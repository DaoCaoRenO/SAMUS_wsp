# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from turtle import shape
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from einops import rearrange
from .Auto_Prompt_Generator import AutoPromptGenerator

# 方案 coarse-to-fine（渐进式边界精炼）全局开关，由 enable_pbr() 在构建模型前打开
_PBR = {"on": False, "hidden": 32, "iters": 1}


def enable_pbr(hidden: int = 32, iters: int = 1) -> None:
    """开启渐进式边界精炼头。iters=精炼迭代次数(>1 即渐进式推理)。"""
    _PBR["on"] = True
    _PBR["hidden"] = hidden
    _PBR["iters"] = iters


class PBRHead(nn.Module):
    """渐进式边界精炼头（coarse-to-fine）。

    输入：解码器特征 feat(b,C,h,w) + 粗 mask logits(b,1,H,W)；
    输出：残差 ΔM(b,1,H,W)。末层**零初始化** -> 接入瞬间 fine==coarse（严格恒等，
    不破坏预训练主干，沿用 FreqAdapter 的 LoRA 式稳定化思路）。

    机制：用 Sobel(粗mask) 定位边界带，轻量 conv 融合[特征,粗mask,边界带]后
    只在边界附近产生修正 -> 区域由粗阶段保证、边界由细阶段精炼，解耦二者矛盾。
    """

    def __init__(self, feat_dim: int = 256, hidden: int = 32) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, hidden, 1),
            nn.GroupNorm(1, hidden), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden + 2, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden), nn.GELU(),
        )
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)   # 零初始化残差 -> 初始恒等
        # 固定 Sobel 核（不回传）
        self.register_buffer("kx", torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer("ky", torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))

    def _sobel(self, m: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(m, self.kx.to(m.dtype), padding=1)
        gy = F.conv2d(m, self.ky.to(m.dtype), padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, feat: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        H, W = coarse_logits.shape[-2:]
        f = F.interpolate(self.proj(feat), size=(H, W), mode="bilinear", align_corners=False)
        m = torch.sigmoid(coarse_logits)
        edge = self._sobel(m)
        x = torch.cat([f, m, edge], dim=1)
        return self.out(self.fuse(x))   # 残差 ΔM


class Samus(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        auto_prompt_generator: AutoPromptGenerator,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.auto_prompt_generator = auto_prompt_generator
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        for param in self.prompt_encoder.parameters():
          param.requires_grad = False
        for param in self.mask_decoder.parameters():
          param.requires_grad = False
        # for param in self.image_encoder.parameters():
        #   param.requires_grad = False
        for n, value in self.image_encoder.named_parameters():
          if "cnn_embed" not in n and "post_pos_embed" not in n and "Adapter" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "upneck" not in n:
            value.requires_grad = False

        if _PBR["on"]:  # 方案 coarse-to-fine：挂载渐进式边界精炼头
            # 用 RNG 状态快照隔离：PBRHead 的默认 conv 初始化会消耗全局随机流，
            # 若不隔离会使后续 dataloader shuffle 顺序与 base50 不同，多出混淆变量。
            cpu_state = torch.get_rng_state()
            cuda_states = None
            if torch.cuda.is_available():
                cuda_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
            self.pbr_head = PBRHead(feat_dim=256, hidden=_PBR["hidden"])  # 256=prompt_embed_dim
            self.pbr_iters = _PBR["iters"]
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                for i, s in enumerate(cuda_states):
                    torch.cuda.set_rng_state(s, i)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward_sam(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def forward(
        self, 
        imgs: torch.Tensor,
        # pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
        # bbox: torch.Tensor=None, # b 4
    ) -> torch.Tensor:
        imge= self.image_encoder(imgs)
        # if len(pt[0].shape) == 3:
        #   se, de = self.prompt_encoder(            # se b 2 256, de b 256 32 32
        #                 points=pt,
        #                 boxes=None,
        #                 masks=None,
        #             )
        imge,se,de = self.auto_prompt_generator(image_embeddings = imge,output_tokens = torch.cat([self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight], dim=0))
        low_res_masks, _ = self.mask_decoder( # low_res_mask b 1 128 128
                    image_embeddings=imge,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                    )
        masks = F.interpolate(low_res_masks, (256, 256), mode="bilinear", align_corners=False)
        outputs = {"low_res_logits": low_res_masks, "masks": masks, "imge": imge}
        if _PBR["on"] and hasattr(self, "pbr_head"):  # 方案 coarse-to-fine：渐进式边界精炼
            fine = low_res_masks
            for _ in range(self.pbr_iters):        # iters>1 即渐进式推理(逐级锐化)
                fine = fine + self.pbr_head(imge, fine)
            outputs["low_res_logits_coarse"] = low_res_masks   # 粗阶段供深监督(纯区域)
            outputs["low_res_logits"] = fine                    # 主输出改为精炼后(下游 eval/test 自动用 fine)
            outputs["masks"] = F.interpolate(fine, (256, 256), mode="bilinear", align_corners=False)
        return outputs
        # else:
        #   low_res_masks, masks = [], []
        #   for i in range(pt[0].shape[1]):
        #     # pti = (pt[0][:, i, :, :], pt[1][:, i, :])
        #     # sei, dei = self.prompt_encoder(            # se b 2 256, de b 256 32 32
        #     #             points=pti,
        #     #             boxes=None,
        #     #             masks=None,
        #     #         )
        #     # print('sei',sei.shape)
        #     # print('dei',dei.shape)
        #     imge, sei, dei = self.auto_prompt_generator(image_embeddings=imge,
        #                                               output_tokens=self.mask_decoder.output_tokens)
        #     low_res_masksi, _ = self.mask_decoder( # low_res_mask b 1 128 128
        #             image_embeddings=imge,
        #             image_pe=self.prompt_encoder.get_dense_pe(),
        #             sparse_prompt_embeddings=sei,
        #             dense_prompt_embeddings=dei,
        #             multimask_output=False,
        #             )
        #     masksi = F.interpolate(low_res_masksi, (256, 256), mode="bilinear", align_corners=False)
        #     low_res_masks.append(low_res_masksi)
        #     masks.append(masksi)
        #   low_res_masks = torch.stack(low_res_masks, dim=1)
        #   masks = torch.stack(masks, dim=1) # b c 1 255 255
        #   masks = masks.reshape(masks.shape[0], -1, masks.shape[3], masks.shape[4])
        #   low_res_masks = low_res_masks.reshape(low_res_masks.shape[0], -1, low_res_masks.shape[3], low_res_masks.shape[4])
        #   outputs = {"low_res_logits": low_res_masks, "masks": masks}
        #   return outputs



    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
