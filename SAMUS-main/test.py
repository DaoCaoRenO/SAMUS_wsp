from ast import arg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
import json
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from thop import profile
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS') 
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS') 
    parser.add_argument('--task', default='BUSI', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # 8 # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') # True
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--load_path', type=str, default='', help='覆盖 config 的权重路径（用于测试不同 checkpoint）')
    parser.add_argument('--freq_adapter', type=int, default=0, help='1=按方案C结构构建模型（测试 Freq_Adapter 权重时必须开）')
    parser.add_argument('--freq_adapter_share', type=int, default=0, help='与训练时保持一致：1=通道共用滤波器')
    parser.add_argument('--c2f', type=int, default=0, help='1=按 coarse-to-fine 结构构建模型（测试 PBR 权重时必须开）')
    parser.add_argument('--c2f_iters', type=int, default=1, help='与训练一致：PBR 精炼迭代次数(>1 即渐进式推理)')
    parser.add_argument('--c2f_hidden', type=int, default=32, help='与训练一致：PBR 头隐藏通道数')
    # ---- RAG PCR 头评测装配路径(与 train.py 一致) ----
    parser.add_argument('--rag_head', type=int, default=0, help='1=装配并经过 PCR 头精炼后评测')
    parser.add_argument('--rag_head_hidden', type=int, default=32, help='PCR 头隐藏通道数(需与训练一致)')
    parser.add_argument('--rag_k', type=int, default=5, help='检索 top-k')
    parser.add_argument('--rag_retr', choices=['roi', 'mean'], default='roi', help='检索向量')
    parser.add_argument('--rag_bank', type=str, default='rag_bank/tn3k_bank.pt', help='源域记忆库 .pt')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    if args.load_path:
        opt.load_path = args.load_path
    print("task", args.task, "checkpoints:", opt.load_path)
    opt.mode = "val"
    #opt.classes=2
    opt.visual = True
    #opt.eval_mode = "patient"
    opt.modelname = args.modelname
    args.device = opt.device
    device = torch.device(opt.device)

     #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 300 # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    opt.batch_size = args.batch_size * args.n_gpu

    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    val_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_val, img_size=args.encoder_input_size, class_id=1)  # return image, mask, and filename
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.modelname=="SAMed":
        opt.classes=2
    if args.freq_adapter:  # 方案C：构建前打开结构开关，否则多出的 Freq_Adapter 键会导致加载失败
        from models.segment_anything_samus.modeling.image_encoder import enable_freq_adapter
        enable_freq_adapter(share=bool(args.freq_adapter_share))
    if args.c2f:  # coarse-to-fine：构建前打开 PBR 头开关，否则权重里的 pbr_head 键会因结构缺失被忽略
        from models.segment_anything_samus.modeling.samus import enable_pbr
        enable_pbr(hidden=args.c2f_hidden, iters=args.c2f_iters)
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    model.train()

    if args.rag_head:   # 先挂 PCR 头，才能接住 checkpoint 里的 pcr.* 键
        from rag_head import build_pcr_head
        model.pcr = build_pcr_head(hidden=args.rag_head_hidden).to(device)

    checkpoint = torch.load(opt.load_path)
    #------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k,v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=not (args.freq_adapter or args.c2f or args.rag_head))
    if args.rag_head:
        n_pcr = sum(1 for k in new_state_dict if k.startswith("pcr."))
        print("rag_head=ON, 权重内 pcr.* 键数 =", n_pcr)
        assert n_pcr > 0, "权重里没有 pcr.*，该 checkpoint 不是 PCR 头训出来的"
    if args.freq_adapter:
        n_fa = sum(1 for k in new_state_dict if "Freq_Adapter" in k)
        print("freq_adapter=ON, 权重内 Freq_Adapter 键数 =", n_fa)
        assert n_fa > 0, "权重里没有 Freq_Adapter，该 checkpoint 不是方案C训出来的"
    if args.c2f:
        n_pbr = sum(1 for k in new_state_dict if "pbr_head" in k)
        print("c2f=ON, 权重内 pbr_head 键数 =", n_pbr)
        assert n_pbr > 0, "权重里没有 pbr_head，该 checkpoint 不是 coarse-to-fine 训出来的"
    

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    criterion = get_criterion(modelname=args.modelname, opt=opt)

#  ========================================================================= begin to evaluate the model ============================================================================

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    # flops, params = profile(model, inputs=(input, points), )
    # print('Gflops:', flops/1000000000, 'params:', params)

    model.eval()

    eval_model = model
    if args.rag_head:   # 走 backbone+PCR 头「检索→精炼」路径，与训练一致(不 shift、不 exclude)
        import rag_common as R
        from rag_head import RagEvalRunner
        rag_bank = torch.load(args.rag_bank, map_location="cpu")
        eval_model = RagEvalRunner(model, rag_bank, args.rag_k, args.rag_retr)

    if opt.mode == "train":
        dices, mean_dice, _, val_losses = get_eval(valloader, eval_model, criterion=criterion, opt=opt, args=args)
        print("mean dice:", mean_dice)
    else:
        mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(valloader, eval_model, criterion=criterion, opt=opt, args=args,epoch='test')
        print("dataset:" + args.task + " -----------model name: "+ args.modelname)
        print("task", args.task, "checkpoints:", opt.load_path)
        print(mean_dice[1:], mean_hdis[1:], mean_iou[1:], mean_acc[1:], mean_se[1:], mean_sp[1:])
        print(std_dice[1:], std_hdis[1:], std_iou[1:], std_acc[1:], std_se[1:], std_sp[1:])
        # 机器可读结果行，供跨域对比汇总脚本解析（dice 等为百分值）
        print("RESULT_JSON " + json.dumps({
            "task": args.task, "model": args.modelname, "ckpt": os.path.basename(opt.load_path),
            "dice": float(mean_dice[1]), "hdis": float(mean_hdis[1]), "iou": float(mean_iou[1]),
            "acc": float(mean_acc[1]), "se": float(mean_se[1]), "sp": float(mean_sp[1]),
            "dice_std": float(std_dice[1])
        }))
        # with open("experiments.txt", "a+") as file:
        #     file.write(args.task + " " + args.modelname + "-pt10 " + '%.2f'%(mean_dice[1]) + "±" + '%.2f'%std_dice[1] + " ")
        #     file.write('%.2f'%mean_hdis[1] + "±" + '%.2f'%std_hdis[1] + " ")
        #     file.write('%.2f'%(mean_iou[1]) + "±" + '%.2f'%std_iou[1] + " ")
        #     file.write('%.2f'%(mean_acc[1]) + "±" + '%.2f'%std_acc[1] + " ")
        #     file.write('%.2f'%(mean_se[1]) + "±" + '%.2f'%std_se[1] + " ")
        #     file.write('%.2f'%(mean_sp[1]) + "±" + '%.2f'%std_sp[1] + "\n")

        #     # file.write("MYO " + args.modelname + " " + '%.2f'%(mean_dice[2]) + "±" + '%.2f'%std_dice[2] + " ")
        #     # file.write('%.2f'%mean_hdis[2] + "±" + '%.2f'%std_hdis[2] + " ")
        #     # file.write('%.2f'%(mean_iou[2]) + "±" + '%.2f'%std_iou[2] + " ")
        #     # file.write('%.2f'%(mean_acc[2]) + "±" + '%.2f'%std_acc[2] + " ")
        #     # file.write('%.2f'%(mean_se[2]) + "±" + '%.2f'%std_se[2] + " ")
        #     # file.write('%.2f'%(mean_sp[2]) + "±" + '%.2f'%std_sp[2] + "\n")
        #     # file.write("LA " + args.modelname + " " + '%.2f'%(mean_dice[3]) + "±" + '%.2f'%std_dice[3] + " ")
        #     # file.write('%.2f'%mean_hdis[3] + "±" + '%.2f'%std_hdis[3] + " ")
        #     # file.write('%.2f'%(mean_iou[3]) + "±" + '%.2f'%std_iou[3] + " ")
        #     # file.write('%.2f'%(mean_acc[3]) + "±" + '%.2f'%std_acc[3] + " ")
        #     # file.write('%.2f'%(mean_se[3]) + "±" + '%.2f'%std_se[3] + " ")
        #     # file.write('%.2f'%(mean_sp[3]) + "±" + '%.2f'%std_sp[3] + "\n")
        #     file.close()
    
if __name__ == '__main__':
    main()
