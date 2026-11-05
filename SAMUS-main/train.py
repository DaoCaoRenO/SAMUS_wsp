from ast import arg
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 本机单卡4090，原代码为'8'
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
import math
def main():

    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='US30K', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=True, help='keep the loss&lr&dice during training or not')
    # ---- 实验开关：频域增广 / 训练轮数 / 输出隔离（均默认保留原始行为）----
    parser.add_argument('--epochs', type=int, default=None, help='覆盖 config 的训练轮数；None 则用 config')
    parser.add_argument('--out_tag', type=str, default='', help='为 save/result/tensorboard 路径加后缀，隔离不同实验（如 base50 / freq50）')
    parser.add_argument('--freq_aug', type=int, default=0, help='1=开启频谱随机化增广（方案A），0=原始行为')
    parser.add_argument('--freq_p', type=float, default=0.5, help='每张图施加频域增广的概率')
    parser.add_argument('--freq_alpha', type=float, default=0.1, help='幅度谱扰动强度')
    parser.add_argument('--freq_sigma', type=float, default=0.1, help='低频随机场尺度')
    parser.add_argument('--freq_adapter', type=int, default=0, help='1=开启方案C（ParaBlock 内并联可学习频域 Adapter），0=原始结构')
    parser.add_argument('--freq_adapter_share', type=int, default=0, help='1=所有通道共用一组复数滤波器（参数量降为 1/dim）')
    parser.add_argument('--freq_loss', type=int, default=0, help='1=开启方案D（在分割损失上叠加 Focal Frequency Loss），0=原损失')
    parser.add_argument('--freq_loss_w', type=float, default=0.1, help='FFL 叠加权重')
    parser.add_argument('--freq_loss_alpha', type=float, default=1.0, help='FFL focal 指数（0=普通频谱L2）')
    parser.add_argument('--freq_loss_hf', type=float, default=0.15, help='FFL 只监督 r>=此值 的高频带（0=全带，不建议）')
    parser.add_argument('--freq_loss_ramp', type=int, default=10, help='FFL 权重前 N 轮线性升温（0=不升温）')
    parser.add_argument('--edge_loss', type=int, default=0, help='1=开启边缘增强（在分割损失上叠加空域 Sobel 边界损失），0=原损失')
    parser.add_argument('--edge_loss_w', type=float, default=1.0, help='边界损失叠加权重')
    parser.add_argument('--edge_loss_ramp', type=int, default=10, help='边界损失权重前 N 轮线性升温（0=不升温）')
    parser.add_argument('--c2f', type=int, default=0, help='1=开启 coarse-to-fine 渐进式边界精炼头(PBR)+深监督，0=关闭')
    parser.add_argument('--c2f_iters', type=int, default=1, help='PBR 精炼迭代次数(>1 即渐进式推理)')
    parser.add_argument('--c2f_hidden', type=int, default=32, help='PBR 头隐藏通道数')
    parser.add_argument('--c2f_lam_coarse', type=float, default=1.0, help='粗阶段区域损失权重')
    parser.add_argument('--c2f_lam_fine', type=float, default=1.0, help='细阶段区域损失权重')
    parser.add_argument('--c2f_edge_w', type=float, default=0.5, help='细阶段边缘损失权重')
    parser.add_argument('--c2f_edge_ramp', type=int, default=10, help='细阶段边缘损失前 N 轮升温')
    # ---- RAG 可训练原型条件化精炼头(PCR head) 开关（均默认关闭，向后兼容原始行为）----
    parser.add_argument('--rag_head', type=int, default=0, help='1=只训 PCR 头(主干全冻结)，0=原始行为')
    parser.add_argument('--rag_head_hidden', type=int, default=32, help='PCR 头隐藏通道数')
    parser.add_argument('--rag_k', type=int, default=5, help='检索 top-k')
    parser.add_argument('--rag_retr', choices=['roi', 'mean'], default='roi', help='检索向量: ROI加权|均值')
    parser.add_argument('--rag_shift', type=int, default=1, help='1=训练时对 query 图做域偏移模拟(迫使头依赖检索)，0=关闭')
    parser.add_argument('--rag_shift_strength', type=float, default=1.0, help='域偏移退化幅度 ∈[0,~1.5]')
    parser.add_argument('--rag_bank', type=str, default='rag_bank/tn3k_bank.pt', help='源域记忆库 .pt')
    parser.add_argument('--rag_freeze_all', type=int, default=1, help='1=冻结全部主干只训头，0=连同 adapter 微调')
    parser.add_argument('--rag_head_lr', type=float, default=1e-3, help='PCR 头学习率')
    parser.add_argument('--rag_base_ckpt', type=str, default='checkpoints/TN3K/SAMUS_11020447_270_0.8349628402308167.pth',
                        help='冻结主干的起点权重(ep270)，只训头时先加载它')

    args = parser.parse_args()
    opt = get_config(args.task)
    args.batch_size = opt.batch_size
    args.device = opt.device
    args.base_lr = opt.learning_rate
    # 实验参数覆盖：训练轮数 + 输出目录隔离
    if args.epochs is not None:
        opt.epochs = args.epochs
    if args.out_tag:
        opt.save_path = opt.save_path.rstrip('/') + '_' + args.out_tag + '/'
        opt.result_path = opt.result_path.rstrip('/') + '_' + args.out_tag + '/'
        opt.tensorboard_path = opt.tensorboard_path.rstrip('/') + '_' + args.out_tag + '/'

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    if args.freq_adapter:  # 方案C：必须在构建模型前打开开关
        from models.segment_anything_samus.modeling.image_encoder import enable_freq_adapter
        enable_freq_adapter(share=bool(args.freq_adapter_share))
    if args.c2f:  # coarse-to-fine：构建模型前打开 PBR 头开关
        from models.segment_anything_samus.modeling.samus import enable_pbr
        enable_pbr(hidden=args.c2f_hidden, iters=args.c2f_iters)
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = opt.batch_size * args.n_gpu

    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True,
                                p_freq=(args.freq_p if args.freq_aug else 0.0), freq_alpha=args.freq_alpha, freq_sigma=args.freq_sigma)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=not (args.freq_adapter or args.c2f))

    # ============ RAG 可训练 PCR 头：冻结主干、挂载唯一可训练头、载入记忆库 ============
    rag_bank = None
    if args.rag_head:
        import rag_common as R
        from rag_head import build_pcr_head
        # 先加载冻结主干的起点(ep270)。config.pre_trained=False 不会自动加载，故显式装。
        if args.rag_base_ckpt and os.path.exists(args.rag_base_ckpt):
            ck = torch.load(args.rag_base_ckpt, map_location="cpu")
            sd = {(k[7:] if k[:7] == "module." else k): v for k, v in ck.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)  # 此时无 pcr.* → missing 属正常
            print("RAG base ckpt loaded: {} (missing={} unexpected={})".format(
                os.path.basename(args.rag_base_ckpt), len(missing), len(unexpected)))
        if args.rag_freeze_all:                     # 默认：只训头，冻结全部主干(含 image_encoder 可训部分/adapter)
            for param in model.parameters():
                param.requires_grad = False
        model.pcr = build_pcr_head(hidden=args.rag_head_hidden).to(opt.device)  # 唯一可训练(RNG 快照隔离)
        rag_bank = torch.load(args.rag_bank, map_location="cpu")                 # no-grad 常量记忆库
        head_tr = sum(p.numel() for p in model.pcr.parameters() if p.requires_grad)
        print("RAG PCR head ON: hidden={} k={} retr={} shift={} strength={} freeze_all={} | head trainable params {}".format(
            args.rag_head_hidden, args.rag_k, args.rag_retr, bool(args.rag_shift), args.rag_shift_strength,
            bool(args.rag_freeze_all), head_tr))

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    if args.rag_head:   # 只训 PCR 头：AdamW(唯一可训练头, lr=1e-3, wd=0.01)
        optimizer = torch.optim.AdamW(model.pcr.parameters(), lr=args.rag_head_lr, betas=(0.9, 0.999), weight_decay=0.01)
   
    criterion = get_criterion(modelname=args.modelname, opt=opt,
                              freq_loss_w=(args.freq_loss_w if args.freq_loss else 0.0),
                              freq_loss_alpha=args.freq_loss_alpha,
                              freq_loss_hf=args.freq_loss_hf,
                              freq_loss_ramp=args.freq_loss_ramp,
                              edge_loss_w=(args.edge_loss_w if args.edge_loss else 0.0),
                              edge_loss_ramp=args.edge_loss_ramp,
                              c2f=args.c2f, c2f_lam_coarse=args.c2f_lam_coarse,
                              c2f_lam_fine=args.c2f_lam_fine, c2f_edge_w=args.c2f_edge_w,
                              c2f_edge_ramp=args.c2f_edge_ramp)
    if args.freq_loss:
        print("FreqLoss(FFL) ON: weight={} alpha={} hf_ratio={} ramp_epochs={}".format(
            args.freq_loss_w, args.freq_loss_alpha, args.freq_loss_hf, args.freq_loss_ramp))
    if args.edge_loss:
        print("EdgeLoss ON: weight={} ramp_epochs={}".format(args.edge_loss_w, args.edge_loss_ramp))
    if args.c2f:
        pbr_all = sum(p.numel() for n, p in model.named_parameters() if "pbr_head" in n)
        pbr_tr = sum(p.numel() for n, p in model.named_parameters() if "pbr_head" in n and p.requires_grad)
        print("CoarseToFine(PBR) ON: iters={} hidden={} lam_c={} lam_f={} edge_w={} edge_ramp={} | PBR params {} (trainable {})".format(
            args.c2f_iters, args.c2f_hidden, args.c2f_lam_coarse, args.c2f_lam_fine, args.c2f_edge_w, args.c2f_edge_ramp, pbr_all, pbr_tr))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    if args.freq_adapter:
        fa_all = sum(p.numel() for n, p in model.named_parameters() if "Freq_Adapter" in n)
        fa_tr = sum(p.numel() for n, p in model.named_parameters() if "Freq_Adapter" in n and p.requires_grad)
        print("Freq_Adapter params: {} (trainable {} / share={})".format(fa_all, fa_tr, bool(args.freq_adapter_share)))

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        train_losses = 0
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
            class_labels = torch.as_tensor(datapack['class_label'],dtype = torch.float32, device=opt.device)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)
            # print(imgs.shape) # 8 1 256 256
            # -------------------------------------------------------- forward --------------------------------------------------------
            if args.rag_head:
                # 域偏移模拟→前向(含 imge)→检索原型逐位置余弦→PCR 头残差精炼→ Dice+BCE(masks_fine, label)
                imgs_q = R.shift_image(imgs, strength=args.rag_shift_strength) if args.rag_shift else imgs
                out = model(imgs_q)
                imge, base = out["imge"], out["masks"]
                names = [str(x) for x in datapack["image_name"]]          # leave-one-out 屏蔽自身
                sim_fg, sim_bg = R.prototypes_for_query(imge, rag_bank, args.rag_k,
                                                        exclude=names, retr=args.rag_retr)
                masks_fine = model.pcr(imge, base, sim_fg, sim_bg)         # (B,1,256,256)
                label = datapack["label"].to(dtype=torch.float32, device=opt.device)
                train_loss = criterion({"low_res_logits": masks_fine}, label)
            else:
                pred = model(imgs) # pred = model(imgs, pt, bbox)
                train_loss = criterion(pred, masks)
            # -------------------------------------------------------- backward -------------------------------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            # print(batch_idx)
            # for name,param in model.auto_prompt_generator.named_parameters():
            #     print(name, param.requires_grad)  # 输出每个参数是否需要梯度
            #     if 'task_tokens' in name:
            #         print(f"Found {name}, Gradient Sum: {param.grad.abs().sum() if param.grad is not None else 'None'}")
            optimizer.step()
            train_losses += train_loss.item()
            print('batch_idx [{}/{}/{}], train loss:{:.4f}'.format(batch_idx,len(trainloader),epoch,train_loss))
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        if hasattr(criterion, "step_epoch"):   # 方案D：推进频域损失的升温调度
            criterion.step_epoch()
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % opt.eval_freq == 0:
            model.eval()
            # 预测图统一先写入 _latest（每轮覆盖），刷新 best 时再搬到 best/，避免按 epoch 累积
            vis_root = opt.result_path + "/Merge-" + opt.modelname + "/"
            latest_dir = vis_root + "_latest"
            best_vis_dir = vis_root + "best"
            opt.vis_folder = "_latest"
            shutil.rmtree(latest_dir, ignore_errors=True)
            eval_target = model
            if args.rag_head:   # val 时同走 PCR 头精炼路径(不 shift、不 exclude，val 图不在库内无泄漏)
                from rag_head import RagEvalRunner
                eval_target = RagEvalRunner(model, rag_bank, args.rag_k, args.rag_retr)
            dices, mean_dice, _, val_losses = get_eval(valloader, eval_target, criterion=criterion, opt=opt, args=args,epoch=epoch)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice
            if mean_dice > best_dice:
                best_dice = mean_dice
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                # 只保留单一最优权重（覆盖），不再按时间戳累积
                torch.save(model.state_dict(), opt.save_path + args.modelname + "_best.pth", _use_new_zipfile_serialization=False)
                print('>>> new best dice {:.4f} @ epoch {}, saved best ckpt & images'.format(best_dice, epoch))
                # 把本轮 _latest 图目录提升为 best/（先删旧 best）
                if os.path.isdir(latest_dir):
                    shutil.rmtree(best_vis_dir, ignore_errors=True)
                    shutil.move(latest_dir, best_vis_dir)
        # 仅最后一个 epoch 存一份 _last.pth + 写训练曲线（不再每 save_freq 存）
        if epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            torch.save(model.state_dict(), opt.save_path + args.modelname + "_last.pth", _use_new_zipfile_serialization=False)
            if args.keep_log:
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                    for i in range(len(loss_log)):
                        f.write(str(loss_log[i])+'\n')
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                    for i in range(len(dice_log)):
                        f.write(str(dice_log[i])+'\n')

if __name__ == '__main__':
    main()