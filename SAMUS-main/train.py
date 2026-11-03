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
        model.load_state_dict(new_state_dict)
      
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
   
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

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
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args,epoch=epoch)
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