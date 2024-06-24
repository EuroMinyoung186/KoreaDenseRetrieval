import argparse
import os
import random
import shutil
import warnings
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.optim import AdamW
from utils.scheduler import GradualWarmupScheduler
from torch import autocast

from dataset.dataset import TranslationDataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from utils.utils import average_pool
import pdb
from multiprocessing import current_process
import wandb

import bitsandbytes as bnb

from typing import List, Literal, Tuple, Dict, Optional
git
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='./dataset',
                    help='path to dataset')

parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=36, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, metavar='LR', dest='lr',
                    help='initial learning rate')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world_size', default=6, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')

parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training') 
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')
parser.add_argument('--wandb_api_key', default=None, type=str)

best_val_loss = sys.maxsize

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def set_device(args):
    if torch.cuda.is_available():
        if args.gpu:
            torch.cuda.set_device(args.gpu)
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
    

def main():
    args = parser.parse_args()
    ### 임시
    args.is_tmp = False # 이거 쓰면, 데이터셋 1000개만 씀
    args.use_wandb = True if args.wandb_api_key else False
    args.loss_type = ['score'] #['score', 'mse']
    args.t_model_name = 'intfloat/multilingual-e5-base'
    args.s_model_name = 'intfloat/multilingual-e5-base'
    ####
    
    if args.seed is not None:
        seed_everything(args.seed)
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    # 분산학습 여부
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # 사용 가능한 GPU를 모두 할당
    # 특정 GPU를 선택하고 싶다면, CUDA_VISIBLE_DEVICES를 설정
    if torch.cuda.is_available():
        ngpus_per_node = min(torch.cuda.device_count(), args.world_size)
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1
    # 작업 수행
    if args.multiprocessing_distributed: # Multiprocessing_distributed라면, 자식 프로세스를 생성하여 각각의 GPU에 할당
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else: # 그렇지 않다면, main_worker를 호출하여 싱글프로세스로 원래 하던 것과 같이 작업 수행
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_val_loss
    
    if 'mse' in args.loss_type:
        if 'score' in args.loss_type:
            loss_str = 'both'
        else:
            loss_str = 'mse'
    else:
        loss_str = 'score'
        
    args.gpu = gpu
    if args.gpu is not None:
        print(f"[GPU: {args.gpu}] start training")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)
        print(f"[GPU: {args.gpu}] Complete init process group")
        
        if args.gpu is not None:
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = max(int((args.workers) / ngpus_per_node), 1)
    
    #wankdb 세팅
    if (args.rank == 0) and args.wandb_api_key:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key
        os.environ['WANDB_LOG_MODEL'] = 'all'
        wandb.login()
        run = wandb.init(
            project='KoDistillE5'
        )
    
    # 작업중인 프로세스 -> 터미널 출력에 필요
    current = current_process()
    
    # 현재 device 설정
    device = set_device(args)
    
    # Create Model & Tokneizer
    t_model_name = args.t_model_name #'intfloat/multilingual-e5-small'
    s_model_name = args.s_model_name #'intfloat/multilingual-e5-small'
    model_str = s_model_name.split('/')[-1]
    
    t_model = AutoModel.from_pretrained(t_model_name)
    for param in t_model.parameters():
        param.requires_grad = False
    t_model.eval()
    
    s_model = AutoModel.from_pretrained(s_model_name)
    
    t_model.to(device)
    s_model.to(device)
    
    if args.distributed:
        s_model = torch.nn.parallel.DistributedDataParallel(s_model, broadcast_buffers=False, find_unused_parameters=True)

    t_tokenizer = AutoTokenizer.from_pretrained(t_model_name)
    s_tokenizer = AutoTokenizer.from_pretrained(s_model_name)
    print(f"[GPU: {args.gpu}] Load Model Complete!")
    
    # Loss
    mse_criterion = nn.MSELoss().to(device)
    kld_criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    # Optimizer & Scheduler
    #optimizer = AdamW(s_model.parameters(), lr=args.lr)
    optimizer = bnb.optim.Adam8bit(s_model.parameters(), lr=args.lr)
    
    print(f"[GPU: {args.gpu}] Load Loss & Scheduler Complete!")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
                
            best_val_loss = checkpoint['best_val_loss']
            if args.gpu is not None:
                best_val_loss = best_val_loss.to(args.gpu)
                
            s_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # Dataset 불러오기
    tokenizer_args = {
        'max_length': 512,
        'truncation': True,
        'padding': 'max_length',
        'return_tensors': 'pt'
        }
    train_dataset = TranslationDataset(os.path.join(args.data, 'train.csv'), t_tokenizer, s_tokenizer, tokenizer_args, is_tmp=args.is_tmp)
    val_dataset = TranslationDataset(os.path.join(args.data, 'test.csv'), t_tokenizer, s_tokenizer, tokenizer_args, is_tmp=args.is_tmp)
    print(f"[GPU: {args.gpu}] Load Dataset Complete!")

    # 분산처리를 위한 데이터 로더 설정
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)  if args.distributed else None
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    print(f"[GPU: {args.gpu}] Load DataLoader Complete!")
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        steps_per_epoch=len(train_loader), 
        epochs=args.epochs
        )
    
    # 학습하기
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        s_model.train()
        train_loss = iteration(
            epoch, t_model, s_model, 
            mse_criterion, kld_criterion, optimizer, scheduler, train_loader, 
            device, current, args, train=True)
        
        s_model.eval()
        with torch.no_grad():
            val_loss = iteration(
                epoch, t_model, s_model, 
                mse_criterion, kld_criterion, optimizer, scheduler, val_loader, 
                device, current, args, train=False)
        
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': s_model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, val_loss, is_best, loss_str, model_str)


def iteration(epoch, 
              t_model, s_model, mse_criterion, kld_criterion, optimizer,
              scheduler, data_loader, device, current_child, 
              args=None, train=True):
    str_code = "train" if train else "test"
    data_iter = tqdm(enumerate(data_loader),
                        desc="[%s]EP_%s: %d" % (device, str_code, epoch),
                        total=len(data_loader),
                        bar_format="{l_bar}{r_bar}",
                        position=current_child._identity[0] - 1)
    scaler = torch.cuda.amp.GradScaler()

    avg_loss = 0.0
    for i, data in data_iter:
        log = {
            f'{str_code}_step': epoch * len(data_iter) + i,
        }
        
        t_en_inputs, s_en_inputs, s_ko_inputs = data
        
        with autocast(device_type='cuda', dtype=torch.float16):
            dict_to_device = lambda x: {key: value.to(device) for key, value in x.items()}
            t_en_inputs, s_en_inputs, s_ko_inputs = map(dict_to_device, [t_en_inputs, s_en_inputs, s_ko_inputs])
            
            t_en_embeds = average_pool(t_model(**t_en_inputs)[0], t_en_inputs['attention_mask']).detach() # N E
            s_en_embeds = average_pool(s_model(**s_en_inputs)[0], s_en_inputs['attention_mask']) # N E
            s_ko_embeds = average_pool(s_model(**s_ko_inputs)[0], s_ko_inputs['attention_mask']) # N E
            
            loss = torch.zeros(1, device=device)
            
            if 'mse' in args.loss_type:
                en_en_loss = mse_criterion(s_en_embeds, t_en_embeds)
                en_ko_loss = mse_criterion(s_ko_embeds, t_en_embeds)
                mse_loss = en_en_loss + en_ko_loss
                
                loss += mse_loss
                log[f'{str_code}_mse_loss'] = mse_loss.item()
            
            if 'score' in args.loss_type:
                t_en2t_en_matrix = F.softmax(torch.matmul(t_en_embeds, t_en_embeds.T), dim=1)
                t_en2s_en_matrix = F.log_softmax(torch.matmul(t_en_embeds, s_en_embeds.T), dim=1)
                t_en2s_ko_matrix = F.log_softmax(torch.matmul(t_en_embeds, s_ko_embeds.T), dim=1)
                score_loss = kld_criterion(t_en2s_en_matrix, t_en2t_en_matrix) + \
                    kld_criterion(t_en2s_ko_matrix, t_en2t_en_matrix)
                
                loss += score_loss
                log[f'{str_code}_score_loss'] = score_loss.item()

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            if scheduler is not None:
                log['lr'] = scheduler.get_last_lr()[0]
        avg_loss += loss.item()
        data_iter.set_description(
                "[%s]EP_%s: %d" % (device, str_code, epoch) + f' | loss: {loss.item():.5f}')
        if (args.rank == 0) and args.use_wandb:
            wandb.log(log)

    return avg_loss


def save_checkpoint(state, val_loss, is_best, loss_str, model_str, save_dir='./checkpoints'):
    save_dir = os.path.join(save_dir, model_str, loss_str)
    # dir 만들기
    def _create_folder(dir):
            try:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            except OSError:
                print('[Error] Creating directory.' + dir)
    _create_folder(save_dir)
    file_path = os.path.join(save_dir, f'checkpoint_{state["epoch"]}_l{val_loss:.3f}.pth')
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(save_dir, 'model_best.pth'))
        

if __name__ == '__main__':
    main()