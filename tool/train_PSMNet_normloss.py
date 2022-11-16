# -*- coding=utf-8 -*-

import os
import random
import cv2
import numpy as np
import logging
import argparse
import time
import apex
from apex.parallel import convert_syncbn_model
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
from scipy.optimize import curve_fit
from util import dataset, transform, config
from util.util import AverageMeter,AverageMeter_list

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def func(x,a,b,c):
    return 0.000001*a*(x**b)+c

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Stereo Matching')
    parser.add_argument('--config', type=str, default=None, help='config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

GLOBAL_SEED = 1
GLOBAL_WORKER_ID = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main():
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = True

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)


    from model.PSMNet import PSMNet_normloss
    model =PSMNet_normloss(args.maxdisp)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    if main_process():#true
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)#tensorboardX
        #logger.info(args)
        logger.info("=> creating model ...")
        #logger.info("Classes: {}".format(args.classes))
        #logger.info(model)
    
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        if args.use_apex:
            model= convert_syncbn_model(model)
            model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level=args.opt_level, keep_batchnorm_fp32=args.keep_batchnorm_fp32, loss_scale=args.loss_scale)
            model = apex.parallel.DistributedDataParallel(model)
        else:
            model=convert_syncbn_model(model)
            model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))


    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))  
    
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform=transform.Compose_disp([
            transform.Crop_disp([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_disp_label),
            transform.ToTensor_disp(),
            transform.Normalize_disp(mean=mean, std=std)])
    if args.SF:#sceneflow
        train_data = dataset.SemData_disp_SF(split='train', data_root=args.train_data_root, data_left_list=args.train_left_list,data_right_list=args.train_right_list,\
            data_disp_list=args.train_disp_list, transform=train_transform)
    else:
        train_data = dataset.SemData_disp(split='train', data_root=args.train_data_root, data_left_list=args.train_left_list,data_right_list=args.train_right_list,\
        data_disp_list=args.train_disp_list, transform=train_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, worker_init_fn=worker_init_fn,pin_memory=True, sampler=train_sampler, drop_last=True)
        
    a,b=0,0
    m,p=0,1
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        loss_train,allEPE_train,allD1E_train,all3px_train,a,b,m,p= train_disp(train_loader, model,optimizer, epoch,a,b,1,m,p)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('allEPE_train', allEPE_train, epoch_log) 
            writer.add_scalar('allD1E_train', allD1E_train, epoch_log)
            writer.add_scalar('all3px_train', all3px_train, epoch_log)
        
        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
        

def train_disp(train_loader, model, optimizer, epoch,a,b,c,m,p):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    end_point_error_meter = AverageMeter()
    D1_error_meter = AverageMeter()
    three_px_error_meter = AverageMeter()
    loss_meter=AverageMeter()

    L1_loss_list_meter=AverageMeter_list(38)
    L1_loss=[0]*38
    n_L1_loss=[0]*38
    L1_loss_list=[0]*38

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    
    for i, (left_input, right_input,disp_target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        left_input = left_input.cuda(non_blocking=True)
        right_input = right_input.cuda(non_blocking=True)
        disp_target = disp_target.cuda(non_blocking=True)
        
        mask = ((disp_target<args.maxdisp)&(disp_target>0))
        mask.detach_()

        loss,output_disp= model(left_input,right_input,disp_target,a,b,c,m,p,args.alpha,args.beta)
        optimizer.zero_grad()

        if args.use_apex and args.multiprocessing_distributed:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        optimizer.step()

        n=len(output_disp[mask])
        #EPE
        end_point_error= torch.sum(torch.abs(output_disp[mask]-disp_target[mask]))
        #D1
        error = torch.abs(output_disp[mask]-disp_target[mask])
        error_number= (error>3)&(error > disp_target[mask]*0.05)
        D1_error =(torch.sum(error_number))*100 
        #3px
        three_px_error_number = error>3
        three_px_error = torch.sum(three_px_error_number)*100 

        for k in range(5,195,5):
            samp=(disp_target<=k) & (disp_target> k-5)
            L1_loss[k//5-1]=torch.sum(torch.abs(output_disp[samp]-disp_target[samp]))
            n_L1_loss[k//5-1]=len(output_disp[samp])

        if args.multiprocessing_distributed:
            end_point_error,D1_error,three_px_error=end_point_error.detach(), D1_error.detach(),three_px_error.detach()
            count = disp_target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(count)
            dist.all_reduce(end_point_error),dist.all_reduce(D1_error),dist.all_reduce(three_px_error)
            n = count.item()

            n_L1_loss = disp_target.new_tensor(n_L1_loss, dtype=torch.long)
            L1_loss=end_point_error.new_tensor(L1_loss,dtype=torch.float)
            
            dist.all_reduce(n_L1_loss)
            dist.all_reduce(L1_loss)

            end_point_error,D1_error,three_px_error=float(end_point_error)/float(n+0.0000001),float(D1_error)/float(n+0.0000001),float(three_px_error)/float(n+0.0000001)
        else:
            end_point_error,D1_error,three_px_error=float(end_point_error)/float(n+0.0000001),float(D1_error)/float(n+0.0000001),float(three_px_error)/float(n+0.0000001)
        

        for k in range(38):
            L1_loss_list[k]=float(L1_loss[k])/(float(n_L1_loss[k])+0.0000001)

        end_point_error_meter.update(end_point_error,1)
        D1_error_meter.update(D1_error,1)
        three_px_error_meter.update(three_px_error,1)
        
        loss_meter.update(loss.item(), 1)

        L1_loss_list_meter.update(L1_loss_list,n_L1_loss.cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'EPE {end_point_error_meter.val:.4f} '
                        'D1E {D1_error_meter.val:.4f} '
                        '3px {three_px_error_meter.val:.4f} '.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          end_point_error_meter=end_point_error_meter,
                                                          D1_error_meter=D1_error_meter,
                                                          three_px_error_meter=three_px_error_meter))
        
        if main_process():#tensorboardX
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('EPE_train_batch', end_point_error_meter.val, current_iter)
            writer.add_scalar('D1E_train_batch', D1_error_meter.val, current_iter)
            writer.add_scalar('3px_train_batch', three_px_error_meter.val, current_iter)
            
    
    L1_loss_list_avg=L1_loss_list_meter.avg
    xdata=np.linspace(5,190,38)
    popt,pconv=curve_fit(func,xdata,L1_loss_list_avg,maxfev=5000)

    allEPE= end_point_error_meter.avg
    allD1E=D1_error_meter.avg
    all3px=three_px_error_meter.avg

    if main_process():
        logger.info('Train result at epoch [{}/{}]: allEPE/allE1D/all3px/a/b/c {:.4f}/{:.4f}/{:.4f}/{:.15f}/{:.15f}/{:.15f}'.format(epoch+1, args.epochs, allEPE,allD1E,all3px,0.000001*popt[0],popt[1],popt[2]))
    
    if epoch==0:
        return loss_meter.avg,allEPE,allD1E,all3px,0.000001*popt[0],popt[1],(0.000001*popt[0]*(args.maxdisp**popt[1])+popt[2])/3, (0.000001*popt[0]*(args.maxdisp**popt[1])+popt[2])
    else:
        return loss_meter.avg,allEPE,allD1E,all3px,0.000001*popt[0],popt[1],args.alpha*m+(1-args.alpha)*p/args.beta, (0.000001*popt[0]*(args.maxdisp**popt[1])+popt[2])

if __name__ == '__main__':
    main()
