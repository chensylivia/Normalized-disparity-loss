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
from util import dataset, transform, config
from util.util import AverageMeter

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)

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
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)

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
    
    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)#tensorboardX
        #logger.info(args)
        logger.info("=> creating model ...")
        #logger.info("Classes: {}".format(args.classes))
        #logger.info(model)
    

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size_test = int(args.batch_size_test / ngpus_per_node)
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
            checkpoint = torch.load(args.weight, map_location=lambda storage, loc: storage.cuda())
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

    
    if args.evaluate:
        test_transform=transform.Compose_disp([
            transform.Crop_disp([args.test_h, args.test_w], crop_type='center', padding=mean, ignore_label=args.ignore_disp_label),
            transform.ToTensor_disp(),
            transform.Normalize_disp(mean=mean, std=std)])
        if args.SF:
            test_data = dataset.SemData_disp_SF(split='test', data_root=args.test_data_root, data_left_list=args.test_left_list,data_right_list=args.test_right_list,\
            data_disp_list=args.test_disp_list,transform=test_transform)
        else:
            test_data = dataset.SemData_disp(split='test', data_root=args.test_data_root, data_left_list=args.test_left_list,data_right_list=args.test_right_list,\
            data_disp_list=args.test_disp_list,transform=test_transform)

        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=test_sampler)
    
        if args.evaluate:
            allEPE_test,allD1E_test,all3px_test= test_disp(test_loader, model)

def test_disp(test_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start testing >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_point_error_meter = AverageMeter()
    D1_error_meter=AverageMeter()
    three_px_error_meter=AverageMeter()

    model.eval()
    end = time.time()
    for i, (left_input, right_input,disp_target) in enumerate(test_loader):
        data_time.update(time.time() - end)
        left_input = left_input.cuda(non_blocking=True)
        right_input = right_input.cuda(non_blocking=True)
        disp_target = disp_target.cuda(non_blocking=True)

        mask = ((disp_target<args.maxdisp)&( disp_target>0))
        mask.detach_()
        with torch.no_grad():
            output_disp = model(left_input,right_input,disp_target)

        n=len(output_disp[mask])
        
        end_point_error= torch.sum(torch.abs(output_disp[mask]-disp_target[mask]))

        error = torch.abs(output_disp[mask]-disp_target[mask])
        error_number= (error>3)&(error > disp_target[mask]*0.05)
        D1_error =torch.sum(error_number)*100 

        three_px_error_number=error>3
        three_px_error=torch.sum(three_px_error_number)*100
        
        end_point_error,D1_error,three_px_error=float(end_point_error)/float(n+0.0000001),float(D1_error)/float(n+0.0000001),float(three_px_error)/float(n+0.0000001)
        
        end_point_error_meter.update(end_point_error,1)
        D1_error_meter.update(D1_error,1)
        three_px_error_meter.update(three_px_error,1)

        batch_time.update(time.time() - end)
        end = time.time()

        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info('Val: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'EPE {end_point_error_meter.val:.4f} '
                        'D1E {D1_error_meter.val:.4f} '
                        '3px {three_px_error_meter.val:.4f} '.format(i + 1, len(test_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          end_point_error_meter=end_point_error_meter,
                                                          D1_error_meter=D1_error_meter,
                                                          three_px_error_meter=three_px_error_meter))

    allEPE= end_point_error_meter.avg
    allD1E=D1_error_meter.avg
    all3px=three_px_error_meter.avg

    if main_process():
        logger.info('Test result: allEPE/allE1D/all3px {:.4f}/{:.4f}/{:.4f}'.format(allEPE,allD1E,all3px))
        logger.info('<<<<<<<<<<<<<<<<< End Testing <<<<<<<<<<<<<<<<<')

    return allEPE,allD1E,all3px


if __name__ == '__main__':
    main()
