# Semi-supervised YOLOv5 by Gen Luo


import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm
import json
from itertools import cycle
FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets_semi import create_dataloader_semi
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_mutation, set_logging, one_cycle, colorstr, methods,non_max_suppression_pseudo_decouple_multi_view,non_max_suppression_pseudo_decouple,xyxy2xywhn,non_max_suppression_pseudo_decouple_multi_view_fix
from utils.downloads import attempt_download
from utils.loss import ComputeLoss_semi
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first,_update_teacher_model
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks

# def cycle_(iterable):
#     # cycle('ABCD') --> A B C D A B C D A B C D ...
#     iters=range(len(iterable))
#     for iter_ in cycle(iters):
#      yield iterable.__getitem__(iter_)

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
def split_label_unlabel_data(data_path,dataseed,sup_percent=0.1,seed='1'):
    # print(seed)
    full_data=open(data_path,'r').readlines()
    label_data=open(data_path.replace('.txt','_label.txt'),'w')
    unlabel_data=open(data_path.replace('.txt','_unlabel.txt'),'w')
    dataseed=json.load(open(dataseed))
    label_inds=[]
    n_label=0
    # for key in dataseed[str(sup_percent)]:
    label_inds+=dataseed[str(sup_percent)][seed]
    full_inds=[]
    for line in full_data:
        ind=line.split()[0].split('/')[-1]
        full_inds.append(ind)
        if ind in label_inds:
            n_label+=1
            label_data.write(line)
        else:
            unlabel_data.write(line)
    # from collections import Counter  # 引入Counter
    # b = dict(Counter(label_inds))
    # # print(b)
    # print({key: value for key, value in b.items() if value > 1})  # 展现重复元素和重复次数
    assert n_label==len(set(label_inds))
    label_data.close()
    unlabel_data.close()
    return data_path.replace('.txt','_label.txt'),data_path.replace('.txt','_unlabel.txt')

def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    image_numpy = input_tensor.cpu().float().numpy()
    mean = [0.485, 0.456, 0.406]  # dataLoader中设置的mean参数
    std = [0.229, 0.224, 0.225]  # dataLoader中设置的std参数
    for i in range(len(mean)):  # 反标准化
        image_numpy[i] = image_numpy[i] * std[i] + mean[i]
    image_numpy = image_numpy * 255
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    # 转成pillow
    im = Image.fromarray(image_numpy)
    im.save(filename)


def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')

    if pretrained:
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model_student'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        model_teacher = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model_student'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load

        csd_t = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd_t = intersect_dicts(csd_t, model_teacher.state_dict(), exclude=exclude)  # intersect
        model_teacher.load_state_dict(csd_t, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))#.to(device)  # create
        model_teacher=Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    #load pretrained imagenet weight
    if opt.imagenet_weights is not None and not pretrained:
        if not os.path.exists(opt.imagenet_weights):
            logging.warning('imagenet weights not exist !')
        pretrained_imagenet_weight=torch.load(opt.imagenet_weights)['state_dict']
        #fix
        fix_pretrained_imagenet_weight={}
        for key in pretrained_imagenet_weight:
            if key.startswith('10'):
                print('not load the weights: ',key)
                continue
            fix_pretrained_imagenet_weight['model.'+key]=pretrained_imagenet_weight[key]
        model.load_state_dict(fix_pretrained_imagenet_weight,strict=False)
    model=model.to(device)

    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # print()
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2


    # Scheduler
    if opt.lr_schedule=='linear':
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    elif opt.lr_schedule=='cosine':
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    elif opt.lr_schedule=='constant':
        lf = lambda x: 1.0
    elif opt.lr_schedule=='step':
        def step_func(x):
            if x >= epochs * 0.8:
                return 0.1
            elif x >= epochs * 0.9:
                return 0.01
            return 1.
        lf = step_func
    else:
        assert NotImplementedError
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] and opt.use_ema else None


    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema is not None and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        # print(start_epoch)
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    #split label and unlabel datas
    train_label_path,train_unlabel_path=split_label_unlabel_data(train_path,data_dict['dataseed'],data_dict['sup_percent'])
    # Trainloader
    train_label_loader, dataset_label = create_dataloader_semi(train_label_path, imgsz, batch_size // WORLD_SIZE, gs,
                                                               single_cls,
                                                               hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect,
                                                               rank=RANK,
                                                               workers=workers, image_weights=opt.image_weights,
                                                               quad=opt.quad,
                                                               prefix=colorstr('train: '))

    train_unlabel_loader, dataset_unlabel = create_dataloader_semi(train_unlabel_path, imgsz, batch_size // WORLD_SIZE,
                                                                   gs, single_cls,
                                                                   hyp=hyp, augment=True, cache=opt.cache,
                                                                   rect=opt.rect, rank=RANK,
                                                                   workers=workers, image_weights=opt.image_weights,
                                                                   quad=opt.quad,
                                                                   prefix=colorstr('train: '))

    mlc = int(np.concatenate(dataset_label.labels, 0)[:,
              0].max())  #max(int(np.concatenate(dataset_label.labels, 0)[:, 0].max()),int(np.concatenate(dataset_unlabel.labels, 0)[:, 0].max()) ) # max label class
    nb = max(len(train_label_loader),len(train_unlabel_loader) ) # number of batches
    if opt.steps is not None:
        epochs=opt.steps//nb
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            labels = np.concatenate(dataset_label.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset_label, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end')

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,find_unused_parameters=True)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset_label.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss_semi(model)  # init loss class
    # check the cls loss, using focal loss may be better
    compute_semi_loss = ComputeLoss_semi(model,semi=True)  # init loss class

    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_label_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------

        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset_label.labels, nc=nc, class_weights=cw)  # image weights
            dataset_label.indices = random.choices(range(dataset_label.n), weights=iw,
                                                   k=dataset_label.n)


        mloss = torch.zeros(4, device=device)  # mean losses
        mloss_semi = torch.zeros(3, device=device)  # mean losses
        semi_loss_items = torch.zeros(3, device=device)
        semi_label = torch.zeros(1, device=device)
        if RANK != -1:
            train_label_loader.sampler.set_epoch(epoch)
            train_unlabel_loader.sampler.set_epoch(epoch)
            train_label_loader.set_length(max(len(train_label_loader),len(train_unlabel_loader)))
            train_unlabel_loader.set_length(max(len(train_label_loader),len(train_unlabel_loader)))
        pbar = enumerate(zip(train_label_loader,train_unlabel_loader))
        LOGGER.info(('\n' + '%10s' * 11) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls','mls','semi_obj', 'semi_cls','semi_mls', 'labels', 'semi_labels'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, data in pbar:  # batch -------------------------------------------------------------
            semi_label_items = torch.zeros(1, device=device)
            (label_imgs, label_targets, label_class_one_hot, label_paths, _), (
                unlabel_imgs, unlabel_targets, unlabel_class_one_hot, unlabel_paths, _) = data
            ni = i + nb * epoch  # number integrated batches (since train start)
            # print(label_class_one_hot)
            #data: label_imgs_strong, label_targets_strong,
            #      unlabel_imgs_strong, unlabel_targets_strong,unlabel_imgs_weak, unlabel_targets_weak,
            label_imgs_weak_aug, label_imgs_strong_aug = label_imgs
            unlabel_imgs_weak_aug, unlabel_imgs_strong_aug = unlabel_imgs
            label_imgs_weak_aug = label_imgs_weak_aug.to(device, non_blocking=True).float()
            label_imgs_strong_aug = label_imgs_strong_aug.to(device, non_blocking=True).float()
            unlabel_imgs_weak_aug = unlabel_imgs_weak_aug.to(device, non_blocking=True).float()
            unlabel_imgs_strong_aug = unlabel_imgs_strong_aug.to(device, non_blocking=True).float()
            label_targets=label_targets.to(device, non_blocking=True)
            unlabel_targets = unlabel_targets.to(device, non_blocking=True)



            #2. semi-superived part
            #2.1 pseudo label prediction
            # pseudo_unlabel_targets=teacher(unlabel_data_weak)
            #2.2
            #loss_semi= student.forward_loss(unlabel_imgs_strong, pseudo_unlabel_targets)
            #2.3
            #loss_total= loss_sup+loss_semi*semi_weights


            # imgs = imgs.to(device, non_blocking=True).float() #/ 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(label_imgs_weak_aug.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in
                          label_imgs_weak_aug.shape[2:]]  # new shape (stretched to gs-multiple)
                    label_imgs_weak_aug = nn.functional.interpolate(label_imgs_weak_aug, size=ns, mode='bilinear',
                                                                    align_corners=False)
                    label_imgs_strong_aug = nn.functional.interpolate(label_imgs_strong_aug, size=ns, mode='bilinear',
                                                                      align_corners=False)
                    unlabel_imgs_weak_aug = nn.functional.interpolate(unlabel_imgs_weak_aug, size=ns, mode='bilinear',
                                                                      align_corners=False)
                    unlabel_imgs_strong_aug = nn.functional.interpolate(unlabel_imgs_strong_aug, size=ns,
                                                                        mode='bilinear', align_corners=False)

            '''
            1. superived part
            label_imgs=concat(label_imgs_strong,label_imgs_weak)
            label_targets=concat(label_targets_strong,label_targets_weak)
            loss_sup= student.forward_loss(label_imgs_strong, label_targets_strong)
            '''

            label_imgs=torch.cat([label_imgs_weak_aug,label_imgs_strong_aug],0)
            label_class_one_hot=torch.cat([label_class_one_hot,label_class_one_hot.detach().clone()],0)
            label_targets_strong=label_targets.clone().detach()
            label_paths=label_paths+label_paths.copy()

            if label_targets.size()[0] == 0 or label_targets_strong.size()[0] == 0:
                print(label_imgs_weak_aug.size())
                continue
            label_targets_strong[:,0]+=(label_targets[-1,0]+1)
            label_targets=torch.cat([label_targets,label_targets_strong],0)
            semi_loss = 0.
            if ni < hyp['burn_up_step']:
                # Forward
                with amp.autocast(enabled=cuda):
                    pred,pred_mls = model(label_imgs)  # forward
                    loss, loss_items = compute_loss(pred, label_targets.to(device),pred_mls,label_class_one_hot.to(device))  # loss scaled by batch_size
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.
            else:
                '''
                2. semi-superived part
                '''

                '''
                2.0. ema part
                ema(student,teacher)
                '''
                if ni == hyp['burn_up_step']:
                    _update_teacher_model(model  if ema is None else ema.ema, model_teacher, word_size=WORLD_SIZE, keep_rate=0.)
                elif (
                        ni - hyp['burn_up_step']
                ) % hyp['teacher_update_iter'] == 0:
                    _update_teacher_model(model if ema is None else ema.ema, model_teacher, word_size=WORLD_SIZE, keep_rate=hyp['ema_keep_rate'])

                '''
                2.1 pseudo label prediction
                pseudo_unlabel_targets=teacher(unlabel_data_weak)
                '''
                model_teacher.eval()
                with torch.no_grad():
                    out, train_out,pseudo_class_one_hot =model_teacher(unlabel_imgs_weak_aug,augment=True)

                    #need post-processing
                    pseudo_class_one_hot_post = (pseudo_class_one_hot.detach().sigmoid() > hyp['mls_threshold']).float()
                    if hyp['mix_label_num'] > 0:
                        if isinstance(out,list):
                            pseudo_boxes_reg, pseudo_boxes_cls = non_max_suppression_pseudo_decouple_multi_view(out, hyp[
                                'bbox_threshold'], hyp['cls_threshold'],
                                                                                                     multi_label=True,
                                                                                                     agnostic=single_cls,
                                                                                                     labels=unlabel_targets,
                                                                                                     )
                        else:
                            pseudo_boxes_reg,pseudo_boxes_cls = non_max_suppression_pseudo_decouple(out, hyp['bbox_threshold'],hyp['cls_threshold'],
                                                                      multi_label=True, agnostic=single_cls,
                                                                          labels=unlabel_targets
                                                                      )

                        temp = unlabel_class_one_hot.to(device) + pseudo_class_one_hot_post
                        temp = temp.long()
                        pseudo_class_one_hot_post = torch.where(temp > 1, 1, temp).float()
                    else:
                        if isinstance(out,list):
                            pseudo_boxes_reg, pseudo_boxes_cls = non_max_suppression_pseudo_decouple_multi_view(out, hyp[
                                'bbox_threshold'], hyp['cls_threshold'],
                                                                                                     multi_label=True,
                                                                                                     agnostic=single_cls,
                                                                                                     )
                        else:

                            pseudo_boxes_reg,pseudo_boxes_cls = non_max_suppression_pseudo_decouple(out, hyp['bbox_threshold'],hyp['cls_threshold'],
                                                                      multi_label=True, agnostic=single_cls
                                                                      )

                    unlabel_class_one_hot = pseudo_class_one_hot_post.clone().detach()

                    unlabel_targets_merge_reg = torch.zeros(0, 6).to(device)
                    unlabel_targets_merge_cls = torch.zeros(0, 6).to(device)
                    for batch_ind, (pseudo_box_reg,pseudo_box_cls, pseudo_one_hot) in enumerate(
                            zip(pseudo_boxes_reg,pseudo_boxes_cls, pseudo_class_one_hot_post)):
                        # two stage filters
                        unlabel_target_probs = torch.gather(pseudo_one_hot.sigmoid(), 0, pseudo_box_cls[:, -1].long())
                        unlabel_target_vaild = unlabel_target_probs > hyp['mls_filter_threshold']
                        pseudo_box_cls = torch.masked_select(pseudo_box_cls, unlabel_target_vaild.unsqueeze(1)).view(-1, 6)

                        n_box = pseudo_box_cls.size()[0]
                        unlabel_target_cls = torch.zeros(n_box, 6).to(device)
                        unlabel_target_cls[:, 0] = batch_ind
                        unlabel_target_cls[:, 1] = pseudo_box_cls[:, -1]
                        unlabel_target_cls[:, 2:] = xyxy2xywhn(pseudo_box_cls[:, 0:4], w=unlabel_imgs_weak_aug.size()[2],
                                                           h=unlabel_imgs_weak_aug.size()[3])
                        unlabel_targets_merge_cls = torch.cat([unlabel_targets_merge_cls, unlabel_target_cls])


                        n_box = pseudo_box_reg.size()[0]
                        unlabel_target_reg = torch.zeros(n_box, 6).to(device)
                        unlabel_target_reg[:, 0] = batch_ind
                        unlabel_target_reg[:, 1] = pseudo_box_reg[:, -1]
                        unlabel_target_reg[:, 2:] = xyxy2xywhn(pseudo_box_reg[:, 0:4], w=unlabel_imgs_weak_aug.size()[2],
                                                           h=unlabel_imgs_weak_aug.size()[3])
                        unlabel_targets_merge_reg = torch.cat([unlabel_targets_merge_reg, unlabel_target_reg])
                        semi_label_items += n_box

                    # print(unlabel_targets)
                #debug code, should remove
                # unlabel_targets=unlabel_targets__.to(device, non_blocking=True)
                '''
                2.2
                loss_semi= student.forward_loss(unlabel_imgs_strong, pseudo_unlabel_targets)
                '''
                Bl=label_imgs.size()[0]
                label_unlabel_imgs=torch.cat([label_imgs,unlabel_imgs_strong_aug])
                with amp.autocast(enabled=cuda):
                    # print(ni)
                    pred,pred_mls = model(label_unlabel_imgs)  # forward
                    sup_pred=[p[:Bl] for p in pred]
                    semi_pred=[p[Bl:] for p in pred]
                    sup_pred_mls,semi_pred_mls=pred_mls[:Bl],pred_mls[Bl:]
                    loss, loss_items = compute_loss(sup_pred, label_targets.to(device),sup_pred_mls,label_class_one_hot.to(device))  # loss scaled by batch_size
                    #pay attention: ignoring the regression term
                    semi_loss_cls, semi_loss_items_cls = compute_semi_loss(semi_pred, unlabel_targets_merge_cls.to(device),semi_pred_mls,unlabel_class_one_hot.to(device),cls_only=True)  # loss scaled by batch_size
                    semi_loss_reg, semi_loss_items_reg = compute_semi_loss(semi_pred, unlabel_targets_merge_reg.to(device),semi_pred_mls,unlabel_class_one_hot.to(device),box_only=True)  # loss scaled by batch_size
                    semi_loss=semi_loss_cls+semi_loss_reg
                    semi_loss_items=semi_loss_items_cls+semi_loss_items_reg
                    # print(semi_loss_items)
                    if RANK != -1:
                        semi_loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                        loss*=WORLD_SIZE
                    if opt.quad:
                        semi_loss *= 4.
                        loss *= 4
            # print(model)

            '''
            2.3
            loss_total= loss_sup+loss_semi*semi_weights
            '''
            loss=loss+semi_loss*hyp['semi_loss_weight']
            semi_loss_items*=hyp['semi_loss_weight']
            # with torch.autograd.set_detect_anomaly(True):
            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
                last_opt_step = ni

            '''
            9.15 改到这里为止
            '''
            #callback needs double check !

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mloss_semi = (mloss_semi * i + semi_loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 9) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, *mloss_semi,label_targets.shape[0], semi_label_items))
                callbacks.run('on_train_batch_end', ni, model, label_imgs, label_targets, label_paths, plots, opt.sync_bn)
            # end batch ------------------------------------------------------------------------------------------------
            if RANK in [-1, 0] and i==len(pbar)-1:
                break
            # if i>100:
            #     break
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in [-1, 0] and epoch % opt.val_per_epoch==0 and epoch>0:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            if ema is not None:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP

                results, maps, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz,
                                           model=deepcopy(model_teacher),#ema.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           save_json=is_coco and final_epoch,
                                           verbose=nc < 50 and final_epoch,
                                           plots=plots and final_epoch,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model_student': deepcopy(de_parallel(model)).half(),
                        'model': deepcopy(de_parallel(model_teacher)).half(),
                        'ema': deepcopy(ema.ema).half() if ema is not None else None,
                        'updates': ema.updates if ema is not None else None,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        if not evolve:
            if is_coco:  # COCO dataset
                for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(m, device,model_name='model_teacher').half(),
                                            iou_thres=0.7,  # NMS IoU threshold for best pycocotools results
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            save_json=True,
                                            plots=False)
            # Strip optimizers
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
        callbacks.run('on_train_end', last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--imagenet-weights', type=str, default=None, help='imagenet weights path')
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.semi.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--val-per-epoch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--lr-schedule', default='constant',help='linear LR',choices=['constant','cosine','linear','step'])
    parser.add_argument('--use-ema', action='store_true', help='if we use ema or not')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    set_logging(RANK)
    if RANK in [-1, 0]:
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
        check_git_status()
        check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=['thop'])

    # Resume
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp)  # check YAMLs
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = 'runs/evolve'
            opt.exist_ok = opt.resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        from datetime import timedelta
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)

            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
