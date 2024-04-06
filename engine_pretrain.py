# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# Un-Mix: https://github.com/szq0214/Un-Mix
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    teach_model,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (sample_1, sample_2, target_1, target_2) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # mix two images in the same batch.
        sample_1.to(device, non_blocking=True)
        lam = np.random.beta(1.0, 1.0)

        mixed_images = lam * sample_1 + (1 - lam) * sample_2 
        im1 = sample_1
        im2 = sample_2
        
        #lam = np.random.beta(1.0, 1.0)
        
        weak_mix = np.argmin([lam, 1-lam]) 
        metric_logger.update(lam=lam)
        
        inps = (mixed_images.to(device, non_blocking=True), im1.to(device, non_blocking=True),im2.to(device, non_blocking=True))
        latent1, mask, ids_shuffle, ids_restore = teach_model.forward_encoder(inps[0], mask_ratio=args.mask_ratio)  
        latent1 = teach_model.decoder_embed(latent1)
        latent2, mask, ids_shuffle, ids_restore = teach_model.forward_encoder(inps[0], mask_ratio=args.mask_ratio, ids_shuffle=ids_shuffle, ids_restore=ids_restore)     
        latent2 = teach_model.decoder_embed(latent2)
        loss, _, _ = model(inps, mask_ratio=args.mask_ratio, weak_idx=weak_mix,ids_shuffle=ids_shuffle, ids_restore=ids_restore,teach_latent=(latent1, latent2))

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}