import os
import sys
from pathlib import Path

# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))

import time
import wandb
import torch
import numpy as np
import torch.utils.tensorboard as tensorboard

from gigapath.classification_head import get_model
from metrics import calculate_metrics_with_task_cfg
from utils import (get_optimizer, get_loss_function, \
                  Monitor_Score, get_records_array,
                  log_writer, adjust_learning_rate)


def train(dataloader, fold, args):
    train_loader, val_loader, test_loader = dataloader
    # set up the writer
    writer_dir = os.path.join(args.save_dir, f'fold_{fold}', 'tensorboard')
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir, exist_ok=True)

    # set up the writer
    writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)
    # set up writer
    if "wandb" in args.report_to:
        wandb.init(
            project=args.exp_code,
            name=args.exp_code + '_fold_' + str(fold),
            id='fold_' + str(fold),
            tags=[],
            config=vars(args),
        )
        writer = wandb
    elif "tensorboard" in args.report_to:
        writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)

    # set up the model
    model = get_model(**vars(args))
    model = model.to(args.device)
    # set up the optimizer
    optimizer = get_optimizer(args, model)
    # set up the loss function
    loss_fn = get_loss_function(args.task_config)
    # set up the monitor
    monitor = Monitor_Score()
    # set up the fp16 scaler
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')

    print('Training on {} samples'.format(len(train_loader.dataset)))
    print('Validating on {} samples'.format(len(val_loader.dataset))) if val_loader is not None else None
    print('Testing on {} samples'.format(len(test_loader.dataset))) if test_loader is not None else None
    print('Training starts!')

    # test evaluate function
    # val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, 0, args)

    val_records, test_records = None, None

    for i in range(args.epochs):
        print('Epoch: {}'.format(i))
        train_records = train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, i, args)

        if val_loader is not None:
            val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, i, args)

            # update the writer for train and val
            log_dict = {'train_' + k: v for k, v in train_records.items() if 'prob' not in k and 'label' not in k}
            log_dict.update({'val_' + k: v for k, v in val_records.items() if 'prob' not in k and 'label' not in k})
            log_writer(log_dict, i, args.report_to, writer)
            # update the monitor scores
            scores = val_records['macro_auroc']

        if args.model_select == 'val' and val_loader is not None:
            monitor(scores, model, ckpt_name=os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt"))
        elif args.model_select == 'last_epoch' and i == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt"))

    # load model for test
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt")))
    # test the model
    test_records = evaluate(test_loader, model, fp16_scaler, loss_fn, i, args)
    # update the writer for test
    log_dict = {'test_' + k: v for k, v in test_records.items() if 'prob' not in k and 'label' not in k}
    log_writer(log_dict, fold, args.report_to, writer)
    wandb.finish() if "wandb" in args.report_to else None

    return val_records, test_records


def train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, args):
    model.train()
    # set the start time
    start_time = time.time()

    # monitoring sequence length
    seq_len = 0

    # setup the records
    records = get_records_array(len(train_loader), args.n_classes)

    for batch_idx, batch in enumerate(train_loader):
        # we use a per iteration lr scheduler
        if batch_idx % args.gc == 0 and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)

        # load the batch and transform this batch
        images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
        images = images.to(args.device, non_blocking=True)
        img_coords = img_coords.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True).long()

        # add the sequence length
        seq_len += images.shape[1]

        with torch.cuda.amp.autocast(dtype=torch.float16 if args.fp16 else torch.float32):

            # get the logits
            logits = model(images, img_coords)
            # get the loss
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()
            else:
                label = label.squeeze(-1).long()

            loss = loss_fn(logits, label)
            loss /= args.gc

            if fp16_scaler is None:
                loss.backward()
                # update the parameters with gradient accumulation
                if (batch_idx + 1) % args.gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                fp16_scaler.scale(loss).backward()
                # update the parameters with gradient accumulation
                if (batch_idx + 1) % args.gc == 0:
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                    optimizer.zero_grad()

        # update the records
        records['loss'] += loss.item() * args.gc

        if (batch_idx + 1) % 20 == 0:
            time_per_it = (time.time() - start_time) / (batch_idx + 1)
            print('Epoch: {}, Batch: {}, Loss: {:.4f}, LR: {:.4f}, Time: {:.4f} sec/it, Seq len: {:.1f}, Slide ID: {}' \
                  .format(epoch, batch_idx, records['loss']/batch_idx, optimizer.param_groups[0]['lr'], time_per_it, \
                          seq_len/(batch_idx+1), batch['slide_id'][-1] if 'slide_id' in batch else 'None'))

    records['loss'] = records['loss'] / len(train_loader)
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss))
    return records


def evaluate(loader, model, fp16_scaler, loss_fn, epoch, args):
    model.eval()

    # set the evaluation records
    records = get_records_array(len(loader), args.n_classes)
    # get the task setting
    task_setting = args.task_config.get('setting', 'multi_class')
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # load the batch and transform this batch
            images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
            images = images.to(args.device, non_blocking=True)
            img_coords = img_coords.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True).long()

            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                # get the logits
                logits = model(images, img_coords)
                # get the loss
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    label = label.squeeze(-1).float()
                else:
                    label = label.squeeze(-1).long()
                loss = loss_fn(logits, label)

            # update the records
            records['loss'] += loss.item()
            if task_setting == 'multi_label':
                Y_prob = torch.sigmoid(logits)
                records['prob'][batch_idx] = Y_prob.cpu().numpy()
                records['label'][batch_idx] = label.cpu().numpy()
            elif task_setting == 'multi_class' or task_setting == 'binary':
                Y_prob = torch.softmax(logits, dim=1).cpu()
                records['prob'][batch_idx] = Y_prob.numpy()
                # convert label to one-hot
                label_ = torch.zeros_like(Y_prob).scatter_(1, label.cpu().unsqueeze(1), 1)
                records['label'][batch_idx] = label_.numpy()

    records.update(calculate_metrics_with_task_cfg(records['prob'], records['label'], args.task_config))
    records['loss'] = records['loss'] / len(loader)

    if task_setting == 'multi_label':
        info = 'Epoch: {}, Loss: {:.4f}, Micro AUROC: {:.4f}, Macro AUROC: {:.4f}, Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}'.format(epoch, records['loss'], records['micro_auroc'], records['macro_auroc'], records['micro_auprc'], records['macro_auprc'])
    else:
        info = 'Epoch: {}, Loss: {:.4f}, AUROC: {:.4f}, ACC: {:.4f}, BACC: {:.4f}'.format(epoch, records['loss'], records['macro_auroc'], records['acc'], records['bacc'])
        for metric in args.task_config.get('add_metrics', []):
            info += ', {}: {:.4f}'.format(metric, records[metric])
    print(info)
    return records