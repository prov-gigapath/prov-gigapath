import os
import math
import torch
import pickle
import random
import numpy as np
import pandas as pd
import torch.optim as optim

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)
        

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def seed_torch(device, seed=7):
    # ------------------------------------------------------------------------------------------
    # References:
    # HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    # ------------------------------------------------------------------------------------------
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_exp_code(args):
    '''Get the experiment code for the current run.'''
    # set up the model code
    model_code = 'eval'
    if len(args.pretrained) > 0:
        model_code += '_pretrained'
    if args.freeze:
        model_code += '_freeze'
        
    # set up the task code
    task_code = args.task
    if args.pat_strat:
        task_code += '_pat_strat'

    # set up the experiment code
    exp_code = '{model_code}_{task_code}'

    return model_code, task_code, exp_code.format(model_code=model_code, task_code=task_code)


def pad_tensors(imgs, coords):
    # ------------------------------------------------------------------------------------------
    # References:
    # mae: https://github.com/facebookresearch/mae/tree/main
    # ------------------------------------------------------------------------------------------
    max_len = max([t.size(0) for t in imgs])  # get the maximum length
    padded_tensors = []  # list to store all padded tensors
    padded_coords = []  # list to store all padded coords
    masks = []  # list to store all masks
    for i in range(len(imgs)):
        # tensor: [L, d]
        tensor = imgs[i]
        # coords: [L, 2]
        coord = coords[i]
        N_i = tensor.size(0)  # get the original length
        # create a new tensor of shape (max_len, d) filled with zeros
        padded_tensor = torch.zeros(max_len, tensor.size(1))
        padded_coord = torch.zeros(max_len, 2)
        # create a new tensor of shape (max_len) filled with zeros for mask
        mask = torch.zeros(max_len)
        # place the original tensor into the padded tensor
        padded_tensor[:N_i] = tensor
        padded_coord[:N_i] = coord
        # the mask is filled with ones at the same indices as the original tensor
        mask[:N_i] = torch.ones(N_i)
        padded_tensors.append(padded_tensor)
        padded_coords.append(padded_coord)
        masks.append(mask)

    # concatenate all tensors along the 0th dimension
    padded_tensors = torch.stack(padded_tensors)
    padded_coords = torch.stack(padded_coords)
    masks = torch.stack(masks)
    # convert masks to bool type
    masks = masks.bool()
    return padded_tensors, padded_coords, masks


def slide_collate_fn(samples):
    '''Separate the inputs and targets into separate lists
    Return value {imgs: [N, L, 256, 384], pad_mask: [N, L]}'''
    image_list = [s['imgs'] for s in samples]
    img_len_list = [s['imgs'].size(0) for s in samples]
    coord_list = [s['coords'] for s in samples]
    label_list = [s['labels'] for s in samples]
    slide_id_list = [s['slide_id'] for s in samples]
    labels = torch.stack(label_list)
    pad_imgs, pad_coords, pad_mask = pad_tensors(image_list, coord_list)
    
    data_dict = {'imgs': pad_imgs, 
            'img_lens': img_len_list,
            'coords': pad_coords,
            'slide_id': slide_id_list,
            'pad_mask': pad_mask,
            'labels': labels}
    return data_dict


def get_splits(df: pd.DataFrame, 
               val_r: float=0.1, test_r: float=0.2, 
               fold: int=0, 
               split_dir: str='', 
               fetch_splits: bool=True, 
               prop: int=1, 
               split_key='slide_id', 
               **kwargs) -> Tuple[List[str], List[str], List[str]]:
    '''Get the splits for the dataset. The default train/val/test split is 70/10/20.'''
    # get the split names
    files = os.listdir(split_dir)
    train_name, val_name, test_name = f'train_{fold}.csv', f'val_{fold}.csv', f'test_{fold}.csv'
    # check split_key is in the columns
    assert split_key in df.columns, f'{split_key} not in the columns of the dataframe'
    # make sure the dataset exists, otherwise create new datasets
    if train_name not in files or val_name not in files or test_name not in files or not fetch_splits:
        samples = df.drop_duplicates(split_key)[split_key].to_list()
        train_samples, temp_samples = train_test_split(samples, test_size=(val_r + test_r), random_state=fold)
        if val_r > 0:
            val_samples, test_samples = train_test_split(temp_samples, test_size=(test_r / (val_r + test_r)), random_state=fold)
        else:
            val_samples, test_samples = [], temp_samples
        train_data = df[df[split_key].isin(train_samples)]
        val_data = df[df[split_key].isin(val_samples)]
        test_data = df[df[split_key].isin(test_samples)]

        # sample the training data
        if prop > 0:
            train_data = train_data.sample(frac=prop, random_state=fold).reset_index(drop=True)
        # save datasets
        train_data.to_csv(os.path.join(split_dir, train_name))
        val_data.to_csv(os.path.join(split_dir, val_name))
        test_data.to_csv(os.path.join(split_dir, test_name))
    # load the dataframe
    train_splits = pd.read_csv(os.path.join(split_dir, train_name))[split_key].to_list()
    val_splits = pd.read_csv(os.path.join(split_dir, val_name))[split_key].to_list()
    test_splits = pd.read_csv(os.path.join(split_dir, test_name))[split_key].to_list()

    return train_splits, val_splits, test_splits


def get_loader(train_dataset, val_dataset, test_dataset, 
               task_config, weighted_sample=False, 
               batch_size=1, num_workers=10, seed=0, 
               **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''Get the dataloader for the dataset.'''
    if weighted_sample and not task_config.get('setting', 'multi_class') == 'multi_label':
        # get the weights for each class, we only do this for multi-class classification
        N = len(train_dataset)
        weights = {}
        for idx in range(N):
            label = int(train_dataset.labels[idx][0])
            if label not in weights: weights[label] = 0
            weights[label] += 1.0 / N
        for l in weights.keys(): weights[l] = 1.0 / weights[l]
        sample_weights = [weights[int(train_dataset.labels[i][0])] for i in range(N)]
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    else:
        train_sampler = RandomSampler(train_dataset)

    # set up generator and worker_init_fn
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # if it's the sequence based model, we use the slide collate function to pad
    train_loader = DataLoader(train_dataset, \
                            num_workers=num_workers, \
                            batch_size=batch_size, sampler=train_sampler, \
                            generator=g, worker_init_fn=seed_worker, \
                            collate_fn=slide_collate_fn)
    val_loader = DataLoader(val_dataset, \
                            num_workers=num_workers, \
                            batch_size=1, sampler=SequentialSampler(val_dataset), \
                            worker_init_fn=seed_worker, \
                            collate_fn=slide_collate_fn) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, \
                            num_workers=num_workers, \
                            batch_size=1, sampler=SequentialSampler(test_dataset), \
                            worker_init_fn=seed_worker, \
                            collate_fn=slide_collate_fn) if test_dataset is not None else None

    return train_loader, val_loader, test_loader


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    # ------------------------------------------------------------------------------------------
    # References:
    # BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    # ------------------------------------------------------------------------------------------
    param_group_names = {}
    param_groups = {}

    num_layers = model.slide_encoder.encoder.num_layers + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        if 'mask_token' in n or 'slide_encoder.decoder' in n:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        
        layer_id = get_layer_id(n, num_layers)

        group_name = n + "_%d_%s" % (layer_id + 1, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id(name, num_layers):
    # ------------------------------------------------------------------------------------------
    # References:
    # BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    # ------------------------------------------------------------------------------------------
    if 'cls_token' in name or 'pos_embed' in name:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('slide_encoder.encoder.layers'):
        return int(name.split('.')[3]) + 1
    else:
        return num_layers


def adjust_learning_rate(optimizer, epoch, args):
    # ------------------------------------------------------------------------------------------
    # References:
    # mae: https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
    # ------------------------------------------------------------------------------------------
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_optimizer(args, model):
    '''Set up the optimizer for the model.'''
    param_groups = param_groups_lrd(model, args.optim_wd,
        layer_decay=args.layer_decay)
    # make the optimizer
    optim_func = torch.optim.AdamW if args.optim == 'adamw' else torch.optim.Adam
    optimizer = optim_func(param_groups, lr=args.lr)

    return optimizer


def get_loss_function(task_config: dict):
    '''Get the loss function based on the task configuration.'''
    task_setting = task_config.get('setting', 'multi_class')
    if task_setting == 'multi_label':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif task_setting == 'multi_class' or task_setting == 'binary':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    return loss_fn


def get_records_array(record_len: int, n_classes) -> dict:
    '''Get the records array based on the task configuration.'''
    record = {
        'prob': np.zeros((record_len, n_classes), dtype=np.float32),
        'label': np.zeros((record_len, n_classes), dtype=np.float32),
        'loss': 0.0,
    }
    return record


class Monitor_Score:
    # ------------------------------------------------------------------------------------------
    # References:
    # MCAT: https://github.com/mahmoodlab/MCAT/blob/master/utils/core_utils.py
    # ------------------------------------------------------------------------------------------
    def __init__(self):
        self.best_score = None

    def __call__(self, val_score, model, ckpt_name:str='checkpoint.pt'):

        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def log_writer(log_dict: dict, step: int, report_to: str='tensorboard', writer=None):
    '''Log the dictionary to the writer.'''
    if report_to == 'tensorboard':
        for k, v in log_dict.items():
            writer.add_scalar(k, v, step)
    elif report_to == 'wandb':
        writer.log(log_dict, step=step)
    else:
        raise NotImplementedError