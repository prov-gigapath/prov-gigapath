import os
import io
import argparse
import zipfile
import pandas as pd

import torch
import itertools
import numpy as np
from torch import nn

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.utils.tensorboard as tensorboard
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support


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


# Make argparser
argparser = argparse.ArgumentParser(description='Linear Probe')
# Dataset
argparser.add_argument('--dataset_csv',            type=str, default='', help='The csv file indicating input samples and labels')
argparser.add_argument('--input_path',          type=str, default='', help='The input embedding files')
argparser.add_argument('--embed_dim',           type=int, default=1536, help='The dimension of the embeddings')
# Training
argparser.add_argument('--batch_size',          type=int, default=512, help='Batch size')
argparser.add_argument('--train_iters',         type=int, default=12500, help='Number of epochs')
argparser.add_argument('--lr',                  type=float, default=0.01, help='Learning rate')
argparser.add_argument('--min_lr',              type=float, default=0.0, help='Minimum learning rate')
argparser.add_argument('--optim',               type=str, default='sgd', help='Optimizer')
argparser.add_argument('--momentum',            type=float, default=0.0, help='Momentum')
argparser.add_argument('--weight_decay',        type=float, default=0.0, help='Weight decay')
argparser.add_argument('--eval_interval',       type=int, default=10000, help='Evaluation interval')
argparser.add_argument('--model_select',        type=str, default='best', help='Model selection')
argparser.add_argument('--num_workers',         type=int, default=10, help='Number of workers')
argparser.add_argument('--seed',                type=int, default=42, help='Random seed')
argparser.add_argument('--z_score',             action='store_true', default=False, help='Whether use z-score normalization')
# Output
argparser.add_argument('--output_dir',          type=str, default='outputs', help='Output directory')


def to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    '''Convert the labels to one-hot encoding'''
    onehot = np.zeros((labels.shape[0], num_classes))
    onehot[np.arange(labels.shape[0]), labels] = 1
    return onehot


def train(model,
          train_loader,
          val_loader,
          test_loader,
          train_iters,
          lr, min_lr,
          optim,
          weight_decay,
          output_dir,
          eval_interval,
          momentum,
          **kwargs):
    """
    Train the linear probe model.

    Arguments:
    ----------
    model: nn.Module
        Linear probe model
    train_loader: DataLoader
        DataLoader for training set
    val_loader: DataLoader
        DataLoader for validation set
    test_loader: DataLoader
        DataLoader for test set
    train_iters: int
        Number of training iterations
    lr: float
        Learning rate
    min_lr: float
        Minimum learning rate
    optim: str
        Optimizer
    weight_decay: float
        Weight decay
    output_dir: str
        Output directory
    eval_interval: int
        Evaluation interval
    momentum: float
        Momentum
    """
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # set Tensorboard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)

    # Set the optimizer
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer')
    print('Set the optimizer as {}'.format(optim))

    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_iters, eta_min=min_lr)

    # Set the loss function
    criterion = nn.CrossEntropyLoss()

    # Set the infinite train loader
    infinite_train_loader = itertools.cycle(train_loader)

    best_f1 = 0
    # Train the model
    print('Start training')
    for i, (embed, target) in enumerate(infinite_train_loader):

        if i >= train_iters:
            break

        embed, target = embed.to(device), target.to(device)

        # Forward pass
        output = model(embed)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (i + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Iteration [{i}/{train_iters}]\tLoss: {loss.item()}\tLR: {lr}')
            writer.add_scalar('Train Loss', loss.item(), i)
            writer.add_scalar('Learning Rate', lr, i)
        # Print the loss
        if (i + 1) % eval_interval == 0 or (i + 1) == train_iters:
            print(f'Start evaluating ...')
            accuracy, f1, precision, recall, auroc, auprc = evaluate(model, criterion, val_loader, device)
            print(f'Val [{i}/{train_iters}] Accuracy: {accuracy} f1: {f1} Precision: {precision} Recall: {recall} AUROC: {auroc} AUPRC: {auprc}')

            writer.add_scalar('Val Accuracy', accuracy, i)
            writer.add_scalar('Val f1', f1, i)
            writer.add_scalar('Val AUROC', auroc, i)
            writer.add_scalar('Val AUPRC', auprc, i)
            writer.add_scalar('Val Precision', precision, i)
            writer.add_scalar('Val Recall', recall, i)
            writer.add_scalar('Best f1', best_f1, i)

            if f1 > best_f1:
                print('Best f1 increase from {} to {}'.format(best_f1, f1))
                best_f1 = f1
                torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

    # Save the model
    torch.save(model.state_dict(), f'{output_dir}/model.pth')

    if kwargs.get('model_select') == 'best':
        val_f1 = best_f1
        model.load_state_dict(torch.load(f'{output_dir}/best_model.pth'))
    else:
        val_f1 = f1
        model.load_state_dict(torch.load(f'{output_dir}/model.pth'))

    # Evaluate the model
    accuracy, f1, precision, recall, auroc, auprc = evaluate(model, criterion, test_loader, device)
    print(f'Test Accuracy: {accuracy} f1: {f1} Precision: {precision} Recall: {recall} AUROC: {auroc} AUPRC: {auprc}')
    writer.add_scalar('Test Accuracy', accuracy, i)
    writer.add_scalar('Test f1', f1, i)
    writer.add_scalar('Test AUROC', auroc, i)
    writer.add_scalar('Test AUPRC', auprc, i)
    writer.add_scalar('Test Precision', precision, i)
    writer.add_scalar('Test Recall', recall, i)

    f = open(f'{output_dir}/results.txt', 'w')
    f.write(f'Val f1: {val_f1}\n')
    f.write(f'Test f1: {f1} Test AUROC: {auroc} Test AUPRC: {auprc}\n')
    f.close()


def evaluate(model, criterion, val_loader, device):
    """
    Evaluate the linear probe model.

    Arguments:
    model (nn.Module): Linear probe model
    val_loader (DataLoader): DataLoader for validation set
    output_dir (str): Output directory
    """

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_loss = 0

        pred_gather, target_gather = [], []
        for _, (embed, target) in enumerate(val_loader):

            embed, target = embed.to(device), target.to(device)

            # forward pass
            output = model(embed)
            loss = criterion(output, target)
            total_loss += loss.item()
            # gather the predictions and targets
            pred_gather.append(output.cpu().numpy())
            target_gather.append(target.cpu().numpy())

    # calculate the accuracy, AUROC, AUPRC
    pred_gather = np.concatenate(pred_gather)
    target_gather = np.concatenate(target_gather)
    accuracy = (pred_gather.argmax(1) == target_gather).mean()
    # calculate the weighted f1 score
    f1 = f1_score(target_gather, pred_gather.argmax(1), average='weighted')
    # calculate the precision and recall, precision is not accuracy
    precision, recall, _, _ = precision_recall_fscore_support(target_gather, pred_gather.argmax(1), average='macro')

    auroc = roc_auc_score(to_onehot(target_gather, pred_gather.shape[1]), pred_gather, average='macro')
    auprc = average_precision_score(to_onehot(target_gather, pred_gather.shape[1]), pred_gather, average='macro')

    return accuracy, f1, precision, recall, auroc, auprc


def main():
    args = argparser.parse_args()
    print(args)

    # set the random seed
    seed_torch(torch.device('cuda'), args.seed)
    # set the processor
    processor = Processor()
    # load the dataset
    splits = ['train', 'val', 'test']
    train_dataset, val_dataset, test_dataset = [EmbeddingDataset(args.dataset_csv, args.input_path, \
                    split=split, z_score=args.z_score, processor=processor) for split in splits]
    # set num_classes
    args.num_classes = len(train_dataset.label_dict)
    print(f'Train: {len(train_dataset)}\tVal: {len(val_dataset)}\tTest: {len(test_dataset)}')

    # infinite sampler for training
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Load the model
    model = LinearProbe(args.embed_dim, args.num_classes)

    # Train the model
    train(model, train_loader, val_loader, test_loader, **vars(args))


class LinearProbe(nn.Module):

    def __init__(self, embed_dim: int = 1536, num_classes: int = 10):
        super(LinearProbe, self).__init__()

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class EmbeddingDataset(Dataset):
    def __init__(self, dataset_csv: str, zip_path: str, split: str = 'train', z_score=False, processor=None):
        """
        Dataset used for training the linear probe based on the embeddings extracted from the pre-trained model.

        Arugments:
        dataset_csv (str): Path to the csv file containing the embeddings and labels.
        """
        df = pd.read_csv(dataset_csv)
        split_df = df[df['split'] == split]

        self.samples = split_df['input'].tolist()
        self.labels = split_df['label'].tolist()

        # load the embeddings
        self.processor = processor
        self.embeds = processor.load_embeddings_from_zip(zip_path, split)

        label_set = list(set(self.labels))
        label_set.sort()
        self.label_dict = {label: i for i, label in enumerate(label_set)}

        self.z_score = z_score

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample, target = self.samples[index], self.labels[index]
        embed = self.embeds[sample]

        if self.z_score:
            # z-score normalization
            embed = (embed - embed.mean()) / embed.std()

        # convert the label to index
        target = self.label_dict[target]

        return embed, target


class Processor:

    def get_sample_name(self, path):
        return os.path.basename(path).replace('.pt', '')
    
    def load_embeddings_from_zip(self, zip_path, split):
        loaded_tensors = {}
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(len(zip_ref.infolist()))

            for file_info in tqdm(zip_ref.infolist()):
                
                if file_info.filename.endswith('.pt') and split in file_info.filename:
                    file_bytes = zip_ref.read(file_info.filename)
                    byte_stream = io.BytesIO(file_bytes)
                    tensor = torch.load(byte_stream)
                    sample_name = self.get_sample_name(file_info.filename)
                    loaded_tensors[sample_name] = tensor
        return loaded_tensors


if __name__ == '__main__':
    main()