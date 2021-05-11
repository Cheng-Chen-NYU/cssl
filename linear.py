import argparse
import pandas as pd
from tqdm import tqdm
from thop import profile, clever_format

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from tools.cvrlDataset import train_transform, test_transform
from model.model import MoCov1, MoCov2, SimCLRv1, SimCLRv2

class Net(nn.Module):
    def __init__(self, model_name, num_class):
        super(Net, self).__init__()

        self.model_name = model_name

        if model_name == 'mocov1':
            # encoder
            self.f = MoCov1().encoder_q
            # classifier
            self.fc = nn.Linear(512, num_class, bias=True)
        elif model_name == 'mocov2':
            # encoder
            self.f = MoCov2().encoder_q
            # classifier
            self.fc = nn.Linear(512, num_class, bias=True)
        elif model_name == 'simclrv1':
            # encoder
            self.f = SimCLRv1().f
            # classifier
            self.fc = nn.Linear(2048, num_class, bias=True)
        elif model_name == 'simclrv2':
            # encoder
            base = SimCLRv2()
            self.f = nn.Sequential(base.f, nn.Flatten(1), base.g1)
            # classifier
            self.fc = nn.Linear(3072, num_class, bias=True)
        else:
            assert(False)

    def forward(self, x):
        if self.model_name.startswith('moco'):
            x = self.f.f(x)
        else:
            x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# train or test for one epoch
def train_val(model_name, model, data_loader, args, is_train):
    
    if model_name.startswith('simclr'):
        scheduler = None
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        if model_name == 'simclrv2':
            model.load_state_dict({k.replace('f.','f.0.'):v for k,v in torch.load(model_path)['state_dict'].items()}, strict=False)
        else:
            model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
    else:
        model = model.cuda()
        model.load_state_dict({k.replace('encoder_q.','f.'):v for k,v in torch.load(model_path)['state_dict'].items()}, strict=False)

    if model_name.startswith('simclr'):
        for param in model.module.f.parameters():
            param.requires_grad = False

        train_optimizer = optim.Adam(model.module.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    else:
        for param in model.f.parameters():
            param.requires_grad = False

        train_optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model.train() if is_train else model.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = model(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_name', type=str, help='model name', default='mocov1')
    parser.add_argument('--model_path', type=str, default='', help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--weight_decay', type=float, help='weight decay factor', default=1e-6)
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    model_name, model_path, batch_size, epochs = args.model_name, args.model_path, args.batch_size, args.epochs
    
    train_data = CIFAR10(root='data/', train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_data = CIFAR10(root='data/', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = Net(model_name, num_class=len(train_data.classes)).cuda()

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [], 'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model_name, model, train_loader, args, True)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model_name, model, test_loader, args, False)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('train_log/linear_statistics.csv', index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'train_log/linear_model.pth')


