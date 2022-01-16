# pip install neptune-client qhoptim

from time import time
import torch
from torchvision import models
import plotly.express as px
from neptune.new.types import File
from tqdm import trange
from torch import nn
from qhoptim.pyt import QHM, QHAdam
from data import run, device, build_dataloaders
from routines import train_epoch, test, solve_test
# def train_epoch(model, dataloader, optimizer, logging=None, interval=None)
# def test(models, dataloader, see_orig=True, aug_iters=0, loss_fn=nn.CrossEntropyLoss())
# def solve_test(model, dataloader, name)

config = {
    'lr': 0.01,
    'wd': 1e-6,
    'dropout': 0.1,
    'beta1': 0.9,
    'beta2': 0.999,
    'nu1': 0.7,
    'nu2': 1.0,
    'epochs': 50,
    'gamma': 0.95
}
run['parameters'] = config
print(f'run config is', *config.items(), flush=True)

train_loader, val_loader, test_loader = build_dataloaders(batch_size=64, download=False)

def train_model(model, optimizer, scheduler, epochs=10**9):
    pathx, pathy = [], []
    train_id = int(time())
    print(f'started train #{train_id}', flush=True)

    for epoch in trange(epochs):
        def train_logging(batch, loss, hx, hy):
            pathx.append(hx)
            pathy.append(hy)
            step = epoch + (batch + 1) / len(train_loader)
            run['train/epoch'].log(step, step=step)
            run['train/train_loss'].log(loss, step=step)
            run['train/path'] = File.as_html(px.line(x=pathx, y=pathy, markers=True))
        def test_logging(loss, acc):
            step = epoch + 1
            run['train/epoch'].log(step, step=step)
            run['train/val_loss'].log(loss, step=step)
            run['train/val_acc'].log(acc, step=step)

        train_epoch(model, train_loader, optimizer, train_logging, 25)
        scheduler.step()
        test_logging(*test([model], val_loader, True, 0))
        name = f'{train_id}_{epoch}'
        solve_test([model], test_loader, name, True, 0)

model = models.resnet18()
model.fc = nn.Linear(512, 10)
for name, module in model.named_children():
    if name == 'fc':
        continue
    module = nn.Sequential(module, nn.Dropout(p=config['dropout']))
model = model.to(device)
optimizer = QHAdam(model.parameters(),
    lr=config['lr'],
    betas=(config['beta1'], config['beta2']),
    nus=(config['nu1'], config['nu2']),
    weight_decay=config['wd'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['gamma'])
train_model(model, optimizer, scheduler, config['epochs'])


for name, module in model.named_children():
    if name == 'fc':
        continue
    module = nn.Sequential(module, nn.Dropout(p=config['dropout']))
model = model.to(device)

optimizer = QHAdam(model.parameters(),
    lr=config['lr'],
    betas=(config['beta1'], config['beta2']),
    nus=(config['nu1'], config['nu2']),
    weight_decay=config['wd'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['gamma'])
train_model(model, optimizer, scheduler, config['epochs'])
