# !pip install neptune-client qhoptim
import torch
from torchvision import models
import plotly.express as px
import neptune.new as neptune
from tqdm import trange
from torch import nn
from qhoptim.pyt import QHM, QHAdam

from data import build_dataloader, Plug
from routines import train_model
from models import *
import static as st
# st.token = ...

def init(smoke=False, config=dict()):
    st.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.project = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')
    st.run = Plug() if smoke else neptune.init(project=st.project_name, api_token=st.token)
    st.run['parameters'] = config
    st.train_loader = build_dataloader('train_v2.bin' if smoke else 'train_v1.bin', batch_size=64, shuffle=True)
    st.val_loader = build_dataloader('val_v2.bin' if smoke else 'val_v2.bin', batch_size=64, shuffle=True)
    st.test_loader = None if smoke else build_dataloader('test_v2.bin', batch_size=64, shuffle=True)
    print(f'run config is', *config.items(), flush=True)

config = {
    'lr': 0.002,
    'wd': 1e-6,
    'dropout': 0.1,
    'beta1': 0.9,
    'beta2': 0.999,
    'nu1': 0.7,
    'nu2': 1.0,
    'epochs': 5,
    'gamma': 0.9
}

init(smoke=True, config=config)

model = M7S1().to(st.device)
print(model)

optimizer = QHAdam(model.parameters(),
    lr=config['lr'],
    betas=(config['beta1'], config['beta2']),
    nus=(config['nu1'], config['nu2']),
    weight_decay=config['wd'])

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['gamma'])

with torch.autograd.profiler.profile() as prof:
    train_model(model, optimizer, scheduler, config['epochs'])