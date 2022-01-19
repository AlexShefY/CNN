# !pip install neptune-client qhoptim
import torch
from routines import run

config = {
    'project_name': 'mlxa/CNN',
    'api_token': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==',
    'register_run': False,

    'model': 'M7S1()',
    'batch_size': 64,
    'train': 'train_v2.bin',
    'val': 'val_v2.bin',
    'test': None,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'optimizer': 'QHAdam',
    'lr': 0.002,
    'wd': 1e-6,
    'dropout': 0.1,
    'beta1': 0.9,
    'beta2': 0.999,
    'nu1': 0.7,
    'nu2': 1.0,
    'epochs': 5,

    'lr_scheduler': 'OneCycleLR',
    'max_lr': 1
}

run(config)