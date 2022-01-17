from os import system
import torch
from data import run, device
from models import *

def save_to_zoo(model, loader, name, val_acc=None, val_loss=None):
	print('save_to_zoo', loader, name, val_acc, val_loss)
	model.cpu()
	torch.save({
		'state_dict': model.state_dict(),
		'loader': loader,
		'val_acc': val_acc,
		'val_loss': val_loss
		}, f'zoo/{loader}_{name}.p')
	model.to(device)

def load_from_zoo(name):
	d = torch.load(f'zoo/{name}.p')
	print(d['loader'])
	model = eval(d['loader'])
	model.load_state_dict(d['state_dict'])
	return model
