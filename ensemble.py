from os import chdir, listdir
from torchvision import models
import torch
import torch.nn as nn
from data import build_dataloaders, device

train_loader, val_loader, test_loader = build_dataloaders(batch_size=64, download=False)

def make_predictions(models, x, coefs=None):
    if not coefs:
        coefs = [torch.ones(1) for m in models]
    return torch.stack([model(x) * k for model, k in zip(models, coefs)]).sum(dim=0)

def test(models, dataloader, coefs=None, loss_fn=nn.CrossEntropyLoss()):
    for model in models:
        model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = make_predictions(models, x, coefs)
        test_loss = test_loss + loss_fn(pred, y)
        correct += (pred.argmax(dim=-1) == y).to(torch.float).mean()
    return test_loss / num_batches, correct / num_batches

chdir('zoo')
zoo = []

for name in listdir():
	if name[-2:] != '.p':
		continue
	if 'resnet' in name:
		model = models.resnet18()
		model.fc = nn.Linear(512, 10)
		model.load_state_dict(torch.load(name, map_location=device))
	else:
		model = torch.load(name, map_location=device)
	zoo.append(model)
	print(name, test([model], val_loader), flush=True)
	break

coefs = [torch.ones(1, requires_grad=True) for m in zoo]
opt = torch.optim.Adam(coefs)

for i in range(100):
	opt.zero_grad()
	loss, acc = test(zoo, val_loader, coefs)
	loss.backward()
	print('c', *[e.item() for e in coefs], flush=True)
	print('g', *[e.grad.item() for e in coefs], flush=True)
	opt.step()
