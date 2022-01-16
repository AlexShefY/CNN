from data import run, device
import numpy as np 
import plotly.express as px
import torch.nn as nn
from autoaug import autoaug
from stats import complex_hash
import torch  

def train_epoch(model, dataloader, optimizer, logging=None, interval=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(autoaug(x)), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if logging and (not interval or batch % interval == 0):
            logging(batch, loss.item(), *complex_hash(model, 2))

def make_predictions(models, x, coefs=None):
    if not coefs:
        coefs = [torch.ones(1).to(device) for m in models]
    return torch.stack([model(x) * k for model, k in zip(models, coefs)]).sum(dim=0)

def test(models, dataloader, coefs=None, loss_fn=nn.CrossEntropyLoss()):
    for model in models:
        model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    def helper():
        nonlocal test_loss, correct
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = make_predictions(models, x, coefs)
            test_loss = test_loss + loss_fn(pred, y)
            correct += (pred.argmax(dim=-1) == y).to(torch.float).mean()
    if not coefs:
        with torch.no_grad():
           helper()
    else:
        helper()
    return test_loss / num_batches, correct / num_batches

def write_solution(filename, labels):
    with open(filename, 'w') as solution:
        print('Id,Category', file=solution)
        for i, label in enumerate(labels):
            print(f'{i},{label}', file=solution)

def solve_test(models, dataloader, name, coefs=None):
    for model in models:
        model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            pred = make_predictions(models, x.to(device), coefs)
            predictions.extend(list(pred.argmax(dim=-1).cpu().numpy()))
    print(len(predictions), 'predictions')
    write_solution(f'solution_{name}.csv', predictions)
