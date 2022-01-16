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

def make_predictions(models, x, see_orig=True, aug_iters=0):
    pred = sum(model(x) for model in models) if see_orig else torch.zeros_like(models[0](x))
    for iter_num in range(aug_iters):
        pred = pred + sum(model(autoaug(x)) for model in models)
    return pred

def test(models, dataloader, see_orig=True, aug_iters=0, loss_fn=nn.CrossEntropyLoss()):
    for model in models:
        model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = make_predictions(models, x)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(dim=-1) == y).to(torch.float).mean()
    return test_loss / num_batches, correct / num_batches

def write_solution(filename, labels):
    with open(filename, 'w') as solution:
        print('Id,Category', file=solution)
        for i, label in enumerate(labels):
            print(f'{i},{label}', file=solution)

def solve_test(models, dataloader, name):
    for model in models:
        model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            pred = make_predictions(models=models, x=x.to(device), see_orig=True, aug_iters=0)
            predictions.extend(list(pred.argmax(dim=-1).cpu().numpy()))
    print(len(predictions), 'predictions')
    torch.save(model, f'model_{name}.p')
    write_solution(f'solution_{name}.csv', predictions)
            