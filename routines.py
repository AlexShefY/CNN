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

def test(model, dataloader, loss_fn=nn.CrossEntropyLoss()):
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        test_loss = test_loss + loss_fn(pred, y)
        correct += (pred.argmax(dim=-1) == y).to(torch.float).mean()
    return test_loss / num_batches, correct / num_batches

def write_solution(filename, labels):
    with open(filename, 'w') as solution:
        print('Id,Category', file=solution)
        for i, label in enumerate(labels):
            print(f'{i},{label}', file=solution)

def solve_test(model, dataloader, name):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            pred = model(x.to(device))
            predictions.extend(list(pred.argmax(dim=-1).cpu().numpy()))
    print(len(predictions), 'predictions')
    write_solution(f'solution_{name}.csv', predictions)