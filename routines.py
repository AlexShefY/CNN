import numpy as np 
import plotly.express as px
import torch.nn as nn
from autoaug import autoaug
from stats import complex_hash
from data import Plug
import torch  
import static as st
from time import time  
from tqdm import trange, tqdm

def train_epoch(model, dataloader, optimizer, logging=None, interval=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch, (x, y) in tqdm(enumerate(dataloader)):
        x, y = x.to(st.device), y.to(st.device)
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
        x, y = x.to(st.device), y.to(st.device)
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
    if dataloader is None:
        print('no test loader, exit', flush=True)
        return
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            pred = model(x.to(st.device))
            predictions.extend(list(pred.argmax(dim=-1).cpu().numpy()))
    print('saved', len(predictions), 'predictions to ', name)
    write_solution(f'{name}.csv', predictions)

def train_model(model, optimizer, scheduler, epochs=10**9, predicate=lambda loss, acc: acc > 93):
    from neptune.new.types import File
    global run, train_loader, val_loader, test_loader

    pathx, pathy = [], []
    train_id = st.run.split('/')[-1] if type(st.run) is not Plug else int(time())
    print(f'started train #{train_id}', flush=True)

    for epoch in range(epochs):
        def train_logging(batch, loss, hx, hy):
            pathx.append(hx)
            pathy.append(hy)
            step = epoch + (batch + 1) / len(st.train_loader)
            st.run['train/epoch'].log(step, step=step)
            st.run['train/train_loss'].log(loss, step=step)
            st.run['train/path'] = File.as_html(px.line(x=pathx, y=pathy))
        def test_logging(loss, acc):
            step = epoch + 1
            st.run['train/epoch'].log(step, step=step)
            st.run['train/val_loss'].log(loss, step=step)
            st.run['train/val_acc'].log(acc, step=step)
            print(f'step: {step}, loss: {loss}, acc: {acc}, hx: {pathx[-1] or None}, hy: {pathy[-1] or None}')
            if predicate(loss, acc):
                name = f'{train_id}_{epoch}'
                save_to_zoo(model, name, val_loss, val_acc)
                solve_test(model, st.test_loader, f'solution_{model.loader()}_{name}')
        train_epoch(model, st.train_loader, optimizer, train_logging, 25)
        scheduler.step()
        test_logging(*test(model, st.val_loader))
        
