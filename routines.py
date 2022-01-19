import static as st

import torch.nn as nn
from stats import complex_hash
from autoaug import autoaug
from tqdm import trange, tqdm
def train_epoch(model, dataloader, optimizer, logging=None, interval=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch, (x, y) in (enumerate(dataloader)):
        optimizer.zero_grad()
        # with torch.autocast(dtype=torch.bfloat16, device_type="cpu"):
        x, y = x.to(st.device), y.to(st.device)
        loss = loss_fn(model(autoaug(x)), y)
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

import torch
def solve_test(model, dataloader, name):
    assert dataloader is not None
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            pred = model(x.to(st.device))
            predictions.extend(list(pred.argmax(dim=-1).cpu().numpy()))
    print('saved', len(predictions), 'predictions to ', name)
    write_solution(f'{name}.csv', predictions)

from neptune.new.types import File
import plotly.express as px
from git_utils import save_to_zoo
def train_model(model, optimizer, scheduler, epochs=10**9, predicate=lambda loss, acc: acc > 93):
    global run, train_loader, val_loader, test_loader

    pathx, pathy = [], []
    print(f'started train #{st.run_id}', flush=True)
    for epoch in trange(epochs):
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
                if st.test_loader is not None:
                    solve_test(model, st.test_loader, f'solution_{model.loader()}_{name}')
        train_epoch(model, st.train_loader, optimizer, train_logging, 25)
        scheduler.step()
        test_logging(*test(model, st.val_loader))

import neptune.new as neptune
from data import Plug, build_dataloader, build_model, build_optimizer, build_lr_scheduler
from time import time
def run(config):
    [print(f'{key}: {value}') for key, value in config.items()]
    st.device = config['device']
    st.project = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')
    st.run = neptune.init(project=config['project_name'], api_token=config['api_token']) if config['register_run'] else Plug()
    st.run_id = st.run.get_run_url().split('/')[-1] if type(st.run) is not Plug else int(time())
    st.run['parameters'] = config
    st.train_loader = build_dataloader(config['train'], batch_size=config['batch_size'], shuffle=True)
    st.val_loader = build_dataloader(config['val'], batch_size=config['batch_size'], shuffle=False)
    st.test_loader = build_dataloader(config['test'], batch_size=config['batch_size'], shuffle=False)
    model = build_model(config)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_lr_scheduler(optimizer, config)

    train_model(model, optimizer, scheduler, config['epochs'])
    save_to_zoo(model, f'{st.run_id}_final', *test(model, st.val_loader))