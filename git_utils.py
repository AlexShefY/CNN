import static as st
from os import system, listdir
import torch
from models import *

def save_to_zoo(model, name, val_loss=None, val_acc=None):
    print('save_to_zoo', name, val_acc, val_loss)
    model.cpu()
    torch.save({
        'state_dict': model.state_dict(),
        'loader': model.loader(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        }, f'model.p')
    model.to(st.device)
    st.project[f'zoo/{model.loader()}_{name}.p'].upload(f'model.p')

def load_from_zoo(name):
    st.project[f'zoo/{name}.p'].download()
    if f'{name}.p' not in listdir():
        system(f'mv {name}.p.p {name}.p')
        system(f'move {name}.p.p {name}.p')
    d = torch.load(f'{name}.p')
    print(d['loader'])
    model = eval(d['loader'])
    model.load_state_dict(d['state_dict'])
    return model
