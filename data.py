import neptune.new as neptune
import pickle
from torch.utils.data import DataLoader, random_split
import torch
import plotly.express as px  

class Plug:
    def log(self, x, step=None):
        pass
    def upload(self, x):
        pass
    def __getitem__(self, s):
        return Plug()
    def __setitem__(self, s, v):
        pass

project = 'mlxa/CNN'
token = token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ=='
run = neptune.init(project=project, api_token=token)
project = neptune.init_project(name=project, api_token=token)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_dataloaders(batch_size, download=True):
    if download:
        project['train_v1.bin'].download()
        project['test_v1.bin'].download()
        project['val_v1.bin'].download()

    with open('train_v1.bin.bin', 'rb') as f:
        train_data = pickle.load(f)
    with open('val_v1.bin.bin', 'rb') as f:
        val_data = pickle.load(f)
    with open('test_v1.bin.bin', 'rb') as f:
        test_data = pickle.load(f)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, y in train_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        # for i, e in enumerate(X[:10]):
        #     print('ans', y[i])
            # fig = px.imshow(e.permute((1, 2, 0)))
            # fig.show()
        break
    
    print("Using {} device".format(device))
    return train_dataloader, val_dataloader, test_dataloader

def smooth(pic):
    assert pic.shape == (3, 32, 32)
    s = torch.ones_like(pic) * pic.mean(dim=(1, 2)).view(3, 1, 1)
    b = (0 < pic) & (pic < 1)
    g = torch.where(b, pic, s)
    s[:, 1:, :] += g[:, :-1, :]
    s[:, :-1, :] += g[:, 1:, :]
    s[:, :, 1:] += g[:, :, :-1]
    s[:, :, :-1] += g[:, :, 1:]
    return torch.where(b, pic, s/4)