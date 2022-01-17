import torch
from torch.utils.data import DataLoader
from os import listdir, system
import static as st


class Plug:
    d = dict()
    def log(self, x, step=None):
        pass
    def upload(self, x):
        pass
    def __getitem__(self, s):
        return self.d[s] if s in self.d else Plug()
    def __setitem__(self, s, v):
        self.d[s] = v

def build_dataloader(name, batch_size, shuffle):
    import pickle
    if not name in listdir():
        st.project[name].download()
        assert f'{name}.bin' in listdir()
        system(f'mv {name}.bin {name}')
        system(f'move {name}.bin {name}')
    try:
        data = torch.load(name)
    except Exception as e:
        print("torch.load() didn't work")
        with open(name) as f:
            data = pickle.load(f)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

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