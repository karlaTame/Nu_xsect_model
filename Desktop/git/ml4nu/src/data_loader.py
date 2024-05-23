import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader
import torch
import numpy as np
from sklearn.neighbors import KernelDensity

def Qfunc(Emu, cost, bandwidth):
    ecos = torch.stack((Emu, cost), dim=1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(ecos)
    log_density = torch.tensor(kde.score_samples(ecos))
    ecosk = torch.stack((Emu, cost, np.exp(log_density)), dim=1)

    return ecosk
    
def plotQfunc(ecosk, width, tag=""):
    y_coords = ecosk[:, 0].numpy()
    x_coords = ecosk[:, 1].numpy()
    color_intensities = ecosk[:, 2].numpy()

    # Create the heatmap plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, c=color_intensities, cmap='viridis')
    plt.colorbar(label='KDE')
    plt.xlabel('cost')
    plt.ylabel('Emu')
    plt.title(f'KDE for width {width}')
    plt.grid(True)
    plt.savefig(f'figures/{tag}_KDE_width_{width}.pdf', bbox_inches='tight')
    plt.close()

class NeutrinoDataset(Dataset):
    def __init__(self, datafile_enu, datafile_emu,
                 device='cpu', nrows=0, bandwidth=0.1):
        print('Loading data...')
        self.device = device
        enu = pd.read_csv(datafile_enu, names=['Enu', 'Emu', 'cost'],
                          skiprows=1, sep=r'\s+')
        self.enu = torch.tensor(enu['Enu'].to_numpy(),
                                dtype=torch.float32,
                                device=device).sort()[0]

        if nrows > 0:
            data = pd.read_csv(datafile_emu, names=['Enu', 'Emu', 'cost'],
                               skiprows=1, nrows=nrows, sep=r'\s+')
        else:
            data = pd.read_csv(datafile_emu, names=['Enu', 'Emu', 'cost'],
                                skiprows=1, sep=r'\s+')
        
        Emu = torch.tensor(data["Emu"].to_numpy())#, dtype=torch.float32, device=device)
        cost = torch.tensor(data["cost"].to_numpy())#, dtype=torch.float32, device=device)
        ecosk = Qfunc(Emu, cost, bandwidth)    
        print(ecosk)
        self.data = ecosk
                     #torch.tensor(data[['Emu', 'cost']].to_numpy(),
                      #           dtype=torch.float32,
                       #          device=device)
        print('Data loaded')
        print('Data size ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_val_split(dataset, val_split=0.25):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(val_split * dataset_size)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train = Subset(dataset, train_indices)
    val = Subset(dataset, val_indices)
    return train, val


def get_data_loaders(train, val, batch_size=1024):
    train_loader = DataLoader(train, batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size, shuffle=False)
    return train_loader, val_loader


def plot_data(data):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    _, emu, cost = data
    bins = (np.linspace(0, 4, 200), np.linspace(-1, 1, 200))
    plt.hist2d(cost, emu.cpu().numpy(), bins=bins, norm=colors.LogNorm())
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = NeutrinoDataset('data/muon_dist_fluxoverxsec',
                           'data/muon_dist_flux', device=device)
    # Time to sample 1024 events
    import time
    start = time.time()
    print(data[0:1024])
    print('Time to sample 1024 events:', time.time()-start)
    train, val = train_val_split(data)
    train_loader, val_loader = get_data_loaders(train, val)
    print('Train size:', len(train))
    print('Val size:', len(val))
    print('Train loader size:', len(train_loader))
    print('Val loader size:', len(val_loader))
    print('Train loader first batch:', next(iter(train_loader)))
