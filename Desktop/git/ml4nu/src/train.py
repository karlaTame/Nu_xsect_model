import torch
from tqdm.rich import tqdm
import matplotlib.pyplot as plt
import numpy as np
from hamiltonians3nu import PMNS, FAKE_PMNS
import scipy.optimize as opt
import random

def train_one_step(data_in, model):
    data, enu = data_in
    loss = model.detector_loss((data, enu))

    return loss


def train_near_detector(train_data, val_data, model, optimizer, epochs=100,
                        display=1, plot=False, clip_scale=1e-1, tag=""):
    loss_hist_train = []
    loss_hist_val = []

    train_hist_std_prob = []

    val_hist_std_prob = []

    enu = train_data.dataset.dataset.enu

    for it in range(epochs):
        print(f'Running for epoch: {it}')
        # Training loss
        train_loss = 0
        for data_batch in tqdm(train_data):
            loss = train_one_step((data_batch, enu), model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(),
                                           clip_scale)
            optimizer.step()
            train_loss += loss
            optimizer.zero_grad()

            train_hist_std_prob.append(model.std_prob.to('cpu').data.numpy())

        # Validation loss
        val_loss = 0
        with torch.no_grad():
            for val_batch in tqdm(val_data):
                loss = train_one_step((val_batch, enu), model)
                val_loss += loss
                val_hist_std_prob.append(model.std_prob.to('cpu').data.numpy())

        if it % display == 0:
            print(f"Epoch {it}: Train loss {train_loss/len(train_data)}, Val loss {val_loss/len(val_data)}")
            if plot:
                model.plot(train_data.dataset.dataset, epoch=it, tag=tag)

        loss_hist_train.append(train_loss.to('cpu').data.numpy()/len(train_data))
        loss_hist_val.append(val_loss.to('cpu').data.numpy()/len(val_data))

    if plot:
        plt.plot(loss_hist_train, label='Train loss')
        plt.plot(loss_hist_val, label='Val loss')
        plt.legend()
        plt.savefig(f'figures/{tag}loss.pdf', bbox_inches='tight')
        plt.close()

        plt.plot(train_hist_std_prob, label='Train std prob')
        plt.plot(val_hist_std_prob, label='Val std prob')
        plt.legend()
        plt.savefig(f'figures/{tag}sig_noise.pdf', bbox_inches='tight')
        plt.close()

    return loss_hist_train, loss_hist_val


def opt_func(osc_param, model, osc, dataloader):
    enu = dataloader.dataset.dataset.enu
    oslayer = osc(torch.tensor([osc_param], device=enu.device))
    loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader):

            osc_log_probs = oslayer(enu)
            osc_prob = torch.exp(osc_log_probs)

            loss += model.detector_loss((data, enu),
                                        weights=osc_prob)
    return loss.cpu()

#def loss_curve(model, osc, dataloader, NLoss=30):
#    enu = dataloader.dataset.dataset.enu
#    losses=[]
#    params=[]
#    with torch.no_grad():
#        for i in range(NLoss):
#            osc_param = i/NLoss
#            params.append(osc_param)
#            oslayer = osc(torch.tensor([osc_param], device=enu.device))
#            loss = 0
#            for data in tqdm(dataloader):
#    
#                osc_log_probs = oslayer(enu)
#                osc_prob = torch.exp(osc_log_probs)
#    
#                loss += model.detector_loss((data, enu),
#                                            weights=osc_prob)
#            losses.append(loss)
#
#        # normalize by loss at osc_param = 0
#        for i in range(1,len(losses)):
#            losses[i] = (losses[i]/losses[0]).cpu().detach().numpy()
#        losses[0] = 1
#
#    return params, losses 


def loss_curve(model, osc, dataloader, param_range=(0, 1), samples=100):
    loss = np.zeros(samples)
    for i, param in enumerate(np.linspace(param_range[0], param_range[1], samples)):
        loss[i] = opt_func(param, model, osc, dataloader)

    return loss / np.mean(loss)


def test_loss_curve(samples=100, tag=""):
    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data_loader.NeutrinoDataset('data/muon_dist_fluxoverxsec',
                                       'data/muon_dist_flux', device=device,
                                       nrows=100)
    train, val = data_loader.train_val_split(data)
    train_loader, val_loader = data_loader.get_data_loaders(train, val)

    data_fd = data_loader.NeutrinoDataset('data/muon_dist_FD',
                                          'data/muon_dist_flux', device=device,
                                          nrows=100)
    train_fd, val_fd = data_loader.train_val_split(data_fd)
    train_loader_fd, val_loader_fd = data_loader.get_data_loaders(train_fd, val_fd)

    model = structure_functions.StructureFunctionModel(37.2247242, 0.10566**2,
                                                       verbose=False,
                                                       device=device)
    param_range = (0, 1)
    loss = loss_curve(model, PMNS, train_loader_fd, param_range, samples)

    plt.plot(np.linspace(param_range[0], param_range[1], samples), loss)
    plt.savefig(f'figures/{tag}test_loss_curve.pdf', bbox_inches='tight')


def train_far_detector(trials, epochs=3, nrows=0, device='cpu', tag=""):
    data = data_loader.NeutrinoDataset('data/muon_dist_fluxoverxsec',
                                       'data/muon_dist_flux', device=device, nrows=nrows)
    train, val = data_loader.train_val_split(data)
    train_loader, val_loader = data_loader.get_data_loaders(train, val)

    data_fd = data_loader.NeutrinoDataset('data/muon_dist_FD',
                                          'data/muon_dist_flux', device=device, nrows=nrows)
    train_fd, val_fd = data_loader.train_val_split(data_fd)
    train_loader_fd, val_loader_fd = data_loader.get_data_loaders(train_fd, val_fd)

    # Systematics
    S23 = []
    loss_plot = []
    for i in range(trials):
        model = structure_functions.StructureFunctionModel(37.2247242, 0.10566**2,
                                                           verbose=False,
                                                           device=device)
        optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)
        train_near_detector(train_loader, val_loader, model, optimizer, epochs=epochs,
                            plot=True, tag=tag)
        print("Starting optimization")
        result = opt.minimize_scalar(opt_func,
                                     args=(model, PMNS, train_loader_fd),
                                     bounds=(0, 1), method='bounded')
        print("Finished optimization, result = ", result.x)
        S23.append(result.x)
        loss_plot.append(loss_curve(model, PMNS, train_loader_fd, (0, 1), 100))
        if i == 0:
            model_save = model

    # Statistics
    train_resampled = torch.utils.data.RandomSampler(train_fd, replacement=True)
    train_loader_resampled = torch.utils.data.DataLoader(train_fd, batch_size=1024, sampler=train_resampled)
    statistics = [S23[0]]
    loss_plot_stat = [loss_plot[0]]
    for _ in range(trials):
        result = opt.minimize_scalar(opt_func,
                                     args=(model_save, PMNS, train_loader_resampled),
                                     bounds=(0, 1), method='bounded')
        statistics.append(result.x)
        loss_plot_stat.append(loss_curve(model_save, PMNS, train_loader_resampled, (0, 1), 100))

    # Plot loss_plot with mean and std
    loss_plot = np.array(loss_plot)
    mean_loss = np.mean(loss_plot, axis=0)
    std_loss = np.std(loss_plot, axis=0)
    loss_plot_stat = np.array(loss_plot_stat)
    mean_loss_stat = np.mean(loss_plot_stat, axis=0)
    std_loss_stat = np.std(loss_plot_stat, axis=0)
    plt.plot(np.linspace(0, 1, 100), mean_loss)
    plt.fill_between(np.linspace(0, 1, 100), mean_loss-std_loss, mean_loss+std_loss, alpha=0.5)
    plt.fill_between(np.linspace(0, 1, 100), mean_loss_stat-std_loss_stat, mean_loss_stat+std_loss_stat, alpha=0.25)
    plt.savefig(f'figures/{tag}loss_errors.pdf', bbox_inches='tight')

    # Print statistics
    mean_23 = np.mean(S23)
    std_dev_23 = np.sqrt(np.var(S23))
    print('The prediction is:{:.3}+-{:.3}'.format(mean_23, std_dev_23))
    print('Upper bound:{:.3}'.format(mean_23+std_dev_23))
    print('Lower bound:{:.3}'.format(mean_23-std_dev_23))
    print('The true value is:{:.3}'.format(np.sqrt(0.572)))


if __name__ == '__main__':
    import data_loader
    import structure_functions
    # import hamiltonians3nu

    # tag for filenames
    tag = 'curvetest_FD_100k_test'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')
    #test_loss_curve(samples=20, tag=tag)
    train_far_detector(2, epochs=2, nrows=10, device=device, tag=tag)
    
    #data = data_loader.NeutrinoDataset('data/muon_dist_fluxoverxsec',
    #                                    'data/muon_dist_flux', nrows=1000000, device=device)
    #train, val = data_loader.train_val_split(data)
    #train_loader, val_loader = data_loader.get_data_loaders(train, val)
    #model = structure_functions.StructureFunctionModel(37.2247242, 0.10566**2,
    #                                                    verbose=False,
    #                                                    device=device)
    #optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)
    #train_near_detector(train_loader, val_loader, model, optimizer, epochs=200,
    #                    plot=True, tag=tag)

   # model.plot(data)
