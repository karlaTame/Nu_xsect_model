import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as colors

MIN_FLOAT = 1e-15


class StructureFunctionNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=20, layers=10,
                 device='cpu'):
        super(StructureFunctionNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.layers = layers
        self.device = device
        #self.model = nn.Sequential(
        #    nn.Linear(self.in_channels, self.hidden_size),
        #    nn.Sigmoid(),
        #    *[nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                    nn.Sigmoid()) for _ in range(self.layers)],
        #    nn.Linear(self.hidden_size, self.out_channels)
        #)
        #self.model.to(self.device)

        self.w0 = nn.Linear(2, hidden_size).to(self.device)
        self.w1 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w3 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w4 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w5 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w6 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w7 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.w8 = nn.Linear(hidden_size, hidden_size).to(self.device)
#         self.w9 = nn.Linear(hidden_size,H)
#         self.w10 = nn.Linear(hidden_size,H)
#         self.w11 = nn.Linear(hidden_size,H)
#         self.w12 = nn.Linear(hidden_size,H)
        self.last = nn.Linear(hidden_size, out_channels).to(self.device)

    def forward(self, inp):
        #out = self.model(x)
        
        out = torch.sigmoid(self.w0(inp)).to(self.device)
        out = torch.sigmoid(self.w1(out)).to(self.device)
        out = torch.sigmoid(self.w2(out)).to(self.device)
        out = torch.sigmoid(self.w3(out)).to(self.device)
        out = torch.sigmoid(self.w4(out)).to(self.device)
        out = torch.sigmoid(self.w5(out)).to(self.device)
        out = torch.sigmoid(self.w6(out)).to(self.device)
        out = torch.sigmoid(self.w7(out)).to(self.device)
        out = torch.sigmoid(self.w8(out)).to(self.device)
# #         out = torch.sigmoid(self.w9(out))
# #         out = torch.sigmoid(self.w10(out))
# #         out = torch.sigmoid(self.w11(out))
# #         out = torch.sigmoid(self.w12(out))
        out = self.last(out).to(self.device)
        
        full_out_pos = torch.exp(out[..., :3])
        full_out_rest = out[..., 3:]*1e-1
        out = torch.cat((full_out_pos, full_out_rest), axis=-1)
        return out


class StructureFunctionModel:
    def __init__(self, ma, ml2, device='cpu', in_channels=2, out_channels=5,
                 hidden_size=20, layers=10, verbose=False):

        self.ma = ma
        self.ml2 = torch.tensor(ml2, device=device)
        self.ml = torch.sqrt(self.ml2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.layers = layers
        self.verbose = verbose
        self.device = device
        self.model = StructureFunctionNet(self.in_channels, self.out_channels,
                                          self.hidden_size, self.layers,
                                          self.device)

    def _generate_enu(self, data, enu_in, trials, weights_in=1):
        enu_min = ((2*self.ma*data[:, 0] - self.ml2)
                   / (2*(torch.sqrt(data[:, 0]**2 - self.ml2)*data[:, 1]
                         + self.ma - data[:, 0])))
        #indices_min = 0*torch.searchsorted(enu_in, enu_min, right=True)
        indices_min = torch.searchsorted(enu_in, enu_min, right=True)
        rand = torch.rand((indices_min.shape[0], trials), device=self.device)
        index_range = len(enu_in) - 1 - indices_min
        indices = indices_min[..., None] + index_range[..., None] * rand
        indices = indices.to(dtype=torch.int64)

        # Mask out of bounds indices
        mask = torch.where(indices >= len(enu_in), 0, 1)
        #mask = torch.where(enu_in[indices] < enu_min[..., None], 0, 1)
        indices = indices * mask + (len(enu_in) - 1) * (1 - mask)
        enu = enu_in[indices].squeeze()
        weights = (1-indices_min/len(enu_in))
        weights = weights / torch.sum(weights)
        #print(enu_in.shape)
        #print(weights.shape)
        weights = weights[..., None]*mask
#        print(weights.shape)
#        print("indices_min = ", indices_min)
#        print("min indices_min = ", torch.min(indices_min))
#        print("max indices_min = ", torch.max(indices_min))
#        print("min mask = ", torch.min(mask))
#        print("max mask = ", torch.max(mask))
#        print("min = ", torch.min(weights))
#        print("max = ", torch.max(weights))
        ones = enu_in*0 + 1
        weights_in = enu_in * ones
        weights *= weights_in[indices].squeeze()
        return enu, weights

    def _eval_x_q2(self, energy, emu, cost):
        emu = emu[..., None]
        cost = cost[..., None]
        energy_diff = energy - emu
        q2 = 2*(energy*emu - energy*torch.sqrt(emu**2 - self.ml2)*cost)-self.ml2
        x = q2 / (2*self.ma*energy_diff)
        q2[x < 0 + MIN_FLOAT] = MIN_FLOAT
        q2[x > 1 - MIN_FLOAT] = MIN_FLOAT
        x[x < 0 + MIN_FLOAT] = MIN_FLOAT
        x[x > 1 - MIN_FLOAT] = MIN_FLOAT
        x[q2 < 0 + MIN_FLOAT] = MIN_FLOAT
        q2[q2 < 0 + MIN_FLOAT] = MIN_FLOAT
        return [x, q2]

    def _eval_kinematics(self, energy, sign, x, q2):
        #print(x.size)
        energyp = energy - q2/(2*x*self.ma)
        y = q2/(2*self.ma*energy*x)
        mass_term_noy = self.ml2/q2
        #print("ml2 = ", self.ml2)
        #print("max(mass_term_noy) = ", torch.max(mass_term_noy))
        mass_term = y * mass_term_noy
        #print("max(mass_term) = ", torch.max(mass_term))
        k1 = y + mass_term
        k2 = (energy/self.ma)*(1 - y - self.ma*x*y/(2*energy)*(1+mass_term_noy))
        k3 = sign * (1 - y/2 - mass_term/2)
        k4 = mass_term * (mass_term_noy + 1)
        k5 = -2*mass_term_noy
        #print("max(k5) = ", torch.max(k5))
        prefactor = energyp*torch.sqrt(1-self.ml2/energyp**2)
        ks = torch.stack((k1, k2, k3, k4, k5), axis=-1)
        #print(ks.shape)
        #print("max(k5) = ", torch.max(ks[:,4]))
        ks[energyp >= self.ml, ...] *= prefactor[energyp >= self.ml].unsqueeze(-1)
        #print("max(k5) = ", torch.max(ks[:,4]))
        return ks

    def xsec(self, enu, sign, x, q2):
        kins = self._eval_kinematics(enu, sign, x, q2)
        form_factor = self.model(torch.stack([x, q2], axis=-1))
#        print("min k1", torch.min(kins[..., 0]))
#        print("max k1", torch.max(kins[..., 0]))
#        print("min k2", torch.min(kins[..., 1]))
#        print("max k2", torch.max(kins[..., 1]))
#        print("min k3", torch.min(kins[..., 2]))
#        print("max k3", torch.max(kins[..., 2]))
#        print("min k4", torch.min(kins[..., 3]))
#        print("max k4", torch.max(kins[..., 3]))
#        print("min k5", torch.min(kins[..., 4]))
#        print("max k5", torch.max(kins[..., 4]))
        sig = torch.sum(kins * form_factor, axis=-1)
        # if self.verbose:
        #     ones = torch.ones_like(sig)
        #     print(f"positive entries: {ones[sig > 0].sum()} / {ones.sum()}")
        #     print(f"negative entries: {ones[sig < 0].sum()} / {ones.sum()}")
        return sig

    def _get_mask(self, target):
        mask = torch.ones(target.shape[:2], dtype=torch.bool,
                          device=target.device)
        mask[target[..., 0] >= 1 - MIN_FLOAT] = 0
        mask[target[..., 0] <= 0 + MIN_FLOAT] = 0
        mask[target[..., 1] <= 0 + MIN_FLOAT] = 0
        mask[torch.isnan(target[..., 0])] = 0
        mask[torch.isnan(target[..., 1])] = 0
        mask[torch.isinf(target[..., 0])] = 0
        mask[torch.isinf(target[..., 1])] = 0
        return mask

    def _get_weights(self, probs, dim_data):
        emu1 = torch.ones(dim_data)
        prob_tensor = torch.flatten(torch.einsum('i,j->ij', emu1, probs))
        return prob_tensor

    def get_flow_variables(self, emu, cost, enu):
        x, q2 = self._eval_x_q2(enu, emu, cost)
        return torch.stack([x, q2], dim=-1)

    def flux_averaged_prob(self, target_in, enu_in,
                           sign=1, weights=1, trials=1024):
        enu, enu_weights = self._generate_enu(target_in, enu_in, trials, weights)
        target = self.get_flow_variables(target_in[:, 0], target_in[:, 1], enu)
        mask = self._get_mask(target)
        prob = self.xsec(enu, sign, target[..., 0], target[..., 1])
        weighted_prob = torch.zeros_like(prob)
        weighted_prob[mask] = (prob*enu_weights)[mask]
        #weighted_prob[mask] = (prob)[mask]
        weighted_prob[torch.isnan(weighted_prob)] = 0
        weighted_prob[torch.isinf(weighted_prob)] = 0
        average_prob = torch.sum(weighted_prob, axis=-1)/trials
        average_prob2 = torch.sum(weighted_prob**2, axis=-1)/trials
        std_prob = torch.sqrt(torch.abs(average_prob2 - average_prob**2)/trials)/(average_prob + MIN_FLOAT)
        std_prob = torch.mean(std_prob)
        #print("StN = ", std_prob)
        return average_prob, std_prob

    def detector_loss(self, data_in, weights=1):
        print(data_in)
        dataq, enu= data_in
        data = torch.tensor(dataq[:,0:2], dtype=torch.float32)
        Q = dataq[:,2]
        print(enu.dtype)
        print(data.dtype)
        avg_prob, self.std_prob = self.flux_averaged_prob(data, enu, weights=weights,
                                                          trials=1024)

        if self.verbose:
            print("avg prob", avg_prob)
            print('avg prob min', torch.min(avg_prob))
            print('avg prob max', torch.max(avg_prob))
        if torch.sum(avg_prob) < 1e-1:
            print(torch.sum(avg_prob))
            print(data)
            print(enu[:30])
            print(enu[-30:])
        #assert(torch.sum(avg_prob) > 1e-1)

        avg_prob_reg = torch.abs(avg_prob) + MIN_FLOAT

        kl_loss_2 = -torch.mean(torch.log(avg_prob_reg))

        kl_loss_1 = torch.log(torch.mean(avg_prob_reg / Q))

        return (kl_loss_1 + kl_loss_2
                - torch.sum(avg_prob[avg_prob < 0])
                - torch.sum(avg_prob[avg_prob < 0]))

    def detector_loss_flat(self, data_in, flat_data_in, weights=1):
        data, enu = data_in
        flat_data, flat_enu = flat_data_in

        avg_prob, self.std_prob = self.flux_averaged_prob(data, enu, weights=weights,
                                                          trials=1024)
        if self.verbose:
            print("avg prob", avg_prob)
            print('avg prob min', torch.min(avg_prob))
            print('avg prob max', torch.max(avg_prob))
        if torch.sum(avg_prob) < 1e-1:
            print(torch.sum(avg_prob))
            print(data)
            print(enu[:30])
            print(enu[-30:])
        #assert(torch.sum(avg_prob) > 1e-1)
        kl_loss_2 = -torch.mean(torch.log(torch.abs(avg_prob) + MIN_FLOAT))

        flat_avg_prob, self.flat_std_prob = self.flux_averaged_prob(flat_data, flat_enu,
                                                                    weights=weights, trials=1024)
        kl_loss_1 = torch.log(torch.mean(torch.abs(flat_avg_prob) + MIN_FLOAT))

        return (kl_loss_1 + kl_loss_2
                - torch.sum(avg_prob[avg_prob < 0])
                - torch.sum(flat_avg_prob[avg_prob < 0]))

    # TODO: Implement plotting
    def plot(self, data, grid_shape=(200, 200), epoch=None, tag=""):
        xx, yy = torch.meshgrid(torch.linspace(0, 4, grid_shape[0]),
                                torch.linspace(-1, 1, grid_shape[1]))
        zz = torch.cat((xx.unsqueeze(2), yy.unsqueeze(2)), axis=2).view(-1, 2)
        zz = zz.to(data.device)
        prob, _ =  self.flux_averaged_prob(zz, data.enu)  
        plt.pcolormesh(yy, xx, prob.view(grid_shape).cpu().detach().numpy(), norm=colors.LogNorm(), shading='auto')
        if epoch is not None:
            #print("xx = ", xx)
            #print("yy = ", yy)
            #print("zz = ", zz)
            #print("Enu = ", data.enu)
            #print("prob = ", prob)
            plt.savefig(f'figures/{tag}model_{epoch}.pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f'figures/{tag}model_base.pdf', bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    import data_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data_loader.NeutrinoDataset('data/muon_dist_fluxoverxsec',
                                       'data/muon_dist_flux', device=device)
    train, val = data_loader.train_val_split(data)
    train_loader, val_loader = data_loader.get_data_loaders(train, val)
    sf_model = StructureFunctionModel(37.2247242, 0.10566**2, verbose=True)
    enu, emu, cost = next(iter(train_loader))
    print(sf_model._eval_x_q2(enu, emu, cost))
