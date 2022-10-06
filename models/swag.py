
import os
import numpy as np
import torch
import torch.nn as nn
import itertools
from utils import get_device
from utils import plot_training_curve


class SWAG():

    def __init__(self, input_dim, output_dim, task, get_aleatoric_uncertainty):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net_arch = [512, 512]
        self.task = task
        self.get_aleatoric_uncertainty = get_aleatoric_uncertainty
        self.device = get_device()
        network_kwargs =   {"input_dim":self.input_dim,
                            "output_dim":self.output_dim,
                            "net_arch":self.net_arch,
                            "task":self.task,
                            "get_aleatoric_uncertainty":self.get_aleatoric_uncertainty}
        self.network = Network(**network_kwargs)
        self.network.to(self.device)

        self.cov_mat = True
        self.max_num_models=20
        self.swag_network = SWAG_Model(Network, no_cov_mat=not self.cov_mat, max_num_models=self.max_num_models,  **network_kwargs)
        self.swag_network.to(self.device)

        if self.task == "regression":
            if self.get_aleatoric_uncertainty:
                self.criterion = torch.nn.GaussianNLLLoss(full=True, reduction='sum')
            else:
                self.criterion = torch.nn.MSELoss(reduction='sum')
        elif self.task == "classification":
            self.criterion = nn.BCELoss(reduction='sum')

    def get_loss(self, output, target):
        if self.task == "regression":
            if self.get_aleatoric_uncertainty:
                return self.criterion(output[0], target, output[1])
            else:
                return self.criterion(output, target)
        elif self.task == "classification":
            return self.criterion(output, target)


    def train(self, 
             train_loader, 
             batch_size, 
             stats, 
             pre_trained_dir, 
             Nbatches, 
             adversarial_training=True,
             **kwargs):
        n_epochs = kwargs.get('n_epochs', 1000)

        print('Training {} model {} adversarial training. Task: {}'.format(type(self).__name__, 
            'with' if adversarial_training else 'without', self.task))
        self.train_loader = train_loader

        swa_start = int(n_epochs/3)
        swa_lr = 1e-3
        lr_init = 1e-3
        wd = 1e-4 
        swa_c_epochs = 1 

        #put cov_mat in cpu (large for gpu)
        for (module, name) in self.swag_network.params:
            cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
            module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt.to(torch.device('cpu')))

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr_init, weight_decay=wd)

        if adversarial_training:
            delta = torch.zeros([batch_size, self.input_dim]).to(self.device)
            X_train_max = torch.tensor(stats['X_train_max'])
            X_train_max = torch.flatten(X_train_max.expand(stats['historical_sequence_length'], -1)).to(self.device)

            X_train_min= torch.tensor(stats['X_train_min'])
            X_train_min = torch.flatten(X_train_min.expand(stats['historical_sequence_length'], -1)).to(self.device)

            if self.task == 'regression':
                clip_eps = 0.2 / stats['X_train_max'].max()
                fgsm_step = 0.2 / stats['X_train_max'].max()
            elif self.task == 'classification':
                clip_eps = 0.5 / stats['X_train_max'].max()
                fgsm_step = 0.5 / stats['X_train_max'].max()

            n_repeats = 4
            n_epochs = int(n_epochs / n_repeats)
            swa_start = int(n_epochs/3)
        
        self.network.train()
        loss_history, lr_history = [], []
        for epoch in range(1, n_epochs+1 ):

            lr = schedule(epoch,swa_start, swa_lr, lr_init)
            adjust_learning_rate(optimizer, lr)

            epoch_loss =[]
            for _ , (features , target) in enumerate(train_loader):
                features  = features.to(self.device)
                target = target.to(self.device)
                
                if adversarial_training:
                    for _ in range(n_repeats):
                        delta_batch = delta[0:features.size(0)]
                        delta_batch.requires_grad = True
                        adv_features = features + delta_batch
                        adv_features.clamp_(X_train_min, X_train_max)
                        output = self.network(adv_features)
                        loss = self.get_loss(output, target)
                        loss.backward()
                        optimizer.step()        
                        optimizer.zero_grad()
                        pert = fgsm_step * torch.sign(delta_batch.grad)
                        delta[0:features.size(0)] += pert.data
                        delta.clamp_(-clip_eps, clip_eps)
                        epoch_loss.append(loss.item())
                else:
                    output = self.network(features)
                    loss = self.get_loss(output, target)
                    loss.backward()
                    optimizer.step()        
                    optimizer.zero_grad()
                    epoch_loss.append(loss.item())

            if (epoch + 1) > swa_start and (epoch + 1 - swa_start) % swa_c_epochs == 0:
                self.swag_network.collect_model(self.network)

            lr_history.append(optimizer.param_groups[0]['lr'])
            loss_history.append(np.mean(epoch_loss))

            if epoch % 10 == 0:
                print("Epoch: {0:0.3g}, NLL: {1:0.3g},  lr: {2:0.3g}".format(epoch,loss_history[-1], lr_history[-1]), end='\r')


        pre_trained_dir = os.path.join(pre_trained_dir, type(self).__name__)
        os.makedirs(pre_trained_dir , exist_ok=True)
        model_save_name = pre_trained_dir + '/trained_network_' + self.task + ('_adv.pt' if adversarial_training else '.pt')
        fig_save_name = pre_trained_dir + '/training_curve_' +self.task + ('_adv.pdf' if adversarial_training else '.pdf')
        torch.save(self.swag_network.state_dict(), model_save_name)
        # torch.save(train_loader, pre_trained_dir +'/trained_network_'+self.task+"train_loader.pt")
        plot_training_curve(loss_history, lr_history, fig_save_name)
        
    def load_parameters_reg(self, pre_trained_dir, train_loader):
        self.train_loader = train_loader
        model_save_name_reg = pre_trained_dir + '/trained_network_'+ 'regression' +  '.pt'
        self.swag_network.load_state_dict(torch.load(model_save_name_reg, map_location=self.device)) 
        for (module, name) in self.swag_network.params:
            cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
            module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt.to(torch.device('cpu')))
        # self.train_loader = torch.load(pre_trained_dir +'/trained_network_'+'regression'+"train_loader.pt")

    def load_parameters_clas(self, pre_trained_dir, train_loader):
        self.train_loader = train_loader
        model_save_name_clas = pre_trained_dir + '/trained_network_'+ 'classification' +  '.pt'
        self.swag_network.load_state_dict(torch.load(model_save_name_clas, map_location=self.device))
        for (module, name) in self.swag_network.params:
            cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
            module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt.to(torch.device('cpu')))
        # self.train_loader = torch.load(pre_trained_dir +'/trained_network_'+'classification'+"train_loader.pt")

    def evaluate_reg(self, X_test, y_true_reg, **kwargs):
        n_samples = kwargs.get('n_samples', 50)
        measurement_error = kwargs.get('measurement_error', 1e-06)
        self.swag_network.eval()
        if self.get_aleatoric_uncertainty:
            samples_mean, samples_var = [], []
            for i in range(n_samples):
                torch.manual_seed(i)
                self.swag_network.sample(scale=0.5, cov=self.cov_mat)
                bn_update(self.train_loader, self.swag_network)
                features = torch.from_numpy(X_test).type(torch.float)
                features  = features.to(self.device)
                pred_mean, pred_var = self.swag_network(features)
                pred_mean = pred_mean.detach().cpu().numpy()
                pred_var = pred_var.detach().cpu().numpy()
                samples_mean.append(pred_mean)
                samples_var.append(pred_var)

            mixture_mean = np.mean(samples_mean, axis=0)
            mixture_var = np.mean(samples_var, axis=0) + np.mean(np.square(samples_mean), axis=0) - np.square(mixture_mean)
        else:
            samples_mean = []
            for i in range(n_samples):
                torch.manual_seed(i)
                self.swag_network.sample(scale=0.5, cov=self.cov_mat)
                bn_update(self.train_loader, self.swag_network)
                features = torch.from_numpy(X_test).type(torch.float)
                features  = features.to(self.device)
                pred_mean = self.swag_network(features)
                pred_mean = pred_mean.detach().cpu().numpy()
                samples_mean.append(pred_mean)
            mixture_mean = np.mean(samples_mean, axis=0)
            mixture_var = np.var(samples_mean, axis=0)
        
        mixture_mean[0] = y_true_reg[0] # enforce measurements
        mixture_var[0] = [measurement_error] # enforce measurement error o
        assert min(mixture_var)>=0, print(mixture_var, samples_var, np.array(samples_var).min())
        return mixture_mean, mixture_var


    def evaluate_clas(self, X_test, y_true_clas, **kwargs):
        n_samples = kwargs.get('n_samples', 50)
        samples = []
        self.network.eval()
        for i in range(n_samples):
            torch.manual_seed(i)
            self.swag_network.sample(scale=0.5, cov=self.cov_mat)
            bn_update(self.train_loader, self.swag_network)
            features = torch.from_numpy(X_test).type(torch.float)
            features  = features.to(self.device)
            output = self.swag_network(features)
            output = output.detach().cpu().numpy()
            output[0] = y_true_clas[0] # enforce measurement
            samples.append(output)
        samples = np.array(samples) 
        return samples

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch, task, get_aleatoric_uncertainty):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task
        self.get_aleatoric_uncertainty = get_aleatoric_uncertainty
        self.dropout_probability = 0.5
        self.layers = nn.ModuleList()
        in_features = self.input_dim
        for hidden_size in net_arch:
            self.layers.append(nn.Linear(in_features=in_features, out_features=hidden_size))
            in_features = hidden_size

        if self.task == 'regression':
            if self.get_aleatoric_uncertainty:
                self.layers.append(nn.Linear(in_features=in_features, out_features=2*self.output_dim))
            else:
                self.layers.append(nn.Linear(in_features=in_features, out_features=self.output_dim))
            self.Softplus= nn.Softplus()
        elif self.task == 'classification':
            self.layers.append(nn.Linear(in_features=in_features, out_features=self.output_dim))
            self.Sigmoid= nn.Sigmoid()
        self.dropout = nn.Dropout(p=self.dropout_probability, inplace=True)                            
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x= self.dropout(x)
            x = self.act(x)
        out = self.layers[-1](x)
 
        if self.task == 'regression':
            mean = out[:, :self.output_dim]
            if self.get_aleatoric_uncertainty:
                # The variance should always be positive (softplus) and la
                variance = self.Softplus(out[:, self.output_dim:])+ 1e-06 
                return mean, variance
            else:
                return mean

        elif self.task == 'classification':
            prob = self.Sigmoid(out)
            return prob


"""
    implementation of a swag model and other utils
    adopted from https://github.com/wjmaddox/swa_gaussian
"""

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def schedule(epoch,swa_start, swa_lr, lr_init):
    t = epoch / swa_start
    lr_ratio = swa_lr / lr_init
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio

    return lr_init * factor

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


def swag_parameters(module, params, no_cov_mat=True):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer('%s_mean' % name, data.new(data.size()).zero_())
        module.register_buffer('%s_sq_mean' % name, data.new(data.size()).zero_())

        if no_cov_mat is False:
            module.register_buffer( '%s_cov_mat_sqrt' % name, data.new_empty((0, data.numel())).zero_().cpu() )

        params.append((module, name))


class SWAG_Model(torch.nn.Module):
    def __init__(self, base, no_cov_mat = True, max_num_models = 0, var_clamp = 1e-30, *args, **kwargs):
        super(SWAG_Model, self).__init__()

        self.register_buffer('n_models', torch.zeros([1], dtype=torch.long))
        self.params = list()

        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models

        self.var_clamp = var_clamp

        self.base = base(*args, **kwargs)
        self.base.apply(lambda module: swag_parameters(module=module, params=self.params, no_cov_mat=self.no_cov_mat))

    def forward(self, *args, **kwargs):
        return self.base( *args, **kwargs)

    def sample(self, scale=1.0, cov=False, seed=None, fullrank = True):
        if seed is not None:
            torch.manual_seed(seed)

        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean = module.__getattr__('%s_mean' % name)
            sq_mean = module.__getattr__('%s_sq_mean' % name)

            if cov:
                cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
                cov_mat_sqrt_list.append( cov_mat_sqrt )

            mean_list.append( mean.cpu() )
            sq_mean_list.append( sq_mean.cpu() )

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad = False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0),), requires_grad=False).normal_())
            cov_sample /= ((self.max_num_models-1)**0.5) 

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale 
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        for (module, name), sample in zip(self.params, samples_list):
            module.register_parameter(name, nn.Parameter(sample.cuda()))

    def collect_model(self, base_model):
        for (module, name), base_param in zip(self.params, base_model.parameters()):
            mean = module.__getattr__('%s_mean' % name)
            sq_mean = module.__getattr__('%s_sq_mean' % name)
            
            #first moment
            mean = mean * self.n_models.item() / (self.n_models.item() + 1.0) + base_param.data / (self.n_models.item() + 1.0)

            #second moment
            sq_mean = sq_mean * self.n_models.item() / (self.n_models.item() + 1.0) + base_param.data ** 2 / (self.n_models.item() + 1.0)

            #square root of covariance matrix
            if self.no_cov_mat is False:
                cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
                
                #block covariance matrices, store deviation from current mean
                dev = (base_param.data - mean).view(-1,1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1,1).t().cpu()),dim=0)

                #remove first column if we have stored too many models
                if (self.n_models.item()+1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt)

            module.__setattr__('%s_mean' % name, mean)
            module.__setattr__('%s_sq_mean' % name, sq_mean)
        self.n_models.add_(1)

    def load_state_dict(self, state_dict, strict=True):
        if not self.no_cov_mat:
            n_models = state_dict['n_models'].item()
            rank = min(n_models, self.max_num_models)
            for module, name in self.params:
                mean = module.__getattr__('%s_mean' % name)
                module.__setattr__('%s_cov_mat_sqrt' % name, mean.new_empty((rank, mean.numel())).zero_())
        super(SWAG_Model, self).load_state_dict(state_dict, strict)
