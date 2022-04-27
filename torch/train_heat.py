from pyexpat import model
import torch
import numpy as np
import scipy.io
import time, math
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import backward
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse
parser = argparse.ArgumentParser(description='Arguments.')

np.random.seed(123)
torch.manual_seed(100)

class PhysicsInformedNN(nn.Module):
    def __init__(self, data_xt, data_z, layers, device='cpu', optim_method='adam', lr=0.01, lmbda=lambda epoch: 0.5): # xyt.size()=(N*T,3), Xbatch=N*T
        super(PhysicsInformedNN, self).__init__()    
        self.lb = data_xt.min()
        self.ub = data_xt.max()

        self.layers = layers
        self.input_dim = 2
        self.output_dim = 1
        self.hidden_dim = 64
        self.hidden_activation = F.tanh  # relu does not work
        
        # Initialize parameters
        # learnable nu
        # self.nu = nn.Parameter(torch.zeros(1, dtype=torch.float64), requires_grad=True)
        # given nu
        self.nu = torch.Tensor([0.02]).to(device)

        self.input_layer =  nn.Linear(self.input_dim, self.hidden_dim).to(device)
        self.hidden_layers = [nn.Linear(self.hidden_dim, self.hidden_dim).to(device) for _ in range(self.layers)]
        self.hidden_layers = nn.ModuleList(self.hidden_layers)  # Have to wrap the list layers with nn.ModuleList to coorectly make those parameters tracked by nn.module! Otherwise those params will not be saved!
        self.ouput_layer = nn.Linear(self.hidden_dim, self.output_dim).to(device)

        self.apply(self._weight_init)


        if optim_method == "adam":
            self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        elif optim_method == "sgd":
            self.optim = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            print("THIS OPTIMIZATION METHOD IS NOT SUPPORTED.")
            return 0
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optim, lr_lambda=lmbda)

    def _weight_init(self, m):
        if isinstance(m, nn.Linear):
            # Use torch default initialization for Linear here: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L44-L48
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.kaiming_uniform_(m.weight)
            nn.init.normal_(m.bias)
    
    def neural_net(self, X): # X.size()=(Xbatch,3), 
        X = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0     

        X = self.hidden_activation(self.input_layer(X))
        for hl in self.hidden_layers:
            X=self.hidden_activation(hl(X))
        Y = self.ouput_layer(X)

        return Y

    def net_NS(self, xt): # xyt.size()=(N*T,3) Xbatch=N*T
        xt.requires_grad=True 
        self.create_graph = True
        self.retain_graph = True

        u = self.neural_net(xt)

        # calculate u,v, p_x, p_y
        Du = grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=self.create_graph, retain_graph=self.retain_graph)

        u_x = Du[0][:,0]
        u_t = Du[0][:,1]

        Du_x = grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=self.create_graph, retain_graph=self.retain_graph)
        u_xx = Du_x[0][:,0]

        # calculate two components of f
        f_u = u_t - self.nu * u_xx

        return u, f_u
        
    def forward(self, xt):
        return self.net_NS(xt)
    
    def loss_function(self, collc_xt, data_xt, data_z):
        pred_u, _ = self.forward(data_xt)
        _, f_u = self.forward(collc_xt)

        u_loss = (pred_u.squeeze() - data_z).pow(2).mean()
        f_u_loss = f_u.pow(2).mean()

        loss = u_loss + f_u_loss

        return loss, u_loss

def load_data(device):
    # Data loading and processing
    data = np.load('data/heat_equation_data.npy', allow_pickle=True)
    data = data.item()
    # print(data.keys())

    collc_xt = torch.FloatTensor(data['collc_points']['xt']).to(device)
    collc_z = torch.FloatTensor(data['collc_points']['z']).to(device)
    data_xt = torch.FloatTensor(data['data_points']['xt']).to(device)
    data_z = torch.FloatTensor(data['data_points']['z']).to(device)
    test_xt = torch.FloatTensor(data['test_points']['xt']).to(device)
    test_z = data['test_points']['z']

    return collc_xt, collc_z, data_xt, data_z, test_xt, test_z

def plot(x):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(x)
    plt.savefig('nn.png')
    # plt.show()

def eval(pinn, test_xt, test_z, verbose=False):
    u_pred, _ = pinn(test_xt)
    nu_value = pinn.nu.detach().cpu().numpy()

    u_pred = u_pred.squeeze().detach().cpu().numpy()
    
    # Error
    # error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)

    error_u = ((u_pred-test_z)**2).mean()

    error_nu = np.abs(nu_value - 0.02)*100

    if verbose:
        print('Error u: %e' % (error_u))    
        print('Error nu: %.5f%%' % (error_nu))                             

    return error_u, error_nu

# Training
def train(pinn, nIter, batch, collc_xt, collc_z, data_xt, data_z, test_xt, test_z, model_path): 
    losses = []
    start_time = time.time()
    adapt_lr = pinn.optim.param_groups[0]['lr']
    for it in range(nIter):
        sample_idx = np.random.choice(collc_xt.shape[0], batch, replace=False)# cannot forward the whole dataset, sample a batch containing data points
        xt = collc_xt[sample_idx]
        loss_train, u_loss = pinn.loss_function(xt, data_xt, data_z,)
        pinn.optim.zero_grad()
        loss_train.backward()
        pinn.optim.step()
        nu = pinn.nu
        losses.append(loss_train.item())
        if (it+1)%1000 == 0:
            pinn.scheduler.step()
            adapt_lr = pinn.optim.param_groups[0]['lr']
            torch.save(pinn.state_dict(), model_path)

        if it % 100 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, nu: %.3f, Time: %.2f' % 
                  (it, loss_train, nu, elapsed))
            start_time = time.time()
            # plot(losses)
            error_u, error_nu = eval(pinn, test_xt, test_z)
            writer.add_scalar('Train/Loss', loss_train, it)
            writer.add_scalar('Train/u', u_loss, it)
            writer.add_scalar('Train/LR', adapt_lr, it)
            writer.add_scalar('Test Error/u', error_u, it)
            writer.add_scalar('Test Error/nu', error_nu, it)


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--data', type=int, default=5000)
    args = parser.parse_args()
    writer = SummaryWriter('runs/'+args.id+str(args.data))

    # Training Process
    N_train = int(args.data) #5000
    N_test = 1000
        
    layers = 8
    nIter = 50000  # original niter is 200000
    lr = 0.001
    batch = 10000

    optim_method = "adam"
    model_path = './model/'
    postfix = args.id if args.id is not '' else '0'
    model_path += postfix

    collc_xt, collc_z, data_xt, data_z, test_xt, test_z = load_data(device)
    data_xt = data_xt[:N_train]
    data_z = data_z[:N_train]
    test_xt = test_xt[:N_test]
    test_z = test_z[:N_test]

    # Training
    pinn = PhysicsInformedNN(data_xt, data_z, layers, device, optim_method, lr).to(device) 
    train(pinn, nIter, batch, collc_xt, collc_z, data_xt, data_z, test_xt, test_z, model_path)

    # Prediction
    eval(pinn, test_xt, test_z, verbose=True)
