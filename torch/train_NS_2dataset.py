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
    def __init__(self, xyt, layers, device='cpu', optim_method='adam', hidden_dim=20, lr=0.01, lmbda=lambda epoch: 0.5): # xyt.size()=(N*T,3), Xbatch=N*T
        super(PhysicsInformedNN, self).__init__()
        self.data_idx = data_idx

        self.layers = layers
        self.input_dim = 3
        self.output_dim = 2
        self.hidden_dim = hidden_dim
        self.hidden_activation = F.tanh  # relu does not work

        self.lb = xyt.min()
        self.ub = xyt.max()

        # Initialize parameters
        # learnable lambda
        # self.lambda_1 = nn.Parameter(torch.zeros(1, dtype=torch.float64), requires_grad=True)
        # self.lambda_2 = nn.Parameter(torch.zeros(1, dtype=torch.float64), requires_grad=True)
        # given lambda
        self.lambda_1 = torch.Tensor([1.]).to(device)
        self.lambda_2 = torch.Tensor([0.01]).to(device)

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

    def net_NS(self, xyt): # xyt.size()=(N*T,3) Xbatch=N*T
        xyt.requires_grad=True 
        self.create_graph = True
        self.retain_graph = True

        psi_and_p = self.neural_net(xyt)
        psi = psi_and_p[:,0] # psi.size(), p.size()=N*T
        p = psi_and_p[:,1]

        # calculate u,v, p_x, p_y
        Dpsi = grad(psi, xyt, grad_outputs=torch.ones_like(psi), create_graph=self.create_graph, retain_graph=self.retain_graph)
        Dp = grad(p, xyt, grad_outputs=torch.ones_like(p), create_graph=self.create_graph, retain_graph=self.retain_graph)

        psi_x = Dpsi[0][:,0]
        psi_y = Dpsi[0][:,1]
        psi_t = Dpsi[0][:,2]

        u = psi_y
        v = -psi_x

        p_x = Dp[0][:,0]
        p_y = Dp[0][:,1]
        p_t = Dp[0][:,2]


        # calculate u_x, u_y, u_t; v_x, v_y, v_t
        Du = grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=self.create_graph, retain_graph=self.retain_graph)
        Dv = grad(v, xyt, grad_outputs=torch.ones_like(v), create_graph=self.create_graph, retain_graph=self.retain_graph)

        u_x = Du[0][:,0]
        u_y = Du[0][:,1]
        u_t = Du[0][:,2]

        v_x = Dv[0][:,0]
        v_y = Dv[0][:,1]
        v_t = Dv[0][:,2]

        # calculate u_xx, u_yy; v_xx, v_yy
        Du_x = grad(u_x, xyt, grad_outputs=torch.ones_like(u), create_graph=self.create_graph, retain_graph=self.retain_graph)
        Du_y = grad(u_y, xyt, grad_outputs=torch.ones_like(u), create_graph=self.create_graph, retain_graph=self.retain_graph)
        Dv_x = grad(v_x, xyt, grad_outputs=torch.ones_like(u), create_graph=self.create_graph, retain_graph=self.retain_graph)
        Dv_y = grad(v_y, xyt, grad_outputs=torch.ones_like(u), create_graph=self.create_graph, retain_graph=self.retain_graph)

        u_xx = Du_x[0][:,0]
        u_yy = Du_y[0][:,1]

        v_xx = Dv_x[0][:,0]
        v_yy = Dv_y[0][:,1]

        # calculate two components of f
        f_u = u_t + self.lambda_1*(u*u_x + v*u_y) + p_x - self.lambda_2*(u_xx + u_yy) 
        f_v = v_t + self.lambda_1*(u*v_x + v*v_y) + p_y - self.lambda_2*(v_xx + v_yy)
    
        return u, v, p, f_u, f_v
        
    def forward(self, xyt):
        return self.net_NS(xyt)
    
    def loss_function(self, xyt, xyt_, u_label, v_label):
        labeled_data_number = xyt_.shape[0]
        xyt = torch.cat([xyt, xyt_], dim=0)
        u, v, p, f_u, f_v = self.forward(xyt)

        u_loss = (u_label.squeeze() - u[-labeled_data_number:]).pow(2).mean()
        v_loss = (v_label.squeeze() - v[-labeled_data_number:]).pow(2).mean()
        f_u_loss = f_u.pow(2).mean()
        f_v_loss = f_v.pow(2).mean()

        loss = u_loss + v_loss + f_u_loss + f_v_loss

        return loss, u_loss, v_loss

        # return  (self.u.squeeze() - u).pow(2).sum()+\
        #         (self.v.squeeze() - v).pow(2).sum()+\
        #         f_u.pow(2).sum()+\
        #         f_v.pow(2).sum()


def load_data(device):
    # Data loading and processing
    data = scipy.io.loadmat('./data/cylinder_nektar_wake.mat')

    U_star = data['U_star'] # N x 2 x T; full data of (u, v) at all x, y, t
    P_star = data['p_star'] # N x T; full data of p at all x, y, t
    t_star = data['t'] # T x 1; t corrdinates of the full data
    XY_star = data['X_star'] # N x 2; (x,y) coordinates of the full data

    N = XY_star.shape[0] # N=5000; size of the full data at a fixed t
    T = t_star.shape[0] # T=200; size of the full data at a fixed x,y


    # Definition of u, v, p
    u = torch.FloatTensor(U_star[:,0,:].flatten()[:,None]) # NT x 1
    v = torch.FloatTensor(U_star[:,1,:].flatten()[:,None]) # NT x 1
    p = torch.FloatTensor(P_star.flatten()[:,None]) # NT x 1

    # definition of xyt, u, v, p
    xy = torch.FloatTensor(XY_star)
    t = torch.FloatTensor(t_star)

    xyxy = torch.cat([torch.tile(xy[:,0:1],(1,T)).flatten().unsqueeze(-1),torch.tile(xy[:,1:2],(1,T)).flatten().unsqueeze(-1)],-1)# xy.size()=(N,2)
    tt = torch.tile(t.unsqueeze(0), (N,1)).flatten().unsqueeze(1) # t.size()=(N*T,1)

    xyt = torch.cat([xyxy,tt], 1) # NT x 3

    return xyt.to(device), u.to(device), v.to(device), p.to(device), N, T

def plot(x):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(x)
    plt.savefig('nn.png')
    # plt.show()

def eval(pinn, xyt_test, u_test, v_test, p_test, verbose=False):
    u_pred, v_pred, p_pred = pinn(xyt_test)[0:3]
    lambda_1_value = pinn.lambda_1.detach().cpu().numpy()
    lambda_2_value = pinn.lambda_2.detach().cpu().numpy()

    u_pred = u_pred.detach().cpu().numpy()
    v_pred = v_pred.detach().cpu().numpy()
    p_pred = p_pred.detach().cpu().numpy()
    u_test = np.squeeze(u_test)
    v_test = np.squeeze(v_test)
    p_test = np.squeeze(p_test)

    # Error
    # error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
    # error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
    # error_p = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)
    error_u = ((u_test-u_pred)**2).mean()
    error_v = ((v_test-v_pred)**2).mean()
    error_p = ((p_test-p_pred)**2).mean()

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100

    if verbose:
        print('Error u: %e' % (error_u))    
        print('Error v: %e' % (error_v))    
        print('Error p: %e' % (error_p))    
        print('Error l1: %.5f%%' % (error_lambda_1))                             
        print('Error l2: %.5f%%' % (error_lambda_2)) 

    return error_u, error_v, error_p, error_lambda_1, error_lambda_2   

# Training
def train(pinn, nIter, batch, xyt_train, xyt_data, u_data, v_data, xyt_test, u_test, v_test, p_test, model_path): 
    losses = []
    start_time = time.time()
    adapt_lr = pinn.optim.param_groups[0]['lr']
    for it in range(nIter):
        sample_idx = np.random.choice(xyt_train.shape[0], batch, replace=False)# cannot forward the whole dataset, sample a batch containing data points
        xyt = xyt_train[sample_idx]
        loss_train, u_loss, v_loss = pinn.loss_function(xyt, xyt_data, u_data, v_data)
        pinn.optim.zero_grad()
        loss_train.backward()
        pinn.optim.step()
        lambda_1_value = pinn.lambda_1
        lambda_2_value = pinn.lambda_2
        losses.append(loss_train.item())
        if (it+1)%10000 == 0:
            pinn.scheduler.step()
            adapt_lr = pinn.optim.param_groups[0]['lr']
            # for param_group in pinn.optim.param_groups:
            #     print(param_group['lr'])
            torch.save(pinn.state_dict(), model_path)

        if it % 100 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, lamd1: %.3f, lamd2: %.5f, Time: %.2f' % 
                  (it, loss_train, lambda_1_value, lambda_2_value, elapsed))
            start_time = time.time()
            # plot(losses)
            error_u, error_v, error_p, error_lambda_1, error_lambda_2 = eval(pinn, xyt_test, u_test, v_test, p_test)
            writer.add_scalar('Train/Loss', loss_train, it)
            # writer.add_scalars('Train/compare_u', {'train': u_loss, 'test': error_u}, it) # this mess up the runs folder
            # writer.add_scalars('Train/compare_v', {'train': v_loss, 'test': error_v}, it)
            writer.add_scalar('Train/u', u_loss, it)
            writer.add_scalar('Train/v', v_loss, it)
            writer.add_scalar('Train/LR', adapt_lr, it)
            writer.add_scalar('Test Error/u', error_u, it)
            writer.add_scalar('Test Error/v', error_v, it)
            writer.add_scalar('Test Error/p', error_p, it)
            writer.add_scalar('Test Error/lambda1', error_lambda_1, it)
            writer.add_scalar('Test Error/lambda2', error_lambda_2, it)


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--data', type=int, default=5000)
    parser.add_argument('--dim', type=int, default=20)
    args = parser.parse_args()
    writer = SummaryWriter('runs/'+args.id+'_'+str(args.data)+'_'+str(args.dim))

    # Training Process
    layers = 2
    nIter = 200000  # original niter is 200000
    lr = 0.01
    batch = 5000
    hidden_dim = int(args.dim)

    optim_method = "adam"
    model_path = './model/'
    postfix = args.id if args.id is not None else '0'
    model_path += postfix

    xyt, u, v, p, N, T = load_data(device)
    print(xyt.shape)
    N_total = N*T
    N_train = int(args.data) # 1000 number of datapoints, not collocation points
    N_test = 1000

    # Test data
    test_idx = np.random.choice(N_total, N_test, replace=False) # Generate a random sample from np.arange(N*T) of size N_train without replacement
    xyt_test = xyt[test_idx,:]
    u_test = u[test_idx,:].cpu().numpy()
    v_test = v[test_idx,:].cpu().numpy()
    p_test = p[test_idx,:].cpu().numpy()

    # Training data as the rest from the test data: collocation points
    train_idx = list(set(np.arange(N_total)) - set(test_idx))
    xyt_train = xyt[train_idx,:]


    data_idx = np.random.choice(train_idx, N_train, replace=False) # data points
    xyt_data = xyt[data_idx,:]
    u_data = u[data_idx,:]
    v_data = v[data_idx,:]
    print(f"Total samples in dataset: {N_total}, collocation points: {len(train_idx)}, data poinst: {len(data_idx)}, test points: {len(test_idx)}")

    # Training
    pinn = PhysicsInformedNN(xyt_data, layers, device, optim_method, hidden_dim, lr).to(device) 
    train(pinn, nIter, batch, xyt_train, xyt_data, u_data, v_data, xyt_test, u_test, v_test, p_test, model_path)

    # Prediction
    eval(pinn, xyt_test, u_test, v_test, p_test, verbose=True)
