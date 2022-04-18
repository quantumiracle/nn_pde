import torch
import numpy as np
# import matplotlib.pyplot as plt
import scipy.io
# from scipy.interpolate import griddata
import time, math
# from itertools import product, combinations
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# # from plotting import newfig, savefig
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.gridspec as gridspec
# from matplotlib import rc
from IPython.display import clear_output
import matplotlib.pyplot as plt

np.random.seed(123)
torch.manual_seed(100)

print(torch.__version__)


# Model
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import backward
import torch.nn.functional as F


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

class PhysicsInformedNN(nn.Module):
    def __init__(self, xyt, u, v, layers, optim_method, lr): # xyt.size()=(N*T,3), Xbatch=N*T
        super().__init__()
        self.xyt = xyt
        self.u = u
        self.v = v
        
        self.lb = xyt.min()
        self.ub = xyt.max()

        self.layers = layers
        self.input_dim = 3
        self.output_dim = 2
        self.hidden_dim = 64
        self.hidden_activation = F.tanh  # relu does not work
        
        # Initialize parameters
        self.lambda_1 = nn.Parameter(torch.zeros(1, dtype=torch.float64), requires_grad=True)
        self.lambda_2 = nn.Parameter(torch.zeros(1, dtype=torch.float64), requires_grad=True)

        self.input_layer =  nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.layers)]
        self.ouput_layer = nn.Linear(self.hidden_dim, self.output_dim)
        # for p in self.parameters():
        #     print(p.data)
        self.apply(self._weight_init)

        if optim_method == "adam":
            self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        elif optim_method == "sgd":
            self.optim = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            print("THIS OPTIMIZATION METHOD IS NOT SUPPORTED.")
            return 0

    # def initialize_NN(self, layers):
    #     weights = []
    #     biases = []
    #     num_layers = len(layers) 
    #     for l in range(0,num_layers-1):
    #         W = self.xavier_init(size=[layers[l], layers[l+1]])
    #         b = nn.Parameter(torch.zeros([layers[l+1]], dtype=torch.float64))
    #         weights.append(W)
    #         biases.append(b) 
    #     return weights, biases

    # def xavier_init(self, size):
    #     in_dim = size[0]
    #     out_dim = size[1]        
    #     xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    #     return nn.Parameter(xavier_stddev*torch.randn([in_dim, out_dim], dtype=torch.float64))

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
        # delta = 1e-7

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
        # e_0 = torch.tensor([[0,1,0]]).tile((xyt.size(0),1))
        # u_diff=1./delta*(self.neural_net(xyt+delta*e_0, self.weights, self.biases) -\
        #         self.neural_net(xyt, self.weights, self.biases) )[:,0]
        # print("Total difference between numerical derivative and pytorch derivative: %e" % ((u_diff-u).pow(2).sum().squeeze()))
        # for idx, (u_d_i, u_i) in enumerate(zip(u_diff, u)):
        #     if torch.abs(u_d_i - u_i)>1e-1:
        #         print(f"u_diff-u: {idx}, {u_d_i-u_i}, {u_d_i:.6f}, {u_i:.6f}")

#                 print("u_diff[65]-u[65]: %e" %(u_diff[65]-u[65]) )
#                 print("u_diff[65]: %e" %(u_diff[65]))
#                 print("u[65]: %e" %(u[65]))
#                 print("u_diff[66]-u[66]: %e" %(u_diff[66]-u[66]) )
#                 print("u_diff[66]: %e" %(u_diff[66]))
#                 print("u[66]: %e" %(u[66]))


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
    
    def loss_function(self):
        u, v, p, f_u, f_v = self.forward(self.xyt)

        return  (self.u.squeeze() - u).pow(2).sum()+\
                (self.v.squeeze() - v).pow(2).sum()+\
                f_u.pow(2).sum()+\
                f_v.pow(2).sum()

def load_data():
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


# Training
def train(pinn, nIter): 
    losses = []
    start_time = time.time()
    for it in range(nIter):
        loss_value = pinn.loss_function()
        pinn.optim.zero_grad()
        loss_value.backward()
        pinn.optim.step()
        lambda_1_value = pinn.lambda_1
        lambda_2_value = pinn.lambda_2
        losses.append(loss_value.item())
        if it % 100 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                  (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
            start_time = time.time()
            plot(losses)


if __name__ == "__main__":
    # Training Process
    N_train = 500 #5000
    N_test = 1000
        
    layers = 3

    nIter = 10000 #2000  # original niter is 200000

    lr = 0.001

    optim_method = "adam"

    xyt, u, v, p, N, T = load_data()

    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False) # Generate a random sample from np.arange(N*T) of size N_train without replacement
    xyt_train = xyt[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    # Training
    pinn = PhysicsInformedNN(xyt_train, u_train, v_train, layers, optim_method, lr).to(device) 
    train(pinn, nIter)

    # Test Data
    idx = np.random.choice(N*T, N_test, replace=False) # Generate a random sample from np.arange(N*T) of size N_train without replacement
    xyt_test = xyt[idx,:]
    u_test = u[idx,:]
    v_test = v[idx,:]
    p_test = p[idx,:]

    # Prediction
    u_pred, v_pred, p_pred = pinn(xyt_test)[0:3]
    lambda_1_value = pinn.lambda_1.detach().cpu().numpy()
    lambda_2_value = pinn.lambda_2.detach().cpu().numpy()

    u_pred = u_pred.detach().cpu().numpy()
    v_pred = v_pred.detach().cpu().numpy()
    p_pred = p_pred.detach().cpu().numpy()

    # Error
    error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
    error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
    error_p = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100

    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))           
