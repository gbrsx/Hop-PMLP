from re import S
from xml.dom import xmlbuilder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import num_nodes, to_dense_adj
import numpy as np
import math
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from conv import *
from hopgnn import *

class PMLP_HOPGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node, edge_index, num_hop=6, hop_interaction="attention"):
        super(PMLP_HOPGNN, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.num_hop = num_hop

        adj = to_dense_adj(edge_index)[0] if isinstance(edge_index, torch.Tensor) else edge_index
        self.hop_gnn = HopGNN(
            g=adj,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # Adjust to match concatenation dimension
            num_hop=num_hop,
            feature_inter=hop_interaction
        )

        # Define feed-forward layers
        self.fcs = nn.ModuleList([nn.Linear(in_channels, hidden_channels)])
        for _ in range(self.num_layers - 2):
            self.fcs.append(nn.Linear(hidden_channels, hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, hidden_channels))

        # Define final layer, setting input to hidden_channels * 2 to accommodate concatenation
        self.final_layer = nn.Linear(hidden_channels * 2, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fcs:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if hasattr(self.hop_gnn, 'reset_parameters'):
            self.hop_gnn.reset_parameters()
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x, edge_index, use_conv=True):
        original_x = x  # Keep a copy of the original input
    
        # Process x through PMLP layers during training
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # During training, apply only a single message-passing layer from HopGNN
        if self.training:
            hop_output = self.hop_gnn(original_x, single_layer=True)
            x = torch.cat((x, hop_output), dim=1)  # Concatenate along the feature dimension
            x = self.final_layer(x)  # Map concatenated features to output size
        else:
            # During inference, use the full HopGNN model
            hop_output = self.hop_gnn(original_x, single_layer=False)
            x = torch.cat((x, hop_output), dim=1)  # Concatenate along the feature dimension
            x = self.final_layer(x)
    
        return x
            
# Implementation of PMLP_GCN, which can become MLP or GCN depending on whether using message passing
class PMLP_GCN(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_GCN, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True  # Use bias for FF layers in default

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.num_layers - 1):
            x = x @ self.fcs[i].weight.t() 
            if use_conv: x = gcn_conv(x, edge_index)  # Optionally replace 'gcn_conv' with other conv functions in conv.py
            if self.ff_bias: x = x + self.fcs[i].bias
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x @ self.fcs[-1].weight.t() 
        if use_conv: x = gcn_conv(x, edge_index)
        if self.ff_bias: x = x + self.fcs[-1].bias
        return x


# Implementation of PMLP_SGC, which can become MLP or SGC depending on whether using message passing
class PMLP_SGC(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_SGC, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.ff_bias = True

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.args.num_mps):
            if use_conv: x = gcn_conv(x, edge_index)

        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fcs[-1](x) 
        return x


# Implementation of PMLP_APP, which can become MLP or SGC depending on whether using message passing
class PMLP_APPNP(nn.Module): #residual connection
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_APPNP, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.fcs[-1](x) 
        for i in range(self.args.num_mps):
            if use_conv: x = gcn_conv(x, edge_index)    
        return x
    


# The rest models are used for additional experiments in the paper

# Implementation of PMLP_GCNII, which can become ResNet (MLP with residual connections) or GCNII depending on whether using message passing
class PMLP_GCNII(nn.Module): #GCNII
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_GCNII, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        x = x @ self.fcs[0].weight.t()
        x = self.activation(self.bns(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_ = x.clone()

        for i in range(1, self.num_layers - 1):
            x = x * (1. - 0.5 / i) + x @ self.fcs[i].weight.t() * (0.5 / i) 
            if use_conv: x = conv_resi(x, edge_index, x_)
            else: x = 0.9 * x + 0.1 * x_
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x =  x @ self.fcs[-1].weight.t() 
        if use_conv: x = gcn_conv(x, edge_index)
        return x


class PMLP_JKNet(nn.Module): #JKNET(concatation pooling)
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_JKNet, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels * (self.num_layers - 1), out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        xs = []
        for i in range(0, self.num_layers - 1):
            x = x @ self.fcs[i].weight.t() 
            if use_conv: x = gcn_conv(x, edge_index)
            x = self.activation(self.bns(x))
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.cat(xs, dim=-1)
        x =  x @ self.fcs[-1].weight.t() 
        return x

class PMLP_SGCres(nn.Module): #SGC with residual connections
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_SGCres, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.ff_bias = True

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)

        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fcs[-1](x) 
        return x



class PMLP_SGCresinf(nn.Module): #SGC with residual connections (in test but not in train)
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_SGCresinf, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.ff_bias = True

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)
            else: x =  gcn_conv(x, edge_index)

        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fcs[-1](x) 
        return x
    
    
class PMLP_APPNPres(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_APPNPres, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.fcs[-1](x) 
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)
        return x


class PMLP_APPNPresinf(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_APPNPresinf, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.fcs[-1](x) 
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)
            else: x =  gcn_conv(x, edge_index)
        return x