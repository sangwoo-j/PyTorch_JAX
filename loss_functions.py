# ------------------------------- #
# Loss functions

import torch
import torch.nn as nn
import numpy as np

default_cost_function = torch.nn.MSELoss()

def pde_loss(u_hat, coll_points, cost_function=None):
    if cost_function is None:
        cost_function = default_cost_function

    x_coll = coll_points[:, 0].reshape(-1, 1)
    y_coll = coll_points[:, 1].reshape(-1, 1)
    
    u = u_hat(x_coll, y_coll)

    u_x = torch.autograd.grad(u.sum(), x_coll, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y_coll, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x.sum(), x_coll, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y_coll, create_graph=True)[0]

    residual = u_xx + u_yy
    
    return cost_function(residual, torch.full_like(residual, 0))

def bc1_loss(u_hat, bc1_points, cost_function=None):
    if cost_function is None:
        cost_function = default_cost_function
    
    x_bc1 = bc1_points[:, 0].reshape(-1, 1)
    y_bc1 = bc1_points[:, 1].reshape(-1, 1)

    u_pred = u_hat(x_bc1, y_bc1)
    
    return cost_function(u_pred, torch.full_like(u_pred, 0))

def bc2_loss(u_hat, bc2_points, cost_function=None):
    if cost_function is None:
        cost_function = default_cost_function
    
    x_bc2 = bc2_points[:, 0].reshape(-1, 1)
    y_bc2 = bc2_points[:, 1].reshape(-1, 1)

    u_pred = u_hat(x_bc2, y_bc2)
    
    return cost_function(u_pred, torch.full_like(u_pred, 1))

def bc3_loss(u_hat, bc3_points, cost_function=None):
    if cost_function is None:
        cost_function = default_cost_function
    
    x_bc3 = bc3_points[:, 0].reshape(-1, 1)
    y_bc3 = bc3_points[:, 1].reshape(-1, 1)

    u_pred = u_hat(x_bc3, y_bc3)
    
    return cost_function(u_pred, torch.full_like(u_pred, 0))

def bc4_loss(u_hat, bc4_points, cost_function=None):
    if cost_function is None:
        cost_function = default_cost_function
    
    x_bc4 = bc4_points[:, 0].reshape(-1, 1)
    y_bc4 = bc4_points[:, 1].reshape(-1, 1)

    u_pred = u_hat(x_bc4, y_bc4)
    
    return cost_function(u_pred, torch.full_like(u_pred, 0))

def total_loss(u_hat, coll_points, bc1_points, bc2_points, bc3_points, bc4_points, cost_function=None):

    return pde_loss(u_hat, coll_points, cost_function)\
            + bc1_loss(u_hat, bc1_points, cost_function)\
            + bc2_loss(u_hat, bc2_points, cost_function)\
            + bc3_loss(u_hat, bc3_points, cost_function)\
            + bc4_loss(u_hat, bc4_points, cost_function)