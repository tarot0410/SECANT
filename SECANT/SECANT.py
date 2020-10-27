#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import math
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import time
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import umap
import matplotlib




# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    #device = get_device()
    return torch.from_numpy(df.values).float()#.to(device)

# Compute log likelihood for the dataset without labels
def logLik(data0, tauVec, muMtx, cov_3D, K, N):   
    # create raw log-tau matrix (n*K)
    logTauVec = torch.log(tauVec)
    tauMtxTemp0 = logTauVec.repeat(N, 1)

    # set up raw log density matrix (K*N)
    logDMtxTemp0 = torch.empty(K, N, dtype = torch.float64).to(device)
    for k in range(K):
        mvn_k = dist.MultivariateNormal(muMtx[k,:], cov_3D[k])
        temp_logD = mvn_k.log_prob(data0)
        logDMtxTemp0[k,:] = temp_logD

    # Compute log likelihood
    tauMtx0 = tauMtxTemp0 - logTauVec[0]
    logDMtx0 = logDMtxTemp0 - logDMtxTemp0[0,:]
    
    logDMtx0[logDMtx0>690.7] = 690.7 # avoid too large value
    if torch.max(logDMtx0) == 690.7:
        print("logDMtx0_max too large!")

    logLikTemp0 = torch.mm(torch.exp(tauMtx0), torch.exp(logDMtx0))

    logLikTemp1 = torch.trace(torch.log(logLikTemp0))
    logLik = logLikTemp1 + N*logTauVec[0] + torch.sum(logDMtxTemp0[0,:])
    return logLik

# Compute log likelihood and posterior prob matrix for the dataset with labels
def logLikDelta(data0, tauVec, muMtx, cov_3D, K, N):
    # create raw log-tau matrix (N*K)
    logTauVec = torch.log(tauVec)
    tauMtxTemp0 = logTauVec.repeat(N, 1)

    # set up raw log density matrix (K*N)
    logDMtxTemp0 = torch.empty(K, N, dtype = torch.float64).to(device)
    for k in range(K):
        mvn_k = dist.MultivariateNormal(muMtx[k,:], cov_3D[k])
        temp_logD = mvn_k.log_prob(data0)
        logDMtxTemp0[k,:] = temp_logD

    # Compute log likelihood
    tauMtx0 = tauMtxTemp0 - logTauVec[0]
    logDMtx0 = logDMtxTemp0 - logDMtxTemp0[0,:]

    logDMtx0[logDMtx0>690.7] = 690.7 # avoid too large value
    if torch.max(logDMtx0) == 690.7:
        print("logDMtx0_max too large!")
    
    logLikTemp0 = torch.mm(torch.exp(tauMtx0), torch.exp(logDMtx0))
    logLikTemp1 = torch.trace(torch.log(logLikTemp0))
    logLik = logLikTemp1 + N*logTauVec[0] + torch.sum(logDMtxTemp0[0,:])

    # Compute delta matrix
    deltaMtx = torch.empty(N, K, dtype=torch.float64).to(device)
    deltaMtx[:,0] = 1/torch.diagonal(logLikTemp0)
    for j in range(1, K):      
        tauMtx_j = tauMtxTemp0 - logTauVec[j]
        logDMtx_j = logDMtxTemp0 - logDMtxTemp0[j,:]
        logLikTemp_j = torch.mm(torch.exp(tauMtx_j), torch.exp(logDMtx_j))
        deltaMtx[:,j] = 1/torch.diagonal(logLikTemp_j)
    
    return logLik, deltaMtx

# compute logLikelihood of observed data for one dataset with labels case
def fullLogLik1(data0, tauVec, muMtx, conMtx, cov3D, classLabel_array, K, N):  
    l1, deltaMtx = logLikDelta(data0, tauVec, muMtx, cov3D, K, N)
    conMtxFull = torch.empty(N, K, dtype = torch.float64).to(device)
    deltaMtxT = torch.transpose(deltaMtx,0,1)
    for i in range(N):
        conMtxFull[i,:] = conMtx[classLabel_array[i].astype(int),:] 
        
    tempMtx = torch.mm(conMtxFull, deltaMtxT)
    temp = torch.trace(torch.log(tempMtx))
    return temp + l1

# compute logLikelihood of observed data for two datasets, one with labels and the other doesn't case
def fullLogLik2(data0, data1, tauVec0, tauVec1, muMtx, conMtx, cov3D, classLabel_array, K, N0, N1):  
    l0, deltaMtx = logLikDelta(data0, tauVec0, muMtx, cov3D, K, N0)
    conMtxFull = torch.empty(N0, K, dtype = torch.float64).to(device)
    deltaMtxT = torch.transpose(deltaMtx,0,1)
    for i in range(N0):
        conMtxFull[i,:] = conMtx[classLabel_array[i].astype(int),:] 
        
    tempMtx = torch.mm(conMtxFull, deltaMtxT)
    l1 = logLik(data1, tauVec1, muMtx, cov3D, K, N1)
    return torch.trace(torch.log(tempMtx)) + l0 + l1



# Function for initialization
# Input numCluter: a list, where each number refers to the number of 
# sub clusters for a common type (matched by index)
# ex. [1,2,2,1] --> type 1 has 1 component, type 2 has 2 components...
# Output structure of conMtx, initialized muMtx and tauVec
def initParam(data0, numCluster, C, K, P, cls_np, init_seed):
    data0_np = data0.cpu().numpy()
    conMtx_temp = torch.zeros(C-1, K, dtype = torch.float64).to(device)
    mu_init = torch.empty(K, P, dtype = torch.float64)
    tau_temp = torch.empty(K, dtype = torch.float64)
    ct_ind = 0
    for c in range(C-1):
        subData = data0_np[cls_np == c, :]
        numC0 = numCluster[c]
        # set up conMtx
        conMtx_temp[c, ct_ind: (ct_ind + numC0)] = 1
        # gmm for sub cluster
        gmmSub = GaussianMixture(n_components=numC0, random_state=init_seed).fit(subData)
        # set up cluster parameters
        mu_init[ct_ind: (ct_ind + numC0)] = torch.tensor(gmmSub.means_, dtype = torch.float64)
        tau_temp[ct_ind: (ct_ind + numC0)] = torch.tensor(gmmSub.weights_, dtype = torch.float64)*sum(cls_np == c)

        ct_ind += numC0

    tau_init = tau_temp/tau_temp.sum()
    return conMtx_temp, mu_init, tau_init

# Set up optimization algorithm for one dataset case
def optimDL1(parameters, data0, conMtx_temp, C, K, P, N, cls_np, learning_rate, nIter):
    # Defines a SGD optimizer to update the parameters
    optimizer = optim.Rprop(parameters, lr=learning_rate) 

    pVec, tauVec, muMtx, scale_3D = parameters
    
    for i in range(nIter):
        optimizer.zero_grad()

        # set up transformed p vector
        pVec_tran = dist.biject_to(dist.Binomial.arg_constraints['probs'])(pVec)

        # set up transformed concordance matrix
        conMtx_tran = torch.empty(C, K, dtype = torch.float64).to(device)
        conMtx_tran[0:(C-1),:] = conMtx_temp*pVec_tran
        conMtx_tran[C-1,:] = 1-pVec_tran

        # set up transformed tau vector
        tauVec_tran = dist.biject_to(dist.Multinomial.arg_constraints['probs'])(tauVec)
        # tauVec_tran = dist.transform_to(dist.Multinomial.arg_constraints['probs'])(tauVec)

        # set up transformed cov3D matrix
        cov3D_tran = torch.empty(K, P, P, dtype = torch.float64).to(device)
        for m in range(K):
            cov3D_chol_m = dist.transform_to(dist.constraints.lower_cholesky)(scale_3D[m])
            cov3D_tran[m] = torch.mm(cov3D_chol_m, cov3D_chol_m.t())

        # Define loss function xMtx, piVec, alphaMtx, K
        NLL = - fullLogLik1(data0, tauVec_tran, muMtx, conMtx_tran, cov3D_tran, cls_np, K, N)
        
        if i % 20 == 0:
            print("")
            print(i, "loglik  =", -NLL.cpu().data.numpy())
            print("conMtx:")
            print(np.around(conMtx_tran.cpu().data.numpy(),3))
            print("tauVec:")
            print(np.around(tauVec_tran.cpu().data.numpy(),3))
        
        NLL.backward()
        optimizer.step()
        
    return conMtx_tran, tauVec_tran, muMtx, cov3D_tran, -NLL

# Final function to run algorithm for one dataset case
def SECANT_CITE(data0, numCluster, K, cls_np, device, uncertain = True, learning_rate=0.01, nIter=100, init_seed=2020):
    torch.manual_seed(init_seed)
    
    N = data0.size()[0]
    P = data0.size()[1] 
    if uncertain:
        C = np.unique(cls_np).size
    else:
        C = np.unique(cls_np).size+1
   
    # concordance p vector
    p_init = torch.ones(K, dtype = torch.float64)*0.5
    pVec = dist.biject_to(dist.Binomial.arg_constraints['probs']).inv(p_init)
    
    # clustering parameters
    conMtx_temp, mu_init, tau_init = initParam(data0, numCluster, C, K, P, cls_np, init_seed)

    # initial cov3D
    scale3D = torch.empty(K, P, P, dtype = torch.float64).to(device)
    for k in range(K):
        scale3D[k] = torch.eye(P, dtype = torch.float64).to(device)*0.01

    tauVec = dist.biject_to(dist.Multinomial.arg_constraints['probs']).inv(tau_init)
    muMtx = mu_init.clone()

    pVec = pVec.to(device)
    tauVec = tauVec.to(device)
    muMtx = muMtx.to(device)

    pVec.requires_grad = True
    tauVec.requires_grad = True
    muMtx.requires_grad = True

    scale3D = scale3D.to(device)
    scale3D.requires_grad = True
    param = [pVec, tauVec, muMtx, scale3D]

    conMtxFinal, tauVecFinal, muMtxFinal, cov3DFinal, loglikFinal = optimDL1(param, data0, conMtx_temp, C, K, P, N, cls_np, learning_rate, nIter)

    l1, deltaMtxFinal = logLikDelta(data0, tauVecFinal, muMtxFinal, cov3DFinal, K, N)
    outLbl = torch.zeros(N)
    for i in range(N):
        values, indices = torch.max(deltaMtxFinal[i,:], 0)
        outLbl[i] = indices
    
    return outLbl, conMtxFinal, tauVecFinal, muMtxFinal, cov3DFinal, loglikFinal

# Set up optimization algorithm for two datasets case
def optimDL2(parameters, data0, data1, conMtx_temp, C, K, P, N0, N1, cls_np, learning_rate, nIter):
    # Defines a SGD optimizer to update the parameters
    optimizer = optim.Rprop(parameters, lr=learning_rate) 

    pVec, tauVec0, tauVec1, muMtx, scale_3D = parameters
    
    for i in range(nIter):
        optimizer.zero_grad()

        # set up transformed p vector
        pVec_tran = dist.biject_to(dist.Binomial.arg_constraints['probs'])(pVec)

        # set up transformed concordance matrix
        conMtx_tran = torch.empty(C, K, dtype = torch.float64).to(device)
        conMtx_tran[0:(C-1),:] = conMtx_temp*pVec_tran
        conMtx_tran[C-1,:] = 1-pVec_tran

        # set up transformed tau vector
        tauVec_tran0 = dist.biject_to(dist.Multinomial.arg_constraints['probs'])(tauVec0)
        tauVec_tran1 = dist.biject_to(dist.Multinomial.arg_constraints['probs'])(tauVec1)

        # set up transformed cov3D matrix
        cov3D_tran = torch.empty(K, P, P, dtype = torch.float64).to(device)
        for m in range(K):
            cov3D_chol_m = dist.transform_to(dist.constraints.lower_cholesky)(scale_3D[m])
            cov3D_tran[m] = torch.mm(cov3D_chol_m, cov3D_chol_m.t())

        # Define loss function xMtx, piVec, alphaMtx, K
        NLL = - fullLogLik2(data0, data1, tauVec_tran0, tauVec_tran1, muMtx, conMtx_tran, cov3D_tran, cls_np, K, N0, N1)
        
        if i % 20 == 0:
            print("")
            print(i, "loglik  =", -NLL.cpu().data.numpy())
            print("conMtx:")
            print(np.around(conMtx_tran.cpu().data.numpy(),3))
            print("tauVec0:")
            print(np.around(tauVec_tran0.cpu().data.numpy(),3))
            print("tauVec1:")
            print(np.around(tauVec_tran1.cpu().data.numpy(),3))
        
        NLL.backward()
        optimizer.step()
        
    return conMtx_tran, tauVec_tran0, tauVec_tran1, muMtx, cov3D_tran, -NLL

# Final function to run algorithm for one dataset case
def SECANT_joint(data0, data1, numCluster, K, cls_np, device, uncertain = True, learning_rate=0.01, nIter=100, init_seed=2020):
    torch.manual_seed(init_seed)
    
    N0 = data0.size()[0]
    N1 = data1.size()[0]
    P = data0.size()[1] 
    
    if uncertain:
        C = np.unique(cls_np).size
    else:
        C = np.unique(cls_np).size+1
    
    # concordance p vector
    p_init = torch.ones(K, dtype = torch.float64)*0.5
    pVec = dist.biject_to(dist.Binomial.arg_constraints['probs']).inv(p_init)
    
    # clustering parameters
    conMtx_temp, mu_init, tau_init = initParam(data0, numCluster, C, K, P, cls_np, init_seed)

    # initial cov3D
    scale3D = torch.empty(K, P, P, dtype = torch.float64).to(device)
    for k in range(K):
        scale3D[k] = torch.eye(P, dtype = torch.float64).to(device)*0.01

    tauVec0 = dist.biject_to(dist.Multinomial.arg_constraints['probs']).inv(tau_init)
    tauVec1 = dist.biject_to(dist.Multinomial.arg_constraints['probs']).inv(tau_init)
    muMtx = mu_init.clone()

    pVec = pVec.to(device)
    tauVec0 = tauVec0.to(device)
    tauVec1 = tauVec1.to(device)
    muMtx = muMtx.to(device)

    pVec.requires_grad = True
    tauVec0.requires_grad = True
    tauVec1.requires_grad = True
    muMtx.requires_grad = True

    scale3D = scale3D.to(device)
    scale3D.requires_grad = True
    param = [pVec, tauVec0, tauVec1, muMtx, scale3D]

    conMtxFinal, tauVec0Final, tauVec1Final, muMtxFinal, cov3DFinal, loglikFinal = optimDL2(param, data0, data1, conMtx_temp, C, K, P, N0, N1, cls_np, learning_rate, nIter)

    l0, deltaMtxFinal0 = logLikDelta(data0, tauVec0Final, muMtxFinal, cov3DFinal, K, N0)
    outLbl0 = torch.zeros(N0)
    for i in range(N0):
        values, indices = torch.max(deltaMtxFinal0[i,:], 0)
        outLbl0[i] = indices

    l1, deltaMtxFinal1 = logLikDelta(data1, tauVec1Final, muMtxFinal, cov3DFinal, K, N1)
    outLbl1 = torch.zeros(N1)
    for j in range(N1):
        values, indices = torch.max(deltaMtxFinal1[j,:], 0)
        outLbl1[j] = indices
    
    preditADT = torch.mm(conMtxFinal, deltaMtxFinal1.t())
    
    return outLbl0, outLbl1, preditADT, conMtxFinal, tauVec0Final, tauVec1Final, muMtxFinal, cov3DFinal, loglikFinal
