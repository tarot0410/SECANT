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
import umap.umap_ as umap
import matplotlib.pyplot as plt
import itertools
from itertools import combinations
import gc

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    return torch.from_numpy(df.values).float().to(device)

def get_likelihoods(X, muMtx, scale3D, log=True):
    """
    :param X: design matrix (examples, features)
    :param mu: the component means (K, features)
    :param logvar: the component log-variances (K, features)
    :param log: return value in log domain?
        Note: exponentiating can be unstable in high dimensions.
    :return likelihoods: (K, examples)
    """

    # get feature-wise log-likelihoods (K, examples)
    log_likelihoods = dist.MultivariateNormal(
        loc = muMtx[:, None], # (K, 1)
        scale_tril = scale3D[:, None] # (K, 1)
    ).log_prob(X)

    if not log:
        log_likelihoods.exp_()
    
    return log_likelihoods

def get_posteriors(log_likelihoods):
    """
    Calculate the the posterior probabilities log p(z|x), assuming a uniform prior over
    components.
    :param likelihoods: the relative likelihood p(x|z), of each data point under each mode (K, examples)
    :return: the log posterior p(z|x) (K, examples)
    """
    posteriors = log_likelihoods - torch.logsumexp(log_likelihoods, dim=0, keepdim=True)
    return posteriors
  
# Function for initialization
# Input numCluter: a list, where each number refers to the number of 
# sub clusters for a common type (matched by index)
# ex. [1,2,2,1] --> type 1 has 1 component, type 2 has 2 components...
def initParam(data0, numCluster, C, K, P, cls, n_gmm_init, init_seed):
    data0_np = data0.cpu().numpy()
    cls_np = cls.cpu().numpy()
    conMtx_temp = torch.zeros(C-1, K, dtype = torch.float32).to(device)
    mu_init = torch.empty(K, P, dtype = torch.float32)
    cov_diag_init = torch.empty(K, P, dtype = torch.float32)

    ct_ind = 0
    for c in range(C-1):
        subData = data0_np[cls_np == c, :]
        numC0 = numCluster[c]
        # set up conMtx
        conMtx_temp[c, ct_ind: (ct_ind + numC0)] = 1
        # gmm for sub cluster
        if numC0 > 1:
            gmmSub_vec = []
            gmmSub_BIC = []
            for i in range(n_gmm_init):
                gmmSub = GaussianMixture(n_components=numC0, covariance_type='diag', reg_covar=1e-5, random_state=init_seed+i*10).fit(subData)
                gmmSub_vec.append(gmmSub)
                gmmSub_BIC.append(gmmSub.bic(subData))
            val, idx = min((val, idx) for (idx, val) in enumerate(gmmSub_BIC))
            gmmSub_best = gmmSub_vec[idx]
        else:
            gmmSub_best = GaussianMixture(n_components=numC0, covariance_type='diag', reg_covar=1e-5, random_state=init_seed).fit(subData)
        # set up cluster parameters
        mu_init[ct_ind: (ct_ind + numC0)] = torch.tensor(gmmSub_best.means_, dtype = torch.float32)
        cov_diag_init[ct_ind: (ct_ind + numC0)] = torch.tensor(gmmSub_best.covariances_, dtype = torch.float32)

        ct_ind += numC0
    return conMtx_temp, mu_init, cov_diag_init

def fullLogLik(data0, conMtx, muMtx, scale3D, cls, K, N):
    log_likelihoods = get_likelihoods(data0, muMtx, scale3D, log=True)
    log_posteriors = get_posteriors(log_likelihoods)
    logP, ind = torch.max(log_posteriors, 0, keepdim=True)
    l1 = torch.sum(log_likelihoods[ind, range(log_likelihoods.size()[1])])

    cls_long = cls.long()
    conMtxFull = conMtx[cls_long, :]

    tempMtx = torch.mm(conMtxFull, torch.exp(log_posteriors))
    temp = torch.trace(torch.log(tempMtx + 1e-5))
    return temp + l1

# Set up optimization algorithm for one dataset case
def optimDL1(parameters, data0, conMtx_temp, C, K, N, P, cls, learning_rate, maxIter, earlystop):
    # Defines a SGD optimizer to update the parameters
    optimizer = optim.Rprop(parameters, lr=learning_rate) 

    tril_indices = torch.tril_indices(row=P, col=P, offset=0)
    pVec, muMtx, lowtri_mtx = parameters
    logLikVec = np.zeros(maxIter)

    for i in range(maxIter):
        optimizer.zero_grad()

        # set up transformed p vector
        pVec_tran = dist.biject_to(dist.Binomial.arg_constraints['probs'])(pVec)
        # set up transformed concordance matrix
        conMtx_tran = torch.empty(C, K, dtype = torch.float32).to(device)
        conMtx_tran[0:(C-1),:] = conMtx_temp*pVec_tran
        conMtx_tran[C-1,:] = 1-pVec_tran

        # set up transformed cov3D matrix
        scale3D = torch.zeros(K, P, P, dtype = torch.float32).to(device)
        scale3D[:, tril_indices[0], tril_indices[1]] = lowtri_mtx
        scale3D[:, range(P), range(P)] = abs(scale3D[:, range(P), range(P)])
        
        # Define loss function xMtx, piVec, alphaMtx, K
        NLL = - fullLogLik(data0, conMtx_tran, muMtx, scale3D, cls, K, N)
        logLikVec[i] = -NLL

        if i % 50 == 0:
            print(i, "th iter...")
            if (i > 0) &  (abs((logLikVec[i]-logLikVec[i-50]-1e-5)/(logLikVec[i-50]+1e-5)) < earlystop):
                break
        
        NLL.backward()
        optimizer.step()
    
    return conMtx_tran, muMtx, scale3D, logLikVec, -NLL

# Final function to run algorithm for one dataset case
def SECANT_CITE(data0, numCluster, cls, learning_rate=0.01, maxIter=500, earlystop = 0.0001, n_gmm_init=5, init_seed=2020):
    torch.manual_seed(init_seed)
    
    C = len(torch.unique(cls))
    N = data0.size()[0]
    P = data0.size()[1] 
    K = sum(numCluster)

    # concordance p vector
    p_init = torch.ones(K, dtype = torch.float32) *0.5
    pVec = dist.biject_to(dist.Binomial.arg_constraints['probs']).inv(p_init)
    
    # clustering parameters
    conMtx_temp, mu_init, cov_diag_init = initParam(data0, numCluster, C, K, P, cls, n_gmm_init, init_seed)

    # initial cov3D
    tril_indices = torch.tril_indices(row=P, col=P, offset=0)
    cov_diag_init = cov_diag_init.to(device)
    temp0 = torch.eye(P, dtype=torch.float32).to(device)
    temp0 = temp0.reshape((1, P, P))
    temp1 = temp0.repeat(K, 1, 1)
    cov_diag_init = cov_diag_init.view(K,P,1)
    cov_int = temp1 * cov_diag_init

    cov3D_decomp = torch.cholesky(cov_int)
    lowtri_mtx = cov3D_decomp[:, tril_indices[0], tril_indices[1]]
    
    muMtx = mu_init.clone()

    pVec = pVec.to(device)
    muMtx = muMtx.to(device)
    lowtri_mtx = lowtri_mtx.to(device)
    
    pVec.requires_grad = True
    muMtx.requires_grad = True
    lowtri_mtx.requires_grad = True

    param = [pVec, muMtx, lowtri_mtx]

    conMtxFinal, mu_out, scale3D_out, logLikVec, logLik_final = optimDL1(param, data0, conMtx_temp, C, K, N, P, cls, learning_rate, maxIter, earlystop)

    logLik_temp = get_likelihoods(data0, mu_out, scale3D_out, log=True)
    log_posteriors_final = get_posteriors(logLik_temp)
    logP, lbl = torch.max(log_posteriors_final, 0, keepdim=True)

    return lbl.view(N), conMtxFinal, mu_out, scale3D_out, log_posteriors_final, logLikVec, logLik_final

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
def SECANT_joint(data0, data1, numCluster, K, cls_np, uncertain = True, learning_rate=0.01, nIter=100, init_seed=2020):
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
