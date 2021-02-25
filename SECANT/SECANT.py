#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.distributions as dist
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import math
import pandas as pd
from sklearn.mixture import GaussianMixture

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
    :Denot N: # of cells; P: # of features; K: # of clusters
    :param X: design matrix (N, P)
    :param muMtx: the component means (K, P)
    :param scale3D: the component log-variances (K, (P+1)P/2)
    :param log: return value in log domain?
        Note: exponentiating can be unstable in high dimensions.
    :return (log)likelihoods: (K, N)
    """

    # utilize tensor broadcasting
    log_likelihoods = dist.MultivariateNormal(
        loc = muMtx[:, None], # (K, 1, 1)
        scale_tril = scale3D[:, None] # (K, 1, 1)
    ).log_prob(X) # (N, 1)

    if not log:
        log_likelihoods.exp_()
    
    return log_likelihoods

def get_posteriors(log_likelihoods, wgt):
    """
    Calculate the the posterior probabilities log p(z|x)
    :param likelihoods: the relative likelihood p(x|z), of each data point under each mode (K, N)
    :param wgt: the weight of each cluster (K)
    :return: the log posterior p(z|x) (K, N)
    """
    # posteriors = log_likelihoods - torch.logsumexp(log_likelihoods, dim=0, keepdim=True)
    temp = log_likelihoods + torch.clamp(wgt[:, None], min=1e-5).log()
    posteriors = temp - torch.logsumexp(temp, dim=0, keepdim=True)
    return posteriors

def initParam(data0, numCluster, C, K, P, cls, n_gmm_init, init_seed):
    data0_np = data0.cpu().numpy()
    cls_np = cls.cpu().numpy()
    conMtx_temp = torch.zeros(C-1, K, dtype = torch.float32, device = device)
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
                gmmSub = GaussianMixture(n_components=numC0, covariance_type='diag', reg_covar=1e-5, random_state=init_seed+i*100).fit(subData)
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

def fullLogLik1(data0, conMtx, wgt, muMtx, scale3D, cls, K, N):
    log_likelihoods = get_likelihoods(data0, muMtx, scale3D, log=True)
    log_posteriors = get_posteriors(log_likelihoods, wgt)
    ind = torch.argmax(log_posteriors, 0)
    l1 = torch.sum(log_likelihoods[ind, range(log_likelihoods.size()[1])])

    cls_long = cls.long()
    conMtxFull = conMtx[cls_long, :]

    tempMtx = torch.mm(conMtxFull, torch.exp(log_posteriors))
    temp = torch.trace(torch.clamp(tempMtx, min=1e-5).log())
    return temp + l1, log_posteriors.exp()

# Set up optimization algorithm for one dataset case
def optimDL1(parameters, data0, conMtx_temp, C, K, N, P, cls, learning_rate, maxIter, earlystop):
    # Defines a SGD optimizer to update the parameters          
    # optimizer = optim.Rprop(parameters, lr=learning_rate) 
    # optimizer = optim.Adam(parameters, lr=learning_rate)
    # optimizer = optim.Adamax(parameters, lr=learning_rate)
    optimizer = optim.Adadelta(parameters)
    
    tril_indices = torch.tril_indices(row=P, col=P, offset=0)
    pVec, muMtx, lowtri_mtx = parameters
    logLikVec = np.zeros(maxIter)
    wgt = torch.ones(K, dtype = torch.float32, device = device)/K

    for i in range(maxIter):
        optimizer.zero_grad()

        # set up transformed p vector and concordance matrix
        pVec_tran = dist.biject_to(dist.Binomial.arg_constraints['probs'])(pVec)
        conMtx_tran = torch.empty(C, K, dtype = torch.float32, device = device)
        conMtx_tran[0:(C-1),:] = conMtx_temp*pVec_tran
        conMtx_tran[C-1,:] = 1-pVec_tran

        # set up transformed cov3D matrix
        scale3D = torch.zeros(K, P, P, dtype = torch.float32, device = device)
        scale3D[:, tril_indices[0], tril_indices[1]] = lowtri_mtx
        scale3D[:, range(P), range(P)] = abs(scale3D[:, range(P), range(P)])
        
        # Define loss function xMtx, piVec, alphaMtx, K
        LL, z = fullLogLik1(data0, conMtx_tran, wgt, muMtx, scale3D, cls, K, N)
        NLL = -LL
        logLikVec[i] = LL
        wgt = (z.sum(1)/N).detach()

        if i % 50 == 0:
            # print(i, "th iter...")
            if (i > 0) &  (abs((logLikVec[i]-logLikVec[i-50]-1e-5)/(logLikVec[i-50]+1e-5)) < earlystop):
                NLL.backward()
                optimizer.step()
                break

        NLL.backward()
        optimizer.step()
        
    return conMtx_tran, wgt, muMtx, scale3D, logLikVec, -NLL

# Final function to run algorithm for one dataset case
def SECANT_CITE(data0, numCluster, cls, uncertain = True, learning_rate=0.01, maxIter=500, earlystop = 0.0001, n_gmm_init=5, init_seed=2020):
    torch.manual_seed(init_seed)
    
    N = data0.size()[0]
    P = data0.size()[1] 
    K = sum(numCluster)
    if uncertain:
        C = len(torch.unique(cls))
    else:
        C = len(torch.unique(cls))+1
        
    # concordance p vector
    p_init = torch.ones(K, dtype = torch.float32, device = device) *0.1
    pVec = dist.biject_to(dist.Binomial.arg_constraints['probs']).inv(p_init)
    # clustering parameters
    conMtx_temp, mu_init, cov_diag_init = initParam(data0, numCluster, C, K, P, cls, n_gmm_init, init_seed)

    # initial cov3D
    tril_indices = torch.tril_indices(row=P, col=P, offset=0)
    cov_diag_init = cov_diag_init.to(device)
    temp0 = torch.eye(P, dtype=torch.float32, device = device)
    temp0 = temp0.reshape((1, P, P))
    temp1 = temp0.repeat(K, 1, 1)
    cov_diag_init = cov_diag_init.view(K,P,1)
    cov_int = temp1 * cov_diag_init

    cov3D_decomp = torch.cholesky(cov_int)
    lowtri_mtx = cov3D_decomp[:, tril_indices[0], tril_indices[1]]
    
    muMtx = mu_init.to(device)
    
    pVec.requires_grad = True
    muMtx.requires_grad = True
    lowtri_mtx.requires_grad = True

    param = [pVec, muMtx, lowtri_mtx]

    conMtxFinal, wgt_out, mu_out, scale3D_out, logLikVec, logLik_final = optimDL1(param, data0, conMtx_temp, C, K, N, P, cls, learning_rate, maxIter, earlystop)

    logLik_temp = get_likelihoods(data0, mu_out, scale3D_out, log=True)
    log_posteriors_final = get_posteriors(logLik_temp, wgt_out)
    lbl = torch.argmax(log_posteriors_final, 0)

    pVec.detach()
    muMtx.detach()
    lowtri_mtx.detach()

    return lbl.view(N), conMtxFinal, wgt_out, mu_out, scale3D_out, log_posteriors_final, logLikVec, logLik_final

# compute logLikelihood of observed data for two datasets, one with labels and the other doesn't case
# data0: CITE-seq data (with ADT confident cell type)
# data1: scRNA-seq data (without ADT info)
def fullLogLik2(data0, data1, conMtx, wgt0, wgt1, muMtx, scale3D, cls, K, N0, N1): 
    # Part1: concordance matrix and likelihood for data0
    log_likelihoods0 = get_likelihoods(data0, muMtx, scale3D, log=True)
    log_posteriors0 = get_posteriors(log_likelihoods0, wgt0)
    ind0 = torch.argmax(log_posteriors0, 0)
    l0 = torch.sum(log_likelihoods0[ind0, range(log_likelihoods0.size()[1])])

    cls_long = cls.long()
    conMtxFull = conMtx[cls_long, :]

    tempMtx = torch.mm(conMtxFull, torch.exp(log_posteriors0))
    temp = torch.trace(torch.clamp(tempMtx, min=1e-5).log())

    # Part2: likelihood for data1
    log_likelihoods1 = get_likelihoods(data1, muMtx, scale3D, log=True)
    log_posteriors1 = get_posteriors(log_likelihoods1, wgt1)
    ind1 = torch.argmax(log_posteriors1, 0)
    l1 = torch.sum(log_likelihoods1[ind1, range(log_likelihoods1.size()[1])])
    return temp + l0 + l1, log_posteriors0.exp(), log_posteriors1.exp()

# Set up optimization algorithm for two datasets case
def optimDL2(parameters, data0, data1, conMtx_temp, C, K, P, N0, N1, cls, learning_rate, maxIter, earlystop):
    # Defines a SGD optimizer to update the parameters
    optimizer = optim.Rprop(parameters, lr=learning_rate) 

    tril_indices = torch.tril_indices(row=P, col=P, offset=0)
    pVec, muMtx, lowtri_mtx = parameters
    logLikVec = np.zeros(maxIter)
    wgt0 = torch.ones(K, dtype = torch.float32, device = device)/K
    wgt1 = torch.ones(K, dtype = torch.float32, device = device)/K

    for i in range(maxIter):
        optimizer.zero_grad()

        # set up transformed p vector
        pVec_tran = dist.biject_to(dist.Binomial.arg_constraints['probs'])(pVec)
        # set up transformed concordance matrix
        conMtx_tran = torch.empty(C, K, dtype = torch.float32, device = device)
        conMtx_tran[0:(C-1),:] = conMtx_temp*pVec_tran
        conMtx_tran[C-1,:] = 1-pVec_tran

        # set up transformed cov3D matrix
        scale3D = torch.zeros(K, P, P, dtype = torch.float32, device = device)
        scale3D[:, tril_indices[0], tril_indices[1]] = lowtri_mtx
        scale3D[:, range(P), range(P)] = abs(scale3D[:, range(P), range(P)])
        
        # Define loss function xMtx, piVec, alphaMtx, K
        LL, z0, z1 = fullLogLik2(data0, data1, conMtx_tran, wgt0, wgt1, muMtx, scale3D, cls, K, N0, N1)
        NLL = -LL
        logLikVec[i] = LL
        wgt0 = (z0.sum(1)/N0).detach()
        wgt1 = (z1.sum(1)/N1).detach()

        if i % 50 == 0:
            if (i > 0) &  (abs((logLikVec[i]-logLikVec[i-50]-1e-5)/(logLikVec[i-50]+1e-5)) < earlystop):
                NLL.backward()
                optimizer.step()
                break
        
        NLL.backward()
        optimizer.step()
    
    return conMtx_tran, wgt0, wgt1, muMtx, scale3D, logLikVec, -NLL

# Final function to run algorithm for jointly analyzing CITE-seq and scRNA-seq data
def SECANT_JOINT(data0, data1, numCluster, cls, uncertain = True, learning_rate=0.01, maxIter=500, earlystop = 0.0001, n_gmm_init=5, init_seed=2020):
    torch.manual_seed(init_seed)
    
    N0 = data0.size()[0]
    N1 = data1.size()[0]
    P = data0.size()[1] 
    K = sum(numCluster)
    if uncertain:
        C = len(torch.unique(cls))
    else:
        C = len(torch.unique(cls))+1

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
    
    conMtxFinal, wgt0_out, wgt1_out, mu_out, scale3D_out, logLikVec, logLik_final = optimDL2(param, data0, data1, conMtx_temp, C, K, P, N0, N1, cls, learning_rate, maxIter, earlystop)

    logLik_temp0 = get_likelihoods(data0, mu_out, scale3D_out, log=True)
    log_posteriors_final0 = get_posteriors(logLik_temp0, wgt0_out)
    lbl0 = torch.argmax(log_posteriors_final0, 0)

    logLik_temp1 = get_likelihoods(data1, mu_out, scale3D_out, log=True)
    log_posteriors_final1 = get_posteriors(logLik_temp1, wgt1_out)
    lbl1 = torch.argmax(log_posteriors_final1, 0)

    preditADT_post_mtx = torch.mm(conMtxFinal, torch.exp(log_posteriors_final1))
    top2 = torch.topk(preditADT_post_mtx, 2, dim=0)

    preditADT_post = torch.mm(conMtxFinal, torch.exp(log_posteriors_final1))
    lbl_predict = torch.argmax(preditADT_post, 0)
    
    return lbl0.view(N0), lbl1.view(N1), lbl_predict.view(N1), conMtxFinal, wgt0_out, wgt1_out, mu_out, scale3D_out, log_posteriors_final0, log_posteriors_final1, preditADT_post, logLikVec, logLik_final
