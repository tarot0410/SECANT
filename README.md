# SECANT (Beta, a more detailed version along with paper will come soon)

SECANT is a biology-guided SEmi-supervised method for Clustering, classification, and ANnoTation of single-cell multi-omics. 

SECANT can be used to analyze CITE-seq data, or jointly analyze CITE-seq and scRNA-seq data. The novelties of SECANT include 
- 1) using confident cell type labels identified from surface protein data as guidance for cell clustering
- 2) providing general annotation of confident cell types for each cell cluster 
- 3) fully utilizing cells with uncertain or missing cell type labels to increase performance
- 4) accurate prediction of confident cell types identified from surface protein data for scRNA-seq data


Paper: will be released soon

## Installation:

### From source

Download a local copy of SECANT and install from the directory:

	git clone https://github.com/tarot0410/SECANT.git
	cd SECANT
	pip install .

### Dependencies

Torch, sklearn, umap and all of their respective dependencies. 

## Example 

First, import the pacakge:

    from SECANT import *

Read in the datasets:
    
    # Input 1
    data0 = pd.read_csv("./simulated_data/data0.csv",header=None)
    # Input 2
    cls_np_0 = pd.read_csv("./simulated_data/cls_np_0.csv",header=None,squeeze=True)
    # Optional input (for joint analysis)
    data1 = pd.read_csv("./simulated_data/data1.csv",header=None)
    
    # Simulated truth data used to assess performance (not used as input)
    cls_np_1 = pd.read_csv("./simulated_data/cls_np_1.csv",header=None,squeeze=True)
    clusterLbl_np_0 = pd.read_csv("./simulated_data/clusterLbl_np_0.csv",header=None,squeeze=True)
    clusterLbl_np_1 = pd.read_csv("./simulated_data/clusterLbl_np_1.csv",header=None,squeeze=True)

Here, for input data, data0 can be viewed as the RNA data from CITE-seq, data1 can be viewed as the optional RNA data from scRNA-seq, and cls_np_0 can be viewed as the confident cell types label from ADT data. For datasets used to assess method performance, cls_np_1 is the simulated confident cell types for scRNA-seq data, clusterLbl_np_0 is the true cluster labels for data from CITE-seq, and clusterLbl_np_1 is the true cluster labels for data from scRNA-seq.

Next, convert the datasets format for SECANT:

    data0 = df_to_tensor(data0)
    data1 = df_to_tensor(data1)
    cls_np_0 = cls_np_0.to_numpy()
    cls_np_1 = cls_np_1.to_numpy()
    clusterLbl_np_0 = clusterLbl_np_0.to_numpy()
    clusterLbl_np_1 = clusterLbl_np_1.to_numpy()
 
Specify the number of clusters for each confident cell types:

    numCluster = [1,2,2,3] 
    K = sum(numCluster)
    
Here, 1 in numCluster stands for 1 cluster for confident cell type 1, 2 for 2 clusters for type 2 and 3, and 3 for 3 clusters for type 4.
The total number of clusters therefore is 8.
    
Run SECANT (for analyzing CITE-seq data only):

    device = get_device() # use GPU if available
    outLbl00, conMtxFinal0, tauVecFinal0, muMtxFinal0, cov3DFinal0, loglikFinal0 = SECANT_CITE(data0, numCluster, K, cls_np_0, uncertain = True, learning_rate=0.01, nIter=100, init_seed = 1)

Get the ARI or AMI:

    new0_ARI= adjusted_rand_score(outLbl00, clusterLbl_np_0)
    new0_AMI= adjusted_mutual_info_score(outLbl00, clusterLbl_np_0)

Print the results:

    print("loglik  =", loglikFinal0.cpu().data.numpy()) 
    loglik  = -18916.006562894596
    
    print("conMtx:")
    print(np.around(conMtxFinal0.cpu().data.numpy(),3))
    conMtx:
    [[0.82  0.    0.    0.    0.    0.    0.    0.   ]
    [0.    0.809 0.79  0.    0.    0.    0.    0.   ]
    [0.    0.    0.    0.776 0.764 0.    0.    0.   ]
    [0.    0.    0.    0.    0.    0.772 0.792 0.785]
    [0.18  0.191 0.21  0.224 0.236 0.228 0.208 0.215]]
    
    print("tauVec0:")
    print(np.around(tauVecFinal0.cpu().data.numpy(),3))
    tauVec0:
    [0.1   0.1   0.2   0.202 0.2   0.05  0.099 0.049]

    print(new0_ARI)
    0.9492018730675573
    
    print(new0_AMI)
    0.9457362239875382

Get UMAP plot (check cell type)

    reducer = umap.UMAP(random_state=42)
    embedding0 = reducer.fit_transform(data0.cpu())
    print(embedding0.shape)

Check UMAP plot (colored by ADT label)

    scatter0 = plt.scatter(embedding0[:, 0],
            embedding0[:, 1],
            c=cls_np_0, s=1, cmap='Spectral')
    plt.title('', fontsize=15)
    mylabel=('Type 1', 'Type 2', 'Type 3','Type 4','Uncertain')
    legend0 = plt.legend(handles=scatter0.legend_elements()[0],labels=mylabel,loc="upper right", title="Cell Type (ADT)",bbox_to_anchor=(1.35, 1))

![plot1](https://user-images.githubusercontent.com/50209236/96532447-efd47680-1259-11eb-8518-c71c5c9d6758.png)

Check UMAP plot (colored by clustering results from SECANT)

	scatter1 = plt.scatter(embedding0[:, 0],
            embedding0[:, 1],
            c= clusterLbl_np_0, s=0.2, cmap='Spectral')
	plt.title('', fontsize=15)
	mylabel=('1', '2', '3', '4', '5' ,'6', '7', '8')
	legend1 = plt.legend(handles=scatter1.legend_elements()[0],labels=mylabel,loc="upper right", title="Clusters",bbox_to_anchor=(1.35, 1))

## Function: SECANT_CITE

SECANT_CITE is used for surface protein-guided cell clustering and cluster annotation for CITE-seq data.

### Usage
SECANT_CITE(data0, numCluster, K, cls_np, uncertain = True, learning_rate=0.01, nIter=100, init_seed=2020)

### Arguments
* *data0* :	tensor of the RNA data from CITE-seq.
* *numCluster* :	a list of number of clusters for each confidence cell type.
* *K* : total number of clusters. 
* *cls_np* :	a numpy array of confident cell type labels built from ADT data. Each value is expected to be an integer, where the last category refers to the "uncertain" type if used.
* *uncertain* :	whether the uncertain cell type is shown in *cls_np*, the default is true.
* *learning_rate* :	the learning rate for SGD, the default is 0.01 
* *nIter* :	the max iteration for optimazation, the default is 100.
* *init_seed* :	the initial seed.


### Values
* *outLbl00* : the final cluster label
* *conMtxFinal0* : the concordance matrix
* *tauVecFinal0* : the proportion of each cluster
* *muMtxFinal0* : cluster-specific mean vector for the multivariste Guassian distribution
* *cov3DFinal0* : cluster-specific covaraince matrix for the multivariate Guassian distribution
* *loglikFinal0* : the final log-likelihood


## Function: SECANT_joint

SECANT_joint is used for surface protein-guided cell clustering, cluster annotation and confident cell type prediction for jointly analyzing CITE-seq and scRNA-seq data.

### Usage
SECANT_joint(data0, data1, numCluster, K, cls_np, uncertain = True, learning_rate=0.01, nIter=100, init_seed=2020)

### Arguments
* *data0* :	tensor of the RNA data from CITE-seq.
* *data1* :	tensor of the RNA data from scRNA-seq.
* *numCluster* :	a list of number of clusters for each confidence cell types.
* *K* : total number of clusters. 
* *cls_np* :	a numpy array of confident cell type labels built from ADT data (CITE-seq). Each value is expected to be an integer, where the last category refers to the "uncertain" type if used.
* *uncertain* :	whether the uncertain cell type is shown in *cls_np*, the default is true.
* *learning_rate* :	the learning rate for SGD, the default is 0.01 
* *nIter* :	the max iteration for optimazation, the default is 100.
* *init_seed* :	the initial seed.


### Values
* *outLbl01* : the final cluster label for CITE-seq data
* *outLbl11* : the final cluster label for scRNA-seq data
* *preditADTMtx* : the predicated ADT label for scRNA-seq data
* *conMtxFinal1* : the concordance matrix
* *tauVecFinal0_1* : the proportion of clusters in CITE-seq data
* *tauVecFinal1_1* : the proportion of clusters in scRNA-seq data
* *muMtxFinal1* : cluster-specific mean vector for the multivariste Guassian distribution
* *cov3DFinal1* : cluster-specific covaraince matrix for the multivariate Guassian distribution
* *loglikFinal1* : the final log-likelihood


