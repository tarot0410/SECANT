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

## Example (using simulated data)

First, import the pacakge:

    from SECANT import *

Read in the datasets:
    
    ### Input data for SECANT
    # Input 1 (RNA data from CITE-seq after dimension reduction, cell*feature)
    data0 = pd.read_csv("./simulated_data/data0.csv",header=None)
    
    # Input 2 (ADT cell type label, integer from 0 to C, where C refers to uncertain cell type if it appears)
    cls_np_0 = pd.read_csv("./simulated_data/cls_np_0.csv",header=None,squeeze=True)
    
    # Optional input (aditional RNA data from scRNA-seq, similar to Input 1, for joint analysis)
    data1 = pd.read_csv("./simulated_data/data1.csv",header=None)
    
    ### Simulated truth data used to assess performance (not used as input)
    # true cell cluster label for RNA data from CITE-seq
    clusterLbl_np_0 = pd.read_csv("./simulated_data/clusterLbl_np_0.csv",header=None,squeeze=True)
    
    # true ADT cell type label for the additional RNA data (optional, for joint analysis)
    cls_np_1 = pd.read_csv("./simulated_data/cls_np_1.csv",header=None,squeeze=True)
    
    # true cell cluster label for the additional RNA data (optional, for joint analysis)
    clusterLbl_np_1 = pd.read_csv("./simulated_data/clusterLbl_np_1.csv",header=None,squeeze=True)

Here, for input data, data0 can be viewed as the RNA data from CITE-seq, data1 can be viewed as the optional RNA data from scRNA-seq, and cls_np_0 can be viewed as the confident cell types label from ADT data. For datasets used to assess method performance, cls_np_1 is the simulated confident cell types for scRNA-seq data, clusterLbl_np_0 is the true cluster labels for data from CITE-seq, and clusterLbl_np_1 is the true cluster labels for data from scRNA-seq.

Next, convert the datasets format for SECANT:

    data0 = df_to_tensor(data0)
    data1 = df_to_tensor(data1)
    cls_np_0 = cls_np_0.to_numpy()
    cls_np_1 = cls_np_1.to_numpy()
    clusterLbl_np_0 = clusterLbl_np_0.to_numpy()
    clusterLbl_np_1 = clusterLbl_np_1.to_numpy()

Check cross table for correspondance of simulated ADT cell type and RNA cell cluster
    
    pd.crosstab(cls_np_0, clusterLbl_np_0, rownames=["true ADT label"], colnames=["true cluster label"])

![plot0](https://user-images.githubusercontent.com/50209236/96535220-5314d780-125f-11eb-80aa-24458fcea842.png)

In this simulated data, there are 3 confident ADT cell types (label 0-3), and ADT label 4 refers to uncertain cell type (proportion set to be 20%). Further, there is 1 cluster for confident cell type 0, 2 clusters for type 1 and 2, and 3 clusters for type 3. The total number of clusters is 8. The cluster proportions are set to be (0.1, 0.1, 0.2, 0.2, 0.2, 0.05, 0.05, 0.1). Cluster-specific parameters are estimated from real data.

Construct UMAP plot for visualization

    reducer = umap.UMAP(random_state=42)
    embedding0 = reducer.fit_transform(data0.cpu())
    print(embedding0.shape)

Visualize simulated data using UMAP plot (colored by simulated ADT label)

    scatter0 = plt.scatter(embedding0[:, 0],
            embedding0[:, 1],
            c=cls_np_0, s=1, cmap='Spectral')
    plt.title('', fontsize=15)
    mylabel=('Type 1', 'Type 2', 'Type 3','Type 4','Uncertain')
    legend0 = plt.legend(handles=scatter0.legend_elements()[0],labels=mylabel,loc="upper right", title="Cell Type (ADT)",bbox_to_anchor=(1.35, 1))

![plot1](https://user-images.githubusercontent.com/50209236/96532447-efd47680-1259-11eb-8518-c71c5c9d6758.png)

Visualize simulated data using UMAP plot (colored by simulated cluster label)

    scatter1 = plt.scatter(embedding0[:, 0],
    	    embedding0[:, 1],
            c= clusterLbl_np_0, s=0.2, cmap='Spectral')
    plt.title('', fontsize=15)
    mylabel=('1', '2', '3', '4', '5' ,'6', '7', '8')
    legend1 = plt.legend(handles=scatter1.legend_elements()[0],labels=mylabel,loc="upper right", title="Clusters",bbox_to_anchor=(1.35, 1))

![plot2](https://user-images.githubusercontent.com/50209236/96532552-1db9bb00-125a-11eb-955c-52d0c376ba76.png)

Run SECANT (for analyzing CITE-seq data only):

First specify the number of clusters for each confident cell types:

    numCluster = [1,2,2,3] 
    K = sum(numCluster)
    
Here, 1 in numCluster stands for 1 cluster for confident cell type 0, 2 for 2 clusters for type 1 and 2, and 3 for 3 clusters for type 3. Therefore, the total number of clusters is 8.

Now run SECANT
    
    device = get_device() # use GPU if available
    outLbl, conMtxFinal0, tauVecFinal0, muMtxFinal0, cov3DFinal0, loglikFinal0 = SECANT_CITE(data0, numCluster, K, cls_np_0, uncertain = True, learning_rate=0.01, nIter=100, init_seed = 1)

Compute ARI and AMI for clustering performance:

    ARI_score= adjusted_rand_score(outLbl, clusterLbl_np_0)
    AMI_score= adjusted_mutual_info_score(outLbl, clusterLbl_np_0)

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
    
    print("ARI =", ARI_score)
    ARI = 0.9492018730675573
    
    print("AMI =", AMI_score)
    AMI = 0.9457362239875382

Visualize SECANT clustering results using UMAP plot (colored by cluster label from SECANT)

    scatter2 = plt.scatter(embedding0[:, 0],
            embedding0[:, 1],
            c=outLbl, s=0.2, cmap='Spectral')
    plt.title('', fontsize=15)
    mylabel=('1', '2', '3', '4', '5' ,'6', '7', '8')
    legend2 = plt.legend(handles=scatter2.legend_elements()[0],labels=mylabel,loc="upper right", title="Clusters",bbox_to_anchor=(1.17, 1.02))

![plot3](https://user-images.githubusercontent.com/50209236/96535222-55773180-125f-11eb-83b8-3421621abfa7.png)

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


