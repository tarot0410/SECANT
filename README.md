# SECANT

SECANT is a biology-guided SEmi-supervised method for Clustering, classification, and ANnoTation of single-cell multi-omics. 

SECANT can be used to analyze CITE-seq data, or jointly analyze CITE-seq and scRNA-seq data. The novelties of SECANT include 
- 1) using confident cell type labels identified from surface protein data as guidance for cell clustering
- 2) providing general annotation of confident cell types for each cell cluster 
- 3) fully utilizing cells with uncertain or missing cell type labels to increase performance
- 4) accurate prediction of confident cell types identified from surface protein data for scRNA-seq data
- 5) quantify the uncertainty of the results


Reference: 
- Paper: 

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

    data0 = pd.read_csv("./simulated_data/data0.csv",header=None)
    data1 = pd.read_csv("./simulated_data/data1.csv",header=None)
    cls_np_0 = pd.read_csv("./simulated_data/cls_np_0.csv",header=None,squeeze=True)
    cls_np_1 = pd.read_csv("./simulated_data/cls_np_1.csv",header=None,squeeze=True)
    clusterLbl_np_0 = pd.read_csv("./simulated_data/clusterLbl_np_0.csv",header=None,squeeze=True)
    clusterLbl_np_1 = pd.read_csv("./simulated_data/clusterLbl_np_1.csv",header=None,squeeze=True)

Here, data0 can be viewed as the RNA-seq data from CITE-seq, and data1 can be viewed as the optional single cell RNA-seq data. cls_np_0 is the xxx, and cls_np is the xxxxx. clusterLbl_np_0 is the true label xxxxxx.

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
    
Run SECANT:

    device = get_device()
    outLbl00, conMtxFinal0, tauVecFinal0, muMtxFinal0, cov3DFinal0, loglikFinal0 = SECANT_CITE(data0, numCluster, K, cls_np_0, uncertain = True, learning_rate=0.01, nIter=100, init_seed = 1)

Get the ARI or AMI:

    new0_ARI= adjusted_rand_score(outLbl00, clusterLbl_np_0)
    new0_AMI= adjusted_mutual_info_score(outLbl00, clusterLbl_np_0)

Print the results:

    print("loglik  =", loglikFinal0.cpu().data.numpy()) 
    print("conMtx:")
    print(np.around(conMtxFinal0.cpu().data.numpy(),3))
    print("tauVec0:")
    print(np.around(tauVecFinal0.cpu().data.numpy(),3))

    print("ARI:", new0_ARI)
    print("AMI:", new0_AMI)

Get UMAP plot (check cell type)

    reducer = umap.UMAP(random_state=42)
    embedding0 = reducer.fit_transform(data0.cpu())
    print(embedding0.shape)

Check UMAP plot (ADT label)

    scatter0 = plt.scatter(embedding0[:, 0],
            embedding0[:, 1],
            c=cls_np_0, s=1, cmap='Spectral')
    plt.title('', fontsize=15)
    mylabel=('Type 1', 'Type 2', 'Type 3','Type 4','Uncertain')
    legend0 = plt.legend(handles=scatter0.legend_elements()[0],labels=mylabel,loc="upper right", title="Cell Type (ADT)",bbox_to_anchor=(1.35, 1))


## Function: SECANT_CITE

### Usage
SECANT_CITE(data0, numCluster, K, cls_np, uncertain = True, learning_rate=0.01, nIter=100, init_seed=2020)

### Arguments
* *data0* :	tensor of the RNA-seq dataset from CITE-seq.
* *numCluster* :	a list of number of clusters for each confidence cell types.
* *K* : total number of cluster. 
* *cls_np* :	a numpy array of 
* *uncertain* :	wheter to add the uncertain type, the default is true.
* *learning_rate* :	the learning rate for SGD, the default is 0.01 
* *nIter* :	the max iteration for optimazation, the default is 100.
* *init_seed* :	the initial seed.


### Values
* *outLbl00* : the cluster label
* *conMtxFinal0* : the concordance matrix
* *tauVecFinal0* : the proportion of each cluster
* *muMtxFinal0* : mean vector for the multivariste Guassian distribution
* *cov3DFinal0* : covaraince matrix for the multivariate Guassian distribution
* *loglikFinal0* : the log likelihood


## Function: SECANT_joint

### Usage
SECANT_joint(data0, data1, numCluster, K, cls_np, uncertain = True, learning_rate=0.01, nIter=100, init_seed=2020)

### Arguments
* *data0* :	tensor of the RNA-seq dataset from CITE-seq.
* *data1* :	tensor of the single cell RNA-seq dataset.
* *numCluster* :	a list of number of clusters for each confidence cell types.
* *K* : total number of clusters. 
* *cls_np* :	a numpy array of 
* *uncertain* :	wheter to add the uncertain type, the default is true.
* *learning_rate* :	the learning rate for SGD, the default is 0.01 
* *nIter* :	the max iteration for optimazation, the default is 100.
* *init_seed* :	the initial seed.


### Values
* *outLbl01* : the cluster label for RNA-seq data
* *outLbl11* : the cluster label for scRNA-seq data
* *preditADTMtx* : the predicated ADT label for the scRNA-seq data
* *conMtxFinal1* : the concordance matrix
* *tauVecFinal0_1* : the proportion of cluster in RNA data from CITE-seq
* *tauVecFinal1_1* : the proportion of cluster in scRNA-seq 
* *muMtxFinal1* : mean vector for the multivariste Guassian distribution
* *cov3DFinal1* : covaraince matrix for the multivariate Guassian distribution
* *loglikFinal1* : the log likelihood


