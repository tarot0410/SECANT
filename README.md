# SECANT (Beta, a more detailed version along with paper will come soon)

SECANT is a biology-guided SEmi-supervised method for Clustering, classification, and ANnoTation of single-cell multi-omics. 

SECANT can be used to analyze CITE-seq data, or jointly analyze CITE-seq and scRNA-seq data. The novelties of SECANT include 
- 1) using confident cell type labels identified from surface protein data as guidance for cell clustering
- 2) providing general annotation of confident cell types for each cell cluster 
- 3) fully utilizing cells with uncertain or missing cell type labels to increase performance
- 4) accurate prediction of confident cell types identified from surface protein data for scRNA-seq data


Paper: will be released soon
## Get Started

### Analyzing CITE-seq data

Here, we demonstrate this functionality with an PBMC10k data, a bone marrow data and a lung data. The same pipeline would generally be used to analyze any CITE-seq dataset. You can find the code in the example folder or you can run it on Google Colab:
- PBMC10k 	<a href="https://colab.research.google.com/drive/10FN1b_og_Sb3InUgrtjpwOl7YBLsPk7t?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>
	
- Bone marrow <a href="https://colab.research.google.com/drive/1azjJhj6DkE0SIJ65sNK8F8MuDxdaw0RD?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>

- Lung <a href="https://colab.research.google.com/drive/1wHucmHyWqgGzH22aGPA2-S1ElfVrMOlD?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>

### Joint analyzing CITE-seq and scRNA-seq data


## Datasets

A collection of datasets are available with SECANT.

### Public data:
<table>
    <tr>
        <th>Dataset</th>
        <th>Size</th>
        <th>Dataset</th>
        <th>Data source</th>
    </tr>
    <tr>
        <td>flchain</td>
        <td>6,524</td>
        <td>
        The Assay of Serum Free Light Chain (FLCHAIN) dataset. See 
        <a href="#references">[1]</a> for preprocessing.
        </td>
        <td><a href="https://github.com/vincentarelbundock/Rdatasets">source</a>
    </tr>
    <tr>
        <td>gbsg</td>
        <td>2,232</td>
        <td>
        The Rotterdam & German Breast Cancer Study Group.
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
    <tr>
        <td>kkbox</td>
        <td>2,814,735</td>
        <td>
        A survival dataset created from the WSDM - KKBox's Churn Prediction Challenge 2017 with administrative censoring.
        See <a href="#references">[1]</a> and <a href="#references">[15]</a> for details.
        Compared to kkbox_v1, this data set has more covariates and censoring times.
        Note: You need 
        <a href="https://github.com/Kaggle/kaggle-api#api-credentials">Kaggle credentials</a> to access the dataset.
        </td>
        <td><a href="https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data">source</a>
    </tr>
    <tr>
        <td>kkbox_v1</td>
        <td>2,646,746</td>
        <td>
        A survival dataset created from the WSDM - KKBox's Churn Prediction Challenge 2017. 
        See <a href="#references">[1]</a> for details.
        This is not the preferred version of this data set. Use kkbox instead.
        Note: You need 
        <a href="https://github.com/Kaggle/kaggle-api#api-credentials">Kaggle credentials</a> to access the dataset.
        </td>
        <td><a href="https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data">source</a>
    </tr>
    <tr>
        <td>metabric</td>
        <td>1,904</td>
        <td>
        The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC).
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
    <tr>
        <td>nwtco</td>
        <td>4,028</td>
        <td>
        Data from the National Wilm's Tumor (NWTCO).
        </td>
        <td><a href="https://github.com/vincentarelbundock/Rdatasets">source</a>
    </tr>
    <tr>
        <td>support</td>
        <td>8,873</td>
        <td>
        Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT).
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
</table>


## Installation:

### From source

Download a local copy of SECANT and install from the directory:

	git clone https://github.com/tarot0410/SECANT.git
	cd SECANT
	pip install .

### Dependencies

Torch, sklearn, umap and all of their respective dependencies. 

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


