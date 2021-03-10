# SECANT (Beta, a more detailed version along with paper will come soon)

SECANT is a biology-guided SEmi-supervised method for Clustering, classification, and ANnoTation of single-cell multi-omics. 

SECANT can be used to analyze CITE-seq data, or jointly analyze CITE-seq and scRNA-seq data. The novelties of SECANT include 
- 1) using confident cell type labels identified from surface protein data as guidance for cell clustering
- 2) providing general annotation of confident cell types for each cell cluster 
- 3) fully utilizing cells with uncertain or missing cell type labels to increase performance
- 4) accurate prediction of confident cell types identified from surface protein data for scRNA-seq data


Paper: will be released soon

# Get Started

### Analyzing CITE-seq data

Here, we demonstrate this functionality with an PBMC10k data, a bone marrow data and a lung data. The same pipeline would generally be used to analyze any CITE-seq dataset. 
- PBMC10k: [SECANT_GitHub_10X10k_PBMC.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_10X10k_PBMC.ipynb)	<a href="https://colab.research.google.com/drive/10FN1b_og_Sb3InUgrtjpwOl7YBLsPk7t?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>
	
- Bone marrow: [SECANT_GitHub_Bone_marrow.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_Bone_marrow.ipynb)<a href="https://colab.research.google.com/drive/1azjJhj6DkE0SIJ65sNK8F8MuDxdaw0RD?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>

- Lung: [SECANT_GitHub_Upper_lobe_lung.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_Upper_lobe_lung.ipynb)<a href="https://colab.research.google.com/drive/1wHucmHyWqgGzH22aGPA2-S1ElfVrMOlD?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>

### Joint analyzing CITE-seq and scRNA-seq data
Here we demonstrate how to joint analyze CITE-seq and scRNA-seq datasets with SECANT using two CITE-seq datasets of peripheral blood mononuclear cells from 10x Genomics. We use the whole 10k dataset while we hold-out the proteins of the 5k dataset. We will store the original values to validate our results.

[SECANT_GitHub_Joint_10X.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_Joint_10X.ipynb)<a href="https://colab.research.google.com/drive/1J8pZUVEApu7shqzFPweCchCvZt8tHR52?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>

### Search for best configuration
It is very important to first search for the best configuration before running SECANT. Here is an example [SECANT_GitHub_Search_Best_Config.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_Search_Best_Config.ipynb)

# Datasets

A collection of datasets are available with SECANT. All the data are pre-processing by [scVI](https://docs.scvi-tools.org/en/stable/index.html). 

### Public data:
<table>
    <tr>
        <th>Dataset</th>
        <th>Size</th>
        <th>Dataset</th>
        <th>Original Data source</th>
    </tr>
    <tr>
        <td>10X10k</td>
        <td>7xxx</td>
        <td>
        Peripheral blood mononuclear cells publicly available from 10X Genomics 
        </td>
        <td><a href="https://support.10xgenomics.com/single-cell-gene-expression/datasets/3.0.0/pbmc_10k_protein_v3">source</a>
    </tr>
    <tr>
        <td>10X5k</td>
        <td>4xxx</td>
        <td>
        Peripheral blood mononuclear cells publicly available from 10X Genomics 
        </td>
        <td><a href="https://support.10xgenomics.com/single-cell-gene-expression/datasets/3.0.2/5k_pbmc_v3_nextgem">source</a>
    </tr>
    <tr>
        <td>bone marrow</td>
        <td>xxx</td>
        <td>
        Bone marrow data
        </td>
        <td><a href="https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis.html">source</a>
    </tr>
    <tr>
        <td>lung</td>
        <td>xxxx</td>
        <td>
        lung
        </td>
        <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3909673a">source</a>
    </tr>
</table>

### In-house data:
In-house data will be available soon.


# Installation:

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


