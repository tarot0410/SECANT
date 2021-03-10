# SECANT (Beta)

SECANT is a biology-guided SEmi-supervised method for Clustering, classification, and ANnoTation of single-cell multi-omics. 

SECANT can be used to analyze CITE-seq data, or jointly analyze CITE-seq and scRNA-seq data. The novelties of SECANT include: 
- 1) using confident cell type labels classified from surface protein data through gating as guidance for cell clustering with RNA data
- 2) providing general annotation of confident cell types for each cell cluster 
- 3) fully utilizing cells with uncertain or missing cell type labels to increase performance
- 4) accurate prediction of confident cell types identified from surface protein data for scRNA-seq data

![workflow](https://user-images.githubusercontent.com/50209236/110571354-757f9280-8125-11eb-9cb8-93c330020c6d.png)

In general, the input of SECANT include:
- 1) ADT confident cell type labels L, where L ranges from 0 to C. Each unique value refers to one confident cell type, such as B cells, Monocytes. The maximum value C indicates uncertain cell type (e.g., cells on the boundary of different cell types in a gating plot)
- 2) RNA data after dimension reduction (e.g., scVI or PCA)
- 3) Optional (for the purpose of jointly analyzing CITE-seq and scRNA-seq data): RNA data after dimension reduction and batch effect correction

Paper: **Wang X**, Xu Z, Zhou X, Zhang Y, Huang H, Ding Y, Duerr RH, Chen W. SECANT: a biology-guided semi-supervised method for clustering, classification, and annotation of single-cell multi-omics. [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.11.06.371849v1). 2020 Jan 1.

# Get Started

### Analyzing CITE-seq data

Here, we demonstrate this functionality with public human PBMC data, bone marrow data and upper lobe lung data. The same pipeline would generally be used to analyze any CITE-seq dataset. 
- PBMC10k: [SECANT_GitHub_10X10k_PBMC.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_10X10k_PBMC.ipynb)	<a href="https://colab.research.google.com/drive/10FN1b_og_Sb3InUgrtjpwOl7YBLsPk7t?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>
	
- Bone marrow: [SECANT_GitHub_Bone_marrow.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_Bone_marrow.ipynb)<a href="https://colab.research.google.com/drive/1azjJhj6DkE0SIJ65sNK8F8MuDxdaw0RD?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>

- Lung: [SECANT_GitHub_Upper_lobe_lung.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_Upper_lobe_lung.ipynb)<a href="https://colab.research.google.com/drive/1wHucmHyWqgGzH22aGPA2-S1ElfVrMOlD?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>

### Jointly analyzing CITE-seq and scRNA-seq data
Here we demonstrate how to jointly analyze CITE-seq and scRNA-seq datasets with SECANT using two public PBMC CITE-seq datasets from 10x Genomics, namely 10X10k and 10X5k. We use the entire 10X10k dataset (i.e., both ADT and RNA) while we hold-out the ADT data of the 10X5k dataset to mimic scRNA-seq. We will store the original values to validate our results.

[SECANT_GitHub_Joint_10X.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_Joint_10X.ipynb)<a href="https://colab.research.google.com/drive/1J8pZUVEApu7shqzFPweCchCvZt8tHR52?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>

### Search for the best configuration of concordance matrix in a data-driven manner
Due to computational burden, we suggest running this step in parallel on a server with multiple CPUs or GPUs. Here is an example [SECANT_GitHub_Search_Best_Config.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_Search_Best_Config.ipynb) <a href="https://colab.research.google.com/drive/1NpVeDP6GP7HYCleLPTnsE-4Qi4QVFfVh?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>

### Simulation study
We provide an example of simulation study, including both how to generate simualted data and assessing performance. For computational burden, we recommend runnining simulation on a server with multiple CPUs or GPUs. To replicate result using Google Colab, one needs to copy all files under [simulation_files](https://github.com/tarot0410/SECANT/tree/main/simulation_files) to Google Drive, and mount Google Colab with Google Drive.
[SECANT_GitHub_simulation.ipynb](https://github.com/tarot0410/SECANT/blob/main/example/SECANT_GitHub_simulation.ipynb)<a href="https://colab.research.google.com/drive/1elVhNgFm5WCy_2cYs1mIpXxRvEgr0S9t?usp=sharing">
  	<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	</a>

# Datasets

A collection of datasets are available with SECANT. All datasets stored in this repository are pre-processed by [scVI](https://docs.scvi-tools.org/en/stable/index.html). 

### Public data:
<table>
    <tr>
        <th>Dataset</th>
        <th>Number of cells</th>
        <th>Dataset</th>
        <th>Original data source</th>
    </tr>
    <tr>
        <td>10X10k_PBMC</td>
        <td>7,865</td>
        <td>
        Human PBMCs (from 10X Genomics) 
        </td>
        <td><a href="https://support.10xgenomics.com/single-cell-gene-expression/datasets/3.0.0/pbmc_10k_protein_v3">source</a>
    </tr>
    <tr>
        <td>10X5k_PBMC</td>
        <td>5,527</td>
        <td>
        Human PBMCs (from 10X Genomics)
        </td>
        <td><a href="https://support.10xgenomics.com/single-cell-gene-expression/datasets/3.0.2/5k_pbmc_v3_nextgem">source</a>
    </tr>
    <tr>
        <td>Bone_marrow</td>
        <td>10,000</td>
        <td>
        Human bone marrow (originaly in Seurat package with >30,000 cells, downsample to 10,000 cells)
        </td>
        <td><a href="https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis.html">source</a>
    </tr>
    <tr>
        <td>Upper_lobe_lung</td>
        <td>5,451</td>
        <td>
        Human upper lobe lung (on GEO, use DropletUtils for pre-processing)
        </td>
        <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3909673">source</a>
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

Torch, sklearn, umap, pandas, numpy and all of their respective dependencies. 
