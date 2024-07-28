# MicroHDF : predicting host phenotypes with metagenomic data using a deep forest-based framework

Input and core components:   __MicroHDF is composed of two modules: generating feature matrix and deep forest (DF) module.__ (1) MicroHDF uses two sequential templates based on posterior traversal and horizontal traversal of the tree structure to construct two phylogenetic tree-based feature matrices.(2) The microbial abundance profile and phylogenetic tree-based feature matrix are the inputs of the modified DF module.

---------------------------------------------------------------------------





<img src="./Figure 1.png" width = "800" height = "450" > 

Workflow: Metagenomic abundance data embedded into the phylogenetic tree is transformed to new feature matrixes by two different tree traversals in the feature matrix generation module.  Next, different feature representation are learned from the cascade layers with different deep forest-based units.  Finally, the learned features are aggregated to output layer and set to perform classification tasks.

---------------------------------------------------------------------------


 

### Dependencies:</BR>



* tensorflow = 1.14.0
>
* python = 3.6
>
* scikit-learn = 0.20.0
>
* pandas = 1.0.1
>
* joblib = 1.0.1
>
* deep-forest = 0.1.5

## Install

- yon can install MicroHDF via [Anaconda](https://anaconda.org/) using the commands below:<BR/>

`git clone https://github.com/liaoherui/MicroHDF.git`<BR/>
`cd MicroHDF`<BR/>

`conda env create -f environment.yaml`<BR/>
`conda activate MicroHDF`<BR/>

`python multi_channel_MicroHDF.py`<BR/>



## Data
#### Instruction about input data.<BR/>

To use MicroHDF, you need to provide microbial species abundance matrix (csv format), raw data set, and phylogenetic tree matrix as input. Here is an example.

-  microbial species abundance matrix<br/>

| sampleID          |s__Methanobrevibacter_smithii  | ...     |
|--------------|------------|------------|
| HD-10 |0.0  |...   |
| HD-11|0.0   |...   |
| HD-12  |0.0   |...   |

- raw data set<br/>

| sampleID     |disease  | ...     |
|--------------|------------|------------|
| HD-10 |0 |...   |
| HD-11|1  |...   |
| HD-12  |1   |...   |

- phylogenetic tree matrix<br/>

|      |95818  | ...     |
|--------------|------------|------------|
|1|0.0 |...   |
| 2|0.0  |...   |
| 3  |0.0   |...   |



#### The processing of data



To use MicroHDF, We provide the modules __ConcatenatTestLoad__ and __feature_selection_test__, where ConcatenatTestLoad is used to read the raw data and feature_selection_test will provide the index for constructing a new feature matrix based on the tree structure information.

 ```
MicroHDF
|-MicroHDF
    |-utils
```


## The code includes MicroHDF and benchmarks 
- Benchmark includes ```Random Forest, SVM, LASSO, MLPNN, CNN1D```，```gcForest``` models.

 ```
MicroHDF
|-Benchmarks
    |-src
        |-train_baseline.py
```
If you want to run benchmarks, look at the following path: train_baselines.py provides a set of processes for running code



#### Run MicroDF

- If only abundance information is used, use

```
main_MicroHDF.py
```

- Suppose that we want to  Reproducing the experiments described in our paper

```
python single_channel_MicroHDF.py
python multi_channel_MicroHDF.py
```
single_channel_MicroHDF is based on a path to replicate our experiment



## - Contact -

If you have any questions, please directed to the corresponding authors : 471745950@qq.com 

