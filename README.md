# MCHNN

The code for paper " Multi-view Contrastive Learning Hypergraph Neural Network for Drug-Microbe-Disease Association Prediction''.

![image-20230110161157410](.\figure\model.png) 

## 1.Overview

The repository is organized as follows:

- `Data/` contains the datasets used in the paper;
- `Utils/` contains the processing functions and tools (e.g., negative sampling strategy functions....);
- `main_cv_hyper_graph.py` contains the training and testing code of 5-CV  experiments on datasets;
- `main_indep_hyper_graph.py` contains the training and testing code of independent test experiments on datasets;
- `model_hyper_graph.py` contains the three body parts of the model, i.e., BioEncoder, HgnnEncoder, and Decoder.

## 2.Dependencies

- python == 3.7.2
- numpy == 1.20.2
- torch == 1.8.0
- torch-geometric == 2.0.1
- networks == 2.5.1
- sklearn == 1.0.2
- rdkit == 2020.09.1.0
- deepchem == 2.5.0

## 3.Example

Here we provide two examples of using MCHNN to run 5-CV experiments and independent test experiments, execute the following command:

`python main_cv_hyper_graph.py --hypergraph_loss_ratio 0.8 --lr 0.005`

`python main_indep_hyper_graph.py --hypergraph_loss_ration 0.8 --lr 0.005`