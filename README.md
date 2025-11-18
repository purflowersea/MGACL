# Multi-source Heterogeneous Data Fusion with Graph-Aware Dual Contrastive Learning for Drug–Drug Interaction Prediction


# Installation & Dependencies


MGACL is mainly tested in a Linux environment, and its dependencies are listed below.


| Package         | Version  |
|-----------------|----------|
| python          | 3.9.21   |
| rdkit           | 2024.3.6 |
| pytorch         | 2.5.0    |
| cuda            | 12.4     |
| torch-geometric | 2.0.2    |
| torch-scatter   | 2.1.2    |
| torch-sparse    | 0.6.18   |
| torchvision     | 0.20.0   |
| scikit-learn    | 1.5.1    |
| tqdm            | 4.67.1   |
| networkx        | 3.2.1    |
| matplotlib      | 3.9.4    |
| pandas          | 2.2.3    |
| numpy           | 1.26.4   |


# Run MHGCL


You can train MGACL with the following command:


```bash
python main.py
```


or


```bash
python main.py --dataset drugbank --extractor adaptive
```




