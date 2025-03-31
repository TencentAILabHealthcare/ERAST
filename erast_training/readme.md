### ERAST Training Code

Due to the file size limit for uploading, only the code for the ERAST training model is uploaded here. 

```python
/owenbhe/buddy1/yenojiang/code/Tencent_AI/erast_training
├── DNA
│   ├── datasets
│   │   ├── dataset2fasta.py
│   │   ├── fasta2frag.py
│   │   └── preprocess.ipynb
│   ├── model
│   │   ├── configuration_caduceus.py
│   │   ├── __init__.py
│   │   ├── modeling_caduceus.py
│   │   ├── modeling_rcps.py
│   │   └── __pycache__
│   └── training
│       └── phy_trainer.py
└── Protein
    ├── datasets
    │   └── save_pairwise.py
    ├── model
    │   ├── embedding_model.py
    │   ├── __init__.py
    │   └── __pycache__
    └── training
        └── score_model_trainer.py
```

- Training ERAST’s Encoding model:
    - /DNA/datasets: describe how to construct the distantly related datasets based on genus partition strategy.
    - /DNA/model: the model’s structure
    - /DNA/training: fine-tuning the pre-trained model on NCBITaxonomic as a classification task on the phylum level
- Training ERAST’s  Homology Searching model:
    - /Protein/datasets: describe how to construct the SCOPe40withScore dataset with a mapping strategy.
    - /DNA/model: the model’s structure
    - /DNA/training: fine-tuning the score model on SCOPe40withScore as a classification task

