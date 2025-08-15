# ERAST: Efficient Retrieval-Augmented Search Tool  
### This repository contains code for training ERAST and running its localized version.
---
**ERAST** is a high-performance framework for large-scale **homology searches** across ~**1 billion** biological sequences — hosted in the largest vector database to date.  
By integrating **state-of-the-art large language models** with **vector database technology**, ERAST delivers **both speed and precision** in detecting homologous biological sequences.  

## Key Features
- **Multi-stage optimization** — Pre-retrieval, retrieval, and post-retrieval stages to boost accuracy.  
- **Broad applicability** — Supports both **nucleotide** and **protein** sequences, unlike most protein-only tools.  
- **Scalable performance** — Efficiently handles billion-scale datasets without compromising search quality.  

---

## Repository Structure

### 1. Training Pipeline  
Train your own ERAST models:  
[`erast_training`](https://github.com/TencentAILabHealthcare/ERAST/tree/main/erast_training)  

### 2. Localized Version  
Run ERAST locally for custom datasets:  
[`pipeline`](https://github.com/TencentAILabHealthcare/ERAST/tree/main/pipeline)  

---

## Environment Setup

**Requirements:**  
- Python 3.8+  
- [HMMER](http://hmmer.org/) (for Pfam scanning)  
- Python packages:  

```
pip install numpy torch datasets transformers
```

---

## Access the ERAST Vector Database

The public ERAST vector database can be accessed here:  
[https://ai4s.tencent.com/erast](https://ai4s.tencent.com/erast)



