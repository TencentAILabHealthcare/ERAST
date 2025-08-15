## Requirements
1. Python 3.8+
2. Required Python packages:

```
pip install numpy torch transformers
```
3. HMMER installed (for Pfam scanning)

## Prepare library of Pfam HMMs

You will need to have a local copy of the Pfam's HMMs library. If you are using bash, you can follow these steps:

1. Download two files from the [Pfam FTP site](ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/):

	```
	wget http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz
	wget http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
	```

2. Unpack the downloaded files to the same directory.

	```
	mkdir pfamdb
	gunzip -c Pfam-A.hmm.dat.gz > pfamdb/Pfam-A.hmm.dat
	gunzip -c Pfam-A.hmm.gz > pfamdb/Pfam-A.hmm
	rm Pfam-A.hmm.gz Pfam-A.hmm.dat.gz
	```

3. Prepare Pfam database for HMMER by creating binary files.
  
	```
	hmmpress pfamdb/Pfam-A.hmm
	```
## Models
It is recommended to first download the esm2_t33_650M_UR50D model weights from [huggingface](https://huggingface.co/facebook/esm2_t33_650M_UR50D). For ERAST Homology Search Model (EHSM), the checkpoint is available on zenodo: https://zenodo.org/records/16879060.
## Quick Usage
### Input
Place your query sequence file in the example/ directory and name it q.fa;

Place your target sequence file in the same example/ directory and name it t.fa.
### Output
The program will automatically generate the output file at:
example/out.jsonl
### Homology Protein Search
```
export PFAMDB="/path/to/your/custom/pfamdb"   # Replace with your actual Pfam DB path
export MODEL_PATH="/path/to/your/custom/model" # Replace with your actual model path
cd pipeline
# Step 1: Generate Pfam annotations 
python pfam_scan.py -out example/pfam_t.json -outfmt json example/t.fa "$PFAMDB"
python pfam_scan.py -out example/pfam_q.json -outfmt json example/q.fa "$PFAMDB"
# Step 2: Generate embeddings 
python encode.py --res_path example/q_emb.npy --model_path "$MODEL_PATH" --fa_path example/q.fa
python encode.py --res_path example/q_emb.npy --model_path "$MODEL_PATH" --fa_path example/q.fa
# Step 3: Run predictions
python predict.py
```
### Homology Nucleotide Search
```
export MODEL_PATH="/path/to/your/custom/model" # Replace with your actual model path
cd pipeline
# Step 1: Generate embeddings 
python encode.py --res_path example/q_emb.npy --model_path "$MODEL_PATH" --fa_path example/q.fa --mode n
python encode.py --res_path example/q_emb.npy --model_path "$MODEL_PATH" --fa_path example/q.fa	--mode n
# Step 2: Run predictions
python predict.py --mode n
```
## Full Usage
### 1. Pfam Domain Annotation (pfam_scan.py)
Generates Pfam domain annotations for protein sequences.(sourced from https://github.com/aziele/pfam_scan)

**Usage**:
```
python pfam_scan.py -out OUTPUT_JSON -outfmt json INPUT_FASTA PFAM_DB_DIR
```
**Parameters:**:
- -out: Path to output JSON file
- -outfmt: Output format (currently only json supported)
- INPUT_FASTA: Input protein sequences in FASTA format
- PFAM_DB_DIR: Directory containing Pfam database files
### 2. Sequence Embedding (encode.py)
Generates sequence embeddings using a pre-trained language model.
**Usage**:
```
python encode.py --res_path EMBEDDING_OUTPUT --model_path MODEL_PATH --fa_path INPUT_FASTA --mode p
```
**Parameters:**:
- --res_path: Path to output embedding file (.npy format)
- --model_path: Directory containing pre-trained language model
- --fa_path: Input sequences in FASTA format
- --mode {p,n}  The type of input sequences; p represents "protein","n" represents "nucleotide". [default: p]
### 3. Homology Prediction (predict.py)
Sort the target sequences based on the homology relationships.
**Usage**:
```
python predict.py --fa_q QUERY_FASTA --fa_t TARGET_FASTA \
                  --emb_q QUERY_EMBEDDING --emb_t TARGET_EMBEDDING \
                  --pfam_q QUERY_PFAM --pfam_t TARGET_PFAM \
                  --out OUTPUT_FILE --reranker RERANKER_PATH \
				  --mode p
```
**Parameters:**:
- --fa_q: Query sequences (FASTA format)
- --fa_t: Target sequences (FASTA format)
- --emb_q: Query embeddings (.npy format)
- --emb_t: Target embeddings (.npy format)
- --pfam_q: Query Pfam annotations (JSON format)
- --pfam_t: Target Pfam annotations (JSON format)
- --out: Path to output prediction file
- --reranker (Optional) Path to the re-ranking model
- --mode {p,n}  The type of input sequences; p represents "protein","n" represents "nucleotide". [default: p]