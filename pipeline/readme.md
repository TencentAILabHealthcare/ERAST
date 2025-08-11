## Requirements
1. Python 3.8+
2. Required Python packages:

```
pip install numpy torch transformers
```
3. HMMER installed (for Pfam scanning)
4. Pre-trained protein language model (downloaded from https://huggingface.co/facebook/esm2_t33_650M_UR50D)
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
## Quick Usage
### Input
Place your query sequence file in the example/ directory and name it q.fa;

Place your target sequence file in the same example/ directory and name it t.fa.
### Output
The program will automatically generate the output file at:
example/out.jsonl
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
### 2. Protein Sequence Embedding (encode.py)
Generates protein sequence embeddings using a pre-trained protein language model.
**Usage**:
```
python encode.py --res_path EMBEDDING_OUTPUT --model_path MODEL_PATH --fa_path INPUT_FASTA
```
**Parameters:**:
- --res_path: Path to output embedding file (.npy format)
- --model_path: Directory containing pre-trained protein language model
- --fa_path: Input protein sequences in FASTA format
### 3. Homology Protein Prediction (predict.py)
Sort the target proteins based on the homology relationships.
**Usage**:
```
python predict.py --fa_q QUERY_FASTA --fa_t TARGET_FASTA \
                  --emb_q QUERY_EMBEDDING --emb_t TARGET_EMBEDDING \
                  --pfam_q QUERY_PFAM --pfam_t TARGET_PFAM \
                  --out OUTPUT_FILE
```
**Parameters:**:
- --fa_q: Query protein sequences (FASTA format)
- --fa_t: Target protein sequences (FASTA format)
- --emb_q: Query protein embeddings (.npy format)
- --emb_t: Target protein embeddings (.npy format)
- --pfam_q: Query Pfam annotations (JSON format)
- --pfam_t: Target Pfam annotations (JSON format)
- --out: Path to output prediction file