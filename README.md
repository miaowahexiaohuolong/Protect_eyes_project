# Evidence-based Health Claims Checker

A local RAG-powered Streamlit app that verifies health product claims using PubMed evidence and a lightweight MLX-based LLM. It extracts keywords from the ad text, searches PubMed (RCTs/Meta-analyses), builds a local knowledge base, and generates an expert-style report with citations context.

> Disclaimer: This tool is for information only and does not constitute medical advice.

## Key Features
- Streamlit web UI with clean English output
- Keyword extraction via local LLM (MLX)
- PubMed search (RCT / Meta-analysis filtered)
- Hybrid retrieval (Faiss vectors + BM25 keywords)
- RAG report generator with clear verdict and recommendation

## Project Structure
- main.py — Streamlit UI and end-to-end pipeline
- agent_flow_extra_keyword_and_search.py — LLM keyword extraction + PubMed search
- step2_build.py — Build Faiss index and metadata store
- step3_rag.py — Hybrid retriever and report generation
- dataset.json — Collected evidence from PubMed (generated)
- my_knowledge_base/ — Faiss index and metadata (generated)


### / deepseek_clear_Data

1. The data from html are collected in  "html_data" dictory
2. The 'process_and_annotate.py' aims to intergate lots of html_data to be a table, and it can use deepseek model to extra the keyword and generate the trianing dataset.
3. Run the 'train.py' that you can the dictory called 'processed_data', this dictory spilt the training data and vaild data, and test.py use test data, you can get the variation of loss while runing the 'trian.py'
4. the dictory 'my_adapters_14b' can be used to put the training weigths from model

After doing these process, now our model can extra keyword more efficient.

## Requirements
- Python 3.9+ (3.10 recommended)
- macOS with Apple Silicon recommended (MLX ecosystem)
- Packages:
  - streamlit, mlx-lm, sentence-transformers, faiss-cpu, rank-bm25, pymed, numpy

Create the environment:
```bash
conda activate eyes
```

Install via pip:
```bash
pip install -r requirements.txt
```
If faiss-cpu fails on macOS, try:
```bash
conda install -c conda-forge faiss-cpu
```

## Configuration
- Model and adapter paths:
  - main.py → class Config (MODEL_NAME, ADAPTER_DIR)
  - step3_rag.py → LLM_MODEL, ADAPTER_PATH, EMBEDDING_MODEL
- Hugging Face mirror (optional):
  - HF_ENDPOINT = https://hf-mirror.com (set in code)
- PubMed tool/email:
  - agent_flow_extra_keyword_and_search.py → TOOL_NAME, EMAIL

Adjust paths for your environment (e.g., adapter directories).

## Run (Streamlit UI)
```bash
streamlit run /Users/cheche/Desktop/git/Protect_eyes_project/main.py
```
Then open the provided local URL. Enter an ad claim (e.g., “Lutein cures myopia”), click “Start Verification,” and wait for the report.

## CLI Usage (Optional)
- Run the PubMed collector (saves dataset.json):
```bash
python /Users/cheche/Desktop/git/Protect_eyes_project/agent_flow_extra_keyword_and_search.py
```
- Build knowledge base from dataset.json:
```bash
python /Users/cheche/Desktop/git/Protect_eyes_project/step2_build.py
```
- Generate a report via RAG (interactive prompt):
```bash
python /Users/cheche/Desktop/git/Protect_eyes_project/step3_rag.py
```

## Workflow
1. Extract keywords from the ad using the local LLM.
2. Search PubMed for RCTs and Meta-analyses with those keywords.
3. Build a vector index (Faiss) and BM25 corpus from abstracts.
4. Retrieve top evidence (vector + keyword, deduplicated).
5. Generate a structured English report with:
   - Core Verdict (🔴/🟡/🟢)
   - Literature Analysis (bulleted)
   - Final Recommendation (paragraph)

## Outputs
- dataset.json — Raw evidence list from PubMed
- my_knowledge_base/health.index — Faiss vector index
- my_knowledge_base/health.pkl — Article metadata

## Troubleshooting
- “Knowledge base files not found”:
  - Ensure dataset.json exists, then run step2_build.py.
- Model loading or VRAM issues:
  - Use the “Clear VRAM” button in the UI; reduce max_tokens in step3_rag.py.
- PubMed rate limiting:
  - The agent sleeps between queries; lower MAX_RESULTS_PER_TERM if needed.
- Embedding model download issues:
  - Ensure internet access or pre-cache SentenceTransformers models.
- Adapter/model path errors:
  - Verify ADAPTER_DIR/ADAPTER_PATH on disk.

## Notes
- UI and generated report are in English.
- Some internal keys may still reference Chinese labels when parsing legacy JSON, but output is normalized to English in the UI.


