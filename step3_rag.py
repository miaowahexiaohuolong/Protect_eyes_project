import os
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate

# ================= 配置调整 =================
# 1. 路径 (保持不变)
KB_DIR = "my_knowledge_base"
INDEX_FILE = os.path.join(KB_DIR, "health.index")
META_FILE = os.path.join(KB_DIR, "health.pkl")

# 2. 模型 (保持不变)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"
ADAPTER_PATH = "deepseek_clear_Data/my_adapters_14b"

# 3. 检索参数 (修改)
RETRIEVAL_TOP_K = 10  # 向量取10个，关键词取10个
# ===========================================

class HybridRetriever:
    def __init__(self):
        print("\n📚 [Init] Initializing hybrid retrieval (full)...")
        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError("Knowledge base files not found")

        # 1. 加载文档
        with open(META_FILE, 'rb') as f:
            self.documents = pickle.load(f)
            
        # 2. 加载向量检索 (Faiss)
        print("   -> Loading Faiss...")
        self.index = faiss.read_index(INDEX_FILE)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        # 3. 加载关键词检索 (BM25)
        print("   -> Building BM25 index...")
        corpus_tokens = [self._tokenize(doc.get('title', '') + " " + doc.get('abstract', '')) for doc in self.documents]
        self.bm25 = BM25Okapi(corpus_tokens)
        
    def _tokenize(self, text):
        return text.lower().split()

    def search(self, query):
        """
        修改版：不进行排序，直接合并两个检索源的所有结果并去重
        """
        results = []
        seen_ids = set() # 用于去重 ID

        # --- 1. 向量检索 (Vector Search) ---
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        _, vector_indices = self.index.search(query_vec, RETRIEVAL_TOP_K)
        vector_ids = vector_indices[0] 
        
        # 收集向量结果
        for doc_id in vector_ids:
            if doc_id == -1: continue
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append({
                    "source": "Vector", # 标记来源方便调试
                    "id": doc_id,
                    "title": self.documents[doc_id].get('title', ''),
                    "abstract": self.documents[doc_id].get('abstract', '')
                })

        # --- 2. 关键词检索 (Keyword Search) ---
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        keyword_ids = np.argsort(bm25_scores)[::-1][:RETRIEVAL_TOP_K]
        
        # 收集关键词结果 (仅添加之前没见过的)
        for doc_id in keyword_ids:
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append({
                    "source": "Keyword",
                    "id": doc_id,
                    "title": self.documents[doc_id].get('title', ''),
                    "abstract": self.documents[doc_id].get('abstract', '')
                })
        
        # 此时 results 包含了 Vector Top 10 + Keyword Top 10 (去重后)
        # 总数在 10 到 20 之间
        
        # 重新为最终列表编号
        for i, res in enumerate(results):
            res['rank'] = i + 1
            
        return results

class LightRAGBot:
    def __init__(self, model, tokenizer):
        self.retriever = HybridRetriever()
        self.model = model
        self.tokenizer = tokenizer

    def verify(self, query):
        print(f"\n🔎 [Step 1] Full retrieval (Vector + Keyword)...")
        evidence = self.retriever.search(query)
        
        print(f"   -> Retrieved {len(evidence)} evidence entries (no truncation):")
        for doc in evidence:
            source_tag = "[V]" if doc.get("source") == "Vector" else "[K]"
            print(f"      {source_tag} [{doc['rank']}] {doc['title'][:50]}...")
            
        context_str = ""
        if not evidence:
            context_str = "【System Notice】: No relevant literature in the database."
        else:
            for doc in evidence:
                context_str += f"--- Evidence [{doc['rank']}] ---\nTitle: {doc['title']}\nAbstract: {doc['abstract']}\n\n"
            
        # English prompt enforcing natural language output
        prompt = f"""
## Strict Instructions
Do not output JSON, dictionaries, or Python lists. Write in clear, natural English as a professional clinician.

## Task
Using the following Evidence, verify the truthfulness of the User Claim.

User Claim: {query}
Evidence:
{context_str}

## Output format (Markdown)

### 1. Core Verdict
*(Write only one sentence, starting with 🔴Do not buy / 🟡Consider with caution / 🟢Reasonable to buy)*

### 2. Detailed Analysis
*(Use unordered list `*`. No quotes, no brackets.)*
* About the claim: [content]
* Scientific evidence: [content]
* Contradictions: [content]

### 3. Final Recommendation
*(Write a short paragraph. No JSON.)*
"""
        print(f"\n📝 [Step 2] AI generating verification report...\n")
        print("-" * 60)
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 注意：因为这里输入了最多20篇文档，Prompt会很长，生成时请确保显存足够
        generate(self.model, self.tokenizer, prompt=text, max_tokens=2048, verbose=True)
        print("-" * 60)

if __name__ == "__main__":
    # 在这里加载模型，避免每次重启
    print("⏳ Loading model...")
    model, tokenizer = load(LLM_MODEL, adapter_path=ADAPTER_PATH)
    
    bot = LightRAGBot(model, tokenizer)
    
    while True:
        q = input("\nEnter ad claim (q to quit): ").strip()
        if q.lower() == 'q': break
        if not q: continue
        bot.verify(q)