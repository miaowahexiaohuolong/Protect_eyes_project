import os
import json
import time
import sys
from pymed import PubMed

# ================= Configuration =================
# 1. AI model config
MODEL_NAME = "mlx-community/Qwen2.5-14B-Instruct-4bit"
ADAPTER_DIR = "deepseek_clear_Data/my_adapters_14b"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. PubMed config
TOOL_NAME = "HealthVerifierAgent"
EMAIL = "mc56508@um.edu.mo"
MAX_RESULTS_PER_TERM = 100
# ===========================================

class HealthAgent:

    def __init__(self,model,tokenizer,filename="dataset.json"):
        print("🤖 Initializing Agent (loading 14B model)...")
        try:
            from mlx_lm import load, generate
            self.load_fn = load
            self.generate_fn = generate
            self.filename = filename
            self.tokenizer = tokenizer
            # Load model (passed in)
            self.model = model
            print("✅ Model loaded!")
        except ImportError:
            print("❌ Error: Please install mlx-lm first")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)

    def step1_extract_keywords(self, ad_text):
        """
        Step 1: Use LLM to extract search keywords
        """
        print(f"\n🧠 [Step 1] Analyzing ad text: {ad_text[:20]}...")
        
        prompt = f"Analyze the product advertisement, extract efficacy claims, and convert them into scientific search keywords.\n\n{ad_text}"
        
        # Build prompt
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        # Generate response
        response_text = self.generate_fn(
            self.model, 
            self.tokenizer, 
            prompt=text, 
            max_tokens=512, 
            verbose=False
        )

        # Parse returned JSON if present
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON not found")
            
            clean_json_str = response_text[start_idx:end_idx]
            data = json.loads(clean_json_str)
            
            keywords = data.get("keywords", [])
            print(f"✅ Keywords extracted: {keywords}")
            return keywords
            
        except Exception as e:
            print(f"❌ Failed to parse keywords: {e}")
            print(f"Raw response: {response_text}")
            return []

    def step2_search_pubmed(self, keywords):
        """
        Step 2: Search PubMed using extracted keywords
        """
        if not keywords:
            print("⚠️ No keywords, skipping search.")
            return []

        print(f"\n🔍 [Step 2] Searching PubMed for evidence...")
        pubmed = PubMed(tool=TOOL_NAME, email=EMAIL)
        all_articles = []

        for query_str in keywords:
            print(f"   Searching: [{query_str}] ...")
            
            # Query: keyword + (RCT or Meta-analysis)
            final_query = f'({query_str}) AND (Randomized Controlled Trial[pt] OR Meta-Analysis[pt])'
            
            try:
                results = pubmed.query(final_query, max_results=MAX_RESULTS_PER_TERM)
                
                count = 0
                for article in results:
                    article_id = article.pubmed_id.split('\n')[0] if article.pubmed_id else "N/A"
                    abstract_text = article.abstract if article.abstract else ""
                    
                    if not abstract_text: continue 
                    
                    article_data = {
                        "id": article_id,
                        "title": article.title,
                        "abstract": abstract_text,
                        "pub_date": str(article.publication_date),
                        "search_term": query_str,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
                    }
                    
                    all_articles.append(article_data)
                    count += 1
                
                print(f"      -> Found {count} articles")
                time.sleep(1)
                
            except Exception as e:
                print(f"      -> Error: {e}")

        return all_articles

    def step3_save_report(self, ad_text, articles):
        """
        Step 3: Save results
        """
        print(f"\n💾 [Step 3] Saving data...")
        
        output_data = {
            "original_ad": ad_text,
            "evidence_count": len(articles),
            "evidence_list": articles
        }
        filename = self.filename
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
            
        print(f"🎉 Process completed! Report saved to: {filename}")
        print(f"Collected {len(articles)} supporting articles.")

    def run(self, ad_text):
        # 1) Extract
        keywords = self.step1_extract_keywords(ad_text)
        # 2) Search
        articles = self.step2_search_pubmed(keywords)
        # 3) Save
        self.step3_save_report(ad_text, articles)

# ================= Entry point =================
if __name__ == "__main__":
    agent = HealthAgent(filename="dataset.json")
    
    # Input an example ad
    test_ad = input("\nEnter the ad text to verify (press Enter to use a default example): \n")
    
    if not test_ad.strip():
        test_ad = "This 'VisionPro' uses quantum wave frequency technology with deep-sea squalene; just 10 minutes of wearing repairs the optic nerve via resonance, reverses 500-degree myopia, and eliminates glasses!"
        print(f"Using default ad text: {test_ad}")
    
    agent.run(test_ad)
