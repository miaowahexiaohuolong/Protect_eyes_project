import json
import time
from pymed import PubMed

# ================= 配置 =================
TOOL_NAME = "HealthVerifierAgent"
EMAIL = "mc56508@um.edu.mo"  # 你的邮箱
OUTPUT_FILE = "raw_data.json" # 爬取结果保存的文件名
MAX_RESULTS = 500              # 为了测试，每个词先爬 10 篇
# =======================================

def crawl_pubmed(keywords):
    print(f"🕷️ 正在初始化爬虫 (Keywords: {keywords})...")
    pubmed = PubMed(tool=TOOL_NAME, email=EMAIL)
    
    all_articles = []
    
    for kw in keywords:
        print(f"   -> 正在搜索关键词: '{kw}' ...")
        try:
            # 执行查询
            results = pubmed.query(kw, max_results=MAX_RESULTS)
            
            count = 0
            for article in results:
                # 提取我们需要的信息
                article_id = article.pubmed_id.split()[0] if article.pubmed_id else str(time.time())
                title = article.title if article.title else "No Title"
                abstract = article.abstract if article.abstract else ""
                
                # 如果没有摘要，这篇文献对 RAG 没用，跳过
                if not abstract: 
                    continue

                all_articles.append({
                    "id": article_id,
                    "title": title,
                    "abstract": abstract,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
                })
                count += 1
            print(f"      成功抓取 {count} 篇。")
            
        except Exception as e:
            print(f"      出错: {e}")
    
    return all_articles

def save_to_json(data):
    if not data:
        print("❌ 没有数据被保存。")
        return

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ 爬取完成！")
    print(f"📦 共收集 {len(data)} 篇文献。")
    print(f"💾 数据已保存至: {OUTPUT_FILE} (你可以打开这个文件检查内容)")

if __name__ == "__main__":
    # 这里你可以手动输入想爬的词，或者写死在代码里
    user_input = input("请输入要爬取的关键词 (用逗号分隔，例如 'Myopia, Eye fatigue'): ")
    
    if not user_input.strip():
        keywords = ["Myopia control", "Blue light eye damage"] # 默认测试词
    else:
        keywords = [k.strip() for k in user_input.split(',')]
        
    data = crawl_pubmed(keywords)
    save_to_json(data)