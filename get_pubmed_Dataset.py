import json
import time
from pymed import PubMed

# --- 配置部分 ---
TOOL_NAME = "HealthVerifierAgent"
EMAIL = "mc56508@um.edu.mo"  # ⚠️ 请替换为你的真实邮箱，否则可能无法连接
MAX_RESULTS_PER_TERM = 100         # 每组关键词爬取的篇数 (测试建议20，正式可改 50-100)

# --- 你指定的关键词列表 ---
SEARCH_QUERIES = [
    "lutein AND visual fatigue AND supplementation",
    "lutein AND macular pigment AND supplementation",
    "zeaxanthin AND macular pigment AND supplementation",
    "lutein AND dry eye AND supplementation",
    "astaxanthin AND visual fatigue AND supplementation",
    "astaxanthin AND eye strain AND supplementation",
    "bilberry AND eye strain AND supplementation",
    "anthocyanins AND visual fatigue AND supplementation",
    "anthocyanins AND macular pigment AND supplementation",
    "omega-3 AND dry eye AND supplementation"
]

def fetch_and_save_json():
    # 初始化 PubMed API
    pubmed = PubMed(tool=TOOL_NAME, email=EMAIL)
    all_articles = []

    print(f"🚀 开始爬取 {len(SEARCH_QUERIES)} 组关键词...\n")

    for query_str in SEARCH_QUERIES:
        print(f"🔍 正在搜索: [{query_str}] ...")
        
        # 为了保证证据质量，我们在你的关键词后追加了 RCT 或 Meta分析 的过滤条件
        # 如果你想要所有类型的文章，可以删除下面这行里的 " AND (...)" 部分
        final_query = f'({query_str}) AND (Randomized Controlled Trial[pt] OR Meta-Analysis[pt])'
        
        try:
            # 执行查询
            results = pubmed.query(final_query, max_results=MAX_RESULTS_PER_TERM)
            
            count = 0
            for article in results:
                # 提取数据
                article_id = article.pubmed_id.split('\n')[0] if article.pubmed_id else "N/A"
                
                # 简单清洗
                abstract_text = article.abstract if article.abstract else ""
                if not abstract_text:
                    continue # 如果没有摘要，跳过
                
                article_data = {
                    "id": article_id,
                    "title": article.title,
                    "abstract": abstract_text,
                    "pub_date": str(article.publication_date),
                    "doi": article.doi if article.doi else "",
                    "search_query": query_str,  # 记录这是哪个关键词搜出来的
                    "source": "PubMed"
                }
                
                all_articles.append(article_data)
                count += 1
            
            print(f"   ✅ 获取到 {count} 篇文献")
            
            # 礼貌性延时，防止请求过快被封 IP
            time.sleep(1) 
            
        except Exception as e:
            print(f"   ❌ 出错: {e}")

    # --- 保存结果为 JSON ---
    output_file = "eye_health_evidence.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        # indent=4 保证格式美观，ensure_ascii=False 保证中文正常显示
        json.dump(all_articles, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 全部完成！")
    print(f"共收集文献: {len(all_articles)} 篇")
    print(f"文件已保存为: {output_file}")

if __name__ == "__main__":
    fetch_and_save_json()