import streamlit as st
import os
import json
import time
import gc
import contextlib
import io
import re
import ast

# MLX 相关
from mlx_lm import load

# ================= 引入自定义模块 =================
try:
    from step2_build import build_knowledge_base
    from agent_flow_extra_keyword_and_search import HealthAgent
    from step3_rag import LightRAGBot
except ImportError as e:
    st.error(f"❌ 缺少必要文件: {e}")
    st.stop()

# ================= 页面配置 & CSS 美化 =================
st.set_page_config(
    page_title="Evidence-based Health Claims Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入 CSS：解决字体不统一，增加列表样式
st.markdown("""
    <style>
    /* Global font */
    html, body, [class*="css"] {
        font-family: -apple-system, "Microsoft YaHei", "Segoe UI", Roboto, sans-serif !important;
        color: #333;
    }
    /* Card */
    .report-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    /* Section header */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 15px;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 8px;
    }
    /* List */
    ul.custom-list {
        margin-top: 5px;
        padding-left: 20px;
    }
    ul.custom-list li {
        margin-bottom: 8px;
        line-height: 1.6;
        font-size: 16px;
    }
    /* Final conclusion box */
    .conclusion-box {
        background-color: #e8f4fd;
        border: 1px solid #b6e0fe;
        border-left: 6px solid #2196f3;
        border-radius: 8px;
        padding: 20px;
        font-size: 17px;
        font-weight: 500;
        color: #0d47a1;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ================= 核心配置 =================
class Config:
    MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    # ✅ 保留您的 Adapter (因为您需要它来提取关键词)
    ADAPTER_DIR = "deepseek_clear_Data/my_adapters_7b" 
    #ADAPTER_DIR = None
    
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
    KB_DIR = "my_knowledge_base"
    INDEX_FILE = os.path.join(KB_DIR, "health.index")
    META_FILE = os.path.join(KB_DIR, "health.pkl")

# ================= 资源管理 =================
def clean_memory():
    gc.collect()
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"): mx.clear_cache()
        elif hasattr(mx, "metal"): mx.metal.clear_cache()
    except: pass

@st.cache_resource(show_spinner=False)
def load_engine():
    print("🚀 [System] 正在初始化 AI 模型...")
    model, tokenizer = load(Config.MODEL_NAME, adapter_path=Config.ADAPTER_DIR)
    print("✅ AI 模型加载完毕！")
    return model, tokenizer

# ================= 核心修复：超级解析器 =================

def format_content_as_list(content):
    """
    将内容强制转换为 HTML 列表格式 (解决不分点的问题)
    """
    if not content:
        return "暂无详细内容"
    
    html = '<ul class="custom-list">'
    
    # 如果已经是列表，直接遍历
    if isinstance(content, list):
        for item in content:
            item_str = str(item).replace('"', '').strip()
            if item_str:
                html += f"<li>{item_str}</li>"
    
    # 如果是字符串，尝试按常见分隔符切分
    elif isinstance(content, str):
        # 去掉 JSON 符号
        clean_str = content.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
        # 按逗号或分号切分
        items = re.split(r'[;；,，]', clean_str)
        for item in items:
            if item.strip():
                html += f"<li>{item.strip()}</li>"
    
    html += '</ul>'
    return html

def extract_all_json(text):
    """
    全文扫描，提取所有可能的 JSON 对象 (解决漏掉最终结论的问题)
    """
    merged_data = {}
    
    # 1. 使用 finditer 查找所有 {...} 块，而不仅仅是第一个
    # 这一步非常关键，因为您的模型输出往往是散落在各处的 JSON
    matches = re.finditer(r'(\{.*?\})', text, re.DOTALL)
    
    for match in matches:
        json_str = match.group(1).replace("\n", " ")
        try:
            # 尝试解析
            data = json.loads(json_str)
        except:
            try:
                data = ast.literal_eval(json_str)
            except:
                continue # 解析失败跳过
        
        if isinstance(data, dict):
            merged_data.update(data) # 合并所有找到的数据
            
    return merged_data

def display_polished_report(raw_text):
    """
    UI render (English)
    """
    data = extract_all_json(raw_text)
    verdict = "Analysis completed"
    literature_html = ""
    final_conclusion = "Please refer to the detailed analysis above."

    # --- Verdict ---
    found_verdict = False
    for k in ["verdict", "conclusion", "核心结论", "最终结论"]:
        if k in data:
            verdict = str(data[k]).replace('"', '').replace("'", "")
            found_verdict = True
            break
    if not found_verdict:
        v_match = re.search(r'(🔴|🟡|🟢|⚪️)\s*([^\n]+)', raw_text)
        if v_match:
            verdict = v_match.group(0)

    # --- Literature sections ---
    field_map = {
        "claims": "🗣 Claims & Ingredients",
        "pros": "✅ Product Pros / Supporting Evidence",
        "evidence": "📚 Scientific Evidence / Literature Conclusions",
        "contradictions": "⚠️ Contradictions / Risk Notes",
        "studies": "📖 Related Studies",
        "ingredients": "🧪 Key Ingredients"
    }
    lit_sections = []
    for key, title in field_map.items():
        if key in data and data[key]:
            content_list = format_content_as_list(data[key])
            if "<li>" in content_list:
                section_html = f"<div><strong>{title}</strong></div>{content_list}"
                lit_sections.append(section_html)
    if lit_sections:
        literature_html = "<br>".join(lit_sections)
    else:
        clean_raw = raw_text.replace("{", "").replace("}", "").replace('"', "")
        literature_html = f"<p>{clean_raw}</p>"

    # --- Final conclusion ---
    for k in ["recommendation", "advice", "suggestion", "expert_advice", "最终结论", "final_conclusion"]:
        if k in data and data[k]:
            final_conclusion = str(data[k])
            if isinstance(data[k], list):
                final_conclusion = "；".join([str(x) for x in data[k]])
            break

    # ================= Render =================
    st.markdown("### 📝 Expert Verification Report")
    st.markdown("---")

    check_str = str(verdict) + str(raw_text)
    if any(x in check_str for x in ["虚假", "假", "Fake", "False", "🔴"]):
        st.error(f"🔴 Core Verdict: {verdict.replace('🔴', '')}")
    elif any(x in check_str for x in ["夸大", "不实", "🟡", "Exaggerated", "Unsubstantiated"]):
        st.warning(f"🟡 Core Verdict: {verdict.replace('🟡', '')}")
    elif any(x in check_str for x in ["有效", "真实", "🟢", "Valid", "True"]):
        st.success(f"🟢 Core Verdict: {verdict.replace('🟢', '')}")
    else:
        st.info(f"🔵 Core Verdict: {verdict}")

    if "<li>" in literature_html or len(lit_sections) > 0:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">2. Literature Analysis</div>', unsafe_allow_html=True)
        st.markdown(literature_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">3. Final Recommendation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="conclusion-box">{final_conclusion}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ================= Web App 主逻辑 =================
def main():
    with st.sidebar:
        st.title("🩺 Evidence-based Health Claims Checker")
        st.markdown("---")
        
        with st.status("System Status", expanded=True):
            try:
                model, tokenizer = load_engine()
                st.write("✅ Model loaded")
                st.write(f"🔧 Adapter: {os.path.basename(Config.ADAPTER_DIR)}")
            except Exception as e:
                st.error(f"Failed to load: {e}")
                st.stop()
        
        if st.button("🧹 Clear VRAM"):
            clean_memory()
            st.toast("VRAM cleared")

    st.header("🔍 Fact-check health products/claims")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_area("Enter ad content", height=100, placeholder="e.g., Taking lutein can completely cure myopia...", label_visibility="collapsed")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        start_btn = st.button("Start Verification ➤", type="primary", use_container_width=True)

    if start_btn and user_input:
        log_buffer = io.StringIO()
        log_placeholder = st.empty()
        
        with st.status("🚀 Running full analysis...", expanded=True) as status:
            try:
                with contextlib.redirect_stdout(log_buffer):
                    
                    # Step 1
                    status.update(label="🌍 [1/3] Extracting keywords & searching...", state="running")
                    print(f"🔹 Target: {user_input}\n")
                    def update_log(): log_placeholder.code(log_buffer.getvalue(), language="bash")

                    agent = HealthAgent(model=model, tokenizer=tokenizer, filename="dataset.json")
                    agent.run(user_input)
                    update_log()
                    clean_memory()

                    # Step 2
                    status.update(label="📚 [2/3] Building knowledge base...", state="running")
                    build_knowledge_base()
                    update_log()

                    # Step 3
                    status.update(label="🧠 [3/3] Generating final report...", state="running")
                    bot = LightRAGBot(model=model, tokenizer=tokenizer)
                    bot.verify(user_input)
                    update_log()

                status.update(label="✅ Analysis completed", state="complete", expanded=False)
                
                full_log = log_buffer.getvalue()
                raw_report = full_log[-4000:] 
                if "-" * 60 in full_log:
                    parts = full_log.split("-" * 60)
                    if len(parts) >= 2: 
                        raw_report = parts[-2] + parts[-1]

                display_polished_report(raw_report)

            except Exception as e:
                status.update(label="❌ Error occurred", state="error")
                st.error(str(e))
                import traceback
                st.code(traceback.format_exc())
            finally:
                log_buffer.close()
                clean_memory()

if __name__ == "__main__":
    main()