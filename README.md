# Agent_for_Identity_eyes_project

1. 项目介绍 (Project Overview)
项目名称： 健康产品宣传真实性识别 + 科学依据智能体 (Health Product Claim Verification Agent) 核心理念： “用数据对抗营销”。 项目简介： 该项目旨在解决市场上健康/保健产品（特别是护眼类）营销宣传夸大其词、缺乏科学依据的问题。通过构建一个AI智能体，一端连接医学权威数据库（如PubMed、Cochrane），一端连接市场营销数据（如小红书、电商文案），通过对比分析，自动鉴别产品宣传的真伪，并为用户提供基于证据等级（Evidence Level）的购买建议。

MVP（最小可行性产品）切入点： 护眼类保健品（叶黄素、花青素等）。

2. 所需数据集 (Datasets)
你需要构建两个核心数据库（对抗数据集）：

A. 科学/医学证据库 (The "Truth" Source)
来源： PubMed, Cochrane Library, Google Scholar, ClinicalTrials.gov。

数据内容：

核心成分 (Ingredients)： 如叶黄素 (Lutein)、花青素 (Anthocyanin)、虾青素 (Astaxanthin)。

研究类型 (Study Type)： 重点抓取 RCT (随机对照试验)、Meta-analysis (荟萃分析)。

功效结论 (Outcome)： 对视力、黄斑变性、眼疲劳的具体影响数据。

证据等级 (Evidence Level)： 依据循证医学标准（A级-强证据，B级-中等，C级-弱/观察性研究，D级-专家意见/无证据）。

B. 市场/营销语料库 (The "Hype" Source)
来源： 小红书 (XiaoHongShu), 淘宝/京东详情页, 抖音 (TikTok) 热门带货视频字幕。

数据内容：

宣传关键词 (Claims)： “抗蓝光”、“逆转近视”、“三天见效”、“保护黄斑”。

营销套路： 焦虑营销话术、KOL 推荐语。

产品信息： 品牌、成分表、含量。

3. 所需技术框架与栈 (Tech Stack & Frameworks)
为了实现从数据获取到智能体交互，建议使用以下技术栈：

数据获取与处理 (ETL)
爬虫框架： Scrapy 或 Selenium/Playwright (用于抓取动态网页如小红书)。

数据清洗： Pandas (Python)。

NLP/聚类： Scikit-learn (用于 TF-IDF/K-Means 聚类营销词汇)，Hugging Face Transformers (用于提取关键词)。

智能体与大模型 (Agent & LLM)
编排框架： LangChain 或 LlamaIndex (最核心框架)。用于构建 RAG (检索增强生成) 流程和 Tool 使用。

大语言模型 (LLM)： GPT-4o, Claude 3.5 Sonnet (推理能力强，适合阅读文献) 或 DeepSeek (高性价比)。

向量数据库 (Vector DB)： Pinecone, Milvus 或 ChromaDB。用于存储医学文献的 Embedding，以便Agent进行语义搜索。

交互界面 (Frontend)
快速演示： Streamlit 或 Gradio (Python 原生，开发极快)。

4. 项目详细流程 (Project Workflow)
根据图片逻辑，项目流程分为三个阶段：

阶段一：数据准备与知识库构建 (Offline)
抓取： 爬取PubMed上关于“Eye health supplements”的论文摘要；爬取电商/社媒上的热门护眼产品文案。

结构化处理：

利用 LLM 将晦涩的医学论文转化为结构化数据：{成分: "叶黄素", 功效: "AMD改善", 证据等级: "A", 来源: "PubMed ID: xxx"}。

利用 NLP 对营销文案进行聚类，提取高频宣传词（如“抗蓝光”）。

入库： 将处理后的医学知识存入向量数据库，作为 Agent 的“大脑”。

阶段二：智能体构建 (Agent Logic)
意图识别： 用户输入“这种蓝莓胶囊能护眼吗？” -> Agent 识别实体“蓝莓(花青素)”和意图“验证功效”。

RAG 检索： Agent 在向量库中搜索“花青素 (Anthocyanin)” + “护眼 (Eye protection)” 的临床证据。

逻辑推理 (CoT)：

Fact Check: 检索到的证据显示花青素对夜视力有一定辅助（C级），但不能治疗近视。

Compare: 营销文案声称“恢复视力”（夸大）。

生成回答： 按照图片3的格式生成结构化报告。

阶段三：用户交互与反馈 (Online)
用户输入产品链接或宣传语。

系统输出：证据等级评分卡、智商税预警、科学替代品推荐。

5. 项目期望达到的目标 (Desired Outcomes/Goals)
信息不对称的消除 (De-bunking)：

通过 AI 自动量化“营销宣传”与“科学事实”之间的差距（Gap Analysis）。

输出直观的对比图（如图2所述的“营销 vs 临床证据差距图”）。

科学依据分级 (Evidence Grading)：

不只是说“好”或“不好”，而是给出 A/B/C/D 等级，建立权威性。

消费决策辅助 (Actionable Advice)：

直接告诉用户：买（性价比高/证据确凿）、不买（证据不足/智商税）、或者平替（推荐叶黄素代替花青素）。

自动化流程：

实现输入一个新产品，Agent 自动跑通全流程并生成报告。
