# RAG 智能客服系统

基于 **LangChain + ChromaDB + 通义千问** 的检索增强生成（RAG）问答系统。支持知识库管理、多轮对话、流式输出、以及 RESTful API。

## 技术栈

| 层       | 技术                                                             |
| -------- | ---------------------------------------------------------------- |
| 大模型   | 通义千问 qwen3-max（Chat）/ text-embedding-v4（Embedding）       |
| RAG 框架 | LangChain LCEL                                                   |
| 向量库   | ChromaDB                                                         |
| 后端 API | FastAPI                                                          |
| 前端     | Streamlit                                                        |
| 容器化   | Docker / docker-compose                                          |

## 项目结构

```
├── main.py                # FastAPI 后端服务（REST API）
├── rag.py                 # RAG 核心：检索 + 重排序 + LLM 生成
├── reranker.py            # 重排序服务（精排检索结果）
├── knowledge_base.py      # 知识库管理：文本分块、向量化、MD5 去重
├── vector_stores.py       # ChromaDB 向量库接入
├── file_history_store.py  # 对话历史管理（文件存储）
├── config_data.py         # 全局配置（模型、检索参数等）
├── app_qa.py              # Streamlit 问答页面
├── app_file_uploader.py   # Streamlit 知识库上传页面
├── requirements.txt       # Python 依赖
├── Dockerfile             # Docker 构建文件
├── docker-compose.yml     # 多服务编排（API + Streamlit）
└── .gitignore
```

## 快速开始

### 方式一：直接运行

```bash
# 1. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 设置通义千问 API Key
export DASHSCOPE_API_KEY=your_api_key_here

# 3. 启动问答页面
streamlit run app_qa.py

# 4. （可选）启动知识库上传页面
streamlit run app_file_uploader.py

# 5. （可选）启动 FastAPI 后端
python main.py
```

### 方式二：Docker 部署

```bash
# 1. 设置 API Key（创建 .env 文件或导出环境变量）
export DASHSCOPE_API_KEY=your_api_key_here

# 2. 一键启动
docker-compose up -d

# 3. 访问服务
#    Streamlit 前端: http://localhost:8501
#    FastAPI 文档:   http://localhost:8000/docs
```

## 架构流程

```
用户提问
    │
    ▼
┌─────────────────────────────────────┐
│  1. Query 传入 RAG Chain            │
│     (含对话历史)                     │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  2. 向量检索（ChromaDB）            │
│     从知识库召回相关文档片段         │
│     （配置: retrieval_k = 10）       │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  3. 重排序（Reranker）              │
│     对召回结果逐条精排              │
│     保留最相关片段                  │
│     （配置: rerank_top_k = 3）       │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  4. LLM 生成回答                    │
│     参考资料 + 对话历史 → 通义千问  │
└─────────┬───────────────────────────┘
          │
          ▼
       返回结果（支持流式）
```

## API 文档

启动 FastAPI 后访问 `http://localhost:8000/docs` 查看交互式文档。

| 方法   | 路径               | 说明               |
| ------ | ------------------ | ------------------ |
| GET    | /api/health        | 健康检查           |
| POST   | /api/chat          | 非流式问答         |
| POST   | /api/chat/stream   | 流式问答（SSE）    |
| POST   | /api/upload        | 上传文件到知识库   |

## 配置说明

所有配置集中在 `config_data.py`：

| 参数               | 默认值            | 说明                         |
| ------------------ | ----------------- | ---------------------------- |
| retrieval_k        | 10                | 初始检索返回的文档数量       |
| enable_reranker    | True              | 是否启用重排序               |
| rerank_model       | text-rerank-v1    | 重排序模型名称               |
| rerank_top_k       | 3                 | 重排序后保留的文档数量       |
| chunk_size         | 1000              | 文本分块大小                 |
| embedding_model_name | text-embedding-v4 | 文本嵌入模型                 |
| chat_model_name    | qwen3-max         | 对话模型                     |
