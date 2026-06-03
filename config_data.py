
md5_path="./md5.txt"

#Chroma
collection_name = "rag"
persist_directory = "./chromadb"

#spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n",".","!","?","。","！","？"," ",""]
max_spliter_char_number = 1000  #文本分割的阈值

#
similarity_threshold = 2        # 检索返回的匹配文档数量（旧参数，建议改用 retrieval_k）
retrieval_k = 10                # 初始检索返回的文档数量（供重排序前使用）
enable_reranker = True          # 是否启用重排序
rerank_model = "text-rerank-v1" # 重排序模型名称
rerank_top_k = 3                # 重排序后保留的文档数量

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}