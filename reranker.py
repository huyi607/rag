"""
重排序服务（Reranker）

作用：
  向量检索返回的结果中可能混入噪音（不相关的内容）。
  Reranker 对检索结果逐条与用户问题进行深度匹配，重新打分排序，
  把真正相关的内容排在前面，提高 LLM 生成质量。

使用方式：
  1. 先以较大的 k 值从向量库检索（如 k=10）
  2. 再通过 Reranker 精排，保留最相关的 top_k 条（如 top_k=3）

与本项目的关系：
  rag.py 中的 RagService 在检索后调用此服务，形成：
    检索(取top-k) → 重排序(精排取top_n) → 格式化 → LLM生成
"""

from typing import List, Optional
from langchain_core.documents import Document


class RerankerService:
    """
    使用 DashScope 的 text-rerank 模型对文档进行重排序。

    Args:
        model: rerank 模型名称，如 "text-rerank-v1"
        top_k: 重排序后保留的文档数量
    """

    def __init__(self, model: str = "text-rerank-v1", top_k: int = 3):
        self.model = model
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        对检索结果进行重排序

        Args:
            query: 用户原始问题
            documents: 检索到的文档列表

        Returns:
            重排序后最相关的 top_k 个文档
        """
        if not documents:
            return []

        # 如果文档数量不足，无需排序
        if len(documents) <= self.top_k:
            return documents

        try:
            from dashscope import TextReRank

            texts = [doc.page_content for doc in documents]
            response = TextReRank.call(
                model=self.model,
                query=query,
                documents=texts
            )

            results = response.output.results
            # 按相关性分数降序排列，取前 top_k 个
            sorted_results = sorted(
                results,
                key=lambda r: r.relevance_score,
                reverse=True
            )
            top_indices = [r.index for r in sorted_results[:self.top_k]]
            reranked = [documents[i] for i in top_indices]
            print(f"[Reranker] {len(documents)}条 → 精排保留{len(reranked)}条")
            return reranked

        except ImportError:
            # dashscope 版本不支持 TextReRank，降级为取前 top_k 条
            print("[Reranker] TextReRank 不可用，返回原始前 top_k 条")
            return documents[:self.top_k]
        except Exception as e:
            print(f"[Reranker] 重排序异常 ({e})，返回原始前 top_k 条")
            return documents[:self.top_k]
