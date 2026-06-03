from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
import config_data as config
from vector_stores import VectorStoreService
from reranker import RerankerService
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class RagService(object):

    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        # 初始化重排序器（由 config_data.py 中的 enable_reranker 控制）
        self.reranker = RerankerService(
            model=config.rerank_model,
            top_k=config.rerank_top_k
        ) if config.enable_reranker else None

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system", "并且我提供的用户对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(
            model=config.chat_model_name,
            streaming=True
        )

        self.chain = self.get_chain()


    def _retrieve_context(self, value: dict) -> str:
        """
        检索 → 重排序 → 格式化为文本

        替换了原来的 format_document + format_for_retriever 组合，

        流程图:
            value["input"] → 向量检索(取retrieval_k条) → Reranker(精排取top_k) → 格式化文本
        """
        try:
            query = value["input"]
            retriever = self.vector_service.get_retriever()
            docs = retriever.invoke(query)

            # 重排序（如果启用）
            if self.reranker:
                docs = self.reranker.rerank(query, docs)

            if not docs:
                return "无参考资料"

            formatted = ""
            for doc in docs:
                formatted += f"文档片段:{doc.page_content}\n文档元数据:{doc.metadata}\n\n"
            return formatted

        except Exception as e:
            print(f"[RAG] 检索失败: {e}")
            return "无参考资料（检索过程出现异常）"

    def get_chain(self):
        """获取最终的执行链"""
        def propmt_print(prompt):
            print(prompt)
            print("=" * 20)
            return prompt

        def format_for_prompt_template(value: dict):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(self._retrieve_context)
            }
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | propmt_print
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain

if __name__ == "__main__":
    #session ID配置
    session_config = {
        "configurable":{
            "session_id":"user_001",
        }
    }
    res = RagService().chain.invoke({"input":"春天穿什么颜色"},session_config)
    print(res)

