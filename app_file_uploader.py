"""基于streamlit完成网页上传服务（支持 TXT / PDF）"""
import streamlit as st
from knowledge_base import KnowledgeBaseService
import time

st.title("知识库更新服务")

uploader_file = st.file_uploader(
    "请上传 TXT 或 PDF 文件",
    type=['txt', 'pdf'],
    accept_multiple_files=False,
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024

    st.subheader(f"文件名: {file_name}")
    st.write(f"格式: {file_type or file_name.split('.')[-1].upper()} | 大小: {file_size:.2f} KB")

    file_bytes = uploader_file.getvalue()

    with st.spinner("载入知识库中。。。"):
        time.sleep(1)
        # 使用 upload_by_file 自动识别文件类型（TXT/PDF）
        result = st.session_state["service"].upload_by_file(file_bytes, file_name)
        st.write(result)


