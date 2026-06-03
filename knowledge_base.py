"""
知识库
支持 TXT / PDF 文件的解析、向量化、去重存储
"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


def check_md5(md5_str:str):
    """检查传入的md5字符串是否已经被处理过
       return False(md5未处理过),True(处理过,已有记录)
       """
    if not os.path.exists(config.md5_path):
        #if进入表示文件不存在，那肯定没处理过这个md5
        open(config.md5_path,'w',encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line = line.strip()     #处理字符串前后的空格和回车
            if line == md5_str:
                return True
        return False


def save_md5(md5_str:str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(config.md5_path,'a',encoding='utf-8') as f:
        f.write(md5_str + '\n')



def get_string_md5(input_str,encoding='utf-8'):
    """将传入的字符串转换为md5字符串"""

    #将字符串转换为byte字节数组
    str_bytes = input_str.encode(encoding=encoding)

    #创建md5对象
    md5_obj = hashlib.md5()        #得到md5对象
    md5_obj.update(str_bytes)      #更新内容
    md5_hex = md5_obj.hexdigest()  #得到md5十六进制字符串

    return md5_hex


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """使用 PyMuPDF 从 PDF 文件中提取文本"""
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except ImportError:
        return "[错误] 未安装 PyMuPDF，无法解析 PDF"
    except Exception as e:
        return f"[错误] PDF 解析失败: {e}"


class KnowledgeBaseService(object):

    def __init__(self):
        #如果文件不存在则创建，存在则跳过
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name = config.collection_name,     #数据库表名
            embedding_function = DashScopeEmbeddings(model = "text-embedding-v4"),      #文本嵌入模型
            persist_directory=config.persist_directory, #数据库本地存储文件夹
        )     #向量存储的实例Chroma向量库对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,       #分割后每一个文本段的最大长度
            chunk_overlap=config.chunk_overlap, #连续文本段之间的字符重叠数量
            separators=config.separators,       #自然段落划分的分隔符
            length_function=len                 #使用python的len函数来进行计算长度
        )     #文本分割器的对象

    def upload_by_file(self, file_bytes: bytes, filename: str) -> str:
        """
        根据文件类型自动处理并上传到知识库
        - .txt : 直接解码 UTF-8 文本
        - .pdf : 提取文本后上传
        """
        try:
            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(file_bytes)
                if text.startswith("[错误]"):
                    return text
            elif filename.lower().endswith(".txt"):
                text = file_bytes.decode("utf-8")
            else:
                return f"[跳过]不支持的文件类型: {filename}"

            return self.upload_by_str(text, filename)
        except Exception as e:
            return f"[失败] 文件处理异常: {e}"

    def upload_by_str(self, data, filename):
        """将传入的字符串，进行向量化，存入向量数据库"""
        try:
            # 先得到传入字符串的md5值
            md5_hex = get_string_md5(data)

            if check_md5(md5_hex):
                return "[跳过]内容已经存在知识库中"

            if len(data) > config.max_spliter_char_number:
                knowledge_chunks = self.spliter.split_text(data)
            else:
                knowledge_chunks = [data]

            metadata = {
                "source": filename,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operator": "小胡",
            }

            self.chroma.add_texts(      # 内容加载到向量库中
                knowledge_chunks,
                metadatas=[metadata for _ in knowledge_chunks]
            )
            save_md5(md5_hex)
            return "[成功]内容已经成功载入向量库"
        except Exception as e:
            return f"[失败]知识库更新异常: {e}"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰伦","testfile")
    print(r)