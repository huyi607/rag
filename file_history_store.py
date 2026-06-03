import os,json

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, BaseMessage, messages_from_dict


def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id    #会话id
        self.storage_path = storage_path   #不同会话id存储的文件路径
        #完整文件路径
        self.file_path = os.path.join(self.storage_path,self.session_id)
        #确保文件夹存在
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)
    #添加消息函数
    def add_messages(self,messages)->None:
        #Sequence序列类似于list，tuple
        all_messages = list(self.messages)      #已有的消息列表
        all_messages.extend(messages)           #新的和已有的融合为一个列表
        #将数据同步到本地文件中
        #类对象写入文件->二进制
        #为了方便，可以将BaseMessage转为字典（借助json模块进行查看），即是message_to_dict
        # new_messages =[]
        # for message in all_messages:
        #     d = message_to_dict(message)
        #     new_messages.append(d)

        new_messages = [message_to_dict(message) for message in all_messages]
        #将数据写入文件
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f)

    #查看消息函数
    @property #property装饰器将messages方法变成成员属性使用
    def messages(self) -> list[BaseMessage]:
        #当前的文件是list[字典]
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages_data = json.load(f)
            return messages_from_dict(messages_data)

        except FileNotFoundError:
            return []

    #清理文件函数
    def clear(self)->None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)