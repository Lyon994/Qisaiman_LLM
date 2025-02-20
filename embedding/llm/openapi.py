# from embedding.llm.base import BaseLLM
# from base import Config
# import openai
#
#
# class Openapi(BaseLLM):
#     def __init__(self, config: 'Config'):
#         self.__config = config
#         openai.api_base = config.openapi_base
#         openai.api_key = config.openapi_key
#         self.__context = []
#
#     def embedding(self, text: str) -> dict[dict]:
#         embedding = openai.Embedding.create(model="text-embedding-ada-002", input=text)
#         return embedding.data[0].embedding
#
#     def ask(self, query: str, context: str) -> str:
#         messages = [
#             {"role": "system", "content": f'你是一个乐于助人的作者，你需要从下文中提取有用的内容来解答用户提出的问题，不能回答不在下文提到的内容，回答请以我的视角回答：\n\n{context}'}
#         ]
#         self.__context.append({"role": "user", "content": query})
#         messages.extend(self.__context)
#         print(messages)
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo-16k",
#             messages=messages
#         )
#         answer = response.choices[0].message.content
#         print("使用的tokens：", response.usage.total_tokens)
#         self.__context.append({"role": "assistant", "content": answer})
#         return answer
#
#     def clear(self):
#         self.__context = []

# import requests
# from embedding.llm.base import BaseLLM
# from base import Config
#
# class KimiAPI(BaseLLM):
#     def __init__(self, config: 'Config'):
#         self.__config = config
#         self.api_base = "https://api.moonshot.cn/v1"# 替换为 Kimi 的 API 地址
#         self.api_key =  "sk-vz7J0N1Tw4xTNaSAV2tdePiMqK9cvCxY8Rte0vOznBtDYA1O"  # 替换为 Kimi 的 API 密钥
#         self.__context = []
#
#     def embedding(self, text: str) -> list:
#         url = f"{self.api_base}/v1/embedding"
#         headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
#         payload = {"model": "kimi-embedding-model", "input": text}
#         response = requests.post(url, json=payload, headers=headers)
#         response_data = response.json()
#
#         if "data" in response_data:
#             return response_data["data"][0]["embedding"]
#         else:
#             raise Exception(f"Embedding API error: {response_data}")
#
#     def ask(self, query: str, context: str) -> str:
#         url = f"{self.api_base}/v1/chat/completions"
#         headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
#         messages = [
#             {"role": "system", "content": f'你是一个乐于助人的作者，你需要从下文中提取有用的内容来解答用户提出的问题，不能回答不在下文提到的内容，回答请以我的视角回答：\n\n{context}'}
#         ]
#         self.__context.append({"role": "user", "content": query})
#         messages.extend(self.__context)
#
#         payload = {
#             "model": "kimi-chat-model",  # 替换为 Kimi 支持的模型
#             "messages": messages
#         }
#
#         response = requests.post(url, json=payload, headers=headers)
#         response_data = response.json()
#
#         if "choices" in response_data:
#             answer = response_data["choices"][0]["message"]["content"]
#             self.__context.append({"role": "assistant", "content": answer})
#             return answer
#         else:
#             raise Exception(f"Chat API error: {response_data}")
#
#     def clear(self):
#         self.__context = []

'''用于调用API实现文本嵌入和问答功能'''
import requests #用于发送HTTP请求，与API通信
from embedding.llm.base import BaseLLM
from base import Config

class ZhipuAPI(BaseLLM):
    def __init__(self, config: 'Config'):
        self.__config = config
        self.api_base = self.__config.zhipu_api_base.rstrip("/")  # 确保结尾没有 "/"
        self.api_key = self.__config.zhipu_api_key  # 从 Config 读取 API Key
        self.__context = [] #初始化一个空列表，用于存储聊天的上下文信息

    def embedding(self, text: str) -> list:
        """获取文本的向量表示（适配 Zhipu API），text是字符串"""
        url = f"{self.api_base}/embeddings"  # 得到完整的本文嵌入API请求
        '''定义一个字典，用于存储HTTP请求的头部信息'''
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        '''定义一个字典，作为HTTP请求的请求体'''
        payload = {"model": "embedding-2", "input": text}  # 确保使用正确的模型

        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()

        if "data" in response_data:
            return response_data["data"][0]["embedding"]
        else:
            raise Exception(f"Embedding API error: {response_data}")

    def ask(self, query: str, context: str) -> str:
        """调用 Zhipu API 进行问答"""
        url = f"{self.api_base}/chat/completions"  # 智普的 chat API
        '''构建请求URL和请求头'''
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        '''构建消息队列'''
        messages = [
            {
                "role": "system",
                "content": f'''您好！我是小智，您的智能助手。
            我会仔细阅读以下上下文，并基于其中的信息为您提供解答。如果问题涉及的内容不在上下文中，我会如实告知。
            回答时会尽量使用亲切易懂的语言，并以您的视角来组织内容。\n\n上下文参考：\n{context}'''
            }
        ]
        '''更新上下文合并消息列表'''
        self.__context.append({"role": "user", "content": query})
        messages.extend(self.__context)

        payload = {
            "model": "glm-4",  # 使用 Zhipu 最新的 glm-4 模型
            "messages": messages
        }

        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()

        if "choices" in response_data:
            answer = response_data["choices"][0]["message"]["content"]
            self.__context.append({"role": "assistant", "content": answer})
            return answer
        else:
            raise Exception(f"Chat API error: {response_data}")

    def clear(self):
        """清空聊天上下文"""
        self.__context = []
