# import chromadb
# import uuid
# from chromadb.utils import embedding_functions
# from base import Config
#
#
# class ChromadbDb:
#     def __init__(self):
#         self.__config = Config()
#         self.__chroma_client = chromadb.PersistentClient(path="db")
#         self.__collection = self.__chroma_client.get_or_create_collection(
#             name="embedding",
#             embedding_function=embedding_functions.OpenAIEmbeddingFunction(
#                 api_base=self.__config.openapi_base,
#                 api_key=self.__config.openapi_key,
#                 model_name="text-embedding-ada-002",
#             ))
#
#     # 添加文本
#     def add_text(self, text: str, embedding: dict[dict], meta: dict):
#         self.__collection.add(
#             documents=[text],
#             embeddings=[embedding],
#             metadatas=[meta],
#             ids=[uuid.uuid4().hex]
#         )
#
#     def query_text(self, query: str, result: int):
#         results = self.__collection.query(
#             query_texts=[query],
#             n_results=result
#         )
#         return results
#
#     def get_data(self, no: int, size: int) -> [int, dict]:
#         count = self.__collection.count()
#         result = self.__collection.get(limit=size, offset=(no - 1) * size)
#         return count, result
#
#     def delete_data(self, ids: list[str]):
#         self.__collection.delete(ids=ids)
# import chromadb
# import uuid
# import requests
# from base import Config
#
#
# class KimiEmbeddingFunction:
#     """自定义 Kimi 词向量（embedding）函数"""
#     def __init__(self, api_base: str, api_key: str, model_name: str):
#         self.api_base = api_base
#         self.api_key = api_key
#         self.model_name = model_name
#
#     def __call__(self, input):  # ⚠️ 这里改成 input，符合 ChromaDB 0.4.16 及以上版本的要求
#         """批量获取文本的向量表示"""
#         url = f"{self.api_base}"
#         print(f"Embedding 请求 URL: {url}")
#         headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
#         payload = {"model": self.model_name, "input": input}  # 这里的 key 仍然是 input
#
#         response = requests.post(url, json=payload, headers=headers)
#         response_data = response.json()
#
#         if "data" in response_data:
#             return [item["embedding"] for item in response_data["data"]]
#         else:
#             raise Exception(f"Embedding API error: {response_data}")
#
#
# class ChromadbDb:
#     def __init__(self):
#         self.__config = Config()
#         print(f"Kimi API Base: {self.__config.kimi_api_base}")
#         self.__chroma_client = chromadb.PersistentClient(path="db")
#
#         # 替换 OpenAIEmbeddingFunction 为自定义的 KimiEmbeddingFunction
#         self.__collection = self.__chroma_client.get_or_create_collection(
#             name="embedding",
#             embedding_function=KimiEmbeddingFunction(
#                 api_base=self.__config.kimi_api_base,  # 确保 Config 变量名一致
#                 api_key=self.__config.kimi_api_key,
#                 model_name="moonshot-v1-8k"  # 替换为 Kimi 实际的 embedding 模型名
#             )
#         )
#
#     # 添加文本
#     def add_text(self, text: str, embedding: list, meta: dict):
#         self.__collection.add(
#             documents=[text],
#             embeddings=[embedding] if isinstance(embedding[0], list) else [embedding],  # 确保 embedding 是 list of list
#             metadatas=[meta],
#             ids=[uuid.uuid4().hex]
#         )
#
#     def query_text(self, query: str, result: int):
#         results = self.__collection.query(
#             query_texts=[query],
#             n_results=result
#         )
#         return results
#
#     def get_data(self, no: int, size: int) -> [int, dict]:
#         count = self.__collection.count()
#         result = self.__collection.get(limit=size, offset=(no - 1) * size)
#         return count, result
#
#     def delete_data(self, ids: list[str]):
#         self.__collection.delete(ids=ids)



import chromadb
import uuid
import requests
from base import Config


class ZhipuEmbeddingFunction:
    """自定义 Zhipu（智普）词向量（embedding）函数"""
    def __init__(self, api_base: str, api_key: str, model_name: str = "embedding-2"):
        self.api_base = api_base.rstrip("/") + "/embeddings"  # 确保 URL 正确拼接
        self.api_key = api_key
        self.model_name = model_name  # 确保默认使用 embedding-2

    def __call__(self, input):
        """批量获取文本的向量表示"""
        url = self.api_base
        print(f"Embedding 请求 URL: {url}")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # 确保 input 是列表格式（符合 API 要求）
        if isinstance(input, str):
            input = [input]

        payload = {"model": self.model_name, "input": input}

        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()

        if "data" in response_data:
            return [item["embedding"] for item in response_data["data"]]
        else:
            raise Exception(f"Embedding API error: {response_data}")


class ChromadbDb:
    def __init__(self):
        self.__config = Config()
        print(f"Zhipu API Base: {self.__config.zhipu_api_base}")
        self.__chroma_client = chromadb.PersistentClient(path="db")

        # 替换 ZhipuEmbeddingFunction，确保 model_name 正确
        self.__collection = self.__chroma_client.get_or_create_collection(
            name="embedding",
            embedding_function=ZhipuEmbeddingFunction(
                api_base=self.__config.zhipu_api_base,
                api_key=self.__config.zhipu_api_key,
                model_name="embedding-2"  # **修正为 embedding-2**
            )
        )

    # 添加文本
    # def add_text(self, text: str, embedding: list, meta: dict):
    #     """添加文本及其向量到数据库"""
    #     self.__collection.add(
    #         documents=[text],
    #         embeddings=[embedding] if isinstance(embedding[0], list) else [[embedding]],  # 确保是 list of list
    #         metadatas=[meta],
    #         ids=[uuid.uuid4().hex]
    #     )
    import uuid

    def add_text(self, text: str, embedding: list, meta: dict):
        """添加文本及其向量到数据库"""
        # 确保 embedding 是一维列表
        if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
            embeddings = [embedding]  # 如果已经是一维列表，直接使用
        elif isinstance(embedding, list) and all(isinstance(x, list) for x in embedding):
            embeddings = embedding  # 如果是二维列表，展平成一维列表
        else:
            raise ValueError("Invalid embedding format. It should be a list of int or float.")

        self.__collection.add(
            documents=[text],
            embeddings=embeddings,  # 确保是 list of list
            metadatas=[meta],
            ids=[uuid.uuid4().hex]
        )
    def query_text(self, query: str, result: int):
        """查询与 query 相关的向量"""
        results = self.__collection.query(
            query_texts=[query],
            n_results=result
        )
        return results

    def get_data(self, no: int, size: int) -> (int, dict):
        """分页获取存储的数据"""
        count = self.__collection.count()
        result = self.__collection.get(limit=size, offset=(no - 1) * size)
        return count, result

    def delete_data(self, ids: list[str]):
        """删除指定 ID 的数据"""
        self.__collection.delete(ids=ids)

