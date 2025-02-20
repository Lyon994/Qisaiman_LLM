'''集中管理应用程序所需的API基础地址和API密钥'''
'''定义了一个配置类Config，用于存储API相关的配置'''
class Config:
    def __init__(self):
        self.zhipu_api_base= "https://open.bigmodel.cn/api/paas/v4/"
        self.zhipu_api_key = "5492f6483309490091434c5f2f444832.xkLBhAlL007naX9f"


#LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
#LLM_API_KEY=8cb803cc363841ab8d0e31849b32b8b5.JUiRf1JJBA4C4pFL
#MODEL_NAME=glm-4