from web import app #从web模块导入Flask应用实例
from embedding import Embedding #导入自定义的嵌入模型类
from base import EMBEDDING #从base模块导入配置常量
import webbrowser
from threading import Timer

def open_browser():
    '''自动打开浏览器访问本地服务'''
    webbrowser.open("http://127.0.0.1:7001")

if __name__ == '__main__':
    # 配置相关服务
    app.config[EMBEDDING] = Embedding() # 将嵌入模型实例挂载到Flask配置中

    Timer(1, open_browser).start()

    # 运行flask内置开发服务器
    app.run(host='0.0.0.0', port=7001) # 运行服务器，允许外部访问

