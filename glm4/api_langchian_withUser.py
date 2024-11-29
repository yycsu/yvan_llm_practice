from typing import List
from fastapi import FastAPI, Request,File,UploadFile,Form
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from glm4LLM_VLLM import ChatGLM4_LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from textEmbeddings import BGEMilvusEmbeddings
# from bge_m3_embeddings import BGEM3Embeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader,Docx2txtLoader,PyMuPDFLoader
from langchain_community.llms import VLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import uvicorn
import json
import os
import datetime
import torch

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_IDS = ["0"]  # CUDA设备ID列表，适配两个GPU

# 组合CUDA设备信息，适配多个GPU
CUDA_DEVICES = [f"{DEVICE}:{device_id}" for device_id in DEVICE_IDS]

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        for cuda_device in CUDA_DEVICES:  # 遍历所有CUDA设备
            with torch.cuda.device(cuda_device):  # 指定当前CUDA设备
                torch.cuda.empty_cache()  # 清空CUDA缓存
                torch.cuda.ipc_collect()  # 收集CUDA内存碎片
                
def persistDataset(agentId: str,split_docs):
    # embedding_model_name = "bge-m3"  
    # embedding_save_directory = "/root/autodl-tmp/bge-m3"
    # embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # embedding_model = AutoModel.from_pretrained(model_name)
    # persist_embedding = BGEM3Embeddings(embedding_model, embedding_tokenizer)
    chroma_directory='/root/autodl-tmp/vectorDatabase/chroma'
    agent_persist_directory=os.path.join(chroma_directory,agentId)
    if not os.path.exists(agent_persist_directory):
        os.mkdir(agent_persist_directory)
    print("开始进行embedding。。。")
    persist_vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        # 允许我们将persist_directory目录保存到磁盘上
        persist_directory=agent_persist_directory  
    )
    print("进行向量数据持久化。。。")
    persist_vectordb.persist()
    print("文件持久化成功！")
    print(f"向量库中存储的数量：{persist_vectordb._collection.count()}")
def cleanse_dataset_format(dataset_pages):
    chinese_and_punctuation = r'\u4e00-\u9fff\u3002\uff0c\u3001\u201c\u201d\u300a\u300b\uff1f\uff01'
    for dataset_page in dataset_pages:
        # pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        pattern = re.compile(rf'[^{chinese_and_punctuation}](\n)[^{chinese_and_punctuation}]', re.DOTALL)
        dataset_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), dataset_page.page_content)
        dataset_page.page_content = dataset_page.page_content.replace('•', '')
        dataset_page.page_content = dataset_page.page_content.replace(' ', '')
    # 知识库中单段文本长度
    CHUNK_SIZE = 500
    # 知识库中相邻文本重合长度
    OVERLAP_SIZE = 50
    print("正在对该文件进行格式清洗。。。")
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE,
    separators=["\n\n", "\n", "。", "，", "：", "！", "？", "；", " "]
    )
    return text_splitter.split_documents(dataset_pages)
def embeddingAndPersistDataSet(agentId: str,file_path: str):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.txt':
        loader = TextLoader(file_path,encoding='utf-8')
        txt_pages = loader.load()
        split_docs = cleanse_dataset_format(txt_pages)
        persistDataset(agentId,split_docs)
    elif file_extension in ('.doc', '.docx'):
        loader = Docx2txtLoader(file_path)
        doc_pages = loader.load()
        split_docs = cleanse_dataset_format(doc_pages)
        persistDataset(agentId,split_docs)
    elif file_extension in ('.pdf'):
        # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
        loader = PyMuPDFLoader(file_path)
        # 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
        doc_pages = loader.load()
        split_docs = cleanse_dataset_format(doc_pages)
        persistDataset(agentId,split_docs)
    # elif file_extension in ('.jpg', '.jpeg'):
    #     file_type = 'JPEG 图像文件'
    # elif file_extension == '.png':
    #     file_type = 'PNG 图像文件'
    else:
        file_type = '未知类型'
    
# 创建FastAPI应用
app = FastAPI()

from pydantic import BaseModel
# 创建问答接口输入数据
class ChatInfoItem(BaseModel):
    message: str
    response: str

class Item(BaseModel):
    question: str
    chatInfo: List[ChatInfoItem]
    isNew: bool
    agentId: str
    otherParams: dict

# 问答接口
@app.post("/")
async def create_item(item: Item):
    global llm, tokenizer,prompt_template,embedding,persist_directory  # 声明全局变量以便在函数内部使用模型和分词器
    # json_post_raw = await request.json()  # 获取POST请求的JSON数据
    # json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    # json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    # question=json_post_list.get('question')
    # chat_info=json_post_list.get('chatInfo')
    # is_new=json_post_list.get('isNew')
    # agent_id=json_post_list.get('agentId')
    # other_params=json_post_list.get('otherParams')

    question = item.question
    chat_info = item.chatInfo
    is_new = item.isNew
    agent_id = item.agentId
    other_params = item.otherParams

    # 获取对应智能体的向量数据库
    agent_persist_directory=os.path.join('/root/autodl-tmp/vectorDatabase/chroma', agent_id)
    print(type(agent_id))
    if os.path.exists(agent_persist_directory) and agent_id:   
        vectordb = Chroma(
            persist_directory=agent_persist_directory,
            embedding_function=embedding
        )
        print(f"{agent_persist_directory}") 
        # 定义检索功能
        retriever=vectordb.as_retriever()
    else:
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        print(f"{persist_directory}") 
        # 定义检索功能
        retriever=vectordb.as_retriever()
    # 定义记忆功能
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    if not is_new:
        # 加载会话历史
        if any(chat_info):
            for entry in chat_info:
                inputs = {"input": entry.message}
                outputs = {"output": entry.response}
                memory.save_context(inputs, outputs)
            print(memory.buffer)
    # 调用模型进行对话生成
    qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    memory=memory
    )
    
    response=qa({"question":question})["answer"] 
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {
        "response": response,
        "status": 200,
        "time": time,
        "otherParams":other_params
    }
    # 构建日志信息
    log = "[" + time + "] " + '", response:"' + repr(response) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应

@app.post("/deleteDataSet")
async def files(agentId: str = Form(...),fileName: str = Form(...)):
    agent_path=f"/root/autodl-tmp/agent"
    save_agent_file_path=os.path.join(agent_path,agentId)
    if not os.path.exists(save_agent_file_path):
        return {
            "response":f"该智能体不存在！",
            "status": 200,
        }
    file_name=fileName
    file_path=os.path.join(save_agent_file_path, file_name)
    try:
        # 删除文件
        os.remove(file_path)
        return {
            "response":f"文件 {file_name} 删除成功",
            "status": 200,
        }
    except OSError as e:
         return {
            "response":f"错误：文件 {file_name} 删除失败 - {e}",
            "status": 200,
        }
@app.post("/saveAgent")
async def upload_files(files: List[UploadFile] = File(...),agentId: str = Form(...)):
    agent_path=f"/root/autodl-tmp/agent"
    save_agent_file_path=os.path.join(agent_path,agentId)
    if not os.path.exists(save_agent_file_path):
        os.mkdir(save_agent_file_path)
    for file in files:
        file_name=file.filename
        file_path=os.path.join(save_agent_file_path, file_name)
        f = open(file_path, 'wb')
        data = await file.read()
        f.write(data)
        f.close()
        embeddingAndPersistDataSet(agentId,file_path)
    return {
        # "filenames": [file.filename for file in files],
        # "content_type": [file.content_type for file in files],
        # "file": [file for file in files],
        # "agentId":agentId
        "response":f"成功保存 {len(files)} 个文件",
        "status": 200,
    }
# 主函数入口
if __name__ == '__main__':
    # 加载预训练的分词器和模型
    # tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/ZhipuAI/glm-4-9b-chat", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     "/root/autodl-tmp/ZhipuAI/glm-4-9b-chat",
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     device_map="auto",
    # )
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    # 加载本地LLM模型
    # llm = ChatGLM4_LLM(model_name_or_path="/root/autodl-tmp/models/glm-4-9b-chat-hf", gen_kwargs=gen_kwargs)
    llm = ChatGLM4_LLM(api_base_url="http://localhost:8002/v1")
    # 加载本地向量数据库与embeddings模型
    embedding = BGEMilvusEmbeddings()
    persist_directory='/root/autodl-tmp/vectorDatabase/publicChroma'
    
    # # 定义模版
    # template = """请结合对话历史以及你自己的知识储备来回答最后的问题。如果你不确定问题的答案，请回答你不知道答案，不要凭空捏造答案。
    # 如果你发现对话历史没有内容，请不要凭空捏造出对话历史。例如在你没有对话历史的时候，当有人问你他的上一个问题是什么，请回答我不知道。
    # 最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    # 定义模版
    template = """请结合对话历史以及你自己的知识储备来回答最后的问题。并且遵循下面的规则：
    1、如果你不确定问题的答案，请回答你不知道答案，不要凭空捏造答案。
    2、如果你发现对话历史没有内容，请不要凭空捏造出对话历史。例如在你没有对话历史的时候，当有人问你他的上一个问题是什么，请回答我不知道。
    3、如果对话历史中有内容，可以回答上一个问题是什么。
    4、最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    对话历史：{chat_history}
    问题: {question}
    """
    # 定义提示词
    prompt_template  = PromptTemplate(input_variables=["context","question","chat_history"],
                                 template=template)
    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)  # 在指定端口和主机上启动应用