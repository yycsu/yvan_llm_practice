from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from textEmbeddings import BGEMilvusEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
import re
# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("/root/autodl-tmp/dataset/LLM-v1.0.0.pdf")
# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader.load()

# 进行文档格式清洗
for pdf_page in pdf_pages:
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
    pdf_page.page_content = pdf_page.page_content.replace('•', '')
    pdf_page.page_content = pdf_page.page_content.replace(' ', '')
    # pdf_page.page_content = pdf_page.page_content.replace('\n', '')
    # print(pdf_page.page_content)

# 知识库中单段文本长度
CHUNK_SIZE = 500

# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50

# 使用递归字符文本分割器
# RecursiveCharacterTextSplitter 递归字符文本分割
# RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
#     这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE,
    separators=["\n\n", "\n", " ", "", "。", "，"]
)

split_docs = text_splitter.split_documents(pdf_pages)

# print(split_docs[150].page_content)


# 定义本地embedding模型
embedding_model = BGEMilvusEmbeddings()

persist_directory='/root/autodl-tmp/vectorDatabase/chroma'

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    # 允许我们将persist_directory目录保存到磁盘上
    persist_directory=persist_directory  
)

vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")