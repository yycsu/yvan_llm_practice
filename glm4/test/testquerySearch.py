from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from textEmbeddings import BGEMilvusEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
import re

embedding = BGEMilvusEmbeddings()

persist_directory='/root/autodl-tmp/vectorDatabase/chroma'

vectordb_load = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
# 相似度检索，k代表返回多少条结果
sim_docs_similarity_search=vectordb_load.similarity_search("如何设计prompt",k=5)
# MMR检索
# sim_docs_MMR_search=vectordb_load.max_marginal_relevance_search("请于此处输入与数据集相关的语句",k=3)

for i, sim_doc in enumerate(sim_docs_similarity_search):
    print(f"通过相似度检索到的第{i}个内容: \n{sim_doc.page_content}", end="\n--------------\n")
# for i, sim_doc in enumerate(sim_docs_MMR_search):
#     print(f"通过MMR检索到的第{i}个内容: \n{sim_doc.page_content}", end="\n--------------\n")