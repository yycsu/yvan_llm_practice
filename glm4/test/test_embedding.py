import os
import sys
# 获取当前工作目录
current_dir = os.getcwd()
# 获取上一级目录
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# 将上一级目录添加到 sys.path
sys.path.insert(0, parent_dir)

from transformers import AutoTokenizer, AutoModel
import torch
from textEmbeddings import BGEMilvusEmbeddings

# model_name = "bge-m3" 
# save_directory = "/root/autodl-tmp/models/bge-m3"
# tokenizer = AutoTokenizer.from_pretrained(model_name, )
# model = AutoModel.from_pretrained(model_name)
# embedding = BGEM3Embeddings()

embedding_model = BGEMilvusEmbeddings()

# 生成文本嵌入
text = "中国的首都是北京"
embedding = embedding_model.embed_query(text)
print(embedding)

# 生成多个文档的嵌入
documents = ["华盛顿是美国的首都", "法国的首都是巴黎", "日本的首都是东京"]
document_embeddings = embedding_model.embed_documents(documents)
print(document_embeddings)

# 生成查询的嵌入
query = "美国首都在哪？"
query_embedding = embedding_model.embed_query(query)
print(query_embedding)