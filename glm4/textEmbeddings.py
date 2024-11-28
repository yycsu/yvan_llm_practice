import torch
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain.embeddings.base import Embeddings

class BGEMilvusEmbeddings(Embeddings):
    def __init__(self):
        self.model = BGEM3EmbeddingFunction(
                    model_name='/root/autodl-tmp/models/bge-m3', # Specify the model name
                    device='cpu', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                    use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
                )

    def embed_documents(self, texts):
        embeddings = self.model.encode_documents(texts)
        return [i.tolist() for i in embeddings["dense"]]

    def embed_query(self, text):
        embedding = self.model.encode_queries([text])
        return embedding["dense"][0].tolist()