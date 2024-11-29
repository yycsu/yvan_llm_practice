# import os
# import sys
# # 获取当前工作目录
# current_dir = os.getcwd()
# # 获取上一级目录
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# # 将上一级目录添加到 sys.path
# sys.path.insert(0, parent_dir)

from transformers import AutoTokenizer, AutoModel
import torch
from glm4LLM_VLLM import ChatGLM4_LLM
from langchain_community.llms import VLLM


# my_gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
# llm = ChatGLM4_LLM(model_name_or_path="/root/autodl-tmp/models/glm-4-9b-chat-hf", gen_kwargs=gen_kwargs)
# llm = VLLM(
#     model="/root/autodl-tmp/models/glm-4-9b-chat",
#     trust_remote_code=True,  # mandatory for hf models
#     do_sample=True,
#     max_new_tokens=2500,
#     top_k=1,
#     top_p=0.95,
#     temperature=0.8,
# )

llm = ChatGLM4_LLM(api_base_url="http://localhost:8002/v1")
print(llm.invoke("你是谁"))

