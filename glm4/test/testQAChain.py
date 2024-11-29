from glm4LLM import ChatGLM4_LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from textEmbeddings import BGEMilvusEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import re

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
# 加载本地LLM模型
llm = ChatGLM4_LLM(model_name_or_path="/root/autodl-tmp/models/glm-4-9b-chat-hf", gen_kwargs=gen_kwargs)
# 加载本地向量数据库与embeddings模型
bge_embedding = BGEMilvusEmbeddings()
persist_directory='/root/autodl-tmp/vectorDatabase/chroma'
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=bge_embedding
)

template = """请结合上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
上下文：{chat_history}
问题: {question}
"""
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
#                                  template=template)
prompt_template  = PromptTemplate(input_variables=["context","question","chat_history"],
                                 template=template)

# 创建检索 QA 链的方法 RetrievalQA.from_chain_type() 有如下参数：
# llm：指定使用的 LLM
# 指定 chain type : RetrievalQA.from_chain_type(chain_type="map_reduce")，也可以利用load_qa_chain()方法指定chain type。
# 自定义 prompt ：通过在RetrievalQA.from_chain_type()方法中，指定chain_type_kwargs参数，而该参数：chain_type_kwargs = {"prompt": PROMPT}
# 返回源文档：通过RetrievalQA.from_chain_type()方法中指定：return_source_documents=True参数；也可以使用RetrievalQAWithSourceChain()方法，返回源文档的引用（坐标或者叫主键、索引）
# 注意该函数不支持记忆功能，若要启用记忆功能可使用ConversationalRetrievalChain
# qa_chain = RetrievalQA.from_chain_type(llm,
#                                        retriever=vectordb.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

# 使用带记忆功能的ConversationalRetrievalChain
memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)

retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    memory=memory
)
question_1 = "你知道什么是大模型吗？"
# result = qa_chain({"query": question_1})
result_1 = qa({"question": question_1})
print("大模型+知识库后回答 question_1 的结果：")
# print(result["result"])
print(result_1["answer"])

question_2 = "怎么设计prompt呢？"
# result = qa_chain({"query": question_1})
result_2 = qa({"question": question_2})
print("大模型+知识库后回答 question_2 的结果：")
# print(result["result"])
print(result_2["answer"])


question_3 = "我的上一个问题是什么？"
result_3 = qa({"question": question_3})
print("大模型+知识库后回答 question_3 的结果：")
print(result_3["answer"])