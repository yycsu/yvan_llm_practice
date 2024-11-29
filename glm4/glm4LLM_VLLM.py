from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from openai import OpenAI


class ChatGLM4_LLM(LLM):
    # 基于本地 ChatGLM4 自定义 LLM 类
    api_client : OpenAI = None
        
    # def __init__(self, api_base_url: str, gen_kwargs: dict = None):
    def __init__(self, api_base_url: str):
        super().__init__()
        print("正在从本地加载模型...")
        self.api_client = OpenAI(api_key="EMPTY", base_url=api_base_url)
        print("完成本地模型的加载")
        
        # if gen_kwargs is None:
        #     gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        # self.gen_kwargs = gen_kwargs
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        # model_inputs = self.tokenizer.apply_chat_template(
        #     messages, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True
        # )
        
        # # 将input_ids移动到与模型相同的设备
        # device = next(self.model.parameters()).device
        # model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
        
        # generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        # ]
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        response = self.api_client.chat.completions.create(
            model="glm-4-9b-chat",
            messages=messages,
        )
        generated_text = response.choices[0].message.content
        return generated_text
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于识别LLM的字典,这对于缓存和跟踪目的至关重要。"""
        return {
            "model_name": "glm-4-9b-chat",
            # "max_length": self.gen_kwargs.get("max_length"),
            # "do_sample": self.gen_kwargs.get("do_sample"),
            # "top_k": self.gen_kwargs.get("top_k"),
        }

    @property
    def _llm_type(self) -> str:
        return "glm-4-9b-chat"