import os
import numpy as np
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status
from openai import OpenAI
import asyncio
from MultiProcessor import MultiProcessor
file_processor=MultiProcessor()

DEEPSEEK_KEY=''

def get_deepseek_key():
    key = os.environ.get("DEEPSEEK_KEY")
    if key is None:
        raise ValueError("DEEPSEEK_KEY 未在环境变量中设置！")
    global DEEPSEEK_KEY
    DEEPSEEK_KEY=key

def set_deepseek_key(deepseek_key:str):
    global DEEPSEEK_KEY
    DEEPSEEK_KEY=deepseek_key


async def deepseek_model_func(
    prompt: str, system_prompt: str = None, history_messages: list = None, keyword_extraction: bool = False, **kwargs
) -> str:
    if history_messages is None:
        history_messages = []


    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    for msg in history_messages:
        messages.append({
            "role": msg["role"].lower() if msg["role"].lower() in ["user", "assistant"] else "user",
            "content": msg["content"]
        })
    
    # Add the current prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Initialize DeepSeek client
        client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
        
        # Make the API call
        response = client.chat.completions.create(
            model="deepseek-chat",  #调用 DeepSeek-V3; 如果model="deepseek-reasoner"可调用 DeepSeek-R1
            messages=messages,
            stream=False
        )
        
        # Extract and return the response
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            print(f"Warning: DeepSeek response object was unexpected: {response}")
            return str(response)
            
    except Exception as e:
        print(f"Error in LLM generation with prompt '{prompt[:50]}...': {type(e).__name__} - {str(e)}")
        raise

async def minilm_embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

async def initialize_rag(work_dir:str,llm_model_func=None,embedding_func=None):
    if llm_model_func is None:
        llm_model_func=deepseek_model_func
    if embedding_func is None:
        embedding_func=minilm_embedding_func
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    rag = LightRAG(
        working_dir=work_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384, # all-MiniLM-L6-v2 的维度
            max_token_size=8192, # SentenceTransformer
            func=embedding_func
        )
    )

    await rag.initialize_storages()
    if asyncio.iscoroutinefunction(initialize_pipeline_status):
         await initialize_pipeline_status()
    else:
         initialize_pipeline_status()

    return rag



def add_document_to_rag(file_path:str|list[str],rag_instance,include_images=True,include_metadata=False):
    text_list=[]
    file_path_list=[]
    if isinstance(file_path, str):
        # 处理单个文件路径
        text, metadata = file_processor.file2text(file_path,include_images=include_images,include_metadata=include_metadata)
        text_list.append(text)
        file_path_list.append(file_path)
    else:
        for single_path in file_path:
            text, metadata = file_processor.file2text(single_path,include_images=include_images,include_metadata=include_metadata)
            text_list.append(text)
            file_path_list.append(single_path)
    rag_instance.insert(text_list,file_paths=file_path_list)


def query_with_rag(query:str,rag_instance,mode='hybrid',param=None):
    if param is None:
        param=QueryParam(mode=mode)
    
    response=rag_instance.query(
            query,
            param=param
        )
    return response