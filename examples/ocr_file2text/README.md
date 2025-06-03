# (LightRAG + DeepSeek + 本地嵌入)

## 主要功能

*   **DeepSeek LLM**: 使用 `deepseek-chat` 模型通过其 API 进行高质量的文本生成。
*   **本地嵌入**: 使用 `sentence-transformers/all-MiniLM-L6-v2` 模型在本地生成文本嵌入，维度为 384。
*   **多文档类型支持**: 通过 `MultiProcessor` 模块，可以从多种文件格式（如 `.docx`, `.md`, `.pdf`, `.xlsx`,`.pptx`,`.png` 等）中提取文本。

## 配置
1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    
2.  **DeepSeek API Key**:
    需要一个 DeepSeek API Key，可以通过以下方式配置：
    *   **环境变量**: 设置名为 `DEEPSEEK_KEY` 的环境变量。
        ```bash
        export DEEPSEEK_KEY="actual_deepseek_api_key"
        ```
        代码中的 `get_deepseek_key()` 函数会尝试从此环境变量加载密钥。
    *   **代码内设置**: 使用 `set_deepseek_key('actual_deepseek_api_key')` 函数在代码中直接设置。

3**工作目录**:
    `initialize_rag` 函数需要一个 `work_dir` 参数，LightRAG 会在这个目录下存储其索引和其他数据。如果目录不存在，会自动创建。


## 使用方法
 `rag_add_file.py`

```python
import rag_add_file as ra
import asyncio

# 设置全局deepseek key
ra.set_deepseek_key('your-deepseek-key')
# 或者获取环境变量中的key
ra.get_deepseek_key()

# 初始化rag实例
rag=asyncio.run(ra.initialize_rag(work_dir='./workdir'))

#需要传入的file_path，支持单个和多个
file_path=['./test.docx','./test.md','./test.pdf','./test.xlsx','./test.docx']

#为实例加入文档
ra.add_document_to_rag(file_path,rag)

#使用rag进行查询
text=ra.query_with_rag('这个文档介绍了什么？',rag)
print(text)
```
可以直接使用 `example.py` 代码测试

```python
python ./example.py
```

**查询**
```python
response = ra.query_with_rag('问题', rag, mode='mix')
```
或者

```python
from lightrag import QueryParam

param=QueryParam(mode ='hybrid', response_type = 'Multiple Paragraphs')
response = ra.query_with_rag('问题', rag, param)
```