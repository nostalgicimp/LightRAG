import rag_add_file as ra

import asyncio

file_path=['./test_file/test.pptx']

ra.set_deepseek_key('')

rag=asyncio.run(ra.initialize_rag(work_dir='./workdir'))

ra.add_document_to_rag(file_path,rag)

text=ra.query_with_rag('这个ppt是关于什么的？',rag)

print(text)