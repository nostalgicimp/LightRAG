import re
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import os

class TxtProcessor:
    """处理 txt 文件内容，提取文本数据"""

    def __init__(self):
        pass

    def extract_text_from_txt(self, txt_path: str, include_metadata: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        从txt文件中提取文本内容
        
        参数:
            txt_path: txt文件路径
            include_metadata: 是否包含文件元数据
        返回:
            元组:(提取的文本内容, 元数据字典)
        """
        content=''
        metadata_dict: Optional[Dict[str, Any]] = None

        try:
            # 1. 提取元数据
            if include_metadata:
                metadata_dict = self._get_txt_metadata(txt_path)
            with open(txt_path, "r") as file:
                content = file.read()
                    
        except Exception as e:
            raise Exception(f"无法处理Txt文件: {str(e)}")

        return content.strip(), metadata_dict

    def _get_txt_metadata(self, txt_path: str) -> Dict[str, Any]:
        """获取txt文件的基本元数据"""
        file_stats = os.stat(txt_path)
        return {
            "file_name": os.path.basename(txt_path),
            "file_size": file_stats.st_size,
            "created_time": file_stats.st_ctime,
            "modified_time": file_stats.st_mtime
        }

