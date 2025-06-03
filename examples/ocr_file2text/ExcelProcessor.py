
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import os

class ExcelProcessor:
    """处理 Excel 文件内容，提取文本数据"""

    def __init__(self):
        pass

    def extract_text_from_excel(self, excel_path: str, include_metadata: bool = False,
                              sheet_names: Optional[list[str]] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        从Excel文件中提取文本内容
        
        参数:
            excel_path: Excel文件路径
            include_metadata: 是否包含文件元数据
            sheet_names: 指定要处理的工作表名称列表，None表示处理所有工作表
            
        返回:
            元组:(提取的文本内容, 元数据字典)
        """
        text_parts = []
        metadata_dict: Optional[Dict[str, Any]] = None

        try:
            # 1. 提取元数据
            if include_metadata:
                metadata_dict = self._get_excel_metadata(excel_path)
            
            # 2. 读取Excel文件
            excel_file = pd.ExcelFile(excel_path)
            
            # 确定要处理的工作表
            sheets_to_process = sheet_names if sheet_names else excel_file.sheet_names
            
            # 3. 提取每个工作表的文本
            for sheet_name in sheets_to_process:
                try:
                    df = excel_file.parse(sheet_name)
                    sheet_text = self._dataframe_to_text(df, sheet_name)
                    text_parts.append(sheet_text)
                except Exception as e:
                    print(f"处理工作表 '{sheet_name}' 时出错: {e}")
                    continue
                    
        except Exception as e:
            raise Exception(f"无法处理Excel文件: {str(e)}")

        return "\n\n".join(text_parts).strip(), metadata_dict

    def _get_excel_metadata(self, excel_path: str) -> Dict[str, Any]:
        """获取Excel文件的基本元数据"""
        file_stats = os.stat(excel_path)
        return {
            "file_name": os.path.basename(excel_path),
            "file_size": file_stats.st_size,
            "created_time": file_stats.st_ctime,
            "modified_time": file_stats.st_mtime
        }

    def _dataframe_to_text(self, df: pd.DataFrame, sheet_name: str) -> str:
        """将DataFrame转换为文本格式"""
        text_parts = [f"=== 工作表: {sheet_name} ==="]
        
        # 处理列名
        columns = [str(col) for col in df.columns]
        text_parts.append("列名: " + ", ".join(columns))
        
        # 处理数据行
        for index, row in df.iterrows():
            row_values = [str(val) if pd.notna(val) else "" for val in row]
            text_parts.append(f"行 {index}: " + " | ".join(row_values))
            
        return "\n".join(text_parts)