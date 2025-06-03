import re
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import pytesseract
import os
import numpy as np # PaddleOCR 需要

# 尝试导入PaddleOCR，并处理未安装的情况
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True  # PaddleOCR可用标志
except ImportError:
    PADDLEOCR_AVAILABLE = False # PaddleOCR不可用标志
    PaddleOCR = None # 定义PaddleOCR为None，以避免后续代码出现NameError

class MarkdownProcessor:
    """处理 Markdown 文件内容，可提取文本、表格，并对图像进行OCR文本提取（支持Tesseract和PaddleOCR）。"""

    def __init__(self,
                 tesseract_cmd_path: Optional[str] = None,
                 default_ocr_engine: str = 'paddle', # 默认OCR引擎
                 paddle_ocr_config: Optional[Dict[str, Any]] = None):
        """
        初始化 MarkdownProcessor

        Args:
            tesseract_cmd_path (Optional[str]): Tesseract OCR的可执行文件路径。
                                                如果Tesseract已在系统PATH中，则无需设置。
            default_ocr_engine (str): 默认使用的OCR引擎 ('tesseract' 或 'paddle')。
            paddle_ocr_config (Optional[Dict[str, Any]]): PaddleOCR的初始化配置。
                例如: {'use_angle_cls': True, 'lang': 'ch', 'show_log': False}
                如果为None，将尝试从 ocr_languages 参数映射语言。
        """
        self.tesseract_available = False
        if tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
        
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
        except pytesseract.TesseractNotFoundError:
            print("警告: Tesseract OCR 未找到或未正确配置。如果选择Tesseract，图像文本提取功能将不可用。")
        except Exception as e:
            print(f"警告: 初始化Tesseract时发生错误: {e}。Tesseract图像文本提取功能可能受影响。")

        self.default_ocr_engine = default_ocr_engine.lower()
        if self.default_ocr_engine == 'paddle' and not PADDLEOCR_AVAILABLE:
            print("警告: PaddleOCR库未安装，但被选为默认OCR引擎。将尝试回退到Tesseract (如果可用)，否则不进行图像OCR。")
            if self.tesseract_available:
                self.default_ocr_engine = 'tesseract'
            else:
                self.default_ocr_engine = None # 默认情况下没有可用的OCR引擎

        self.paddle_ocr_instance: Optional[PaddleOCR] = None # PaddleOCR实例，惰性初始化
        self.user_paddle_ocr_config = paddle_ocr_config      # 用户提供的PaddleOCR配置
        self.current_paddle_lang_init = None # 用于跟踪PaddleOCR初始化时使用的语言

    def _initialize_paddleocr(self, lang_for_paddle: str) -> bool:
        """【私有方法】惰性初始化PaddleOCR实例。"""
        if not PADDLEOCR_AVAILABLE:
            print("警告: PaddleOCR库未安装，无法使用PaddleOCR引擎。")
            return False

        if self.user_paddle_ocr_config:
            should_reinitialize_user_config = self.paddle_ocr_instance is None
            if 'lang' in self.user_paddle_ocr_config:
                 should_reinitialize_user_config = should_reinitialize_user_config or \
                    (self.current_paddle_lang_init != self.user_paddle_ocr_config.get('lang'))
            if should_reinitialize_user_config:
                try:
                    print(f"正在使用用户配置初始化PaddleOCR: {self.user_paddle_ocr_config}...")
                    self.paddle_ocr_instance = PaddleOCR(**self.user_paddle_ocr_config)
                    self.current_paddle_lang_init = self.user_paddle_ocr_config.get('lang', 'ch')
                    print("PaddleOCR已通过用户配置成功初始化。")
                    return True
                except Exception as e:
                    print(f"警告: 使用用户配置初始化PaddleOCR失败: {e}")
                    self.paddle_ocr_instance = None
                    return False
            return True

        if self.paddle_ocr_instance is None or self.current_paddle_lang_init != lang_for_paddle:
            try:
                default_config = {'use_angle_cls': True, 'lang': lang_for_paddle, 'show_log': False}
                print(f"正在使用默认配置初始化PaddleOCR (语言: {lang_for_paddle})...")
                self.paddle_ocr_instance = PaddleOCR(**default_config)
                self.current_paddle_lang_init = lang_for_paddle
                print("PaddleOCR已成功初始化。")
                return True
            except Exception as e:
                print(f"警告: 初始化PaddleOCR失败 (语言={lang_for_paddle}): {e}")
                self.paddle_ocr_instance = None
                return False
        return True

    def _map_tesseract_lang_to_paddle(self, tesseract_langs: str) -> str:
        """【私有方法】启发式地将Tesseract语言字符串映射到PaddleOCR语言代码。"""
        if self.user_paddle_ocr_config and 'lang' in self.user_paddle_ocr_config:
            return self.user_paddle_ocr_config['lang']

        t_langs = tesseract_langs.lower()
        if 'chi_sim' in t_langs or 'chi_tra' in t_langs:
            return 'ch'
        if 'eng' in t_langs:
            return 'en'
        # 可以添加更多映射
        print(f"警告: 无法将Tesseract语言 '{tesseract_langs}'直接映射到PaddleOCR支持的语言。将默认使用 'ch'。")
        return 'ch'

    def extract_text_from_markdown(self, md_path: str, 
                                   include_metadata: bool = False,
                                   include_images: bool = False, 
                                   ocr_languages: str = 'chi_sim+eng',
                                   ocr_engine: Optional[str] = None
                                   ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        从 Markdown 文件中提取文本内容，包括正文、元数据（可选）和图像中的文本（可选，通过OCR）。

        Args:
            md_path (str): Markdown 文件的路径。
            include_metadata (bool): 是否提取并返回 YAML Front Matter 元数据。
            include_images (bool): 是否对 Markdown 中引用的本地图像进行 OCR 识别。
            ocr_languages (str): OCR 识别时使用的语言。
                                 - Tesseract: 'eng', 'chi_sim', 'eng+chi_sim'
                                 - PaddleOCR: 会尝试映射到 'ch', 'en' 等。
            ocr_engine (Optional[str]): 使用的OCR引擎 ('tesseract', 'paddle')。如果None，使用默认引擎。

        Returns:
            Tuple[str, Optional[Dict[str, Any]]]: 提取的文本字符串和元数据字典（如果 `include_metadata` 为True）。
        """
        text_parts = []
        metadata_dict: Optional[Dict[str, Any]] = None
        chosen_ocr_engine = (ocr_engine or self.default_ocr_engine or "none").lower()

        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # 1. 提取 YAML 元数据
            if include_metadata:
                # YAML Front Matter 通常位于文件开头，由 '---' 包围
                yaml_match = re.match(r'^\s*---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
                if yaml_match:
                    yaml_text = yaml_match.group(1)
                    try:
                        import yaml # 尝试使用PyYAML进行更健壮的解析
                        metadata_dict = yaml.safe_load(yaml_text)
                    except ImportError:
                        print("提示: PyYAML库未安装，将使用基础方法解析元数据。建议安装 'pip install PyYAML' 以获得更好的元数据解析。")
                        metadata_dict = self._parse_yaml_metadata_basic(yaml_text)
                    except Exception as e_yaml: #捕获yaml解析错误
                        print(f"警告: 解析YAML元数据时出错: {e_yaml}。将使用基础方法解析。")
                        metadata_dict = self._parse_yaml_metadata_basic(yaml_text)
                    
                    content = content[yaml_match.end():]  # 从内容中移除元数据部分
                else:
                    print(f"提示: 在文件 {md_path} 中未找到标准的YAML元数据块。")


            # 2. 提取正文文本（初步剥离Markdown特定语法，如图片引用和表格标记）
            # 移除图片链接: ![alt text](path/to/image.png)
            interim_text = re.sub(r'!\[.*?\]\(.*?\)', '', content) 
            # 移除HTML标签 (Markdown中可能内嵌HTML)
            interim_text = re.sub(r'<[^>]+>', '', interim_text)
            # 移除行内代码标记: `code`
            interim_text = re.sub(r'`(.*?)`', r'\1', interim_text)
            # 移除代码块标记: ```lang ... ```
            interim_text = re.sub(r'```[\s\S]*?```', '', interim_text)
            # 移除表格的横线分隔符行 (例如 |---|---|)
            interim_text = re.sub(r'^\s*\|?[\s\-|:]+\|?\s*$', '', interim_text, flags=re.MULTILINE)
            # 移除纯粹的 Markdown 表格标记符（例如行首和行尾的 |，以及单元格间的 |）
            interim_text = interim_text.replace('|', ' ') # 将竖线替换为空格，以分离单元格内容
            
            # 移除其他一些常见的Markdown标记，如标题符号、列表符号等
            interim_text = re.sub(r'^[#*->]+\s*', '', interim_text, flags=re.MULTILINE) # 标题, 列表, 引用
            interim_text = re.sub(r'^\s*[-=]{3,}\s*$', '', interim_text, flags=re.MULTILINE) # 水平线

            text_parts.append(interim_text.strip())

            # 3. 对 Markdown 中的图像进行 OCR 识别
            if include_images:
                if chosen_ocr_engine == "none":
                    print("信息: 未选择OCR引擎 (或默认引擎不可用)，跳过图像文本提取。")
                elif (chosen_ocr_engine == 'tesseract' and not self.tesseract_available) or \
                     (chosen_ocr_engine == 'paddle' and not PADDLEOCR_AVAILABLE):
                    print(f"警告: 选择的OCR引擎 '{chosen_ocr_engine}' 不可用。跳过图像文本提取。")
                else:
                    image_text_content = self._extract_text_from_markdown_images(
                        content, md_path, ocr_languages, chosen_ocr_engine
                    )
                    if image_text_content.strip():
                        text_parts.append(f"\n--- 来自图像的文本 ({chosen_ocr_engine}) ---\n{image_text_content.strip()}")

        except FileNotFoundError:
            raise FileNotFoundError(f"错误: Markdown文件未找到: {md_path}")
        except Exception as e:
            raise Exception(f"处理Markdown文件 '{md_path}' 时发生错误: {str(e)}")

        return "\n\n".join(filter(None, text_parts)).strip(), metadata_dict # 使用\n\n连接，过滤空部分

    def _parse_yaml_metadata_basic(self, yaml_text: str) -> Dict[str, Any]:
        """【私有方法】基础的YAML元数据解析，不支持复杂结构。"""
        meta = {}
        for line in yaml_text.splitlines():
            if ':' in line:
                key, val = line.split(':', 1)
                meta[key.strip()] = val.strip().strip('"').strip("'") # 移除可能的引号
        return meta

    def _ocr_image_with_tesseract(self, image_path: str, languages: str) -> str:
        """【私有方法】使用Tesseract对单个图像进行OCR。"""
        try:
            image = Image.open(image_path)
            # 确保图像是Tesseract可以处理的模式，例如RGB或L
            if image.mode == 'RGBA' or image.mode == 'P': # P是调色板模式
                image = image.convert('RGB')
            text = pytesseract.image_to_string(image, lang=languages)
            return text.strip()
        except pytesseract.TesseractError as te:
            print(f"  - Tesseract OCR错误 (图像: {os.path.basename(image_path)}): {te}")
        except Image.UnidentifiedImageError:
            print(f"  - Tesseract警告: 无法识别图像格式: {os.path.basename(image_path)}。")
        except Exception as e_img:
            print(f"  - Tesseract处理图像 {os.path.basename(image_path)} 时发生未知错误: {e_img}")
        return ""

    def _ocr_image_with_paddle(self, image_path: str, lang_for_paddle: str) -> str:
        """【私有方法】使用PaddleOCR对单个图像进行OCR。"""
        if not self._initialize_paddleocr(lang_for_paddle) or not self.paddle_ocr_instance:
            print(f"  - PaddleOCR初始化失败或不可用，无法处理图像: {os.path.basename(image_path)}")
            return ""
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode == 'RGBA' or pil_image.mode == 'P':
                pil_image = pil_image.convert('RGB')
            elif pil_image.mode == 'L':
                 pil_image = pil_image.convert('RGB')
            
            img_np = np.array(pil_image)
            ocr_results = self.paddle_ocr_instance.ocr(img_np, cls=True)
            
            image_texts = []
            if ocr_results and ocr_results[0] is not None:
                for line_info in ocr_results[0]:
                    text_content = line_info[1][0]
                    if text_content.strip():
                        image_texts.append(text_content.strip())
            return "\n".join(image_texts)
        except Image.UnidentifiedImageError:
            print(f"  - PaddleOCR警告: 无法识别图像格式: {os.path.basename(image_path)}。")
        except Exception as e_img:
            print(f"  - PaddleOCR处理图像 {os.path.basename(image_path)} 时发生错误: {e_img}")
        return ""

    def _extract_text_from_markdown_images(self, 
                                           md_content: str, 
                                           md_file_path: str, 
                                           ocr_languages: str = 'chi_sim+eng',
                                           ocr_engine: str = 'tesseract') -> str:
        """
        【私有方法】查找 Markdown 内容中的本地图像引用，并使用指定的OCR引擎提取文本。
        仅支持相对路径和绝对路径的本地图像，例如：
        ![描述](./relative/path/to/image.png)
        ![描述](/absolute/path/to/image.png)
        """
        image_text_parts = []
        # Markdown文件的目录，用于解析相对路径的图像
        markdown_file_directory = os.path.dirname(os.path.abspath(md_file_path))

        # 正则表达式查找 Markdown 图像语法: ![alt text](image_path "optional title")
        # 我们只关心 image_path
        image_path_matches = re.findall(r'!\[.*?\]\((.*?)(?:\s+".*?")?\)', md_content)
        
        print(f"OCR ({ocr_engine}): 在Markdown中找到 {len(image_path_matches)} 个图像引用。")
        processed_image_count = 0

        for img_index, raw_image_path in enumerate(image_path_matches):
            # 清理路径中的潜在URL编码或多余字符 (虽然通常不需要对本地路径做这个)
            image_path_cleaned = raw_image_path.strip()

            # 忽略URL链接的图像
            if image_path_cleaned.lower().startswith(("http://", "https://")):
                # print(f"  - 跳过远程图像: {image_path_cleaned}")
                continue

            # 解析图像的绝对路径
            if os.path.isabs(image_path_cleaned):
                absolute_image_path = image_path_cleaned
            else:
                # 将相对路径转换为绝对路径 (相对于Markdown文件所在目录)
                absolute_image_path = os.path.abspath(os.path.join(markdown_file_directory, image_path_cleaned))

            if not os.path.isfile(absolute_image_path):
                print(f"  - 警告: 图像文件未找到: {absolute_image_path} (原始路径: '{raw_image_path}')")
                continue
            
            processed_image_count +=1
            # print(f"  - 正在处理图像 {processed_image_count}: {os.path.basename(absolute_image_path)}")
            
            text_from_one_image = ""
            if ocr_engine == 'tesseract':
                if self.tesseract_available:
                    text_from_one_image = self._ocr_image_with_tesseract(absolute_image_path, ocr_languages)
                else:
                    print(f"  - Tesseract不可用，跳过图像: {os.path.basename(absolute_image_path)}")
            elif ocr_engine == 'paddle':
                if PADDLEOCR_AVAILABLE:
                    lang_for_paddle = self._map_tesseract_lang_to_paddle(ocr_languages)
                    text_from_one_image = self._ocr_image_with_paddle(absolute_image_path, lang_for_paddle)
                else:
                    print(f"  - PaddleOCR不可用，跳过图像: {os.path.basename(absolute_image_path)}")
            
            if text_from_one_image:
                image_text_parts.append(f"\n--- 图像 '{os.path.basename(absolute_image_path)}' (第 {img_index + 1} 个引用) ---\n{text_from_one_image}")
        
        if processed_image_count > 0:
            print(f"OCR ({ocr_engine}): 完成处理 {processed_image_count} 个本地图像。")
        elif len(image_path_matches) > 0:
             print(f"OCR ({ocr_engine}): 未找到可处理的本地图像。")


        return "\n".join(image_text_parts)

    def extract_tables(self, md_path: str) -> List[List[List[str]]]:
        """
        从 Markdown 文件中提取所有表格数据。
        表格被解析为三维列表: [表格索引][行索引][单元格文本]。
        包括表头行。

        Args:
            md_path (str): Markdown 文件的路径。

        Returns:
            List[List[List[str]]]: 提取的表格列表。
        """
        all_extracted_tables: List[List[List[str]]] = []
        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                lines = [line.rstrip('\n') for line in file.readlines()] # 读取时移除换行符

            current_line_index = 0
            num_lines = len(lines)
            
            print(f"表格提取: 正在扫描 {md_path} 以查找Markdown表格...")

            while current_line_index < num_lines:
                line = lines[current_line_index]

                # 简单判断是否可能是表格的表头或分隔行
                # 表格行通常包含'|'，分隔行主要由'|', '-', ':'构成
                if '|' not in line:
                    current_line_index += 1
                    continue

                # 检查下一行是否为分隔行 (例如: |---|---| 或 |:---|:---:|)
                if current_line_index + 1 < num_lines:
                    next_line = lines[current_line_index + 1]
                    # 正则表达式匹配Markdown表格的分隔行
                    # 允许可选的前导/尾随空格和可选的行首/行尾'|'
                    # 核心是单元格之间必须有'-'，可以包含':'用于对齐
                    is_separator_line = bool(re.match(r'^\s*\|?(:?-+:?\|)+:?-+:?\|?\s*$', next_line))

                    if is_separator_line:
                        # 找到了一个表格的开始 (表头 + 分隔行)
                        current_table: List[List[str]] = []
                        
                        # 1. 解析表头行
                        header_cells_text = self._parse_table_row(line)
                        if not any(cell.strip() for cell in header_cells_text): # 跳过空的表头行
                             current_line_index += 1
                             continue
                        current_table.append(header_cells_text)

                        # 2. 跳过分隔行，开始解析数据行
                        current_line_index += 2 
                        
                        while current_line_index < num_lines and '|' in lines[current_line_index]:
                            data_row_text = lines[current_line_index]
                            # 如果数据行看起来像另一个分隔符（不太可能但以防万一）或者完全是空的，则停止
                            if re.match(r'^\s*\|?(:?-+:?\|)+:?-+:?\|?\s*$', data_row_text) or not data_row_text.strip():
                                break
                            
                            data_cells_text = self._parse_table_row(data_row_text)
                            current_table.append(data_cells_text)
                            current_line_index += 1
                        
                        if current_table: # 确保表格不是空的
                            all_extracted_tables.append(current_table)
                            # print(f"  - 提取到1个表格，包含 {len(current_table)} 行。")
                        # 继续在文件的剩余部分查找下一个表格 (current_line_index 已经指向新位置)
                        continue # 直接进入下一次while循环的迭代
                
                current_line_index += 1 # 如果不是表格的开始，则前进到下一行

        except FileNotFoundError:
            raise FileNotFoundError(f"错误: Markdown文件未找到: {md_path}")
        except Exception as e:
            raise Exception(f"提取Markdown表格时发生错误 ({md_path}): {str(e)}")
        
        if all_extracted_tables:
            print(f"表格提取: 共提取到 {len(all_extracted_tables)} 个表格。")
        else:
            print(f"表格提取: 在 {md_path} 中未检测到符合格式的Markdown表格。")

        return all_extracted_tables

    def _parse_table_row(self, row_string: str) -> List[str]:
        """【私有方法】解析Markdown表格的一行，返回单元格文本列表。"""
        # 移除行首和行尾的'|' (如果存在) 和两端的空白
        cleaned_row = row_string.strip().strip('|').strip()
        # 按'|'分割单元格，并去除每个单元格两端的空白
        cells = [cell.strip() for cell in cleaned_row.split('|')]
        return cells