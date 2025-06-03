import fitz  # PyMuPDF
from typing import List, Tuple, Optional, Dict, Any
import re
from PIL import Image # 导入Pillow库
import pytesseract    # 导入pytesseract
import io             # 导入io用于处理字节流
import numpy as np    # 导入numpy用于图像数组转换 (PaddleOCR需要)

# 尝试导入PaddleOCR，并处理未安装的情况
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True  # PaddleOCR可用标志
except ImportError:
    PADDLEOCR_AVAILABLE = False # PaddleOCR不可用标志
    PaddleOCR = None # 定义PaddleOCR为None，以避免后续代码出现NameError

class PDFProcessor:
    """
    使用Fitz(PyMuPDF)实现的PDF文件处理器，支持Tesseract和PaddleOCR进行图像文字识别。
    """
    def __init__(self,
                 tesseract_cmd_path: Optional[str] = None,
                 default_ocr_engine: str = 'paddle', # 默认OCR引擎
                 paddle_ocr_config: Optional[Dict[str, Any]] = None):
        """
        初始化PDFProcessor

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
            print("请确保已安装Tesseract OCR并将其添加到系统PATH，或在初始化PDFProcessor时提供其路径。")
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
        """惰性初始化PaddleOCR实例"""
        if not PADDLEOCR_AVAILABLE:
            print("警告: PaddleOCR库未安装，无法使用PaddleOCR引擎。")
            return False

        # 如果用户提供了完整的配置，并且我们尚未初始化，或者语言配置发生变化
        if self.user_paddle_ocr_config:
            # 检查是否需要重新初始化 (实例不存在，或配置中的语言与当前初始化的语言不同)
            # 注意: 如果用户配置中没有'lang'，则认为不需要基于语言的重新初始化
            should_reinitialize_user_config = self.paddle_ocr_instance is None
            if 'lang' in self.user_paddle_ocr_config:
                 should_reinitialize_user_config = should_reinitialize_user_config or \
                    (self.current_paddle_lang_init != self.user_paddle_ocr_config.get('lang'))

            if should_reinitialize_user_config:
                try:
                    print(f"正在使用用户配置初始化PaddleOCR: {self.user_paddle_ocr_config}...")
                    self.paddle_ocr_instance = PaddleOCR(**self.user_paddle_ocr_config)
                    self.current_paddle_lang_init = self.user_paddle_ocr_config.get('lang', 'ch') # 默认'ch'
                    print("PaddleOCR已通过用户配置成功初始化。")
                    return True
                except Exception as e:
                    print(f"警告: 使用用户配置初始化PaddleOCR失败: {e}")
                    self.paddle_ocr_instance = None
                    return False
            return True # 已使用用户配置初始化

        # 如果没有用户配置，或者实例不存在，或者基于参数推断的语言发生变化
        if self.paddle_ocr_instance is None or self.current_paddle_lang_init != lang_for_paddle:
            try:
                # 默认配置，如果用户未指定
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
        return True # 已正确初始化

    def _map_tesseract_lang_to_paddle(self, tesseract_langs: str) -> str:
        """
        启发式地将Tesseract语言字符串映射到PaddleOCR语言代码。
        PaddleOCR的 'ch' 模型通常也能很好地处理英文。

        Args:
            tesseract_langs (str): Tesseract的语言字符串，例如 'chi_sim+eng'.

        Returns:
            str: PaddleOCR的语言代码，例如 'ch', 'en'.
        """
        # 如果用户在初始化时指定了paddle_ocr_config并包含lang，则优先使用该配置
        if self.user_paddle_ocr_config and 'lang' in self.user_paddle_ocr_config:
            return self.user_paddle_ocr_config['lang']

        t_langs = tesseract_langs.lower()
        if 'chi_sim' in t_langs or 'chi_tra' in t_langs:
            return 'ch'  # PaddleOCR 中文模型通常包含简体和繁体，并且能处理中英混合
        if 'eng' in t_langs:
            return 'en'
        if 'kor' in t_langs:
            return 'korean'
        if 'japan' in t_langs:
            return 'japan'
        # 可以根据需要添加更多语言的映射
        print(f"警告: 无法将Tesseract语言 '{tesseract_langs}'直接映射到PaddleOCR支持的语言。将默认使用 'ch'。")
        return 'ch' # 一个通用的默认值，因为中文模型通常也能处理一些英文

    def extract_text_from_pdf(self, pdf_path: str,
                           include_metadata: bool = False,
                           include_images: bool = False,
                           ocr_languages: str = 'chi_sim+eng', # Tesseract和PaddleOCR都尝试使用
                           ocr_engine: Optional[str] = None) -> Tuple[str, Optional[dict]]:
        """
        从PDF文件中提取文本内容。

        Args:
            pdf_path (str): PDF文件路径。
            include_metadata (bool): 是否在返回结果中包含PDF的元数据。
            include_images (bool): 是否尝试从PDF中的图像提取文本（需要OCR支持）。
            ocr_languages (str): OCR识别时使用的语言。
                                 - 对于Tesseract: 例如 'eng' (英语), 'chi_sim' (简体中文), 'eng+chi_sim' (英语和简体中文)。
                                 - 对于PaddleOCR: 此参数会尝试映射到PaddleOCR支持的语言代码 (如 'ch', 'en')。
                                                  更精确的控制可以通过在初始化PDFProcessor时传入 `paddle_ocr_config` 实现。
            ocr_engine (Optional[str]): 指定使用的OCR引擎。可以是 'tesseract' 或 'paddle'。
                                        如果为 None，则使用初始化时设置的 `default_ocr_engine`。

        Returns:
            Tuple[str, Optional[dict]]: 一个元组，包含提取的文本字符串和元数据字典 (如果 `include_metadata` 为True)。
        """
        text_parts = []
        metadata = None
        chosen_ocr_engine = (ocr_engine or self.default_ocr_engine or "none").lower()

        try:
            with fitz.open(pdf_path) as doc:
                # 1. 提取原生文本
                for page_num, page in enumerate(doc):
                    page_text = page.get_text("text")
                    if page_text:
                        text_parts.append(page_text)

                # 2. 提取元数据
                if include_metadata:
                    metadata = doc.metadata

                # 3. 从图像中提取文本 (如果需要)
                if include_images:
                    if chosen_ocr_engine == 'tesseract':
                        if self.tesseract_available:
                            try:
                                image_text = self._extract_text_from_images_tesseract(doc, languages=ocr_languages)
                                if image_text:
                                    text_parts.append(image_text)
                            except pytesseract.TesseractNotFoundError: # 这个异常理论上在init时已捕获，但双重检查
                                print(f"警告: Tesseract OCR 未找到或配置不正确，无法从 {pdf_path} 的图像中提取文本。")
                            except Exception as e_ocr:
                                print(f"警告: 使用Tesseract从 {pdf_path} 的图像中提取文本时发生错误: {e_ocr}")
                        else:
                            print("警告: Tesseract被选为OCR引擎，但不可用。跳过图像文本提取。")
                    elif chosen_ocr_engine == 'paddle':
                        if PADDLEOCR_AVAILABLE: # 再次检查，因为可能在init后卸载了
                            try:
                                image_text = self._extract_text_from_images_paddle(doc, tesseract_fallback_lang=ocr_languages)
                                if image_text:
                                    text_parts.append(image_text)
                            except Exception as e_ocr:
                                print(f"警告: 使用PaddleOCR从 {pdf_path} 的图像中提取文本时发生错误: {e_ocr}")
                        else:
                            print("警告: PaddleOCR被选为OCR引擎，但PADDLEOCR_AVAILABLE为False (可能未安装或初始化失败)。跳过图像文本提取。")
                    elif chosen_ocr_engine == "none":
                         print("信息: 未选择OCR引擎 (或默认引擎不可用)，跳过图像文本提取。")
                    else:
                        print(f"警告: 未知的OCR引擎 '{chosen_ocr_engine}'。跳过图像文本提取。")

        except Exception as e:
            raise Exception(f"处理PDF文件 {pdf_path} 时发生一般错误: {str(e)}")

        full_text = "\n".join(text_parts).strip()

        if include_metadata:
            return full_text, metadata
        return full_text, None

    def _extract_text_from_images_tesseract(self, doc: fitz.Document, languages: str = 'chi_sim+eng') -> str:
        """
        【私有方法】使用Tesseract从PDF文档中的图像提取文本。

        Args:
            doc (fitz.Document): PyMuPDF的文档对象。
            languages (str): Tesseract OCR识别时使用的语言。

        Returns:
            str: 从所有图像中提取并合并的文本。
        """
        if not self.tesseract_available:
            # 此检查主要用于内部调用，外部调用应由 extract_text_from_pdf 处理
            print("Tesseract OCR不可用，无法从图像提取文本。")
            return ""
        
        image_text_parts = []
        print(f"Tesseract OCR: 正在扫描 {len(doc)} 个页面以查找图像...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            if not image_list:
                continue
            # print(f"Tesseract OCR: 页面 {page_num + 1} 发现 {len(image_list)} 张图像。")

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]  # 图像的交叉引用号
                base_image = doc.extract_image(xref) # 提取原始图像数据
                if not base_image or not base_image.get("image"):
                    # print(f"  - Tesseract警告: 无法提取图像 {img_index + 1} (页面 {page_num + 1}) 的数据。")
                    continue
                image_bytes = base_image["image"] # 图像的字节数据

                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    # 使用pytesseract进行OCR
                    text_from_image = pytesseract.image_to_string(pil_image, lang=languages)

                    if text_from_image.strip():
                        # 可以选择添加标记来区分图像文本来源
                        # image_text_parts.append(f"\n--- Tesseract识别自图像 {img_index + 1} (页面 {page_num + 1}) ---\n")
                        image_text_parts.append(text_from_image.strip())
                        # image_text_parts.append("\n--- 图像文本结束 ---\n")
                except pytesseract.TesseractError as te:
                    print(f"  - Tesseract警告: 处理图像 {img_index + 1} (页面 {page_num + 1}) 时Tesseract发生错误: {te}")
                except Image.UnidentifiedImageError:
                    print(f"  - Tesseract警告: 无法识别图像 {img_index + 1} (页面 {page_num + 1}) 的格式。跳过。")
                except Exception as e:
                    print(f"  - Tesseract警告: OCR处理图像 {img_index + 1} (页面 {page_num + 1}) 时发生未知错误: {e}")
        
        return "\n".join(image_text_parts)

    def _extract_text_from_images_paddle(self, doc: fitz.Document, tesseract_fallback_lang: str = 'chi_sim+eng') -> str:
        """
        【私有方法】使用PaddleOCR从PDF文档中的图像提取文本。

        Args:
            doc (fitz.Document): PyMuPDF的文档对象。
            tesseract_fallback_lang (str): 用于映射到PaddleOCR语言的Tesseract语言字符串。

        Returns:
            str: 从所有图像中提取并合并的文本。
        """
        lang_for_paddle = self._map_tesseract_lang_to_paddle(tesseract_fallback_lang)
        if not self._initialize_paddleocr(lang_for_paddle) or not self.paddle_ocr_instance:
            print("PaddleOCR不可用或初始化失败，无法从图像提取文本。")
            return ""

        image_text_parts = []
        print(f"PaddleOCR: 正在扫描 {len(doc)} 个页面以查找图像...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            if not image_list:
                continue
            # print(f"PaddleOCR: 页面 {page_num + 1} 发现 {len(image_list)} 张图像。")

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if not base_image or not base_image.get("image"):
                    # print(f"  - PaddleOCR警告: 无法提取图像 {img_index + 1} (页面 {page_num + 1}) 的数据。")
                    continue
                image_bytes = base_image["image"]

                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    # PaddleOCR需要NumPy数组。PIL图像是RGB，PaddleOCR通常可以处理。
                    # 确保图像是RGB，如果PaddleOCR模型有特定要求（例如，某些模型可能在BGR上训练）
                    # 但通常直接传递RGB的NumPy数组即可。
                    if pil_image.mode == 'RGBA':
                        pil_image = pil_image.convert('RGB') # 转换为RGB，去除Alpha通道
                    elif pil_image.mode == 'P': # 调色板模式
                        pil_image = pil_image.convert('RGB') # 转换为RGB
                    elif pil_image.mode == 'L': # 灰度图
                         pil_image = pil_image.convert('RGB') # 转换为RGB，因为很多模型期望3通道

                    img_np = np.array(pil_image) # 将PIL图像转换为NumPy数组
                    
                    # PaddleOCR的ocr方法返回一个列表，每个图像一个结果列表。
                    # 对于单张图片，结果是 ocr_results[0]
                    # ocr_results[0] 是一个包含多个检测框信息的列表，例如:
                    # [
                    #   [[box_points_1], (text_1, confidence_1)],
                    #   [[box_points_2], (text_2, confidence_2)],
                    #   ...
                    # ]
                    # 对于 paddleocr >= 2.7, 单张图片的输出结构为:
                    # [ [detection_result_1, detection_result_2, ...] ]
                    # 即一个列表，包含一个子列表，子列表里是所有检测结果。
                    # cls=True 启用文本方向分类，有助于提高识别准确率
                    ocr_results = self.paddle_ocr_instance.ocr(img_np, cls=True) 
                    
                    page_image_texts = []
                    # 检查是否有结果，并且结果不为None
                    if ocr_results and ocr_results[0] is not None:
                        for line_info in ocr_results[0]: # 遍历图像中的每一个检测到的文本行
                            # line_info 结构通常是: [[box_points], (text, confidence)]
                            text_content = line_info[1][0] # 提取文本内容
                            if text_content.strip():
                                page_image_texts.append(text_content.strip())
                    
                    if page_image_texts:
                        # 可以选择添加标记
                        # image_text_parts.append(f"\n--- PaddleOCR识别自图像 {img_index + 1} (页面 {page_num + 1}) ---\n")
                        image_text_parts.append("\n".join(page_image_texts)) # 将同一张图片内的多行文本合并
                        # image_text_parts.append("\n--- 图像文本结束 ---\n")

                except Image.UnidentifiedImageError:
                    print(f"  - PaddleOCR警告: 无法识别图像 {img_index + 1} (页面 {page_num + 1}) 的格式。跳过。")
                except Exception as e:
                    print(f"  - PaddleOCR警告: OCR处理图像 {img_index + 1} (页面 {page_num + 1}) 时发生错误: {e}")
        
        return "\n".join(image_text_parts)



    def extract_tables(self, pdf_path: str) -> List[List[List[str]]]:
        """
        从PDF文件中提取表格数据。

        Args:
            pdf_path (str): PDF文件路径。

        Returns:
            List[List[List[str]]]: 一个三维列表，结构为 [表格索引][行索引][单元格文本]。
                                   所有页面上的表格会被平铺到一个列表中。
        """
        all_tables_extracted = [] 
        try:
            with fitz.open(pdf_path) as doc:
                print(f"表格提取: 正在扫描 {len(doc)} 个页面以查找表格...")
                for page_num, page in enumerate(doc):
                    # page.find_tables() 返回一个 TableFinder 对象
                    # TableFinder.tables 是一个表格对象列表 (fitz.table.Table)
                    page_tables_finder = page.find_tables()
                    if page_tables_finder.tables: # 检查当前页面是否找到了表格
                        # print(f"  页面 {page_num + 1}: 找到 {len(page_tables_finder.tables)} 个表格。")
                        for table_obj in page_tables_finder.tables:
                            # table.extract() 返回一个列表的列表，代表表格的行和单元格
                            # 例如: [[cell_11, cell_12, ...], [cell_21, cell_22, ...], ...]
                            # 单元格内容可能是字符串，也可能是 None (如果单元格为空或是合并单元格的一部分)
                            extracted_table_data = table_obj.extract()
                            # 清理None值，将其替换为空字符串，或根据需要进行其他处理
                            cleaned_table = [
                                [str(cell) if cell is not None else "" for cell in row] 
                                for row in extracted_table_data
                            ]
                            all_tables_extracted.append(cleaned_table)
        except Exception as e:
            raise Exception(f"从PDF提取表格时发生错误: {str(e)}")
        if not all_tables_extracted:
            print("  未在PDF中检测到表格。")
        return all_tables_extracted


