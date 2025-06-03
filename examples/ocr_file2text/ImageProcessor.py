import os
import io
from typing import Tuple, Optional, Dict, Any

from PIL import Image # Pillow 用于图像处理
import pytesseract     # Tesseract OCR
import numpy as np     # PaddleOCR 需要 numpy

# 尝试导入PaddleOCR并设置标志
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True  # PaddleOCR可用标志
except ImportError:
    PADDLEOCR_AVAILABLE = False # PaddleOCR不可用标志
    PaddleOCR = None # 定义PaddleOCR为None，以避免后续代码出现NameError

class ImageProcessor:
    """
    处理图像文件（.png, .jpg, .jpeg, .bmp, .tiff），使用OCR提取文本。
    支持 Tesseract 和 PaddleOCR 引擎。
    """

    def __init__(self,
                 tesseract_cmd_path: Optional[str] = None,
                 default_ocr_engine: str = 'paddle', # 默认OCR引擎
                 paddle_ocr_config: Optional[Dict[str, Any]] = None):
        """
        初始化 ImageProcessor。

        Args:
            tesseract_cmd_path (Optional[str]): Tesseract OCR 的可执行文件路径。
                                                如果 Tesseract 已在系统 PATH 中，则无需设置。
            default_ocr_engine (str): 默认使用的OCR引擎 ('tesseract' 或 'paddle')。
            paddle_ocr_config (Optional[Dict[str, Any]]): PaddleOCR 的初始化配置。
                例如: {'use_angle_cls': True, 'lang': 'ch', 'show_log': False}
        """
        self.tesseract_available = False
        if tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
        except pytesseract.TesseractNotFoundError:
            print("警告: Tesseract OCR 未找到或未正确配置。"
                  "如果选择Tesseract，图像文本提取功能将不可用。")
        except Exception as e:
            print(f"警告: 初始化Tesseract时发生错误: {e}。"
                  "Tesseract图像文本提取功能可能受影响。")

        self.default_ocr_engine = default_ocr_engine.lower()
        if self.default_ocr_engine == 'paddle' and not PADDLEOCR_AVAILABLE:
            print("警告: PaddleOCR库未安装，但被选为默认OCR引擎。"
                  "将尝试回退到Tesseract (如果可用)，否则不进行图像OCR。")
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

        # 如果用户提供了完整的配置
        if self.user_paddle_ocr_config:
            # 检查是否需要重新初始化 (实例不存在，或配置中的语言与当前初始化的语言不同)
            should_reinitialize_user_config = self.paddle_ocr_instance is None
            if 'lang' in self.user_paddle_ocr_config: # 如果用户配置了语言
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
            return True # 已使用用户配置初始化，且语言匹配（或用户配置中无语言）

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
        """【私有方法】启发式地将Tesseract语言字符串映射到PaddleOCR语言代码。"""
        # 如果用户在初始化时指定了paddle_ocr_config并包含lang，则优先使用该配置
        if self.user_paddle_ocr_config and 'lang' in self.user_paddle_ocr_config:
            return self.user_paddle_ocr_config['lang']
        
        t_langs = tesseract_langs.lower()
        if 'chi_sim' in t_langs or 'chi_tra' in t_langs: return 'ch'
        if 'eng' in t_langs: return 'en'
        if 'kor' in t_langs: return 'korean' # 韩语
        if 'japan' in t_langs: return 'japan' # 日语
        # 可以根据需要添加更多语言的映射
        print(f"警告: 无法将Tesseract语言 '{tesseract_langs}' 直接映射到PaddleOCR支持的语言。"
              f"将默认使用 'ch'。")
        return 'ch' # 一个通用的默认值

    def _get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """【私有方法】获取图像文件的基本元数据。"""
        file_stats = os.stat(image_path)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                img_format = img.format
                mode = img.mode
        except Exception: # 如果无法打开图像获取详细信息
            width, height, img_format, mode = None, None, None, None

        return {
            "文件名称": os.path.basename(image_path),
            "文件大小_字节": file_stats.st_size,
            "创建时间_时间戳": file_stats.st_ctime,
            "修改时间_时间戳": file_stats.st_mtime,
            "图像宽度_像素": width,
            "图像高度_像素": height,
            "图像格式": img_format, # 例如 'PNG', 'JPEG'
            "图像模式": mode,      # 例如 'RGB', 'L' (灰度), 'RGBA'
        }

    def _ocr_with_tesseract_from_bytes(self, image_bytes: bytes, languages: str, image_id: str) -> str:
        """【私有方法】使用Tesseract从图像字节流进行OCR。"""
        if not self.tesseract_available:
            print(f"Tesseract OCR 不可用。无法处理图像 {image_id}。")
            return ""
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            # Tesseract 通常更喜欢简单格式，如RGB
            if pil_image.mode == 'RGBA' or pil_image.mode == 'P': # RGBA带Alpha通道, P是调色板模式
                pil_image = pil_image.convert('RGB')
            text = pytesseract.image_to_string(pil_image, lang=languages)
            return text.strip()
        except pytesseract.TesseractError as te:
            print(f"  - Tesseract OCR错误 (图像: {image_id}): {te}")
        except Image.UnidentifiedImageError: # Pillow无法识别图像格式
            print(f"  - Tesseract警告: 无法识别图像格式 (图像: {image_id})。")
        except Exception as e_img: # 其他处理图像时的未知错误
            print(f"  - Tesseract处理图像 (图像: {image_id}) 时发生未知错误: {e_img}")
        return ""

    def _ocr_with_paddle_from_bytes(self, image_bytes: bytes, lang_for_paddle: str, image_id: str) -> str:
        """【私有方法】使用PaddleOCR从图像字节流进行OCR。"""
        if not self._initialize_paddleocr(lang_for_paddle) or not self.paddle_ocr_instance:
            print(f"PaddleOCR初始化失败或实例不可用。无法处理图像 {image_id}。")
            return ""
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            # 确保图像是PaddleOCR能很好处理的格式 (例如RGB)
            if pil_image.mode == 'RGBA' or pil_image.mode == 'P':
                pil_image = pil_image.convert('RGB')
            elif pil_image.mode == 'L': # 灰度图
                 pil_image = pil_image.convert('RGB') # 很多模型期望3通道输入

            img_np = np.array(pil_image) # PaddleOCR需要NumPy数组
            # cls=True 启用文本方向分类，有助于提高识别准确率
            ocr_results = self.paddle_ocr_instance.ocr(img_np, cls=True)

            image_texts = []
            # 检查是否有结果，并且结果不为None (ocr_results[0]对应单张图片的结果)
            if ocr_results and ocr_results[0] is not None:
                for line_info in ocr_results[0]: # 遍历图像中的每一个检测到的文本行
                                                  # line_info 结构: [[box_points], (text, confidence)]
                    text_content = line_info[1][0] # 文本内容在元组的第一个元素
                    if text_content.strip():
                        image_texts.append(text_content.strip())
            return "\n".join(image_texts)
        except Image.UnidentifiedImageError:
            print(f"  - PaddleOCR警告: 无法识别图像格式 (图像: {image_id})。")
        except Exception as e_img:
            print(f"  - PaddleOCR处理图像 (图像: {image_id}) 时发生错误: {e_img}")
        return ""

    def extract_text_from_image(self,
                                image_path: str,
                                include_metadata: bool = False,
                                ocr_languages: str = 'chi_sim+eng', # Tesseract和PaddleOCR都尝试使用
                                ocr_engine: Optional[str] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        从单个图像文件中提取文本内容。

        Args:
            image_path (str): 图像文件路径。
            include_metadata (bool): 是否在返回结果中包含图像的元数据。
            ocr_languages (str): OCR识别时使用的语言。
                                 - 对于Tesseract: 例如 'eng' (英语), 'chi_sim' (简体中文)。
                                 - 对于PaddleOCR: 此参数会尝试映射到PaddleOCR支持的语言代码。
                                                  更精确控制可通过初始化时传入 `paddle_ocr_config`。
            ocr_engine (Optional[str]): 指定使用的OCR引擎 ('tesseract' 或 'paddle')。
                                        如果为 None，则使用初始化时设置的 `default_ocr_engine`。

        Returns:
            Tuple[str, Optional[Dict[str, Any]]]: 一个元组，包含 (提取的文本字符串, 元数据字典或None)。
        """
        text_content = ""
        metadata_dict: Optional[Dict[str, Any]] = None
        chosen_ocr_engine = (ocr_engine or self.default_ocr_engine or "none").lower()
        image_id = os.path.basename(image_path) # 用于日志记录

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件未找到: {image_path}")

        try:
            # 1. 提取元数据 (如果需要)
            if include_metadata:
                metadata_dict = self._get_image_metadata(image_path)

            # 2. 读取图像字节
            with open(image_path, "rb") as f_img:
                image_bytes = f_img.read()

            # 3. 执行OCR
            if chosen_ocr_engine == "none":
                print(f"信息: 未选择OCR引擎 (或默认引擎不可用) 处理图像 {image_id}。跳过文本提取。")
            elif chosen_ocr_engine == 'tesseract':
                if self.tesseract_available:
                    print(f"OCR ({chosen_ocr_engine}): 正在使用Tesseract处理图像 {image_id} (语言: {ocr_languages})...")
                    text_content = self._ocr_with_tesseract_from_bytes(image_bytes, ocr_languages, image_id)
                else:
                    print(f"警告: Tesseract被选为OCR引擎，但不可用。跳过图像 {image_id} 的OCR处理。")
            elif chosen_ocr_engine == 'paddle':
                if PADDLEOCR_AVAILABLE: # 再次检查，因为可能在init后卸载了
                    lang_for_paddle = self._map_tesseract_lang_to_paddle(ocr_languages)
                    print(f"OCR ({chosen_ocr_engine}): 正在使用PaddleOCR处理图像 {image_id} (映射语言: {lang_for_paddle})...")
                    text_content = self._ocr_with_paddle_from_bytes(image_bytes, lang_for_paddle, image_id)
                else:
                    print(f"警告: PaddleOCR被选为OCR引擎，但不可用。跳过图像 {image_id} 的OCR处理。")
            else:
                print(f"警告: 指定了未知的OCR引擎 '{chosen_ocr_engine}' 处理图像 {image_id}。跳过OCR处理。")
            
            if text_content:
                print(f"OCR ({chosen_ocr_engine}): 已从图像 {image_id} 提取文本。")
            elif chosen_ocr_engine not in ["none", None]: # 如果尝试了OCR但没结果
                 print(f"OCR ({chosen_ocr_engine}): 未能从图像 {image_id} 提取文本，或引擎处理失败。")


        except FileNotFoundError: # 已在开头检查，但作为安全措施
            raise
        except Image.UnidentifiedImageError: # Pillow无法识别图像
            raise ValueError(f"无法识别图像文件格式或图像已损坏: {image_path}")
        except Exception as e:
            raise Exception(f"处理图像文件 '{image_path}' 时发生错误: {str(e)}")

        return text_content.strip(), metadata_dict

