
from ultralytics import YOLO
import os
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from loguru import logger

        
class Detector:
    @staticmethod
    def getclsname(cls) -> str:
        """
        将类别映射为名称
        """
        clsmap = {
            0: "fan",
            1: "zheng",
            2: "touxiang",
            3: "guohui",
        }

        return clsmap.get(cls, "unknown")

    def __init__(self, model_path):
        """
        初始化检测器
        """
        self.model = YOLO(model_path,  verbose=False)

    def calculate_rectangle_dimensions(self, coords: list[float]) -> bool:
        """
        计算矩形的宽度和高度, 如果宽度大于高度, 则返回 True, 否则返回 False
        coords 表示矩形的坐标, 是一个列表, 包含 4 个元素, 分别是 x1, y1, x2, y2, 表示左上角和右下角的坐标
        """
        assert isinstance(coords, list), "coords 参数必须是列表"
        assert len(coords) == 4, "coords 参数必须是 4 个元素的列表"
        x1, y1, x2, y2 = coords
        width = x2 - x1
        height = y2 - y1
        return width >= height
        
    def _outdict(self):
        """
        格式化输出
        """
        d = {
            "cls": None,  # int  目标类别
            "conf": None, # float , 置信度
            "xyxy": None, #list , 边界框坐标
            "xywh": None, #list, 边界框坐标
            "is_width": None, #bool 表示是否是宽矩形, 宽度大于高度
        }
        return d
    
    def _outfromat(self, d):
        """
        对输出进行格式化, 比如保留 2 位小数
        """
        # 如果元素是一个列表, 只有一个元素, 则转为元素
        if isinstance(d['cls'], list) and len(d['cls']) == 1:
            d['cls'] = int(d['cls'][0])

        if isinstance(d['conf'], list) and len(d['conf']) == 1:
            d['conf'] = round(d['conf'][0], 2)

        if isinstance(d['xyxy'], list) and len(d['xyxy']) == 1 and len(d['xyxy'][0]) == 4:
            d['xyxy'] = d['xyxy'][0]
        if isinstance(d['xywh'], list) and len(d['xywh']) == 1 and len(d['xywh'][0]) == 4:
            d['xywh'] = d['xywh'][0]

        d['conf'] = round(d['conf'], 2)
        d['xyxy'] = [round(i, 2) for i in d['xyxy']] 
        d['xywh'] = [round(i, 2) for i in d['xywh']]
        return d

    def  _set_d(self, d, cls, conf, xyxy, xywh, is_width):
        """
        设置输出
        """
        d['cls'] = cls
        d['conf'] = conf
        d['xyxy'] = xyxy
        d['xywh'] = xywh
        d['is_width'] = is_width
        return d

    
    def check_empty(self, out):
        """
        检查 out 的字段值是否有空值
        """
        for d in out:
            for k, v in d.items():
                if v is None:
                    logger.warning(f"key: {k}, value: {v}, 有空值, 路径: {self.source_path}")
        return out
    


    def check_prediction(self, out):
        """
        提取所有的 cls,
        """
        # 0. 对 conf 进行检查, 过滤掉置信度小于 0.7 的
        out = [d for d in out if d['conf'] >= 0.7]

        # 1. out 的长度应该为 2, 每个元素是一个字典, 即 d
        if len(out) != 2:
            logger.warning(f"预测错误, out 的长度应该为 2, 实际长度为 {len(out)}, 路径: {self.source_path}")

        # 2. 提取所有的 cls, cls 只能是 {0,3} 和 {1,2}
        cls = [d['cls'] for d in out]
        if not set(cls) in [{0,3}, {1,2}]:
            logger.warning(f"预测错误, cls: {cls}, 只能是 {0,3} 和 {1,2}, 路径: {self.source_path}")

        # 3. 检查是否有空值
        out = self.check_empty(out)
        return out
    
    def detect(self, source):
        """
        传入图片路径, 返回图片中的物体的类别和位置
        输入:
        - source: 图片路径, 也可以是 PIL.Image.Image 或 numpy.ndarray
        返回: list[dict], 每个元素是一个字典, 包含以下字段
        - cls: 目标类别
        - conf: 置信度
        - xyxy: 边界框坐标
        - xywh: 边界框坐标
        - is_width: 是否是宽矩形, 宽度大于高度
        """
        # 如果是路径, 则保存在 source_path 中
        if (isinstance(source, str) and len(source) <= 50) or isinstance(source, Path):
            self.source_path = source
        else:
            self.source_path = None
        results = self.model.predict(source, verbose=False)  

        out = []
        for result in results:
            boxes = result.boxes
            # boxes.cls.cpu().numpy().tolist()  # 类别 [1.0, 2.0]  # 如果有多个目标, 则是一个列表
            # boxes.xyxy.cpu().numpy().tolist() # 边界框坐标 [[x1, y1, x2, y2], [x1, y1, x2, y2]]
            # 因此, 需要遍历 boxes
            for box in boxes:
                d = self._outdict()
                cls = box.cls.cpu().numpy().tolist() 
                conf = box.conf.cpu().numpy().tolist()  
                xyxy = box.xyxy.cpu().numpy().tolist() 
                xywh = box.xywh.cpu().numpy().tolist() 
                # 必须都是非空的, 才能继续
                if cls and conf and xyxy and xywh:
                    is_width = self.calculate_rectangle_dimensions(xyxy[0])  # 是否是宽矩形, 宽度大于高度
                    d = self._set_d(d, cls, conf, xyxy, xywh, is_width)
                    d = self._outfromat(d)
                    out.append(d)
                else:
                    out.append(d)
        # 检查是否预测的正确
        out = self.check_prediction(out)
        return out

    def frist_out(self, out) -> list[dict]:
        """
        第一次预测, 即只要正反面的结果,
        只返回cls==1 或 cls==0 的结果, 返回的是一个list, list 只有一个元素,且是字典
        """
        out = [d for d in out if d['cls'] in [0, 1]]
        return out
    def second_out(self, out) -> list[dict]:
        """
        第二次预测, 即只要头像和国徽的结果,
        只返回cls==2 或 cls==3 的结果, 返回的是一个list, list 只有一个元素,且是字典
        """
        out = [d for d in out if d['cls'] in [2, 3]]
        return out
    
    def crop_image(self, img, size):
        """
        裁剪图片,根据给定的坐标, 返回裁剪后的图片
        size = [x1, y1, x2, y2], 表示裁剪的区域,左上角和右下角的坐标
        is_width: 是否是宽矩形, 如果是宽矩形, 则不需要旋转, 否则旋转 90 度 , 默认根据 size 来判断
        """
        # 检查 img 是路径还是图片
        if isinstance(img, str) or isinstance(img, Path):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise TypeError("img 参数必须是路径或者 np.ndarray 类型的图片")
        # 2,获取图片的高度和宽度
        h, w = img.shape[:2]

        # 3. 检查 size 参数
        assert isinstance(size, list), "size 参数必须是列表"
        assert len(size) == 4, "size 参数必须是 4 个元素的列表"
        x1, y1, x2, y2 = map(int, size)
        # 4. 对给定的大小进行扩展, 
        expand_n = max(1, min(int(min(w, h) * 0.02), 5))  # 扩展 2% 的像素, 但是不能小于 1, 也不能大于 5
        x1, y1 = max(0, x1 - expand_n), max(0, y1 - expand_n)
        x2, y2 = min(w, x2 + expand_n), min(h, y2 + expand_n)
        # 5. 裁剪
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img2 = img[y1:y2, x1:x2]
        # 5.1. 如果是不是宽矩形, 则旋转 90 度, 如何知道选择旋转 90 度还是 270 度呢?
        if not self.calculate_rectangle_dimensions(size):
            img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
        
        # 6. 调整大小 
        n = 4
        dst_width =  240 * n  # 提高分辨率，原为240
        dst_height = 151 * n # 提高分辨率，原为151
        img2 = cv2.resize(img2, (dst_width, dst_height), interpolation=cv2.INTER_CUBIC)
        return img2


    def draw_rectangle(self, img, size, color=(0, 255, 0), text="", thickness=2, font_scale=1.0, text_color=None):
        """
        在图像上绘制矩形并可选地添加文字。

        参数:
        - img: 要绘制的图像。
        - size: 定义矩形坐标的四个整数的列表 [x1, y1, x2, y2]。
        - color: 矩形颜色的 BGR 格式元组。默认为绿色 (0, 255, 0)。
        - text: 矩形内部要绘制的可选文字。默认为空字符串。
        - thickness: 矩形边框的厚度。默认值为 2。
        - font_scale: 文字大小的缩放因子。默认值为 1.0。
        - text_color: 文字颜色的 BGR 格式。如果未指定，默认为矩形颜色。

        返回:
        - 绘制了矩形（和可选文字）的图像。
        """
        if not isinstance(size, list):
            raise TypeError("参数 'size' 必须是列表。")
        if len(size) != 4:
            raise ValueError("参数 'size' 必须是包含四个元素的列表。")
        
        try:
            x1, y1, x2, y2 = map(int, size)
        except ValueError as e:
            raise ValueError("列表 'size' 中的所有元素必须可转换为整数。") from e

        # 画矩形
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 画文字
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            if text_color is None:
                text_color = color
            text_position = (x1, y1 - 10) if y1 - 10 > 10 else (x1, y1 + 20)
            cv2.putText(img, text, text_position, font, font_scale, text_color, thickness, cv2.LINE_AA)

        return img

    def is_need_rotate(self, img, cls, size):
        """
        判断是否需要旋转180度,
        - img: 图片
        - cls: 目标类别, 0: fan, 1: zheng, 2: touxiang, 3: guohui
        - size: 定义矩形坐标的四个整数的列表 [x1, y1, x2, y2]。
        返回, 旋转后的图片
        """
        # 0. 初步检查, 获取图片的高度和宽度, 必须是宽矩形, 宽度大于高度
        h, w = img.shape[:2]
        assert h < w, "图片的高度必须小于宽度"
        center_x = w // 2
        center_y = h // 2
        assert cls in [2, 3], "目标类别必须是 2 或 3"
        # 1. 要求输入 touxiang 和 guohui 的坐标
        x1, y1, x2, y2 = map(int, size)
        assert x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h, "坐标超出范围"
        if cls == 2:
            # 表示这个图片是正面, 头像
            cx1, cy1 = (x2 - x1)//2 + x1, (y2 - y1)//2 + y1    # 头像的中心坐标
            if cx1 < center_x:
                # 需要旋转180度
                img = cv2.rotate(img, cv2.ROTATE_180)

        else:
            # 表示这个图片是反面, 国徽
            cx1, cy1 = (x2 - x1)//2 + x1, (y2 - y1)//2 + y1    # 国徽的中心坐标
            if cx1 > center_x:
                # 需要旋转180度
                img = cv2.rotate(img, cv2.ROTATE_180)
        return img
    

