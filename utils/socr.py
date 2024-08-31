from utils.idcardocr import IdCardOCR
from utils.detector import Detector
import cv2
from paddleocr import PaddleOCR
from pathlib import Path


class Socr:
    def __init__(self, model_path="weight/weight_1000/best.pt"):
        self.model_path = model_path
        self.detector = Detector(self.model_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, use_gpu=True)
        self.idcardocr = IdCardOCR(self.ocr)

    def imgocr(self, img_path):
        # 1. 图片检测，获取正反面结果
        out = self.detector.detect(img_path)
        assert out is not None, "图片检测失败"
        assert len(out) > 0, "图片检测失败"
        first = self.detector.frist_out(out)[0]  # 只返回正反面的结果, cls==1 或 cls==0

        # 2. 裁剪图片，并检测头像或国徽
        img = self.detector.crop_image(img_path, first['xyxy'])  # 裁剪后的图片
        out = self.detector.detect(img)
        assert out is not None, "图片检测失败"
        assert len(out) > 0, "图片检测失败"
        second = self.detector.second_out(out)[0]  # 只返回头像和国徽的结果, cls==2 或 cls==3

        # 3. 判断图片的正立或倒立，并再次检测
        img = self.detector.is_need_rotate(img, second['cls'], second['xyxy'])  # 返回正立的图片
        out = self.detector.detect(img)
        assert out is not None, "图片检测失败"
        assert len(out) > 0, "图片检测失败"
        second = self.detector.second_out(out)[0]
        first = self.detector.frist_out(out)[0]
        
        # 4. 绘制矩形框
        # img = self.detector.draw_rectangle(img, out[0]['xyxy'], color=(0, 0, 255), text=self.detector.getclsname(out[0]['cls']))    
        # img = self.detector.draw_rectangle(img, out[1]['xyxy'], color=(0, 255, 0), text=self.detector.getclsname(out[1]['cls']))

        # 5. 识别身份证信息
        mask = second['xyxy']  # 国徽的坐标
        info = self.idcardocr.extarct_info(img, first['cls'], mask)
        
        return info