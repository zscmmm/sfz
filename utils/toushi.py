"""
由于裁剪出来的身份证图片是倾斜的，所以需要对其进行透视变换，将其矫正为正矩形。
基本思想, 通过轮廓检测找到身份证的四个角，然后计算透视变换矩阵，将旋转矩形矫正为正矩形。
- 其中,四个顶点坐标通过 minAreaRect 找到最小旋转矩形，然后获取旋转矩形的顶点坐标。
"""

import cv2
import numpy as np
import copy

class IDcardTouShi:
    def __init__(self, image_path):
        """
        Initialize ImageProcessor with an image path.
        """
        self.image = self._load_image(image_path)
        self.original_image = self.image.copy()

    def _load_image(self, image):
        if isinstance(image, str) and image:
            image = cv2.imread(image)
            self.image_path = image
            if image is None:
                raise ValueError(f"无法加载图像，可能路径无效")
            return image
        elif isinstance(image, np.ndarray):
            # 如果是 cv2 图像对象
            return image
        else:
            raise ValueError(f"无效的输入，必须是文件路径或cv2图像对象")

    def image_preprocess(self, image):
        """
        预处理图像，返回包含身份证区域的图像、四个顶点坐标和旋转矩形。
        """
        # image = self.image
        h, w = image.shape[:2]
        # 图像预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度图像
        canny = cv2.Canny(gray, 50, 150, apertureSize=3)  # 边缘检测
        # 膨胀
        kernel = np.ones((3, 3), np.uint8)
        canny = cv2.dilate(canny, kernel, iterations=3)
        # 闭运算
        kernel = np.ones((3, 3), np.uint8)
        canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
        # 通过轮廓检测找到身份证的四个角
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 合并所有轮廓点到一个数组中
        all_points = np.vstack(contours).squeeze()

        # 使用 minAreaRect 找到最小旋转矩形
        rect = cv2.minAreaRect(all_points)
        # print(f"旋转矩形: {rect}") # ((x, y), (w, h), angle)
        # 获取旋转矩形的顶点坐标
        box = cv2.boxPoints(rect)
        box = np.intp(box) # 返回的是四个点的坐标: 形如 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]
        box = self.order_points(box)  # 对顶点坐标进行排序, 顺序为: top-left, top-right, bottom-right, bottom-left
        
        # 调整顶点坐标确保在图像内部
        box[:, 0] = np.clip(box[:, 0], 0, w - 1) #限制 box 的 x 坐标在 0 到 w-1 之间
        box[:, 1] = np.clip(box[:, 1], 0, h - 1) # 限制 box 的 y 坐标在 0 到 h-1 之间
        
        # 画出包含所有轮廓的最小矩形
        # image = cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        return image, box

    def order_points(self, pts):
        """
        Orders the points in the order: top-left, top-right, bottom-right, bottom-left.
        """
        rect = np.zeros((4, 2), dtype="float32")

        # sum the (x, y) coordinates and sort based on the sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        # compute the difference between (x, y) coordinates and sort based on the difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect
    
    @staticmethod
    def perspective_transform(image, box):
        """
        透视变换，将旋转矩形矫正为正矩形。
        """
        # 计算模板图像的宽度和高度 (保持宽度不变)
        height, width = image.shape[:2]

        # 定义目标矩形的顶点坐标
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(np.float32(box), dst_pts)

        # 应用透视变换，将旋转矩形矫正为正矩形
        warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_CUBIC)
        
        return warped

    def transform_image(self, output_path: str = None):
        """
        保存矫正后的图像。
        """
        # 预处理图像
        image, box = self.image_preprocess(self.image)
        warped = self.perspective_transform(image, box)
        if output_path:
            cv2.imwrite(output_path, warped)
            return None
        else:
            return warped
    


if __name__ == "__main__":
    image_path = "adjusted_image.jpg"
    output_path = 'output_image.jpg'
    toushi = IDcardTouShi(image_path)
    toushi.transform_image(output_path)
