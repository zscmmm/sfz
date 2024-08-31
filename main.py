from utils.socr import Socr

img_path = "assets/2_2.jpg"  #识别的图片路径
socr = Socr()
result = socr.imgocr(img_path)
print(result)

