import torch
import cv2
from IPython.display import Image, clear_output
from PIL import Image

clear_output()

# 토치 사용 확인
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

# Model : 파이토치허브에서 바로 불러오는 방법 (커스텀 신경망도 가능)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS

# 이미지 로드
img1 = Image.open('skate.mp4')  # PIL image
imgs = [img1] # [img1, img2]  # batched list of images

# Inference
results = model(imgs, size=640)  # includes NMS : NMS (non-maximum-suppression)란 현재 픽셀을 기준으로 주변의 픽셀과 비교했을 때 최대값인 경우 그대로 놔두고, 아닐 경우(비 최대) 억제(제거)하는 것이다. 
# Results
results.print()  # print results to screen
results.show()  # display results
results.save()  # save as results1.jpg, results2.jpg... etc.

# Data
print('\n', results.xyxy[0])