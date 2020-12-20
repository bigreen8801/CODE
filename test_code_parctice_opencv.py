import os
import test_def

import cv2
import numpy as np
import pandas as pd
## 1강 테스트, 이미지 불러오기
# img = cv2.imread('OpenCV_study_root\Resource\dog.jpg')
# cv2.imshow('dog',img)
# test_def.safe_run_imshow()

## 2강, 내장 카메라 또는 외장 카메라 이미지 프레임 재생
# capture = cv2.VideoCapture(1) # 0: 내장 카메라, 1~n : 외장 카메라
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# while True:
#     ret, frame = capture.read()
#     cv2.imshow("VideoFrame", frame)
#     if cv2.waitKey(1) == ord('q'): break

# capture.release() # 카메라에서 받아온 메모리 해제
# cv2.destroyAllWindows()

# 3강, image 출력
# img = cv2.imread('drone_detect_root\Resource_data_in\data_practice\dog.jpg', cv2.IMREAD_UNCHANGED)
# # mode
# # cv2.IMREAD_UNCHANGED : 원본 사용
# # cv2.IMREAD_GRAYSCALE : 1 채널, 그레이스케일 적용
# # cv2.IMREAD_COLOR : 3 채널, BGR 이미지 사용
# # cv2.IMREAD_ANYDEPTH : 이미지에 따라 정밀도를 16/32비트 또는 8비트로 사용
# # cv2.IMREAD_ANYCOLOR : 가능한 3 채널, 색상 이미지로 사용
# # cv2.IMREAD_REDUCED_GRAYSCALE_2 : 1 채널, 1/2 크기, 그레이스케일 적용
# # cv2.IMREAD_REDUCED_GRAYSCALE_4 : 1 채널, 1/4 크기, 그레이스케일 적용
# # cv2.IMREAD_REDUCED_GRAYSCALE_8 : 1 채널, 1/8 크기, 그레이스케일 적용
# # cv2.IMREAD_REDUCED_COLOR_2 : 3 채널, 1/2 크기, BGR 이미지 사용
# # cv2.IMREAD_REDUCED_COLOR_4 : 3 채널, 1/4 크기, BGR 이미지 사용
# # cv2.IMREAD_REDUCED_COLOR_8 : 3 채널, 1/8 크기, BGR 이미지 사용
# cv2.imshow('dog',img)
# test_def.safe_run_imshow()

# height, width, channel = img.shape # 이미지 사이즈 파악
# print(height, width , channel)

## 4강, VIDEO 출력

# # 동영상 프레임 가져옴
# video_filename = 'OpenCV_study_root\Resource\Andy Anderson_ a Short Skate Film.mp4'
# capture = cv2.VideoCapture(video_filename)
# while True:
#     # 현재 프레임 갯수랑 총 프레임 속성.갯수 를 가져와서 끝까지 비디오가 다 돌면 다시 재생
#     if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
#         capture.open(video_filename)

#     ret, frame = capture.read()
#     cv2.imshow("VideoFrame", frame)

#     if cv2.waitKey(33) > 0: break # 33ms마다 프레임을 재생하도록 함, 어떤 키라도 누르면 종료

# capture.release()
# # capture의 맴버
# # capture.get(속성) : VideoCapture의 속성을 반환합니다.
# # capture.grab() : Frame의 호출 성공 유/무를 반환합니다.
# # capture.isOpened() : VideoCapture의 성공 유/무를 반환합니다.
# # capture.open(카메라 장치 번호 또는 경로) : 카메라나 동영상 파일을 엽니다.
# # capture.release() : VideoCapture의 장치를 닫고 메모리를 해제합니다.
# # capture.retrieve() : VideoCapture의 프레임과 플래그를 반환합니다.
# # capture.set(속성, 값) : VideoCapture의 속성의 값을 설정합니다.
# cv2.destroyAllWindows()

## 7강, image pyramid를 활용하여 이미지 확대 축소
# img_filename = 'OpenCV_study_root\Resource\dog.jpg'
# img = cv2.imread(img_filename)

# height, width, channel = img.shape
# dst = cv2.pyrUp(img, dstsize=(width*2, height*2), borderType=cv2.BORDER_DEFAULT);
# dst2 = cv2.pyrDown(img);

# cv2.imshow("img", img)
# cv2.imshow("dst", dst)
# cv2.imshow("dst2", dst2)
# test_def.safe_run_imshow()

## 8강, resize
# img_filename = 'OpenCV_study_root\Resource\dog.jpg'
# img = cv2.imread(img_filename)
# dst = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA) # : 원본 이미지, 이미지 크기, 보간법
# dst2 = cv2.resize(img, dsize=(0,0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR) # : 비율로 조절할 수 있음

# cv2.imshow('img',img)
# cv2.imshow('dst',dst)
# cv2.imshow('dst2',dst2)
# test_def.safe_run_imshow()

## 9강, 이미지 자르기
# # 1)
# img_filename = 'OpenCV_study_root\Resource\dog.jpg'
# img = cv2.imread(img_filename)
# # dst = img 로 복재하게 되면 원본에도 영향을 미치게 됨
# dst = img.copy() #의 형식으로 복재해야 함
# dst = img[100:600, 200:700] # 이런식으로 잘라낼 영역을 설정한다.

# cv2.imshow("img", img)
# cv2.imshow('dst', dst)
# test_def.safe_run_imshow()

# # 2)
# img_filename = 'OpenCV_study_root\Resource\dog.jpg'
# img = cv2.imread(img_filename)

# dst = img.copy() #의 형식으로 복재해야 함
# roi = img[100:600, 200:700]
# dst[0:500, 0:500] = roi

# cv2.imshow("img", img)
# cv2.imshow("dst", dst)
# test_def.safe_run_imshow()

## 12강 이진화
# img_filename = 'OpenCV_study_root\Resource\dog.jpg'
# img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) # 그레이스케일 이미지, 임계값, 최댓값, 임계값 종류

# cv2.imshow('dat', dst)
# test_def.safe_run_imshow()

## 13강 흐림효과
# img_filename = 'OpenCV_study_root\Resource\dog.jpg'
# img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
# dst = cv2.blur(img, (9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
# cv2.imshow("dst", dst)
# test_def.safe_run_imshow()


## 14강, 가장자리 검출(Edge)
# img_filename = 'OpenCV_study_root\Resource\dog.jpg'
# img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# canny = cv2.Canny(img, 100, 255) #: 원본이미지, 임계값1, 임계값2, 커널 크기, L2그라디언트(디폴트는 L1)를 이용하여 가장자리 검출을 함
# sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3) # : 그레이스케일 이미지, 정밀도, x방향 미분, y방향 미분, 커널, 배율, 델타, 픽셀 외삽법을 이용하여 가장자리 검출을 함
# laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3) # : 그레이스케일 이미지, 정밀도, 커널, 배율, 델타, 픽셀 외삽법로 가장자리 검출을 함

# cv2.imshow("canny", canny)
# cv2.imshow("sobel", sobel)
# cv2.imshow("laplacian", laplacian)
# test_def.safe_run_imshow()

## 15강, HSV, Hue, Saturation, Value 색체 검지하는 것임
# HUE : 색의 질로 색 자체를 나타낸다
# SATURATION : 색의 선명도로 채도라고 한다
# VALUE : 색의 밝기로 명도라고 한다.

# 1) RGB 이미지를 HSV로 바꾸기
# img_filename = 'OpenCV_study_root\Resource\dog.jpg'
# img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 초기 속성은 BRG --> HSV로 바꿔줌(분리된 채널은 그레이 컬러로 단일 채널이므로 흑백의 색상으로만 표현)
# h, s, v = cv2.split(hsv)

# cv2.imshow('h',h)
# cv2.imshow('s',s)
# cv2.imshow('v',v)
# test_def.safe_run_imshow()

# 2) HUE의 범위를 조정하여 특정 색상만 출력하기
# img_filename = 'OpenCV_study_root\Resource\dog.jpg'
# img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 초기 속성은 BRG --> HSV로 바꿔줌(분리된 채널은 그레이 컬러로 단일 채널이므로 흑백의 색상으로만 표현)
# h, s, v = cv2.split(hsv)
# h = cv2.inRange(h, 8, 20)
# orange = cv2.bitwise_a(hsv, hsv, mask = h)
# orange = cv2.cvtColor(orange, cv2.COLOR_HSV2BGR)

# cv2.imshow('orange', orange)
# test_def.safe_run_imshow()

## 16강, 채널 범위 병합
# img_filename = 'OpenCV_study_root\Resource\internet_img.jpg'
# img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)

# lower_red = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255)) # 붉은 영역을 나눴다. 0~5
# upper_red = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))  # 붉은 영역을 나눴다. 170~180
# added_red = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0) # cv2.addWeighted(이미지1, 이미지1 비율, 이미지2, 이미지2 비율, 가중치) 방식으로 나눴다.

# red = cv2.bitwise_and(hsv, hsv, mask = added_red)
# red = cv2.cvtColor(red, cv2.COLOR_HSV2BGR)

# cv2.imshow("red", red)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## 17강, 채널 분리 및 병합 * opencv의 기본 색 정렬은 BRG이다.
# 1) Opencv 방식
# img_filename = 'OpenCV_study_root\Resource\internet_img.jpg'
# img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
# b, g, r = cv2.split(img) # 채널 분리
# inversebgr = cv2.merge((r, g, b)) # 채널 재정렬 및 합성

# cv2.imshow("b", b)
# cv2.imshow("g", g)
# cv2.imshow("r", r)
# cv2.imshow("inverse", inversebgr)
# test_def.safe_run_imshow()

# 2) numpy 방식
# img_filename = 'OpenCV_study_root\Resource\internet_img.jpg'
# img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
# b, g, r = cv2.split(img) # 채널 분리
# b = img[:,:,0] # 이미지[높이, 너비, 채널] 이라고 생각하면 된다.
# g = img[:,:,1]
# r = img[:,:,2]

# height, width, channel = img.shape
# zero = np.zeros((height,width,1), dtype = np.uint8) # 값이 빈 모양이 같은 이미지를 만든다.
# bgz = cv2.merge((b, g, zero)) # 레드가 있어야 될 부분에 빈것을 넣음
# cv2.imshow('bgz',bgz)
# test_def.safe_run_imshow()

## 18강 그림그리기

# src = np.zeros((768, 1366, 3), dtype = np.uint8) # BRG의 값이 빈 판을 만든다.

# cv2.line(src, (100, 100), (1200, 100), (0, 0, 255), 3, cv2.LINE_AA) # cv2.line(이미지, (x1, y1), (x2, y2), (B, G, R), 두께, 선형 타입)을 이용하여 선을 그릴 수 있습니다.
# cv2.circle(src, (300, 300), 50, (0, 255, 0), cv2.FILLED, cv2.LINE_4) # cv2.circle(이미지, (x, y), 반지름, (B, G, R), 두께, 선형 타입)을 이용하여 원
# cv2.rectangle(src, (500, 200), (1000, 400), (255, 0, 0), 5, cv2.LINE_8) # cv2.rectangle(이미지, (x1, y1), (x2, y2), (B, G, R), 두께, 선형 타입)을 이용하여 사각형
# cv2.ellipse(src, (1200, 300), (100, 50), 0, 90, 180, (255, 255, 0), 2) # cv2.ellipse(이미지, (x, y), (lr, sr), 각도, 시작 각도, 종료 각도, (B, G, R), 두께, 선형 타입)을 이용하여 타원

# # poly를 사용하여 그래픽을 그릴때에는 numpy형태의 위치 좌표가 필요함, n개의 점이 있으면 n각형이 됨
# pts1 = np.array([[100, 500], [300, 500], [200, 600]])
# pts2 = np.array([[600, 500], [800, 500], [700, 600]])
# cv2.polylines(src, [pts1], True, (0, 255, 255), 2) # cv2.polylines(이미지, [위치 좌표], 닫힘 유/무, (B, G, R), 두께, 선형 타입 )
# cv2.fillPoly(src, [pts2], (255, 0, 255), cv2.LINE_AA) # 위와 동일하나 내부가 찬 다각형을 그릴 수 있음

# cv2.putText(src, "DICK!!", (900, 600), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
# cv2.imshow('src',src)
# test_def.safe_run_imshow()

## 19강, 기하학적 변환

# img_filename = 'OpenCV_study_root\Resource\internet_img2.jpg'
# img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
# height, width, channel = img.shape

# srcPoint=np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
# dstPoint=np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
# matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)

# dst = cv2.warpPerspective(img, matrix,(width,height)) # 4개의 점을 매핑함
# cv2.imshow('dst', dst)
# test_def.safe_run_imshow()

## 20강, 캠쳐 및 녹화

# import datetime

# video_file_name = video_filename = 'OpenCV_study_root\Resource\Andy Anderson_ a Short Skate Film.mp4'
# output_folder_name = "OpenCV_study_root/Out_Resource/"
# capture = cv2.VideoCapture(video_file_name)
# fourcc = cv2.VideoWriter_fourcc(*'XVID') # 비디오에 해당하는 코덱을 사용해야함(코덱의 인코딩 방식)
# record = False # 녹화 유/무 설정

# while True:
#     if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
#         capture.open(video_file_name)

#     ret, frame = capture.read()
#     cv2.imshow("VideoFrame", frame)

#     now = datetime.datetime.now().strftime("%d_%H-%M-%S") # 현재 시간을 받아와서 제목으로 설정하도록
#     key = cv2.waitKey(33) # 현재 눌러진 키보드의 키값이 저장 33ms 마다 갱신됨

# # 눌러진 키값을 판단함
#     if key == 27: # ESC
#         break
#     elif key == 26: # Ctrl+z
#         print("캡쳐")
#         cv2.imwrite(output_folder_name + str(now) + ".png", frame)
#     elif key == 24: # Ctrl+x
#         print("녹화 시작")
#         record = True
#         video = cv2.VideoWriter(output_folder_name + str(now) + ".avi",
#                             fourcc, 20.0, (frame.shape[1], frame.shape[0]))
#                 # cv2.VideoWriter("경로 및 제목", 비디오 포맷 코드, FPS, (녹화 파일 너비, 녹화 파일 높이))를 의미함
#     elif key == 3: # Ctrl+c
#         print("녹화 중지")
#         record = False
#         video.release()

#     if record == True:
#         print("녹화 중..")
#         video.write(frame) # 저장할 프레임을 설정함

# capture.release()
# cv2.destroyAllWindows()

# import cv2
# img_file_name = 'OpenCV_study_root\Resource\contours.png'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)

# gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) # 이미지 이진화
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # 이미지 임계처리 : 흰색에 가까운 값만 살림
# binary = cv2.bitwise_not(binary) # 반전시켜서 검출하려는 물체가 하얀색의 성질을 띄도록 변환

# contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# # findContours로 이진화 윤곽선을 검색함 : cv2.findContours(이진화 이미지, 검색 방법, 근사화 방법)
# # contours : 윤곽선의 지점, hierarchy : 계층
# # 계층구조는 외곽 윤곽선, 내곽 윤곽선, 모든 윤관선...

# for i in range(len(contours)):
#     cv2.drawContours(src, [contours[i]], 0, (0, 0, 255), 2) # 검출된 윤곽선을 그림
#     cv2.putText(src, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
#     print(i, hierarchy[0][i])
#     cv2.imshow("src", src)
#     cv2.waitKey(0) # 이러면 눌릴때마다 넘어감 오

# cv2.destroyAllWindows()

## 22강, 다각형 근사 (윤곽선 검출)
# img_file_name = 'OpenCV_study_root\Resource\internet_img3.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)

# gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) # 그레이 칼러 변환
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU) # 임계처리 : OTSU는 임계갑승ㄹ 자동으로 계산해줌
# binary = cv2.bitwise_not(binary) # 두개의 이미지를 비트연산 임계값만 제외처리함

# contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
# # RETR_LIST : 계층에 상관없이 이미지에서 발견한 모든 Contour을 나열하는 것
# # cv2.CHAIN_APPROX_TC89_KCOS :근사 알고리즘 적용

# # 색인값과 하위 윤곽선 정보 반복
# for contour in contours:
#     epsilon = cv2.arcLength(contour, True) * 0.02 # 근사치의 정확도르 계산하기 위해 전체 길이의 2%만 사용
#     # arcLength : 윤곽선의 길이 조사 (윤곽선, 폐곡선)을 의미함

#     # 윤곽선들의 윤곽점들로 근사해 근사 다각형으로 변환 : (윤곽선, 근사치 정확도, 폐곡선)을 의미함
#     approx_poly = cv2.approxPolyDP(contour, epsilon, True)

#     # 근사 다각형을 반복해 근사점을 이미지 위에 표기함
#     for approx in approx_poly:
#         cv2.circle(src, tuple(approx[0]), 3, (255, 0, 0), -1)

# cv2.imshow("src", src)
# test_def.safe_run_imshow()

## 23강, 코너 검출 (트래킹하기 좋은 지점 검출)
# img_file_name = 'OpenCV_study_root\Resource\internet_img4.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# dst = src.copy()

# gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
# corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 5, blockSize=3, useHarrisDetector=True, k=0.03)
# # goodFeaturesToTrack : 윤곽선들의 이미지에서 코너를 검출함
# # cv2.goodFeaturesToTrack(입력 이미지, 코너 최댓값, 코너 품질, 최소 거리, 마스크, 블록 크기, 해리스 코너 검출기 유/무, 해리스 코너 계수)
# '''
# 입력 이미지는 8비트 또는 32비트의 단일 채널 이미지를 사용합니다.

# 코너 최댓값은 검출할 최대 코너의 수를 제한합니다. 코너 최댓값보다 낮은 개수만 반환합니다.

# 코너 품질은 반환할 코너의 최소 품질을 설정합니다. 코너 품질은 0.0 ~ 1.0 사이의 값으로 할당할 수 있으며, 일반적으로 0.01 ~ 0.10 사이의 값을 사용합니다.

# 최소 거리는 검출된 코너들의 최소 근접 거리를 나타내며, 설정된 최소 거리 이상의 값만 검출합니다.

# 마스크는 입력 이미지와 같은 차원을 사용하며, 마스크 요솟값이 0인 곳은 코너로 계산하지 않습니다.

# 블록 크기는 코너를 계산할 때, 고려하는 코너 주변 영역의 크기를 의미합니다.

# 해리스 코너 검출기 유/무는 해리스 코너 검출 방법 사용 여부를 설정합니다.

# 해리스 코너 계수는 해리스 알고리즘을 사용할 때 할당하며 해리스 대각합의 감도 계수를 의미합니다.

# Tip : 코너 품질에서 가장 좋은 코너의 강도가 1000이고, 코너 품질이 0.01이라면 10 이하의 코너 강도를 갖는 코너들은 검출하지 않습니다.
# Tip : 최소 거리의 값이 5일 경우, 거리가 5 이하인 코너점은 검출하지 않습니다.
# '''
# # corners : 코너들의 좌표가 저장되어 있음
# for i in corners:
#     cv2.circle(dst, tuple(i[0]), 3, (0, 0, 255), 2)

# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 24강, 블록 껍질 :  경계면을 둘러싸는 다각형을 구하는 알고리즘
# img_file_name = 'OpenCV_study_root\Resource\internet_png.png'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# dst = src.copy()

# gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
# ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) # 임계값 주기

# contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # 윤곽선 잡기

# for i in contours:
#     hull = cv2.convexHull(i, clockwise=True) # 윤곽선의 블록 껍질을 검출함 : cv2.convexHull(윤곽선, 방향)을 의미함
#     cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2)

# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 25강, 모멘트 : 이미지의 중심을 잡는데 자주 사용된다.
# img_file_name = 'OpenCV_study_root\Resource\internet_png.png'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# dst = src.copy()

# gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
# ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) # 그레이 스케일로 전환

# contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # 윤곽선을 검출함

# for i in contours:
#     M = cv2.moments(i) # 윤관선에서 모멘트를 계산함
#     cX = int(M['m10'] / M['m00'])
#     cY = int(M['m01'] / M['m00'])
#     # 위의 공식을 이용하여 무개중심(중심점을 구함)
#     cv2.circle(dst, (cX, cY), 3, (255, 0, 0), -1)
#     cv2.drawContours(dst, [i], 0, (0, 0, 255), 2)

# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## 26강, 모폴로지 변환 : 영상이나 이미지를 형태학적 관점에서 접근하는 기법
# '''
# 모폴로지 변환은 주로 영상 내 픽셀값 대체에 사용됩니다. 이를 응용해서 노이즈 제거, 요소 결합 및 분리, 강도 피크 검출 등에 이용할 수 있습니다.

# 집합의 포함 관계, 이동(translation), 대칭(reflection), 여집합(complement), 차집합(difference) 등의 성질을 사용합니다.

# 기본적인 모폴로지 변환으로는 팽창(dilation)과 침식(erosion)이 있습니다.

# 팽창과 침식은 이미지와 커널의 컨벌루션 연산이며, 이 두 가지 기본 연산을 기반으로 복잡하고 다양한 모폴로지 연산을 구현할 수 있습니다.
# '''
# # 팽창(Dilation) : 모든 픽셀의 값을 커널 내부의 극댓값(local maximum)으로 대체, 어두운 영역이 줄어들고 밝은 영역이 늘어납니다.
# #                  노이즈 제거 후 줄어든 크기를 복구하고자 할 때 사용함

# # 침식 : 픽셀의 값을 커널 내부의 극솟값(local minimum)으로 대체, 밝은 영역이 줄어들고 어두운 영역이 늘어납니다.
#         #  침식 연산은 노이즈 제거에 주로 사용
# img_file_name = 'OpenCV_study_root\Resource\internet_img5.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)

# # 구조요소를 생성 cv2.getStructuringElement(커널의 형태, 커널의 크기, 중심점)
# # 커널의 크기는 구조 요소의 크기를 의미
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))

# dilate = cv2.dilate(src, kernel, anchor=(-1, -1), iterations=5) # 팽창함수
# erode = cv2.erode(src, kernel, anchor=(-1, -1), iterations=5) # 침식함수
# # cv2.erode(원본 배열, 구조 요소, 고정점, 반복 횟수, 테두리 외삽법, 테두리 색상)
# # 고정점을 (-1, -1)로 할당할 경우, 커널의 중심부에 고정점이 위치하게 됩니다.

# # 연결 함수(np.concatenate)로 원본 이미지, 팽창 결과, 침식 결과를 하나의 이미지로 연결합니다.
# # 동시에 보여주기 위해 사용
# # np.concatenate(연결할 이미지 배열들, 축 방향)
# dst = np.concatenate((src, dilate, erode), axis=1)


# Resize = test_def.ez_resizeing(dst, width=1280)
# cv2.imshow('Resize', Resize)
# test_def.safe_run_imshow()

## 27강, 모폴로지 연산 :  모폴로지 변환의 팽창(dilation)과 침식(erosion)을 기본 연산으로 사용해 고급 형태학을 적용하는 변환 연산
# '''
# 입력 이미지가 이진화된 이미지라면 팽창과 침식 연산으로도 우수한 결과를 얻을 수 있습니다.

# 하지만, 그레이스케일이나 다중 채널 이미지를 사용하는 경우 더 복잡한 연산을 필요로 합니다.

# 이때 모폴로지 연산을 활용해 우수한 결과를 얻을 수 있습니다.
# '''
# # 열림(Opening) : 팽창 연산자와 침식 연산자의 조합이며, 침식 연산을 적용한 다음, 팽창 연산을 적용합니다.,스펙클(speckle)이 사라지면서 발생한 객체의 크기 감소를 원래대로 복구 
# # 닫힘(Closeing) : 팽창 연산자와 침식 연산자의 조합이며, 열림과 반대로 팽창 연산을 적용한 다음, 침식 연산을 적용합니다., 체 내부의 홀(holes)이 사라지면서 발생한 크기 증가를 원래대로 복구할 수 있다.
# # 그레이디언트(Gradient) : 팽창 연산자와 침식 연산자의 조합이며, 열림 연산이나 닫힘 연산과 달리 입력 이미지에 각각 팽창 연산과 침식 연산을 적용하고 감산을 진행합니다., 그레이스케일 이미지가 가장 급격하게 변하는 곳에서 가장 높은 결과를 반환합니다.
# # 탑햇(TopHat) : 입력 이미지(src)와 열림(Opening)의 조합이며, 그레이디언트 연산과 비슷하게 입력 이미지에 열림 연산을 적용한 이미지를 감산합니다., 입력 이미지의 객체들이 제외되고 국소적으로 밝았던 부분들이 분리됩니다
# # 블랙햇(BlackHat) : 입력 이미지(src)와 닫힘(Closing)의 조합이며, 탑햇 연산과 비슷하게 닫힘 연산을 적용한 이미지에 입력 이미지를 감산합니다., 입력 이미지의 객체들이 제외되고 국소적으로 어두웠던 홀들이 분리됩니다.
# # 히트미스(HitMiss) : 이미지의 전경이나 배경 픽셀의 특정 패턴을 찾는 데 사용하는 이진 형태학으로서 구조 요소의 형태에 큰 영향을 받습니다., 커널 내부의 0은 해당 픽셀을 고려하지 않는다는 의미이며, 1은 해당 요소를 유지하겠다는 의미입니다.
# #                    이 특성 덕분에 히트미스 연산을 모서리(Corner)를 검출하는 데 활용하기도 합니다.
# #                    * 제한 조건 - 8-bit unsigned integers, 1-Channel

# img_file_name = 'OpenCV_study_root\Resource\internet_img6.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# dst = src.copy()

# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)) # 커널 구성요소 생성(커널 매트릭스 생성) 직사각형 9*9 크기의 매트릭스 생성
# dst = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=9)
# # cv2.morphologyEx(원본 배열, 연산 방법, 구조 요소, 고정점, 반복 횟수, 테두리 외삽법, 테두리 색상)
# '''
# cv2.MORPH_DILATE	팽창 연산
# cv2.MORPH_ERODE	침식 연산
# cv2.MORPH_OPEN	열림 연산
# cv2.MORPH_CLOSE	닫힘 연산
# cv2.MORPH_GRADIENT	그레이디언트 연산
# cv2.MORPH_TOPHAT	탑햇 연산
# cv2.MORPH_BLACKHAT	블랙햇 연산
# cv2.MORPH_HITMISS	히트미스 연산
# '''
# dst_resize = test_def.ez_resizeing(dst, width=1280)
# cv2.imshow('dst_resize', dst_resize)
# test_def.safe_run_imshow()

## 28강 직선 검출
'''
직선 검출 알고리즘은 허프 변환(Hough Transform)을 활용해 직선을 검출합니다.

허프 변환은 이미지에서 직선을 찾는 가장 보편적인 알고리즘입니다.

이미지에서 선과 같은 단순한 형태를 빠르게 검출할 수 있으며, 직선을 찾아 이미지나 영상을 보정하거나 복원합니다.

허프 선 변환은 이미지 내의 어떤 점이라도 선 집합의 일부일 수 있다는 가정하에 직선의 방정식을 이용해 직선을 검출한다.

직선 검출은 직선의 방정식을 활용해 
y=ax+b
를 극좌표(ρ, θ)의 점으로 변환해서 사용합니다.

극좌표 방정식으로 변환한다면 
p=xsinθ+ycosθ
이 되어, 직선과 원점의 거리(ρ)와 직선과 x축이 이루는 각도(θ)를 구할 수 있습니다.
'''
# 표준 허프 변환(Standard Hough Transform) & 멀티 스케일 허프 변환(Multi-Scale Hough Transform)
# 표준 허프 변환(Standard Hough Transform) : 입력 이미지(x, y 평면) 내의 점 p를 지나는 직선의 방정식을 구합니다
# 멀티 스케일 허프 변환(Multi-Scale Hough Transform) : 표준 허프 변환을 개선한 방법, 검출한 직선의 값이 더 정확한 값으로 반환되도록, 거리(ρ)와 각도(θ)의 값을 조정해 사용

# img_file_name = 'OpenCV_study_root\Resource\internet_img7.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# # 직선검출전 전처리 작업 진행 : 그레이스케일 이미지와 케니 엣지 이미지를 사용함
# dst = src.copy() # 결과이미지
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# canny = cv2.Canny(gray, 5000, 1500, apertureSize = 5, L2gradient = True) # 케니 엣지 알고리즘의 임계값을 각각 5000과 1500으로 주요한 가장자리만 남기게 함

# 직선 검출을 진행한다.
# 1)
# lines = cv2.HoughLines(canny, 0.8, np.pi / 180, 150, srn = 100, stn = 200, min_theta = 0, max_theta = np.pi)
# cv2.HoughLines(검출 이미지, 거리, 각도, 임곗값, 거리 약수, 각도 약수, 최소 각도, 최대 각도)
'''
거리와 각도는 누산 평면에서 사용되는 해상도를 나타냅니다.
거리의 단위는 픽셀을 의미하며, 0.0 ~ 1.0의 실수 범위를 갖습니다.
각도의 단위는 라디안을 사용하며 0 ~ 180의 범위를 갖습니다.
임곗값은 허프 변환 알고리즘이 직선을 결정하기 위해 만족해야 하는 누산 평면의 값을 의미합니다.
누산 평면은 각도 × 거리의 차원을 갖는 2차원 히스토그램으로 구성됩니다.
거리 약수와 각도 약수는 거리와 각도에 대한 약수(divisor)를 의미합니다.
두 값 모두 0의 값을 인수로 활용할 경우, 표준 허프 변환이 적용되며, 하나 이상의 값이 0이 아니라면 멀티 스케일 허프 변환이 적용됩니다.
최소 각도와 최대 각도는 검출할 각도의 범위를 설정'''
# for i in lines: # lines에는 (n, 거리, 각도)를 포함하고 있음
#     rho, theta = i[0][0], i[0][1]
#     a, b = np.cos(theta), np.sin(theta)
#     x0, y0 = a*rho, b*rho

#     scale = src.shape[0] + src.shape[1]

#     x1 = int(x0 + scale * -b)
#     y1 = int(y0 + scale * a)
#     x2 = int(x0 - scale * -b)
#     y2 = int(y0 - scale * a)

#     cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv2.circle(dst, (x0, y0), 3, (255, 0, 0), 5, cv2.FILLED)
# cv2.imshow("dst", dst)
# dst_resize = test_def.ez_resizeing(dst, width=1280)
# cv2.imshow('dst_resize', dst_resize)
# test_def.safe_run_imshow()

# 2)
# 점진성 확률적 허프 변환(Progressive Probabilistic Hough Transform)
# 점진성 확률적 허프 변환(Progressive Probabilistic Hough Transform)은 또 다른 허프 변환 함수를 사용해 직선을 검출
# lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength = 10, maxLineGap = 100)
# # cv2.HoughLinesP(검출 이미지, 거리, 각도, 임곗값, 최소 선 길이, 최대 선 간격)
# for i in lines: # lines에는 (n,거리, 각도,임계값)를 포함하고 있음 : 시작점과 끝점을 포함함
#     cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)

# # cv2.imshow("dst", dst)
# dst_resize = test_def.ez_resizeing(dst, width=1280)
# cv2.imshow('dst_resize', dst_resize)
# test_def.safe_run_imshow()

## 29강, 원 검출
# 허프 원 변환(Hough Circle Transform) 알고리즘 :  3차원 누산 평면으로 검출
# OpenCV 원 검출 함수는 2단계 허프 변환(Two stage Hough Transform) 방법을 활용해 원을 검출

# import cv2

# img_file_name = 'OpenCV_study_root\Resource\internet_img8.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# dst = src.copy()
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1 = 250, param2 = 10, minRadius = 80, maxRadius = 120)
# # cv2.HoughCircles(검출 이미지, 검출 방법, 해상도 비율, 최소 거리, 캐니 엣지 임곗값, 중심 임곗값, 최소 반지름, 최대 반지름)
# '''
# 검출 방법은 항상 2단계 허프 변환 방법(21HT, 그레이디언트)만 사용합니다.
# 해상도 비율은 원의 중심을 검출하는 데 사용되는 누산 평면의 해상도를 의미합니다.
# 인수를 1로 지정할 경우 입력한 이미지와 동일한 해상도를 가집니다. 즉, 입력 이미지 너비와 높이가 동일한 누산 평면이 생성됩니다.
# 또한 인수를 2로 지정하면 누산 평면의 해상도가 절반으로 줄어 입력 이미지의 크기와 반비례합니다.
# 최소 거리는 일차적으로 검출된 원과 원 사이의 최소 거리입니다. 이 값은 원이 여러 개 검출되는 것을 줄이는 역할을 합니다.
# 캐니 엣지 임곗값은 허프 변환에서 자체적으로 캐니 엣지를 적용하게 되는데, 이때 사용되는 상위 임곗값을 의미합니다.
# 하위 임곗값은 자동으로 할당되며, 상위 임곗값의 절반에 해당하는 값을 사용합니다.
# 중심 임곗값은 그레이디언트 방법에 적용된 중심 히스토그램(누산 평면)에 대한 임곗값입니다. 이 값이 낮을 경우 더 많은 원이 검출됩니다.
# 최소 반지름과 최대 반지름은 검출될 원의 반지름 범위입니다. 0을 입력할 경우 검출할 수 있는 반지름에 제한 조건을 두지 않습니다.
# 최소 반지름과 최대 반지름에 각각 0을 입력할 경우 반지름을 고려하지 않고 검출하며, 최대 반지름에 음수를 입력할 경우 검출된 원의 중심만 반환합니다.
# '''
# # circles 변수는 (1, N, 3)차원 형태, 내부 차원의 요소로는 검출된 중심점(x, y)과 반지름(r)이 저장
# for i in circles[0]:
#     cv2.circle(dst, (i[0], i[1]), int(i[2]), (255, 255, 255), 5)

# # cv2.imshow("dst", dst)
# dst_resize = test_def.ez_resizeing(dst, width=1280)
# cv2.imshow('dst_resize', dst_resize)
# test_def.safe_run_imshow()

## 30강 이미지 연산
# 하나 또는 둘 이상의 이미지에 대해 수학적인 연산을 수행

# img_file_name = 'OpenCV_study_root\Resource\internet_img9.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# number1 = np.ones_like(src) * 127 # : 회색 이미지(127, 127, 127) 사용
# number2 = np.ones_like(src) * 2 # : 검은색 이미지(2, 2, 2) 사용

# # 결괏값이 0보다 작다면, 0으로 반환되며, 결괏값이 255보다 크다면, 255로 반환됩니다.
# add = cv2.add(src, number1)
# sub = cv2.subtract(src, number1)
# mul = cv2.multiply(src, number2)
# div = cv2.divide(src, number2)

# src = np.concatenate((src, src, src, src), axis = 1)
# number = np.concatenate((number1, number1, number2, number2), axis = 1)
# dst = np.concatenate((add, sub, mul, div), axis = 1)

# dst = np.concatenate((src, number, dst), axis = 0)
# '''
# 나오는 모양
# src	src	src	src
# number1	number1	number2	number2
# add	sub	mul	div
# '''

# dst_resize = test_def.ez_resizeing(dst, width=1280)
# cv2.imshow('dst_resize', dst_resize)
# test_def.safe_run_imshow()


# ## 31강, 이미지 연산 2
# img_file_name = 'OpenCV_study_root\Resource\internet_img9.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# number = np.ones_like(src) * 127

# _max = cv2.max(src, number) # 최댓값 함수는 두 이미지의 요소별 최댓값
# _min = cv2.min(src, number) # 최솟값 함수는 두 이미지의 요소별 최솟값을 계산
# _abs = cv2.absdiff(src, number) # 절댓값 차이 함수는 두 이미지의 요소별 절댓값 차이를 계산
# compare = cv2.compare(src, number, cv2.CMP_GT)

# src = np.concatenate((src, src, src, src), axis = 1)
# number = np.concatenate((number, number, number, number), axis = 1)
# dst = np.concatenate((_max, _min, _abs, compare), axis = 1)

# dst = np.concatenate((src, number, dst), axis = 0)

# dst_resize = test_def.ez_resizeing(dst, width=1280)
# cv2.imshow('dst_resize', dst_resize)
# test_def.safe_run_imshow()


## 32강, 비트 연산

# img_file_name = 'OpenCV_study_root\Resource\internet_img10.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # 127 임계값의 이진화를 사용함

# _and = cv2.bitwise_and(gray, binary) # 연산 이미지1과 연산 이미지2의 값을 비트 단위로 파악하며, 해당 비트에 대해 AND 연산을 진행
# _or = cv2.bitwise_or(gray, binary) # 연산 이미지1과 연산 이미지2의 값을 비트 단위로 파악하며, 해당 비트에 대해 OR 연산을 진행
# _xor = cv2.bitwise_xor(gray, binary) # 연산 이미지1과 연산 이미지2의 값을 비트 단위로 파악하며, 해당 비트에 대해 XOR 연산을 진행
# _not = cv2.bitwise_not(gray) # 연산 이미지1의 값을 비트 단위로 파악하며, 해당 비트에 대해 NOT 연산을 진행

# src = np.concatenate((np.zeros_like(gray), gray, binary, np.zeros_like(gray)), axis = 1)
# dst = np.concatenate((_and, _or, _xor, _not), axis = 1)
# dst = np.concatenate((src, dst), axis = 0)
# '''
# 구성
# None	gray	binary	None
# _and	_or	_xor	_not
# '''
# dst_resize = test_def.ez_resizeing(dst, width=1280)
# cv2.imshow('dst_resize', dst_resize)
# test_def.safe_run_imshow()

## 33강, 히스토그램

# X 축을 픽셀의 값으로 사용하고 Y 축을 해당 픽셀의 개수로 표현,  이미지의 특성을 쉽게 확인할 수 있습
# '''
# 히스토그램의 핵심요소
# 1. 빈도 수(BINS): 히스토그램 그래프의 X 축 간격
# 2. 차원 수(DIMS): 히스토그램을 분석할 이미지의 차원
# 3. 범위(RANGE): 히스토그램 그래프의 X 축 범위
# '''
# img_file_name = 'OpenCV_study_root\Resource\internet_img11.jpg'
# # 원본 이미지(src)와 그레이스케일(gray), 히스토그램 이미지(result)을 선언
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# result = np.zeros((src.shape[0], 256), dtype=np.uint8)

# # 히스토그램 계산 함수를 통해 분포를 계산 할 수 있음
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# # cv2.calcHist(연산 이미지, 특정 채널, 마스크, 히스토그램 크기, 히스토그램 범위)
# cv2.normalize(hist, hist, 0, result.shape[0], cv2.NORM_MINMAX)
# # cv2.normalize(입력 배열, 결과 배열, alpha, beta, 정규화 기준)
# # 히스토그램으로 인한 값은 정규화 되어 지지 않은 값이므로 정규화 시켜줌

# for x, y in enumerate(hist):
#     cv2.line(result, (x, result.shape[0]), (x, result.shape[0] - y), 255)

# dst = np.hstack([gray, result])

# dst_resize = test_def.ez_resizeing(dst, width=1280)
# cv2.imshow('dst_resize', dst_resize)
# test_def.safe_run_imshow()

## 34강, 픽셀 접근 : 이미지 배열에서 특정 좌표에 대한 값을 받아오거나, 변경할 때 사용합니다.
# Numpy 배열의 요소 접근 방식과 동일하며, 직접 값을 변경하거나 할당할 수 있습니다.

# gray = np.linspace(0, 255, num=90000, endpoint=True, retstep=False, dtype=np.uint8).reshape(300, 300, 1)
# color = np.zeros((300, 300, 3), np.uint8)
# color[0:150, :, 0] = gray[0:150, :, 0]
# color[:, 150:300, 2] = gray[:, 150:300, 0]
# # 배열의 접근 방식은 배열[행 시작:행 끝, 열 시작: 열 끝, 차원 시작:차원 끝]의 구조

# x, y, c = 200, 100, 0
# access_gray = gray[y, x, c]
# access_color_blue = color[y, x, c]
# access_color = color[y, x]

# print(access_gray)
# print(access_color_blue)
# print(access_color)

# cv2.imshow("gray", gray)
# cv2.imshow("color", color)

# test_def.safe_run_imshow()

## 35강, 트랙 바 : 스크롤 바의 하나로, 슬라이더 바의 형태를 갖고 있습니다.

# # 트랙 바는 일정 범위 내의 값을 변경할 때 사용하며, 적절한 임곗값을 찾거나 변경하기 위해 사용,
# # 생성된 윈도우 창에 트랙바를 부착해 사용할 수 있습니다.

# def onChange(pos):
#     pass

# img_file_name = 'OpenCV_study_root\Resource\internet_img12.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_GRAYSCALE)
# cv2.namedWindow("Trackbar Windows") # 트랙 바를 윈도우 창에 부착하기 위해서는 윈도우 창에 생성된 상태여야 함

# cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange) # 트랙 바를 생성
# cv2.createTrackbar("maxValue", "Trackbar Windows", 0, 255, lambda x : x)
# # cv2.createTrackbar("트랙 바 이름", "윈도우 창 제목", 최솟값, 최댓값, 콜백 함수)
# # 여기서 onChange 함수의 pos는 현재 발생한 트랙 바 값을 반환

# cv2.setTrackbarPos("threshold", "Trackbar Windows", 127) #  트랙 바의 값을 설정
# cv2.setTrackbarPos("maxValue", "Trackbar Windows", 255)
# # cv2.setTrackbarPos("트랙 바 이름", "윈도우 창 제목", 설정값)
# # 설정값은 초기에 할당된 값이나, 특정 조건 등을 만족했을 때 강제로 할당할 값을 설정
# '''
# 트랙 바 이름은 트랙 바의 명칭이며, 윈도우 창 제목과 같이 변수와 비슷한 역할을 합니다.
# 윈도우 창 제목은 트랙 바를 부착할 윈도우 창을 의미합니다.
# 최솟값과 최댓값은 트랙 바를 조절할 때 사용할 최소/최대 값을 의미합니다.
# 콜백 함수는 트랙 바의 바를 조절할 때 위치한 값을 전달합니다.
# onChange 함수의 pos는 현재 발생한 트랙 바 값을 반환합니다.'''
# while cv2.waitKey(1) != ord('q'): # 지속적으로 화면 갱신
    
#     # getTrackbarPos는 트랙바 받기 함수
#     thresh = cv2.getTrackbarPos("threshold", "Trackbar Windows")
#     maxval = cv2.getTrackbarPos("maxValue", "Trackbar Windows")
#     # cv2.getTrackbarPos("트랙 바 이름", "윈도우 창 제목")
#     _, binary = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)

#     cv2.imshow("Trackbar Windows", binary)

# cv2.destroyAllWindows()

## 36강, 적응형 이진화 : 적응형 이진화 알고리즘은 입력 이미지에 따라 임곗값이 스스로 다른 값을 할당할 수 있도록 구성된 이진화 알고리즘
# '''
# 이미지에 따라 어떠한 임곗값을 주더라도 이진화 처리가 어려운 이미지가 존재합니다.
# 예를 들어, 조명의 변화나 반사가 심한 경우 이미지 내의 밝기 분포가 달라 국소적으로 임곗값을 적용해야 하는 경우가 있습니다.
# 이러한 경우 적응형 이진화 알고리즘을 적용한다면 우수한 결과를 얻을 수 있습니다.
# '''
# img_file_name = 'OpenCV_study_root\Resource/tree.jpg'
# src = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # 그레이 스케일 적용
# # 적응형 이진화 적용
# binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 467, 37)
# # cv2.adaptiveThreshold(입력 이미지, 최댓값, 적응형 이진화 플래그, 임곗값 형식, 블록 크기, 감산값)
# '''
# cv2.ADAPTIVE_THRESH_MEAN_C : blockSize 영역의 모든 픽셀에 평균 가중치를 적용
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C : blockSize 영역의 모든 픽셀에 중심점으로부터의 거리에 대한 가우시안 가중치 적용
# '''
# cv2.imshow("binary", binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## 37강, 탬플릿 매칭 : 원본 이미지에서 템플릿 이미지와 일치하는 영역을 찾는 알고리즘

# img_file_name1 = 'OpenCV_study_root\Resource\hats.png'
# img_file_name2 = 'OpenCV_study_root\Resource\hatss.png'

# src = cv2.imread(img_file_name2, cv2.IMREAD_GRAYSCALE) # 원본 이미지(src) 선언
# templit = cv2.imread(img_file_name1, cv2.IMREAD_GRAYSCALE) # 템플릿 이미지(templit) 선언
# dst = cv2.imread(img_file_name2) # 결과 이미지(dst) 선언

# result = cv2.matchTemplate(src, templit, cv2.TM_SQDIFF_NORMED)
# # cv2.matchTemplate(원본 이미지, 템플릿 이미지, 템플릿 매칭 플래그), 원본 이미지와 템플릿 이미지는 8비트의 단일 채널 이미지를 사용
# '''
# 반환되는 결괏값(dst)은 32비트의 단일 채널 이미지로 반환됩니다.
# 또한, 배열의 크기는 W - w + 1, H - h + 1의 크기를 갖습니다.
# (W, H)는 원본 이미지의 크기이며, (w, h)는 템플릿 이미지의 크기입니다.
# 결괏값이 위와 같은 크기를 갖는 이유는 원본 이미지에서 템플릿 이미지를 일일히 비교하기 때문입니다.
# 예를 들어, 4×4 크기의 원본 이미지와 3×3 크기의 템플릿 이미지가 있다면 아래의 그림과 같이 표현할 수 있습니다.
# '''

# # 결괏값(dst)에서 가장 유사한 부분을 찾기 위해 최소/최대 위치 함수(cv2.minMaxLoc)로 검출값을 찾습니다.
# minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
# x, y = minLoc # 최소 포인터, 최대 포인터,
# h, w = templit.shape # 최소 지점, 최대 지점

# dst = cv2.rectangle(dst, (x, y), (x +  w, y + h) , (0, 0, 255), 1) #검출된 결과를 결과 이미지(dst)위에 표시
# dst_resize = test_def.ez_resizeing(dst, width=1280)
# cv2.imshow('dst_resize', dst_resize)
# test_def.safe_run_imshow()

## 38강, ORB(Oriented FAST and Rotated BRIEF) : 특징점을 찾아내는 방법
'''
FAST(Features from Accelerated Segment Test) 알고리즘
FAST 알고리즘은 로스텐(Rosten)과 드리먼드(Drummond)가 제안한 피처 검출기 알고리즘으로서 픽셀 P와 픽셀 주변의 작은 원 위에 있는 픽셀의 집합을 비교하는 방식입니다.
픽셀 P의 주변 픽셀에 임곗값을 적용해 어두운 픽셀, 밝은 픽셀, 유사한 픽셀로 분류해 원 위의 픽셀이 연속적으로 어둡거나 밝아야 하며 이 연속성이 절반 이상이 돼야 합니다.
이 조건을 만족하는 경우 해당 픽셀은 우수한 특징점으로 볼 수 있다는 개념입니다.'''
'''
BRIEF(Binary Robust Independent Elementary Features) 알고리즘
BRIEF 알고리즘은 칼론더(Calonder) 연구진이 개발해 칼론더 피처라고도 불립니다
이 알고리즘은 특징점(Key Point)을 검출하는 알고리즘이 아닌 검출된 특징점에 대한 기술자(Descriptor)를 생성하는 데 사용합니다.
특징점 주변 영역의 픽셀을 다른 픽셀과 비교해 어느 부분이 더 밝은지를 찾아 이진 형식으로 저장합니다.
가우시안 커널을 사용해 이미지를 컨벌루션 처리하며, 피처 중심 주변의 가우스 분포를 통해 첫 번째 지점과 두 번째 지점을 계산해 모든 픽셀을 한 쌍으로 생성합니다.
즉, 두 개의 픽셀을 하나의 그룹으로 묶는 방식입니다.'''
# *Tip : 기술자(Descriptor)란 서로 다른 이미지에서 특징점(Key Point)이 어떤 연관성을 가졌는지 구분하게 하는 역할을 합니다.

'''
ORB(Oriented FAST and Rotated BRIEF) 알고리즘
ORB 알고리즘은 FAST 알고리즘을 사용해 특징점을 검출합니다.
FAST 알고리즘은 코너뿐만 아니라 가장자리에도 반응하는 문제점으로 인해 해리스 코너 검출 알고리즘을 적용해 최상위 특징점만 추출합니다.
이 과정에서 이미지 피라미드를 구성해 스케일 공간 검색을 수행합니다.
이후 스케일 크기에 따라 피처 주변 박스 안의 강도 분포에 대해 X축과 Y축을 기준으로 1차 모멘트를 계산합니다.
1차 모멘트는 그레이디언트의 방향을 제공하므로 피처의 방향을 지정할 수 있습니다.
방향이 지정되면 해당 방향에 대해 피처 벡터를 계산할 수 있으며, 피처는 회전 불변성을 갖고 있으며 방향 정보를 포함하고 있습니다.
하나의 ORB 피처를 가져와 피처 주변의 박스에서 1차 모멘트와 방위 벡터를 계산합니다.
피처의 중심에서 모멘트가 가리키는 위치까지 벡터를 피처 방향으로 부여하게 됩니다. ORB의 기술자는 BRIEF 기술자에 없는 방향 정보를 갖고 있습니다.
ORB 알고리즘은 SIFT(Scale-Invariant Feature Trasnform) 알고리즘과 SURF(Speeded-Up Robust Features) 알고리즘 을 대체하기 위해 OpenCV Labs에서 개발됐으며 속도 또한 더 빨라졌습니다.
'''
# *Tip : 회전 불변성이란 이미지가 회전돼 있어도 기술자는 회전 전과 같은 값으로 계산됩니다. 회전 불변성을 갖고 있지 않다면 회전된 이미지에서 피처는 서로 다른 의미(값)를 지니게 됩니다.
# *Tip : OpenCV 4 부터는 SIFT 알고리즘과 SURF 알고리즘을 지원하지 않습니다.

# src = cv2.imread("OpenCV_study_root\Resource/apple_books.jpg")
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 원본이미지 선언
# target = cv2.imread("OpenCV_study_root\Resource/apple.jpg", cv2.IMREAD_GRAYSCALE) # 타겟 이미지선언

# # ORB 객체 생성
# orb = cv2.ORB_create(
#     nfeatures=40000, # 최대 피처 수는 ORB 객체가 한 번에 검출하고자 하는 특징점의 개수
#     scaleFactor=1.2, # 스케일 계수는 이미지 피라미드를 설정합니다. 인수를 2로 지정할 경우, 이미지 크기가 절반이 되는 고전적인 이미지 피라미드를 의미
#     nlevels=8, # 피라미드 레벨은 이미지 피라미드의 레벨 수
#     edgeThreshold=31, # 엣지 임곗값은 이미지 테두리에서 발생하는 특징점을 무시하기 위한 경계의 크기를 나타냄
#     firstLevel=0, # 시작 피라미드 레벨은 원본 이미지를 넣을 피라미드의 레벨을 의미
#     WTA_K=2, # 비교점은 BRIEF 기술자가 구성하는 비교 비트를 나타냅 *2를 지정할 경우 이진 형식(0, 1)을 사용하며, 3의 값을 사용할 경우 3자 간 비교 결과로 (0, 1, 2)를 사용한다. 4의 값을 사용할 경우 4자 간 비교 결과로 (0, 1, 2, 3)을 사용합니다.
#     scoreType=cv2.ORB_HARRIS_SCORE, # 점수 방식은 피처의 순위를 매기는 데 사용되며, 해리스 코너(cv2.ORB_HARRIS_SCORE) 방식과 FAST(cv2.ORB_FAST_SCORE) 방식을 사용할 수 있습니다.
#     patchSize=31, # 패치 크기는 방향성을 갖는 BFIEF 기술자가 사용하는 개별 피처의 패치 크기
#     fastThreshold=20, # AST 임곗값은 FAST 검출기에서 사용되는 임곗값을 의미
# )
# # cv2.ORB_create(최대 피처 수, 스케일 계수, 피라미드 레벨, 엣지 임곗값, 시작 피라미드 레벨, 비교점, 점수 방식, 패치 크기, FAST 임곗값)

# # 각각의 이미지에 특징점 및 기술자 계산 메서드(orb.detectAndCompute)로 특징점 및 기술자 계산
# kp1, des1 = orb.detectAndCompute(gray, None) 
# # 특징점, 기술자 = orb.detectAndCompute(입력 이미지, 마스크)
# kp2, des2 = orb.detectAndCompute(target, None)
# # orb.detectAndCompute(거리 측정법, 교차 검사) : 거리 측정법은 질의 기술자(Query Descriptors)와 훈련 기술자(Train Descriptors)를 비교할 때 사용되는 거리 계산 측정법을 지정

# # *특징점은 좌표(pt), 지름(size), 각도(angle), 응답(response), 옥타브(octave), 클래스 ID(class_id)를 포함
# '''
# 좌표는 특징점의 위치를 알려주며, 지름은 특징점의 주변 영역을 의미합니다.
# 각도는 특징점의 방향이며, -1일 경우 방향이 없음을 나타냅니다.
# 응답은 피처가 존재할 확률로 해석하며, 옥타브는 특징점을 추출한 피라미드의 스케일을 의미합니다.
# 클래스 ID는 특징점에 대한 저장공간을 생성할 때 객체를 구분하기 위한 클러스터링한 객체 ID를 뜻합니다.
# 기술자는 각 특징점을 설명하기 위한 2차원 배열로 표현됩니다. 이 배열은 두 특징점이 같은지 판단할 때 사용됩니다.
# '''
# # 특징점과 기술자 검출이 완료되면, 전수 조사 매칭(Brute force matching)을 활용해 객체를 인식하거나 추적
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # 매치 함수를 활용해 최적의 메칭을 검출함
# matches = bf.match(des1, des2) # bf.detectAndCompute(기술자1, 기술자2)를 의미
# matches = sorted(matches, key=lambda x: x.distance)
# # *질의 색인(queryIdx), 훈련 색인(trainIdx), 이미지 색인(imgIdx), 거리(distance)로 구성

# # 반복문을 통하여 우수한 상위 100개에 대해서만 표시함
# for i in matches[:100]:
#     idx = i.queryIdx
#     x1, y1 = kp1[idx].pt
#     cv2.circle(src, (int(x1), int(y1)), 3, (255, 0, 0), 3)
#     print(x1,y1)

# dst_resize = test_def.ez_resizeing(src, width=1280)
# cv2.imshow('dst_resize', dst_resize)
# test_def.safe_run_imshow()
# cv2.waitKey()
