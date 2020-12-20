import cv2
import os

# open_cv imshow prevent freezing of imshow
# 환경에 따라서 윈도우가 프리징되는 현상을 막는다.
def safe_run_imshow():
    while True:
        key = cv2.waitKey(0)
        if key in [27, ord('q'), ord('Q')]:
            cv2.destroyAllWindows()

# 이미지를 열어보기 전에 리사이징을 쉽게 할 수 있다.
def ez_resizeing(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)