import cv2 as cv
import numpy as np
import math 

vfov = 120 # 수직 시야각(degree 단위)
hfov = 150 # 수평 시야각(degree 단위)

def erp2front(image):
    f = height / (2 * math.pi) # focal length 계산
    hp = int(2 * f * math.tan(math.radians(vfov/2)) + 0.5) # 투영면의 수직 해상도
    wp = int(2 * f * math.tan(math.radians(hfov/2)) + 0.5) # 투영면의 수평 해상도
    dst = np.zeros((hp, wp, 3), dtype=np.uint8) # 투영할 평면 정의

    dst_cx = wp / 2 # 투영면의 중심 x좌표
    dst_cy = hp / 2 # 투영면의 중심 y좌표

    for x in range(wp):
        xth = math.atan((x - dst_cx) / f) # 투영면의 x좌표에 해당하는 ERP에서의 수평각
        erp_x = int((xth + math.pi) * width / (2 * math.pi) + 0.5) # 투영면의 x좌표에 해당하는 ERP이미지의 x좌표
        yf = f / math.cos(xth)
        for y in range(hp):
            yth = math.atan((y - dst_cy) / yf) # 투영면의 y좌표에 해당하는 ERP에서의 수직각
            erp_y = int(yth * height / math.pi + height / 2 + 0.5) # 투영면의 y좌표에 해당하는 ERP이미지의 y좌표
            dst[y][x] = image[erp_y][erp_x] # 투영면 좌표에 해당하는 ERP이미지 좌표값 저장
    return dst

def erp2top(image):
    f = height / (2 * math.pi) # focal length 계산
    hp = int(2 * f * math.tan(math.radians(vfov/2)) + 0.5) # 투영면의 수직 해상도
    wp = int(2 * f * math.tan(math.radians(hfov/2)) + 0.5) # 투영면의 수평 해상도
    dst = np.zeros((hp, wp, 3), dtype=np.uint8) # 투영할 평면 정의

    dst_cx = wp / 2 # 투영면의 중심 x좌표
    dst_cy = hp / 2 # 투영면의 중심 y좌표

    for x in range(wp):
        for y in range(hp):
            # 투영면의 (x, y)좌표의 위치에 따른 ERP에서의 수평각 계산
            if x - dst_cx >= 0 and y - dst_cy > 0:
                xth = math.atan((x - dst_cx) / (y - dst_cy))
            elif x - dst_cx < 0 and y - dst_cy > 0:
                xth = math.atan((x - dst_cx) / (y - dst_cy))
            elif x - dst_cx < 0 and y - dst_cy < 0:
                xth = - math.pi + abs(math.atan((x - dst_cx) / (y - dst_cy)))
            elif x - dst_cx >= 0 and y - dst_cy < 0:
                xth = math.pi - abs(math.atan((x - dst_cx) / (y - dst_cy)))
            elif x - dst_cx >= 0 and y - dst_cy == 0:
                xth = math.pi * 3 / 2
            elif x - dst_cx < 0 and y - dst_cy == 0:
                xth = math.pi / 2
            yth = math.atan(math.sqrt((x - dst_cx)**2 + (y - dst_cy)**2) / f) # 투영면의 y좌표에 해당하는 ERP에서의 수직각
            erp_x = int(xth * width / (2 * math.pi) + 0.5) # 투영면의 x좌표에 해당하는 ERP이미지의 x좌표
            erp_y = int(height - yth * height / math.pi + 0.5) # 투영면의 y좌표에 해당하는 ERP이미지의 y좌표
            if erp_y == height:
                erp_y -= erp_y
            dst[y][x] = image[erp_y][erp_x] # 투영면 좌표에 해당하는 ERP이미지 좌표값 저장
    return dst

img = cv.imread("erp.png") # ERP 이미지 불러오기

height = img.shape[0] # 불러온 ERP 이미지의 수직 해상도
width = img.shape[1] # 불러온 ERP 이미지의 수평 해상도

front_view = erp2front(img)
top_view = erp2top(img)

cv.imshow("Raw image",img)
cv.imshow("Front View", front_view) # 정면 이미지 표시
cv.imshow("Top View", top_view) # top view 이미지 표시

if cv.waitKey(0) == ord('q'): # q 누르면 창 꺼짐
   cv.destroyAllWindows()
