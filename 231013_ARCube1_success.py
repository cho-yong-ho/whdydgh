import cv2
import numpy as np

# 체커보드의 가로 및 세로 꼭지점 수 설정
rows = 4
cols = 7

# 카메라 해상도 설정
width = 640  # 원하는 너비로 설정
height = 480  # 원하는 높이로 설정

# 체커보드의 꼭지점 좌표 생성
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

# 이미지에서 체커보드 검출을 위한 리스트
objpoints = []  # 3D 꼭지점 좌표
imgpoints = []  # 2D 이미지 상의 꼭지점 좌표

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # 큐브 그리기
    for i in range(4):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[(i+1) % 4]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[4]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[1]), tuple(imgpts[5]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[2]), tuple(imgpts[6]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[7]), (0, 0, 255), 3)

    # 바닥면 그리기
    img = cv2.line(img, tuple(imgpts[4]), tuple(imgpts[5]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[5]), tuple(imgpts[6]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[6]), tuple(imgpts[7]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[7]), tuple(imgpts[4]), (0, 0, 255), 3)

    return img

# 비디오 캡처 또는 이미지 불러오기
cap = cv2.VideoCapture(0)  # 카메라를 사용하려면 0을 사용하거나 동영상 파일 경로를 지정할 수 있습니다.

# 카메라 해상도 설정
cap.set(3, width)  # 너비 설정
cap.set(4, height)  # 높이 설정

while True:
    ret, frame = cap.read()  # 비디오 프레임 읽기

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 체커보드 검출
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        objpoints.append(objp)
        imgpoints.append(corners)

        # 카메라 캘리브레이션 실행
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print('mtx')
        print(mtx)
        print(type(mtx))
        print('dist')
        print(dist)
        print(type(dist))
        print('rvecs')
        print(rvecs)
        print(type(rvecs))
        print('tvecs')
        print(tvecs)
        print(type(tvecs))

        # 3D 큐브 좌표 설정
        axis_length = rows-1  # 큐브의 변 길이
        axis_points = np.float32([[0, 0, 0], [0, axis_length, 0], [axis_length, axis_length, 0], [axis_length, 0, 0], [0, 0, -axis_length], [0, axis_length, -axis_length], [axis_length, axis_length, -axis_length], [axis_length, 0, -axis_length]])

        # 카메라의 위치와 방향 추정 (3D 큐브 위치 결정)
        _, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

        # 3D 큐브의 꼭지점 이미지 좌표 계산
        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)

        # 3D 큐브 그리기
        frame = draw_cube(frame, corners, imgpts)

    cv2.imshow('Checkerboard Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()