import cv2
import numpy as np

# 체커보드의 가로 및 세로 꼭지점 수 설정
rows = 4
cols = 7

# 카메라 파라미터
fx = 563.296738
fy = 563.296738
cx = 320.000000
cy = 240.000000
k1 = 0.256847
k2 = -1.085800
p1 = 0.001880
p2 = -0.009034
k3 = 0

# 카메라 매트릭스
mtx = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])

# 왜곡 계수
dist = np.array([k1, k2, p1, p2, k3])

# 체커보드의 꼭지점 좌표 생성
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

# AR 큐브 변의 길이
axis_length = rows-1

# AR 큐브 월드 좌표 설정
axis_points = np.float32([[0, 0, 0], [0, axis_length, 0], 
                    [axis_length, axis_length, 0], [axis_length, 0, 0],
                    [0, 0, -axis_length], [0, axis_length, -axis_length], 
                    [axis_length, axis_length, -axis_length], [axis_length, 0, -axis_length]])

def draw_cube(img, corners, imgpts):
    # 큐브 아랫면 그리기
    for i in range(4):
        # AR 큐브 아랫면 그리기
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[(i+1) % 4]), (0, 0, 255), 1)
        # AR 큐브 측면 모서리 그리기
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i+4]), (0, 0, 255), 1)
        # AR 큐브 윗면 그리기
        img = cv2.line(img, tuple(imgpts[i+4]), tuple(imgpts[4 + ((i+1) % 4)]), (0, 0, 255), 1)
 
    return img

# 웹캠 사용
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # 비디오 프레임 읽기

    if not ret:
        break

    # 체커보드 검출
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        # 체커보드 꼭지점을 일정 수준 이상의 정확도로 탐지
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 이미지에서 체커보드 검출을 위한 리스트
        objpoints = []  # 체커보드 꼭지점의 월드 좌표
        imgpoints = []  # 체커보드 꼭지점의 픽셀 좌표

        # 각 리스트에 체스보드 꼭지점 좌표 추가
        objpoints.append(objp)
        imgpoints.append(corners)
    
        # 카메라의 위치와 방향 추정 (3D 큐브 위치 결정)
        _, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)

        # 회전 벡터를 회전 매트릭스로 변환
        R, _ = cv2.Rodrigues(rvec)

        # 투영 매트릭스 계산 (3x4 matrix)
        # proj_matrix = np.dot(mtx, np.hstack((R, tvec.reshape(3,1))))
        proj_matrix = mtx @ np.hstack((R, tvec.reshape(3,1)))

        # AR 큐브 월드 좌표를 2D 픽셀 좌표로 변환
        axis_points_h = np.vstack((axis_points.T, np.ones(8)))  # 1을 추가한 homogeneous 좌표로 변환
        imgpts_homogeneous = proj_matrix @ axis_points_h
        imgpts = imgpts_homogeneous[:2, :] / imgpts_homogeneous[2, :]
        imgpts = np.int32(imgpts.T)
        
        # 3D 큐브 그리기
        frame = draw_cube(frame, corners, imgpts)
    
    # 화면 출력
    cv2.imshow('Checkerboard Detection', frame)

    # q를 누르면 프로그램 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
