import cv2
import os

# Read from the first camera device
cap = cv2.VideoCapture(0)
w = 640#1280#1920
h = 480#720#1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
img_counter = 1
# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Display
    cv2.imshow("Camera",frame)
    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        # 파일 이름 겹치지 않도록 자동 증가
        while True:
            filename = f"{img_counter:03}.jpg"
            if not os.path.exists(filename):
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
                img_counter += 1
                break
            img_counter += 1