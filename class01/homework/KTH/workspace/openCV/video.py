import numpy as np
import cv2
import os

# Read from the recorded video file
cap = cv2.VideoCapture("ronaldinho.mp4")

img_counter = 1

# 동영상 파일이 성공적으로 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        cap = cv2.VideoCapture("ronaldinho.mp4")
        continue
        # print("Can't receive frame (stream end?). Exiting ...")
        # break

    # Display
    cv2.imshow("Frame",frame)

    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(42)
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

cap.release()
cv2.destroyAllWindows()