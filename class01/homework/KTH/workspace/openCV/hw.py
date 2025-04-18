import torch
import cv2
import os

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 캡처 카운터 딕셔너리
capture_counters = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # 객체 인식
    results = model(frame)
    annotated_frame = results.render()[0]

    # 결과 정보 추출
    detections = results.pred[0]  # [x1, y1, x2, y2, conf, cls]
    labels = results.names

    largest_area = 0
    largest_obj_name = None

    # 가장 큰 박스 찾기
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_obj_name = labels[int(cls)]

    # 화면 출력
    cv2.imshow("YOLOv5 Webcam", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        if largest_obj_name is None:
            print("No object detected. Capture skipped.")
            continue

        # 파일 저장
        name = largest_obj_name
        count = capture_counters.get(name, 1)
        while True:
            filename = f"{name}_{count:03}.jpg"
            if not os.path.exists(filename):
                cv2.imwrite(filename, frame)
                print(f"Captured: {filename}")
                capture_counters[name] = count + 1
                break
            count += 1

cap.release()
cv2.destroyAllWindows()
