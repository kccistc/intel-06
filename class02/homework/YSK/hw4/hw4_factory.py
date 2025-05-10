#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from iotdemo import FactoryController, MotionDetector, ColorDetector

FORCE_STOP = False


def thread_cam1(q):
    # 모션 감지기 초기화
    Motion1 = MotionDetector()
    Motion1.load_preset('resources/motion.cfg')

    # OpenVINO 로드 및 초기화
    device = "CPU"

    base_artifacts_dir = Path("resources").expanduser()
    model_name = "exported_model"
    model_xml_name = f"{model_name}.xml"
    model_xml_path = base_artifacts_dir / model_xml_name

    core = ov.Core()
    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name=device)
    output_layer = compiled_model.output(0)
    
    # 비디오 파일 열기
    cap = cv2.VideoCapture('resources/conveyor.mp4')
    
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # 라이브 프레임 큐에 추가
        q.put(('VIDEO:Cam1 live', frame))
        
        # 모션 감지 수행
        detected = Motion1.detect(frame)
        if detected is None:
            continue
            
        # 감지된 프레임 큐에 추가
        q.put(('VIDEO:Cam1 detected', detected))
        
        # 이미지 전처리
        frame = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
        reshaped = frame[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # OpenVINO 추론 실행
        results = compiled_model([batch_tensor])[output_layer]
        
        # 결과 처리 (모델에 따라 조정 필요)
        # 예시: 결과에서 x와 원 비율 추출
        x_ratio = results[0][1] * 100
        circle_ratio = results[0][0] * 100
        
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # 액추에이터 1 제어
        if circle_ratio < np.abs(x_ratio):  # 임계값은 적절히 조정
            q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # 모션 감지기 초기화
    Motion2 = MotionDetector()
    Motion2.load_preset('resources/motion.cfg')

    # 색상 감지기 초기화
    Color2 = ColorDetector()
    Color2.load_preset('resources/color.cfg')

    # 비디오 파일 열기
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # 라이브 프레임 큐에 추가
        q.put(('VIDEO:Cam2 live', frame))
        
        # 모션 감지 수행
        detected = Motion2.detect(frame)
        if detected is None:
            continue
            
        # 감지된 프레임 큐에 추가
        q.put(('VIDEO:Cam2 detected', detected))
        
        # 색상 감지 수행
        name, ratio = Color2.detect(detected)[0]
        
        # 비율 계산 및 출력
        print(f"{name}: {ratio*100:.2f}%")

        # 액추에이터 2 제어
        if name == 'blue':
            q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # 큐 생성
    q = Queue()

    # 카메라 스레드 생성 및 시작
    cam1 = threading.Thread(target=thread_cam1, args=(q,))
    cam2 = threading.Thread(target=thread_cam2, args=(q,))
    cam1.start()
    cam2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # 큐에서 이벤트 가져오기
            try:
                event = q.get_nowait()
            except Empty:
                continue
                
            name, data = event

            # 라이브 및 감지된 비디오 표시
            if name[-4:] == 'live':
                imshow(name[6:], data,)
            elif name[-8:] == 'detected':
                imshow(name[6:], data)
            # 액추에이터 제어
            elif name == 'PUSH':
                ctrl.push_actuator(data)
            elif name == 'DONE':
                FORCE_STOP = True

            q.task_done()
            
    # 스레드 종료 대기
    cam1.join()
    cam2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
