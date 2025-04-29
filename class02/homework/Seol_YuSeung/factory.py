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
    model_name = "model"
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

        
        q.put(('VIDEO:Cam1 live', frame))
        
        
        detected = Motion1.detect(frame)
        if detected is None:
            continue
            
        
        q.put(('VIDEO:Cam1 detected', detected))
        
        
        frame = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
        reshaped = frame[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        
        results = compiled_model([batch_tensor])[output_layer]
        
       
        x_ratio = results[0][0] * 100
        circle_ratio = results[0][1] * 100
        
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        
        if x_ratio > 60:
            q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    
    Motion2 = MotionDetector()
    Motion2.load_preset('resources/motion.cfg')

    
    Color2 = ColorDetector()
    Color2.load_preset('resources/color.cfg')

    
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        
        q.put(('VIDEO:Cam2 live', frame))
        
        
        detected = Motion2.detect(frame)
        if detected is None:
            continue
            
        
        q.put(('VIDEO:Cam2 detected', detected))
        
        
        name, ratio = Color2.detect(detected)[0]
        
        
        print(f"{name}: {ratio*100:.2f}%")

        
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

    
    q = Queue()

    
    cam1 = threading.Thread(target=thread_cam1, args=(q,))
    cam2 = threading.Thread(target=thread_cam2, args=(q,))
    cam1.start()
    cam2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            
            try:
                event = q.get_nowait()
            except Empty:
                continue
                
            name, data = event

            
            if name[-4:] == 'live':
                imshow(name[6:], data, pos=None)
            elif name[-8:] == 'detected':
                imshow(name[6:], data, pos=None)
            
            elif name == 'PUSH':
                ctrl.push_actuator(data)
            elif name == 'DONE':
                FORCE_STOP = True

            q.task_done()
            
    cam1.join()
    cam2.join()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()