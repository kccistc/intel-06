#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue # Queue = queue.Queue
from time import sleep

import cv2
import numpy as np
#from openvino.inference_engine import IECore

from iotdemo import FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):
    # TODO: MotionDetector
    cMotion = MotionDetector()
    cMotion.load_preset('resources/motion.cfg')
    # TODO: Load and initialize OpenVINO

    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('resources/conveyor.mp4')
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info #이름:VIDEO:Cam1, data:frame
        q.put(("VIDEO:Cam1 live", frame))
        # TODO: Motion detect
        detected = cMotion.detect(frame)
        if detected is None:
            continue
        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected", detected))
        # abnormal detect
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #reshaped = detected[:, :, [2, 1, 0]]
        #np_data = np.moveaxis(reshaped, -1, 0)
        #preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        #batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # TODO: Inference OpenVINO

        # TODO: Calculate ratios
        #print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    cMotion = MotionDetector()
    cMotion.load_preset('resources/motion.cfg')
    # TODO: ColorDetector
    #cColor = ColorDetector()
    #cColor.load_preset('resources/color_1.cfg')
    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

        # TODO: Detect motion
        detected = cMotion.detect(frame)
        if detected is None:
            continue
        
        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))

        # TODO: Detect color
        #output = cColor.detect(detected)[0] # 더 높은 확률을 출력하게 하기 위해
        #name, ratio = cColor.detect(detected)[0]
        #print(name, ratio)
        # TODO: Compute ratio
        #print(f"{name}: {ratio:.2f}%")
        print(f"{name}: {ratio:.2f}%")
        # TODO: Enqueue to handle actuator 2
        #if name == 'blue': # 파란색이면 액추에이터 2번에 푸시해준다
        #    q.put(('PUSH',2))

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

    # TODO: HW2 Create a Queue
    q = Queue()
    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.
    thread1 = threading.Thread(target=thread_cam1, args=(q,))
    thread2 = threading.Thread(target=thread_cam2, args=(q,))

    thread1.start()
    thread2.start()
    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue.
            # You might need to properly handle exceptions.
            # de-queue name and data
            try:
                event = q.get_nowait()  # queue에서 바로 꺼내려고 함 (비어있으면 예외 발생)
            except Empty:
                continue
            name, data = event # 꺼낸 event를 name과 data로 분리 name:VIDEO:Cam Live, data:frame
            
            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if name[-4:]=='live':
                imshow(name[6:],data)
            elif name[-8:]=='detected':
                imshow(name[6:],data)
            # TODO: Control actuator, name == 'PUSH'

            elif name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
