from iotdemo import FactoryController


with FactoryController('/dev/ttyACM0') as ctrl:
    while True:
        ctrl.red = False # ON
        ctrl.orange = True # OFF
        ctrl.green = False
        a=input()
        if a=='q' or a=='Q':
            break
        else:
            a = int(a)
            ctrl.push_actuator(a)