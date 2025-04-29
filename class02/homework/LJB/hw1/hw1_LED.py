from iotdemo import FactoryController

# 1. with FactoryController('/dev/ttyACM0') as crtl:
# with FactoryController('/dev/ttyACM0') as ctrl:
#     ctrl.~~
#     ctrl.~~~
# 2. ctrl = FactoryController('/dev/ttyACM0')
# ctrl = FactoryController('/dev/ttyACM0')
# ctrl.~~~

with FactoryController('/dev/ttyACM0') as ctrl:
    while True:
        ctrl.red = False # 이게 On이다
        ctrl.orange = True # 이게 Off이다
        ctrl.green = False
        a=input()
        if a=='q' or a=='Q':
            break
        else:
            a = int(a)
            ctrl.push_actuator(a)

