from iotdemo import FactoryController

# version 1. with FactoryController('/dev/ttyACM0') as ctrl:
# with FactoryController('/dev/ttyACM0') as ctrl:
#     ctrl.~~
#     ctrl.~~~

# version 2. ctrl = FactoryController('/dev/ttyACM0')
#ctrl = FactoryController('dev/ttyACM0')
#ctrl.~~~


with FactoryController('/dev/ttyACM0') as ctrl:
    red_state=True
    orange_state=True
    green_state=True
    conveyor_state=True
    while(1):
        led=input()
        if led=='3':
            red_state=not red_state
            ctrl.red=red_state
        elif led=='4':
            orange_state=not orange_state
            ctrl.orange=ctrl.orange
            print(ctrl.orange)
        elif led=='5':
            green_state=not green_state
            ctrl.green=green_state
        elif led=='6':
            conveyor_state=not conveyor_state
            ctrl.conveyor=conveyor_state
        elif led=='7':
            ctrl.push_actuator(1)
        elif led=='8':
            ctrl.push_actuator(2)
        elif led=='q':
            break
