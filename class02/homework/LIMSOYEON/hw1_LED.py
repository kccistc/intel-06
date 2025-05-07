from iotdemo import FactoryController

#1.with FactoryController('dev/ttyACM0') as ctrl:

# with FactoryController('dev/ttyACM0') as ctrl:
#     ctrl.~~
#     ctrl.~~~

#2. ctrl = FactroyController('/dev/ttyACM0')
#ctrl = FactroyController('/dev/ttyACM0')
#ctrl.~~~

with FactoryController('/dev/ttyACM0') as ctrl:
    ctrl.red = True  # False : ON / True : OFF
    ctrl.orange = True
    ctrl.green = True
    ctrl.push_actuator(1)
    ctrl.push_actuator(2)

    red_state = True
    orange_state = True
    green_state = True

    while (1):
        cmd = input('input number : ')
        if cmd == '3':
            red_state = not red_state
            ctrl.red = red_state
        elif cmd == '4':
            orange_state = not orange_state
            ctrl.orange = orange_state
        elif cmd == '5':
            green_state = not green_state
            ctrl.green = green_state
        elif cmd == '7':
            ctrl.push_actuator(1)
        elif cmd == '8':
            ctrl.push_actuator(2)
        elif cmd == 'q':
            break