import threading
import time

def func1():
    for i in range(50):
        print('func1: ', i)
        time.sleep(2)

def func2():
    for i in range(50):
        print('func2: ', i)
        time.sleep(2)

threading.Thread(target=func1).start()
threading.Thread(target=func2).start()
