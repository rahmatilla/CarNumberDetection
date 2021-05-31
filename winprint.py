"""
import win32ui
dc = win32ui.CreateDC()
dc.CreatePrinterDC()
dc.StartDoc('My Python Document')

dc.StartPage()
dc.TextOut(100,100, 'Python Prints!')
dc.TextOut(100,100, 'Python Prints!')
#dc.MoveTo(100, 102)
#dc.LineTo(200, 102)
dc.EndPage()
dc.EndDoc()
"""

import os
import datetime


number = '01T101LB'
time = datetime.datetime.now()

f = open("TestFile.txt", "w")
f.write('Car Number: ' + number + '\nEnter Time: ' + str(time)[:19])
f.close()
os.startfile("TestFile.txt", "print")