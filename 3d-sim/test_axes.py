from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QApplication,  QVBoxLayout, QWidget, QPushButton
import pyqtgraph.opengl as gl
import numpy as np
import itertools

def start_flight():
    print("starting flight")

def reset_flight():
    print("resetting flight")

app = QApplication.instance()
if app is None:
    app = QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('A cube')

layout = QVBoxLayout()

layout.addWidget(w)

vertexes = np.array(list(itertools.product(range(2),repeat=3)))

faces = []

for i in range(2):
    temp = np.where(vertexes==i)
    for j in range(3):
        temp2 = temp[0][np.where(temp[1]==j)]
        for k in range(2):
            faces.append([temp2[0],temp2[1+k],temp2[3]])

faces = np.array(faces)

colors = np.array([[1,0,0,1] for i in range(12)])


cube = gl.GLMeshItem(vertexes=vertexes, faces=faces, faceColors=colors,
                     drawEdges=True, edgeColor=(0, 0, 0, 1))

w.addItem(cube)

glGrid = gl.GLGridItem()
glGrid.setSize(200, 200)
glGrid.setSpacing(5, 5)
w.addItem(glGrid)

glAxes = gl.GLAxisItem()
glAxes.setSize(100, 100, 100)
w.addItem(glAxes)

btn_start_stop = QPushButton('Start/Stop', w)
btn_start_stop.clicked.connect(start_flight)
layout.addWidget(btn_start_stop)

btn_reset_flight = QPushButton('Reset', w)
btn_reset_flight.clicked.connect(reset_flight)
layout.addWidget(btn_reset_flight)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QApplication.instance().exec_()