# Following libraries required:
# pip install PyOpenGL PyQt5 numpy pyqtgraph numpy-stl pandas

import sys
import math
import numpy as np
import itertools
import pyqtgraph.opengl as gl
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
# from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from stl import mesh

# class GLWidget(QGLWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.rotate_angle = 0
#         self.x_pos = 0
#         self.y_pos = 0
#         self.z_pos = -5.0
#         # self.timer = self.startTimer(10)  # Timer for animation

#     def initializeGL(self):
#         glClearColor(0.0, 0.0, 0.0, 1.0)
#         glEnable(GL_DEPTH_TEST)

#     def resizeGL(self, width, height):
#         glViewport(0, 0, width, height)
#         glMatrixMode(GL_PROJECTION)
#         glLoadIdentity()
#         aspect_ratio = width / height
#         fov_degrees = 45.0
#         near_clip = 0.1
#         far_clip = 100.0

#         # Calculate perspective projection matrix manually
#         top = near_clip * math.tan(math.radians(fov_degrees / 2.0))
#         bottom = -top
#         left = bottom * aspect_ratio
#         right = top * aspect_ratio

#         glFrustum(left, right, bottom, top, near_clip, far_clip)
#         glMatrixMode(GL_MODELVIEW)

#     def resetGL(self):
#         self.x_pos = 0
#         self.y_pos = 0
#         self.z_pos = -5.0

#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         glLoadIdentity()
#         glTranslatef(self.x_pos, self.y_pos, self.z_pos)

#         self.update()

#     def paintGL(self):
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         glLoadIdentity()
#         glTranslatef(self.x_pos, self.y_pos, self.z_pos)
#         # glTranslatef(0.0, 0.0, -5.0)
#         # glRotatef(self.rotate_angle, 1, 1, 1)

#         glBegin(GL_QUADS)
#         glColor3f(1.0, 0.0, 0.0)  # Red
#         glVertex3f(1.0, 0.5, -0.5)
#         glVertex3f(-1.0, 0.5, -0.5)
#         glVertex3f(-1.0, 0.5, 0.5)
#         glVertex3f(1.0, 0.5, 0.5)

#         glColor3f(0.0, 1.0, 0.0)  # Green
#         glVertex3f(1.0, -0.5, 0.5)
#         glVertex3f(-1.0, -0.5, 0.5)
#         glVertex3f(-1.0, -0.5, -0.5)
#         glVertex3f(1.0, -0.5, -0.5)

#         glColor3f(0.0, 0.0, 1.0)  # Blue
#         glVertex3f(1.0, 0.5, 0.5)
#         glVertex3f(-1.0, 0.5, 0.5)
#         glVertex3f(-1.0, -0.5, 0.5)
#         glVertex3f(1.0, -0.5, 0.5)

#         glColor3f(1.0, 1.0, 0.0)  # Yellow
#         glVertex3f(1.0, -0.5, -0.5)
#         glVertex3f(-1.0, -0.5, -0.5)
#         glVertex3f(-1.0, 0.5, -0.5)
#         glVertex3f(1.0, 0.5, -0.5)

#         glColor3f(0.0, 1.0, 1.0)  # Cyan
#         glVertex3f(-1.0, 0.5, 0.5)
#         glVertex3f(-1.0, 0.5, -0.5)
#         glVertex3f(-1.0, -0.5, -0.5)
#         glVertex3f(-1.0, -0.5, 0.5)

#         glColor3f(1.0, 0.0, 1.0)  # Magenta
#         glVertex3f(1.0, 0.5, -0.5)
#         glVertex3f(1.0, 0.5, 0.5)
#         glVertex3f(1.0, -0.5, 0.5)
#         glVertex3f(1.0, -0.5, -0.5)
#         glEnd()

#         self.rotate_angle += 1
#         self.x_pos += 0.1
#         self.y_pos += 0.1
#         # self.z_pos -= 0.1

#     def timerEvent(self, event):
#         self.update()

# def create_cube():
#     vertexes = np.array(list(itertools.product(range(2),repeat=3)))

#     faces = []

#     for i in range(2):
#         temp = np.where(vertexes==i)
#         for j in range(3):
#             temp2 = temp[0][np.where(temp[1]==j)]
#             for k in range(2):
#                 faces.append([temp2[0],temp2[1+k],temp2[3]])

#     faces = np.array(faces)

#     colors = np.array([[1,0,0,1] for i in range(12)])

#     cube = gl.GLMeshItem(vertexes=vertexes, faces=faces, faceColors=colors,
#                         drawEdges=True, edgeColor=(0, 0, 0, 1))

#     return cube

def load_drone():
    filename = "stl_files\\uploads_files_4318187_Drone.stl"
    m = mesh.Mesh.from_file(filename)
    shape = m.points.shape
    points = m.points.reshape(-1, 3)
    faces = np.arange(points.shape[0]).reshape(-1, 3)
    return points, faces

# Main GUI

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.animation_running = False
        self.init_ui()
        self.flight_plan = []

    def init_ui(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('OptiDrone Flight Simulator')

        layout = QVBoxLayout()
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # self.glWidget = GLWidget()
        # layout.addWidget(self.glWidget)

        self.viewer = gl.GLViewWidget()
        layout.addWidget(self.viewer)

        self.glGrid = gl.GLGridItem()
        self.glGrid.setSize(200, 200)
        self.glGrid.setSpacing(5, 5)
        self.viewer.addItem(self.glGrid)

        # cube = create_cube()
        # self.viewer.addItem(cube)

        drone_points, drone_faces = load_drone()
        meshdata = gl.MeshData(vertexes=drone_points, faces=drone_faces)
        self.mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
        self.viewer.addItem(self.mesh)

        self.glAxes = gl.GLAxisItem()
        self.glAxes.setSize(100, 100, 100)
        self.viewer.addItem(self.glAxes)

        btn_start_stop = QPushButton('Start/Stop', self)
        btn_start_stop.clicked.connect(self.start_flight)
        layout.addWidget(btn_start_stop)

        btn_reset_flight = QPushButton('Reset', self)
        btn_reset_flight.clicked.connect(self.reset_flight)
        layout.addWidget(btn_reset_flight)

        btn_import_flight = QPushButton('Import Flight Plan', self)
        btn_import_flight.clicked.connect(self.import_flight)
        layout.addWidget(btn_import_flight)

        self.show()

    def import_flight(self):
        self.flight_plan_in = pd.read_csv('predictions_test.csv')

        self.flight_plan_x = self.flight_plan_in["TX"][1]
        self.flight_plan_y = self.flight_plan_in["TY"][1]
        self.flight_plan_z = self.flight_plan_in["TZ"][1]
        print(self.flight_plan_x)

        self.flight_plan.clear()
        
        # curr_row = (self.flight_plan_x, self.flight_plan_y, self.flight_plan_z)
        curr_row = (0, 0, 0)
        self.flight_plan.append(curr_row)
        prev_x = self.flight_plan_x
        prev_y = self.flight_plan_y
        prev_z = self.flight_plan_z

        factor = 10

        for idx, row in self.flight_plan_in.iterrows():
            if idx == 0 or idx == 1:
                pass
            else:
                curr_row = ((row["TX"] - prev_x)/factor, (row["TY"] - prev_y)/factor, (row["TZ"] - prev_z)/factor)
                # curr_row = ((row["TX"] - self.flight_plan_x)/factor, (row["TY"] - self.flight_plan_y)/factor, (row["TZ"] - self.flight_plan_z)/factor)
                # curr_row = (row["TX"]/factor, row["TY"]/factor, row["TZ"]/factor)
                self.flight_plan.append(curr_row)

                prev_x, prev_y, prev_z = row["TX"], row["TY"], row["TZ"]

        print(self.flight_plan)
        self.curr_frame = 0


    def start_flight(self):
        # if self.animation_running:
        #     print('Stopping animation...')
        #     # self.glWidget.killTimer(self.glWidget.timer)
        #     self.animation_running = False
        # else:
        #     print('Starting animation...')
        #     # self.glWidget.timer = self.glWidget.startTimer(50)  # Timer for animation
        #     self.animation_running = True

        #     # while(self.animation_running):
        #     #     self.mesh.translate(0.1,0,0)
        #     #     app.processEvents()

        #     for frame in self.flight_plan:
        #         print(frame)
        #         self.mesh.translate(frame[0], frame[1], frame[2])
        #         app.processEvents()

        for frame in self.flight_plan:
            print(frame)
            self.mesh.translate(frame[0], frame[1], frame[2])
            app.processEvents()

    def reset_flight(self):
        print('Resetting animation...')
        # if self.animation_running:
        #     # self.glWidget.killTimer(self.glWidget.timer)
        #     self.animation_running = False
        #     self.mesh.translate(0, 0, 0)
        #     app.processEvents()
        self.mesh.resetTransform()

        # self.glWidget.resetGL()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())
