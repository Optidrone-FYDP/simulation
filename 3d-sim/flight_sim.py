# Following libraries required:
# pip install PyOpenGL PyQt5 numpy pyqtgraph numpy-stl pandas

import os
import sys
import time
import numpy as np
import pyqtgraph.opengl as gl
import pandas as pd
from PyQt5.QtWidgets import *
from OpenGL.GL import *
from stl import mesh

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.drone_mesh_filename = ""
        self.flight_plan_filename = ""
        self.flight_plan_filename_2 = ""
        self.flight_plan = []
        self.flight_plan_2 = []
        self.time_delay = 0.10
        self.time_delay_display = 0

        self.flying = False
        self.curr_frame = 0

        self.init_ui()


    def init_ui(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('OptiDrone Flight Simulator')

        self.error_dialog = QErrorMessage()

        layout = QGridLayout()
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # ROW 0

        self.viewer = gl.GLViewWidget()
        layout.addWidget(self.viewer, 0, 0, 1, 3)

        self.glGrid = gl.GLGridItem()
        self.glGrid.setSize(200, 200)
        self.glGrid.setSpacing(5, 5)
        self.viewer.addItem(self.glGrid)

        self.glAxes = gl.GLAxisItem()
        self.glAxes.setSize(100, 100, 100)
        self.viewer.addItem(self.glAxes)

        # ROW 1

        btn_import_drone = QPushButton('Import Drone Mesh', self)
        btn_import_drone.clicked.connect(self.import_drone_mesh)
        layout.addWidget(btn_import_drone, 1, 0, 1, 1)

        self.lbl_import_drone = QLineEdit(self)
        self.lbl_import_drone.setText("STL file path...")
        self.lbl_import_drone.setReadOnly(True)
        layout.addWidget(self.lbl_import_drone, 1, 1, 1, 2)

        # ROW 2

        btn_import_flight = QPushButton('Import Flight Plan', self)
        btn_import_flight.clicked.connect(self.import_flight)
        layout.addWidget(btn_import_flight, 2, 0, 1, 1)

        self.lbl_import_flight = QLineEdit(self)
        self.lbl_import_flight.setText("CSV file path...")
        self.lbl_import_flight.setReadOnly(True)
        layout.addWidget(self.lbl_import_flight, 2, 1, 1, 2)

        # ROW 3

        btn_import_flight_2 = QPushButton('Import Second Flight Plan (Optional)', self)
        btn_import_flight_2.clicked.connect(self.import_flight_2)
        layout.addWidget(btn_import_flight_2, 3, 0, 1, 1)

        self.lbl_import_flight_2 = QLineEdit(self)
        self.lbl_import_flight_2.setText("CSV file path...")
        self.lbl_import_flight_2.setReadOnly(True)
        layout.addWidget(self.lbl_import_flight_2, 3, 1, 1, 2)

        # ROW 4

        self.btn_start_stop = QPushButton('Start/Stop Flight', self)
        self.btn_start_stop.clicked.connect(self.start_stop_flight)
        layout.addWidget(self.btn_start_stop, 4, 0, 1, 3)

        # ROW 5

        btn_slow_down = QPushButton('Decrease speed by 0.10', self)
        btn_slow_down.clicked.connect(self.slow_down)
        layout.addWidget(btn_slow_down, 5, 0)

        self.lbl_speed = QLineEdit(self)
        self.lbl_speed.setText("0.00 (Default Speed)")
        self.lbl_speed.setReadOnly(True)
        layout.addWidget(self.lbl_speed, 5, 1)

        btn_speed_up = QPushButton('Increase speed by 0.10', self)
        btn_speed_up.clicked.connect(self.speed_up)
        layout.addWidget(btn_speed_up, 5, 2)

        # ROW 6

        btn_reset_all = QPushButton('Reset All', self)
        btn_reset_all.clicked.connect(self.reset_all)
        layout.addWidget(btn_reset_all, 6, 0, 1, 1)

        btn_reset_flight = QPushButton('Reset Flight', self)
        btn_reset_flight.clicked.connect(self.reset_flight)
        layout.addWidget(btn_reset_flight, 6, 1, 1, 2)

        self.show()


    def import_drone_mesh(self):

        self.lbl_import_drone.setText("Importing...")
        self.drone_mesh_filename, filter = QFileDialog.getOpenFileName(self,"Select an STL file...", self.curr_dir, "STL Files (*.stl)")

        if not self.drone_mesh_filename:
            self.lbl_import_drone.setText("No drone mesh selected.")
            return

        m = mesh.Mesh.from_file(self.drone_mesh_filename)
        shape = m.points.shape
        points = m.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)

        self.meshdata = gl.MeshData(vertexes=points, faces=faces)
        self.mesh = gl.GLMeshItem(meshdata=self.meshdata, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
        self.viewer.addItem(self.mesh)

        self.lbl_import_drone.setText(self.drone_mesh_filename)


    def import_flight(self):
        if not self.drone_mesh_filename:
            self.error_dialog.showMessage('No drone mesh imported!')
            return

        self.lbl_import_flight.setText("Reading flight plan...")
        self.flight_plan_filename, filter = QFileDialog.getOpenFileName(self,"Select a CSV file...", self.curr_dir, "CSV Files (*.csv)")

        if not self.flight_plan_filename:
            self.lbl_import_flight.setText("No flight plan selected.")
            return

        self.flight_plan_in = pd.read_csv(self.flight_plan_filename)

        factor = 100

        self.flight_plan_x = self.flight_plan_in["TX"][1] / factor
        self.flight_plan_y = self.flight_plan_in["TY"][1] / factor
        self.flight_plan_z = self.flight_plan_in["TZ"][1] / factor
        print(self.flight_plan_x)

        self.flight_plan.clear()

        curr_row = (self.flight_plan_x, self.flight_plan_y, self.flight_plan_z)
        self.flight_plan.append(curr_row)
        prev_x = self.flight_plan_x
        prev_y = self.flight_plan_y
        prev_z = self.flight_plan_z

        for idx, row in self.flight_plan_in.iterrows():
            if idx == 0 or idx == 1:
                pass
            else:
                curr_row = ((row["TX"] - prev_x)/factor, (row["TY"] - prev_y)/factor, (row["TZ"] - prev_z)/factor)
                self.flight_plan.append(curr_row)

                prev_x, prev_y, prev_z = row["TX"], row["TY"], row["TZ"]

        print(self.flight_plan)

        self.lbl_import_flight.setText(self.flight_plan_filename)


    def import_flight_2(self):
        if not self.drone_mesh_filename:
            self.error_dialog.showMessage('No drone mesh imported!')
            return

        self.mesh_2 = gl.GLMeshItem(meshdata=self.meshdata, smooth=True, drawFaces=False, drawEdges=True)
        self.viewer.addItem(self.mesh_2)

        self.lbl_import_flight_2.setText("Reading flight plan...")
        self.flight_plan_filename_2, filter = QFileDialog.getOpenFileName(self,"Select a CSV file...", self.curr_dir, "CSV Files (*.csv)")

        if not self.flight_plan_filename_2:
            self.lbl_import_flight_2.setText("No flight plan selected.")
            return

        self.flight_plan_in_2 = pd.read_csv(self.flight_plan_filename_2)

        factor = 100

        self.flight_plan_x_2 = self.flight_plan_in_2["TX"][1] / factor
        self.flight_plan_y_2 = self.flight_plan_in_2["TY"][1] / factor
        self.flight_plan_z_2 = self.flight_plan_in_2["TZ"][1] / factor

        self.flight_plan_2.clear()

        curr_row = (self.flight_plan_x_2, self.flight_plan_y_2, self.flight_plan_z_2)
        self.flight_plan_2.append(curr_row)
        prev_x = self.flight_plan_x_2
        prev_y = self.flight_plan_y_2
        prev_z = self.flight_plan_z_2

        for idx, row in self.flight_plan_in_2.iterrows():
            if idx == 0 or idx == 1:
                pass
            else:
                curr_row = ((row["TX"] - prev_x)/factor, (row["TY"] - prev_y)/factor, (row["TZ"] - prev_z)/factor)
                self.flight_plan_2.append(curr_row)

                prev_x, prev_y, prev_z = row["TX"], row["TY"], row["TZ"]

        print(self.mesh_2)

        self.lbl_import_flight_2.setText(self.flight_plan_filename_2)


    def start_stop_flight(self):
        if not self.drone_mesh_filename:
            self.error_dialog.showMessage('No drone mesh imported!')
            return

        if not self.flight_plan_filename:
            self.error_dialog.showMessage('No flight plan imported!')
            return

        if self.flying:
            self.flying = False
            self.btn_start_stop.setText("Start")
        else:
            self.flying = True
            self.btn_start_stop.setText("Stop")

        second_flight = False
        if self.flight_plan_filename_2:
            second_flight = True

        if self.curr_frame >= len(self.flight_plan):
            self.curr_frame = 0

        if len(self.flight_plan_2) < len(self.flight_plan):
            for i in range(0, (len(self.flight_plan) - len(self.flight_plan_2)) + 1):
                blank_row = (0, 0, 0)
                self.flight_plan_2.append(blank_row)

        for frame in self.flight_plan[self.curr_frame:]:
        # for idx, frame in enumerate(self.flight_plan):
            if self.flying:
                print(frame)
                time.sleep(self.time_delay)
                self.mesh.translate(frame[0], frame[1], frame[2])
                app.processEvents()

                if second_flight:
                    self.mesh_2.translate(self.flight_plan_2[self.curr_frame][0], self.flight_plan_2[self.curr_frame][1], self.flight_plan_2[self.curr_frame][2])
                    app.processEvents()

                self.curr_frame += 1
            else:
                break

        # if loop broke because EOF
        if self.flying:
            self.flying = False
            self.btn_start_stop.setText("Start/Stop")

    def slow_down(self):
        self.time_delay += 0.01
        self.time_delay_display -= 0.10

        if self.time_delay >= 0.20:
            self.time_delay = 0.20
            self.lbl_speed.setText("-1.00 (Minimum Speed)")
        elif self.time_delay == 0:
            self.lbl_speed.setText("0.00 (Default Speed)")
        else:
            self.lbl_speed.setText("%.2f" % self.time_delay_display)


    def speed_up(self):
        self.time_delay -= 0.01
        self.time_delay_display += 0.10

        if self.time_delay <= 0:
            self.time_delay = 0
            self.lbl_speed.setText("1.00 (Maximum Speed)")
        elif self.time_delay == 0:
            self.lbl_speed.setText("0.00 (Default Speed)")
        else:
            self.lbl_speed.setText("%.2f" % self.time_delay_display)


    def reset_all(self):
        self.viewer.removeItem(self.mesh)
        if self.flight_plan_filename_2:
            self.viewer.removeItem(self.mesh_2)
        self.drone_mesh_filename = ""
        self.flight_plan_filename = ""
        self.flight_plan_filename_2 = ""
        self.btn_start_stop.setText("Start/Stop")
        self.flight_plan = []
        self.flight_plan_2 = []
        self.time_delay = 0.10
        self.time_delay_display = 0
        self.flying = False
        self.curr_frame = 0

        self.lbl_import_drone.setText("STL file path...")
        self.lbl_import_flight.setText("CSV file path...")
        self.lbl_import_flight_2.setText("CSV file path...")
        self.lbl_speed.setText("0.00 (Default Speed)")


    def reset_flight(self):
        self.btn_start_stop.setText("Start/Stop")
        print('Resetting animation...')
        self.mesh.resetTransform()
        self.mesh_2.resetTransform()
        self.flying = False
        self.curr_frame = 0


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())
