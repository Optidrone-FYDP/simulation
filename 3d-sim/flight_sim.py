# Following libraries required:
# pip install PyOpenGL PyQt5 numpy pyqtgraph numpy-stl pandas

import os
import sys
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
        self.init_ui()
        self.flight_plan = []


    def init_ui(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('OptiDrone Flight Simulator')

        self.error_dialog = QErrorMessage()

        layout = QGridLayout()
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.viewer = gl.GLViewWidget()
        layout.addWidget(self.viewer, 0, 0, 1, 2)

        self.glGrid = gl.GLGridItem()
        self.glGrid.setSize(200, 200)
        self.glGrid.setSpacing(5, 5)
        self.viewer.addItem(self.glGrid)

        self.glAxes = gl.GLAxisItem()
        self.glAxes.setSize(100, 100, 100)
        self.viewer.addItem(self.glAxes)

        btn_import_drone = QPushButton('Import Drone Mesh', self)
        btn_import_drone.clicked.connect(self.import_drone_mesh)
        layout.addWidget(btn_import_drone, 1, 0)

        self.lbl_import_drone = QLineEdit(self)
        self.lbl_import_drone.setText("STL file path...")
        self.lbl_import_drone.setReadOnly(True)
        layout.addWidget(self.lbl_import_drone, 1, 1)

        btn_import_flight = QPushButton('Import Flight Plan', self)
        btn_import_flight.clicked.connect(self.import_flight)
        layout.addWidget(btn_import_flight, 2, 0)

        self.lbl_import_flight = QLineEdit(self)
        self.lbl_import_flight.setText("CSV file path...")
        self.lbl_import_flight.setReadOnly(True)
        layout.addWidget(self.lbl_import_flight, 2, 1)

        btn_start_stop = QPushButton('Start Flight', self)
        btn_start_stop.clicked.connect(self.start_flight)
        layout.addWidget(btn_start_stop, 3, 0, 1, 2)

        btn_reset_all = QPushButton('Reset All', self)
        btn_reset_all.clicked.connect(self.reset_all)
        layout.addWidget(btn_reset_all, 4, 0)

        btn_reset_flight = QPushButton('Reset Flight', self)
        btn_reset_flight.clicked.connect(self.reset_flight)
        layout.addWidget(btn_reset_flight, 4, 1)

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

        meshdata = gl.MeshData(vertexes=points, faces=faces)
        self.mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
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

        self.flight_plan_x = self.flight_plan_in["TX"][1]
        self.flight_plan_y = self.flight_plan_in["TY"][1]
        self.flight_plan_z = self.flight_plan_in["TZ"][1]
        print(self.flight_plan_x)

        self.flight_plan.clear()

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
                self.flight_plan.append(curr_row)

                prev_x, prev_y, prev_z = row["TX"], row["TY"], row["TZ"]

        print(self.flight_plan)

        self.lbl_import_flight.setText(self.flight_plan_filename)


    def start_flight(self):
        if not self.drone_mesh_filename:
            self.error_dialog.showMessage('No drone mesh imported!')
            return

        if not self.flight_plan_filename:
            self.error_dialog.showMessage('No flight plan imported!')
            return

        for frame in self.flight_plan:
            print(frame)
            self.mesh.translate(frame[0], frame[1], frame[2])
            app.processEvents()


    def reset_all(self):
        self.viewer.removeItem(self.mesh)
        self.drone_mesh_filename = ""
        self.flight_plan_filename = ""
        self.flight_plan = []

        self.lbl_import_drone.setText("STL file path...")
        self.lbl_import_flight.setText("CSV file path...")


    def reset_flight(self):
        print('Resetting animation...')
        self.mesh.resetTransform()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())
