from vicon_handler import Vicon
from vicon_dssdk import *
import time

vicon = Vicon()
vicon.connect('localhost:801')
vicon.setViconMode(ViconDataStream.Client.StreamMode.EServerPush)

while True:
    vicon.get_frame()
    print(vicon.curr_pos)