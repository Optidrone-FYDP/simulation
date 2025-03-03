from vicon_dssdk import ViconDataStream

class Vicon:
    def __init__(self):
        self.client = ViconDataStream.Client()
        self.drone_name = None
        self.segment_name = None
        self.FrameBuffer = []
        self.curr_pos = None

    def connect(self, host='localhost:801'):
        while not self.client.IsConnected():
            print(f"Connecting to {host}...")
            self.client.Connect(host)
        if self.client.IsConnected():
            print("Connected to Vicon!")
            self.client.EnableSegmentData()

    def setViconMode(self, mode):
        self.client.SetStreamMode(mode)

    def get_frame(self):
        ret = self.client.GetFrame()
        if ret:
            print("Frame received")
        else:
            print(f"Frame not received. Error code: {ret}")
        print(f"Frame {self.client.GetFrameNumber()} received.")
        
        if self.drone_name is None:
            self.drone_name = self.client.GetSubjectNames()[0]
            print(f"Drone name: {self.drone_name}")
        if self.segment_name is None and not self.drone_name is None:
            self.segment_name = self.client.GetSegmentNames(self.drone_name)[0]
            print(f"Segment name: {self.segment_name}")
        position = self.client.GetSegmentGlobalTranslation(self.drone_name, self.segment_name)
        rotation = self.client.GetSegmentGlobalRotationEulerXYZ(self.drone_name, self.segment_name)
        self.curr_pos = [ position[0][0], position[0][1], position[0][2], rotation[0][0], rotation[0][1], rotation[0][2] ]
        return self.curr_pos
