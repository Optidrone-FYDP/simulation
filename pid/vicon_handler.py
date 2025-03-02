from vicon_dssdk import ViconDataStream

class Vicon:
    def __init__(self):
        self.client = ViconDataStream.Client()
        self.drone_name = None
        self.segment_name = None
        self.FrameBuffer = []
        self.curr_pos = None

    def connect(self, host='localhost:801'):
        self.client.Connect(host)

    def setViconMode(self, mode):
        self.client.SetStreamingMode(mode)

    def get_frame(self):
        self.client.GetFrame()
        
        if self.drone_name is None:
            self.drone_name = self.client.GetSubjectName(0)
        if self.segment_name is None and not self.drone_name is None:
            self.segment_name = self.client.GetSegmentName(self.drone_name, 0)

        position = self.client.GetSegmentGlobalTranslation(self.drone_name, self.segment_name)
        rotation = self.client.GetSegmentGlobalRotationEulerXYZ(self.drone_name, self.segment_name)
        return [ position.Translation[0]*0.001, position.Translation[1]*0.001, position.Translation[2]*0.001, rotation.Rotation[0], rotation.Rotation[1], rotation.Rotation[2] ]
