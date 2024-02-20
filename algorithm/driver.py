
import carla
import numpy as np
from laneinfo import LaneInfo, LaneType
from lanetrace import LaneTrace


class Driver(object):

    def __init__(self, actor, laneinfo):
        self.actor = actor
        self.survive = True
        self.lanetrace = LaneTrace(laneinfo, 5)
        self.lanetrace_result = []

    def tick(self):
        try:
            self.tr = self.actor.get_transform()
            self.v = self.actor.get_velocity()

            self.lanetrace_result = self.lanetrace.Trace(self.tr.location.x, self.tr.location.y)
        except:
            self.survive = False
    
    def destroy(self):
        self.actor.destroy()