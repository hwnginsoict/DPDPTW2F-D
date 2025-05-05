class Node(object):
    def __init__(self, nid: int, x: float, y: float, demand: float, ready_time: float, due_time: float, service_time: float, pid: int, did: int, time: float, drone:int):
        # Initialize the instance attributes
        self.nid = nid
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
        self.pid = pid
        self.did = did
        self.time = time
        self.drone = drone