import numpy as np
from .node import Node
import csv

class Graph:
    def __init__(self, file_path, cd=0.7, xi=1, kappa=44, p=1.2, A=3.192, mk=3.2, g=9.81, cr=0.01, psi=737, pi=0.2, R=165, eta=0.36):
        self.cd = cd
        self.p = p
        self.A = A 
        self.mk = mk
        self.g = g
        self.cr = cr
        self.xi = xi
        self.kappa = kappa
        self.psi = psi 
        self.pi = pi
        self.R = R 
        self.eta = eta

        self.read_file(file_path)


    def read_file(self, file_path):
        node_list = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header row
            for row in reader:
                node_list.append(row)

        num_nodes = len(node_list)
        nodes = [
            Node(
                int(item[0]),   # nid
                float(item[1]), # x
                float(item[2]), # y
                float(item[3]), # ready_time
                float(item[4]), # due_time
                float(item[5]), # demand
                float(item[6]), # service_time
                int(item[7]),   # pid 
                int(item[8]),   # did 
                float(item[9]),  # time (hoặc speed)
                float(item[10])
            )
            for item in node_list
        ]

        self.depot = nodes[0]


        pickup_nodes = [node for node in nodes if node.pid == 0 and node.did != 0]

        # Khởi tạo ma trận khoảng cách NxN
        dist = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            node_a = nodes[i]
            for j in range(i+1, num_nodes):
                node_b = nodes[j]
                d = self.calculate_dist(node_a, node_b)
                dist[i][j] = d
                dist[j][i] = d  # Ghi đối xứng

        vehicle_num = int(nodes[0].demand)
        # vehicle_num = 10
        vehicle_cap = nodes[0].service_time
        vehicle_speed = 4/6

        self.pickup_nodes = pickup_nodes
    
        self.num_pickup_nodes = len(pickup_nodes)
        self.num_nodes = num_nodes
        self.nodes = nodes
        self.dist = dist
        self.vehicle_num = vehicle_num
        self.vehicle_cap = vehicle_cap
        self.vehicle_speed = vehicle_speed

        # Tạo dictionary requests: p -> d
        # pickup node: pid=0, did != 0
        # => requests[p] = d, trong đó p = node.nid, d = node.did
        self.requests = {}
        for node in nodes:
            if node.pid == 0 and node.did != 0:
                self.requests[node.nid] = node.did

    @staticmethod
    def calculate_dist(node_a, node_b):
        """
        Tính khoảng cách Euclidean giữa node_a và node_b.
        Trả về float('inf') nếu khoảng cách = 0 (tránh chia 0?).
        """
        if node_a.nid == node_b.nid:
            return float('inf')
        dis = np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))
        return dis


if __name__ == "__main__":
    graph = Graph('F:\\CodingEnvironment\\DPDPTW2F\\data\\dpdptw\\200\\LC1_2_1.csv')
    print("num_nodes =", graph.num_nodes)
    print("vehicle_num =", graph.vehicle_num)
    print("vehicle_cap =", graph.vehicle_cap)
    print("vehicle_speed =", graph.vehicle_speed)
    print(len(graph.pickup_nodes), "pickup nodes")
    
    # Kiểm tra requests
    # mỗi cặp p->d trong graph.requests
    for p, d in graph.requests.items():
        print(f"Pickup node {p} -> Delivery node {d}")
