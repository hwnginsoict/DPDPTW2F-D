import numpy as np
import random
from population import Individual, Population

import numpy as np

def create_individual_pickup(graph):
    """
    Tạo ra một cá thể (chromosome) dựa trên integer permutation.
    
    1. Tạo leader_keys (kiểu integer) cho các vehicle.
    2. Tạo permutation các pickup node (trừ node 0 nếu 0 là depot).
    3. Gộp hai mảng lại thành chromosome.
    4. Trả về đối tượng Individual (cần định nghĩa class Individual phù hợp).
    """

    num_nodes = graph.num_nodes
    pickup_nodes = graph.pickup_nodes
    # num_pickup_nodes = len(pickup_nodes)
    vehicle_num = graph.vehicle_num

    # 1) Sinh leader_keys kiểu integer cho các vehicle
    #    Ví dụ: random từ [num_nodes, 300)
    leader_keys = np.random.randint(low=num_nodes, high=num_nodes + vehicle_num + 10, size=vehicle_num)

    # 2) Sinh permutation cho các pickup node (bỏ qua depot = 0)
    #    Nếu graph.pickup_nodes = [0, 1, 2, 3, ...], ta chỉ hoán vị [1, 2, 3, ...]
    #    Giả sử pickup_nodes đã sort, ta bỏ pickup_nodes[0] nếu nó là 0
    #    Nếu chắc chắn node 0 là depot, ta làm như dưới:

    valid_pickup_nodes = []
    for node in pickup_nodes:
        valid_pickup_nodes.append(node.nid)

    node_keys = np.random.permutation(valid_pickup_nodes)

    # 3) Ghép 2 mảng lại
    chromosome = np.concatenate((leader_keys, node_keys))

    # Tạo đối tượng Individual (bạn cần định nghĩa hoặc import lớp Individual)
    individual = Individual(chromosome)

    return individual


def repair_time(graph, solution):
    """
    Hàm sắp xếp thứ tự các điểm (node) trong mỗi route của 'solution' 
    theo trường due_time (từ nhỏ đến lớn) trong graph.

    Args:
        graph: Đối tượng graph, trong đó graph.nodes[node_id].due_time 
               trả về due_time của node_id.
        solution: Danh sách các route, mỗi route là một list chứa 
                  các chỉ số node (VD: [leader, n1, n2, ..., nK, leader, ...]).
                  - Thông thường node >= graph.num_nodes có thể được coi là 
                    "leader" hoặc "break" node tùy quy ước.

    Returns:
        new_solution: Bản sao của solution, với mỗi route được sắp xếp 
                      lại các node < graph.num_nodes theo due_time tăng dần.
    """
    new_solution = []

    for route in solution:
        # Tách "leader" hoặc node >= graph.num_nodes (nếu có)
        # và các node "thật" trong cùng route
        
        real_nodes_sorted = sorted(
            route[1:], 
            key=lambda x: graph.nodes[x].due_time
        )

        # Tuỳ thuộc vào logic của bạn:
        # 1) Giữ nguyên leader ở đầu route và cuối route
        # 2) Chèn leader xen kẽ
        #
        # Ở đây, ví dụ đơn giản: 
        # - Đặt leader_nodes (nếu có) ở đầu route, rồi đến real_nodes_sorted
        # - Nếu có nhiều "leader" node, bạn tự quyết định cách chèn phù hợp
        new_route = route[:1] + real_nodes_sorted
        new_solution.append(new_route)

    return new_solution

def split_chromosome(individual, vehicle_num, pickup_num_minus_1):
    """
    Tách chromosome ra 2 phần:
    - leader_keys (độ dài = vehicle_num)
    - permutation (độ dài = pickup_num_minus_1)
    """
    chromosome = individual.chromosome
    leader_keys = chromosome[:vehicle_num]
    permutation = chromosome[vehicle_num: vehicle_num + pickup_num_minus_1]
    return leader_keys, permutation

def merge_chromosome(leader_keys, permutation):
    """
    Gộp 2 mảng leader_keys và permutation thành 1 chromosome duy nhất.
    """
    return np.concatenate((leader_keys, permutation))

def crossover_leader_keys(p1_leader, p2_leader, prob=0.5):
    """
    Uniform crossover giữa 2 mảng leader_keys.
    """
    size = len(p1_leader)
    mask = np.random.rand(size) < prob
    child1 = p1_leader.copy()
    child2 = p2_leader.copy()

    child1[mask] = p2_leader[mask]
    child2[mask] = p1_leader[mask]

    return child1, child2


def crossover_leader_keys_onepoint(p1_leader, p2_leader):
    """
    1-point crossover giữa 2 mảng leader_keys.
    """
    size = len(p1_leader)
    point = np.random.randint(1, size)  # cắt ở đâu đó giữa
    child1 = np.concatenate((p1_leader[:point], p2_leader[point:]))
    child2 = np.concatenate((p2_leader[:point], p1_leader[point:]))
    return child1, child2

def pmx_crossover(parent1, parent2):
    """
    PMX crossover cho 2 mảng permutation parent1, parent2 có cùng độ dài.
    Trả về 2 mảng con (child1, child2).
    """
    size = len(parent1)
    # Chọn 2 điểm cắt (cut1, cut2)
    cut1, cut2 = sorted(np.random.choice(range(size), 2, replace=False))
    
    # Khởi tạo con
    child1 = [-1] * size
    child2 = [-1] * size

    # Copy đoạn cắt từ parent
    child1[cut1:cut2+1] = parent1[cut1:cut2+1]
    child2[cut1:cut2+1] = parent2[cut1:cut2+1]

    # Đánh dấu đoạn đã copy để tạo mapping
    mapping_p1 = parent1[cut1:cut2+1]
    mapping_p2 = parent2[cut1:cut2+1]

    # Xử lý các phần tử ngoài đoạn cắt cho child1
    for i in range(size):
        if i < cut1 or i > cut2:
            val = parent2[i]
            # Nếu val đã nằm trong child1, ta thay thế dựa trên mapping
            while val in mapping_p1:
                # idx = mapping_p1.index(val)  # Lỗi vì mapping_p1 là ndarray
                idx = np.where(mapping_p1 == val)[0][0]
                val = mapping_p2[idx]
            child1[i] = val

    # Tương tự cho child2
    for i in range(size):
        if i < cut1 or i > cut2:
            val = parent1[i]
            while val in mapping_p2:
                idx = np.where(mapping_p2 == val)[0][0]
                val = mapping_p1[idx]
            child2[i] = val

    return np.array(child1), np.array(child2)


def ox_crossover(parent1, parent2):
    """
    Order Crossover (OX) cho 2 mảng permutation.
    """
    size = len(parent1)
    child1 = [-1]*size
    child2 = [-1]*size

    start, end = sorted(np.random.choice(range(size), 2, replace=False))

    # Sao chép đoạn [start, end] từ cha sang con
    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1]

    # Thứ tự còn lại fill vào con 1
    pos = (end + 1) % size
    for i in range(size):
        idx = (end + 1 + i) % size
        if parent2[idx] not in child1:
            child1[pos] = parent2[idx]
            pos = (pos + 1) % size

    # Thứ tự còn lại fill vào con 2
    pos = (end + 1) % size
    for i in range(size):
        idx = (end + 1 + i) % size
        if parent1[idx] not in child2:
            child2[pos] = parent1[idx]
            pos = (pos + 1) % size

    return np.array(child1), np.array(child2)


def cx_crossover(parent1, parent2):
    """
    Cycle Crossover (CX) cho 2 mảng permutation parent1, parent2.
    Trả về child1, child2.
    """

    size = len(parent1)
    child1 = [-1] * size
    child2 = [-1] * size

    visited = [False] * size
    idx = 0

    # Tính số chu kỳ (cycle)
    while False in visited:
        if not visited[idx]:
            start = idx
            val1 = parent1[idx]
            val2 = parent2[idx]

            while True:
                # Gán cho child
                child1[idx] = parent1[idx]
                child2[idx] = parent2[idx]
                visited[idx] = True

                # print("Parent1:", parent1)
                # print("Parent2:", parent2)
                # print("Child1:", child1)
                # print("Child2:", child2)

                # Tìm vị trí tiếp theo trong parent1 trùng với val2
                matching_indices = np.where(parent1 == val2)[0]
                if len(matching_indices) == 0:
                    raise ValueError(f"Value {val2} không tồn tại trong parent1 (không phải permutation?)")

                idx = matching_indices[0]   # Lấy vị trí đầu tiên

                val2 = parent2[idx]
                if idx == start:
                    break

        # Tìm vị trí idx mới chưa được duyệt
        unvisited = [i for i, v in enumerate(visited) if not v]
        if unvisited:
            idx = unvisited[0]

    return np.array(child1), np.array(child2)




def crossover_operator(graph, parent1, parent2, method='PMX'):
    """
    Thực hiện lai ghép giữa 2 cá thể cha/mẹ (parent1, parent2)
    Trả về 2 cá thể con (child1, child2) chưa tính fitness.
    
    method: 'PMX', 'OX', hay 'CX'.
    """
    vehicle_num = graph.vehicle_num
    pickup_num_minus_1 = len(graph.pickup_nodes)

    # Tách phần leader keys
    p1_leader, p1_perm = split_chromosome(parent1, vehicle_num, pickup_num_minus_1)
    p2_leader, p2_perm = split_chromosome(parent2, vehicle_num, pickup_num_minus_1)

    # -- Crossover leader keys -- (ví dụ uniform)
    c1_leader, c2_leader = crossover_leader_keys(p1_leader, p2_leader)
    # Hoặc dùng one-point
    # c1_leader, c2_leader = crossover_leader_keys_onepoint(p1_leader, p2_leader)
    
    # -- Crossover permutation --
    if method == 'PMX':
        c1_perm, c2_perm = pmx_crossover(p1_perm, p2_perm)
    elif method == 'OX':
        c1_perm, c2_perm = ox_crossover(p1_perm, p2_perm)
    elif method == 'CX':
        c1_perm, c2_perm = cx_crossover(p1_perm, p2_perm)
    else:
        # Mặc định dùng PMX
        c1_perm, c2_perm = pmx_crossover(p1_perm, p2_perm)

    # Gộp lại
    c1_chromosome = merge_chromosome(c1_leader, c1_perm)
    c2_chromosome = merge_chromosome(c2_leader, c2_perm)

    # Tạo cá thể con
    child1 = Individual(c1_chromosome)
    child2 = Individual(c2_chromosome)

    # print("Parent1:", parent1.chromosome)
    # print("Parent2:", parent2.chromosome)
    # print("Child1:", child1.chromosome)
    # print("Child2:", child2.chromosome)

    return child1, child2


def mutate_leader_keys(leader_keys, mutation_rate, min_val, max_val):
    """
    Đột biến trên leader_keys (số nguyên), 
    random reset trong khoảng [min_val, max_val).
    """
    for i in range(len(leader_keys)):
        if np.random.rand() < mutation_rate:
            leader_keys[i] = np.random.randint(low=min_val, high=max_val)
    return leader_keys



def swap_mutation(permutation, mutation_rate):
    """
    Swap mutation cho permutation:
    Với xác suất mutation_rate, chọn 2 vị trí ngẫu nhiên và swap.
    """
    perm = permutation.copy()
    for i in range(len(perm)):
        if np.random.rand() < mutation_rate:
            j = np.random.randint(0, len(perm))
            # Swap
            perm[i], perm[j] = perm[j], perm[i]
    return perm

def change_mutation(permutation, mutation_rate):
    """
    Change mutation cho permutation:
    Với xác suất mutation_rate, chọn 1 vị trí ngẫu nhiên và đổi chỗ với 1 vị trí khác.
    """
    perm = permutation.copy()
    for i in range(len(perm)):
        if np.random.rand() < mutation_rate:
            j = np.random.randint(0, len(perm))
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def scramble_mutation(permutation, mutation_rate):
    perm = permutation.copy()
    if np.random.rand() < mutation_rate:
        start, end = sorted(np.random.choice(range(len(perm)), 2, replace=False))
        np.random.shuffle(perm[start:end+1])
    return perm


def mutation_operator(graph, individual, mutation_rate=0.1):
    """
    Đột biến (mutation) lên 1 cá thể, trả về 1 cá thể con.
    """
    vehicle_num = graph.vehicle_num
    pickup_num_minus_1 = len(graph.pickup_nodes)

    # Tách
    leader_keys, permutation = split_chromosome(individual, vehicle_num, pickup_num_minus_1)

    # Đột biến leader keys (ví dụ random reset trong [graph.num_nodes, 300))
    leader_keys = mutate_leader_keys(
        leader_keys, 
        mutation_rate, 
        min_val=graph.num_nodes, 
        max_val=graph.num_nodes + vehicle_num + 10
    )

    # Đột biến permutation (ví dụ swap mutation)
    permutation = swap_mutation(permutation, mutation_rate)
    # permutation = scramble_mutation(permutation, mutation_rate)
    # Hoặc scramble_mutation, inversion_mutation, ...

    # Gộp lại
    new_chromosome = merge_chromosome(leader_keys, permutation)

    # Tạo offspring
    offspring = Individual(new_chromosome)
    return offspring


def cost_full(graph, solution):
    """
    Computes:
      1) total_energy       - total energy consumption of the entire solution (litres of fuel)
      2) vehicle_fairness   - standard deviation of energy consumption among vehicles (litres)
      3) customer_fairness  - standard deviation of total request tardiness (minutes),
                              where request tardiness = tardiness_pickup + tardiness_delivery
      4) max_time           - the maximum tardiness of any single node (or you could adapt it
                              to be max request tardiness if desired)

    Args:
        graph: Graph object containing attributes like dist, nodes, 
               vehicle_speed, and energy-related parameters.
        solution: list of routes, each route is a list of nodes 
                  [0, i1, i2, ..., 0] (assuming 0 is depot)
    
    Returns:
        (total_energy, vehicle_fairness, customer_fairness, max_time)
    """

    # Extract energy-related constants from graph or define them here
    cd = graph.cd
    xi = graph.xi
    kappa = graph.kappa
    p = graph.p
    A = graph.A
    mk = graph.mk
    g = graph.g
    cr = graph.cr
    psi = graph.psi
    pi_val = graph.pi
    R = graph.R
    eta = graph.eta

    # Speeds, etc.
    v_speed_km_h = 40
    v_speed_km_m = 40 / 60.0  # ~0.6667 km/min

    total_energy = 0.0
    ve_energy = []                 # energy consumption per vehicle
    node_tardiness = dict()        # dictionary to store tardiness per node, e.g. node_tardiness[node_id] = tardiness

    # Initialize all node tardiness to 0
    for node in graph.nodes:
        node_tardiness[node.nid] = 0.0

    # Function to compute energy for a segment between two nodes
    def energy_for_leg(current_node, next_node, current_capacity):
        # Calculate distance between nodes
        d_ij = graph.dist[current_node][next_node]

        # Power consumption terms
        p_ij = 0.5 * cd * p * A * (v_speed_km_h ** 3) + (mk + current_capacity) * g * cr * v_speed_km_h

        # Compute energy consumption L_ij using the provided formula:
        L_ij = (xi / (kappa * psi)) * (pi_val * R + (p_ij / eta)) * (d_ij / v_speed_km_h)
        return L_ij, d_ij

    # --- Process each route ---
    for route in solution:
        route_energy = 0.0
        current_capacity = 0.0
        time = 0.0  # track time along the route for tardiness

        # ----- Leg from depot (0) to first customer -----
        if len(route) > 1:
            L_ij, d_ij = energy_for_leg(0, route[1], current_capacity)
            route_energy += L_ij
            travel_time = d_ij / v_speed_km_m
            time += travel_time
            current_capacity += graph.nodes[route[1]].demand

        # ----- Leg from first customer to intermediate, ... , until last -----
        for i in range(1, len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

            # If next_node is not depot, we handle time windows, capacity, etc.
            if next_node != 0:
                # The node object
                customer = graph.nodes[next_node]

                # Arrive at this next_node
                time = max(time, customer.ready_time)  # if we arrive too early, we wait

                # Tardiness check BEFORE service
                if time > customer.due_time:
                    tardiness = time - customer.due_time
                    node_tardiness[next_node] = tardiness
                else:
                    node_tardiness[next_node] = 0.0

                # Service time
                time += customer.service_time
                
                current_capacity += customer.demand

            # Compute energy and distance for this segment (current_node -> next_node)
            L_ij, d_ij = energy_for_leg(current_node, next_node, current_capacity)
            route_energy += L_ij

            # Travel time (in minutes)
            travel_time = d_ij / v_speed_km_m
            time += travel_time

        # ----- Leg from last node back to depot (0) -----
        if len(route) > 1:
            L_ij, d_ij = energy_for_leg(route[-1], 0, current_capacity)
            route_energy += L_ij
            # current_capacity += graph.nodes[route[-1]].demand

        ve_energy.append(route_energy)
        total_energy += route_energy

        # if current_capacity!=0:
        #     print(current_capacity)
        #     print("Route: ", route)
        #     raise ValueError("Capacity not zero at the end of the route!")

    # Calculate vehicle_fairness
    vehicle_fairness = standard_deviation(ve_energy)

    # ------------------------------------------------------------------
    #   Compute request-level tardiness
    # ------------------------------------------------------------------
    # We sum the tardiness of pickup_node and its corresponding delivery_node
    # for each request: requests[pickup_node_id] = delivery_node_id.
    request_tardiness_list = []
    for pickup_nid, delivery_nid in graph.requests.items():
        tardiness_pickup = node_tardiness.get(pickup_nid, 0.0)
        tardiness_delivery = node_tardiness.get(delivery_nid, 0.0)
        request_tardiness_list.append(tardiness_pickup + tardiness_delivery)

    # Now the fairness is the standard deviation of the request-level sums
    customer_fairness = standard_deviation(request_tardiness_list) if request_tardiness_list else 0.0

    max_time = max(node_tardiness.values()) if node_tardiness else 0.0

    return total_energy, vehicle_fairness, customer_fairness, max_time


def cost_drone(graph, solution, check_drone):
    """
    Computes:
      1) total_energy       - total energy consumption of the entire solution (litres of fuel)
      2) vehicle_fairness   - standard deviation of energy consumption among vehicles (litres)
      3) customer_fairness  - standard deviation of total request tardiness (minutes),
                              where request tardiness = tardiness_pickup + tardiness_delivery
      4) max_time           - the maximum tardiness of any single node (or you could adapt it
                              to be max request tardiness if desired)

    Args:
        graph: Graph object containing attributes like dist, nodes, 
               vehicle_speed, and energy-related parameters.
        solution: list of routes, each route is a list of nodes 
                  [0, i1, i2, ..., 0] (assuming 0 is depot)
    
    Returns:
        (total_energy, vehicle_fairness, customer_fairness, max_time)
    """

    # Extract energy-related constants from graph or define them here
    cd = graph.cd
    xi = graph.xi
    kappa = graph.kappa
    p = graph.p
    A = graph.A
    mk = graph.mk
    g = graph.g
    cr = graph.cr
    psi = graph.psi
    pi_val = graph.pi
    R = graph.R
    eta = graph.eta

    beta = 210.8 #w/kg
    gamma = 181.2 #w

    # Speeds, etc.
    v_speed_km_h = 40
    v_speed_km_m = 40 / 60.0  # ~0.6667 km/min


    drone_speed = 0.9388

    total_energy = 0.0
    ve_energy = []                 # energy consumption per vehicle
    node_tardiness = dict()        # dictionary to store tardiness per node, e.g. node_tardiness[node_id] = tardiness

    # Initialize all node tardiness to 0
    for node in graph.nodes:
        node_tardiness[node.nid] = 0.0

    # Function to compute energy for a segment between two nodes
    def energy_for_leg(current_node, next_node, current_capacity):
        # Calculate distance between nodes
        d_ij = graph.dist[current_node][next_node]

        # Power consumption terms
        p_ij = 0.5 * cd * p * A * (v_speed_km_h ** 3) + (mk + current_capacity) * g * cr * v_speed_km_h

        # Compute energy consumption L_ij using the provided formula:
        L_ij = (xi / (kappa * psi)) * (pi_val * R + (p_ij / eta)) * (d_ij / v_speed_km_h)
        
        return L_ij * 28.525, d_ij / v_speed_km_m

    
    def energy_drone(current_node, next_node, current_capacity):
        d_ij = graph.dist[current_node][next_node]
        energy_per_minute = (gamma + beta * 20) * 60
        time = d_ij / drone_speed
        L_ij  = (time + 2 + 2) * energy_per_minute
        return L_ij * 2.450 / 60000, d_ij / drone_speed


    # --- Process each route ---
    for idx, route in enumerate(solution):
        route_energy = 0.0
        current_capacity = 0.0
        time = 0.0  # track time along the route for tardiness

        # ----- Leg from depot (0) to first customer -----
        if len(route) > 1:

            if check_drone[idx][1] == 1:
                L_ij, t_ij = energy_drone(0, route[1], current_capacity)
                truck_node = 0
                time_truck = time
            else:
                L_ij, t_ij = energy_for_leg(0, route[1], current_capacity)

            route_energy += L_ij
            travel_time = t_ij
            time += travel_time
            current_capacity += graph.nodes[route[1]].demand

        # ----- Leg from first customer to intermediate, ... , until last -----
        for i in range(1, len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

            # If next_node is not depot, we handle time windows, capacity, etc.
            if next_node != 0:
                # The node object
                customer = graph.nodes[next_node]

                # Arrive at this next_node
                time = max(time, customer.ready_time)  # if we arrive too early, we wait

                # Tardiness check BEFORE service
                if time > customer.due_time:
                    tardiness = time - customer.due_time
                    node_tardiness[next_node] = tardiness
                else:
                    node_tardiness[next_node] = 0.0

                # Service time
                time += customer.service_time
                
                current_capacity += customer.demand

            # Compute energy and distance for this segment (current_node -> next_node)

            # print("len route", len(route))
            # print("len check drone", len(check_drone))
            # print("i", i)
            # print("Route: ", route)

            if check_drone[idx][i+1] == 1:
                L_ij, t_ij = energy_drone(current_node, next_node, current_capacity)
                truck_node = current_node
                time_truck = time
            else:
                if check_drone[idx][i] == 1:
                    L_ij_truck, t_ij_truck = energy_for_leg(truck_node, next_node, current_capacity)
                    L_ij_drone, t_ij_drone = energy_drone(current_node, next_node, current_capacity)

                    L_ij = L_ij_truck + L_ij_drone

                    if time_truck + t_ij_truck >= time + t_ij_drone:
                        t_ij = t_ij_truck
                        time = time_truck
                    else:
                        t_ij = t_ij_drone
                    # t_ij = t_ij_truck + t_ij_drone
                else:
                    L_ij, t_ij = energy_for_leg(current_node, next_node, current_capacity)

            route_energy += L_ij

            # Travel time (in minutes)
            travel_time = t_ij
            time += travel_time

        # ----- Leg from last node back to depot (0) -----
        if len(route) > 1:
            L_ij, t_ij = energy_for_leg(route[-1], 0, current_capacity)
            route_energy += L_ij
            # current_capacity += graph.nodes[route[-1]].demand

        ve_energy.append(route_energy)
        total_energy += route_energy

        # if current_capacity!=0:
        #     print(current_capacity)
        #     print("Route: ", route)
        #     raise ValueError("Capacity not zero at the end of the route!")

    # Calculate vehicle_fairness
    vehicle_fairness = standard_deviation(ve_energy)

    # ------------------------------------------------------------------
    #   Compute request-level tardiness
    # ------------------------------------------------------------------
    # We sum the tardiness of pickup_node and its corresponding delivery_node
    # for each request: requests[pickup_node_id] = delivery_node_id.
    request_tardiness_list = []
    for pickup_nid, delivery_nid in graph.requests.items():
        tardiness_pickup = node_tardiness.get(pickup_nid, 0.0)
        tardiness_delivery = node_tardiness.get(delivery_nid, 0.0)
        request_tardiness_list.append(tardiness_pickup + tardiness_delivery)

    # Now the fairness is the standard deviation of the request-level sums
    customer_fairness = standard_deviation(request_tardiness_list) if request_tardiness_list else 0.0

    max_time = max(node_tardiness.values()) if node_tardiness else 0.0

    return total_energy, vehicle_fairness, customer_fairness, max_time


def variance(list):
    mean = sum(list) / len(list)
    variance = sum((x - mean) ** 2 for x in list) / len(list)
    return variance

def standard_deviation(list):
    return np.sqrt(variance(list))


def decode_solution_pickup(problem, keys):
    """
    Giải mã chromosome (gồm leader_keys + permutation các pickup node) 
    thành danh sách route cho mỗi vehicle.
    
    1) Tách leader_keys và node_perm (phần permutation).
    2) Sắp xếp chỉ số vehicle theo thứ tự tăng dần của leader_keys (argsort).
    3) Lấy các pickup node theo thứ tự trong node_perm, chia đều cho các xe.
    4) Mỗi route: [leader_idx, p1, d1, p2, d2, ...].
    5) Gọi repair_time (nếu có) để sửa thêm (nếu cần).

    Args:
        problem: 
            - problem.vehicle_num: số xe
            - problem.pickup_nodes: danh sách Node (có .nid)
            - problem.nodes[node_id].did: delivery node ứng với pickup node_id
        keys: np.array
            - keys[:vehicle_num]       = leader_keys (số nguyên)
            - keys[vehicle_num:]       = node_perm (hoán vị các pickup node, *trừ* depot)
    Returns:
        solution: list các route, mỗi route là một list chỉ số node (int).
    """

    vehicle_num = problem.vehicle_num

    # ----------------------------------------------------------
    # 1) TÁCH LEADER_KEYS VÀ NODE_PERM
    # ----------------------------------------------------------
    leader_keys = keys[:vehicle_num]            # Mảng int (leader_keys)
    node_perm   = keys[vehicle_num:]            # Mảng int (permutation của pickup node IDs, *không* gồm depot)

    # ----------------------------------------------------------
    # 2) SẮP XẾP VỊ TRÍ XE THEO THỨ TỰ LEADER_KEYS TỪ NHỎ ĐẾN LỚN
    # ----------------------------------------------------------
    sorted_leader_indices = np.argsort(leader_keys)

    # ----------------------------------------------------------
    # 3) CHUẨN BỊ ROUTE CHO MỖI XE, CHIA PICKUP THEO NAIVE CAPACITY
    # ----------------------------------------------------------
    solution = [[] for _ in range(vehicle_num)]
    # naive_capacity ở đây chỉ là cách chia đều theo số pickup
    naive_capacity = len(node_perm) // vehicle_num
    idx_pickup = 0

    # ----------------------------------------------------------
    # 4) GÁN PICKUP + DELIVERY THEO THỨ TỰ node_perm CHO CÁC XE
    # ----------------------------------------------------------
    for i, leader_idx in enumerate(sorted_leader_indices):
        # Bắt đầu route: [leader_idx, ...]
        solution[i].append(int(leader_idx))
        
        # Lấy một mẻ pickup
        assigned_pickups = node_perm[idx_pickup: idx_pickup + naive_capacity]
        idx_pickup += naive_capacity

        # Thêm pickup & delivery
        for p_id in assigned_pickups:
            d_id = problem.nodes[p_id].did  # delivery node tương ứng
            solution[i].append(p_id)
            solution[i].append(d_id)

    # ----------------------------------------------------------
    # 5) GÁN CÁC PICKUP DƯ (NẾU CÓ) THEO VÒNG TRÒN
    # ----------------------------------------------------------
    leftover_pickups = node_perm[idx_pickup:]
    for j, p_id in enumerate(leftover_pickups):
        route_idx = j % vehicle_num
        d_id = problem.nodes[p_id].did
        solution[route_idx].append(p_id)
        solution[route_idx].append(d_id)

    # ----------------------------------------------------------
    # 6) GỌI REPAIR_TIME (NẾU CÓ) ĐỂ SẮP XẾP THEO TIME WINDOW HOẶC LOGIC KHÁC
    # ----------------------------------------------------------
    solution = repair_time(problem, solution)  # giả sử bạn đã định nghĩa hàm repair_time

    return solution


def drone_heuristics(graph, solution):
    remain = 10
    cap = 10
    drone_speed = 0.9388
    check_drone = []
    for route in solution:
        check = [0 for i in range(len(route))]
        for i in range(2,len(route)-1):
            if check[i-1] == 0 and graph.nodes[route[i]].demand <= cap and graph.nodes[route[i]].drone == 1:
                # print(graph.nodes[route[i-1]])
                # print(route[i-1], route[i], route[i+1])
                drone_time = (graph.dist[route[i-1]][route[i]] + graph.dist[route[i]][route[i+1]])/drone_speed - 2 - 2
                if drone_time < 30:
                    check[i] = 1
        check_drone.append(check)

    
    return check_drone

        

def calculate_fitness(problem, individual):
    """
    Tính toán fitness (đa mục tiêu) cho một cá thể (individual).
    Ở đây, ta sử dụng hàm cost(route) trả về:
        total_distance, vehicle_fairness, customer_fairness
    Lưu các giá trị đó thành list và gán vào individual["objectives"].
    
    Args:
        problem: đối tượng chứa thông tin bài toán (trong đó có .cost(route)).
        individual: đối tượng, có .chromosome và .objectives

    Returns:
        list: [total_distance, vehicle_fairness, customer_fairness]
    """
    # 1) Giải mã từ random keys -> route (dạng một list duy nhất 
    #    có chèn sentinel >= problem.graph.num_nodes để đánh dấu chia tuyến)
    route = decode_solution_pickup(problem, individual.chromosome)

    # print("Route: ", route)

    # raise Exception

    check_drone = drone_heuristics(problem, route)
    # print("Check drone: ", check_drone)
    # raise
    
    # 2) Tính cost
    total_distance, vehicle_fairness, customer_fairness, max_time = cost_drone(problem, route, check_drone)
    
    # 3) Lưu vào individual["objectives"] (mục tiêu đa mục tiêu)
    # individual.objectives = [(total_distance/10000)/0.00284, vehicle_fairness/200, customer_fairness/20]
    individual.objectives = [(total_distance), vehicle_fairness, customer_fairness, max_time]
    
    return individual.objectives