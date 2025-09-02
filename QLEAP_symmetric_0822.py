import random
import math
import copy
import time
from collections import namedtuple, deque
import netsquid as ns
import netsquid.components.instructions as instr
import netsquid.qubits.ketstates as ks
import pydynaa
import numpy as np
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.models import FixedDelayModel, FibreDelayModel, DephaseNoiseModel, DepolarNoiseModel
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate, IRotationGate
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals, ServiceProtocol
from netsquid.nodes import Node, Network
from netsquid.nodes.connections import Connection
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from netsquid.qubits import operators as ops, StateSampler, qubitapi as qapi
from netsquid.util.datacollector import DataCollector
from netsquid.components.qmemory import QuantumMemory
from scipy.stats import binom
from queue import Queue
from collections import namedtuple
from collections import defaultdict


temp = [0 for i in range(1024)] 

# side len(node num) of square network
scale = 32
# available position num of each node      
pos_num = 500
req_num = 50
# physical distance between nodes
distance = 2
req_bandwidth = 6
ori_f = 0.85
#set k which  makes the magnitudes in the equation match
# k1*10^6
k = [1e-6, 10, 1e-6]
requests = []
# files
filename = 'base_output.txt'
failfile = 'fail_allocation.txt'
configure_file = '/home/ustc-test1/quantum/Q-Network/fidelity_configure_32_0.85.txt'
filename1 = '/home/ustc-test1/quantum/Q-Network/positions1.txt'
filename2 = '/home/ustc-test1/quantum/Q-Network/positions2.txt'
filename3='/home/ustc-test1/quantum/Q-Network/DEJMPS_request.txt'
src_dst_file = '/home/ustc-test1/quantum/Q-Network/src_dst_50_32_48.txt'

# parameters
GateA=IGate('RxA', ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
GateB=IGate('RxB', ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate = True))
GateC=IGate('CNOT', ops.CNOT)
_INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
_INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))
gate_noise_model = DephaseNoiseModel(200)
mem_noise_model = DepolarNoiseModel(200)
Ins = [
    PhysicalInstruction(GateA, duration=1, parallel=True), 
    PhysicalInstruction(GateB, duration=1, parallel=True), 
    PhysicalInstruction(GateC, duration=1, parallel=True), 
    PhysicalInstruction(INSTR_MEASURE, duration=1), 
    PhysicalInstruction(_INSTR_Rx, duration=1, parallel=True), 
    PhysicalInstruction(_INSTR_RxC, duration=1, parallel=True), 
    PhysicalInstruction(INSTR_CNOT, duration=1, parallel=True), 
    PhysicalInstruction(instr.INSTR_X, duration=1), 
    PhysicalInstruction(instr.INSTR_Z, duration=1), 
    PhysicalInstruction(instr.INSTR_MEASURE_BELL, duration=1)
    ]
# time_unit: ns
finished_path = 0
link_entangle_time = 10000000
link_deviation_time = 1500000
purify_round_time = 2000000
swap_time = 10000000
swap_deviation_time= 1000000
allocation_query_time = 100000
# mean_arrival_time = 1/frequency
frequency = 2e-10
# distillation_fidelity = []
distillation_timecost = []
waiting_time = []
all_fidelities = []
# 定义一个命名元组来存储路径ID和保真度
PathFidelity = namedtuple('PathFidelity', ['path_id', 'fidelity'])
last_request_path_id = []
request_path_id = []

routing_req= []
routing_T = []
routing_C = []
routing_F = []
routing_D = []
routing_T_success = []
routing_C_success = []
routing_F_success = []
routing_D_success = []
routing_T_average = []
routing_C_average = []
routing_F_average = []
routing_D_average = []
distillation_R = [] # 纯化资源
distillation_T = []
distillation_R_success = []
distillation_T_success = []
distillation_R_average = []
distillation_T_average = []
all_qubit_fidelity = []


def write_to_file(data, filename):
    with open(filename, "a") as file:
        file.write(data + "\n")

class ClassicalConnection(Connection):
    # 单向经典信道
    def __init__(self, name, length):
        super().__init__(name=name)
        self.add_subcomponent(ClassicalChannel("Channel_A2B", length=length, models={"delay_model": FibreDelayModel(c=250000)}))
        self.ports['A'].forward_input(self.subcomponents["Channel_A2B"].ports['send'])
        self.subcomponents["Channel_A2B"].ports['recv'].forward_output(self.ports['B'])

class Link:
    def __init__(self, left_node, right_node, initial_fidelity):
        self.left_node = left_node                     #node ID of previous hop
        self.right_node = right_node
        self.max_capacity = 0
        self.rescapacity=0
        self.Pmk = []
        self.allocatedcapacity=0
        self.initial_fidelity = initial_fidelity
        self.decomposition_fidelity = 0
        self.eta = []
        self.fidelity_list = []
        self.probability_list = []
        self.r = 0
        self.n_backup_list = []  # 存储每轮目标量子比特数量的列表
        self.threshold_all = 0.95  # 纯化总阈值
        # self.threshold_all = 0.5

    def prior_knowledge(self):
        fidelity = float(self.initial_fidelity)
        self.fidelity_list.append(fidelity)
        while fidelity - float(self.decomposition_fidelity) < -0:
            probability = (fidelity * fidelity) + (1 - fidelity) * (1 - fidelity)
            self.probability_list.append(probability)
            fidelity = (fidelity * fidelity) / (fidelity * fidelity + (1 - fidelity) * (1 - fidelity))
            self.fidelity_list.append(fidelity)
            self.r += 1
        #self.r = 2
        # = math.ceil(self.r/2)*2
        biaoji2=0


    # def back_up(self):
    #     num = 1
    #     for i in range(self.r + 1):
    #         self.n_backup_list.insert(0,num)
    #         num *= 2

    def back_up(self):
        num = 1
        for i in range(self.r + 1):
            self.n_backup_list.insert(0,num)
            num *= 2

    def calculate_rescapacity(self):
        self.rescapacity = self.max_capacity // self.n_backup_list[0]
        self.allocatedcapacity = self.rescapacity

    def calculate_Pmk(self):  # 计算Pmk的函数
        for i in range(self.allocatedcapacity + 1):
            Pmk_i = Cmn(self.allocatedcapacity, i) * (self.threshold_all ** i) * (
                        (1 - self.threshold_all) ** (self.allocatedcapacity - i))
            self.Pmk.append(Pmk_i)

    def run(self):
        # 对传入的每个链路调用类内函数run()即可。
        self.prior_knowledge()
       # print("back up")
        self.back_up()
       # print("calculate_res")
        self.calculate_rescapacity()


class Request:
    #one-qubit gate reliabiliy, two-qubit gate reliability, bsm reliability 
    ita_1, ita_2, ita_proj = 1, 1, 1
    req_num = 0        

    def __init__(self,src,dst,fidelity,bandwidth):
        self.src = src                  # QNode object!
        self.dst = dst                  
        self.id = self.req_num
        # self.req_num = self.req_num + 1  只会修改对象的公共变量副本的值！！
        self.fidelity = fidelity
        self.bandwidth = bandwidth
        self.start_time = ns.sim_time()
        self.finished_bandwidth = 0
        self.path_set = []
        # for analysis
        self.achieved_bandwidth = 0
        self.achieved_fidelity = []
        self.average_fidelity = 0
        self.success = 0                # -1: fail allocation ; 0: fail; 1:success
        self.finish_time = 0
        self.transtart_time = 0
        self.distillation_cost = []
    def path_finder(self, nodes: list, edges: list, k: list):
        #edges:list of (node1.id,node2.id,raw_fidelity,classical_delay)
        #nodes:list of QNode
        for node in nodes:
           node.get_waiting_time()
        #    if node.T != 0:
        #        print("node",node.ID," T :",node.T)
        graph = {node.ID:{'C':len(node.qmemory.unused_positions),'T':node.T, 'Link_count': 0} for node in nodes}
#        print(len(edges))
        for edge in edges:
            node1,node2 = edge[:2]
            graph[node1][node2] = edge
            graph[node2][node1] = edge
#        print(len(graph.keys()), graph)
        routing_T.append([])
        routing_C.append([])
        routing_F.append([])
        routing_D.append([])

        for i in range(4):                    #degree of nodes in lattice network <= 4
            path = find_best_path(self,self.src.ID,self.dst.ID,graph,k)
            if path:
                print(path)
#                self.path_set.append(path)
                link_path = []
                for j in range(len(path)-1):
                    # set network parameters in Links
                    edge = graph[path[j]][path[j+1]]
                    graph[path[j]]['Link_count'] = graph[path[j]]['Link_count'] + 1
                    graph[path[j + 1]]['Link_count'] = graph[path[j + 1]]['Link_count'] + 1
                    link = Link(left_node = path[j], right_node = path[j + 1], initial_fidelity = edge[2])
                    link_path.append(link)
                    del graph[path[j]][path[j+1]]
                    del graph[path[j+1]][path[j]]
                #update path_set
                self.path_set.append(link_path)
            else: break
        #averagely split node capacity to links in path_set of the request
        for path in self.path_set:
            for link in path:
                node1, node2 = link.left_node, link.right_node
                # capacity = min(graph[node1]['C'] // graph[node1]['Link_count'], graph[node2]['C'] // graph[node2]['Link_count'])
                capacity = graph[node1]['C']//4
                link.max_capacity = capacity
    def fidelity_decomposition(self):
        interVal_1 = 1 / 3 * (self.ita_1 ** 2) * self.ita_2 * (4 * (self.ita_proj ** 2) - 1)
        for path in self.path_set:
            interVal_2 = 3 * math.pow(interVal_1, len(path) - 1)
            individual_fidelity_target = (3 * math.pow((4 * self.fidelity - 1) / interVal_2, 1 / len(path)) + 1) / 4
            # print("end to end fidelity requirement", self.fidelity, "fidelity constraint", individual_fidelity_target)
#            bottleneck = 10000
            for link in path:
                link.decomposition_fidelity = individual_fidelity_target
                #link.decomposition_fidelity = 0.9
#                temporal value for test, cancel distillation

#               print("end to end fidelity", self.fidelity, "target", link.decomposition_fidelity, "original", link.initial_fidelity)
                link.run()

    def resource_allocation(self):
        routing_req.append(self.id)
        for i in range(len(self.path_set)):
            for j in range(len(self.path_set[i])):
                self.path_set[i][j].allocatedcapacity = min(math.ceil(self.bandwidth*0.8), self.path_set[i][j].allocatedcapacity)
                #self.path_set[i][j].allocatedcapacity = 10 #因Netsquid的限制，此参数不能大于7
        
        # for i in range(len(self.path_set)):
        #     for j in range(len(self.path_set[i])):
        #         self.path_set[i][j].allocatedcapacity = min(self.bandwidth, self.path_set[i][j].allocatedcapacity)
        

    def generate_node_requests(self, nodes: list):
    #convert link source demand to NodeRequests and attach them to corresponding QNode 
    #put in node.waiting_list[]
        node_set = set()
        # for i in range(len(self.path_set)):
        #     for j in range(len(self.path_set[i])):
        #         print(self.path_set[i][j].allocatedcapacity , end=' ')
        #     print("")
        for path in self.path_set:
            # if any link in the path is allocated 0 qubits, skip the path
            # print("new")
            if any(link.allocatedcapacity == 0 for link in path):
                print("has zero bandwidth!")
                continue
            NodeRequest.path_num += 1
            l_req = NodeRequest(self.id)
            r_req = NodeRequest(self.id)
            request_path_id[self.id].append(l_req.path_id)
            if self.id == 19:
                last_request_path_id.append(l_req.path_id)
            # print(self.id,l_req.path_id,r_req.path_id)
            # update: add direction info into repeater nodes' swapping & fwd protocols
            for index,link in enumerate(path):
                l_req.update_req_info(kind = "next", link = link)
                r_req.update_req_info(kind = "prev", link = link)
                self.distillation_cost.append(l_req.next_qubit_num + r_req.prev_qubit_num)
                # print(f'distillation cost:{l_req.prev_qubit_num + r_req.next_qubit_num}')
                l_req.calculate_service_time()
                # print(f'node{link.left_node} path {l_req.path_id} estimated_time{l_req.estimated_service_time}' )
                nodes[link.left_node - 1].waiting_queue.append(l_req)
                temp[link.left_node - 1] += l_req.prev_qubit_num + l_req.next_qubit_num
#                print("insert into node:", nodes[link.left_node - 1].ID, "total qubits:", l_req.prev_qubit_num + l_req.next_qubit_num, end = ' ')
                if index > 0:
                    nodes[link.left_node - 1].swap_protocol.clarify_path_direction(NodeRequest.path_num, get_direction(link.right_node - link.left_node))
                    nodes[link.left_node - 1].fwd_protocol.clarify_path_direction(NodeRequest.path_num, get_direction(link.right_node - link.left_node))
                # trigger_allocation(nodes[link.left_node - 1])
                node_set.add(nodes[link.left_node - 1])
                l_req = r_req
                r_req = NodeRequest(self.id)
            l_req.calculate_service_time()
            nodes[path[-1].right_node - 1].waiting_queue.append(l_req)
            nodes[path[-1].right_node - 1].dst_protocol.clarify_path_length(NodeRequest.path_num,len(path) + 1)
            # trigger_allocation(nodes[path[-1].right_node - 1])
            node_set.add(nodes[path[-1].right_node - 1])
#            print("insert into node:", nodes[path[-1].right_node - 1].ID, "total qubits:", l_req.prev_qubit_num + l_req.next_qubit_num)
            temp[path[-1].right_node - 1] += l_req.prev_qubit_num + l_req.next_qubit_num
        for repeater in node_set:
            trigger_allocation(repeater) 
        #individually inform dst node's HandleCinProtocol
        # self.dst.rec_protocol.rec_message()

class NodeRequest:
    path_num = 0
    def __init__(self, req_id):
        self.req_id = req_id
        self.path_id = self.path_num
        self.prev_node = 0      #id
        self.next_node = 0      #id
        self.prev_qubit_num = 0
        self.next_qubit_num = 0
        self.prev_backup_list = []
        self.next_backup_list = []
        # self.prev_purify_round = 0
        # self.next_purify_round = 0
        self.start_time = 0
        self.estimated_service_time = 0
        self.mem_positions = []
    def update_req_info(self, kind, link:Link):
        if kind == 'prev':
            self.prev_node = link.left_node
            self.prev_qubit_num = link.n_backup_list[0] * link.allocatedcapacity
            self.prev_backup_list = link.n_backup_list
        elif kind == 'next':
            self.next_node = link.right_node
            self.next_qubit_num = link.n_backup_list[0] * link.allocatedcapacity
            self.next_backup_list = link.n_backup_list
    def calculate_service_time(self):
        pre_purify_round = len(self.prev_backup_list) - 1
        next_purify_round = len(self.next_backup_list) - 1
        estimated_purify_time = max(pre_purify_round, next_purify_round) * purify_round_time
        self.estimated_service_time = link_entangle_time + estimated_purify_time + swap_time

def trigger_allocation(node):
    node.allocation_service.trigger()

def get_req_by_pathid(queue, value):
    for req in queue:
        if req.path_id == value:
            return req
    return None

def check_entanglement(req_list, node_req, bit_id):
    #check fidelity, refresh original req's achieved_fidelity & achieved_bandwidth
    path_id = node_req.path_id
    req_id = node_req.req_id
    req = req_list[req_id]
    qubit_a, = req.src.qmemory.peek(req.src.positions[path_id][bit_id])
    qubit_b, = req.dst.qmemory.peek(req.dst.positions[path_id][bit_id])
    fidelity = ns.qubits.fidelity([qubit_a,qubit_b],ns.b00,squared=True)
    all_qubit_fidelity.append(fidelity)
    # print( "req", req_id, "get entanglement:",ns.sim_time(ns.MILLISECOND),"ms")
    # print(f'req {req_id} required fidelity {req.fidelity}, path {path_id} bit {bit_id} achieved fidelity {fidelity}')
    if fidelity > req.fidelity:
        req.achieved_fidelity.append(fidelity)
        req.achieved_bandwidth += 1
    #req.achieved_bandwidth == req.bandwidth
    if req.achieved_bandwidth >= req.bandwidth:
        req.finish_time = ns.sim_time()
        req.transtart_time = max(req.transtart_time,node_req.start_time)
        req.success = 1
    
def free_path_resource(node_list, node, node_req):
    # always starts from dst node, ends at src
    path_id = node_req.path_id
    node.qmemory.pop(node_req.mem_positions)
    # check the memory status, ensure poped positions are available for other positions
    '''
        passed!!
        print(f'poped positions:{node_req.mem_positions}, available positions:{node.qmemory.unused_positions}')
    '''
    # remove node_req from the node's running_list
    if node_req in node.running_list:
        node.running_list.remove(node_req)
        trigger_allocation(node)
    else:
        node.waiting_queue.remove(node_req)
    
    real_time = ns.sim_time() - node_req.start_time
    node.service_time[path_id] = [real_time, node_req.estimated_service_time]
    # check running_list status
    '''
        passed!
        print(f'search result of path {path_id} in node {node.ID}\'s running_list: {get_req_by_pathid(node.running_list,path_id)}')
    '''
    # find next node & node_req to deal with
    if node_req.prev_qubit_num > 0:
        next_node = node_list[node_req.prev_node - 1]
        next_req = get_req_by_pathid(next_node.running_list, path_id)
        if next_req is None:
            # 上游节点还没开始任务
            next_req = get_req_by_pathid(next_node.waiting_queue, path_id)
        free_path_resource(node_list, next_node, next_req)


def get_direction(distance):
    if distance == 1:
        return 'R'
    elif distance == -1:
        return 'L'
    elif distance > 1:
        return 'D'
    else:
        return 'U'

# def entangle_time(node_a:Node,node_b:Node):
#     return 1000                                                 # to be fixed

# def purify_time(node_a:Node,node_b:Node):                       
#     return 1000                                                 # to be fixed

class QNode(Node):
    def __init__(self, name, ID=None, qmemory=None, port_names=None):
        super().__init__(name, ID, qmemory, port_names)
        self.waiting_queue: deque[NodeRequest] = deque()
        self.running_list: list[NodeRequest] = []
#        self.available_qubits_number=len(self.qmemory.unused_positions)
        self.T = 0
        self.link_service: LinkProtocol = None
        self.purify_service = {'L': None, 'R': None, 'U': None, 'D': None}
        self.allocation_service: None
        self.swap_service: None
        self.correct_service: None
        self.positions = {}
        self.fwd_protocol = None
        self.src_protocol = None
        self.dst_protocol = None
        self.swap_protocol = None
        self.service_time = {}
        # self.rec_protocol = None
                      
    def get_waiting_time(self):
        # avoid changing real list in nodes
        # here only revise start_time of req in waiting queue(will be overwritten)
        if len(self.waiting_queue) == 0 and len(self.running_list) == 0:
            return
        predict_queue = copy.copy(self.waiting_queue)
        running_queue = copy.copy(self.running_list)
        occupied_qubits = sum(req.prev_qubit_num + req.next_qubit_num for req in running_queue)
        while len(predict_queue) > 0:
            next_request = predict_queue.popleft()
            t = ns.sim_time()
            # print("curr:", t)
            while(occupied_qubits + next_request.prev_qubit_num + next_request.next_qubit_num > len(self.qmemory.mem_positions)):
                nearest_request = min(running_queue,key=lambda req:req.start_time + req.estimated_service_time)
                t = nearest_request.start_time + nearest_request.estimated_service_time
                # print("next req start at:", nearest_request.start_time, "estimate service time:", nearest_request.estimated_service_time,"req finish time:")
                occupied_qubits = occupied_qubits - nearest_request.prev_qubit_num - nearest_request.next_qubit_num
                running_queue.remove(nearest_request)
            running_queue.append(next_request)
            occupied_qubits = occupied_qubits + next_request.prev_qubit_num - next_request.next_qubit_num
            next_request.start_time = t                          #influences need to be judge   
        # the last req in predict queue must in it
        latest_req = max(running_queue, key = lambda req: req.start_time + req.estimated_service_time)
        self.T = max(0, latest_req.start_time + latest_req.estimated_service_time - ns.sim_time())
        if self.T != 0:
            waiting_time.append(self.T)
            # print("node:", self.ID, " T:", self.T)
    
    # def running_time_estimation(self,request:NodeRequest):
    #     estimated_entangle_time = max(entangle_time(self,request.prev_node),entangle_time(self,request.next_node))
    #     estimated_purification_time = max(request.prev_purify_round*purify_time(self,request.prev_node), \
    #                                         request.next_purify_round*purify_time(self,request.next_node))
    #     estimated_swapping_time = 1000          # to be fixed, depending on swapping program and PhysicalInstructions defined on this node
    #     request.estimated_service_time = estimated_entangle_time + estimated_purification_time + estimated_swapping_time 
def calculate_ext(W, h, p, q):
    """
    计算期望吞吐量
    
    参数：
    W -- 路径宽度
    h -- 路径长度
    p -- 单个通道的成功率
    q -- 每次纠缠交换的成功率
    
    返回：
    EXT -- 期望吞吐量
    """
    # 初始化概率变量
   # print(2)
    Q = [0] * (W + 1)
    P = [[0] * (W + 1) for _ in range(h + 1)]
    
    # 计算单跳的概率
    for i in range(1, W + 1):
        Q[i] = math.comb(W, i) * (p ** i) * ((1 - p) ** (W - i))
    
    # 计算多跳的概率
    for k in range(1, h + 1):
        for i in range(1, W + 1):
            if k==1:
                P[k][i]=Q[i]
            else:
                P[k][i] = P[k-1][i] * sum(Q[l] for l in range(i, W + 1)) + Q[i] * sum(P[k-1][l] for l in range(i + 1, W + 1))
            #P[k][i] = P[k - 1][i] * sum(Q[l] for l in range(i,W+1))
            #if i > 1:
             #   P[k][i] += Q[i] * sum(P[k - 1][l] for l in range(i+1,W+1))
    # 计算EXT
    #print(sum(i * P[h][i] for i in range(1, W + 1)))
    ext = q ** h * sum(i * P[h][i] for i in range(1, W + 1))
    return ext

def get_p(graph,node1,node2,fidelity_target):
    fidelity=graph[node1][node2][2]
    new_fidelity_list=[]
    p_list=[]
    i=0
    new_fidelity_list.append(fidelity)
    p_list.append(1)
    p_f=1
    while fidelity < fidelity_target:
        i=i+1
        F1, F2 = fidelity, fidelity
        p_new = F1 * F2+ (1-F1)*(1-F2)
        fidelity=F1*F2/p_new
        new_fidelity_list.append(fidelity)
        p_list.append(p_new)
        if fidelity < fidelity_target:
            p_f=p_f*p_list[i]*p_f*p_list[i]
        else:    
            p_f=p_f*p_list[i]
            
        if fidelity>0.999:
            break
    return p_f


def find_best_path2(request,src:int,dst:int,graph:dict,k:list,bandwith,fidelity_target):
    T_path = [graph[src]['T']] * len(graph)
    C_path = [graph[src]['C']] * len(graph)
    old_metric = [float('inf')] * len(graph)
    old_metric[src - 1] = k[0] * T_path[src - 1] / C_path[src - 1]
    #h=1
    unvisited = list(graph.keys())
    visited=[False]*len(graph)
    metric = [float('-inf')] * len(graph)
    path = [[] for i in range(len(graph))]
    metric[src - 1] = float('inf')
    path[src - 1] =[src]
    while(unvisited):
        current_node_id = max(unvisited, key = lambda node_id: metric[node_id - 1])
        h = len(path[current_node_id - 1])
        #print(current_node_id)
#        print(current_node_id)
        current_neighbors = {node_id for node_id in graph[current_node_id].keys() if isinstance(node_id, int)}
        for neighbor_node_id in current_neighbors:
            temp_C = min(graph[neighbor_node_id]['C'], C_path[current_node_id - 1])
            temp_T = max(graph[neighbor_node_id]['T'], T_path[current_node_id - 1])
            new_old_metric = old_metric[current_node_id - 1] + k[0] * (temp_T / temp_C - T_path[current_node_id - 1] / C_path[current_node_id - 1]) - \
                 k[1] * math.log(graph[current_node_id][neighbor_node_id][2]) + k[2] *graph[current_node_id][neighbor_node_id][3]
            if new_old_metric < old_metric[neighbor_node_id - 1]:
                old_metric[neighbor_node_id - 1] = new_old_metric
                C_path[neighbor_node_id - 1] = temp_C
                T_path[neighbor_node_id - 1] = temp_T
            #h=len(path[neighbor_node_id - 1])
            p=get_p(graph,current_node_id,neighbor_node_id,fidelity_target)
            #p=graph[current_node_id][neighbor_node_id][2]
            #p=0.995
            W=bandwith
            new_metric = calculate_ext(W,h,p,0.99)
            #print(new_metric)
            #print('good')
            # test proper parameters:
            # if temp_T != 0:
            #     print(f"first term: { k[0] * temp_T / temp_C}, one-hop second term: {-k[1] * math.log(graph[current_node_id][neighbor_node_id][2])}, one-hop last term:{k[2] *graph[current_node_id][neighbor_node_id][3]}")
            if new_metric > metric[neighbor_node_id - 1]:
                metric[neighbor_node_id - 1] = new_metric
                path[neighbor_node_id - 1] = path[current_node_id - 1] + [neighbor_node_id]
            if neighbor_node_id == dst and request.req_num == 20:
                F_path = 0
                D_path = 0
                for i in range(len(path[dst - 1]) - 1):
                    F_path -= k[1] * math.log(graph[path[dst - 1][i]][path[dst - 1][i + 1]][2])
                    D_path += k[2] * graph[path[dst - 1][i]][path[dst - 1][i + 1]][3]
                print(f"path:{path[dst - 1]}, T_path:{T_path[dst - 1]}, C_path:{C_path[dst - 1]}, F_path:{F_path}, D_path:{D_path}, metric:{old_metric[dst - 1]}") 
            if neighbor_node_id == dst:
                return path[dst - 1]
                #visited[neighbor_node_id-1]=True
            #h=h+1
               # print('good',path[neighbor_node_id - 1])
#       print(unvisited, current_node_id)
        #print(metric)
        unvisited.remove(current_node_id)
 #   print(metric)
    # if request.req_num == 20:
    #     F_path = 0
    #     D_path = 0
    #     for i in range(len(path[dst - 1]) - 1):
    #         F_path -= k[1] * math.log(graph[path[dst - 1][i]][path[dst - 1][i + 1]][2])
    #         D_path += k[2] * graph[path[dst - 1][i]][path[dst - 1][i + 1]][3]
    #     print(f"path:{path[dst - 1]}, T_path:{T_path[dst - 1]}, C_path:{C_path[dst - 1]}, F_path:{F_path}, D_path:{D_path}, metric:{old_metric[dst - 1]}") 
    if metric[dst - 1] == float('-inf'):
        #print('good')
        return None
    else: 
        #print('good',path[dst-1])
        return path[dst - 1]

             

def find_best_path(request,src:int,dst:int,graph:dict,k:list):
    T_path = [graph[src]['T']] * len(graph)
    C_path = [graph[src]['C']] * len(graph)
    unvisited = list(graph.keys())
    # mind the difference between node_id and list index!!!
    # T_path = [graph[src]['T']] * len(graph)
    # C_path = [graph[src]['C']] * len(graph)
    metric = [float('inf')] * len(graph)
    path = [[] for i in range(len(graph))]
    metric[src - 1] = 0
    path[src - 1] =[src]
    while(unvisited):
        current_node_id = min(unvisited, key = lambda node_id: metric[node_id - 1]) 
#        print(current_node_id)
        current_neighbors = {node_id for node_id in graph[current_node_id].keys() if isinstance(node_id, int)}
        for neighbor_node_id in current_neighbors:
            temp_C = min(graph[neighbor_node_id]['C'], C_path[current_node_id - 1])
            temp_T = max(graph[neighbor_node_id]['T'], T_path[current_node_id - 1])
            new_metric = metric[current_node_id - 1] - k[1] * math.log(graph[current_node_id][neighbor_node_id][2]) 
            # test proper parameters:
            # if temp_T != 0:
            #     print(f"first term: { k[0] * temp_T / temp_C}, one-hop second term: {-k[1] * math.log(graph[current_node_id][neighbor_node_id][2])}, one-hop last term:{k[2] *graph[current_node_id][neighbor_node_id][3]}")
            if new_metric < metric[neighbor_node_id - 1]:
                metric[neighbor_node_id - 1] = new_metric
                path[neighbor_node_id - 1] = path[current_node_id - 1] + [neighbor_node_id]
                C_path[neighbor_node_id - 1] = temp_C
                T_path[neighbor_node_id - 1] = temp_T
#        print(unvisited, current_node_id)
        unvisited.remove(current_node_id)

    F_path = 0
    D_path = 0
    for i in range(len(path[dst - 1]) - 1):
        F_path += graph[path[dst - 1][i]][path[dst - 1][i + 1]][2]
        D_path += graph[path[dst - 1][i]][path[dst - 1][i + 1]][3]/100000
    if path[dst - 1]:
        routing_T[request.req_num - 1].append(T_path[dst - 1])
        routing_C[request.req_num - 1].append(C_path[dst - 1])
        routing_F[request.req_num - 1].append(F_path/(len(path[dst - 1]) - 1))
        routing_D[request.req_num - 1].append(D_path)
    
    """ if request.req_num == 20:
        F_path = 0
        D_path = 0
        for i in range(len(path[dst - 1]) - 1):
            F_path -= k[1] * math.log(graph[path[dst - 1][i]][path[dst - 1][i + 1]][2])
            D_path += k[2] * graph[path[dst - 1][i]][path[dst - 1][i + 1]][3]
        print(f"path:{path[dst - 1]}, T_path:{T_path[dst - 1]}, C_path:{C_path[dst - 1]}, F_path:{F_path}, D_path:{D_path}, metric:{metric[dst - 1]}") """
#    print(metric)

    if metric[dst - 1] == float('inf'):
        return None
    else: return path[dst - 1]



def Cmn(m,n):  #组合数的函数
    from math import factorial
    if m>=n:
        return factorial(m)//factorial(n)//factorial(m-n)    
    else:
        return 0

class Path:
    #swapping probability, set identically for all nodes in network
    swapping_probability = 0.99
    
    def __init__(self):
        self.Pbm = []
        self.Pnm = []
        
    def calculate_Pbm(self, path_set, m, bm):    #计算Pbm的函数
        pbm1 = 1
        pbm2 = 1 
        for k in range(len(path_set[m])):
            sum1 = 0
            for i in range(bm, path_set[m][k].allocatedcapacity + 1):
                sum1 += path_set[m][k].Pmk[i]
            pbm1 *= sum1
            
        for k in range(len(path_set[m])):
            sum2 = 0
            for i in range(bm + 1, path_set[m][k].allocatedcapacity + 1):
                sum2 += path_set[m][k].Pmk[i]
            pbm2 *= sum2
            
        Pbm_bm = pbm1 - pbm2
        self.Pbm.append(Pbm_bm)
        
    def calculate_Pnm(self, path_set, m, nm):    #计算Pnm的函数
        Pnm_nm = 0
        for bm in range(nm, min(path_set[m], key=lambda x: x.allocatedcapacity).allocatedcapacity+1):
            Pnm_nm += Cmn(bm,nm)  * self.swapping_probability ** (len(path_set[m]) * nm) * (1 - self.swapping_probability ** len(path_set[m])) ** (bm - nm) * self.Pbm[bm]
#        print("Cmn(bm,nm)" , Pnm_nm)
        self.Pnm.append(Pnm_nm)

                              
def Calculate_Pu(path, qubit_num):
#    time0 = time.time()
    def backtrack(m, path_indices, product):
        if m == len(path):
            if sum(path_indices) > qubit_num:
                res.append(product)
            return
        for nm in range(len(path[m].Pnm)):
            path_indices.append(nm)
            sum1 = 0
            for k in range( m + 1, len(path)):
                sum1 += len(path[k].Pnm)
            if sum(path_indices) + sum1 >= qubit_num:
                backtrack(m + 1, path_indices, product * path[m].Pnm[nm])
                path_indices.pop()
#        print(len(path[m].Pnm))

    res = []
    backtrack(0, [], 1)
#    print(res)
#    time1 = time.time()
    # print("pu :",sum(res))
    return sum(res)

def optimize(path_set, path, qubit_num):
    count = 1
    threshold = 0.95
    time1 = time.time()
    for i in range(len(path_set)):
        for j in range(len(path_set[i])):
            if path_set[i][j].allocatedcapacity > int(1.2 * min(path_set[i], key=lambda x: x.allocatedcapacity).rescapacity):
                path_set[i][j].allocatedcapacity =int(1.2 * min(path_set[i], key=lambda x: x.allocatedcapacity).rescapacity)
            path_set[i][j].calculate_Pmk()  #根据分配的容量进行Pmk的计算        

#    print("path_set_len:", len(path_set))
    for m in range(len(path_set)):
        for bm in range(min(path_set[m], key=lambda x: x.allocatedcapacity).allocatedcapacity+1):
            path[m].calculate_Pbm(path_set, m, bm) #路径为m，遍历所有路径和bm，计算Pbm
        #print("Pbm" , path[m].Pbm, sum(path[m].Pbm))
    # time2 = time.time()
    # print("Pbm time:", time2 - time1)     

    for m in range(len(path_set)):
        for nm in range(min(path_set[m], key=lambda x: x.allocatedcapacity).allocatedcapacity+1):
            path[m].calculate_Pnm(path_set, m, nm) #遍历所有路径和nm，计算Pnm
        #print("Pnm" , path[m].Pnm, sum(path[m].Pnm))
    # time3 = time.time()
    # print("Pnm time:", time3 - time1) 
    Pu = Calculate_Pu(path, qubit_num)
    while(count):        
        #Pu = Calculate_Pu(path, qubit_num) #用回溯算法递归实现寻找满足传输要求的所有组合，C为任务要求的比特数，计算Pu
#        print("Pu" , Pu)
        if(Pu >= threshold):        
            if((-np.log10(1-Pu))>5):
                for i in range(len(path_set)):
                    res=int(min(path_set[i], key=lambda x: x.allocatedcapacity).allocatedcapacity * (-np.log10(1-Pu)-4)/(-np.log10(1-Pu)))
                    for j in range(len(path_set[i])):
                        path_set[i][j].allocatedcapacity -=res
                        if path_set[i][j].allocatedcapacity > int(1.2 * min(path_set[i], key=lambda x: x.allocatedcapacity).allocatedcapacity):
                            path_set[i][j].allocatedcapacity =int(1.2 * min(path_set[i], key=lambda x: x.allocatedcapacity).allocatedcapacity)
                        path_set[i][j].Pmk=[]
                        path_set[i][j].calculate_Pmk()

                    
                for m in range(len(path_set)):
                    path[m].Pbm=[]
                    for bm in range(min(path_set[m], key=lambda x: x.allocatedcapacity).allocatedcapacity+1):
                        path[m].calculate_Pbm(path_set,m,bm) #路径为m，遍历所有路径和bm，计算Pbm
                    #print("Pbm" , path[m].Pbm,sum(path[m].Pbm))
                        
                for m in range(len(path_set)):
                    path[m].Pnm=[]
                    for nm in range(min(path_set[m], key=lambda x: x.allocatedcapacity).allocatedcapacity+1):
                        path[m].calculate_Pnm(path_set,m,nm) #遍历所有路径和nm，计算Pnm
                    #print("Pnm" , path[m].Pnm,sum(path[m].Pnm))
                Pu = Calculate_Pu(path, qubit_num)
            else:
                max_row, max_col = max(((i, j) for i, row in enumerate(path_set) for j, x in enumerate(row)), key=lambda x: (path_set[x[0]][x[1]].allocatedcapacity, path_set[x[0]][x[1]].n_backup_list[0]))  #按照已分配的纯化后比特数和纯化所需资源进行排序
                path_set[max_row][max_col].allocatedcapacity -= 1  #满足要求则递减
                path_set[max_row][max_col].Pmk=[]
                path_set[max_row][max_col].calculate_Pmk()
                path[max_row].Pbm=[]
                for bm in range(min(path_set[max_row], key=lambda x: x.allocatedcapacity).allocatedcapacity+1):
                    path[max_row].calculate_Pbm(path_set, max_row, bm)
                path[max_row].Pnm=[]
                for nm in range(min(path_set[max_row], key=lambda x: x.allocatedcapacity).allocatedcapacity+1):
                    path[max_row].calculate_Pnm(path_set, max_row, nm)
                Pu = Calculate_Pu(path, qubit_num)
        
        elif(count>1 and Pu <= threshold):
            path_set[max_row][max_col].allocatedcapacity += 1  #刚好低于阈值则回退一次容量，并退出循环
            path_set[max_row][max_col].Pmk=[]
            path_set[max_row][max_col].calculate_Pmk()
            path[max_row].Pbm=[]
            for bm in range(min(path_set[max_row], key=lambda x: x.allocatedcapacity).allocatedcapacity+1):
                path[max_row].calculate_Pbm(path_set,max_row,bm)
            path[max_row].Pnm=[]
            for nm in range(min(path_set[max_row], key=lambda x: x.allocatedcapacity).allocatedcapacity+1):
                path[max_row].calculate_Pnm(path_set, max_row, nm)
            Pu = Calculate_Pu(path, qubit_num)
            if(Pu >= threshold):
                break
        
        else:
            print("fail to allocate capacity for request")           #fail to meet the probability requirement for the first time
            for path in path_set:
                for link in path:
                    link.allocatedcapacity = 0
            print("last Pu" , Pu)
            return False
        count += 1
    time4 = time.time()
    # print("whole allocation time:", time4 - time1)    
    # print("last Pu" , Pu)
    print("最终容量分配为：")
    for i in range(len(path_set)):
        for j in range(len(path_set[i])):
            print(path_set[i][j].allocatedcapacity , end=' ')
        print("")
    return True


def random_request_generator(scale:int,network:Network,node_capacity:int,s,d):
    # src = scale * random.randint(0, scale - 1) + random.randint(0, scale - 1)     # need to confirm names of nodes
    # while True:
    #     x1 = random.randint(0, scale - 1)
    #     y1 = random.randint(0, scale - 1)
    #     x2 = random.randint(0, scale - 1)
    #     y2 = random.randint(0, scale - 1)
    #     if abs(x2 - x1) + abs(y2 - y1) > scale * 0.7:
    #         src = x1 * scale + y1
    #         dst = x2 * scale + y2
    #         break
    src = s
    dst = d
    # src = 11
    # dst = 98
    fidelity = 0.8
    # fidelity = 0.5 + 0.2*random.random()
    # fidelity = 0.8 + 0.1*random.random()
#    bandwidth = random.randint(1, node_capacity//100 + 1)
# only for test    
    bandwidth = req_bandwidth
    nodes = sorted(network.nodes.values(), key=lambda x: x.ID) 
    req = Request(nodes[src], nodes[dst], fidelity, bandwidth)
    return req

def noise_parameter(fidelity, random_flag):
# get a parameter to simulate bit-flip error
# following steps are:
# noise_sim = ops.create_rotation_op(theta)
# operate(qubit, noise_sim)
    theta = 2 * math.acos(math.sqrt(fidelity))
    if random_flag:
        return random.gauss(theta, 0.02)
    else:
        return theta

# protocols
class RandomRequest(LocalProtocol):
    def __init__(self, network: Network, edges:list, frequency: int, position_num = None, nodes=None, name=None, max_nodes=-1):
        super().__init__(nodes, name, max_nodes)
        self.edges = edges
        self.network = network
        self.pos_num = position_num
        self.frequency = frequency
        self.src_dst_list = []
    def run(self):
        with open(src_dst_file, 'r') as f:
            for line in f:
                self.src_dst_list.append(int(line.strip()))
        while Request.req_num < req_num :
            # frequency = 1/mean_arrival_time
            duration = random.expovariate(self.frequency)
            if len(requests) > 0:
                yield self.await_timer(duration)
            # assume node_capacity = 100, could be changed
            nodes = sorted(network.nodes.values(), key=lambda x: x.ID)
            scale = int(math.sqrt(len(self.network.nodes)))
            src = self.src_dst_list[2 * Request.req_num]
            dst = self.src_dst_list[2 * Request.req_num + 1]
            new_req = random_request_generator(scale,self.network,self.pos_num,src,dst)
            Request.req_num += 1
            print("time:", ns.sim_time(), new_req.src, new_req.dst, new_req.bandwidth, new_req.fidelity)
            #set k which  makes the magnitudes in the equation match
            k = [1e-5, 2, 1e-6]                             # metric weights to be specified
            time1 = time.time()
            new_req.path_finder(nodes, self.edges, k)
            time2 = time.time()
            print("finish discovery in:", time2 - time1)
            new_req.fidelity_decomposition()
            new_req.resource_allocation()
            time3 = time.time()
            # print("finish allocation at:", time3 - time1)
            if new_req.success != -1:
                new_req.generate_node_requests(nodes)
            requests.append(new_req)


class SRInitProtocol(NodeProtocol):
    # 初始化协议，部署在源端节点，负责启动通信过程。

    position=namedtuple('position', ['path_id', 'positions'])

    def __init__(self, node=None, name=None):
        super().__init__(node, name)
    
    def run(self):
        self.send_signal(signal_label=Signals.SUCCESS)
    
    def get_positions(self,position):
        # 接收DEJMPS协议对象发来的存储位。
        self.node.positions[position.path_id]=position.positions # 这是为了方便最后展示路径保真度结果。


class FCMProtocol(NodeProtocol):
    """经典消息转发协议，部署到路径上的中间节点，使得中间节点接收到上游节点通过经典信道传来的信息后，将其继续向下游节点转发。
    最终这些经典信息会到达发送方。"""
    def __init__(self, node=None, name=None, prompt: bool = True):
        """directions：各条路径的离开方向。字典的键为路径号，值为该路径的离开方向。
        direction_order：一个为使程序更简洁而设置的成员，列表中的项分别代表左、上、右、下。
        prompt：是否需要在运行过程中向屏幕上输出运行状态信息。"""
        super().__init__(node, name)
        self.name=name
        self.directions={}
        self.direction_order=['L','U','R','D']
        self.prompt = prompt

    def clarify_path_direction(self, path_id, direction):
        # 声明路径的离开方向。
        if len(direction)!=1:
            raise Exception(f'{self.name}: Path direction format error on path {path_id}.')
        self.directions[path_id]=direction
        
    def run(self):
        while True:
            expr1 = self.await_port_input(self.node.ports[f'cinL'])
            expr2 = self.await_port_input(self.node.ports[f'cinU'])
            expr3 = self.await_port_input(self.node.ports[f'cinR'])
            expr4 = self.await_port_input(self.node.ports[f'cinD'])
            expr = yield (expr1 | expr2) | (expr3 | expr4)
            # expr=self.await_port_input(self.node.ports[f'cinL']) | self.await_port_input(self.node.ports[f'cinU']) | self.await_port_input(self.node.ports[f'cinR']) | self.await_port_input(self.node.ports[f'cinD'])
            # yield expr
            # print(f"{self.name}FCMP triggered")
            m=[] # 此列表用来容纳经典信道接收到的消息。
            flag = 0
            for d in self.direction_order:
                m.append(self.node.ports[f'cin{d}'].rx_input())
            c = 0 # 此变量用来检查收到的信息是否有错误。
            for i in range (4):
                # 按路径号检查收到的经典信息是来自哪条路径，并将其向下游中间节点转发。
                if m[i] is None :
                    c+=1
                    continue
                elif len(m[i].items)%4 != 0: # 经典信息的长度必须为4的整数倍。详见SwapProtocol发送经典信息的过程。
                    print(f'Warning: {self.name} detected message error. The error message is {m[i].items}.')
                    c+=1
                    continue
                if not m[i].items[0] in self.directions.keys():
                    # print(f'{self.name} received message {m[i]} from {self.direction_order[i]}')
                    for j in range(0,len(m[i].items),4):
                        # 将合并在一起的数据包分开，并将其转换为cmessage
                        t=self.node.dst_protocol.cmessage(m[i].items[j], m[i].items[j+1], m[i].items[j+2], m[i].items[j+3])
                        self.node.dst_protocol.get_cmessage(t)
                    # t=self.node.dst_protocol.cmessage(m[i].items[0], m[i].items[1], m[i].items[2], m[i].items[3])
                    # self.node.dst_protocol.get_cmessage(t)
                    # print(f'{self.name} received message {m[i]} from {self.direction_order[i]} and give it to CorrectProtocol')
                    flag = 1
                    break
                    # raise Exception(f'{self.name}: directions of path {m[i].items[0]} has not been clarified.')
                forward_direction=self.directions[m[i].items[0]] # 获取经典数据包所在路径的离开方向
                self.node.ports[f'cout{forward_direction}'].tx_output(m[i])
                # if self.prompt:
                    # print(f'{self.name} received message {m[i]} from {self.direction_order[i]} and forward it to {forward_direction}')
                c-=1
            if(flag):
                continue
            if c==4:
                print(f'Warning: {self.name} triggered but no valid message received.')


class SwapProtocol(NodeProtocol):
    """该协议是纠缠交换（中间节点）协议，执行中间节点的纠缠交换操作。它从DEJMPS协议对象接收需要进行纠缠交换的存储位，
    执行纠缠交换后，将贝尔基下的联合测量结果经由经典信道向目的端方向发送。每个节点需要分配一个本协议对象。"""

    position=namedtuple('position', ['path_id', 'positions'])
    """DEJMPS协议对象按此格式给本协议对象发送存储位信息。
    path_id：路径编号
    positions：需要进行纠缠交换的存储位"""

    def __init__(self, node=None, name=None, fail_rate=0):
        """direction：每一条路径的穿越方向。字典的键为路径号，值为穿越方向。
        fr：纠缠交换失败的概率。
        key_counter：此成员部分决定了纯化过程中所用到的量子程序的key值。这是为防止不同纯化轮的程序输出结果被弄混。
        lsn：上一次纠缠交换当中，交换成功的比特数。
        position_queue：本协议第二次收到某一路径上需要进行纠缠交换的存储位时，它将被存储在本队列中。若队列里没有其它等待处理的
        存储位，则它将触发一次纠缠交换。
        ppt：是否需要在运行过程中向屏幕上输出运行状态信息。
        prepared_positions：本协议第一次收到某一路径上需要进行纠缠交换的存储位时，它将被存储在本队列中，等待该路径上的第二个
        纠缠交换存储位。
        record_result：若量子处理器执行量子程序出现混乱，则将非本轮纯化的量子程序输出暂存在此。这是为多线程并行运行而设计的成员变量。
        sP：源端的初始化协议对象。
        _program：纠缠交换过程中涉及到的量子程序。"""
        super().__init__(node, name)
        self.direction={}
        self.fr=fail_rate
        self.key_counter=0
        self.lsn=0
        self.name = name
        self.positions_queue=Queue()
        #change ppt to a directory
        self.ppt = True
        # 这里！！del之后下标会改变！需要修改，或许用字典好些
        # 简单起见暂时取消了del
        self.prepared_positions=[]
        self.record_result={}
        self.sP=None
        self._qprogram_complete_signal='Quantum Program Finished'
        self._start_swap_signal='Position Matched'
        self.add_signal(self._qprogram_complete_signal)
        self.add_signal(self._start_swap_signal)
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(instr.INSTR_MEASURE_BELL, [q1, q2], output_key = name, inplace=False)
    
    
    def clarify_path_direction(self,path_id ,direction: str=''):
        # 声明本节点的路径穿越方向。
        if len(direction)!=1:
            raise Exception('Path string length error!')
        self.direction[path_id]=direction

    def get_positions(self, positions):
        """接收需要进行纠缠交换的存储位。本方法由本协议对象所在节点的DEJMPS协议调用。
        纠缠交换需要知道两套存储位：上游方向的存储位和下游方向的存储位。因此，纠缠交换的过程需要等待两套存储位全部被接收后才能
        开始。两套存储位都通过本方法来接收。"""
        matched=False # 该变量记录本次接收到的是第一套存储位还是第二套存储位
        # 判断本次接收到的是第一套存储位还是第二套存储位
        for i in range(len(self.prepared_positions)):
            # 尝试在prepared_positions[]中寻找同一路径号的存储位
            if positions.path_id==self.prepared_positions[i].path_id:
                self.positions_queue.put([i,positions]) # 将新收到的存储位放入positions_queue
                matched=True
                self.send_signal(self._start_swap_signal, result=i) # 发送信号，触发纠缠交换
                """ file3 = open(filename3, 'a')
                for pos in positions.positions:
                    file3.write(f'Node {self.node.ID}, path_id {positions.path_id} get {self.node.qmemory.peek(pos)}\n')
                file3.close() """
                # if self.ppt:
                #     print(f'{self.name} get {positions}')
                break
        """ file2 = open(filename2, 'a')
        file2.write(f'Node {self.node.ID} Swap, {matched}, path_id={positions.path_id}, succeeded positions={positions.positions}\n')
        file2.close() """
        if not matched:
            self.prepared_positions.append(positions) # 将新收到的存储位放入prepared_positions[]
            # if self.ppt:
            #     print(f'{self.name} get {positions}.')
    
    # def get_qprogram_result(self, result, output_key):
    #     # 带队列的量子处理器向本协议对象交付量子程序输出结果。该方法是为多线程并行运行而设计。
    #     m=result[output_key][0]
    #     if output_key==f'{self.name}_{self.key_counter}':
    #         self.send_signal(self._qprogram_complete_signal, result)
    #     self.record_result[output_key]=m
    
    # def get_sender_protocol(self, sP:NodeProtocol=None):
    #     # 声明源端的初始化协议
    #     self.sP=sP
    
    def refresh_key(self):
        # 更新key_counter成员。
        if self.key_counter==99:
            self.key_counter=0
            self.record_result.clear()
        else:
            self.key_counter+=1

    def run(self):
        # if self.sP==None:
        #     raise Exception(f'{self.name}: Sender Initialization Protocol has not been clarified!')
        # # 等待源端初始化完成
        # yield self.await_signal(sender=self.sP, signal_label=Signals.SUCCESS)
        while True:
            yield self.await_signal(self, self._start_swap_signal)
            # 获取路径上游和下游方向上的存储位（均为namedtuple）
            # 其中A[0]是path_id, A[1]是纯化好的存储位列表
            i,A=self.positions_queue.get()
            B=self.prepared_positions[i]
            # del self.prepared_positions[i]
            if A[0]!=B[0]:
                raise Exception(f'{self.name}: path ids do not match.')
            # print('path_id',A[0])
            bmin=min(len(A[1]), len(B[1])) # 判断需要进行纠缠交换的比特数量
            # print(bmin)
            # add time duration of swapping operation
            yield self.await_timer(random.gauss(swap_time, swap_deviation_time))

            for b in range(bmin):
                # self.fr = 0
                if random.uniform(0.0,1.0)>=self.fr: # 使用随机数推算本次纠缠交换应成功还是失败。若应失败，则不进行纠缠交换。
                    qubit_map=[A[1][b], B[1][b]]
                    # req=self.node.subcomponents[f'QPU_{self.node.ID}_1'].request(self,self._program,qubit_map,f'{self.name}_{self.key_counter}')
                    while self.node.qmemory.busy:
                        yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(self._program,qubit_mapping=qubit_map)
                    try:
                        m=self._program.output[self.name][0]
                    except KeyError:
                        m = 'failure'
                    if not A[0] in self.direction.keys():
                        raise Exception(f'{self.name}: Direction of path {A[0]} has not been clarified.')
                    if len(self.direction[A[0]])!=1:
                        raise Exception(f'{self.name}: Path direction length error on path {A[0]}.')
                    # 用经典数据包，向路径下游方向发送贝尔基下的联合测量结果。
                    # 经典数据包格式：[路径号，比特号，测量结果，将进行纠缠交换的比特总数]。
                    # 发送将要进行纠缠交换的比特总数的原因是，接收方可以根据路径上不同节点发来的这一数字，计算出纯化、纠缠交换均成功的比特数。
                    if self.direction[A[0]]=='L':
                        self.node.ports[f'coutL'].tx_output(Message([A[0], b, m, bmin], header=self.name))
                    elif self.direction[A[0]]=='U':
                        self.node.ports[f'coutU'].tx_output(Message([A[0], b, m, bmin], header=self.name))
                    elif self.direction[A[0]]=='R':
                        self.node.ports[f'coutR'].tx_output(Message([A[0], b, m, bmin], header=self.name))
                    elif self.direction[A[0]]=='D':
                        self.node.ports[f'coutD'].tx_output(Message([A[0], b, m, bmin], header=self.name))
                    else:
                        raise Exception(f'{self.name}: Path format error!')
                    # if self.ppt:
                    #     print(f"{self.name}: last processing result is {[A[0], b, m, bmin]}")
                    self.refresh_key()
                # 纠缠交换失败的情形，用经典数据包向目的端进行通告
                else:
                    # print(f'{self.name}: Swapping failed in qubit {b}, path {A[0]}.')
                    if self.direction[A[0]]=='L':
                        self.node.ports[f'coutL'].tx_output(Message([A[0], b, 'failure', bmin], header=self.name))
                    elif self.direction[A[0]]=='U':
                        self.node.ports[f'coutU'].tx_output(Message([A[0], b, 'failure', bmin], header=self.name))
                    elif self.direction[A[0]]=='R':
                        self.node.ports[f'coutR'].tx_output(Message([A[0], b, 'failure', bmin], header=self.name))
                    elif self.direction[A[0]]=='D':
                        self.node.ports[f'coutD'].tx_output(Message([A[0], b, 'failure', bmin], header=self.name))
            # if self.ppt:
            #     print(f'{self.name}: Swapping finished.')

class SwapCorrectProgram(QuantumProgram):
    # Correct Protocol当中涉及到的用以执行纠正变换的量子程序。
    default_num_qubits=1

    def set_corrections(self, x_corr, z_corr):
        self.x_corr = x_corr % 2
        self.z_corr = z_corr % 2

    def program(self):
        q1, = self.get_qubit_indices(1)
        if self.x_corr == 1:
            self.apply(instr.INSTR_X, q1)
        if self.z_corr == 1:
            self.apply(instr.INSTR_Z, q1)
        yield self.run()

class HandleCinProtocol(NodeProtocol):
    # 需要修改触发方式！！只有作为端结点才管
    """此协议是纠缠交换（目的端）协议的一部分，用来使目的端节点统一而有序地接受由经典信道发来的经典数据。该协议接收所有收到的
    经典数据包，将它们解包，并将其中的信息以指定的格式交付给本节点的纠缠交换（目的端）协议的另一部分（CorrectProtocol）。
    每个节点需要分配一个本协议对象。"""
    cmessage=namedtuple('ClassicalMessage', ['path_id', 'bit_no', 'result', 'bmin'])
    """将经典数据包中的信息转交给CorrectProtocol的格式。本协议会将经典数据包所含数据转换为该namedtuple对象，然后转交给CorrectProtocol。
    path_id：路径号
    bit_no：比特号
    result：贝尔基下的联合测量结果
    bmin：中间节点进行了纠缠交换的比特数"""
    position=namedtuple('position', ['path_id', 'positions'])
    """将成功完成纯化、纠缠交换和纠正变换的量子比特所在的存储位交付给链路层协议的格式。
    path_id：路径号
    positions：成功完成所有流程的量子比特所在的存储位"""

    def __init__(self, node=None, name=None, prompt=True):
        """由于HandleCinProtocol和CorrectProtocol是同一个协议的两个部分，为简化编程过程，他们的成员变量被设计为了两个协议
        所需的成员的并集。
        cmessages：CorrectProtocol存储各条路径上先于纯化成功的存储位到达的经典信息。此时这些经典信息还无法被处理。
        counter：记录各条路径上到达的经典数据包数量。字典的键为路径号，值为当前该路径上到达的经典数据包数量。字典的值决定了每个
        存储位应在何时执行纠正变换对应的量子程序。
        cP：本节点上的CorrectProtocol协议对象。
        failure_bit：各条路径上纯化成功，但由于中间节点纠缠交换失败而无法使用的量子比特号。字典的键为路径号，值为无法使用的
        量子比特号。
        LLP：本节点上的链路层协议对象。
        path_bit_num：各条路径上纯化、纠缠交换、纠正变换全部成功的量子比特数。字典的键为路径号，值为成功的量子比特数。该字典的值
        会随着纠正变换的进程不断更新，在CorrectProtocol向链路层协议交付存储位之前达到正确值。
        path_length：各条路径的路径长度（即所含节点个数）。字典的键为路径号，值为该路径所含节点个数。
        positions：CorrectProtocol用来存放从DEJMPS协议对象接收到的纯化成功的比特的位的队列。
        ppt：是否需要在运行过程中向屏幕上输出运行状态信息。
        processing_path：CorrectProtocol当前正在处理的所有路径号。
        processing_positions：CorrectProtocol当前正在处理的所有存储位。字典的键为路径号，值为该路径上正在处理的所有存储位。
        process_progress：指示需要进行纠正变换的量子比特是否已完成变换。字典的键为路径号，值为该路径需处理的量子比特的处理状态。
        若中间节点纠缠交换失败，则对应的量子比特的处理状态会直接被标记为已完成。
        processing_queue：CorrectProtocol存储各条路径上后于纯化成功的存储位到达的经典信息。这些信息可以立即被处理。
        program：纠正交换对应的量子程序。
        x_corr：记录各条路径上各个量子比特对应的量子程序x_corr参数。字典的键为路径号，值为包含所有量子比特x_corr参数的列表。
        z_corr：记录各条路径上各个量子比特对应的量子程序z_corr参数。字典的键为路径号，值为包含所有量子比特z_corr参数的列表。"""
        super().__init__(node, name)
        self.cmessages=[]
        self.counter={}
        self.cP=None
        self.failure_bit={}
        self.LLP=None
        self.path_bit_num={}
        self.path_length={}
        self.positions=Queue()
        self.ppt=prompt
        self.processing_path=[]
        self.processing_positions={}
        self.process_progress={}
        self.processing_queue=Queue()
        self.program=SwapCorrectProgram()
        self.x_corr={}
        self.z_corr={}
        self._receive_signal = 'Get Destination Information'
        self._qprogram_complete_signal='Quantum Program Finished'
        self.add_signal(self._qprogram_complete_signal)
        self.add_signal(self._receive_signal)
    
    def clarify_correct_protocol(self,cP):
        # 声明本节点上的CorrectProtocol协议对象。
        if cP==None:
            raise Exception(f'{self.name}: correct protocol input error.')
        self.cP=cP

    def rec_message(self):
        self.send_signal(self._receive_signal)

    def run(self): #handle classical input
        while True:
            # only triggered when the node is receiver
            yield self.await_signal(sender = self, signal_label = self._receive_signal)
            # 准备接收所有方向上到来的经典数据包
            expr1 = self.await_port_input(self.node.ports[f'cinL'])
            expr2 = self.await_port_input(self.node.ports[f'cinU'])
            expr3 = self.await_port_input(self.node.ports[f'cinR'])
            expr4 = self.await_port_input(self.node.ports[f'cinD'])
            expr = yield (expr1 | expr2) | (expr3 | expr4)
            # 根据经典信息发来的方向为message赋值
            if expr.first_term.value:
                message1 = self.node.ports[f"cinL"].rx_input()
                message2 = self.node.ports[f"cinU"].rx_input()
                if message1 is None or len(message1.items)%4 != 0: # 经典信息的长度应为2的整数倍。详见SwapProtocol的消息发送过程。
                    message=message2
                else:
                    message=message1
            elif expr.second_term.value:
                message1 = self.node.ports[f"cinR"].rx_input()
                message2 = self.node.ports[f"cinD"].rx_input()
                if message1 is None or len(message1.items)%4 != 0:
                    message=message2
                else:
                    message=message1
            if self.ppt:
                print(f"{self.name} receives {message}.")
            # 处理经典信息
            if message is None or len(message.items)%4 != 0:
                print(f'Warning: {self.name} detected message error. The error message is {message.items}.')
                continue
            if self.cP==None:
                raise Exception(f'{self.name}: Correct protocol has not been clarified.')
            for i in range(0,len(message.items),4):
                # 将合并在一起的数据包分开，并将其转换为cmessage
                t=self.cmessage(message.items[i], message.items[i+1], message.items[i+2], message.items[i+3])
                self.cP.get_cmessage(t)


class CorrectProtocol(NodeProtocol):
    """此协议是纠缠交换（目的端）协议的一部分。它接收DEJMPS协议对象发来的纯化成功的量子比特存储位，根据协议的另一部分
    （HandleCinProtocol）处理好的经典信息，来对这些存储位上的量子比特进行纠正变换。完成之后，它将成功进行了纯化、纠缠交换
    以及纠正变换的所有量子比特的存储位，联同他们的路径号递交给本节点所在的链路层协议。这样，一次完整的物理层请求就执行完毕了。"""

    cmessage=namedtuple('ClassicalMessage', ['path_id', 'bit_no', 'result', 'bmin'])
    """将经典数据包中的信息转交给CorrectProtocol的格式。本协议会将经典数据包所含数据转换为该namedtuple对象，然后转交给CorrectProtocol。
    path_id：路径号
    bit_no：比特号
    result：贝尔基下的联合测量结果
    bmin：中间节点进行了纠缠交换的比特数"""
    position=namedtuple('position', ['path_id', 'positions'])
    """将成功完成纯化、纠缠交换和纠正变换的量子比特所在的存储位交付给链路层协议的格式。
    path_id：路径号
    positions：成功完成所有流程的量子比特所在的存储位"""

    def __init__(self, node=None, name=None):
        """由于HandleCinProtocol和CorrectProtocol是同一个协议的两个部分，为简化编程过程，他们的成员变量被设计为了两个协议
        所需的成员的并集。
        cmessages：CorrectProtocol存储各条路径上先于纯化成功的存储位到达的经典信息。此时这些经典信息还无法被处理。
        counter：记录各条路径上到达的经典数据包数量。字典的键为路径号，值为当前该路径上到达的经典数据包数量。字典的值决定了每个
        存储位应在何时执行纠正变换对应的量子程序。
        cP：本节点上的CorrectProtocol协议对象。
        failure_bit：各条路径上纯化成功，但由于中间节点纠缠交换失败而无法使用的量子比特号。字典的键为路径号，值为无法使用的
        量子比特号。
        LLP：本节点上的链路层协议对象。
        path_bit_num：各条路径上纯化、纠缠交换、纠正变换全部成功的量子比特数。字典的键为路径号，值为成功的量子比特数。该字典的值
        会随着纠正变换的进程不断更新，在CorrectProtocol向链路层协议交付存储位之前达到正确值。
        path_length：各条路径的路径长度（即所含节点个数）。字典的键为路径号，值为该路径所含节点个数。
        positions：CorrectProtocol用来存放从DEJMPS协议对象接收到的纯化成功的比特的位的队列。
        ppt：是否需要在运行过程中向屏幕上输出运行状态信息。
        processing_path：CorrectProtocol当前正在处理的所有路径号。
        processing_positions：CorrectProtocol当前正在处理的所有存储位。字典的键为路径号，值为该路径上正在处理的所有存储位。
        process_progress：指示需要进行纠正变换的量子比特是否已完成变换。字典的键为路径号，值为该路径需处理的量子比特的处理状态。
        若中间节点纠缠交换失败，则对应的量子比特的处理状态会直接被标记为已完成。True：失败的比特
        processing_queue：CorrectProtocol存储各条路径上后于纯化成功的存储位到达的经典信息。这些信息可以立即被处理。
        program：纠正交换对应的量子程序。
        x_corr：记录各条路径上各个量子比特对应的量子程序x_corr参数。字典的键为路径号，值为包含所有量子比特x_corr参数的列表。
        z_corr：记录各条路径上各个量子比特对应的量子程序z_corr参数。字典的键为路径号，值为包含所有量子比特z_corr参数的列表。"""
        super().__init__(node, name)
        self.cmessages=[]
        self.counter={}
        self.cP=None
        self.failure_bit={}
        self.LLP=None
        self.path_bit_num={}
        self.path_length={}
        self.positions=Queue()
        self.processing_path=[]
        self.processing_positions={}
        self.process_progress={}
        self.processing_queue=Queue()
        self.program=SwapCorrectProgram()
        self.x_corr={}
        self.z_corr={}
        # self.ppt=prompt
        self._get_cmessage_signal='Get Correction Information'
        self._get_position_signal='Get Distillation Result'
        self._qprogram_complete_signal='Quantum Program Finished'
        self.add_signal(self._get_cmessage_signal)
        self.add_signal(self._get_position_signal)
        self.add_signal(self._qprogram_complete_signal)

    # def clarify_LLP(self, LLP):
    #     # 声明本节点的链路层协议对象。
    #     if LLP==None:
    #         raise Exception(f'{self.name}: Link layer protocol input error.')
    #     self.LLP=LLP
    
    def clarify_path_length(self, path_id, path_length):
        # 声明路径长度。
        # if path_length<2:
        #     raise Exception(f'{self.name}: path length input error.')
        self.path_length[path_id]=path_length

    def get_cmessage(self,message):
        # 接收HandleCinProtocol处理完成的经典信息，按情况将它们放入processing_queue或cmessages中。
        # self.processing_path中是本节点“正在处理”的路径列表
        if message.path_id in self.processing_path:
            self.processing_queue.put(message)
            # self.send_signal(self._get_cmessage_signal)
        else:
            self.cmessages.append(message)
        self.send_signal(self._get_cmessage_signal)

    def get_positions(self, position):
        # 接收DEJMPS发来的存储位。
        # print("corr ready")
        # print(f"{self.name} get positions of path {position.path_id} with length {self.path_length[position.path_id]}")
        if len(position)==0:
            raise Exception(f'{self.name} received failed distillation result.')
        # 对只有一跳的路径，不需要纯化，直接丢到节点的positions列表等待结果
        self.node.positions[position.path_id] = position.positions # 这是为了方便最后展示路径保真度结果。
        """ file2 = open(filename2, 'a')
        file2.write(f'Node {self.node.ID} Correct, path_id={position.path_id}, succeeded positions={position.positions}\n')
        file2.close() """
        # 对需要correct的路径，初始化所需的字典和列表
        if self.path_length[position.path_id] > 2:
            self.processing_positions[position.path_id]=position.positions
            self.processing_path.append(position.path_id)
            self.counter[position.path_id]={}
            self.path_bit_num[position.path_id]=len(position.positions)
            if not (position.path_id in self.failure_bit.keys()):
                self.failure_bit[position.path_id]=[]
            if not (position.path_id in self.process_progress.keys()):
                self.process_progress[position.path_id]=[]
            if not (position.path_id in self.x_corr.keys()):
                self.x_corr[position.path_id]={}
            if not (position.path_id in self.z_corr.keys()):
                self.z_corr[position.path_id]={}
            for i in range(len(position.positions)):
                self.process_progress[position.path_id].append(False)        
        # self.positions.put(position)
        # self.send_signal(self._get_position_signal)
    
    # def get_qprogram_result(self, result, output_key):
    #     # 带队列的量子处理器向本协议对象交付量子程序输出结果。该方法是为多线程并行运行而设计。
    #     self.send_signal(self._qprogram_complete_signal)

    def run(self):
        while True:
            # 关注迟到的get_positions_signal!
            yield self.await_signal(self, self._get_cmessage_signal) | self.await_signal(self, self._get_position_signal)
            # find=False
            while True:
                # 在cmessage[]中寻找先于存储位到达的经典信息
                find = False
                for i in range(len(self.cmessages)):
                    if self.cmessages[i].path_id in self.processing_path:
                        p=self.cmessages.pop(i)
                        find=True
                        break
                if (not find) and self.processing_queue.empty():
                    break
                    # if len(self.processing_path)==0: # 所有路径均处理完毕
                    #     break
                    # if self.processing_queue.qsize()==0:
                    #     yield self.await_signal(self, self._get_cmessage_signal)
                if not find:
                    p=self.processing_queue.get()
                # 解包经典信息
                p_id = p.path_id
                b = p.bit_no
                m = p.result
                bm = p.bmin
                # first term: 比最后一跳纯化的数量多，用不到； second term: 其他节点在此处交换失败
                if (not p_id in self.process_progress):
                    continue
                if b > len(self.process_progress[p_id]) - 1 or self.process_progress[p_id][b]:
                    # 若进入该分支，表示该比特对应的中间节点手中的量子比特遭遇了纠缠交换失败，故无需继续被处理。
                    continue
                # 更新path_bit_num列表
                if bm < self.path_bit_num[p_id]:
                    for j in range(bm, self.path_bit_num[p_id]):
                        self.process_progress[p_id][j] = True
                    self.path_bit_num[p_id] = bm
                if m == 'failure':
                    # 若遇到纠缠交换失败的对应比特，直接更新failure_bit和process_progress
                    self.failure_bit[p_id].append(b)
                    self.process_progress[p_id][b]=True
                    # !!! 某路径上最后一个比特是失败的话，不会释放资源呢
                    # continue
                # 根据贝尔基下的联合测量结果，更新x_corr和z_corr字典
                else:
                    if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                        if b in self.x_corr[p_id].keys():
                            self.x_corr[p_id][b] += 1
                        else:
                            self.x_corr[p_id][b] = 1
                    else:
                        if not (b in self.x_corr[p_id].keys()):
                            self.x_corr[p_id][b] = 0
                    if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                        if b in self.z_corr[p_id].keys():
                            self.z_corr[p_id][b] += 1
                        else:
                            self.z_corr[p_id][b] = 1
                    else:
                        if not(b in self.z_corr[p_id].keys()):
                            self.z_corr[p_id][b] = 0
                # 更新counter字典
                    if b in self.counter[p_id].keys():
                        self.counter[p_id][b]+=1
                    else:
                        self.counter[p_id][b] = 1

                    if not p_id in self.path_length.keys() or self.path_length[p_id] == None:
                        print(f'{self.name} warning: processing qubits on path {p_id} without path length clarified.')
                    elif self.counter[p_id][b]==self.path_length[p_id]-2:
                        # 路径上的所有中间节点发送的经典信息已全部到达，对存储器中对应的量子比特做纠正变换
                        if self.x_corr[p_id][b] or self.z_corr[p_id][b]:
                            self.program.set_corrections(self.x_corr[p_id][b], self.z_corr[p_id][b])    
                            while self.node.qmemory.busy:
                                yield self.await_program(self.node.qmemory)
                            yield self.node.qmemory.execute_program(self.program, qubit_mapping=[self.processing_positions[p_id][b]])
                            # print(f"Executed Swap Correct Program in node {self.node.name} for Path {p_id}, bit {b}")
                        # calculate fidelity of this entangled pair, save results to original req
                        node_req = get_req_by_pathid(self.node.running_list, p_id)
                        check_entanglement(requests,node_req,b)
                        # 纠正变换完成后，更新process_progress，复位x_corr、z_corr、counter字典
                        self.process_progress[p_id][b]=True
                        self.x_corr[p_id][b] = 0
                        self.z_corr[p_id][b] = 0
                        self.counter[p_id][b] = 0
                    elif self.counter[p_id][b] > self.path_length[p_id]-2:
                        # 这里路径长度是节点数！
                        # 路径长度声明过晚的情形
                        self.failure_bit[p_id].append(b)
                        self.process_progress[p_id][b]=True
                        print(f'{self.name}: the path length of path {p_id} is clarified too late and too many classical message of qubit {b} is received. You need to measure the fidelity manually, or pop the qubit.')
                if min(self.process_progress[p_id]):
                    # min(True, False) = False
                    # 路径p_id的所有量子比特都已处理完毕，准备发送信息给链路层协议
                    # print(f'{self.name}: path {p_id} with length {self.path_length[p_id]} finished at', ns.sim_time())
                    successful_positions=[] # 所有成功的量子比特所在存储位
                    successful_bitnum=[] # 所有成功的量子比特号
                    for i in range(self.path_bit_num[p_id]):
                        if i in self.failure_bit[p_id]:
                            continue
                        successful_positions.append(self.processing_positions[p_id][i])
                        successful_bitnum.append(i)
                    # free resources along the path (fidelities has been recorded)
                    node_req = get_req_by_pathid(self.node.running_list, p_id)
                    # print("path ",p_id," freed resources")
                    free_path_resource(nodes,self.node,node_req)
                    # for analysis
                    global finished_path
                    finished_path = finished_path + 1

                    # 对相关的列表和字典进行复位
                    del self.failure_bit[p_id]  
                    del self.path_bit_num[p_id]
                    del self.path_length[p_id]
                    self.processing_path.remove(p_id)
                    del self.processing_positions[p_id]
                    del self.process_progress[p_id]
                    # print(f'{self.name}: path {p_id} finished at', ns.sim_time())
                    # for p in self.processing_path:
                    #     print(f"path{p} unfinished")
                    #     print("position status:", self.process_progress[p])
                        # if self.process_progress[p].count(False) == 1:
                        #     print("strange condition, still free resources")
                        #     node_req = get_req_by_pathid(self.node.running_list, p)
                        #      # print("path ",p_id," freed resources")
                        #     free_path_resource(nodes,self.node,node_req)
                        #     del self.failure_bit[p]  
                        #     del self.path_bit_num[p]
                        #     del self.path_length[p]
                        #     self.processing_path.remove(p)
                        #     del self.processing_positions[p]
                        #     del self.process_progress[p]
                    if len(self.processing_path)==0:
                        break
                # else:
                #     print("path ",p_id," process: " ,self.process_progress[p_id])
                #     print("failure bits:",self.failure_bit[p_id])

 


class DEJMPS_Distillation(ServiceProtocol):

    request = namedtuple('DistillationRequest', ['path_id', 'backup_amount', 'positions', 'bit_num'])
    res_ok = namedtuple('DistillationOK', ['path_id', 'backup_amount', 'positions', 'bit_num'])

    def __init__(self, node, name = None, direction = '', role = True):
        super().__init__(node, name)
        self.register_request(self.request, self.request)
        self.register_response(self.res_ok)
        self._new_req_signal = 'New Distillation Request'
        self._prepared_signal = 'Qubits Prepared'
        # self._round_sync_signal = "Round Synchronous"
        self._result_signal = 'Bell Measure Result'
        self._prep_ent_signal='Prepared for Establishing Entangle Pair'
        self._entangle_complete_signal='Entangle Pair Established'
        self.add_signal(self._new_req_signal)
        self.add_signal(self._prepared_signal)
        self.add_signal(self._result_signal)
        self.add_signal(self._prep_ent_signal)
        self.add_signal(self._entangle_complete_signal)
        # self.add_signal(self._round_sync_signal)
        self.direction = direction
        self.neighbour = None
        self.ready_list = {}
        self.key_counter = 0
        self.queue = deque()
        self.result=None # 0：失败；1：部分成功（进行了纯化，但未达到所需的轮数）；2：成功
        self.role = role
        self.make_entangle_pair_token=True
        self.program = DEJMPS_Program(num_qubits=2, parallel=True, role=self.role)
        # self.swap = {}
        self.result_pos=[]
        self.measure_results = []
        self.measure_pos =[]
        self.result_positions =[]
        self.success_num =[]
        self.succ_recording=[]
        #self.neighbourPositions ={}
        self.node_statistics={}
        self.round_count=[]
        self.check_fidelity_waiting=True
        self.check_fidelity_signal='Check Fidelity'
        self.add_signal(self.check_fidelity_signal)

    def clarify_neighbour(self, DDl: ServiceProtocol = None):
        if DDl == None: 
            raise Exception(f'{self.name}: DEJMPS Distillation Protocol Neighbour Input Error!')
        self.neighbour = DDl        

    # 有必要存这么多不？还是只存对应的swap/dst/src节点是哪些，直接判断下啊
    # def clarify_swap(self, SWl: ServiceProtocol = None, path_id: int = 1):
    #     if SWl == None:
    #         raise Exception(f'{self.name}: Swap Protocol Input Error!')
    #     self.swaps[f'{path_id}'] = SWl
    #     self.path_id=path_id
    
    def refresh_key(self):
        if self.key_counter == 99:
            self.key_counter = 0
        else:
            self.key_counter += 1

    def get_result(self, result):
        self.send_signal(self._result_signal, result)


    def handle_request_2(self, request, start_time = None, **kwargs):
        # for i in range(len(request.positions)):
        #     print(self.node.qmemory.peek(request.positions[i]))
        """ with open(filename3, 'a') as file3:
            file3.write(f'Node {self.node.ID} Distillation, role={self.role}, path_id={request.path_id}, receive distillation request\n') """
        if start_time is None:
            start_time=ns.sim_time()
        self.queue.append((start_time, request))
        self.send_signal(self._new_req_signal)
        return kwargs

    def prepared_entangle(self, i):
        self.send_signal(self._prep_ent_signal, i)

    def run(self):
        if self.neighbour is None:
            raise Exception(f'{self.name}: DEJMPS Distillation Protocol Neighbour Has Not been Clarified!')
        while True:
            yield self.await_signal(self, self._new_req_signal)
            while len(self.queue) > 0:
                start_time, request = self.queue.popleft()
                if len(request.positions)<request.bit_num*2:
                    raise Exception(f'The resources is not enough to handle the request. The positions given is {len(request.positions)}, but the request need {request.bit_num*math.pow(2, len(request.backup_amount)-1)} qubits.')
                elif len(request.positions)< request.bit_num*request.backup_amount[0]:
                    print('Warning: the resource maybe enough to handle the request, but cannot match the backup requirement.')
                # if start_time > ns.sim_time():
                #     yield self.await_timer(end_time=start_time)
                yield from self.symmetric(request)

    def symmetric(self, request):

        #biaoji1=0

        self.succ_recording=[]
        start_time = time.time()
        l = request.path_id
        round_num = len(request.backup_amount)-1
        # print("node:", self.node, self.direction, "r_num", round_num)
        positions = request.positions

        ava_positions = request.positions
        self.ava_positions = ava_positions
        self.make_entangle_pair_token = True
        r_aim = len(request.backup_amount) - 1

        self.round_count = [0] * len(request.positions)
        if self.node not in self.node_statistics:
            self.node_statistics[self.node] = {
                "bit_consumed": 0,  # 量子比特消耗
                "pur_round":0
             }
        #ori_f = 0.85

        self.node_statistics[self.node]["bit_consumed"] +=len(positions)
        self.node_statistics[self.node]["pur_round"] +=round_num


        if len(ava_positions) < request.bit_num * (2**r_aim):
            raise Exception(f'{self.name}: Provided bit number is not enough to handle the request.')
        if request.path_id in self.neighbour.ready_list:
            self.send_signal(self._prepared_signal)
        else:
            self.ready_list[request.path_id] = request.positions
            yield self.await_signal(self.neighbour, self._prepared_signal)
        r_cur = []
        for i in range(int(len(request.positions) / 2)):
            r_cur.append(0)
        self.succ_positions = []
        self.fail_positions = list(range(len(request.positions)))
        results = []
        
        if request.path_id in self.neighbour.ready_list:
            self.send_signal(self._prepared_signal)
        else:
            self.ready_list[request.path_id] = request.positions
            yield self.await_signal(self.neighbour, self._prepared_signal) 
        
        counter=0

        #self.succ_recording+=self.fail_positions[0:10]
        
        while (min(r_cur) < r_aim and counter<2):
            if counter>0:
                self.node_statistics[self.node]["bit_consumed"] +=len(self.fail_positions)
                self.node_statistics[self.node]["pur_round"] +=round_num

                if self.neighbour.make_entangle_pair_token:
                    self.make_entangle_pair_token = False
                    yield self.await_signal(self, self._prep_ent_signal)
                    neighbour_actual_positions=self.get_signal_result(self._prep_ent_signal)
                    for n in range(len(neighbour_actual_positions)):
                        if not neighbour_actual_positions[n] in prepared_entangle_positions:
                            prepared_entangle_positions.append(neighbour_actual_positions[n])
                    qubits = ns.qubits.create_qubits(len(self.fail_positions)*2)
                    # self.node_statistics[self.node]["bit_consumed"] += len(self.fail_positions)*2
                    for j in range(len(self.fail_positions)):
                        ns.qubits.assign_qstate([qubits[2 * j], qubits[2 * j + 1]], ns.b00)
                        theta = noise_parameter(ori_f, 0)
                        noise_sim = ops.create_rotation_op(theta)
                        qapi.operate(qubits[2 * j + 1], noise_sim)
                    # print(self.fidelity[direction]-ns.qubits.fidelity(qubits, ks.b00, squared = True))
                    for o in range(len(self.fail_positions)):
                        while self.node.qmemory.busy:
                            yield self.await_program(self.node.qmemory)
                        self.node.qmemory.put(qubits[2 * o], positions=request.positions[self.fail_positions[o]])
                        while self.neighbour.node.qmemory.busy:
                            yield self.await_program(self.neighbour.node.qmemory)
                        self.neighbour.node.qmemory.put(qubits[2 * o + 1], positions=prepared_entangle_positions[o])
                    self.neighbour.send_signal(self.neighbour._entangle_complete_signal)
                else:
                    actual_positions = []
                    for i in range(len(self.fail_positions)):
                        actual_positions.append(request.positions[self.fail_positions[i]])
                    self.neighbour.prepared_entangle(actual_positions) # 直接传位置，不要传索引
                    yield self.await_signal(self, self._entangle_complete_signal)
                    #self.neighbour.make_entangle_pair_token=True
                self.make_entangle_pair_token = True
                
            self.succ_positions=self.fail_positions
            self.fail_positions=[]

            for k in range(r_aim):
                i = 0
                results = []
                res_nei=[]
                if r_aim>0:
                    while i < len(self.succ_positions):
                        while self.node.qmemory.busy:
                            yield self.await_program(self.node.qmemory)
                        yield self.node.qmemory.execute_program(self.program, qubit_mapping=[request.positions[self.succ_positions[i]], request.positions[self.succ_positions[i + 1]]])
                        try:
                            m, = self.program.output["m"]
                        except KeyError:
                            m = -2
                        results.append(m)
                        i += 2

                self.refresh_key()
                # print("node", self.node, self.direction,"key counter",self.key_counter)
                if self.key_counter == self.neighbour.key_counter:
                    # self.send_signal(self._round_sync_signal)
                    self.neighbour.get_result([results, self.succ_positions])
                    res_nei = self.neighbour.measure_results
                    pos_nei = self.neighbour.measure_pos
                else:
                    # yield self.await_signal(self.neighbour, self.neighbour._round_sync_signal)
                    self.measure_results = results
                    self.measure_pos = ava_positions
                    yield self.await_signal(self, self._result_signal)
                    res_nei, pos_nei = self.get_signal_result(self._result_signal)
                if len(results) != len(res_nei):
                    raise Exception(f'{self.name}: Measure results length is different from the neighbour.')
                if len(results) != int(len(self.succ_positions) / 2):
                    print(f'Warning: given {int(len(self.succ_positions) / 2)} positions, but only get {len(results)} results.')
                prepared_entangle_positions = []
                remove_list=[]
                
                for i in range(len(results)):
                    #self.make_entangle_pair_token=True                    
                    if results[i] == res_nei[i] and results[i] != -2:
                        remove_list.append(self.succ_positions[2*i])
                        for m in range(int(2**(r_cur[int(self.succ_positions[2 * i+1]/2)]))): #j的值有问题
                            r_cur[int(self.succ_positions[2 * i+1]/2)-m] += 1
                    else:
                        for m in range(int(self.succ_positions[2 * i]/(2**r_aim))*(2**r_aim), (int(self.succ_positions[2 * i]/(2**r_aim))+1)*(2**r_aim)):
                            if not m in self.fail_positions:
                                self.fail_positions.append(m)
                            if not m in remove_list:
                                remove_list.append(m)                    
                    for j in range(len(self.fail_positions)):
                        r_cur[int(self.fail_positions[j]/2)] = 0

                for i in remove_list:
                    if i in self.succ_positions:
                        self.succ_positions.remove(i) 
                remove_list=[]

            while self.node.qmemory.busy:
                yield self.await_program(self.node.qmemory)
            for i in self.fail_positions:
                self.node.qmemory.pop(request.positions[i])
            for i in self.succ_positions:
                for j in range(2**r_aim-1):
                    self.node.qmemory.pop(request.positions[i-j-1])
            
            self.succ_recording+=self.succ_positions
          
            counter+=1
        

        
        req = get_req_by_pathid(self.node.running_list, l)
        #print('self.ava_positions',self.ava_positions)
        #把ava_positions中元素为奇数位取出
        # self.succ_positions = [pos for i, pos in enumerate(self.ava_positions) if i % 2 == 1]
        self.swap_positions=[]

        """ file1=open(filename1, 'a')
        file1.write(f'node: {self.node}, path_id: {l}, direction:{self.direction}, role:{self.role}, succ_recording: {self.succ_recording}, succ_positions: {self.succ_positions}\n')        
        file1.close() """

        for i in self.succ_recording:
            self.swap_positions.append(request.positions[i])


        if (req.prev_qubit_num == 0):
            self.node.src_protocol.get_positions(self.node.src_protocol.position(l, self.swap_positions))
        elif (req.next_qubit_num == 0):
            self.node.dst_protocol.get_positions(self.node.dst_protocol.position(l, self.swap_positions))
        else:
            self.node.swap_protocol.get_positions(self.node.swap_protocol.position(l, self.swap_positions))
        if l in self.ready_list:
            del self.ready_list[l]

        for i in range(len(request_path_id)):
            if l in request_path_id[i]:
                distillation_R[i].append(sum(data["bit_consumed"] for data in self.node_statistics.values()))
                distillation_T[i].append(sum(data["pur_round"] for data in self.node_statistics.values()))

        self.node_statistics[self.node]["pur_round"] =max(self.round_count)
        end_time = time.time()  # 记录纯化结束时间
        elapsed_time = end_time - start_time  # 计算纯化所用时间
        # Purification_time.append(elapsed_time)

# add time factor
    def distillation(self, request):
        l = request.path_id
        round_num = len(request.backup_amount)-1
        # print("node:", self.node, self.direction, "r_num", round_num)
        positions = request.positions
        if l in last_request_path_id:
            print(f'node: {self.node}, path_id: {l}, cost: {len(positions)}, r_num: {round_num}')
        ava_positions = positions
        # for i in range(len(positions)):
        #     write_to_file(str(self.node.qmemory.peek(request.positions[i])), filename2)
        o_fidelity = 0
        if request.path_id in self.neighbour.ready_list:
            # for test only
            qa, = self.node.qmemory.peek(positions[0])
            qb, = self.neighbour.node.qmemory.peek(self.neighbour.ready_list[request.path_id][0])
            o_fidelity = ns.qubits.fidelity([qa,qb],ns.b00,squared = True)
            # print("original  fidelity:", fidelity)
            # test end
            self.send_signal(self._prepared_signal)
        else:
            self.ready_list[request.path_id] = request.positions
            yield self.await_signal(self.neighbour, self._prepared_signal)
        #print("node:", self.node, self.direction, request.positions)
        # nec_resource=math.pow(2, round_num)
        # print(self.node.ID, "purify start:",ns.sim_time(ns.MILLISECOND),"ms")
        t1 = ns.sim_time(ns.MILLISECOND)
        for r in range(round_num):
            round_m = []
            self.measure_results.clear()
            self.measure_pos.clear()
            yield self.await_timer(purify_round_time)
            # print(self.node.ID, "purify round:",r+1, " ",ns.sim_time(ns.MILLISECOND),"ms")          
            #print("node:", self.node,"direction:",self.direction, f'round {r+1}:')
            for i in range(int(len(ava_positions)/2)): # 单轮纯化，对positions[2*i]和positions[2*i+1]进行。若成功，则positions[2*i+1]是输出比特
                # highlight!!!!
                while self.node.qmemory.busy:
                    yield self.await_program(self.node.qmemory)   
                yield self.node.qmemory.execute_program(self.program, qubit_mapping=[ava_positions[2*i], ava_positions[2*i+1]]) # 此处需要setup_network()函数同步改动以实现统一
                try:
                    m, = self.program.output["m"]
                except KeyError:
                    m = -1
                round_m.append(m)
            # synchronize round process
            self.refresh_key()
            #print("node", self.node, self.direction,"key counter",self.key_counter)
            if self.key_counter == self.neighbour.key_counter:
                # self.send_signal(self._round_sync_signal)
                self.neighbour.get_result([round_m, ava_positions])
                round_n = self.neighbour.measure_results
                pos = self.neighbour.measure_pos
            else:
                # yield self.await_signal(self.neighbour, self.neighbour._round_sync_signal)
                self.measure_results = round_m
                self.measure_pos = ava_positions
                yield self.await_signal(self, self._result_signal)
                round_n, pos = self.get_signal_result(self._result_signal)

            #print("node ", self.node, "round ", r + 1)
            if len(round_m) != len(round_n):
                raise Exception(f'{self.name} distillation error: the amount of qubit being operated is different from the neighbour.')
            count = 0
            for i in range(len(round_m)):
                if round_m[i] == round_n[i]:
                    count += 1
                    self.result_pos.append(ava_positions[2*i+1])
                    qa, = self.node.qmemory.peek(ava_positions[2*i+1]) #输出纯化后保真度
                    qb, = self.neighbour.node.qmemory.peek(pos[2*i+1]) #输出纯化后保真度
                    fidelity = ns.qubits.fidelity([qa,qb],ns.b00,squared = True) #输出纯化后保真度
                    all_fidelities.append(PathFidelity(request.path_id, fidelity))
                    # distillation_fidelity.append(fidelity)
                    # print( "pair:", count,  fidelity) #输出纯化后保真度
                    # for test only
                    # if self.role & r > 0:
                    #     qa, = self.node.qmemory.peek(ava_positions[2*i+1])
                    #     qb, = self.neighbour.node.qmemory.peek(pos[2*i+1])
                    #     fidelity = ns.qubits.fidelity([qa,qb],ns.b00,squared = True)
                    #     print( "pair:", count,  fidelity)
                    # test end
            # print("expected:", request.backup_amount[r + 1] * request.bit_num, "count:", count, "success rate:", count/len(round_m))
#                    self.node.subcomponents[f'QPU_{self.node.ID}_{l}'].pop(ava_positions[2*i+1])
            if len(self.result_pos)>1 or r+1 == round_num:
                ava_positions=self.result_pos
                # print(f'The successfully distillated qubits in round {r+1} are {ava_positions}.')
                if (r+1) == round_num:
                    # if len(self.result_pos) < request.bit_num:
                    #     print(f'The successfully distillated qubit amount cannot match the request. The qubit amount is {len(self.result_pos)}.')
                    self.result=2
                    ## add special conditions for end nodes of a path!! no need for swapping
                    # give results to swapping protocols: including path_id & positions
                    req = get_req_by_pathid(self.node.running_list, l)
                    # print(f'{self.node}: distillation succeed in positions:',self.result_pos)
                    if(req.prev_qubit_num == 0):
                        self.node.src_protocol.get_positions(self.node.src_protocol.position(l,self.result_pos))
                    elif(req.next_qubit_num == 0):
                        self.node.dst_protocol.get_positions(self.node.dst_protocol.position(l,self.result_pos))
                    else:        
                        self.node.swap_protocol.get_positions(self.node.swap_protocol.position(l,self.result_pos))
                    if l in self.ready_list:
                        del self.ready_list[l]
                    # print(f'{self.name}: distillation succeed.')
                # else:
                    # if len(self.result_pos)<request.backup_amount[r + 1] * request.bit_num:
                        # print(f'Warning: The successfully distillated qubits in round {r+1} cannot match the backup requirement. The successful qubit amount is {len(self.result_pos)}, the requirement is {request.backup_amount[r]*request.bit_num}.')
                # nec_resource=nec_resource/2
            elif len(self.result_pos)==1:
                self.result = 1
                print(f'{self.name}: distillation round number cannot match the requirement. The current round number is {r+1}.')
                if l in self.ready_list:
                    del self.ready_list[l]
                break
            else:   
                self.result = 0
                # print(f'{self.name}: distillation failed in round {r+1}.')
                if l in self.ready_list:
                    del self.ready_list[l]
                break
            i=0
            if r!=round_num-1:
                self.result_pos=[]  
        t2 = ns.sim_time(ns.MILLISECOND)    
        distillation_timecost.append(t2 - t1)   
        ava_positions=[]
        self.result_pos=[]

class DEJMPS_Program(QuantumProgram):
    default_num_qubits = 2

    def __init__(self, num_qubits=None, parallel=True, qubit_mapping=None, role=True):
        super().__init__(num_qubits, parallel, qubit_mapping)
        self.role=role

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        if self.role:
            self.apply(GateA, [q1], inplace=True)
            self.apply(GateA, [q2], inplace=True)
        else:
            self.apply(GateB, [q1], inplace=True)
            self.apply(GateB, [q2], inplace=True)
        self.apply(GateC, [q2, q1], inplace=True)
        self.apply(INSTR_MEASURE, q1, output_key="m", inplace=False)
        yield self.run(parallel = False)

class LinkProtocol(ServiceProtocol):
    #Define the requests and responses as class attributes
    req_entangle = namedtuple('NewLinkEntanglement', ['path_id', 'start_time',  'positions', 'the_other_node', 'node_req_tag'])
    res_ok = namedtuple('LinkOk', ['path_id'])

    def __init__(self, node, name = None):
        super().__init__(node, name)
        self.res = "hello"
        self.add_signal(self.res)
        self.register_request(self.req_entangle,self.entangle)
        self.register_response(self.res_ok)
        self.queue = deque()
        self._new_req_signal = "New request in queue"
        self.add_signal(self._new_req_signal)
        self._create_id = 0
        self.ready_list = {'L': {}, 'R': {}, 'U': {}, 'D': {}}
        self.fidelity = {'L': None, 'R': None, 'U': None, 'D': None}
        self._other_service = {'L': None, 'R': None, 'U': None, 'D': None}
        self._purify_service = {'L': None, 'R': None, 'U': None, 'D': None}

    def add_other_service(self, direction, service):
        self._other_service[direction] = service
    
    def update_fidelity(self, direction, value):
        self.fidelity[direction] = value

    def add_purify_service(self, direction, service):
        self._purify_service[direction] = service

    def handle_request(self, request, id):
        self.queue.append(request)
#        print(self.node.ID, "handle_request", "used positions:", self.node.qmemory.used_positions)
        self.send_signal(self._new_req_signal)
    
    def receive_signal_result(self, result):
        self.send_signal(self.res, result)
    
    
    def run(self):
        while True:
            yield self.await_signal(self, self._new_req_signal)
#            print(self.node.ID, self.queue)
            while len(self.queue) > 0:
                request = self.queue.popleft()
                # time synchronization with the other node
                # local_start_time = request.start_time
                distance = self.node.ID - request.the_other_node
                if distance == -1:
                    direction = 'R'
                    op_direction = 'L'
                elif distance == 1:
                    direction = 'L'
                    op_direction ='R'
                elif distance < -1:
                    direction = 'D'
                    op_direction ='U'
                elif distance > 1:
                    direction = 'U'
                    op_direction ='D'
                if request.path_id in self._other_service[direction].ready_list[op_direction]:
#                    if len(request.positions) != len(self._other_service[direction].ready_list[op_direction][request.path_id]):
#                        raise Exception('Link Requests Size Unpaired Error!')
                    yield from self.entangle(request, direction, 's')
                else:
                    self.ready_list[direction][request.path_id] = request.positions
                    #print(self.node, self.ready_list, self._other_service, self._other_service[direction].ready_list)
                    yield from self.entangle(request, direction, 'r')
                

    def entangle(self, req, direction, role):
        # print("enter entangle", self.node.ID, role)
        o_dict = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U'}
        op_direction = o_dict[direction]
        if role == 's':
            # set a duration time t_e, here set 1000000 temporarily
            time = random.gauss(link_entangle_time, link_deviation_time)
            yield self.await_timer(time)
            # print("s",self.node.ID, req.the_other_node,"linklayer prepared:",ns.sim_time(ns.MILLISECOND),"ms")
#            print("start entangle", self.node.ID, req.the_other_node)
            if len(req.positions) != len(self._other_service[direction].ready_list[op_direction][req.path_id]):
                raise Exception('Link Requests Size Unpaired Error!')
            for i in range(len(req.positions)):
                # defaultly create state |Phi+>, how to control fidelity?!
                qubits = ns.qubits.create_qubits(2)
                ns.qubits.assign_qstate(qubits, ns.b00)
                theta = noise_parameter(self.fidelity[direction], 0)
                noise_sim = ops.create_rotation_op(theta)
                qapi.operate(qubits[1], noise_sim)
                # print(self.fidelity[direction]-ns.qubits.fidelity(qubits, ks.b00, squared = True))
                while self.node.qmemory.busy:
                    yield self.await_program(self.node.qmemory)
                self.node.qmemory.put(qubits[0], positions = req.positions[i])
                while self._other_service[direction].node.qmemory.busy:
                    yield self.await_program(self._other_service[direction].node.qmemory)
                self._other_service[direction].node.qmemory.put(qubits[1], \
                    positions = self._other_service[direction].ready_list[op_direction][req.path_id][i])
            #print("finish entangle", self.node.ID, req.the_other_node)
            #request = namedtuple('DistillationRequest', ['path_id', 'backup_amount', 'positions', 'bit_num'])
            node_req = get_req_by_pathid(self.node.running_list, req.path_id)
            #怎么会有一些请求已经被释放了呢？
            if node_req is None:
                return
            if req.node_req_tag == "p":
                if len(node_req.prev_backup_list) > 1:
                    pur_req = self._purify_service[direction].request(req.path_id, node_req.prev_backup_list, req.positions, int(node_req.prev_qubit_num/node_req.prev_backup_list[0]))
                    # pur_req_new = self._purify_service[direction].request(req.path_id, node_req.prev_backup_list, req.positions, int(node_req.prev_qubit_num/node_req.prev_backup_list[0]))
                    self._purify_service[direction].handle_request_2(pur_req)  
                else:
                    # give positions info to swap/src/dst protocol
                    if node_req.next_qubit_num:
                        # has next hop, deliver to swap protocol
                        self.node.swap_protocol.get_positions(self.node.swap_protocol.position(req.path_id, req.positions))
                    else:
                        # be dst node!
                        self.node.dst_protocol.get_positions(self.node.dst_protocol.position(req.path_id, req.positions))
                    #print("no purification needed for", direction," link of node ", self.node)
            else:
                if len(node_req.next_backup_list) > 1:
                    pur_req = self._purify_service[direction].request(req.path_id, node_req.next_backup_list, req.positions, int(node_req.next_qubit_num/node_req.next_backup_list[0]))
                    self._purify_service[direction].handle_request_2(pur_req)                
                else:
                    # give positions info to swap/src/dst protocol
                    if node_req.prev_qubit_num:
                        # has prev hop, deliver to swap protocol
                        self.node.swap_protocol.get_positions(self.node.swap_protocol.position(req.path_id, req.positions))
                    else:
                        # be src node!
                        self.node.src_protocol.get_positions(self.node.src_protocol.position(req.path_id, req.positions))
                    # print("no purification needed for", direction," link of node ", self.node)
            
            # print("node", self.node.ID,"put pur req with", direction, "neighbour")
            self._other_service[direction].receive_signal_result([req.path_id, self._other_service[direction].node.ID])

        if role == 'r':
            # how to control its waiting time specified by path_id
            yield self._other_service[direction].await_signal(sender = self, signal_label = self.res)
            #print(self.node.qmemory.used_positions)
            #qa, = nodes[0].qmemory.peek([0])
            #qb, = nodes[1].qmemory.peek([1])
            #fidelity = ns.qubits.fidelity([qa,qb], ks.b00, squared = True)
            #print(fidelity)
 #           self.ready_list[direction].pop(req.path_id, None)
            # print("r",self.node.ID, req.the_other_node,"linklayer prepared:",ns.sim_time(ns.MILLISECOND),"ms")
            node_req = get_req_by_pathid(self.node.running_list, req.path_id)
            if node_req is None:
                return
            if req.node_req_tag == "p":
                if len(node_req.prev_backup_list) > 1:
                    pur_req = self._purify_service[direction].request(req.path_id, node_req.prev_backup_list, req.positions, int(node_req.prev_qubit_num/node_req.prev_backup_list[0]))
                    self._purify_service[direction].handle_request_2(pur_req)  
                else:
                    # give positions info to swap/src/dst protocol
                    if node_req.next_qubit_num:
                        # has next hop, deliver to swap protocol
                        self.node.swap_protocol.get_positions(self.node.swap_protocol.position(req.path_id, req.positions))
                    else:
                        # be dst node!
                        self.node.dst_protocol.get_positions(self.node.dst_protocol.position(req.path_id, req.positions))

            else:
                if len(node_req.next_backup_list) > 1:
                    pur_req = self._purify_service[direction].request(req.path_id, node_req.next_backup_list, req.positions, int(node_req.next_qubit_num/node_req.next_backup_list[0]))
                    self._purify_service[direction].handle_request_2(pur_req)
                else:
                    # give positions info to swap/src/dst protocol
                    if node_req.prev_qubit_num:
                        # has prev hop, deliver to swap protocol
                        self.node.swap_protocol.get_positions(self.node.swap_protocol.position(req.path_id, req.positions))
                    else:
                        # be src node!
                        self.node.src_protocol.get_positions(self.node.src_protocol.position(req.path_id, req.positions))
            # print("node", self.node.ID,"put pur req with", direction, "neighbour")            
            res = self.res_ok(req.path_id)
            self.send_response(res)
        

class ResourceAllocationProtocol(NodeProtocol):
    def __init__(self, node=None, name=None):
        super().__init__(node, name)
        self._trigger_signal = "try allocation"
        self.add_signal(self._trigger_signal)

    def add_linklayer_protocol(self,protocol):
        self.link_layer_protocol = protocol

    def trigger(self):
        self.send_signal(self._trigger_signal)

    def run(self):
        while True:
            #to avoid occupying cpu all the time, or waiting for a period(idle situation)
            #when resources is enough for more than one node requests, how to deal with?
            #can be fixed to activate the second choice only when there is no position being used
            # 12/21 fix:只在被使用的qubit位置有释放或者有新请求到达时触发！但这个used_positions好像不太能用,反转完可能获取不到了
            # 有位置释放等价于等待节点释放资源时候发个信号通知
            # yield self.await_mempos_in_use_toggle(self.node.qmemory, self.node.qmemory.used_positions) | self.await_signal(self,self._new_req_signal)
            yield self.await_signal(self,self._trigger_signal)
            # print(self.node.ID, "trigger")
            if len(self.node.waiting_queue) == 0:
                continue
            available_positions = self.node.qmemory.unused_positions
            for req in self.node.running_list:
                available_positions = [x for x in available_positions if x not in req.mem_positions]
            if bool(self.node.waiting_queue) and len(available_positions) >= \
                self.node.waiting_queue[0].prev_qubit_num + self.node.waiting_queue[0].next_qubit_num:
            #triggered when there's enough space for the next request
            #    print("node", self.node.ID, "waiting_list_length", len(self.node.waiting_queue), "running_list_length", len(self.node.running_list))
                current_request = self.node.waiting_queue.popleft()
                current_request.mem_positions = available_positions[:current_request.prev_qubit_num + current_request.next_qubit_num]
                current_request.start_time = ns.sim_time()
                self.node.running_list.append(current_request)
                # print("current_request.mem_positions", current_request.mem_positions)
            # split a node request into two parts 
                # print(self.node.ID, "allocated")
                # req_entangle = namedtuple('NewLinkEntanglement', ['path_id', 'start_time',  'positions', 'the_other_node', 'node_req_tag'])
                if current_request.prev_qubit_num:
                    req_prev = self.link_layer_protocol.req_entangle(current_request.path_id, current_request.start_time,  \
                        current_request.mem_positions[:current_request.prev_qubit_num], current_request.prev_node, "p") 
                    self.link_layer_protocol.put(req_prev)

                if current_request.next_qubit_num:
                    req_next = self.link_layer_protocol.req_entangle(current_request.path_id, current_request.start_time,  \
                        current_request.mem_positions[-current_request.next_qubit_num:],current_request.next_node, "n")
                    self.link_layer_protocol.put(req_next)

                # 尝试继续为waiting_list中其他请求分配资源：
                if len(self.node.waiting_queue) > 0:
                    self.trigger()

#setup network
def setup_network(scale, pos_num, distance, configured):
    network = Network("Test")
    nodes=[]
    link_protocols = []
    allocation_protocols = []
    # global configure_file
    for i in range(scale):
        for j in range(scale):
            nodes.append(QNode(f'Node_{i * scale + j + 1}', i * scale + j + 1, None, ['cinL','cinR','cinU','cinD','coutL','coutR','coutU','coutD']))
            qmemory = QuantumProcessor(f'QPU_{i * scale + j + 1}', num_positions = pos_num, fallback_to_nonphysical=True, phys_instructions=Ins)
            nodes[i * scale + j].add_subcomponent(qmemory)
    # for node in nodes:
    #     print(node.ID)
    network.add_nodes(nodes)
    edges = []
    # initialize edge parameters
    # edges:list of (node1.id,node2.id,raw_fidelity,classical_delay)
    # fidelity: 0.81~0.99，mu = 0.9
    # unit: delay (ns)
    if configured:
        fidelities = []
        with open(configure_file, 'r') as file:
            for line in file:
                fidelities.append(float(line.strip()))
    count = 0
    for i in range(scale):
        for j in range(scale - 1):
            conn = ClassicalConnection(name = f"node[{i},{j}]-node[{i},{j+1}]", length = distance)
            network.add_connection(nodes[i * scale + j], nodes[i * scale + j + 1], conn, port_name_node1 = 'coutR', port_name_node2= 'cinL')
            conn = ClassicalConnection(name = f"node[{i},{j + 1}]-node[{i},{j}]", length = distance)
            network.add_connection(nodes[i * scale + j + 1], nodes[i * scale + j], conn, port_name_node1 = 'coutL', port_name_node2= 'cinR')
            conn = ClassicalConnection(name = f"node[{j},{i}]-node[{j+1},{i}]", length = distance)
            network.add_connection(nodes[i + scale * j], nodes[i + scale * j + scale], conn, port_name_node1 = 'coutD', port_name_node2= 'cinU')
            conn = ClassicalConnection(name = f"node[{j + 1},{i}]-node[{j},{i}]", length = distance)
            network.add_connection(nodes[i + scale * j + scale], nodes[i + scale * j], conn, port_name_node1 = 'coutU', port_name_node2= 'cinD')
            # fidelity on horizontal edges
            if configured:
                edge = [nodes[scale * i + j].ID, nodes[scale * i + j + 1].ID, fidelities[count], 100000]
                count = count + 1
                edges.append(edge)
                edge = [nodes[i + scale * j].ID, nodes[i + scale * j + scale].ID, fidelities[count], 100000]
                count = count + 1
                edges.append(edge)
            else:    
                fidelity = random.gauss(0.80, 0.03)
                # print(fidelity)
                if fidelity > 1:
                    fidelity = fidelity - 0.1
                write_to_file(f'{fidelity}',configure_file)
                edge = [nodes[scale * i + j].ID, nodes[scale * i + j + 1].ID, fidelity, 100000]
                edges.append(edge)
                # fidelity on vertical edges
                fidelity = random.gauss(0.80, 0.03)
                if fidelity > 1:
                    fidelity = fidelity - 0.1
                write_to_file(f'{fidelity}',configure_file)
                edge = [nodes[i + scale * j].ID, nodes[i + scale * j + scale].ID, fidelity, 100000]
                edges.append(edge)
    return network, edges

#setup protocols
def setup_protocols(network:Network, scale: int, pos_num: int, edges: list):
    nodes = sorted(network.nodes.values(), key=lambda x: x.ID)
    main_protocol = LocalProtocol(nodes)
    link_protocols = []
    allocation_protocols = []
    L_purify_protocols = []
    R_purify_protocols = []
    U_purify_protocols = []
    D_purify_protocols = []
    swap_protocols = []
    fwd_protocols = []
    src_protocols = []
    dst_protocols = []
    # msg_collect_protocols = []
    # frequency = 1e-7
    gen_req_protocol = RandomRequest(network, edges, frequency, pos_num)
    # main_protocol.add_subprotocol(gen_req_protocol)
    for i in range(scale * scale):
        L_purify_protocols.append(DEJMPS_Distillation(node = nodes[i], direction = 'L', role = False))
        R_purify_protocols.append(DEJMPS_Distillation(node = nodes[i], direction = 'R'))
        U_purify_protocols.append(DEJMPS_Distillation(node = nodes[i], direction = 'U', role = False))
        D_purify_protocols.append(DEJMPS_Distillation(node = nodes[i], direction = 'D'))
        link_protocols.append(LinkProtocol(nodes[i]))
        swap_protocols.append(SwapProtocol(nodes[i], f'Swap_{i+1}', fail_rate = 0.01))
        src_protocols.append(SRInitProtocol(nodes[i],f'Src_{i+1}'))
        dst_protocols.append(CorrectProtocol(nodes[i],f'Correct_{i+1}'))
        fwd_protocols.append(FCMProtocol(nodes[i], f'Forward_{i+1}'))
        # msg_collect_protocols.append(HandleCinProtocol(nodes[i],f'CollectMessage_{i+1}'))
        # msg_collect_protocols[i].clarify_correct_protocol(dst_protocols[i])
        nodes[i].fwd_protocol = fwd_protocols[i]
        nodes[i].swap_protocol = swap_protocols[i]
        nodes[i].src_protocol = src_protocols[i]
        nodes[i].dst_protocol = dst_protocols[i]
        # nodes[i].rec_protocol = msg_collect_protocols[i]
        link_protocols[i].add_purify_service('L', L_purify_protocols[i])
        link_protocols[i].add_purify_service('R', R_purify_protocols[i])
        link_protocols[i].add_purify_service('U', U_purify_protocols[i])
        link_protocols[i].add_purify_service('D', D_purify_protocols[i])
        allocation_protocols.append(ResourceAllocationProtocol(nodes[i]))
        nodes[i].allocation_service = allocation_protocols[i]
        allocation_protocols[i].add_linklayer_protocol(link_protocols[i])
        # Error: Sub-protocols should be able to signal parent protocol
        # main_protocol.add_subprotocol(link_protocols[i])
        # main_protocol.add_subprotocol(allocation_protocols[i])
    for i in range(scale):
        for j in range(scale - 1):
            link_protocols[scale * i + j].add_other_service('R', link_protocols[scale * i + j + 1])
            link_protocols[scale * i + j + 1].add_other_service('L', link_protocols[scale * i + j])
            link_protocols[i + scale * j].add_other_service('D', link_protocols[i + scale * j + scale])
            link_protocols[i + scale * j + scale].add_other_service('U', link_protocols[i + scale * j])
            R_purify_protocols[scale * i + j].clarify_neighbour(L_purify_protocols[scale * i + j + 1])
            L_purify_protocols[scale * i + j + 1].clarify_neighbour(R_purify_protocols[scale * i + j])
            D_purify_protocols[i + scale * j].clarify_neighbour(U_purify_protocols[i + scale * j + scale])
            U_purify_protocols[i + scale * j + scale].clarify_neighbour(D_purify_protocols[i + scale * j])            
    for edge in edges:
        distance = edge[1] - edge[0]
        if distance == 1:
            link_protocols[edge[0] - 1].update_fidelity('R', edge[2])
            link_protocols[edge[1] - 1].update_fidelity('L', edge[2])
        else:
            link_protocols[edge[0] - 1].update_fidelity('D', edge[2])
            link_protocols[edge[1] - 1].update_fidelity('U', edge[2])
    # only for test
    # for i in range(scale * scale):
    #     print("node:", i+1, "purify DN:", D_purify_protocols[i].neighbour)

    # adjusted: 先初始化各节点协议再生成随机请求
    # gen_req_protocol.start()
    for i in range(scale * scale):
        link_protocols[i].start()
        allocation_protocols[i].start()
        swap_protocols[i].start()
        fwd_protocols[i].start()
        src_protocols[i].start()
        dst_protocols[i].start()
        # msg_collect_protocols[i].start()
        if i % scale:
            L_purify_protocols[i].start()
        if (i + 1) % scale:
            R_purify_protocols[i].start()
        if i >= scale:
            U_purify_protocols[i].start()
        if i < scale * (scale - 1):
            D_purify_protocols[i].start()
    gen_req_protocol.start()
#    return main_protocol

# # side len(node num) of square network
# scale = 10
# # available position num of each node      
# pos_num = 100
# # physical distance between nodes
# distance = 2
network, edges= setup_network(scale, pos_num, distance,True)
nodes = sorted(network.nodes.values(), key=lambda x: x.ID)
#nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]
#we only test alg in this file, so manually give each node a T attribute
# for node in nodes:
#     # print(node)
#     factor = random.uniform(3, 7)
#     node.T = factor * 1000000
# #set k which  makes the magnitudes in the equation match
# # k1*10^6
# k = [1e-4, 10, 1e-6]
# requests = []
# frequency = 1/mean_arrival_time

# frequency = 1e-7
# gen_req_protocol = RandomRequest(network, edges, frequency, pos_num)
# gen_req_protocol.start()
#main_protocol = setup_protocols(network, scale, pos_num)
req_sucnum=0
for i in range(req_num):
    distillation_R.append([]) # 纯化资源
    distillation_T.append([])
    request_path_id.append([])



time1 = time.time()
setup_protocols(network, scale, pos_num, edges)
time2 = time.time()
print("finish setup at:", time2 - time1)
ns.sim_run(50000000000000000)
# filename = 'base_output(k1=100).txt'
# failfile = 'fail_allocation.txt'
# configure_file = 'fidelity_configure'

        
for req in requests:
    print('*******************************')
    src_i = (req.src.ID - 1) // scale
    src_j = (req.src.ID - 1) % scale
    dst_i = (req.dst.ID - 1) // scale
    dst_j = (req.dst.ID - 1) % scale
    dis = abs(dst_j - src_j) + abs(dst_i - src_i)
    dur = (req.finish_time - req.start_time) / 1000000
    trans_dur = (req.finish_time - req.transtart_time) / 1000000
    print(f'id:{req.id},src:({src_i},{src_j}),dst:({dst_i},{dst_j})')
    print(f'distance: {dis}')
    print(f'success status:{req.success}')
    print(f'distillation cost:{sum(req.distillation_cost)}')
    if req.success == 1:
        avg_fidelity = sum(req.achieved_fidelity)/ len(req.achieved_fidelity)
        print(f'target fidelity:{req.fidelity}, average_fidelity:{avg_fidelity}')
        print(f'target bandwidth:{req.bandwidth}, acquired_bandwidth:{req.achieved_bandwidth}')
        print(f'started at{req.start_time/1000000}ms, finished at{req.finish_time/1000000}ms, duration:{dur}')
        req_sucnum+=1
        # write_to_file(f'{req.id},{dis},{req.success},{req.fidelity},{avg_fidelity},{req.bandwidth},{req.achieved_bandwidth},{dur},{trans_dur}',filename)
    else:
        if req.success == -1:
            print('resource num in per node cannot satisfy!')
            # write_to_file(f'{req.id},{dis},{req.success},0,0,{req.bandwidth},0,0',failfile)
        else:
            print(f'started at{req.start_time/1000000}ms,target bandwidth:{req.bandwidth}, acquired_bandwidth:{req.achieved_bandwidth}')
            # write_to_file(f'{req.id},{dis},{req.success},{req.fidelity},{avg_fidelity},{req.bandwidth},{req.achieved_bandwidth},{dur}',filename)
            
path_fidelities = defaultdict(list)
for fidelity_data in all_fidelities:
    path_fidelities[fidelity_data.path_id].append(fidelity_data.fidelity)

        # 计算每个路径的平均保真度
average_fidelities = {}
for path_id, fidelities in path_fidelities.items():
    average_fidelities[path_id] = sum(fidelities) / len(fidelities)

        # 打印平均保真度
# for path_id, avg_fidelity in average_fidelities.items():
#     print(f"Path {path_id} average fidelity: {avg_fidelity}")            
# print(f'average waiting time {sum(waiting_time)/len(waiting_time)}')
for i in range(len(routing_T)):
    if i in routing_req:
        routing_T_success.append(routing_T[i])
        routing_C_success.append(routing_C[i])
        routing_F_success.append(routing_F[i])
        routing_D_success.append(routing_D[i])
        distillation_R_success.append(distillation_R[i])
        distillation_T_success.append(distillation_T[i])

for i in range(len(distillation_R_success)):
    if len(distillation_R_success[i])==0:
        distillation_R_success[i].append(0)
for i in range(len(distillation_T_success)):
    if len(distillation_T_success[i])==0:
        distillation_T_success[i].append(0)
# print(routing_T_success)
for i in range(len(routing_req)):
    routing_T_average.append(sum(routing_T_success[i]) / len(routing_T_success[i]))
    print(f'request {routing_req[i]} average routing T:{routing_T_average[i]}')
    routing_C_average.append(sum(routing_C_success[i]) / len(routing_C_success[i]))
    print(f'request {routing_req[i]} average routing C:{routing_C_average[i]}')
    routing_F_average.append(sum(routing_F_success[i]) / len(routing_F_success[i]))
    print(f'request {routing_req[i]} average routing F:{routing_F_average[i]}')
    routing_D_average.append(sum(routing_D_success[i]) / len(routing_D_success[i]))
    print(f'request {routing_req[i]} average routing D:{routing_D_average[i]}')
    distillation_R_average.append(sum(distillation_R_success[i]) / len(distillation_R_success[i]))
    print(f'request {routing_req[i]} average distillation R:{distillation_R_average[i]}')
    distillation_T_average.append(sum(distillation_T_success[i]) / len(distillation_T_success[i]))
    print(f'request {routing_req[i]} average distillation T:{distillation_T_average[i]}')
print(f'routing_T min:{min(routing_T_average)},max:{max(routing_T_average)},average:{sum(routing_T_average)/len(routing_T_average)}')
print(f'routing_C min:{min(routing_C_average)},max:{max(routing_C_average)},average:{sum(routing_C_average)/len(routing_C_average)}')
print(f'routing_F min:{min(routing_F_average)},max:{max(routing_F_average)},average:{sum(routing_F_average)/len(routing_F_average)}')
print(f'routing_D min:{min(routing_D_average)},max:{max(routing_D_average)},average:{sum(routing_D_average)/len(routing_D_average)}')
print(f'distillation_R min:{min(distillation_R_average)},max:{max(distillation_R_average)},average:{sum(distillation_R_average)/len(distillation_R_average)}')
print(f'distillation_T min:{min(distillation_T_average)},max:{max(distillation_T_average)},average:{sum(distillation_T_average)/len(distillation_T_average)}')


# print(f'average distillation fidelity {sum(distillation_fidelity)/len(distillation_fidelity)}')
# print(f'average distillation time {sum(distillation_timecost)/len(distillation_timecost)}')
print(f'allocated {NodeRequest.path_num} path. finished {finished_path}')
print(f'success request {req_sucnum}')
print(f'average qubit fidelity {sum(all_qubit_fidelity)/len(all_qubit_fidelity)}')
#test unfreed situation
            
# for node in nodes:
#    if node.qmemory.used_positions:
#        print(f'node{node} still have positions{node.qmemory.used_positions}')
#    if node.running_list:
#        for node_req in node.running_list:
#            print(f'a request belong to req{node_req.req_id} path {node_req.path_id} unfreed')
#            print(f'it requires positions:{node_req.mem_positions}')
           


# for node in nodes:
#     if len(node.service_time) > 0:
#         print('----------------------------')
#         print(f'node {node.ID} service time:')
#         for key,value in node.service_time.items():
#             print(f'path {key} estimated {value[1]} ns, actual spent {value[0]}ns')


#test random request generator

# test_req = random_request_generator(scale, network, 100)
# print(test_req.src, test_req.dst, test_req.bandwidth, test_req.fidelity)

# non_zero_list = [item for item in temp if item != 0]
# if non_zero_list:
#     avg = sum(non_zero_list) / len(non_zero_list)
#     max_value = max(temp)
#     max_index = temp.index(max_value)
#     print("avg resource:", avg, "in related nodes")
#     print("max resource:", max_value, "at node:", max_index)