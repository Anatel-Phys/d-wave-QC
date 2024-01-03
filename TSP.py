import dwave_networkx as dx
from dwave_networkx.algorithms import traveling_salesperson_qubo
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import dimod

####################
# GRAPH GENERATION #
####################

n_of_towns = 8
graph_matrix = np.matrix([[0,7,0,10,0,5,0,9],
                        [7,0,9,0,0,9,9,0],
                        [0,9,0,0,10,8,8,2],
                        [10,0,0,0,2,8,0,1],
                        [0,0,10,2,0,3,6,4],
                        [5,9,8,8,3,0,0,1],
                        [0,9,8,0,6,0,0,6],
                        [9,0,2,1,4,1,6,0]])

weighted_edges = [] # [(node_1, node_2, weight_12), ...]
for i in range(n_of_towns):
    for j in range(n_of_towns):
        if (graph_matrix[i,j] != 0):
            weighted_edges.append((i+1,j+1,graph_matrix[i,j])) #+1 because node counting begins at 1 in the exam question

graph = nx.Graph()
graph.add_weighted_edges_from(weighted_edges)

#################
# GRAPH DISPLAY #
#################

pos = nx.spring_layout(graph, seed=0)
labels = {x[:2]:graph.get_edge_data(*x)['weight'] for x in graph.edges}
nx.draw_networkx(graph, pos)
nx.draw_networkx_edge_labels(graph, pos,labels)
plt.show()

###############
# TSP SOLVING #
###############

annealing_time = 80
nreads = 1000

dws = DWaveSampler(token="DEV-5c0f1949132356e0df6b12031862ac936ed9dacd")
sampler = EmbeddingComposite(dws)

qubo = traveling_salesperson_qubo(graph) 
bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

while (True):
    response = sampler.sample_qubo(qubo, num_reads=nreads, annealing_time=annealing_time, label="TSP - 2nd method")

    sample = response.first.sample
    cost = response.first.energy
    print(response.info)
    route = [None] * n_of_towns

    for (city, time), val in sample.items():
        if val:
            route[time] = city
    print(route)
    if (None not in route): 
        if (len(route) == len(set(route))): #checks for duplicates
            break

print("First valid route : ")
print(route)




