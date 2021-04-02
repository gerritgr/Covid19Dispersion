import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import imageio
import glob
import random


########################################################
# Random graph models
########################################################


def household_simpleGeom(num_nodes=100, hh_size = 4, degree_out=5):
    # Important: seed only for power law
    assert(num_nodes % hh_size == 0)
    hh_num = int(num_nodes/hh_size)

    meta_graph = geom(hh_num, mean_degree=degree_out*hh_size) #erdos_renyi(hh_num, mean_degree=degree_out*hh_size, connect_prob=None)
    pl_edges = list(meta_graph.edges())
    hh_edges = list()

    for node in meta_graph.nodes():
        new_nodes = [str(node)+'_'+str(ind) for ind in range(hh_size)]
        for n1 in new_nodes:
            for n2 in new_nodes:
                if n1 != n2:
                    hh_edges.append((n1,n2))

    for n1, n2 in pl_edges:
        new_edge = np.random.choice(range(hh_size), 2)
        n1_new = new_edge[0]
        n2_new = new_edge[1]

        n1_new = str(n1)+'_'+str(n1_new)
        n2_new = str(n2)+'_'+str(n2_new)
        hh_edges.append((n1_new, n2_new))

    hh_graph = nx.Graph(hh_edges)
    hh_graph = nx.convert_node_labels_to_integers(hh_graph)

    return hh_graph



def power_law_meandegree(num_nodes=100, min_truncate=2, mean_degree=8):
    for _ in range(10):
        for gamma in np.linspace(1.01, 5.0, 1001):
            G, _ = power_law_graph(num_nodes=num_nodes, gamma=gamma, min_truncate=min_truncate)
            if nx.is_connected(G) and mean_degree - 0.1 < sum(dict(G.degree()).values()) / num_nodes < mean_degree + 0.1:
                return G
    print('power_law_meandegree failed')



# seed does not really work
def power_law_graph(num_nodes=100, gamma=2.0, min_truncate=2, max_truncate=None):
    degree_distribution = [0] + [(k + 1) ** (-gamma) for k in range(num_nodes)]
    degree_distribution[:min_truncate] = [0.0] * min_truncate
    if max_truncate is not None:
        # max truncate and everything larger is zero
        degree_distribution[max_truncate:] = [0.0] * (len(degree_distribution) - max_truncate)
    assert (len(degree_distribution) == num_nodes + 1)
    z = np.sum(degree_distribution)
    degree_distribution = [p / z for p in degree_distribution]
    while True:
        degee_sequence = [np.random.choice(range(num_nodes + 1), p=degree_distribution) for _ in range(num_nodes)]
        degee_sequence = [1 if d==0 else d for d in degee_sequence]
        if np.sum(degee_sequence) % 2 == 0:
            break

    np.random.seed(None)

    for seed in range(100000):
        seed += 42
        contact_network = nx.configuration_model(degee_sequence, create_using=nx.Graph)
        for n in contact_network.nodes():
            try:
                contact_network.remove_edge(n, n)  # hack, how to remove self-loops in nx 2.4??
            except:
                pass
        if nx.is_connected(contact_network):
            return contact_network, None

    print('Failed graph generation')


########################################################
# From other file
########################################################

def barabasi_mean(num_nodes=100,  mean_degree=5):
    graphs = [nx.barabasi_albert_graph(num_nodes,i+1) for i in range(10)]
    graphs = [(G, np.abs(mean_degree-sum(dict(G.degree()).values())/num_nodes)) for G in graphs]
    graphs = sorted(graphs, key = lambda x: x[1])
    return graphs[0][0]



def geom(num_nodes=100,  mean_degree=5):
    for _ in range(100000):
        for radius in np.linspace(0.0001, 1.0, 100000):
            G = nx.random_geometric_graph(num_nodes, radius)
            if nx.is_connected(G) and nx.is_connected(G) and mean_degree - 0.1 < sum(dict(G.degree()).values())/num_nodes < mean_degree + 0.1:
                G = nx.convert_node_labels_to_integers(G)
                return G
    print('failed graph generation geom_graph2')




def newman(num_nodes=100, k=6, p=0.2, nondet=True):
    for seed in range(10000):
        seed += 222
        if nondet:
            seed = None
        G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p, seed=seed)
        if nx.is_connected(G) and nx.is_connected(G):
            G = nx.convert_node_labels_to_integers(G)
            return G
    print('failed graph generation')


#for _ in range(20):
#    household_simpleGeom_plot()