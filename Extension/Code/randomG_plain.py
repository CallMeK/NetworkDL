import networkx as nx
import sys
import random
import os
import cPickle
import pandas as pd
import numpy as np

def checkG(G_sub):
    for k in G_sub.keys():
        for n in G_sub[k]:
            try:
                assert(k in G_sub[n])
                assert(k!=n)
            except:
                print G_sub
    return 0

# Samples a random walk of length n from graph G
def SampleRandomWalk(G, n):
    #returns a list of nodes
    #I will use a much simpler approach, generate v's  and create adjacency matrix directly
    walk = set()
    head_node = random.choice(G.keys())
    walk.add(head_node)
    candidates = G[head_node].copy()

    while(len(walk)!=n):
        new_node = random.choice(list(candidates))
        walk.add(new_node)
        candidates.remove(new_node)
        candidates |= G[new_node] #union
        if(len(candidates)==0):
            #no more nodes to explore
            break

    checkG(G)

    return walk


def GenerateInduceGraph(G,walk):
    #walk should be a set to reduce the inquiry time
    G_sub = {}
    for node in walk:
        G_sub[node] = set([neig for neig in G[node] if neig in walk])

    checkG(G_sub)
    return G_sub


def GenerateAdjMat(G,order=None):
    #no ordering
    matsize = len(G)
    adjmat = np.zeros([matsize,matsize])
    #fill in the matrix by row
    if(order is None):
        keys_list = G.keys()
    else:
        keys_list = order

    keys_dict = {x:i for i,x in enumerate(keys_list)}
    visited=set([])
    for s in keys_dict:
        if s not in visited:
            for d in G[s]:
                index_s = keys_dict[s]
                index_d = keys_dict[d]
                adjmat[index_s,index_d]=1
                adjmat[index_d,index_s]=1
            visited.add(s)
    return adjmat

def OrderByBFS(G_sub):
    degree_list = {node:len(G_sub[node]) for node in G_sub}
    keys_list = G_sub.keys()
    order = []
    visited = set([])
    #should return the same form of the keys_list
    start_node = max(degree_list,key=degree_list.get)
    order.append(start_node)
    nextlayer = [start_node]
    visited = set(order)

    while(len(order)!=len(keys_list)):
        temp_nextlayer=[]
        for node in nextlayer:
            children = [n for n in G_sub[node] if n not in visited]
            temp_nextlayer += children
            temp_nextlayer = list(set(temp_nextlayer))
        if(len(temp_nextlayer)==0):
            continue #should terminate correctly
        nextlayer = temp_nextlayer
        #sort by the degree_list
        nextlayer = sorted(nextlayer,key=lambda x: degree_list[x],reverse=True)
        #by random for tie-breaker
        order.extend(nextlayer)
        visited = visited | set(nextlayer)
    return order

def OrderByDFS(G_sub):
    degree_list = {node: len(G_sub[node]) for node in G_sub}
    keys_list = G_sub.keys()
    order = []
    start_node = max(degree_list, key=degree_list.get)
    order.append(start_node)
    head = start_node
    visited = set(order)
    stack = [start_node]

    while(len(order)!=len(keys_list)):
        children = [n for n in G_sub[head] if n not in visited]
        if(len(children) == 0):
            # already a leaf
            stack.pop()
            head = stack[-1]
        else:
            head = max(children, key=lambda x: degree_list[x])
            stack.append(head)
            visited.add(head)
            order.append(head)
    return order


if __name__ == '__main__':	
    X_8 = np.load("../AdjmatData/8/X_data_image.npy")
    sample = X_8[1].reshape(8,8)
    from matplotlib import pyplot as plt
    plt.imshow(sample)
    plt.show()
    G_sub = nx.from_numpy_matrix(sample)
    G_sub = nx.to_dict_of_lists(G_sub)
    # order_BFS = OrderByBFS(G_sub)
    # adjmat_BFS = GenerateAdjMat(G_sub,order_BFS)
    # plt.imshow(adjmat_BFS)
    # plt.show()
    order_DFS = OrderByDFS(G_sub)
    adjmat_DFS = GenerateAdjMat(G_sub, order_DFS)
    plt.imshow(adjmat_DFS)
    plt.show()
    print order_DFS
    print G_sub

