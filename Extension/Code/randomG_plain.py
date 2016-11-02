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
				sys.exit()
				raise("shit")
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
	#should return the same form of the keys_list
	start_node = max(degree_list,key=degree_list.get)
	order.append(start_node)
	nextlayer = [start_node]

	while(len(order)!=len(keys_list)):
		temp_nextlayer=[]
		for node in nextlayer:
			children = [n for n in G_sub[node] if n not in order]
			temp_nextlayer += children
			temp_nextlayer = list(set(temp_nextlayer))
		if(len(temp_nextlayer)==0):
			continue #should terminate correcly
		nextlayer = temp_nextlayer
		#sort by the degree_list
		nextlayer = sorted(nextlayer,key=lambda x: degree_list[x],reverse=True)
		#by random for tie-breaker
		order.extend(nextlayer)
	return order


if __name__ == '__main__':	
	N_sub = 8
	df = pd.read_pickle('../Fulldata.pkl')
	#df = pd.read_pickle('../Smalldata_test.pkl')
	df_use = df[df['Labels']!=2]
	print df_use.shape
	X = np.zeros([df_use.shape[0],N_sub*N_sub])
	for i,g in df_use['GraphObjs'].iteritems():
		print 'start sampling'
		checkG(g)
		walk = SampleRandomWalk(g,N_sub)
		print 'generate'
		gsub = GenerateInduceGraph(g, walk)
		print 'bfs'
		order = OrderByBFS(gsub)
		assert len(walk) ==N_sub
		print(GenerateAdjMat(gsub,order))


