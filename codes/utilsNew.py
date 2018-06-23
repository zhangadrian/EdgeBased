from __future__ import print_function

import numpy as np 
import random
import sys
import os
import six.moves.cPickle as pickle 


dataPath = '../../DeepHawkes/dataset_weibo.txt'
picklePath = 'subgraphs.pickle'
def saveSubgraph(sourceDataPath, targetDataPath):
	print(os.getcwd())
	subgraphs = {}
	tempIndex = 0
	with open(sourceDataPath, 'r') as graphs:
		for graph in graphs:
			elements = graph.strip().split('\t')
			graphID = int(elements[0])
			traces = elements[4]
			traceList = traces.split()[1:-1]
			allEdge = np.empty([3,len(traceList)])
			index = 0
			tempIndex += 1
			if tempIndex % 10 == 0:
				print(tempIndex)

			for trace in traceList:
				tempElement = trace.split(':')
				tempTime = int(tempElement[1])
				tempEdge = tempElement[0].split('/')
				allEdge[:, index] = np.array([(tempTime), (int(tempEdge[-2])), (int(tempEdge[-1]))])
				allEdge = np.argsort(allEdge, axis=1)
				index += 1
				# if index % 1000 == 0:
				# 	print(index)

			subgraphs[graphID] = allEdge
	with open(targetDataPath, 'wb') as handle:
		pickle.dump(subgraphs, handle)
		print(1)
	with open(targetDataPath, 'rb') as handle:
		test = pickle.load(handle)
		print(1)
	print(test == subgraphs)

def getSnapshot(graph, timeSlot, targetDataPath):

	diffEndTime = graph[0,-1]
	snapshotNum = math.floor(diffEndTime/timeSlot) + 1

	snapshots = [0] * snapshotNum
	index = 0
	edgeNum = np.shape(graph)[1]

	for i in range(1, snapshotNum+1):
		tempTime = timeSlot * i
		tempsnapshot = np.array([(0), (0)])
		while graph[0, index] < tempTime and index < edgeNum:
			tempsnapshot = np.append([tempsnapshot], [graph[1:3, index]], axis=1)
			index += 1

	with open(targetDataPath, 'wb') as handle:
		pickle.dump(snapshots, handle)
	return snapshots

def getSnapshot(graph, snapshotNum, targetDataPath):

	diffEndTime = graph[0,-1]
	# snapshotNum = math.floor(diffEndTime/timeSlot) + 1
	timeSlot = (diffEndTime+1)/snapshots

	snapshots = [0] * snapshotNum
	index = 0
	edgeNum = np.shape(graph)[1]

	for i in range(1, snapshotNum+1):
		tempTime = timeSlot * i
		tempsnapshot = np.array([(0), (0)])
		while graph[0, index] < tempTime and index < edgeNum:
			tempsnapshot = np.append([tempsnapshot], [graph[1:3, index]], axis=1)
			index += 1

	with open(targetDataPath, 'wb') as handle:
		pickle.dump(snapshots, handle)
	return snapshots


# saveSubgraph(dataPath, picklePath)







			


