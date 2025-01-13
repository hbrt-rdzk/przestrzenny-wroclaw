import pandas as pd
import numpy as np
import heapq
from geopy.distance import geodesic
import ast
import re
from tqdm import tqdm


def parse_coords(coord_str):
    match = re.match(r'POINT \(([-+]?\d*\.\d+|\d+)\s([-+]?\d*\.\d+|\d+)\)', coord_str)
    if match:
        return (float(match.group(2)), float(match.group(1)))  
    else:
        return None
    

def preprocess(df):
    df = df.dropna(subset=['Koordynaty startowe', 'Koordynaty końcowe'])
    df['Koordynaty startowe'] = df['Koordynaty startowe'].apply(parse_coords)
    df['Koordynaty końcowe'] = df['Koordynaty końcowe'].apply(parse_coords)
    df = df.dropna()
    df['Czasy przejazdu'] = df['Czasy przejazdu'].apply(ast.literal_eval)
    return df


def timeToInt(time):
    timeSplit = time.split(":")
    timeInt = int(timeSplit[0]) * 60 + int(timeSplit[1])
    return timeInt


def timeToString(time):
    timeString = ""
    h = (time // 60) % 24
    m = time % 60
    if h < 10:
        timeString += "0"
    timeString += str(h) + ":"
    if m < 10:
        timeString += "0"
    timeString += str(m)
    return timeString


def deltaTimeToString(delta):
    h = (delta // 60) % 24
    m = delta % 60
    return f"{h} godz {m} min"


def costT(start, rideStart, rideEnd):
    start2 = start % 1440
    rideStart2 = rideStart % 1440
    rideEnd2 = rideEnd % 1440
    if start2 <= rideStart2 and start2 < rideEnd2:
        return rideEnd2 - start2
    return rideEnd2 - start2 + 1440


class Node:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.stops = []

    def addStop(self, endNode, line, times):
        self.stops.append(BigEdge(endNode, line, times))

    def __str__(self):
        s = f"{self.name}, {self.x}, {self.y}, ["
        for stop in self.stops:
            s += str(stop) + ", "
        return s + "]"


class Edge:
    def __init__(self, endNode, line, startTime, endTime):
        self.endNode = endNode
        self.line = line
        self.startTime = startTime
        self.endTime = endTime

    def __str__(self):
        return f"{self.endNode.name} {self.line} {timeToString(self.startTime)} {timeToString(self.endTime)}"
    
class BigEdge:
    def __init__(self, endNode, line, times):
        self.endNode = endNode
        self.line = line
        self.times = times

    def __str__(self):
        return f"{self.endNode.name} {self.line} {timeToString(self.times)}"


class Graph:
    def __init__(self):
        self.nodes = {}
        self.distances = {}

    def newEdge(self, line, times, name1, x1, y1, name2, x2, y2):
        self.nodes.setdefault(name1, Node(name1, x1, y1))
        self.nodes.setdefault(name2, Node(name2, x2, y2))
        self.nodes[name1].addStop(self.nodes[name2], line, times)

    def calculate_distances(self):
        for name1, node1 in tqdm(list(self.nodes.items())):
            self.distances[name1] = {}
            for name2, node2 in self.nodes.items():
                if name1 != name2:
                    distance = int(np.ceil(geodesic((node1.x, node1.y), (node2.x, node2.y)).meters * 0.012))
                    self.distances[name1][name2] = distance
                else:
                    self.distances[name1][name2] = 0
    

class DijkstraNode:
    def __init__(self, node):
        self.node = node
        self.d = float('inf')
        self.p = None
        self.pEdge = None

    def resetNode(self):
        self.d = float('inf')
        self.p = None
        self.pEdge = None


class Dijkstra:
    def __init__(self, graph):
        self.stops = {}
        self.Q = []
        self.graph = graph
        for keys, value in graph.nodes.items():
            self.stops[keys] = DijkstraNode(value)

    def printRoadHelper(self, stop, line, startTime):
        edge = stop.pEdge
        prev = stop.p
        if edge is None:
            print(f"{'Linia ' if line != 'Pieszo' else ''}{line}: {stop.node.name} {timeToString(startTime)} - ", end="")
        else:
            if edge.line == line:
                self.printRoadHelper(prev, line, edge.startTime)
            else:
                self.printRoadHelper(prev, edge.line, edge.startTime)
                print(f"{timeToString(edge.endTime)} {stop.node.name}")
                print(f"{'Linia ' if line != 'Pieszo' else ''}{line}: {stop.node.name} {timeToString(startTime)} - ", end="")

    def printRoad(self, stop):
        edge = stop.pEdge
        prev = stop.p
        self.printRoadHelper(prev, edge.line, edge.startTime)
        print(f"{timeToString(edge.endTime)} {stop.node.name}")

    def getNodeTime(self, stop):
        return self.stops[stop].d
    
    def getAllNodesTime(self):
        return {stop: self.stops[stop].d for stop in self.stops}


class Dijkstra:
    def __init__(self, graph):
        self.stops = {}
        self.Q = []
        self.graph = graph
        for keys, value in graph.nodes.items():
            self.stops[keys] = DijkstraNode(value)

    def reset(self):
        self.Q = []
        for keys, value in self.stops.items():
            value.resetNode()
            self.Q.append(value)

    def printRoadHelper(self, stop, line, startTime):
        edge = stop.pEdge
        prev = stop.p
        if edge is None:
            print(f"{'Linia ' if line != 'Pieszo' else ''}{line}: {stop.node.name} {timeToString(startTime)} - ", end="")
        else:
            if edge.line == line:
                self.printRoadHelper(prev, line, edge.startTime)
            else:
                self.printRoadHelper(prev, edge.line, edge.startTime)
                print(f"{timeToString(edge.endTime)} {stop.node.name}")
                print(f"{'Linia ' if line != 'Pieszo' else ''}{line}: {stop.node.name} {timeToString(startTime)} - ", end="")

    def printRoad(self, stop):
        edge = stop.pEdge
        prev = stop.p
        self.printRoadHelper(prev, edge.line, edge.startTime)
        print(f"{timeToString(edge.endTime)} {stop.node.name}")

    def getNodeTime(self, stop):
        return self.stops[stop].d
    
    def getAllNodesTime(self):
        return {stop: self.stops[stop].d for stop in self.stops}

    def dijkstra(self, start, timeStart, allowWalk=True):
        self.reset()
        startNode = DijkstraNode(Node("start", start[0], start[1]))
        for node in self.stops.values():
            node.d = int(np.ceil(geodesic(start, (node.node.x, node.node.y)).meters * 0.012))
            node.p = startNode
            node.pEdge = Edge(startNode, "Pieszo", 0, node.d)
            
        timeInt = timeToInt(timeStart)

        while self.Q:
            uMin = self.Q[0]
            dMin = self.Q[0].d
            for u in self.Q:
                if u.d < dMin:
                    uMin = u
                    dMin = u.d
            self.Q.remove(uMin)
            currTime = timeInt + uMin.d
            for v in uMin.node.stops:
                vNode = self.stops[v.endNode.name]
                times = v.times
                startTime, endTime = None, None
                bestWait = float('inf')
                for s, e in times:
                    w = s - currTime
                    if w < 0:
                        w += 1440
                    if w < bestWait:
                        bestWait = w
                        startTime = s
                        endTime = e

                if vNode.d > uMin.d + costT(timeInt + uMin.d, startTime, endTime):
                    vNode.d = uMin.d + costT(timeInt + uMin.d, startTime, endTime)
                    vNode.p = uMin
                    vNode.pEdge = Edge(uMin.node, v.line, startTime, endTime)
                if allowWalk:
                    for other_node_name, walk_time in self.graph.distances[uMin.node.name].items():
                        other_node = self.stops[other_node_name]
                        if other_node.d > uMin.d + walk_time:
                            other_node.d = uMin.d + walk_time
                            other_node.p = uMin
                            other_node.pEdge = Edge(uMin.node, "Pieszo", uMin.d, uMin.d + walk_time)
