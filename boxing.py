from scipy import misc
import miscellaneous
import numpy as np 
from itertools import product
from itertools import chain


class Box:
    def __init__(self, xlist, ylist, zlist):
        self.xbounds = xlist
        self.ybounds = ylist
        self.zbounds = zlist

def classify_boxes(xmin, xmax, ymin, ymax, zmin, zmax, d): # d is the number of boxes in each direction, d is integer
    xvalues = np.linspace(xmin, xmax, num=d+1)
    yvalues = np.linspace(ymin, ymax, num=d+1)
    zvalues = np.linspace(zmin, zmax, num=d+1)

    boxDict = {}

    dlist = [x+1 for x in range(d)]
    boxLabelList = product(dlist, repeat=3)
    for label in boxLabelList:
        xlist = [xvalues[label[0]-1], xvalues[label[0]]]
        ylist = [yvalues[label[1]-1], yvalues[label[1]]]
        zlist = [zvalues[label[2]-1], zvalues[label[2]]]

        box = Box(xlist, ylist, zlist)
        boxDict[label] = box
    
    return boxDict

def get_boundary_values(coordinates): # coordinates is list of lists
    xvalues, yvalues, zvalues = np.array([x[0] for x in coordinates]), np.array([x[1] for x in coordinates]), np.array([x[2] for x in coordinates])
    xmin, xmax = min(xvalues), max(xvalues)
    ymin, ymax = min(yvalues), max(yvalues)
    zmin, zmax = min(zvalues), max(zvalues)

    return xmin, xmax, ymin, ymax, zmin, zmax

def locate_box(boxDict, coordinates, nodeLabel):
    coord = coordinates[nodeLabel - 1]

    for box in boxDict:
        if boxDict[box].xbounds[0] <= coord[0] <= boxDict[box].xbounds[1] and boxDict[box].ybounds[0] <= coord[1] <= boxDict[box].ybounds[1] and boxDict[box].zbounds[0] <= coord[2] <= boxDict[box].zbounds[1]:
            return box


def classify_nodes(graph, coordinates, boxDict): # updates boxDict by removing any empty/unoccupied boxes
    nodeList = list(graph.nodes)
    filledBoxes = []
    for node in nodeList:
        # coord = coordinates[node - 1]
        # boxLabel = [box for box in boxDict if boxDict[box].xbounds[0] <= coord[0] <= boxDict[box].xbounds[1] and boxDict[box].ybounds[0] <= coord[1] <= boxDict[box].ybounds[1] and boxDict[box].zbounds[0] <= coord[2] <= boxDict[box].zbounds[1]][0]
        boxLabel = locate_box(boxDict, coordinates, node)
        graph.nodes[node]['box'] = boxLabel
        filledBoxes.append(boxLabel)
    
    filledBoxes = list(dict.fromkeys(filledBoxes))
    # emptyBoxes = [x for x in boxDict if x not in filledBoxes]
    emptyBoxes = list(set(boxDict) - set(filledBoxes))
    for box in emptyBoxes:
        boxDict.pop(box)

    return graph, boxDict

def classify_donors_acceptors(graph, daDict): # will perform this for both donors and acceptors
    for da in daDict:
        nodeList = daDict[da].nodes
        boxLabelList = [graph.nodes[x]['box'] for x in nodeList]
        boxLabelList = list(dict.fromkeys(boxLabelList))
        daDict[da].add_boxLabels(boxLabelList)
    
    return daDict

def classify_cycles(graph, cycleDict):
    for cycle in cycleDict:
        nodeList = [x for x in set(chain(*cycleDict[cycle].edgeList))]
        boxLabelList = [graph.nodes[x]['box'] for x in nodeList]
        boxLabelList = list(dict.fromkeys(boxLabelList))
        cycleDict[cycle].add_boxLabels(boxLabelList)
    return cycleDict

def classify_aromsys(graph, aromaticDict):
    for asys in aromaticDict:
        nodeList = [x for x in set(chain(*miscellaneous.flatten(aromaticDict[asys].cycle_list)))]
        boxLabelList = [graph.nodes[x]['box'] for x in nodeList]
        boxLabelList = list(dict.fromkeys(boxLabelList))
        aromaticDict[asys].add_boxLabels(boxLabelList)
    return aromaticDict

def all_classification(graph, donorDict, acceptorDict, cycleDict, aromaticDict):
    
    donorDict, acceptorDict = classify_donors_acceptors(graph, donorDict), classify_donors_acceptors(graph, acceptorDict)
    # cycleDict = classify_cycles(graph, cycleDict)
    aromaticDict = classify_aromsys(graph, aromaticDict)
    return donorDict, acceptorDict, aromaticDict

def box_classification(coordinates, graph, d=5):
    xmin, xmax, ymin, ymax, zmin, zmax = get_boundary_values(coordinates)
    boxDict = classify_boxes(xmin, xmax, ymin, ymax, zmin, zmax, d)
    graph, boxDict = classify_nodes(graph, coordinates, boxDict)
    
    return graph, boxDict


def neighbouring_boxes(boxLabelList, boxDict):
    binaryList = [-1,0,1]
    possibilities = product(binaryList, repeat=3)
    binList = [np.array(x) for x in possibilities]
    npboxLabels = [np.array(x) for x in boxLabelList]
    neighbourList = []
    for label in npboxLabels:
        neighbours = [label + x for x in binList]
        neighbourList.extend(neighbours)
    
    neighboxList = np.array(neighbourList) #+ npboxLabels
    neighboxList = list(np.unique(neighboxList, axis=0))
    # neighboxList = list(set(neighbourList))
    # neighboxList = list(dict.fromkeys(neighboxList)) # get the unique values
    neighboxList = [tuple(x) for x in neighboxList]
    # print('neighboxList', neighboxList)
    # print('boxes: ', [box for box in boxDict])
    # neighboxList = [x for x in neighboxList if x in boxDict]
    neighboxList = list(set(neighboxList).intersection(boxDict))
    return neighboxList

# def adjacent_status_da(donorBoxLabelList, acceptorBoxLabelList, boxDict):
#     donorBoxesNeigh = neighbouring_boxes(donorBoxLabelList, boxDict)
#     if len(set(donorBoxesNeigh).intersection(acceptorBoxLabelList)) > 0:
#         return True
#     else:
#         return False

def adjacent_da(da, donorDict, acceptorDict, boxDict):
    donorBoxLabels, acceptorBoxLabels = donorDict[da[0]].boxLabelList, acceptorDict[da[1]].boxLabelList
    donorBoxesNeigh = neighbouring_boxes(donorBoxLabels, boxDict)
    if len(set(donorBoxesNeigh).intersection(acceptorBoxLabels)) > 0:
        return da
