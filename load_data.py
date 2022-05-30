import sys
import re
import numpy as np
import math
import collections
from itertools import combinations

def read_input(inputPath):
    with open(inputPath, 'r+') as f:
        info = f.readlines()
        for line in info:
            category, data = line.split()
            if 'coordinates' == category:
                xyzFile = data
            elif 'charges' == category:
                chargefile = data
            elif 'fragSize' == category:
                minAtomNo = float(data)
    try:
        return xyzFile, chargefile, minAtomNo
    except NameError:
        return xyzFile, None, minAtomNo

def read_xyz(xyzFile, chargefile=False):
    if chargefile:
        chargeList = np.genfromtxt(chargefile, dtype='int')[:,1]
        # print(chargeList)

    xyzPattern='-?[0-9]*\.[0-9]*\s*-?[0-9]*\.[0-9]*\s*-?[0-9]*\.[0-9]*\s*'
    coordinates, atoms = [], []
    nodeList = []

    with open(xyzFile, 'r+') as f:
        xyzData = f.readlines()
    count = 0
    for line in xyzData:
        if (re.search(xyzPattern,line)) or (re.search('-?[0-9]*e-[0-9]*\s',line)):
                # print(line.strip())
                atom,x,y,z = line.split()
                coordinates.append([float(x),float(y),float(z)])
                atoms.append(atom)
                if chargefile:
                    nodeList.append((len(atoms), {"element": atom, "charge": chargeList[count]}))
                    
                else:
                    nodeList.append((len(atoms), {"element": atom, "charge": 0}))
                count += 1
    # print(nodeList)
    return atoms, coordinates, nodeList

def EDM(A,B):
    # taken from https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f
    p1 = np.sum(A**2, axis=1)[:, np.newaxis]
    p2 = np.sum(B**2, axis=1)
    p3 = -2 * np.dot(A,B.T)
    # return np.round(np.sqrt(p1+p2+p3),8)
    return np.tril(np.sqrt(np.round(p1+p2+p3,8)))

def EDMbox(A,B):
    # taken from https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f
    p1 = np.sum(A**2, axis=1)[:, np.newaxis]
    p2 = np.sum(B**2, axis=1)
    p3 = -2 * np.dot(A,B.T)
    # return np.round(np.sqrt(p1+p2+p3),8)
    return np.sqrt(np.round(p1+p2+p3,8))

def get_bond_order(atom1, atom2, dist, tol):
    bondDict = {('C', 'C'): {1: [1.37, 1.596], 1.5: [1.37, 1.432], 2: [1.243, 1.382], 3: [1.187, 1.268]}, ('C', 'Br'): {1: [1.789, 1.95]}, ('C', 'Cl'): {1: [1.612, 1.813]}, ('C', 'F'): {1: [1.262, 1.401]}, ('C', 'I'): {1: [1.992, 2.157]}, ('C', 'N'): {1: [1.347, 1.492], 1.5: [1.328, 1.35], 2: [1.207, 1.338], 3: [1.14, 1.177]}, ('C', 'O'): {1: [1.273, 1.448], 2: [1.135, 1.272], 3: [1.115, 1.145]}, ('C', 'P'): {1: [1.858, 1.858], 2: [1.673, 1.673], 3: [1.542, 1.562]}, ('C', 'S'): {1: [1.714, 1.849], 2: [1.553, 1.647], 3: [1.478, 1.535]}, ('C', 'Se'): {1: [1.855, 1.959], 2: [1.676, 1.709]}, ('C', 'Si'): {1: [1.722, 1.848]}, ('O', 'O'): {1: [1.116, 1.516], 2: [1.2, 1.208]}, ('N', 'O'): {1: [1.184, 1.507], 2: [1.066, 1.258]}, ('O', 'S'): {2: [1.405, 1.5]}, ('C', 'H'): {1: [0.931, 1.14]}, ('N', 'H'): {1: [0.836, 1.09]}, ('S', 'H'): {1: [1.322, 1.4]}, ('O', 'H'): {1: [0.912, 1.033]}, ('H', 'F'): {1: [0.917, 1.014]}, ('N', 'N'): {1: [1.181, 1.864], 1.5: [1.332, 1.332], 2: [1.139, 1.252], 3: [1.098, 1.133]}, ('S', 'S'): {1: [1.89, 2.155], 2: [1.825, 1.898]}, ('H', 'H') : {1: [0.741, 0.741]}, ('F', 'O'): {1: [1.421, 1.421]}, ('F', 'F'): {1: [1.322, 1.412]}, ('H', 'Cl'): {1: [1.275, 1.321]}, ('O', 'Cl'): {1: [1.641, 1.704], 2: [1.404, 1.414]}, ('Cl', 'Cl'): {1: [1.9879, 1.9879]}, ('N', 'F'): {1: [1.317, 1.512]}}

    if (atom1, atom2) or (atom2, atom1) in bondDict:
        try:
            dictionary = bondDict[(atom1, atom2)]
        except KeyError:
            dictionary = bondDict[(atom2, atom1)]
    
        try:
            bo = max([k for k in dictionary if dictionary[k][0] - tol <= dist <= dictionary[k][1] + tol])
            return bo
        except ValueError:
            return 0
    
    else:
        print('this program only handles the elements H, B, C, N, O, F, P, S, Cl, Br and I')
        sys.exit()

def get_valence(elem):
    valence = {'H': 1,
    'B': 3,
    'C': 4,
    'N': 5,
    'O': 6,
    'F':7,
    'P': 5,
    'S': 6,
    'Cl': 7,
    'Br': 7,
    'I': 7}
    val = valence[elem]
    return val

def get_radii(atom):
        # vdw radii provided by Bondi
    radii = {"H"  : 1.20,
         "He" : 1.40,
         "C"  : 1.70,
         "N"  : 1.55,
         "O"  : 1.52,
         "F"  : 1.47,
         "Ne" : 1.54,
         "Si" : 2.10,
         "P"  : 1.80,
         "S"  : 1.80,
         "Cl" : 1.75,
         "Ar" : 1.88,
         "As" : 1.85,
         "Se" : 1.90,
         "Br" : 1.85,
         "Kr" : 2.02,
         "Te" : 2.06,
         "I"  : 1.98,
         "Xe" : 2.16}
    
    return radii[atom]


def get_volume(graph, proxMatrix): # uses atom centred Gaussian functions for volume representation 
    # uses formula provided by https://pubs.acs.org/doi/full/10.1021/ci800315d (ShaEP)
    
    nodeList = list(graph.nodes)

    # different atoms
    pairList = combinations(nodeList, 2)
    pairVol = 0 
    # print(proxMatrix)
    for pair in pairList:
        atom1 = graph.nodes[pair[0]]['element']
        atom2 = graph.nodes[pair[1]]['element']

        sortedPair = sorted(pair)
        dist = proxMatrix[sortedPair[1]-1, sortedPair[0]-1] # requires sorting because only the lower triangular elements in the proximity matrix are populated

        alpha1 = math.pi * pow((3 * 2 * math.sqrt(2))/(4 * math.pi * get_radii(atom1)**3), 2/3) # these are the decay factors
        alpha2 = math.pi * pow((3 * 2 * math.sqrt(2))/(4 * math.pi * get_radii(atom2)**3), 2/3) # 2√2 is taken to be the amplitude of the Gaussian

        incVol = 8 * math.exp( -1 * (alpha1 * alpha2 * dist**2)/(alpha1 + alpha2)) * pow(math.pi/(alpha1 + alpha2), 3/2) # incremental volume
        pairVol = pairVol + (2 * incVol)
    
    # del incVol # resetting variable

    # same atoms
    sameVol = 0 
    nodeCount = collections.Counter([graph.nodes[x]['element'] for x in nodeList]).most_common()
    # print(nodeCount)
    for item in nodeCount:
        atom = item[0]
        count = item[1]
        alpha = math.pi * pow((3 * 2 * math.sqrt(2))/(4 * math.pi * get_radii(atom)**3), 2/3)

        incVol = 8 * pow(math.pi/(2 * alpha), 3/2)
        sameVol = sameVol + (count * incVol)
    
    volume = pairVol + sameVol

    return volume