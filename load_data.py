import pandas as pd
import sys
import re
from numpy import linalg
import numpy as np
import math
import collections
from itertools import combinations

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
        if (re.search(xyzPattern,line)):
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

def proxMat(atoms, coordinates):
    '''making use of symmetric nature of matrix'''
    matrix = np.full((len(atoms), len(atoms)), 0, dtype='float64')

    for i in range(len(atoms)):
        a1 = np.array(coordinates[i])
        colInd = range(0, i+1) ## this ensures i > j, we have a lower triangular matrix
        for j in colInd:
            a2 = np.array(coordinates[j])
            dist = linalg.norm(a1-a2)
            dist = round(dist, 5)
            if dist > 20:
                dist = 100.00000
            matrix[i,j] = dist ## only the lower triangular elements are populated
    # print('matrix', matrix)
    return matrix

def EDM(A,B):
    # taken from https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f
    p1 = np.sum(A**2, axis=1)[:, np.newaxis]
    p2 = np.sum(B**2, axis=1)
    p3 = -2 * np.dot(A,B.T)
    # return np.round(np.sqrt(p1+p2+p3),8)
    return np.tril(np.sqrt(np.round(p1+p2+p3,8)))


def retrieveBI(atom1, atom2): # retrieve bond information, returns subsection of dataframe of the possible bonding between two atoms
    # bond lengths taken from https://cccbdb.nist.gov/expbondlengths1.asp 
    # print(atom1, atom2)
    d = {'atom1': ['C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','O','O','N','N','O','C','N','S','O','H','N','N','N','N','S','S'],
    'atom2': ['C','C','C','C','Br','Cl','F','I','N','N','N','N','O','O','O','P','P','P','S','S','S','Se','Se','Si','O','O','O','O','S','H','H','H','H','F','N','N','N','N','S','S'],
    'bond_type': ['single','aromatic','double','triple','single','single','single','single','single','aromatic','double','triple','single','double','triple','single','double','triple','single','double','triple','single','double','single','single','double','single','double','double','single','single','single','single','single','single','aromatic','double','triple','single','double'],
    'av': [1.508,1.396,1.333,1.213,1.9,1.746,1.337,2.124,1.417,1.339,1.286,1.16,1.399,1.197,1.137,1.858,1.673,1.552,1.8,1.587,1.507,1.919,1.693,1.802,1.363,1.204,1.368,1.184,1.435,1.09,1.009,1.345,0.967,0.966,1.523,1.332,1.202,1.12,2.009,1.874],
    'std': [0.039,0.014,0.024,0.021,0.053,0.047,0.03,0.054,0.05,0.008,0.041,0.008,0.034,0.027,0.012,0,0,0.014,0.038,0.037,0.04,0.056,0.013,0.07,0.119,0.005,0.094,0.038,0.044,0.018,0.043,0.02,0.022,0.069,0.332,0,0.057,0.019,0.084,0.033],
    'min': [1.37,1.37,1.243,1.187,1.789,1.612,1.262,1.992,1.347,1.328,1.207,1.14,1.32,1.115,1.128,1.858,1.673,1.542,1.714,1.553,1.478,1.855,1.676,1.722,1.116,1.2,1.184,1.066,1.405,0.931,0.836,1.322,0.912,0.917,1.181,1.332,1.139,1.098,1.890,1.825],
    'max': [1.596,1.432,1.382,1.268,1.95,1.813,1.401,2.157,1.492,1.35,1.338,1.177,1.448,1.272,1.145,1.858,1.673,1.562,1.849,1.647,1.535,1.959,1.709,1.848,1.516,1.208,1.507,1.258,1.5,1.14,1.09,1.4,1.033,1.014,2.236,1.332,1.252,1.133,2.155,1.898]}
    df = pd.DataFrame(d)
    if atom1 != atom2:
        rdf1 = df.loc[(df['atom1'] == atom1) & (df['atom2'] == atom2)]
        # print(rdf1)
        rdf2 = df.loc[(df['atom1'] == atom2) & (df['atom2'] == atom1)]
        # print(rdf2)

        df_list = [rdf1, rdf2]
        status = [rdf1.empty, rdf2.empty]
        # print([i for i, x in enumerate(status) if not x][0])
        rdf = df_list[[i for i, x in enumerate(status) if not x][0]]
        # print(rdf)
    
        if rdf.empty:
            print('this program only handles the elements H, B, C, N, O, F, P, S, Cl, Br and I')
            sys.exit()
        else:
            return rdf

    elif atom1 == atom2:
        rdf = df.loc[(df['atom1'] == atom1) & (df['atom2'] == atom2)]

        if rdf.empty:
            print('this program only handles the elements H, B, C, N, O, F, P, S, Cl, Br and I')
            sys.exit()
        else:
            return rdf

def get_range(df, bondtype):
    rdf = df.loc[df['bond_type'] == bondtype]
    min = rdf['min'].to_list()[0]
    max = rdf['max'].to_list()[0]
    return min, max

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


def get_volume(graph, proxMatrix): # uses atom centred Gaussian functions for volume representation 
    # uses formula provided by https://pubs.acs.org/doi/full/10.1021/ci800315d (ShaEP)

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

        alpha1 = math.pi * pow((3 * 2 * math.sqrt(2))/(4 * math.pi * radii[atom1]**3), 2/3) # these are the decay factors
        alpha2 = math.pi * pow((3 * 2 * math.sqrt(2))/(4 * math.pi * radii[atom2]**3), 2/3) # 2√2 is taken to be the amplitude of the Gaussian

        incVol = 8 * math.exp( -1 * (alpha1 * alpha2 * dist**2)/(alpha1 + alpha2)) * pow(math.pi/(alpha1 + alpha2), 3/2) # incremental volume
        pairVol = pairVol + (2 * incVol)
    
    del incVol # resetting variable

    # same atoms
    sameVol = 0 
    nodeCount = collections.Counter([graph.nodes[x]['element'] for x in nodeList]).most_common()
    # print(nodeCount)
    for item in nodeCount:
        atom = item[0]
        count = item[1]
        alpha = math.pi * pow((3 * 2 * math.sqrt(2))/(4 * math.pi * radii[atom]**3), 2/3)

        incVol = 8 * pow(math.pi/(2 * alpha), 3/2)
        sameVol = sameVol + (count * incVol)
    
    volume = pairVol + sameVol

    return volume
       