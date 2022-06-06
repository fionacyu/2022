import math
from itertools import combinations
from itertools import product
import numpy as np 
from numpy import linalg
import miscellaneous
import load_data

# parameters
KCAL_TO_KJ = 4.184
DEG_TO_RADIAN = math.pi/180

class parameter:
    def add_r1(self, r1):
        self.r1 = r1
    
    def add_theta0(self, theta0):
        self.theta0 = theta0
    
    def add_x1(self, x1):
        self.x1 = x1
    
    def add_D1(self, D1):
        self.D1 = D1
    
    def add_Z1(self, Z1):
        self.Z1 = Z1

    def add_Vi(self, Vi):
        self.Vi = Vi

    def add_Uj(self, Uj):
        self.Uj = Uj
    
    def add_Xi(self, Xi):
        self.Xi = Xi

def bond_order(graph, node1, node2):
    bondorder = graph[node1][node2]['bo']

        # amide C-N set to have bondorder of 1.41
    if (graph.nodes[node1]['element'] == 'C' and graph.nodes[node2]['element'] == 'N') or (graph.nodes[node2]['element'] == 'C' and graph.nodes[node1]['element'] == 'N'):
        cnode = [x for x in (node1, node2) if graph.nodes[x]['element'] == 'C'][0]
        cneigh = [x for x in graph.neighbors(cnode)]
        cneighel = [graph.nodes[x]['element'] for x in cneigh]
        if 'O' in cneighel:
            oxygennodes = [x for x in cneigh if graph.nodes[x]['element'] == 'O']
            boList = [graph[x][cnode]['bo'] for x in oxygennodes]
            if 2 in boList:
                bondorder = 1.41
    return bondorder
        

def bond_distance(prmDict, graph, node1, node2):
    ri, rj = prmDict[graph.nodes[node1]['at']].r1, prmDict[graph.nodes[node2]['at']].r1
    chiI, chiJ = prmDict[graph.nodes[node1]['at']].Xi, prmDict[graph.nodes[node2]['at']].Xi
    bondorder = bond_order(graph, node1, node2)
    rbo = -0.1332 * (ri+rj) * math.log(bondorder) # equation 3
    
    ren = ri*rj*(pow((math.sqrt(chiI) - math.sqrt(chiJ)),2.0)) / (chiI*ri + chiJ*rj) # equation 4

    r0 = ri + rj + rbo - ren # typo in original published paper
    return r0
# want to have a function to calculate each type of energy with a graph as an input 

def bond_energy(graph, prmDict):
    energy = 0

    edgeList = list(graph.edges)
    for edge in edgeList:
        node1, node2 = edge[0], edge[1]
        # first need to obtain the equilibrium bond distance
        r0 = bond_distance(prmDict, graph, node1, node2)
    
        # second need to obtain the force constant kb
        ZI, ZJ = prmDict[graph.nodes[node1]['at']].Z1, prmDict[graph.nodes[node2]['at']].Z1
        
        # include the 1/2 from equation 1a in the force constant
        kb = (0.5 * KCAL_TO_KJ * 664.12 * ZI * ZJ) / (r0 * r0 * r0)  # equation 6
        # kb = round(kb,3)

        delta2 = (graph[node1][node2]['r'] - r0)**2
        energy += kb * delta2

        # print(node1, node2, graph.nodes[node1]['at'], graph.nodes[node2]['at'], bondorder, graph[node1][node2]['r'], round(r0,3), round(kb,3), round(graph[node1][node2]['r'] - r0,3), round(kb * delta2,3))

    return energy


def angle_energy(graph, prmDict):
    energy = 0.0 

    nodeList = [x for x in list(graph.nodes) if graph.degree[x] > 1]
    for node in nodeList:
        theta0 = prmDict[graph.nodes[node]['at']].theta0 * DEG_TO_RADIAN
        cosT0 = np.cos(theta0)
        sinT0 = np.sin(theta0)
        c2 = 1.0 / (4.0 * sinT0 * sinT0)
        c1 = -4.0 * c2 * cosT0
        c0 = c2*(2.0*cosT0*cosT0 + 1.0)

        neighList = list(graph.neighbors(node))
        neighPairs = combinations(neighList, 2)
        for pair in neighPairs:
            # print('pair', pair[0], pair[1])
            a, b, c = np.array(graph.nodes[node]['coord']), np.array(graph.nodes[pair[0]]['coord']), np.array(graph.nodes[pair[1]]['coord'])
            rab = bond_distance(prmDict, graph, node, pair[0])
            rac = bond_distance(prmDict, graph, node, pair[1])
            rbc = math.sqrt(rab*rab + rac*rac - 2.0 * rab*rac*cosT0)
            ab = b - a
            ac = c - a
            # rbc = linalg.norm(b - c)
            # print('first rbc', rbc)
            # rbc = math.sqrt(linalg.norm(ab)*linalg.norm(ab) + linalg.norm(ac)*linalg.norm(ac) - 2.0 * linalg.norm(ab)*linalg.norm(ac)*cosT0)
            # print('second rbc', rbc)
            cosT = np.dot(ab, ac) / (linalg.norm(ab) * linalg.norm(ac))
            theta = np.arccos(cosT) # angle in radians

            # force constant
            ka = (664.12 * KCAL_TO_KJ) * (prmDict[graph.nodes[pair[0]]['at']].Z1 * prmDict[graph.nodes[pair[1]]['at']].Z1 / (pow(rbc, 5.0)))
            ka *= (3.0 * rab * rac * (1.0 - cosT0 * cosT0) - rbc * rbc * cosT0)
            # print('ka', ka)
            # value for n 
            if graph.nodes[node]['ed'] == 2: # linear case
                inc = ka * (1 + cosT) # typo in Rappe paper, not 1 - cosT src: openbabel
            elif graph.nodes[node]['ed'] == 3: # sp2
                n = 3
                ka /= 9
                # a penalty for angles close to zero is added, based on ESFF
                inc = ka * (1 - np.cos(n*theta))# + math.exp(-20.0*(theta - theta0 + 0.25))
            elif graph.nodes[node]['ed'] == 4:
                inc = ka*(c0 + c1*cosT + c2*(2.0*cosT*cosT - 1.0))
            
            energy += inc
            # print(graph.nodes[node]['at'], graph.nodes[pair[0]]['at'], graph.nodes[pair[1]]['at'], round(theta * 180 / math.pi, 3), round(theta0, 3), ka, inc)

    return energy

def torsional_energy(graph, prmDict):
    # bonds IJ and KL connected by common bond JK
    # bonds ab and cd connected by common bond bc
    energy = 0.0
    # count = 0 
    # getting the edges which serve as the common bond
    edgeList = [e for e in list(graph.edges) if graph.degree[e[0]] > 1 and graph.degree[e[0]] > 1 ]
    # refine edgeList to remove linear centres
    edgeList = [e for e in edgeList if graph.nodes[e[0]]['ed'] > 2 and graph.nodes[e[1]]['ed'] > 2]

    for edge in edgeList:
        nodeb, nodec = edge[0], edge[1]
        Jneigh, Kneigh = list(graph.neighbors(nodeb)), list(graph.neighbors(nodec))
        Jneigh.remove(nodec), Kneigh.remove(nodeb)

        if graph.nodes[nodeb]['ed'] == 4 and graph.nodes[nodec]['ed'] == 4: # two sp3 centers
            phi0 = 60.0
            n = 3
            vi, vj = prmDict[graph.nodes[nodeb]['at']].Vi, prmDict[graph.nodes[nodec]['at']].Vi

            if load_data.get_valence(graph.nodes[nodeb]['element']) == 6: 
                # exception when both atoms are group 16

                # atom b:
                if graph.nodes[nodeb]['element'] == 'O':
                    vi = 2.0
                    n = 2
                    phi0 = 90
                elif graph.nodes[nodeb]['element'] == 'S':
                    vi = 6.8
                    n = 2
                    phi0 = 90

            if load_data.get_valence(graph.nodes[nodec]['element']) == 6:
                # atom c
                if graph.nodes[nodec]['element'] == 'O':
                    vj = 2.0
                    n = 2
                    phi0 = 90
                elif graph.nodes[nodec]['element'] == 'S':
                    vj = 6.8
                    n = 2
                    phi0 = 90
            V = 0.5 * KCAL_TO_KJ * math.sqrt(vi * vj)

        elif graph.nodes[nodeb]['ed'] == 3 and graph.nodes[nodec]['ed'] == 3: # two sp2 centers
            phi0 = 180.0
            n = 2
            torsiontype = bond_order(graph, nodeb, nodec)
            V = 0.5 * KCAL_TO_KJ * 5.0 * math.sqrt(prmDict[graph.nodes[nodeb]['at']].Uj * prmDict[graph.nodes[nodec]['at']].Uj) * (1.0 + 4.18 * math.log(torsiontype))

        elif (graph.nodes[nodeb]['ed'] == 4 and graph.nodes[nodec]['ed'] == 3) or (graph.nodes[nodeb]['ed'] == 3 and graph.nodes[nodec]['ed'] == 4): # one sp3 and one sp2 center
            phi0 = 0.0
            n = 6
            
            # exception when an atom is group 16 

            if load_data.get_valence(graph.nodes[nodeb]['element']) == 6:

                # atom b:
                if graph.nodes[nodeb]['element'] == 'O':
                    n = 2
                    phi0 = 90
                elif graph.nodes[nodeb]['element'] == 'S':
                    n = 2
                    phi0 = 90

            if load_data.get_valence(graph.nodes[nodec]['element']) == 6:
                # atom c
                if graph.nodes[nodec]['element'] == 'O':
                    n = 2
                    phi0 = 90
                elif graph.nodes[nodec]['element'] == 'S':
                    n = 2
                    phi0 = 90

            V = 0.5 * KCAL_TO_KJ * 1.0

        neighPairs = list(product(Jneigh, Kneigh))
        for pair in neighPairs:
            nodea, noded = pair[0], pair[1]

            vba = np.array(graph.nodes[nodea]['coord']) - np.array(graph.nodes[nodeb]['coord'])
            vcb = np.array(graph.nodes[nodeb]['coord']) - np.array(graph.nodes[nodec]['coord'])
            vdc = np.array(graph.nodes[nodec]['coord']) - np.array(graph.nodes[noded]['coord'])
            abbc = np.cross(vba, vcb)
            bccd = np.cross(vcb, vdc)

            dotAbbcBccd = np.dot(abbc,bccd)
            # print('dotAbbcBccd', dotAbbcBccd)
            # print('linalg.norm(abbc) * linalg.norm(bccd)', linalg.norm(abbc) * linalg.norm(bccd))
            costor = (dotAbbcBccd / (linalg.norm(abbc) * linalg.norm(bccd)))
            # print('costor', costor)
            if math.isnan(costor):
                tor = 0.001
            else:
                tor = np.arccos(costor)

            if dotAbbcBccd > 0.0:
                tor = -tor
            
            cosine = np.cos(tor * n)
            cosNPhi0 = np.cos(n * DEG_TO_RADIAN * phi0)
            inc = V * (1.0 - cosNPhi0*cosine)

            energy += inc
            # count += 1

            # print(graph.nodes[nodea]['at'], graph.nodes[nodeb]['at'], graph.nodes[nodec]['at'], graph.nodes[noded]['at'], round(V,3), round(tor * 180/math.pi,3), round(inc, 3))
    # print(count)
    return energy

def vdw_energy(graph, prmDict):
    energy = 0.0
    gdict = {n: set(graph.neighbors(n)) for n in graph.nodes}
    # print(gdict)
    pairList = combinations(list(graph.nodes), 2)
    
    count = 0
    for pair in pairList:
        
        node1, node2 = pair[0], pair[1]
        dist = miscellaneous.shortest_path_length(gdict, node1, node2)[1]
        if dist >= 3:
            print(pair, file=open('pair.txt', 'a'))
            print(dist, file=open('pair.txt', 'a'))
            count += 1
            
            Ra = prmDict[graph.nodes[node1]['at']].x1
            ka = prmDict[graph.nodes[node1]['at']].D1
            Rb = prmDict[graph.nodes[node2]['at']].x1
            kb = prmDict[graph.nodes[node2]['at']].D1

            kab = KCAL_TO_KJ * math.sqrt(ka * kb)
            kaSquared = (Ra * Rb)
            # ka now represents the xij in equation 20 -- the expected vdw distance
            ka = math.sqrt(kaSquared)

            va, vb = np.array(graph.nodes[node1]['coord']), np.array(graph.nodes[node2]['coord'])
            rabSquared = linalg.norm(va-vb)**2

            if rabSquared < 0.00001:
                rabSquared = 0.00001
            
            term6 = kaSquared / rabSquared
            term6 = term6 * term6 * term6
            term12 = term6 * term6
            inc = kab * ((term12) - (2.0 * term6))
            energy += inc

            print(graph.nodes[node1]['at'], graph.nodes[node2]['at'], round(math.sqrt(rabSquared),3), round(kab,3), round(inc,3))

    print('vdw count', count)
    return energy




            


            
    

