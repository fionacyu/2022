import math
from itertools import combinations
from itertools import product
import numpy as np 
from numpy import linalg
from scipy import misc
import miscellaneous
import load_data
import networkx as nx
import time

# parameters
KCAL_TO_KJ = 4.184
DEG_TO_RADIAN = math.pi/180
RAD_TO_DEG = 180 / math.pi

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
    # print(node1, node2)
    # print('\t', graph.nodes[node1]['element'], graph.nodes[node2]['element'])
    # print('\t', graph.nodes[node1]['ed'], graph.nodes[node2]['ed'])
    # print('\t', graph.nodes[node1]['charge'], graph.nodes[node2]['charge'])
    ri, rj = prmDict[graph.nodes[node1]['at']].r1, prmDict[graph.nodes[node2]['at']].r1
    chiI, chiJ = prmDict[graph.nodes[node1]['at']].Xi, prmDict[graph.nodes[node2]['at']].Xi
    bondorder = bond_order(graph, node1, node2)
    rbo = -0.1332 * (ri+rj) * math.log(bondorder) # equation 3
    
    ren = ri*rj*(pow((math.sqrt(chiI) - math.sqrt(chiJ)),2.0)) / (chiI*ri + chiJ*rj) # equation 4

    r0 = ri + rj + rbo - ren # typo in original published paper
    return r0
# want to have a function to calculate each type of energy with a graph as an input 

def vectorAngle (v1, v2 ):
    dp = np.dot(v1,v2)/ (linalg.norm(v1) * linalg.norm(v2))

    if (dp < -0.999999):
        dp = -0.9999999

    if (dp > 0.9999999):
        dp = 0.9999999


    return((RAD_TO_DEG * np.arccos(dp)))

def Point2PlaneAngle(a, b, c, d):
    # a, b, c, d are np arrays with size 3
    ac = a - c
    bc = b - c
    cd = c - d

    normal = np.cross(bc, cd)
    angle = 90.0 - vectorAngle(normal, ac)
    return angle


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

def bond_edges(graph, edgeList, prmDict):
    energy = 0.0
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
    for node in nodeList: # node is the vertex
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

def angleE(graph, cnode, onode1, onode2, prmDict):
    theta0 = prmDict[graph.nodes[cnode]['at']].theta0 * DEG_TO_RADIAN
    cosT0 = np.cos(theta0)
    sinT0 = np.sin(theta0)
    c2 = 1.0 / (4.0 * sinT0 * sinT0)
    c1 = -4.0 * c2 * cosT0
    c0 = c2*(2.0*cosT0*cosT0 + 1.0)

    a, b, c = np.array(graph.nodes[cnode]['coord']), np.array(graph.nodes[onode1]['coord']), np.array(graph.nodes[onode2]['coord'])
    rab = bond_distance(prmDict, graph, cnode, onode1)
    rac = bond_distance(prmDict, graph, cnode, onode2)
    rbc = math.sqrt(rab*rab + rac*rac - 2.0 * rab*rac*cosT0)
    ab = b - a
    ac = c - a
    cosT = np.dot(ab, ac) / (linalg.norm(ab) * linalg.norm(ac))
    theta = np.arccos(cosT) # angle in radians

    # force constant
    ka = (664.12 * KCAL_TO_KJ) * (prmDict[graph.nodes[onode1]['at']].Z1 * prmDict[graph.nodes[onode2]['at']].Z1 / (pow(rbc, 5.0)))
    ka *= (3.0 * rab * rac * (1.0 - cosT0 * cosT0) - rbc * rbc * cosT0)
    # print('ka', ka)
    # value for n 
    if graph.nodes[cnode]['ed'] == 2: # linear case
        inc = ka * (1 + cosT) # typo in Rappe paper, not 1 - cosT src: openbabel
    elif graph.nodes[cnode]['ed'] == 3: # sp2
        n = 3
        ka /= 9
        # a penalty for angles close to zero is added, based on ESFF
        inc = ka * (1 - np.cos(n*theta))# + math.exp(-20.0*(theta - theta0 + 0.25))
    elif graph.nodes[cnode]['ed'] == 4:
        inc = ka*(c0 + c1*cosT + c2*(2.0*cosT*cosT - 1.0))
    
    return inc

def angleE_edge(graph, edgeList, prmDict):
    energy = 0.0
    for edge in edgeList:
        node1, node2 = edge[0], edge[1]
        n1neigh = list(graph.neighbors(node1))
        n1neigh.remove(node2)
        n2neigh = list(graph.neighbors(node2))
        n2neigh.remove(node1)

        if n1neigh:
            for n1 in n1neigh:
                energy += angleE(graph, node1, node2, n1, prmDict)
        
        if n2neigh:
            for n2 in n2neigh:
                energy += angleE(graph, node2, node1, n2, prmDict)
    return energy

def angleE_edge2(graph, edgeList, prmDict, angList):
    energy = 0.0
    for edge in edgeList:
        node1, node2 = edge[0], edge[1]
        n1neigh = list(graph.neighbors(node1))
        n1neigh.remove(node2)
        n2neigh = list(graph.neighbors(node2))
        n2neigh.remove(node1)

        if n1neigh:
            for n1 in n1neigh:
                if frozenset([node1, node2, n1]) in angList:
                    continue
                else:
                    angList.add(frozenset([node1, node2, n1]))
                    energy += angleE(graph, node1, node2, n1, prmDict)
        
        if n2neigh:
            for n2 in n2neigh:
                if frozenset([node2, node1, n2]) in angList:
                    continue
                else:
                    angList.add(frozenset([node2, node1, n2]))
                    energy += angleE(graph, node2, node1, n2, prmDict)
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
            
            # exception when an sp3 atom is group 16 
            blist = [graph.nodes[nodeb]['ed'] == 4, graph.nodes[nodec]['ed'] == 4]
            sp3node = [e for i,e in enumerate(edge) if blist[i]][0] # obtain atom that is the sp3 center

            if load_data.get_valence(graph.nodes[sp3node]['element']) == 6:

                if graph.nodes[sp3node]['element'] == 'O' or graph.nodes[sp3node]['element'] == 'S':
                    n = 2
                    phi0 = 90

            V = 0.5 * KCAL_TO_KJ * 1.0

        neighPairs = list(product(Jneigh, Kneigh))
        neighPairs = [x for x in neighPairs if x[1] != x[0]] # cases like cyclopropane, ensures we are dealing with 4 distinct atoms
        for pair in neighPairs:
            nodea, noded = pair[0], pair[1]

            vba = np.array(graph.nodes[nodea]['coord']) - np.array(graph.nodes[nodeb]['coord'])
            vcb = np.array(graph.nodes[nodeb]['coord']) - np.array(graph.nodes[nodec]['coord'])
            vdc = np.array(graph.nodes[nodec]['coord']) - np.array(graph.nodes[noded]['coord'])
            abbc = np.cross(vba, vcb)
            bccd = np.cross(vcb, vdc)

            dotAbbcBccd = np.dot(abbc,bccd)

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

def torE(graph, nodea, nodeb, nodec, noded, prmDict):
    # print(nodeb, nodec)
    # print('\t', graph.nodes[nodeb]['ed'], graph.nodes[nodec]['ed'])
    # print('\t', graph.nodes[nodeb]['element'], graph.nodes[nodec]['element'])
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
        
        # exception when an sp3 atom is group 16 
        blist = [graph.nodes[nodeb]['ed'] == 4, graph.nodes[nodec]['ed'] == 4]
        sp3node = [e for i,e in enumerate([nodeb, nodec]) if blist[i]][0] # obtain atom that is the sp3 center

        if load_data.get_valence(graph.nodes[sp3node]['element']) == 6:

            if graph.nodes[sp3node]['element'] == 'O' or graph.nodes[sp3node]['element'] == 'S':
                n = 2
                phi0 = 90

        V = 0.5 * KCAL_TO_KJ * 1.0
    
    else:
        return 0

    vba = np.array(graph.nodes[nodea]['coord']) - np.array(graph.nodes[nodeb]['coord'])
    vcb = np.array(graph.nodes[nodeb]['coord']) - np.array(graph.nodes[nodec]['coord'])
    vdc = np.array(graph.nodes[nodec]['coord']) - np.array(graph.nodes[noded]['coord'])
    abbc = np.cross(vba, vcb)
    bccd = np.cross(vcb, vdc)

    dotAbbcBccd = np.dot(abbc,bccd)

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

    return inc


def torE_edge(graph, edgeList, prmDict):
    energy = 0.0
    for edge in edgeList:
        # torsional: bond ab is connectd to bond cd by bond bc 
        # consider three cases: 1. edge is bond bc, 2. edge is bond ab, 3. edge is cd

        node1, node2 = edge[0], edge[1]
        # case 1:
        n1neigh = list(graph.neighbors(node1))
        n1neigh.remove(node2)
        n2neigh = list(graph.neighbors(node2))
        n2neigh.remove(node1)

        if n1neigh and n2neigh:
            pairList = product(n1neigh, n2neigh)
            pairList = [x for x in pairList if x[0] != x[1]] # cases like cyclopropane
            for pair in pairList:
                nodea = pair[0]
                nodeb = node1
                nodec = node2
                noded = pair[1]
                energy += torE(graph, nodea, nodeb, nodec, noded, prmDict)
        
        # case 2:
        if n2neigh:
            nodea = node1
            nodeb = node2
            for n2 in n2neigh:
                nodec = n2
                ncneigh = list(graph.neighbors(nodec))
                ncneigh = [x for x in ncneigh if x not in edge] # cases like cyclopropane
                for nc in ncneigh:
                    noded = nc
                    energy += torE(graph, nodea, nodeb, nodec, noded, prmDict)
        
        # case 3:
        if n1neigh:
            noded = node2
            nodec = node1
            for n1 in n1neigh:
                nodeb = n1
                nbneigh = list(graph.neighbors(nodeb))
                nbneigh = [x for x in nbneigh if x not in edge] # cases like cyclopropane
                for nb in nbneigh:
                    nodea = nb
                    energy += torE(graph, nodea, nodeb, nodec, noded, prmDict)
    
    return energy
    


def torE_edge2(graph, edgeList, prmDict, torList):
    energy = 0.0
    for edge in edgeList:
        # torsional: bond ab is connectd to bond cd by bond bc 
        # consider three cases: 1. edge is bond bc, 2. edge is bond ab, 3. edge is cd

        node1, node2 = edge[0], edge[1]
        # case 1:
        n1neigh = list(graph.neighbors(node1))
        n1neigh.remove(node2)
        n2neigh = list(graph.neighbors(node2))
        n2neigh.remove(node1)

        if n1neigh and n2neigh:
            pairList = product(n1neigh, n2neigh)
            pairList = [x for x in pairList if x[0] != x[1]] # cases like cyclopropane
            for pair in pairList:
                nodea = pair[0]
                nodeb = node1
                nodec = node2
                noded = pair[1]
                if frozenset([nodea, nodeb, nodec, noded]) in torList:
                    continue
                else:
                    torList.add(frozenset([nodea, nodeb, nodec, noded]))
                    energy += torE(graph, nodea, nodeb, nodec, noded, prmDict)
        
        # case 2:
        if n2neigh:
            nodea = node1
            nodeb = node2
            for n2 in n2neigh:
                nodec = n2
                ncneigh = list(graph.neighbors(nodec))
                ncneigh = [x for x in ncneigh if x not in edge] # cases like cyclopropane
                for nc in ncneigh:
                    noded = nc
                    if frozenset([nodea, nodeb, nodec, noded]) in torList:
                        continue
                    else:
                        torList.add(frozenset([nodea, nodeb, nodec, noded]))
                        energy += torE(graph, nodea, nodeb, nodec, noded, prmDict)
        
        # case 3:
        if n1neigh:
            noded = node2
            nodec = node1
            for n1 in n1neigh:
                nodeb = n1
                nbneigh = list(graph.neighbors(nodeb))
                nbneigh = [x for x in nbneigh if x not in edge] # cases like cyclopropane
                for nb in nbneigh:
                    nodea = nb
                    if frozenset([nodea, nodeb, nodec, noded]) in torList:
                        continue
                    else:
                        torList.add(frozenset([nodea, nodeb, nodec, noded]))
                        energy += torE(graph, nodea, nodeb, nodec, noded, prmDict)
    
    return energy

def vdw_energy(graph, prmDict):
    energy = 0.0
    gdict = {n: set(graph.neighbors(n)) for n in graph.nodes}
    # print(gdict)
    pairList = combinations(list(graph.nodes), 2)
    
    # count = 0
    for pair in pairList:
        
        node1, node2 = pair[0], pair[1]
        dist = miscellaneous.shortest_path_length(gdict, node1, node2)[1]
        # r = linalg.norm(np.array(graph.nodes[node1]['coord']) - np.array(graph.nodes[node2][]))
        if dist >= 3 or dist == 0:
            # print(pair, file=open('pair.txt', 'a'))
            # print(dist, file=open('pair.txt', 'a'))
            # count += 1
            
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

            # print(graph.nodes[node1]['at'], graph.nodes[node2]['at'], round(math.sqrt(rabSquared),3), round(kab,3), round(inc,3))

    # print('vdw count', count)
    return energy

def vdw_pair(graph, pairList, prmDict):
    energy = 0.0
    for pair in pairList:
        node1, node2 = pair[0], pair[1]
        # dist = miscellaneous.shortest_path_length(gdict, node1, node2)[1]
        # # r = linalg.norm(np.array(graph.nodes[node1]['coord']) - np.array(graph.nodes[node2][]))
        # if dist >= 3 or dist == 0:
            # print(pair, file=open('pair.txt', 'a'))
            # print(dist, file=open('pair.txt', 'a'))
            # count += 1
        
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

            # print(graph.nodes[node1]['at'], graph.nodes[node2]['at'], round(math.sqrt(rabSquared),3), round(kab,3), round(inc,3))

    # print('vdw count', count)
    return energy

def dimer_comp(dname):
    return dname.split('_')[0], dname.split('_')[1]


def total_energy(graph):
    prmDict = load_data.read_prm()
    energy = 0.0
    t2 = time.process_time()
    energy += bond_energy(graph, prmDict)
    # print('bond time', time.process_time() - t2)
    t3 = time.process_time()
    energy += angle_energy(graph, prmDict)
    # print('angle time', time.process_time() - t3)
    t4  = time.process_time()
    energy += torsional_energy(graph, prmDict)
    # print('tor time', time.process_time() - t4)
    t1 = time.process_time()
    energy += vdw_energy(graph, prmDict)
    # print('vdw time', time.process_time() - t1)
    return energy



def mbe2_eff(monFrags, monHcaps, jdimerFrags, jdHcaps, jdimerEdges, prmDict):

    monEnergy = 0.0
    monE = {}
    for mon in monFrags:
        _monE = total_energy(monFrags[mon])
        monE[mon] = _monE
        monEnergy += _monE

    # getting disjoint dimers
    monKeys = list(monFrags)
    monPairs = combinations(monKeys, 2)
    ddList = []
    for pair in monPairs:
        if "%s_%s" % (pair[0], pair[1]) in jdimerFrags or "%s_%s" % (pair[1], pair[0]) in jdimerFrags:
            continue
        else:
            ddList.append("%s_%s" % (pair[0], pair[1]))

    # sum of disjoint dimer interaction energies
    ddimerEnergy = 0.0
    # pair of atoms on separate fragments
    for dd in ddList:
        mon1, mon2 = dimer_comp(dd)
        datomPairs = list(product(list(monFrags[mon1].nodes), list(monFrags[mon2].nodes)))
        ddg =  nx.compose(monFrags[mon1],monFrags[mon2])
        # print(dd, vdw_pair(ddg, datomPairs, prmDict))
        ddimerEnergy += vdw_pair(ddg, datomPairs, prmDict)
    
    jdimerEnergy = 0.0 
    for jd in list(jdimerFrags):
        # print(jd)
        mon1, mon2 = dimer_comp(jd)
        edgeList = jdimerEdges[jd] # bonds belonging on the original graph that were broken
        # print("\t", 'edgeList', edgeList)
        jdg = jdimerFrags[jd]
        bondE = bond_edges(jdg, edgeList, prmDict)

        angE = angleE_edge(jdg, edgeList, prmDict)

        torE = torE_edge(jdg, edgeList, prmDict)

        # hcapsM = (set(monFrags[mon1].nodes) | set(monFrags[mon2].nodes)) - set(jdg.nodes)
        monBondE = 0.0
        hcapList, nonHnodes = [], []
        nonHnodes2 = []
        sbond, sangle, stor = 0.0, 0.0, 0.0
        monjdH  = (set(monFrags[mon1].nodes) | set(monFrags[mon2].nodes)) - set(jdg.nodes)
        # print('\t', 'monjdH', monjdH)
        jdmonH = set(jdg.nodes) - (set(monFrags[mon1].nodes) | set(monFrags[mon2].nodes))
        # print('\t', 'jdmonH', jdmonH)
        mon2g = nx.compose(monFrags[mon1], monFrags[mon2])
        sameH = set()
        hmonjd = {}
        for h in [i for i in monjdH]:
            blist = [np.array_equal(np.array(mon2g.nodes[h]['coord']), np.array(jdg.nodes[x]['coord'])) for x in jdmonH]
            if any(blist):
                monjdH.remove(h)
                
                bidx = np.where(blist)[0]
                sameH.add(list(jdmonH)[bidx[0]])
                hmonjd[h] = list(jdmonH)[bidx[0]]

        # print('\t', 'relevant hcaps', monjdH)
        # print('\t', 'sameH', sameH)
        relevant_hcaps = monjdH
        for _mon in [mon1, mon2]:
            # hcaps = set(monFrags[_mon].nodes) - set(jdg.nodes)
            hcaps = relevant_hcaps & set(monFrags[_mon].nodes)
            # print("\t", hcaps)
            hcapList.append(list(hcaps))
            nonHnodes.append(list(set(monFrags[_mon].nodes) - set(hcaps)))

            _nonHnodes2 = []
            for x in list(set(monFrags[_mon].nodes) - set(hcaps)):
                if x in hmonjd:
                    _nonHnodes2.append(hmonjd[x])
                else:
                    _nonHnodes2.append(x)
            nonHnodes2.append(_nonHnodes2)

            hedgeList = [e for e in monFrags[_mon].edges if any([e[0] in hcaps, e[1] in hcaps]) and any([e[0] in miscellaneous.flatten(edgeList), e[1] in miscellaneous.flatten(edgeList)])]
            bondEM = bond_edges(monFrags[_mon], hedgeList, prmDict)
            sbond += bondEM
            monBondE += bondEM
            angEM = angleE_edge(monFrags[_mon], hedgeList, prmDict)
            monBondE += angEM
            sangle += angEM
            torEM = torE_edge(monFrags[_mon], hedgeList, prmDict)
            monBondE += torEM
            stor += torEM
        
        deltaBondE = bondE + angE + torE - monBondE
        
        # print('\t', 'deltabond', bondE - sbond)
        # print('\t', 'deltaangle', angE - sangle)
        # print('\t', 'deltator', torE - stor)
        # jdimerEnergy += deltaBondE


        ijpairs = product(nonHnodes2[0], nonHnodes2[1])
        jdgdict = {n: set(jdg.neighbors(n)) for n in jdg.nodes}
        ijpairs = [x for x in ijpairs if miscellaneous.shortest_path_length(jdgdict, x[0], x[1])[1] >= 3 ]
        Eijvdw = vdw_pair(jdg, ijpairs, prmDict)
        
        iHpairs = product(nonHnodes[0], hcapList[0])
        # iHpairs = list(iHpairs) + list(combinations(hcapList[0], 2))
        gdict1 = {n: set(monFrags[mon1].neighbors(n)) for n in monFrags[mon1].nodes}
        iHpairs = [x for x in iHpairs if miscellaneous.shortest_path_length(gdict1, x[0], x[1])[1] >= 3]
        
        Eivdw = vdw_pair(monFrags[mon1], iHpairs, prmDict)

        jHpairs = product(nonHnodes[1], hcapList[1])
        # jHpairs = list(jHpairs) + list(combinations(hcapList[1], 2))
        gdict2 = {n: set(monFrags[mon2].neighbors(n)) for n in monFrags[mon2].nodes}
        jHpairs = [x for x in jHpairs if miscellaneous.shortest_path_length(gdict2, x[0], x[1])[1] >= 3]
        
        Ejvdw = vdw_pair(monFrags[mon2], jHpairs, prmDict)

        deltaVDW = Eijvdw - Eivdw - Ejvdw
        # print('\t', 'Eijvdw', Eijvdw)
        # print('\t', 'Eivdw', Eivdw)
        # print('\t', 'Ejvdw', Ejvdw)
        # print('\t', 'deltavdw', deltaVDW)
        jdimerEnergy += (deltaBondE + deltaVDW)
    
    # print( round((monEnergy + ddimerEnergy + jdimerEnergy), 4))
    return round((monEnergy + ddimerEnergy + jdimerEnergy), 4)


def peff_mbe2(graph, edges_to_cut_list, monFrags, monHcaps, jdimerFrags, jdimerHcaps, jdimerEdges, prmDict):
    # the penalty is equal to E - E(MBE2)
    # E is the energy of the entire molecule, and E(MBE2) = \sum_i E_i + \sum_ij (E_ij - E_i - E_j)
    # penalty is calculated in two parts: 1. E - \sum_i E_i 2. \sum_ij (E_ij - E_i - E_j)
    # then subtract 2. from 1. and the penalty is given 

    monKeys = list(monFrags)

    # part 1.: monomers
    # the original molecule: bonding terms only
    Eog = 0.0
    angList, torList = set(), set()
    Eogbond = bond_edges(graph, edges_to_cut_list, prmDict)
    Eog += Eogbond
    Eogangle = angleE_edge2(graph, edges_to_cut_list, prmDict, angList)
    Eog += Eogangle
    Eogtor = torE_edge2(graph, edges_to_cut_list, prmDict, torList)
    Eog += Eogtor

    # the monomers: bonding terms only (only need to account for the hydrogen caps)
    Emon = 0.0
    EIvdw = 0.0
    # Embond, Emang, Emtor = 0.0, 0.0, 0.0
    nonHcapnodes = []
    for k in monKeys:
        angList2, torList2 = set(), set()
        hcaps = set(monFrags[k].nodes) - set(graph.nodes)
        nonHcaps = list(set(monFrags[k].nodes) - hcaps)
        nonHcapnodes.append(nonHcaps)
        hedgeList = [e for e in monFrags[k].edges if any([e[0] in hcaps, e[1] in hcaps])]
        Emon += bond_edges(monFrags[k], hedgeList, prmDict)
        # Embond += bond_edges(monFrags[k], hedgeList, prmDict)
        Emon += angleE_edge2(monFrags[k], hedgeList, prmDict, angList2)
        # Emang += angleE_edge2(monFrags[k], hedgeList, prmDict, angList2)
        Emon += torE_edge2(monFrags[k], hedgeList, prmDict, torList2) 
        # Emtor += torE_edge2(monFrags[k], hedgeList, prmDict, torList2) 

        IHpairs = list(product(nonHcaps, hcaps)) + list(combinations(hcaps, 2))
        mondict = {n: set(monFrags[k].neighbors(n)) for n in monFrags[k].nodes}
        IHpairs = [x for x in IHpairs if miscellaneous.shortest_path_length(mondict, x[0], x[1])[1] >= 3]
        EIvdw += vdw_pair(monFrags[k], IHpairs, prmDict)
    
    # dealing with van der waals interactions
    # deltavdw = E^vdw - \sum_i E_i^vdw
    # deltavdw = interaction between pair of atoms on distinct fragments (excluding hydrogen caps) - \sum^Nfrags interactin of hydrogen caps to other atoms in the same fragment 

    # pairs on distinct fragments:
    IJpairs = list(miscellaneous.pairs(*nonHcapnodes))
    gdict = {n: set(graph.neighbors(n)) for n in graph.nodes}
    IJpairs = [x for x in IJpairs if miscellaneous.shortest_path_length(gdict, x[0], x[1])[1] >= 3 ]
    EIJvdw = vdw_pair(graph, IJpairs, prmDict)

    deltavdw = EIJvdw - EIvdw 
    deltaEmon = (Eog - Emon) + deltavdw
    print('\t', 'deltabond', (Eog - Emon))
    # print('\t', 'deltabond', (Eogbond - Embond))
    # print('\t', 'deltaangle', (Eogangle - Emang))
    # print('\t', 'deltator', (Eogtor - Emtor))
    print('\t', 'deltavdw', deltavdw)

    monPairs = combinations(monKeys, 2)
    ddList = []
    for pair in monPairs:
        if "%s_%s" % (pair[0], pair[1]) in jdimerFrags or "%s_%s" % (pair[1], pair[0]) in jdimerFrags:
            continue
        else:
            ddList.append("%s_%s" % (pair[0], pair[1]))

    # sum of disjoint dimer interaction energies
    ddimerEnergy = 0.0
    # pair of atoms on separate fragments
    for dd in ddList:
        mon1, mon2 = dimer_comp(dd)
        datomPairs = list(product(list(monFrags[mon1].nodes), list(monFrags[mon2].nodes)))
        ddg =  nx.compose(monFrags[mon1],monFrags[mon2])
        # print(dd, vdw_pair(ddg, datomPairs, prmDict))
        ddimerEnergy += vdw_pair(ddg, datomPairs, prmDict)
    
    jdimerEnergy = 0.0 
    for jd in list(jdimerFrags):
        # print(jd)
        mon1, mon2 = dimer_comp(jd)
        edgeList = jdimerEdges[jd] # bonds belonging on the original graph that were broken
        # print("\t", 'edgeList', edgeList)
        jdg = jdimerFrags[jd]
        bondE = bond_edges(jdg, edgeList, prmDict)

        angE = angleE_edge(jdg, edgeList, prmDict)

        torE = torE_edge(jdg, edgeList, prmDict)

        # hcapsM = (set(monFrags[mon1].nodes) | set(monFrags[mon2].nodes)) - set(jdg.nodes)
        monBondE = 0.0
        hcapList, nonHnodes = [], []
        nonHnodes2 = []
        sbond, sangle, stor = 0.0, 0.0, 0.0
        monjdH  = (set(monFrags[mon1].nodes) | set(monFrags[mon2].nodes)) - set(jdg.nodes)
        # print('\t', 'monjdH', monjdH)
        jdmonH = set(jdg.nodes) - (set(monFrags[mon1].nodes) | set(monFrags[mon2].nodes))
        # print('\t', 'jdmonH', jdmonH)
        mon2g = nx.compose(monFrags[mon1], monFrags[mon2])
        sameH = set()
        hmonjd = {}
        for h in [i for i in monjdH]:
            blist = [np.array_equal(np.array(mon2g.nodes[h]['coord']), np.array(jdg.nodes[x]['coord'])) for x in jdmonH]
            if any(blist):
                monjdH.remove(h)
                
                bidx = np.where(blist)[0]
                sameH.add(list(jdmonH)[bidx[0]])
                hmonjd[h] = list(jdmonH)[bidx[0]]

        # print('\t', 'relevant hcaps', monjdH)
        # print('\t', 'sameH', sameH)
        relevant_hcaps = monjdH
        for _mon in [mon1, mon2]:
            # hcaps = set(monFrags[_mon].nodes) - set(jdg.nodes)
            hcaps = relevant_hcaps & set(monFrags[_mon].nodes)
            # print("\t", hcaps)
            hcapList.append(list(hcaps))
            nonHnodes.append(list(set(monFrags[_mon].nodes) - set(hcaps)))

            _nonHnodes2 = []
            for x in list(set(monFrags[_mon].nodes) - set(hcaps)):
                if x in hmonjd:
                    _nonHnodes2.append(hmonjd[x])
                else:
                    _nonHnodes2.append(x)
            nonHnodes2.append(_nonHnodes2)

            hedgeList = [e for e in monFrags[_mon].edges if any([e[0] in hcaps, e[1] in hcaps]) and any([e[0] in miscellaneous.flatten(edgeList), e[1] in miscellaneous.flatten(edgeList)])]
            bondEM = bond_edges(monFrags[_mon], hedgeList, prmDict)
            sbond += bondEM
            monBondE += bondEM
            angEM = angleE_edge(monFrags[_mon], hedgeList, prmDict)
            monBondE += angEM
            sangle += angEM
            torEM = torE_edge(monFrags[_mon], hedgeList, prmDict)
            monBondE += torEM
            stor += torEM
        
        deltaBondE = bondE + angE + torE - monBondE
        
        # print('\t', 'deltabond', bondE - sbond)
        # print('\t', 'deltaangle', angE - sangle)
        # print('\t', 'deltator', torE - stor)
        # jdimerEnergy += deltaBondE


        ijpairs = product(nonHnodes2[0], nonHnodes2[1])
        jdgdict = {n: set(jdg.neighbors(n)) for n in jdg.nodes}
        ijpairs = [x for x in ijpairs if miscellaneous.shortest_path_length(jdgdict, x[0], x[1])[1] >= 3 ]
        Eijvdw = vdw_pair(jdg, ijpairs, prmDict)
        
        iHpairs = product(nonHnodes[0], hcapList[0])
        # iHpairs = list(iHpairs) + list(combinations(hcapList[0], 2))
        gdict1 = {n: set(monFrags[mon1].neighbors(n)) for n in monFrags[mon1].nodes}
        iHpairs = [x for x in iHpairs if miscellaneous.shortest_path_length(gdict1, x[0], x[1])[1] >= 3]
        
        Eivdw = vdw_pair(monFrags[mon1], iHpairs, prmDict)

        jHpairs = product(nonHnodes[1], hcapList[1])
        # jHpairs = list(jHpairs) + list(combinations(hcapList[1], 2))
        gdict2 = {n: set(monFrags[mon2].neighbors(n)) for n in monFrags[mon2].nodes}
        jHpairs = [x for x in jHpairs if miscellaneous.shortest_path_length(gdict2, x[0], x[1])[1] >= 3]
        
        Ejvdw = vdw_pair(monFrags[mon2], jHpairs, prmDict)

        deltaVDW = Eijvdw - Eivdw - Ejvdw
        # print('\t', 'Eijvdw', Eijvdw)
        # print('\t', 'Eivdw', Eivdw)
        # print('\t', 'Ejvdw', Ejvdw)
        # print('\t', 'deltavdw', deltaVDW)
        jdimerEnergy += (deltaBondE + deltaVDW)
    return round((deltaEmon - ddimerEnergy - jdimerEnergy), 4)


# oop bending not used 
# def oop_energy(graph): # out of plane bending energy
#     energy = 0.0

#     validAtoms = {'C', 'O', 'N', 'P'}
#     nodeList = [x for x in list(graph.nodes) if graph.degree[x] == 3 and graph.nodes[x]['element'] in validAtoms] # only concerns atoms that are bonded to exactly three other atoms

#     # node I is connected to atoms J K and L
#     for node in nodeList:
#         neighList = list(graph.neighbors(node))

#         at = graph.nodes[node]['at']

#         at1 = {'N_3', 'N_2', 'N_R', 'O_2', 'O_R'}
#         if at in at1:
#             c0 = 1.0
#             c1 = -1.0
#             c2 = 0.0
#             koop = 6.0 * KCAL_TO_KJ
#         if at == 'C_2' or at == 'C_R':
#             c0 = 1.0
#             c1 = -1.0
#             c2 = 0.0
#             koop = 6.0 * KCAL_TO_KJ

#             neighat = [graph.nodes[x]['at'] for x in neighList]
#             if 'O_2' in neighat:
#                 koop = 50.0 * KCAL_TO_KJ
        
#         koop /= 3.0 # three inversion centers

#         b = np.array(graph.nodes[node]['coord'])
#         a, c, d =  np.array(graph.nodes[neighList[0]]['coord']), np.array(graph.nodes[neighList[1]]['coord']), np.array(graph.nodes[neighList[2]]['coord'])
#         angle = DEG_TO_RADIAN * Point2PlaneAngle(d, a, b, c)

#         inc = koop * (c0 + c1 * np.cos(angle) + c2 * np.cos(2.0*angle))
#         energy += inc
        
#         print(graph.nodes[neighList[2]]['at'], graph.nodes[node]['at'], graph.nodes[neighList[0]]['at'], graph.nodes[neighList[1]]['at'], round(angle * RAD_TO_DEG, 3), round(koop, 3), round(inc, 3))
#     return energy


def mbe2(monFrag, jdFrag, ddFrag, monHcaps, jdimerHcaps):
    monEnergies = {}
    jdEnergies, ddEnergies = {}, {}

    t1 = time.process_time()
    for mon in monFrag:
        monEnergies[mon] = total_energy(monFrag[mon])
        # print(mon, total_energy(monFrag[mon]))
    # print('mon energy time', time.process_time() - t1)
    t2 = time.process_time()
    for jd in jdFrag:
        jdEnergies[jd] = total_energy(jdFrag[jd])
        # print(jd, total_energy(jdFrag[jd]))
    # print('jd energy time', time.process_time() - t2)
    t3 = time.process_time()
    for dd in ddFrag:
        ddEnergies[dd] = total_energy(ddFrag[dd])
        # print(dd, total_energy(ddFrag[dd]))
    # print('dd energy time', time.process_time() - t3)
    sumMonEnergiesH = sum(monEnergies[x] for x in monEnergies)
    sumMonEnergies = sumMonEnergiesH - 0.5 *2625.5 * sum([v for _, v in monHcaps.items()])

    dimerEnergies = {}
    dimerEnergies.update(jdEnergies)
    dimerEnergies.update(ddEnergies)
    intEnergy = 0
    for dimer in dimerEnergies:
        mon1, mon2 = dimer_comp(dimer)
        if dimer in jdEnergies:
            # minus energy of hydrogen = 0.5 hartrees 
            inc = (dimerEnergies[dimer] - 2625.5 * 0.5 * int(jdimerHcaps[dimer])) - (monEnergies[mon1] - 2625.5 * 0.5 * int(monHcaps[mon1])) - (monEnergies[mon2] - 2625.5 * 0.5 * int(monHcaps[mon2]))
        else:
            # with disjoint dimers, the hydrogens cancel out 
            inc = (dimerEnergies[dimer]) - (monEnergies[mon1]) - (monEnergies[mon2])
        
        intEnergy += inc
    
    mbe2 = sumMonEnergies + intEnergy
    return mbe2
    
    

            


            
    

