from openbabel import pybel
from openbabel import openbabel

pybel.ob.obErrorLog.SetOutputLevel(0)

def molecule_energy(xyz_str):
    mol = pybel.readstring('xyz', xyz_str)
    ff = openbabel.OBForceField.FindForceField("uff")
    ff.Setup(mol.OBMol)
    return ff.Energy()

def conv_graph_xyzstr(graph, comment=''):
    # atoms = len(graph.nodes)
    symbols = [graph.nodes[x]['element'] for x in graph.nodes]
    coords = [graph.nodes[x]['coord'] for x in graph.nodes]
    cblock = ['{0}  {1[0]: .10f}  {1[1]: .10f}  {1[2]: .10f}'.format(s, c) for s, c in zip(symbols, coords)]
    return str(len(symbols)) + '\n' + comment + '\n' + '\n'.join(cblock)

def dimer_comp(dname):
    return dname.split('_')[0], dname.split('_')[1]

def mbe2(monFrags, jdimerFrags, ddimerFrags, monHcaps, jdimerHcaps):
    monEnergies = {}
    jdEnergies, ddEnergies = {}, {}

    for mon in monFrags:
        mon_xyzstr = conv_graph_xyzstr(monFrags[mon])
        monEnergies[mon] = molecule_energy(mon_xyzstr)
    
    for jd in jdimerFrags:
        jd_xyzstr = conv_graph_xyzstr(jdimerFrags[jd])
        jdEnergies[jd] = molecule_energy(jd_xyzstr)
    
    for dd in ddimerFrags:
        dd_xyzstr = conv_graph_xyzstr(ddimerFrags[dd])
        ddEnergies[dd] = molecule_energy(dd_xyzstr)
    
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