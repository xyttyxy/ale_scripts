import os
import gc
import numpy as np
import random, string
import pytim
import MDAnalysis as mda
from tqdm import tqdm
from scipy.signal import find_peaks
from ase.io import read, write
from fast_lammpsdump import read_dump
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def get_thicknesses(steps_ctrs, steps_brkt, method):
    """ Wrapper around get_thickness used in stitch_runs
    Note: retrieves structures (ase.Atoms) from 2 possibilities
    1. if db file exists, read sql db. This is the fastest. 
    2. if db file does not exist, read dumpfile. Also fast enough.
    
    so make sure you do not have broken/incomplete db files named run_*.db in the working folder.
    """
    
    thicknesses = []
    nos = []
    atomss = []
    j = 0
    def foo(s):
        r = np.where(s >= steps_brkt)[0][-1]+1
        if r == len(steps_brkt):
            r -= 1
            
        if os.path.exists('dump.{:}.thickness'.format(r)):
            nparr = np.loadtxt('dump.{:}.thickness'.format(r))
            print(s, np.where(nparr[:, 0].astype(np.int) == s))
            row = nparr[np.where(nparr[:, 0].astype(np.int) == s)[0][0], :]
            t = row[1]
            no = row[2]
        else:
            if os.path.exists('run_{:}.db'.format(r)):
                atoms = read('run_{:}.db@timestep={:}'.format(r, s))[0]
            else:
                atoms = read_dump('dump.{:}.high_freq'.format(r), s)
                
            if len([a for a in atoms if a.symbol == 'O']) == 0:
                t = 0
                no = 0
            else:
                t, no = get_thickness(atoms, method)
        return t, no
    
    thicknesses, nos = zip(*Parallel(n_jobs=8)(delayed(foo)(s) for s in tqdm(steps_ctrs)))

    return np.array(thicknesses), np.array(nos)


def get_all_thicknesses(filename, byteidx):

    
    thicknesses = np.empty(1)
    num_o = np.empty(1)
    
    for step in tqdm(byteidx[:, 1]):
        atoms = read_dump(filename, step)
        t, no = get_thickness(atoms, 3)
        thicknesses=np.append(thicknesses, t)
        num_o=np.append(num_o, no)

    return np.vstack((byteidx[:, 1], thicknesses[1:], num_o[1:]))
    

def get_all_thicknesses_parallel(filename, byteidx, ncore=None, batch=None):


    def foo(s):
        atoms = read_dump(filename, s)
        t, no = get_thickness(atoms, 4)
        return t, no
    if ncore:
        cpus = ncore
    else:
        cpus = os.cpu_count()
    
    thickness, num_o = zip(*Parallel(n_jobs=cpus)(delayed(foo)(s) for s in tqdm(byteidx[batch[0]:batch[1], 1])))
    return np.vstack((byteidx[batch[0]:batch[1], 1], thickness, num_o))

        
def get_thickness(atoms, method=1, debug=False, raw_coords=False):
    """ Calculates the thickness of the oxide

    method: selects definition of thickness
       1: z of highest cu (excluding sputtered) - z of lowest o
    Note:
    this code tries to reduce the noise due to
    1) few O atoms penetrating deep into bulk Cu, and
    2) few Cu atoms protruding from the oxide surface

       2: iteratively check the cu-cu rdf for significant peaks
    corresponding to bulk Cu

    raw_coords returns the z coordinates of max_cu and z_bottom
    """

    z = atoms.get_positions()[:, 2]
    is_o = atoms.symbols == 'O'

    min_O = min(z[is_o])
    # method 1
    # highest cu z coords
    high_Cu = np.sort(z[~is_o])[-100:]
    thresh = np.mean(high_Cu) + 2 * np.std(high_Cu)
    high_Cu = high_Cu[high_Cu < thresh]
    max_Cu = max(high_Cu)

    # method 2
    rMax = 10
    nBins = 500
    x = np.arange(nBins) * rMax / nBins

    z_bottom = min_O
    if method == 2:
        z_bounds = np.linspace(max_Cu-5, min_O-20, 20)

        j = 0
        atoms_view = []
        for z in z_bounds:
            atoms_rdf = Atoms([a for a in atoms if a.position[2] > z],
                              cell=atoms.cell.cellpar()[0:3])
            atoms_view.append(atoms_rdf)
            RDFobj = RDF(atoms_rdf, rMax, nBins)

            # check for cu-cu features
            rdf = RDFobj.get_rdf(elements=(29, 29))
            # rdf is the signal
            peaks, properties = find_peaks(rdf,
                                           prominence=10,
                                           height=20,
                                           distance=25,
                                           width=2)
            is_bottom = np.any(np.logical_and(x[peaks] < 2.7, x[peaks] > 2.5))
            if debug:
                plt.plot(rdf+j, label=j+1)
                plt.plot(peaks, rdf[peaks]+j, "x")
#            breakpoint()
            if is_bottom:
                # found bulk Cu features! stop here
                z_bottom = z
                if z_bottom - min_O > 5:

                    breakpoint(context=20)

                print(z_bottom, min_O)
                break

            j += 1

        if debug:
            plt.show()
    elif method == 3:
        positions = atoms.get_positions()
        num_divs = 3
        min_Os = np.zeros((num_divs, num_divs))
        max_Cus = np.zeros((num_divs, num_divs))
        cell = atoms.cell.cellpar()[0:3]

        x_step = cell[0] / num_divs
        y_step = cell[1] / num_divs
        for ix in np.arange(0, num_divs):
            for iy in np.arange(0, num_divs):
                min_Os[ix][iy] = positions[(positions[:,0] > ix*x_step)
                                           & (positions[:,0] < (ix+1)*x_step)
                                           & (positions[:,1] > iy*y_step)
                                           & (positions[:,1] < (iy+1)*y_step)
                                           & is_o, 2].min()
                max_Cus[ix][iy] = positions[(positions[:,0] > ix*x_step)
                                            & (positions[:,0] < (ix+1)*x_step)
                                            & (positions[:,1] > iy*y_step)
                                            & (positions[:,1] < (iy+1)*y_step)
                                            & ~is_o, 2].max()

        # remove highest and lowest
        min_Os = np.sort(min_Os.flatten())[1:-1]
        max_Cus = np.sort(max_Cus.flatten())[1:-1]

        max_Cu = max_Cus.mean()
        min_O = min_O.mean()

        num_O = np.sum((positions[:,2] < max_Cu) & (positions[:,2] > min_O) & (is_o))
        if raw_coords:
            return max_Cu, min_O, num_O
        else:
            return max_Cu - min_O, num_O
    elif method == 4:
        # convert to MDAnalysis.Universe
        # which has a low-level memory leak
        letters = string.ascii_lowercase

        random.seed(None)
        filename = ''.join(random.choice(letters) for i in range(10))+'.data'
        write(filename, atoms, format='lammps-data')
        u = mda.Universe(filename, format='DATA', atom_style='id type x y z')
        
        inter = pytim.ITIM(u, max_layers=1, molecular=False, cluster_cut=4, alpha=3,normal='z')
        top_idx = np.array([int(elm) for elm in inter.layers[0][0].elements]) - 1
        top_z = np.mean(atoms.get_positions()[top_idx, 2])
        del u, inter
        gc.collect()
        
        atoms_o = atoms[np.array(atoms.get_chemical_symbols()) == 'O']
        write(filename, atoms_o, format='lammps-data')
        u = mda.Universe(filename, format='DATA', atom_style='id type x y z')
        inter = pytim.ITIM(u, max_layers=1, molecular=False, cluster_cut=4, alpha=3, normal='z')
        bot_idx = np.array([int(elm) for elm in inter.layers[1][0].elements]) - 1
        num_O = len(atoms_o)
        if len(bot_idx) != 0 and len(bot_idx) != 0:
            bot_z = np.mean(atoms_o.get_positions()[bot_idx, 2])
            os.remove(filename)
            del atoms, atoms_o, u, inter
            gc.collect()
            if top_z > bot_z:
                thickness = top_z - bot_z
            else:
                thickness = 0
        else:
            thickness = 0
        
        if not raw_coords:
            return thickness, num_O
        else:
            return top_z, bot_z

    return max_Cu - z_bottom


