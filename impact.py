import os
import mmap
import numpy as np
import pandas as pd
from PIL import Image
from scipy import special
from scipy import constants as consts
from scipy.signal import find_peaks
from ase import Atoms, Atom
from ase.db import connect
from ase.io import read, write
from ase.neighborlist import build_neighbor_list
from fast_lammpsdump import read_dump
from pymatgen.io.lammps.outputs import LammpsDump, parse_lammps_dumps, parse_lammps_log
import matplotlib.pyplot as plt


""" post-processing for low-temperature plasma impact MD simulations """


def find_impact(atoms_out):
    # detect when new ions are added
    len_save = len(atoms_out[0])
    tracking_flag = False
    impact_flag = False
    counter = 0
    impacts = []
    additions = []
    for i, atoms in enumerate(atoms_out):
        # print('image number ', i)
        len_new = len(atoms)
        if len_new > len_save:
            # atoms added
            if atoms[-1].symbol == 'O' and atoms[-2].symbol == 'O':
                # print(string_template.format('new ions added', i))
                additions.append(i)
                new_ion_idx = [len_new - 1, len_new - 2]
                tracking_flag = True
                impact_flag = False

        elif len_new < len_save:
            # atoms deleted, redo tracking
            tracking_flag = False
            impact_flag = False

        # determine if impact
        if tracking_flag:
            # print('now tracking')
            nl = build_neighbor_list(atoms, bothways=True)
            for n in new_ion_idx:
                # if any of the new ions have Cu atoms close by
                jndices, offsets = nl.get_neighbors(n)
                jndices = unique_list(jndices)
                # print('neighbors: ', jndices)

                for j in jndices:
                    if atoms[j].symbol == 'Cu':
                        tracking_flag = False
                        impact_flag = True
                        counter = 0
                        impacts.append(i)
                        # print(string_template.format('impact found', i))
                if impact_flag:
                    break

        if impact_flag:
            counter += 1
            if counter > 50:
                tracking_flag = False
                impact_flag = False
                # print('after_impact: ', counter)
                # print(atoms[indices])
                # break
        len_save = len_new
    return impacts, additions


def check_short_distances(atoms, elm_1, elm_2):
    indices_1 = [a.index for a in atoms if a.symbol == elm_1]
    indices_2 = [a.index for a in atoms if a.symbol == elm_2]

    if len(indices_1) == 0 or len(indices_2) == 0:
        return -1
    if elm_1 == elm_2 and len(indices_1) < 2:
        return -1

    max_dist = np.zeros(len(indices_1))
    for i in list(range(len(indices_1))):
        arr = atoms.get_distances(indices_1[i], indices_2, mic=True)
        arr = arr[arr > 0]
        max_dist[i] = np.min(arr)

    print(np.min(max_dist))
    return np.min(max_dist)


def unique_list(list_in):
    return list(dict.fromkeys(list_in))


def stitch_logs(runs):
    """ Stitches logs from runs to complete, nonoverallping log

    Note: relies on log.*.noew, generated from log.*.impact using
    sed '/NNP EW/d'
    """
    last_time = 0
    log = []

    for i in runs:
        log_read = parse_lammps_log('log.{:}.noew'.format(i))
        log_read = [l for l in log_read if l.shape[0] > 10]
        # log_read = log_read[1:]

        log_cat = pd.concat(log_read, ignore_index=True)
        if i != 1:
            last_step = log[-1]['Step'].iloc[-1]
            # this step may fail when thermo frequency is changed
            iloc = (log_cat['Step']-last_step).abs().idxmin()
            overran_time = log_cat['Time'].iloc[iloc]
            # overran_time = log_cat['Time'].loc[log_cat['Step']
            #                                  == last_step].iloc[0]

            last_time = log[-1]['Time'].iloc[-1] - overran_time

            log_cat = log_cat[log_cat['Step'] > last_step]
            log_cat['Time'] += last_time

        log.append(log_cat)

    log = pd.concat(log, ignore_index=True)
    return log


def build_brkt(runs):
    """ Builds brackets of steps from dump.*.timesteps files.

    Note: these timesteps files do not correspond to the
    nonoverallping log file.
    """
    steps_brkt = []
    for r in runs:
        if os.path.exists('dump.{:}.timesteps'.format(r)):
            timesteps = np.loadtxt('dump.{:}.timesteps'.format(r))
            if timesteps.ndim > 1:
                steps_brkt.append(timesteps[0, 0])
            else:
                steps_brkt.append(timesteps[0])
        else:
            timesteps = np.loadtxt('dump.{:}.byteidx'.format(r))[:,1]
            steps_brkt.append(timesteps[0])
    steps_brkt.append(timesteps[-1])
    steps_brkt = np.array(steps_brkt)

    return steps_brkt


def steps4averaging(log, num_ctrs, num_smooth):
    """ Generates a list of timesteps lists.

    Note: top level is center, lower level is timesteps for averaging
    """

    step_ctrs, _ = get_steps(log, num_ctrs, 'log10')
    num_sample = num_smooth # number of points around which to smooth the rdf
    # 100 is needed because 1 dump every 100 steps
    smoothing = np.linspace(-10, 10, num_sample)*100
    stepss = []
    for step_ctr in step_ctrs:
        stepss.append((step_ctr + smoothing))

    return step_ctrs, stepss


def get_steps(log, num_ctrs, spacing):

    total_time = log['Time'].iloc[-1]
    init_time = log['Time'].iloc[0]

    if spacing == 'log10':
        time_ctrs = np.logspace(5, np.log10(total_time), num_ctrs)
    elif spacing == 'linear':
        time_ctrs = np.linspace(init_time, total_time, num_ctrs)

    # this idxmin has duplicates every 10000 rows,
    # unless ignore_index is used in concat
    closest_iloc = [(log['Time']-s).abs().idxmin() for s in time_ctrs]

    # step_ctrs in general will not coincide with dump indices
    # since frequencies are different
    step_ctrs = [int(log.iloc[idx]['Step']) for idx in closest_iloc]
    # need to find nearest integer divisible by 100, the dump frequency
    step_ctrs = [s-s % 100 if s % 100 < 50 else s+(100-s % 100) for s in step_ctrs]

    return step_ctrs, time_ctrs


def get_rdf(stepss, steps_brkt):
    """ Calculates time-averaged radial distribution functions

    Note: relies on run_*.db files holding correct atoms
    Note: relies on asap3 for rdf routine which is not available on Windows conda-forge
    """

    if not os.name == 'nt':
        from asap3.analysis.rdf import RadialDistributionFunction as RDF
    else:
        print('impact.get_rdf(): RDF is not available on Windows')
        exit()
    rMax = 10
    nBins = 500
    elements = (8, 29)
    x = np.arange(nBins) * rMax / nBins

    rdfs = []
    for steps in stepss:
        rdf = np.zeros(nBins)
        for s in steps:
            r = np.where(s > steps_brkt)[0][-1]+1
            atoms = read('run_{:}.db@timestep={:}'.format(r,s))[0]
            RDFobj = RDF(atoms, rMax, nBins)
            rdf += RDFobj.get_rdf(elements=elements)

        rdf = rdf / len(steps)
        rdfs.append(rdf)

    return rdfs, x


def get_snap_imgs(steps_ctrs, steps_brkt):
    """ Generates png images zoomed in to the oxide layers """
    for s in steps_ctrs:
        r = np.where(s > steps_brkt)[0][-1]+1

        im_filename = '{:}.png'.format(s)
        atoms = read('run_{:}.db@timestep={:}'.format(r, s))[0]
        write(im_filename, atoms, rotation='-90x')
        im = Image.open(im_filename)

        z = atoms.get_positions()[:, 2]
        is_o = atoms.symbols == 'O'
        max_O = max(z[is_o])
        min_O = min(z[is_o])

        cell_z = atoms.cell.cellpar()[2]

        width, height = im.size
        # top of slab, also top of image
        top = np.ceil(height / cell_z * (cell_z-max_O)) - 50
        # bottom of slab, also bottom of image
        bottom = np.floor(height / cell_z * (cell_z-min_O)) + 50
        left = 0
        right = width

        im1 = im.crop((left, top, right, bottom))
        im1.save(im_filename)


def get_thicknesses(steps_ctrs, steps_brkt):
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
    for s in steps_ctrs:
        r = np.where(s >= steps_brkt)[0][-1]+1
        if r == len(steps_brkt):
            r -= 1
            
        if os.path.exists('dump.{:}.thickness'.format(r)):
            nparr = np.loadtxt('dump.{:}.thickness'.format(r))
            t = nparr[np.where(np.arr[0] == s), 1]
            no = nparr[np.where(np.arr[0] == s), 2]
        else:
            if os.path.exists('run_{:}.db'.format(r)):
                atoms = read('run_{:}.db@timestep={:}'.format(r, s))[0]
            else:
                atoms = read_dump('dump.{:}.high_freq'.format(r), s)
                
            if len([a for a in atoms if a.symbol == 'O']) == 0:
                t = 0
                no = 0
            else:
                t, no = get_thickness(atoms, 3)
        
        thicknesses.append(t)
        nos.append(no)
        atomss.append(atoms)
        j += 1
    return np.array(thicknesses), np.array(nos)


def get_all_thicknesses(filename, byteidx):
    from tqdm import tqdm
    
    thicknesses = np.empty(1)
    num_o = np.empty(1)
    
    for step in tqdm(byteidx[:, 1]):
        atoms = read_dump(filename, step)
        t, no = get_thickness(atoms, 3)
        thicknesses=np.append(thicknesses, t)
        num_o=np.append(num_o, no)

    return np.vstack((byteidx[:, 1], thicknesses[1:], num_o[1:]))
    

def get_all_thicknesses_parallel(filename, byteidx):
    from pqdm.processes import pqdm
    from tqdm import tqdm
    from joblib import Parallel, delayed
    
    def foo(s):
        atoms = read_dump(filename, s)
        t, no = get_thickness(atoms, 3)
        return t, no

    thickness, num_o = zip(*Parallel(n_jobs=32)(delayed(foo)(s) for s in tqdm(byteidx[:, 1])))
    return np.vstack((byteidx[0:100, 1], thickness, num_o))

        
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

    return max_Cu - z_bottom


def find_impact_site(atomss):
    """ Determines the impact site from a sequence of atoms objects

    Note:
    atomss[0] is before the ion addition,
    atomss[1] is the first instant of ion addition
    """
    for i, atoms in enumerate(atomss):
        if len(atoms) > len(atomss[0]):
            addition_idx = i
            break

    dist = 1e3
    for idx0, atoms in enumerate(atomss[addition_idx:]):
        arr = atoms.get_distances(list(range(len(atoms)-1)), -1, mic=True)
        arr = arr[arr > 0]
        new_dist = np.amin(arr)

        if new_dist > dist:
            idx = arr.argmin()
            impact_site = atoms[idx].position
            return impact_site, idx0+addition_idx, idx
        else:
            dist = new_dist


def get_spike_range(atomss=None, temp=353, prob_cutoff=1e-5):
    """ Calcuates range of heat spikes after impact

    Note: range is defined as the distance of atom that is
    excited to velocities 'cutoff' times higher than the
    v_rms at thermostated temperature.

    atoms passed in is a single impact trajectory
    """

    kb = consts.value('Boltzmann constant in eV/K')
    amu = consts.value('atomic mass constant')

    # m_d       mass in Dalton
    # m_kg      mass in kg
    # v_apfs    velocity in angstroms per femtosecond
    # v_mps     velocity in meters per second
    # ke_j      kinetic energy in joules

    def v_mps(m_d, T):
        """ rms velocity in meters per second """
        return np.sqrt((3*kb*T*consts.eV)/(m_d*amu))

    def ke_j(m_d, v_apfs):
        """ kinetic energy in joules """
        return 1/2*(v_apfs * 1e-10 / 1e-15)**2 * (m_d * amu)

    def int_mb(m_d, v_mps, T):
        """ maxwell-boltzmann integrated from v to infty """
        m_kg = m_d*amu
        k = consts.k # in SI units
        b = (m_kg/(2*np.pi*k*T))**(3/2)
        a = m_kg / (2*k*T)
        non_integrated = b*4*np.pi*v_mps**2*np.exp(-a*v_mps**2)
        integrated = b*4*np.pi * (np.sqrt(np.pi)*special.erfc(np.sqrt(a)*v_mps)/(4*a**(3/2)) + v_mps*np.exp(-a*v_mps**2/(2*a)))

        return non_integrated, integrated # nondimensional
    
    def v_cutoff(m_d, T):
        # determine cutoff velocity
        v_rms = v_mps(m_d, T)
        vs = np.linspace(v_rms, v_rms*10, 100)
        _, integrated = int_mb(m_d, vs, T)
        # import matplotlib.pyplot as plt
        # plt.plot(vs, integrated)
        # plt.show()
        # exit()
        # probablity of such hot atoms arising in thermal
        # motion is 0.00001, very unlikely given our number
        # of atoms (4000)
        mps2apfs = 1e10/1e15
        v_cut = vs[integrated < prob_cutoff][0] * mps2apfs
        return v_cut
    
    v_cut_cu = v_cutoff(63.54, T=temp)
    v_cut_o  = v_cutoff(16, T=temp)
    # determine impact site
    try:
        impact_site, impact_idx, atom_idx = find_impact_site(atomss)
    except:
        return -1, -1, -1

    hot_list = np.array([len(atomss[impact_idx])-1, atom_idx], dtype=int) # initial hot cu atom
    # print('hot_list: ', hot_list)
    dist = 0
    deep = 0
    for atoms in atomss[impact_idx+1:]:
        vel = atoms.get_velocities()
        vel_normed = np.linalg.norm(vel, axis=1)

        # need to maintain a list of hot atoms, and require that
        # new hot atoms are within certain distance of the
        # existing hot atoms

        # current list
        hot_mask_cu = (vel_normed > v_cut_cu) & (np.array(atoms.get_chemical_symbols()) == 'Cu')
        hot_mask_o = (vel_normed > v_cut_o) & (np.array(atoms.get_chemical_symbols()) == 'O')
        hot_mask = hot_mask_cu | hot_mask_o
        hot_list_now = np.where(hot_mask)[0]
        max_change = 2.7 # > bond length

        # compare to old list
        new_hot_list = []
        for h in hot_list_now:
            if h in hot_list:
                # if h is already one of the old hot atoms,
                # it stays in the list
                continue

            atoms_hot = atoms[hot_list] + atoms[h]
            atoms_hot.set_pbc(True)
        # breakpoint()
            arr = atoms_hot.get_distances(np.arange(hot_list.size), -1,
                                          mic=True)

            if np.amin(arr) > max_change:
                # h is far from all other hot atoms
                continue
            else:
                new_hot_list.append(h)

        # this should not have duplicate elements
        hot_list = np.concatenate((hot_list,
                                   np.array(new_hot_list, dtype=int)))
        
        # check if any atom in this list has drifted past the boundary, remove from the hot list
        to_delete = [idx for idx, elm in enumerate(hot_list) if atoms[elm].position[2] < 5 and atoms[elm].symbol == 'O']
        hot_list = np.delete(hot_list, to_delete)

        # find the max distance to impact_site
        atoms_hot = atoms + Atom('F', position=impact_site)
        atoms_hot.set_pbc(True)

        arr = atoms_hot.get_distances(hot_list, -1, mic=True)
        arr = np.sort(arr)[::-1]
        new_dist = arr[0]
        if new_dist > dist:
            # print('dist, new_dist = {:4.2f}, {:4.2f}'.format(dist, new_dist))
            # if len(new_hot_list) > 0:
            #     breakpoint()
            # print('new_hot_list: ', new_hot_list)
            # print('hot_list_now: ', hot_list_now)
            # print('updated hot_list: ', hot_list)
            dist = new_dist

        arr = impact_site[2] - atoms_hot[hot_list].get_positions()[:, 2]
        arr = np.sort(arr)[::-1] # lowest first
        new_deep = arr[0]
        if new_deep > deep:
            deep = new_deep
    return dist, deep, atoms_hot[hot_list]
