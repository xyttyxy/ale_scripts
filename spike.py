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
import multiprocessing as mp

kb = consts.value('Boltzmann constant in eV/K')
amu = consts.value('atomic mass constant')

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

def v_cutoff(m_d, T, prob_cutoff):
    # determine cutoff velocity
    v_rms = v_mps(m_d, T)
    vs = np.linspace(v_rms, v_rms*10, 100)
    _, integrated = int_mb(m_d, vs, T)

    # probablity of such hot atoms arising in thermal
    # motion is 0.00001, very unlikely given our number
    # of atoms (4000)
    mps2apfs = 1e10/1e15
    v_cut = vs[integrated < prob_cutoff][0] * mps2apfs
    return v_cut

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


def get_spike_range(atomss=None, temp=353, prob_cutoff=1e-5, pids=None, impact_idx=0):
    """ Calcuates range of heat spikes after impact

    Note: range is defined as the distance of atom that is
    excited to velocities 'cutoff' times higher than the
    v_rms at thermostated temperature.

    atoms passed in is a single impact trajectory
    """
    from tqdm import tqdm
    current = mp.current_process()

    # m_d       mass in Dalton
    # m_kg      mass in kg
    # v_apfs    velocity in angstroms per femtosecond
    # v_mps     velocity in meters per second
    # ke_j      kinetic energy in joules
    
    v_cut_cu = v_cutoff(63.54, T=temp, prob_cutoff=prob_cutoff)
    v_cut_o  = v_cutoff(16, T=temp, prob_cutoff=prob_cutoff)
    # determine impact site
    try:
        impact_site, impact_idx, atom_idx = find_impact_site(atomss)
    except:
        return -1, -1, -1
        
    hot_list = np.array([len(atomss[impact_idx])-1, atom_idx], dtype=int) # initial hot cu atom
    # print('hot_list: ', hot_list)
    dist = 0
    deep = 0

    position = current._identity[0]-33
    
    for atoms in atomss[impact_idx+1:]:
        # for atoms in tqdm(atomss[impact_idx+1:], desc='SpikeRange-{:}-{:}'.format(position, impact_idx), position=position, leave=False):
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

def get_temp_grad(atomss=None, temp=353, prob_cutoff=1e-5, pids=None, run_id=0):
    """ helper function in calculating the temperature gradient
    """
    
    def v2t(mass, atom_v):
        mps2apfs = 1e10/1e15
        tidx = (1/2*(mass*amu)*(atom_v/mps2apfs)**2)/(3*kb*consts.eV)
        return tidx
    
    from tqdm import tqdm
    # current = mp.current_process()
    # position = current._identity[0]-33
        
    v_cut_cu_original = v_cutoff(63.54, T=temp, prob_cutoff=prob_cutoff)
    v_cut_o_original  = v_cutoff(16, T=temp, prob_cutoff=prob_cutoff)
    
    # determine impact site
    try:
        impact_site, impact_idx, atom_idx = find_impact_site(atomss)
    except:
        return -1, -1
    impact_atoms = atomss[impact_idx]
    min_o = min([at.position[2] for at in impact_atoms if at.symbol == 'O'])
    max_cu = max([at.position[2] for at in impact_atoms if at.symbol == 'Cu'])
    z_coord = np.linspace(min_o, max_cu, 20)
    tz = np.zeros(20)-1
    
    hot_list = np.array([len(atomss[impact_idx])-1, atom_idx], dtype=int) # initial hot cu atom

    
    #for atoms in tqdm(atomss[impact_idx+1:], desc='SpikeRange-{:}-{:}'.format(position, impact_idx), position=position, leave=False):
    for atoms in atomss[impact_idx+1:]:
        vel = atoms.get_velocities()
        vel_normed = np.linalg.norm(vel, axis=1)

        # if len(hot_list) > 1:
        #     v_cut_cu = np.append(vel_normed[[h for h in hot_list if atoms[h].symbol == 'Cu']]*0.8, v_cut_cu_original).min()
        #     v_cut_o = np.append(vel_normed[[h for h in hot_list if atoms[h].symbol == 'O']]*0.8, v_cut_o_original).min()
        #     v_cut_cu = max(v_cut_cu_original/1.414, v_cut_cu)
        #     v_cut_o = max(v_cut_o_original/1.414, v_cut_o)
        # else:
        v_cut_cu = v_cut_cu_original
        v_cut_o = v_cut_o_original

        # need to maintain a list of hot atoms, and require that
        # new hot atoms are within certain distance of the
        # existing hot atoms
        # breakpoint()
        # current list
        # print(v2t(64, v_cut_cu), v2t(16, v_cut_o))
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

        # for each atom in hot_list, update corresponding temperature
        for atom, atom_v in zip(atoms[hot_list], vel_normed[hot_list]):
            atom_z = atom.position[2]
            # index of the first z coordinate less than hot atom
            argwhere_res = np.argwhere(z_coord < atom_z)
            if len(argwhere_res) == 0:
                breakpoint()
                print('bug: run_'+str(run_id))
                return -1, -1
            else:
                idx = np.argwhere(z_coord < atom_z)[-1][0]
                tidx = ke_j(atom.mass, atom_v) / (3/2*consts.k)
                tz[idx] = max(tidx, tz[idx])
                
        # for z_idx in range(len(z_coord)-1):
        #     if tz[z_idx] == -1:
        #         layer_idxs = [at.index for at in atoms if at.position[2] < z_coord[z_idx+1] and at.position[2] >z_coord[z_idx]]
        #         layer_vel = vel_normed[layer_idxs]
        #         # kb in eV/K
        #         kb_j = consts.k
        #         layer_ke = np.array([ke_j(atoms[a_idx].mass, a_vel) for a_idx, a_vel in zip(layer_idxs, layer_vel)]).sum()
        #         t_mean = layer_ke / (3/2*kb_j*len(layer_idxs))
        #         #tz[z_idx] = t_mean

    return z_coord, tz


