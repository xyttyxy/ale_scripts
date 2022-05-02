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
prob_cutoff = 1e-3

from spike import v_mps, ke_j, int_mb, v_cutoff, find_impact_site

def get_temp_grad(atomss=None, temp=353, prob_cutoff=1e-5, pids=None, run_id=0):
    """ helper function in calculating the temperature gradient
    """
    
    def v2t(mass, atom_v):
        mps2apfs = 1e10/1e15
        tidx = (1/2*(mass*amu)*(atom_v/mps2apfs)**2)/(3*kb*consts.eV)
        return tidx
    
    from tqdm import tqdm
    current = mp.current_process()
    position = current._identity[0]-33
    
    v_cut_cu_original = v_cutoff(63.54, T=temp, prob_cutoff=prob_cutoff)
    v_cut_o_original  = v_cutoff(16, T=temp, prob_cutoff=prob_cutoff)
    
    # determine impact site
    try:
        impact_site, impact_idx, atom_idx = find_impact_site(atomss)
    except:
        print('failed to find impact site')
        return -1, -1, -1
    
    impact_atoms = atomss[impact_idx]
    min_o = min([at.position[2] for at in impact_atoms if at.symbol == 'O'])
    max_cu = max([at.position[2] for at in impact_atoms if at.symbol == 'Cu'])
    z_coord = np.linspace(min_o, max_cu, 20)
    tz = np.zeros(20)-1

    # [ion, atom_hit_by_ion]
    spike_list = np.array([len(atomss[impact_idx])-1, atom_idx], dtype=int) 
    df_traj = []
    stepnumber = 0
    for atoms in tqdm(atomss[impact_idx+1:], desc='TempGrad-{:}-{:}'.format(position, impact_idx), position=position, leave=False):
        vel = atoms.get_velocities()
        vel_normed = np.linalg.norm(vel, axis=1)
        df_atoms = []
        for z_idx in reversed(range(len(z_coord)-1)):
            ## characterize the layer
            layer_idxs = [at.index for at in atoms if at.position[2] < z_coord[z_idx+1] and at.position[2] >z_coord[z_idx]] # NL

            layer_mask = np.zeros(len(atoms), dtype=bool) 
            layer_mask[layer_idxs] = True # N

            layer_vel = vel_normed[layer_idxs] # NL

            kb_j = consts.k
            layer_ke = np.array([ke_j(atoms[a_idx].mass, a_vel) for a_idx, a_vel in zip(layer_idxs, layer_vel)]).sum() # 1
            t_mean = layer_ke / (3/2*kb_j*len(layer_idxs)) # 1

            ## get hot atoms in current layer
            v_cut_cu = min([v_cutoff(63.54,
                                     T=t_mean,
                                     prob_cutoff=prob_cutoff),
                            v_cut_cu_original])

            v_cut_o = min([v_cutoff(16,
                                    T=t_mean,
                                    prob_cutoff=prob_cutoff),
                           v_cut_o_original])

            hot_mask_cu = (vel_normed > v_cut_cu) & (np.array(atoms.get_chemical_symbols()) == 'Cu') & layer_mask # N
            hot_mask_o = (vel_normed > v_cut_o) & (np.array(atoms.get_chemical_symbols()) == 'O') & layer_mask # N
            
            hot_mask = hot_mask_cu | hot_mask_o # N
            hot_layer = np.where(hot_mask)[0] # NLH

            ## get only those connected to the spike
            max_change = 2.7 # > bond length
            spike_layer = []
            for h in hot_layer:
                if h in spike_list:
                    spike_layer.append(h)
                    continue

                atoms_hot = atoms[spike_list] + atoms[h]
                atoms_hot.set_pbc(True)
                arr = atoms_hot.get_distances(np.arange(spike_list.size), -1,
                                              mic=True)
                if np.amin(arr) <= max_change:
                    spike_layer.append(h)
                    
            # breakpoint()
            ## check if any atom in this list has drifted past the boundary, remove from the hot list
            to_delete = [idx for idx, elm in enumerate(spike_list) if atoms[elm].position[2] < 5 and atoms[elm].symbol == 'O']
            spike_list = np.delete(spike_list, to_delete)
            
            ## store results
            if len(spike_layer) > 0:
                temp_max = np.array([v2t(atoms[idx].mass, vel_normed[idx]) for idx in spike_layer]).max()
                tz[z_idx] = max(temp_max, tz[z_idx])
                spike_list = np.unique(np.concatenate((spike_list,
                                                       np.array(spike_layer, dtype=int))))
                # get element in hot_layer and in spike_list
                conc = np.concatenate((spike_list, hot_layer))
                m = np.zeros_like(conc, dtype=bool)
                m[np.unique(conc, return_index=True)[1]] = True
                # non_unique elements
                spike_layer_past_inc = conc[~m]
                # construct dataframe
                temp_all = np.array([v2t(atoms[idx].mass, vel_normed[idx]) for idx in spike_layer_past_inc])
                df_layer = pd.DataFrame({'z': atoms.get_positions()[:, 2][spike_layer_past_inc],
                                         'vel': temp_all,
                                         'idx': np.array(spike_layer_past_inc),
                                         'element': [atoms[idx].symbol for idx in spike_layer_past_inc],
                                         'timestep': np.zeros_like(spike_layer_past_inc)+stepnumber+impact_idx+1})
                df_atoms.append(df_layer)
                
            elif tz[z_idx] == -1:
                tz[z_idx] = t_mean

        try:
            df_traj.append(pd.concat(df_atoms))
        except:
            print('nothing to concat, continuing')
            pass
        stepnumber += 1
        
    return z_coord, tz, pd.concat(df_traj)


