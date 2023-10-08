import os
import numpy as np
import pandas as pd
from PIL import Image
from ase.io import read, write
from ase.neighborlist import build_neighbor_list
from pymatgen.io.lammps.outputs import parse_lammps_log


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

        if log_cat.shape[0] == 0:
            print('AHHHHH! You have duplicate runs!')
        else:
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

def find_cluster(atoms):
    from ase.neighborlist import NeighborList
    cutoffs = [0.1] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(atoms)
    conn_mat = nl.get_connectivity_matrix(sparse=False)
    molecules = np.argwhere(conn_mat)
    from ase.atoms import Atoms
    markers = Atoms([atoms[m] for m in molecules.flatten()], cell=atoms.cell, pbc=True)
    atoms_copy = atoms.copy()

    del atoms_copy[[molecules[0] for molecule in molecules]]

    cutoffs = [0.6 if at.symbol == 'O' else 0 for at in atoms_copy]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(atoms_copy)
    conn_mat = nl.get_connectivity_matrix(sparse=False)
    molecules = np.argwhere(conn_mat)
    markers = Atoms([atoms_copy[m] for m in molecules.flatten()], cell=atoms_copy.cell, pbc=True)
    return markers


