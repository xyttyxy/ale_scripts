# from pyscal import traj_process as ptp
# from myase.io.lammpsdump import read_lammps_dump_pymatgen, read_lammps_dump
# from ase.io import read
# from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
from fast_lammpsdump import read_lammps_dump_pymatgen, read_dump
import numpy as np
import pyscal as pc
from impact import get_thickness
from ase.db import connect
from ase.io import read
import os

def get_q(atoms):
    sys = pc.System()
    sys.read_inputfile(atoms, format='ase')
    sys.find_neighbors(method="voronoi")
    # cannot call get_all_neighbors_voronoi twice, leads to segfault
    # neighbors are assigned in process_neighbors
    # just need to hijack that
    sys.calculate_q([4,6])
    q = np.array(sys.get_qvals([4,6]))
    return q


def calc_boo_db(dbfilename):
    q = np.empty((2,1))
    # read cuo database
    db = connect(dbfilename)
    dfs = []
    for row in db.select():
        atoms = row.toatoms()
        symbols = np.array(atoms.get_chemical_symbols()) == 'Cu'
        q = np.transpose(get_q(atoms))
        df = pd.DataFrame({'q4': q[:, 0],
                           'q6': q[:, 1],
                           'sym': symbols})
        dfs.append(df)
    return pd.concat(dfs) * 1


def calc_boo_slab(atomss):
    ql = np.empty((3, 1))
    dfs = []
    for atoms in atomss:
        # filter some atoms
        max_cu, min_o, num_o = get_thickness(atoms, method=3, raw_coords=True)
        min_cu = min([at.position[2] for at in atoms if at.symbol == 'Cu' ])
        del atoms[[at.index for at in atoms if at.position[2] < min_o]]
        del atoms[[at.index for at in atoms if at.position[2] > max_cu]]
        del atoms[[at.index for at in atoms if at.position[2] < min_cu - 5]]
        
        pz = atoms.get_positions()[:, 2]
        # pz_mask = (pz > min_o + 2.5) & (pz < max_cu - 2.5)
        
        # breakpoint()
        ql_now = get_q(atoms)
        ql_now = np.transpose(np.vstack((ql_now, pz)))
        symbols = np.array(atoms.get_chemical_symbols()) == 'Cu'
        df = pd.DataFrame({'q4': ql_now[:, 0],
                           'q6': ql_now[:, 1],
                           'z': ql_now[:, 2],
                           'sym': symbols})
        dfs.append(df)

    return pd.concat(dfs)*1


base = False
if base:
    basedir = '/mnt/d/Work/ALE/hdnnp/dataset/bulk_oxide/'
    # q_cuo_2000 = calc_boo_db(basedir+'CuO_mp1692/2x2x2/dft_singlepoints/2000K/spin.db')
    q_cuo_1000 = calc_boo_db(basedir+'CuO_mp1692/2x2x2/dft_singlepoints/1000K/spin.db')
    q_cuo_1000_3x3 = calc_boo_db(basedir+'CuO_mp1692/3x3x2/dft_singlepoints/spin.db')

    q_cu2o = calc_boo_db(basedir+'Cu2O_mp361/dft_singlepoints/spin.db')
    q_cuo = pd.concat([q_cuo_1000, q_cuo_1000_3x3])

    q_cuo[q_cuo['sym']==1].to_csv('q4q6_cuo_1000K_cu.csv', index=False, index_label=False)
    q_cuo[q_cuo['sym']==0].to_csv('q4q6_cuo_1000K_o.csv', index=False, index_label=False)

    q_cu2o[q_cu2o['sym']==1].to_csv('q4q6_cu2o_1000K_cu.csv', index=False, index_label=False)
    q_cu2o[q_cu2o['sym']==0].to_csv('q4q6_cu2o_1000K_o.csv', index=False, index_label=False)
    exit()

# this is roughly 100 structures
slab = True
if slab:
    atomss = read_lammps_dump_pymatgen('/mnt/e/production/10x10x50/10eV/353K/10/afterrun_6/dump.low_freq', every_n=1)
    q = calc_boo_slab(atomss)
    q[q['sym']==1].to_csv('10x353K10n2i_cu.csv', index=False, index_label=False)
    q[q['sym']==0].to_csv('10x353K10n2i_o.csv', index=False, index_label=False)
    exit()

evo = True
if evo:
    basedir = '20x353K20eV10n2i'
    trajfiles = [f for f in os.listdir(basedir) if 'run' in f and os.path.isdir(basedir+'/'+f)]
    runs = [int(f.split('_')[1]) for f in trajfiles]
    runs, trajfiles = zip(*sorted(zip(runs, trajfiles)))
    atomss = []
    trajfiles = [t+'/dump.run' for t in trajfiles]
    for r, f in zip(runs, trajfiles):
        atomss = read_lammps_dump_pymatgen(basedir+'/'+f, every_n = 50)
        q = calc_boo_slab(atomss)
        plt.clf()
        plt.scatter(q[q['sym'] == 1]['q6'], q[q['sym'] == 1]['z'], c = 'k', s = 1, label = 'cu')
        plt.scatter(q[q['sym'] == 0]['q6'], q[q['sym'] == 0]['z'], c = 'r', s = 1, label = 'o')
        plt.ylim([90, 110])
        plt.xlim([0, 0.8])
        plt.savefig(basedir+'/q6vz_run_{:}.png'.format(r))
        q[q['sym'] == 1].to_csv(basedir+'/run_{:}_cu.csv'.format(r), index=False, index_label=False)
        q[q['sym'] == 0].to_csv(basedir+'/run_{:}_o.csv'.format(r), index=False, index_label=False)
    exit()


q = calc_boo_slab([read('./20xminimized/run_10_last/lastmin.traj')])
plt.scatter(q['q4'], q['z'], c='k', s=1)
plt.show()
        
