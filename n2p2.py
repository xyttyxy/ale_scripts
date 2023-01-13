
# -*- mode: python; python-indent-offset: 4 -*-
import numpy as np
import pandas as pd
from io import StringIO
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator as SPC


def n2p2_data(atoms, energy, name):
    """ n2p2 data string from ase.Atoms """
    out_str = ''
    out_str += 'begin\n'
    out_str += 'comment ' + name + '\n'
        
    cell = atoms.get_cell()
    cell_template = 'lattice {:10.6f} {:10.6f} {:10.6f}\n'
    for c in cell:
        out_str += cell_template.format(c[0], c[1], c[2])

    atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}\n'
    forces = atoms.get_forces()
    for a in atoms:
        force = forces[a.index]
        out_str += atom_template.format(a.position[0], a.position[1], a.position[2],
                                        a.symbol, 0.0, 0.0,
                                        force[0], force[1], force[2])
                
    out_str += 'energy {:10.6f}\n'.format(energy)
    out_str += 'charge 0.0\n'
    out_str += 'end\n'

    return out_str

def read_n2p2_data(filename, sort = True):
    """ ase.Atoms from n2p2 data format """
    begin_indx = []
    end_indx = []
    with open(filename) as f:
        lines = f.readlines()
    begin_indx = [i for i, l in enumerate(lines) if 'begin' in l]
    end_indx = [i for i, l in enumerate(lines) if 'end' in l]
    atoms_lst = []
    for atoms_indx, begin_linum in enumerate(begin_indx):
        comment_line = lines[begin_linum+1]
        name = comment_line.split()[1]
        
        lattice_lines = lines[begin_linum+2:begin_linum+5]
        df_lattice = pd.read_table(StringIO(''.join(lattice_lines)), delim_whitespace=True, names=['lattice','lx','ly','lz'])
        lattice = np.array(df_lattice[['lx','ly','lz']])
        
        atom_lines = lines[begin_linum+5:end_indx[atoms_indx]-2]
        df_atoms = pd.read_table(StringIO(''.join(atom_lines)), delim_whitespace=True, names=['atom', 'x', 'y', 'z', 'e',  'c', 'n', 'fx', 'fy', 'fz'])
        positions = np.array(df_atoms[['x','y','z']])
        forces = np.array(df_atoms[['fx', 'fy', 'fz']])
        atoms = Atoms(symbols = df_atoms['e'].tolist(), positions=positions, cell=lattice, info={'name': name})
        
        energy_line = lines[end_indx[atoms_indx]-2]
        e = float(energy_line.split()[1])

        atoms.set_calculator(SPC(atoms, energy=e, forces=forces))
        atoms_lst.append(atoms)
    comments = [atoms.info['name'] for atoms in atoms_lst]

    # ensure sorted by comment, fix MPI messing with ordering
    if sort:
        atoms_lst = [atoms for _, atoms in sorted(zip(comments, atoms_lst))]
    return atoms_lst

def write_dft_db(dbname, dirs, dir_parser):
    """ Collect VASP results into db file """
    from ase.db import connect
    from ase.calculators.vasp import Vasp
    db = connect(dbname, append=False)
    for dirname in dirs:
        name = dir_parser(dirname)
        try:
            db.get(name=name)
            print('Found in db: ', name)
        except KeyError:
            try:
                calc = Vasp(directory=dirname, restart=True)
                num_iter = calc.get_number_of_iterations()
                if num_iter >= 60:
                    print('Calc SCF failed: ', name, num_iter)
                    continue
                else:
                    print('Calc success: ', name, num_iter)
                    atoms = calc.get_atoms()
                    db.write(atoms, name=name)
            except Exception:
                print('Calc error: ', name)
                continue

