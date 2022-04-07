from myase.io import myread
from ase.io import read
from ase.calculators.vasp import Vasp, Vasp2
from myase.calculators.gaussian import Gaussian
from myase.calculators.lammpslib import LAMMPSlib
def read_struct_from_args(args):
    if '-r' in args:
        idx = args.index('-r')
        params = (args[idx+1].split(','))
        format=None
        for param in params:
            param_lst = param.split('=')
            if param_lst[0] == 'format':
                format=param_lst[1]
            elif param_lst[0] == 'filename':
                filename=param_lst[1]

        atoms = myread(format=format, filename=filename)
        try:
            atoms = myread(format=format, filename=filename)
        except FileNotFoundError:
            print('ERROR: GEOMETRY FILE NOT FOUND')
            sys.exit()
    elif '-g' in args:
        try:
            from gen import generate
        except ModuleNotFoundError:
            print('ERROR: GEOMETRY GENERATION SCRIPT NOT FOUND')
            sys.exit()
        atoms = generate()
    else:
        print('ERROR: GEOMETRY NOT SPECIFIED')
        sys.exit()
    return atoms

def is_test(args):
    if '-t' in args:
        print("TEST COMPLETE. NO ERRORS")
        return True
    else:
        return False

import sys
def parse(args, calculator, label, atom_types=None, cell=None):
    try:
        atoms = read_struct_from_args(args)
        # check the type of calculator passed in
        if isinstance(calculator, Vasp) or isinstance(calculator, Vasp2):
            program = 'vasp'
        elif isinstance(calculator, Gaussian):
            program = 'gaussian'
        elif isinstance(calculator, LAMMPSlib):
            program = 'lammps'

        atoms.set_calculator(calculator)
        # CLEANUP THE ATOMS OBJECT(PBC, CELL)
        # vasp is always PBC
        if program == 'vasp':
            atoms.set_pbc([1,1,1])
            
        if cell != None:
            atoms.set_cell(cell)

        # if atoms cell not set, as when you read XSD's from material studio
        from numpy import abs
        if abs(atoms.get_cell()).max() == 0.0:
            atoms.set_cell([20,20,20])
            atoms.wrap()
            atoms.center()

        if '-v' in args:
            from ase.visualize import view
            view(atoms)

        if is_test(args):
            sys.exit()
        else:
            print('EXECUTING')
            print(atoms.get_potential_energy())
            print(atoms.get_forces())
            
            # if vasp read CONTCAR
            if program == 'vasp':
                after = myread('CONTCAR')
            # if gaussian read .out
            if program == 'gaussian':
                try: 
                    after = myread(format='GAUSSIAN_OUT', filename=label+'.out')
                except IndexError:
                    print('ERROR: GAUSSIAN OUT EMPTY')
                    sys.exit()
            # if lammps read dump file
            if program == 'lammps':
                after = read('dump.Cu', format='lammps-dump', index=':', atom_types=atom_types)

            write(filename = label + '_after' + '.xsd', images = after, format = 'xsd')
    except IndexError:
        print(help_message)

    # return the optimized atom
    return atoms


def run(atoms, calculator, is_test):
    if is_test:
        pass
    else:
        print('EXECUTING')
        atoms.set_calculator(calculator)
        print(atoms.get_potential_energy())
        print(atoms.get_forces())
    return atoms
        
