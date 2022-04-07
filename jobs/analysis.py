# script to check for convergence and print energies
# as a replacement for executing for i in *; do grep..
import json
import os
import warnings

import numpy as np
from ase.calculators.vasp import Vasp, Vasp2
from ase.io import read
from myase.calculators.gaussian import Gaussian
from myase.thermochemistry import SimpleCrystalThermo

warnings.simplefilter('ignore')
# ASE reading existing calculation into a calculator.
# VASP2(FileIOCalculator) is a lot (3X) faster
def check_convergence(interactive = True, prefix = 't'):
    # must be called in the parent of workdirs
    min_eng = None
    dirs = next(os.walk('.'))[1]
    #dirs.reverse()
    paths = []
    energies = []
    for i in dirs:
        if '__' in i:
            continue
        elif prefix not in i:
            continue
        pwd = os.getcwd()
        os.chdir(i)
        try:
            calc = Vasp2(restart=True)
            atoms = calc.get_atoms()
            if calc.converged:
                energy = calc.get_potential_energy(atoms)
                paths.append(i)
                energies.append(energy)
                if interactive:
                    print("{:<10} {:<20} {:<15}".format(i+'/', "CONVERGED:", energy))
            else:
                if interactive:
                    print("{:<10} {:<20}".format(i+'/', "NOT CONVERGED!"))

        except FileNotFoundError:
            if interactive:
                print("{:<10} {:<20}".format(i+'/', "FILES NOT FOUND!"))
        except:
            if interactive:
                print("{:<10} {:<20}".format(i+'/', "RUNNING!"))

        os.chdir(pwd)
    optimal = None
    if energies:
        min_eng = min(energies)
        min_idx = energies.index(min(energies))
        if interactive:
            print("Minimum is {} at {}".format(paths[min_idx], energies[min_idx]))
            exit()
        else:
            pwd = os.getcwd()
            os.chdir(paths[min_idx])
            calc = Vasp2(restart=True)
            atoms = calc.get_atoms()
            os.chdir(pwd)
            return atoms
    else:
        return None

def atoms_count(atoms, adsorbate, metal):
    num_A = len([a for a in atoms if a.symbol == adsorbate])
    num_M = len([a for a in atoms if a.symbol == metal])
    return num_A, num_M

def read_free(acc_dir, vib_dir, temperature, pressure, metal, adsorbate=None, coverages=None):
    # print(coverages)
    pwd = os.getcwd()

    G = np.array([])
    num_ads = np.array([])

    os.chdir(pwd)

    cov_energy = {}
    if not coverages: # bare
        [atoms, pot_en, vib_en] = extract_energies(acc_dir, vib_dir, 0)
        free_en = surface_thermo(vib_en,
                                 pot_en,
                                 atoms,
                                 temperature,
                                 pressure)
        num_metal = len([ b for b in atoms if b.symbol == metal ])
        return free_en, num_metal
    else:
        for idx, cov in enumerate(coverages):
            cov_acc = acc_dir + cov
            cov_vib = vib_dir + cov
            #  print(cov_vib)
            [atoms, pot_en, vib_en] = extract_energies(cov_acc, cov_vib, 0)
            num_A, num_M = atoms_count(atoms, adsorbate, metal)
            free_en = surface_thermo(vib_en,
                                     pot_en,
                                     atoms,
                                     temperature,
                                     pressure)
            G = np.append(G, free_en)
            num_ads = np.append(num_ads, num_A)
        return G, num_ads

def surface_thermo(vib_en, pot_en, atoms, temperature, pressure):
    thermo = SimpleCrystalThermo(vib_energies = vib_en,
                                 potentialenergy = pot_en,
                                 atoms = atoms,
                                 geometry = 'nonlinear',
                                 symmetrynumber = 1, spin = 0)
    enthalpy = thermo.get_enthalpy(temperature)
    S_vib = thermo.get_vib_entropy(temperature, pressure)

    free_en = enthalpy - S_vib * temperature
    return free_en

def extract_energies(acc_dir, vib_dir, num_freqs_fixed):
    pwd = os.getcwd()

    os.chdir(acc_dir)
    cache_filename = 'cache.json'
    try:
        with open(cache_filename) as cache_file:
            data = json.load(cache_file)
            pot_en = data['pot_en']
            atoms = read('CONTCAR')
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        if os.path.exists(cache_filename):
            os.remove(cache_filename)
        calc = Vasp2(restart=True)
        atoms = calc.get_atoms()
        pot_en = calc.get_potential_energy()
        with open(cache_filename, 'w') as cache_file:
            json.dump({'pot_en': pot_en}, cache_file)
        
    os.chdir(vib_dir)
    try:
        with open(cache_filename) as cache_file:
            data = json.load(cache_file)
            vib_en = np.array([float(v) for v in data['vib_en']])
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        if os.path.exists(cache_filename):
            os.remove(cache_filename)
        calc = Vasp2(restart=True)
        vib_en = calc.read_vib_freq()[0] # meV
        vib_en_imaginary = calc.read_vib_freq()[1] # meV

        # ensure correct number of modes discarded for gas molecules
        # for surfaces with fixed bottom, keep all real frequencies and shift the lowest ones
        # bulk crystal is treated by phonopy
        num_discard_real = max(0, num_freqs_fixed - len(vib_en_imaginary))
        # if somm is to be discarded:
        if num_discard_real != 0:
            vib_en = vib_en[0:-num_discard_real]
        vib_en = np.array(vib_en)
        vib_en[vib_en < 12] = 12
        vib_en = vib_en/1000
        with open(cache_filename, 'w') as cache_file:
            json.dump({'vib_en': vib_en.tolist()}, cache_file)
    os.chdir(pwd)
    return [atoms, pot_en, vib_en]

def adsorption_energies(bare_path, coverages, adsorbate, metal, ref, raw_energy=False, print_energies=True, workdir = None, prefix = None):
    E_ref = {'N_mol': -16.9713100855191/2, #G, (80C,350torr)
             'O_mol': -10.377593118499604/2, #G, (80C,350torr)
             'N_atom': -.31246907E+01}
    E = []
    # bare is path to the bare directory
    # coverage is a list of relative paths(strings) in which the optimized structures can be found
    pwd = os.getcwd()
    os.chdir(bare_path)
    calc = Vasp2(restart=True)
    bare = calc.get_atoms()
    num_A_bare, num_M_bare = atoms_count(bare, adsorbate, metal)

    if calc.converged:
        bare_energy = calc.get_potential_energy(bare)
    else:
        print("bare is not converged")
        sys.exit()
    # Go through the coverages
    os.chdir(pwd)
    
    cov_energy = {}
    for idx, cov in enumerate(coverages):
        os.chdir(cov)
        if raw_energy:
            cache = "raw.data"
        else:
            cache = "adsorption.data"            
        if os.path.exists(cache):
            #print(os.getcwd())
            f = open(cache,'r')
            contents = f.readlines()
            energy = float(contents[0])
        else:
            f = open(cache,"w+")
            if workdir:
                print('workdir 1')
                if workdir[idx]:
                    #print('no conf_ '+os.getcwd())
                    calc = Vasp2(restart=True)
                    optimal = calc.get_atoms()
                else:
                    #print('conf_ '+os.getcwd())
                    optimal = check_convergence(False, prefix)
            else:
                optimal = check_convergence(False, prefix)
            opt_energy = optimal.get_potential_energy()
            num_A, num_M = atoms_count(optimal, adsorbate, metal)
            assert(num_M == num_M_bare)
            if raw_energy:
                energy = opt_energy
            else:
                num_A_ads = num_A - num_A_bare
                energy = (opt_energy - bare_energy - E_ref[adsorbate+ref] * num_A_ads) / num_A_ads
            f.write(str(energy))
        if print_energies:
            print(energy)
        E.append(energy)
        f.close()
        os.chdir(pwd)

    return E



# def adsorption_energies(bare_path, coverages, adsorbate, metal, ref, raw_energy=False, print_energies=True):
#     E_ref = {'N_mol': -.16611553E+02/2,
#              'N_atom': -.31246907E+01}
#     E = []
#     # bare is path to the bare directory
#     # coverage is a list of relative paths(strings) in which the optimized structures can be found
#     pwd = os.getcwd()
#     os.chdir(bare_path)
#     calc = Vasp2(restart=True)
#     bare = calc.get_atoms()
#     num_A_bare, num_M_bare = atoms_count(bare, adsorbate, metal)

#     if calc.converged:
#         bare_energy = calc.get_potential_energy(bare)
#     else:
#         print("bare is not converged")
#         sys.exit()
#     # Go through the coverages
#     os.chdir(pwd)

#     cov_energy = {}
#     for idx, cov in enumerate(coverages):
#         os.chdir(cov)
#         if raw_energy:
#             cache = "raw.data"
#         else:
#             cache = "adsorption.data"            
#         if os.path.exists(cache):
#             f = open(cache,'r')
#             contents = f.readlines()
#             energy = float(contents[0])
#         else:
#             f = open(cache,"w+")
#             optimal = check_convergence(False)
#             opt_energy = optimal.get_potential_energy()
#             num_A, num_M = atoms_count(optimal, adsorbate, metal)
#             assert(num_M == num_M_bare)
#             if raw_energy:
#                 energy = opt_energy
#             else:
#                 num_A_ads = num_A - num_A_bare
#                 energy = (opt_energy - bare_energy - E_ref[adsorbate+ref] * num_A_ads) / num_A_ads
#             f.write(str(energy))
#         if print_energies:
#             print(energy)
#         E.append(energy)
#         f.close()
#         os.chdir(pwd)

#     return E

def get_area(bare_dir):
    pwd = os.getcwd()

    os.chdir(bare_dir)
    calc = Vasp2(restart=True)
    atoms = calc.get_atoms()
    cs_area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))

    os.chdir(pwd)
    return cs_area
