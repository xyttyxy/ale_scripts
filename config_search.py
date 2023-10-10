import pandas as pd
import numpy as np
from ase.calculators.vasp import Vasp
from ase.thermochemistry import IdealGasThermo
from ase import units
from ase.io import read
import json
import os


def frac_to_float(pd_srs):
    """Converts pd.Series fields with text literal fraction to floating-point"""
    for key in pd_srs.keys():
        if 'a_' in key:
            val = pd_srs[key]
            if isinstance(val, str) and '/' in val:
                val_splt = val.split('/')
                assert len(val_splt) == 2, f'value {val} not recognizable as a fraction'
                val_ret = float(val_splt[0]) / float(val_splt[1])
            else:
                val_ret = float(val)
            pd_srs[key] = val_ret
        else:
            continue


def etching_energy(surface_name, rxn_name, 
                   rxn_db, surf_db, gas_db, bulk_db, 
                   condition, 
                   free_energy=True):
    """Calculates etching free energies according to layer model"""
    # etching energy of Ni_s|Ni_x N_y -> 
    rxn_data = pd.read_table(rxn_db, delimiter=',', index_col=False)
    # bare surface reference is defined in surf_db
    # etchants, complex, bulk ref., and rxn. stoich. defined in rxn_db
    rxn_row = rxn_data.loc[rxn_data['name'] == rxn_name].iloc[0]
    frac_to_float(rxn_row)
    
    # ab-initio thermodynamics calculator
    bulkthermo = BulkThermo(bulk_db, free_energy)
    surfacethermo = SurfaceThermo(surf_db, free_energy)
    gasthermo = GasThermo(gas_db, free_energy)
    
    # evaluate energy/free_energy
    E_surf = surfacethermo.energy(surface_name, condition)
    E_bulk = bulkthermo.energy(rxn_row['name_bulk'], condition)
    E_etchant = gasthermo.energy(rxn_row['name_etchant'], condition)
    E_complex = gasthermo.energy(rxn_row['name_complex'], condition)
    E_hydride = gasthermo.energy(rxn_row['name_hydride'], condition)
    
    n_mod = read_db(surf_db, surface_name, 'n_mod_total')
    E_bare = surfacethermo.energy(read_db(surf_db, surface_name, 'name_bare'), condition)
    E_etch = -E_bulk \
        + (rxn_row['a_adsorbed'] / n_mod) * (E_bare-E_surf) \
        + (rxn_row['a_complex'] * E_complex + rxn_row['a_hydride'] * E_hydride - rxn_row['a_etchant'] * E_etchant)
    return E_etch


def read_db(db, name, key):
    """reads one field (row & column) from a db file (.csv)"""
    data = pd.read_table(db,
                         delimiter=',',
                         index_col=False)
    return data.loc[data['name'] == name].iloc[0][key]


class AbstractThermo():
    def __init__(self, db, free_energy, verbose=False):
        data = pd.read_table(db,
                             delimiter=',',
                             index_col=False)
        # replace empty directories
        data.loc[data['en_dir'].isnull(),'en_dir'] = ''
        data.loc[data['vib_dir'].isnull(),'vib_dir'] = ''
        # convert list
        from ast import literal_eval
        
        if 'supercell' in data.columns:
            data.supercell = data.supercell.apply(literal_eval)
        self.db = data
        self.free_energy = free_energy
        self.verbose = verbose

    def energy(self, name, condition):
        if self.free_energy:
            return self.get_free_energy(name, condition)
        else:
            return self.get_potential_energy(name)

    def get_potential_energy(name):
        raise NotImplementedError
    
    def get_free_energy(name, condition):
        raise NotImplementedError
    
    @staticmethod
    def read_vib(vib_dir, **kwargs):
        """Read vibrational modes in eV"""
        key = 'vib_en'
        val = read_with_cache(vib_dir, key, AbstractThermo.vib_parser, **kwargs)
        vib_en = np.array(val)
        return vib_en

    @staticmethod
    def vib_parser(path, num_freqs_fixed, **kwargs):
        """Parses VASP vibrational modes calculaton"""
        calc = Vasp(restart=True, directory = path)
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

        return vib_en.tolist()

    @staticmethod
    def read_en(en_dir):
        """Read electronic potential energy from VASP working folders"""
        key = 'pot_en'
        val = read_with_cache(en_dir, key, AbstractThermo.en_parser)
        return val

    @staticmethod
    def en_parser(path, **kwargs):
        """Parses VASP energy calculation"""
        calc = Vasp(restart=True, directory=path)
        pot_en = calc.get_potential_energy()
        return pot_en
    

class BulkThermo(AbstractThermo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_potential_energy(self, name):
        row = self.db.loc[self.db['name'] == name].iloc[0]
        en = AbstractThermo.read_en(row.en_dir)
        return en / row['formula_units']

    def get_free_energy(self, name, condition):
        """thermodynamics of bulk crystal using full phonon dispersion"""
        import phonopy
        # assuming no duplicated names
        row = self.db.loc[self.db['name'] == name].iloc[0]
        en = AbstractThermo.read_en(row.en_dir)
        phonon = phonopy.load(phonopy_yaml=f'{row.vib_dir}/phonopy.yaml',
                              supercell_matrix=row.supercell,
                              unitcell_filename=f'{row.vib_dir}/POSCAR-unitcell',
                              force_constants_filename=f'{row.vib_dir}/FORCE_CONSTANTS')

        phonon.run_mesh([20, 20, 20])
        phonon.run_thermal_properties(t_step=0,
                                      t_max=condition['T'],
                                      t_min=condition['T'])
        tp_dict = phonon.get_thermal_properties_dict()
        kj2ev = 1.0364e-2
        free_energy_correction = tp_dict['free_energy'][0]

        return (en + free_energy_correction*kj2ev) / row['formula_units']
    

class SurfaceThermo(AbstractThermo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_potential_energy(self, name):
        row = self.db.loc[self.db['name'] == name].iloc[0]
        en = AbstractThermo.read_en(row.en_dir)
        return en
    
    def get_free_energy(self, name, condition):
        row = self.db.loc[self.db['name'] == name].iloc[0]
        en_dir = row['en_dir']
        vib_dir = row['vib_dir']

        vib_en = AbstractThermo.read_vib(vib_dir, num_freqs_fixed=0)
        atoms = read(f'{en_dir}/CONTCAR')
        en = AbstractThermo.read_en(en_dir)

        # thermochemistry
        thermo = SimpleCrystalThermo(vib_energies = vib_en,
                                     potentialenergy = en,
                                     atoms = atoms,
                                     geometry = 'nonlinear',
                                     symmetrynumber = 1, spin = 0)
        G = thermo.get_gibbs_energy(condition['T'], condition['P'], self.verbose)
        return G


class GasThermo(AbstractThermo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_potential_energy(self, name):
        row = self.db.loc[self.db['name'] == name].iloc[0]
        en = AbstractThermo.read_en(row.en_dir)
        return en / row['nuclearity']

    def get_free_energy(self, name, condition):
        row = self.db.loc[self.db['name'] == name].iloc[0]
        en_dir = row['en_dir']
        vib_dir = row['vib_dir']
        vib_en = AbstractThermo.read_vib(vib_dir, num_freqs_fixed=row['num_freqs_fixed'])

        if row['linearity'] == 'moatomic':
            # no vibrations
            vib_en = [0]
        else:
            # read vib modes
            vib_en = AbstractThermo.read_vib(vib_dir, num_freqs_fixed=row['num_freqs_fixed'])

        atoms = read(f'{en_dir}/CONTCAR')
        en = AbstractThermo.read_en(en_dir)

        # thermochemistry
        thermo = IdealGasThermo(vib_energies=vib_en,
                                potentialenergy=en,
                                atoms=atoms,
                                geometry=row['linearity'],
                                symmetrynumber=row['symmetrynumber'],
                                spin=0)

        G = thermo.get_gibbs_energy(condition['T'],
                                    condition['P'],
                                    self.verbose)
        return G / row['nuclearity']


def read_with_cache(path, key, raw_parser, **kwargs):
    """abstract reader with transparent cache handling"""
    cache_filename = 'cache.json'
    cache_path = f'{path}/{cache_filename}'
    try:
        with open(cache_path, 'r') as cache_file:
            data = json.load(cache_file)
        val = data[key]
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        if os.path.exists(cache_path):
            os.remove(cache_path)
        
        val = raw_parser(path, **kwargs)
        data = {key: val}
        with open(cache_path, 'w') as cache_file:
            json.dump(data, cache_file)

    return val


class SimpleCrystalThermo(IdealGasThermo):
    """"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _vprint(self, text):
        if self.verbose:
            print(text)
        
    def get_entropy(self, temperature, pressure, verbose=False):
        """Returns the entropy, in eV/K, in the ideal gas approximation
        at a specified temperature (K) and pressure (Pa)."""
        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'

        S = 0.0
        S_v = self._vibrational_entropy_contribution(temperature)
        write(fmt % ('S_vib', S_v, S_v * temperature))
        S += S_v
        
        return S

    def get_enthalpy(self, temperature, verbose=False):
        """Returns the enthalpy, in eV, in the ideal gas approximation
        at a specified temperature (K)."""
        self.verbose = verbose        
        write = self._vprint
        fmt = '%-15s%13.3f eV'
        
        H = 0.
        
        H += self.potentialenergy
        write(fmt % ('E_pot', self.potentialenergy))

        zpe = self.get_ZPE_correction()
        write(fmt % ('E_ZPE', zpe))
        H += zpe
        
        dH_v = self._vibrational_energy_contribution(temperature)
        write(fmt % ('Cv_vib (0->T)', dH_v))
        H += dH_v
        
        Cp_corr = units.kB * temperature
        write(fmt % ('(C_v -> C_p)', Cp_corr))
        H += Cp_corr
        
        return H
