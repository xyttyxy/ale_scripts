from config_search import GasThermo, SurfaceThermo, BulkThermo, etching_energy
import pandas as pd


def test_gases():
    gases = ['FA',
             'NiFA2',
             'Ni2FA4',
             'Ni3FA6',
             'Ni4FA8_1',
             'Ni4FA8_2',
             'NH3',
             'H2',
             'N2']
    for gas in gases:
        gasthermo = GasThermo('/home/xyttyxy/Work/ALE/nin/gas/mole_configs.csv',
                              free_energy=True,
                              verbose=False)
        G = gasthermo.energy(name = gas,
                             condition = {'T': 80,
                                          'P': 350})
        print(f'{gas}: {G:.2f}')


def test_surfaces():
    db = '/home/xyttyxy/Work/ALE/nin/high_coverage/surf_configs.csv'
    data = pd.read_table(db,
                         delimiter=',',
                         index_col=False)
    names = data['name']
    names = [elm for elm in names if '100' in elm and 'bare' not in elm]
    surfacethermo = SurfaceThermo(db, free_energy=True)
    for name in names:
        G = surfacethermo.energy(name, condition={'T': 80, 'P': 350})
        print(f'{name}: {G:.2f}')


def test_bulk_thermo():
    db = '/home/xyttyxy/Work/ALE/nin/bulk/bulk_configs.csv'
    data = pd.read_table(db,
                         delimiter=',',
                         index_col=False)
    names = data['name']
    bulkthermo = BulkThermo(db, free_energy=True)
    for name in names:
        G = bulkthermo.energy(name, condition={'T': 80, 'P': 350})
        print(f'{name}: {G:.2f}')


def test_etching_energy():
    G_etch = etching_energy('Ni_100_mixed_1_2ML_1_2ML_1',
                            'Ni_etch_N_FA',
                            '/home/xyttyxy/Work/ALE/nin/high_coverage/rxn_configs.csv',
                            '/home/xyttyxy/Work/ALE/nin/high_coverage/surf_configs.csv',
                            '/home/xyttyxy/Work/ALE/nin/gas/mole_configs.csv',
                            '/home/xyttyxy/Work/ALE/nin/bulk/bulk_configs.csv',
                            condition = {'T': 80,
                                         'P': 350},
                            free_energy=True)
    print(G_etch)
