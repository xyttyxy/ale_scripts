#!/usr/bin/env python
from ase.db import connect
from n2p2 import n2p2_data
import argparse

parser = argparse.ArgumentParser(description='clean and write dft snapshots to n2p2 data')
parser.add_argument('-i', '--input', default='input.data.test')
parser.add_argument('-d', '--database', default='dft.db')
parser.add_argument('-p', '--prefix')
args = parser.parse_args()

input = args.input
db = connect(args.database)
prefix = args.prefix

with open(input, 'w') as input_data:
    for row in db.select():
        name = row.get('name')
        atoms = row.toatoms()
        pot_rpbe = row.get('energy')
        input_data.write(n2p2_data(atoms, pot_rpbe, prefix+'_'+name))
