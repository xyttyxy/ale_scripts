#!/bin/python

# this is a template for cleaning scripts which should be placed in sub-dataset folders
# should not be run by itself 
from ase.db import connect

import pdb
def cleaner_demo(atoms):
    return False

def clean_db(db_in, db_out, cleaners = [cleaner_demo]):
    # list of function handles
    with db_out as output:
        for row in db_in.select():
            atoms = row.toatoms()
            flag = False
            # loop over 
            for clnr in cleaners:
                new_flag = clnr(atoms)
                flag = flag or new_flag
                if new_flag:
                    print(row.name, ': ', clnr.__name__)

            if not flag:
                output.write(atoms, name=row.name)
            else:
                print(row.name)

if __name__ == '__main__':
    clean_db()
