from ase import Atom, Atoms
from ase.io import read
from ase.visualize import view
from ase.data import covalent_radii, atomic_numbers, vdw_radii
from numpy import arange
import os, sys

from utilities.autoads import *
sys.path.append('/oasis/scratch/comet/xyttyxyx/temp_project/coverage_search/Cu/100/1N/')
import ads_1N

if __name__ == '__main__':
    args = sys.argv
else:
    args = []

def next_ads():
    # 1. select last_opt
    last_opt = read('/oasis/scratch/comet/xyttyxyx/temp_project/coverage_search/Cu/100/1N/finished/t3/CONTCAR')
    # 2. select last kit
    last_kit = ads_1N.get_kit(3)
    # 3. choose adsorbate
    adsorbate = Atoms('N')

    coords, connect = next_site(last_kit)

    if '--cds' in args:
        print("All Coords: \n", coords)
        print("All Connectivity: \n", connect)
        sys.argv = []
    # 4. change mask
    ## MANUAL MASKING
    # tolerance = 0
    # mask = masking(len(connect), [2,4,5])
    ## AUTOMATIC MASKING
    distance = covalent_radii[atomic_numbers['N']] + covalent_radii[atomic_numbers['Cu']]
    distance *= 1
    mask = [c == 4 for c in connect]

    next_opt, next_kit = build_next(last_opt=last_opt, last_kit=last_kit,
                                    adsorbate=adsorbate, mask=mask, distance=distance)

    return next_opt, next_kit
opt, kit = next_ads()
#################### BELOW DOES NOT CHANGE ####################
if '-w' in args:
    write_imgs(opt)

if '--vnkit' in args:
    view([kit[i] for i, l in enumerate(kit) if isinstance(l, Atoms)])

if '--vnopt' in args:
    view([l for l in opt if isinstance(l, Atoms)])
    
def get_kit(idx = None):
    if idx != None:
        return kit[idx]
    else:
        return kit


