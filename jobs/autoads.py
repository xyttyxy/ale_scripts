# Yantao Xia 20191114
# Helper functions to automate the optimal coverage search
import copy
from ase.io import read, write
from ase import Atom, Atoms
from ase.data import covalent_radii, atomic_numbers, vdw_radii
import numpy as np
from myase.utils.analysis import check_convergence
from catkit.gen.surface import SlabGenerator
from catkit.gen.adsorption import Builder, AdsorptionSites, get_adsorption_sites
from catkit.gen.utils import to_gratoms
from numpy import arange, array, zeros, linspace
import os, sys
from ase.visualize import view
# recombined only tested for if the last added atoms recombined.
# this is wrong
# instead it must test for all atom pairs
def recombined(atoms, cutoff, symbol = 'N'):
    indices = [a.index for a in atoms if a.symbol == symbol]
    #print(indices)
    recomb = False
    if not indices:
        return False
    for idx, i in enumerate(indices):
        temp = copy.deepcopy(indices)
        temp.remove(i)
        if not temp:
            # only one atom
            return False
        arr = atoms.get_distances(i, temp, mic=True)
        arr = arr[arr >= 0]
        try:
            minimum_distance = min(arr)
        except ValueError:
            if len(arr) == 0:
                return False
            else:
                print("Check! Something's up in myase.utils.autoads.recombined")
                exit()
        if min(arr) < cutoff:
            recomb = True
    return recomb

def view_opt(idx):    view(next_opt[idx])
def view_kit(idx):    view(next_kit[idx])

def write_imgs(structure, path = None, prefix='t'):
    # write next_struct to a folder structure
    pwd = os.getcwd()
    
    if path:
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
    for i, e in enumerate(structure):
        if not isinstance(e, Atoms):
            continue
        label = prefix+str(i)
        if not os.path.exists(label):
            os.mkdir(label)
        e.write(filename=label+'/before.vasp')
    os.chdir(pwd)

def view_imgs(structure):
    view([s for s in structure if isinstance(s, Atoms)])
    
def masking(length, lst):
    tolerance = 0
    indices = array(lst)
    mask = zeros(length, dtype=bool)
    mask[indices] = True
    mask = mask.tolist()
    return mask

def mask_on_mask(old_mask, lst):
    i = 0
    mask = []
    for idx, elm in enumerate(old_mask):
        if elm == True:
            if i not in lst:
                mask.append(False)
            else:
                mask.append(True)
            i += 1
        else:
            mask.append(False)
    return mask

def next_opt(last_opt, sites, adsorbate, opt_vecs, distance, search):
    """Build slabs with one more adsorbant from optimized slabs
    
    Does not work with adsorbates with more than one atom

    Parameters
    ----------
    previous_opt: optimized geometry of last step
    
    sites: list of coordinates of adsorption sites
    
    adsorbate: ASE Atom object

    tol: tolerance before consider duplicate sites

    Returns
    -------
    Next structures to be optimized
    """
    ads_sym = adsorbate[0].symbol
    returned = []
    for i, s in enumerate(sites):
        if np.isnan(s.any()):
            returned.append(object())
            continue
        candidate = copy.deepcopy(last_opt)
        if search:
            # linear search along the adsorption vector
            search_success = False
            for l in linspace(0, 4, 30):
                adsorbate[0].position = s + opt_vecs[i] * l
                candidate += adsorbate
                indices = [c.index for c in candidate if c.symbol != ads_sym]
                closest_distance = min(candidate.get_distances(-1, indices,mic=True))
                if closest_distance > distance:
                    search_success = True
                    break
                else:
                    del candidate[-1]

            # check candidate adsorbate distance larger than threshold
            if search_success and not recombined(candidate, recomb_length, ads_sym):
                returned.append(candidate)
            else:
                returned.append(object())
        else:
            adsorbate[0].position = s + opt_vecs[i] * distance
            candidate += adsorbate
            returned.append(candidate)
    return returned

def next_kit(last_kit, idx, adsorbate, tol = 0.1, search = True):
    """Build slabs with one more adsorbant from 
    clean catkit gratoms slabs
    
    Parameters
    ----------
    previous_kit: catkit generated gratoms(with the networkx graph)
    
    idx: list of indices of adsorption sites 
    Assuming catkit.gen.adsorption.get_adsorption_sites find sites in the same
    order as catkit.gen.adsorption.Builder.add_adsorbate. 
    
    adsorbate: ASE Atoms

        tol: tolerance before considering duplicate sites
    
        Returns
        -------
        Gratoms used in site search in the next iteration
        """
    builder = Builder(slab=last_kit)
    adsorbate = to_gratoms(adsorbate)
    returned = []
    ads_sym = adsorbate[0].symbol
    for i in idx:
        if i == -1:
            returned.append(object())
            continue
        candidate = builder.add_adsorbate(adsorbate=adsorbate, bonds=[0], index=i, auto_construct=True)
        returned.append(candidate)

    return returned
    
def next_site(last_kit):
    coords, connect = get_adsorption_sites(last_kit, symmetry_reduced=True)
    return coords, connect
    
def build_next(last_opt, last_kit, adsorbate, distance, mask = None, search=True):
    coords, connect = get_adsorption_sites(last_kit, symmetry_reduced=True)
    AdsSites = AdsorptionSites(slab = last_kit)
    adsorption_vectors = AdsSites.get_adsorption_vectors()
    print(coords, connect)
    if mask != None:
        mask = [not m for m in mask]

        kit_indices = copy.deepcopy(arange(len(coords)))
        kit_indices[mask] = -1

        opt_coords = copy.deepcopy(coords)
        opt_coords[mask] = np.nan
        opt_vecs = copy.deepcopy(adsorption_vectors)
        opt_vecs[mask] = np.nan
    else:
        opt_coords = coords[mask][0]
        kit_indices = arange(len(coords))
        opt_vecs = adsorption_vectors[mask][0]

    if distance != None:
        opt = next_opt(last_opt, opt_coords, adsorbate, opt_vecs, distance, search)
        kit = next_kit(last_kit, kit_indices, adsorbate, distance, search)
    else:
        opt = next_opt(last_opt, opt_coords, adsorbate, opt_vecs, distance, search)
        kit = next_kit(last_kit, kit_indices, adsorbate, distance, search)

    # match up the opt and kit indices
    for i, o in enumerate(opt):
        if not isinstance(o, Atoms):
            kit[i] = object()
    return opt, kit

    
def adjust_vac(atoms, new_vac):
    cell = atoms.get_cell()
    bottom = min(atoms.get_positions()[:,2])
    atoms.translate([0,0,-bottom])
    height = max(atoms.get_positions()[:,2])
    cell[2][2] = height + new_vac*2
    atoms.set_cell(cell)
    atoms.center()
    return atoms

def accurate_atoms(path_to_fin_folder, prefix):
    # identify optimal configuration
    pwd = os.getcwd()
    os.chdir(path_to_fin_folder)
    atoms = check_convergence(False, prefix)
    os.chdir(pwd)
    return adjust_vac(atoms, 15)

# find the bond length
def get_bond_length(metal, adsorbate):
    ads_rad = covalent_radii[atomic_numbers[adsorbate]]
    metal_rad = covalent_radii[atomic_numbers[metal]]
    return ads_rad + metal_rad


def line_search(slab, raw_site, adsorbate, ads_vec = np.array([0,0,1]), bond_length = None):
    # find the metal species
    syms = [s.symbol for s in slab]
    metal_sym = max(set(syms), key = syms.count)
    metal_pos = [s.position for s in slab if s.symbol == metal_sym]
    metal_idx = [s.index for s in slab if s.symbol == metal_sym]

    if bond_length == None:
        bond_length = get_bond_length(metal_sym, adsorbate.symbol)
    if bond_length == 0:
        return raw_site
    # do the linear search
    # cannot just use euclidean distance because of pbc
    # have to use mic=True, minimum image convention
    # todo: speed it up by geometrically calculating the distance
    ads_pos = raw_site
    success = False
    for l in np.linspace(0, 4, 40):
        ads_pos = raw_site + ads_vec * l
        adsorbate.position = ads_pos
        slab += adsorbate
        closest_distance = min(slab.get_distances(-1, metal_idx, mic=True))
        del slab[-1]
        if closest_distance > bond_length:
            success = True
            break

    if success:
        return ads_pos
    else:
        print('Site line search failed. This should not happen! Exiting...')
        return np.array([None])

def next_ads(last_opt, prim_slab, repeat, prim_vec, prim_site, adsorbate, occ = [-1], cutoff = None):
    x_ax = prim_slab.get_cell()[0]
    y_ax = prim_slab.get_cell()[1]

    candidate = []
    ads_sym = adsorbate.symbol
    if cutoff:
        recomb_length = cutoff
    else:
        recomb_length = get_bond_length(ads_sym, ads_sym)

    for idx, p in enumerate(prim_site):
        for i in range(0, repeat[0]):
            for j in range(0, repeat[1]):
                if repeat[0]*repeat[1]*idx + i*repeat[1] + j in occ:
                    
                    candidate.append(object())
                    continue
                shift = i*x_ax + j*y_ax
                adsorbate.position = p + shift
                temp = copy.deepcopy(last_opt)

                temp += adsorbate
                
                if recombined(temp, recomb_length, ads_sym):
                    candidate.append(object())
                    continue
                else:
                    candidate.append(temp)
    return candidate

def prim_sites(prim_slab, adsorbate, site_idx, bond_length=None):
    raw_sites, connect = next_site(prim_slab)
    ads_sites_obj = AdsorptionSites(slab = prim_slab)
    ads_vec = ads_sites_obj.get_adsorption_vectors()
    prim_site = []
    prim_vec = []

    # print(raw_sites)
    for idx, c in enumerate(raw_sites):
        if site_idx == [-1]:
            pass
        elif idx not in site_idx:
            continue
        ads_pos = line_search(prim_slab,
                              c,
                              adsorbate,
                              ads_vec[idx],
                              bond_length)

        if len(ads_pos) < 3:
            continue
        prim_site.append(ads_pos) # do your linear search here
        prim_vec.append(ads_vec[idx])

    return [prim_site, prim_vec]

def init_config(site_idx, repeat, adsorbate, prim, first_iter = False, do_write = False):
    supercell = prim.repeat(repeat)
    if first_iter:
        site_idx = [-1]
        repeat = (1,1,1)
    [prim_site, prim_vec] = prim_sites(prim, adsorbate, site_idx)
    iter_1 = next_ads(supercell, prim, repeat, prim_vec, prim_site, adsorbate)

    if first_iter:
        if do_write:
            write_imgs(iter_1, path = 'iter_1/', prefix = 'conf_')
            exit()
    return [iter_1, prim_site, prim_vec]

def conf_or_acc(iter_1, select, setup, start_pos = 0, write_acc = True, do_write = False, cutoff = None):
    prim = setup[0]
    prim_site = setup[1]
    prim_vec = setup[2]
    repeat = setup[3]
    adsorbate = setup[4]
    slab_ads = [iter_1]

    for idx, s in enumerate(select):
        if idx < start_pos:
            continue

        acc = slab_ads[-1][s]
        sel = select[start_pos:idx+1]

        temp = next_ads(acc, prim, repeat, prim_vec, prim_site, adsorbate, sel, cutoff)
        if do_write and write_acc:
            write('iter_'+str(idx+1)+'/acc/POSCAR', acc)
        slab_ads.append(temp)    

    if do_write and not write_acc:
        write_imgs(slab_ads[-1], path = 'iter_'+str(len(select)+1)+'/', prefix='conf_')

    return slab_ads

def add_site_idx(new_idx, setup):
    prim = setup[0]
    adsorbate = setup[4]
    [prim_site_new, prim_vec_new] = prim_sites(prim, adsorbate, new_idx)
    setup[1] += prim_site_new
    setup[2] += prim_vec_new

def site_dict(metal, facet, sites):
    Ni_dict = {'100':{'t': 0,
                      'b': 1,
                      '4h': 2},
               '110': {'t-t':0,
                       't-p':1,
                       'sb':2,
                       'lb':4,
                       '3h':5},
               '111': {'t':0,
                       '3h-hcp':3,
                       '3h-fcc':2},
               '210': {'t-p':2,
                       'b-t':6,
                       'b-p':7,
                       '4h':10,
                       '3h':8,
                       't-p':2,
                       't-t':11}, 
               '211': {'t-s':1,
                       't-p':2,
                       'b-p':3,
                       'b-cs':8,
                       '4h':13,
                       '3h-hcp-s-l':11,
                       '3h-hcp-s-h':10,
                       '3h-fcc-s-l':9,
                       '3h-fcc-s-h':12},
               '221': {'t-t':0,
                       't-p':3,
                       'lb': 8,
                       'b-p': 10,
                       '3h-hcp-s-l':18,
                       '3h-hcp-s-h':13,
                       '3h-fcc-s-m':12,
                       '3h-fcc-s-l':19,
                       '3h-fcc-s-h':14,
                       '3h-fcc-cs':17,
                       '3h-fcc-s-m-new':13,
                       '3h-fcc-s-l-new':14,
                       '3h-fcc-s-h-new':16,
                       '3h-fcc-cs-new':19,
                       '3h-hcp-m-new':12,
                       '3h-hcp-l-new':15,
                       '3h-hcp-h-new':17,
                       '3h-hcp-cs-new': 18},
               '311': {'b-t':5,
                       'b-p':4,
                       '4h':8,
                       '3h-hcp':7,
                       '3h-fcc':6}}
    Cu_dict = {'100':{'t': 0,
                      'b': 1,
                      '4h': 2},
               '110': {'t-t':0,
                       't-p':1,
                       'sb':2,
                       'lb':4,
                       '3h':5},
               '111': {'t':0,
                       '3h-hcp':2,
                       '3h-fcc':3},
               '210': {'t-pb':0,
                       't-p':2,
                       't-t':1,
                       'b-t':5,
                       'b-p':7,
                       '4h':10,
                       '3h':8},
               '211': {'b-cs':8,
                       '4h':13,
                       '3h-hcp-s-l-new':10,
                       '3h-hcp-s-h-new':12,
                       '3h-fcc-s-l-new':9,
                       '3h-fcc-s-h-new':11,
                       '3h-hcp-s-l':11,
                       '3h-hcp-s-h':10,
                       '3h-fcc-s-l':9,
                       '3h-fcc-s-h':12},
               '221': {'t-t':0,
                       't-p':3,
                       'lb': 8,
                       '3h-hcp-s-l-new': 16,
                       '3h-hcp-s-l':18,
                       '3h-hcp-s-h':13,
                       '3h-hcp-cs': 15,
                       '3h-fcc-s-m':12,
                       '3h-fcc-s-l':19,
                       '3h-fcc-s-h':14,
                       '3h-fcc-cs':17},
               '311': {'b-t':5,
                       'b-p':4,
                       '4h':8,
                       '3h-hcp':7,
                       '3h-fcc':6}}
    big_dict = {'Ni': Ni_dict,'Cu': Cu_dict}
    
    site_idx = []
    for s in sites:
        idx = big_dict[metal][facet][s]
        if isinstance(idx, list):
            site_idx += idx
        else:
            site_idx += [idx]
    return site_idx
            
