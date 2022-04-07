import filecmp
import os
import shutil
from itertools import islice
from os.path import isfile, join

import numpy as np
import termplotlib as tpl


# 1. copy the last section of fort.83 to ffield
# read fort.83
def update_83():
    ffield_readin = []
    for idx, line in enumerate(reversed(open("fort.83").readlines())):
        if idx >= 166:
            break
        ffield_readin.append(line.rstrip())

    shutil.copyfile('ffield', 'ffield_save')
    with open ('ffield', 'w+') as ffield_writeout:
        for ff in reversed(ffield_readin):
            ffield_writeout.write(ff+'\n')
    # no need to close explicitly

def save_99():
    # read all files matching pattern 99.fort.*
    save99_olds = [f for f in os.listdir('.') if isfile(join('.', f)) and '99.fort.' in f]

    if save99_olds:
        suffix = max([int(elm.split('.')[-1]) for elm in save99_olds])
        save99_old = '99.fort.'+str(suffix)
        save99_new = '99.fort.'+str(suffix+1)
        if filecmp.cmp(save99_old, 'fort.99'):
            print('already saved in ' + save99_old)
            return
        else:
            print('fort.99 saved in ' + save99_new)
            shutil.copyfile('fort.99', save99_new)
    else:
        print('no previous saves found, writing 99.fort.1')
        shutil.copyfile('fort.99', '99.fort.1')

def check_79(atom, param):
    O_params = np.zeros((200,3))
    Cu_params = np.zeros((200,3))

    with open('fort.79') as fort79:
        O_flags = [False, False, False]
        Cu_flags = [False, False, False]
        O_count = 0
        Cu_count = 0
        for line in fort79:
            line = line.rstrip()
            if ' 2  3 30' in line:
                param_line = list(islice(fort79, 7))[-1]
                O_p1 = float(param_line.rstrip().split()[-1])
                O_flags[0] = True
            if ' 2  3 31' in line:
                param_line = list(islice(fort79, 7))[-1]
                O_p2 = float(param_line.rstrip().split()[-1])
                O_flags[1] = True
            if ' 2  3 32' in line:
                param_line = list(islice(fort79, 7))[-1]
                O_p3 = float(param_line.rstrip().split()[-1])
                O_flags[2] = True
            if all(O_flags):
                O_params[O_count] = np.array([O_p1, O_p2, O_p3])
                O_count += 1
                O_flags = [False, False, False]
            if ' 2  4 30' in line:
                param_line = list(islice(fort79, 7))[-1]
                Cu_p1 = float(param_line.rstrip().split()[-1])
                Cu_flags[0] = True
            if ' 2  4 31' in line:
                param_line = list(islice(fort79, 7))[-1]
                Cu_p2 = float(param_line.rstrip().split()[-1])
                Cu_flags[1] = True
            if ' 2  4 32' in line:
                param_line = list(islice(fort79, 7))[-1]
                Cu_p3 = float(param_line.rstrip().split()[-1])
                Cu_flags[2] = True
            if all(Cu_flags):
                Cu_params[Cu_count] = np.array([Cu_p1, Cu_p2, Cu_p3])
                Cu_count += 1
                Cu_flags = [False, False, False]
    data = {'O': O_params,
            'Cu': Cu_params}
    counters = {'O': O_count,
                'Cu': Cu_count}
    x = np.arange(0, counters[atom])
    y = data[atom][0:counters[atom], param]
    fig = tpl.figure()
    fig.plot(x, y)
    fig.show()


def comp_99():
    fort99_original = {''}
    with open('fort.99') as fort99:
        for line in fort99:
            line = line.rstrip()
            if '+o2_2.9' in line:
                original_val = 129.5314
                tmp = line.split()
                new_val = float(tmp[3])
                if abs(original_val - new_val) > 1:
                    print('Warning: O2 long distance error')

# check constraint is successfully applied
def check_90():
    with open("fort.90", "r") as f:
        geo_file = f.read().splitlines()
        # print(geo_file)
    distances = []
    for linum, x in enumerate(geo_file):
        if linum % 17 == 0:
            atom_1 = geo_file[linum+8].split()
            pos_1 = np.array([float(a) for a in atom_1[3:6]])
            atom_2 = geo_file[linum+9].split()
            pos_2 = np.array([float(a) for a in atom_2[3:6]])
            distances.append(np.linalg.norm(pos_1-pos_2))
        else:
            continue
    return distances
