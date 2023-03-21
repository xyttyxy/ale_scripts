from kart_helpers import calculate_mcs, match_subgraph, find_topo_ids, find_atom_id_for_event, find_index_in_cluster, atoms2nx
from joblib import Parallel, delayed
import sqlite3
import pickle
import numpy as np
import pandas as pd
import subprocess
from ase.io import read, write
import re
import os
import networkx as nx
import random 
import matplotlib.pyplot as plt
    

def distance(G_atoms, H_atoms, G_sub, H_sub):
    # from mcs we can extract the non-matched atoms
    # those atoms should serve to similarity lower / distance higher
    # and their influence should depend on their distance to the center atom
    
    G = G_atoms.info['graph']
    H = H_atoms.info['graph']

    if len(G_sub) == 0:
        unmatched_G = G
    else:
        unmatched_G = match_subgraph(G_sub, G)

    if len(H_sub) == 0:
        unmatched_H = H
    else:
        unmatched_H = match_subgraph(H_sub, H)
        
    # get_sub = lambda graph, mcs, idx : graph.subgraph([sorted(graph.nodes())[l] for l in [int(edge.split('-')[idx]) for edge in mcs]])
    # G_sub = get_sub(G, mcs, 0)
    # H_sub = get_sub(H, mcs, 1)

    G_nosub = unmatched_G.nodes
    H_nosub = unmatched_H.nodes
    topo_rad = 4.8
    try:
        dist_G = np.array([G_atoms.get_distance(n, G_atoms.info['center_atom'], mic=True) for n in G_nosub])/topo_rad
        dist_H = np.array([H_atoms.get_distance(n, H_atoms.info['center_atom'], mic=True) for n in H_nosub])/topo_rad
    except IndexError:
        breakpoint()
    
    dist = np.concatenate((dist_G, dist_H))
    
    # in practice dist almost always have elements > 0.9 or so
    # these are atoms far away from the center, toward the outer shell
    # these atoms should contribute little to the norm
    # if we do 1-dist, so these are close to 0
    # when we do lp norm, these contribute very little
    # on the other hand, atoms close to the center will be close to 1, and will contribute a lot
    # also, we don't normalize by number of atoms anymore. It seem to mess up things
    p = 2
    if len(dist) == 0:
        # if two graphs are isomorphic, distance is 0
        d = 0.0
    else:
        # if two graphs do not share subgraph at all,
        # then naturally all atoms near center atoms
        # (including the center atom itself)
        # will be included in the distance calculation
        d = (np.sum((1-dist)**p))**(1/p)

    if d < 0 or np.isnan(d):
        breakpoint()
    return d

def parworker(ii, jj, selected):
    conn = sqlite3.connect('mcs.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    G = selected[ii].info['graph']
    H = selected[jj].info['graph']
    # key = sorted([selected[ii].info['topoid'], selected[jj].info['topoid']])
    # key = [str(k) for k in key]
    # key = '_'.join(key)

    key_hash = '_'.join([str(ii), str(jj)])
    sql_check = '''SELECT EXISTS(SELECT 1 FROM MCS WHERE id=?)'''
    sql_insert = '''INSERT INTO MCS(id,topo_i,topo_j,sub_i,sub_j,distance) VALUES(?,?,?,?,?,?)'''
    sql_read = '''SELECT * FROM MCS where id=?'''
    cur = conn.cursor()
    check = cur.execute(sql_check, [key_hash])
    
    if check.fetchone()[0] == 0:
        # key does not exist
        G_sub, H_sub, solution_size = calculate_mcs(G, H)
        # G_sub = nx.Graph()
        # H_sub = nx.Graph()
        # solution_size = 0
        if solution_size == 0:
            d = -1.0
        else:
            d = distance(selected[ii], selected[jj], G_sub, H_sub)
        G_data = pickle.dumps(G_sub, pickle.HIGHEST_PROTOCOL)
        H_data = pickle.dumps(H_sub, pickle.HIGHEST_PROTOCOL)
        cur.execute(sql_insert, (key_hash, selected[ii].info['topoid'], selected[jj].info['topoid'], sqlite3.Binary(G_data), sqlite3.Binary(H_data), d))
        conn.commit()
        print(ii, jj, d)
    else:
        cur.execute(sql_read, [key_hash]) # should only be one row
        record = cur.fetchall()
        
        assert len(record)==1, 'More than one matching row found in database!'
        data_row = record[0]
        G_sub = pickle.loads(data_row[3])
        H_sub = pickle.loads(data_row[4])
        d = data_row[5]
    conn.close()
    return (G_sub, H_sub, d)

if __name__ == '__main__':
    random.seed(114514)

    mode = 'reconstruct'
    if mode == 'calculate':
        for search_id in range(0, 52):
            # search_id = 1 # debug
            directory = f'search_{search_id:04d}'
            try:
                events = [int(d.split('.')[0][5:]) for d in os.listdir(f'{directory}/EVENTS_DIR') if 'event' in d and 'xyz' in d]
                fout = open(f'{directory}/init_saddle_distance', 'w')
                print(f'working on {directory}')
            except FileNotFoundError:
                print(f'{directory} has no events')
                continue
            fout.write('eventid, distance\n')
            for _e in events:
                # _e = 32766 # debug
                init_sd_fin = read(f'{directory}/EVENTS_DIR/event{_e:d}.xyz', index=':')
                initial_topo_id = find_topo_ids(f'{directory}/EVENTS_DIR/event{_e:d}')
                center_atom_id = find_atom_id_for_event(_e, filename=f'{directory}/sortieproc.0')
                initial_index = find_index_in_cluster(f'{directory}/EVENTS_DIR/event{_e:d}', center_atom_id)

                initial = init_sd_fin[0]
                initial.info['graph'] = atoms2nx(initial, 3.5)

                saddle = init_sd_fin[1]
                saddle.info['graph'] = atoms2nx(saddle, 3.5)
                G_sub, H_sub, solution_size = calculate_mcs(initial.info['graph'], saddle.info['graph'])

                initial.info['subgraph'] = G_sub
                saddle.info['subgraph'] = H_sub

                d = distance(initial_index, initial, saddle, G_sub, H_sub)
                if np.isnan(d):
                    breakpoint()
                fout.write(f'{_e}, {d}\n')
            fout.close()
    elif mode == 'plot':
        dists = []
        for search_id in range(0, 52):
            try:
                df = pd.read_table(f'search_{search_id:04d}/init_saddle_distance', delimiter=', ', engine='python')
                dists.append(df)
            except FileNotFoundError:
                print(f'search_{search_id:04d} has no events')
                continue
        dists = pd.concat(dists)['distance']
        plt.hist(dists, bins=30)
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.savefig('init_saddle.png')

    elif mode == 'reconstruct':
        initial_configs = []
        saddle_configs = []
        for search_id in range(0, 52):
            # search_id = 1 # debug
            directory = f'search_{search_id:04d}'
            try:
                events = [int(d.split('.')[0][5:]) for d in os.listdir(f'{directory}/EVENTS_DIR') if 'event' in d and 'xyz' in d]
                fout = open(f'{directory}/init_saddle_distance', 'w')
                print(f'working on {directory}')
            except FileNotFoundError:
                print(f'{directory} has no events')
                continue
            fout.write('eventid, distance\n')
            for _e in events:
                # _e = 32766 # debug
                init_sd_fin = read(f'{directory}/EVENTS_DIR/event{_e:d}.xyz', index=':')
                initial_topo_id, saddle_topo_id = find_topo_ids(f'{directory}/EVENTS_DIR/event{_e:d}')
                center_atom_id = find_atom_id_for_event(_e, filename=f'{directory}/sortieproc.0')
                initial_index = find_index_in_cluster(f'{directory}/EVENTS_DIR/event{_e:d}', center_atom_id)
                
                initial = init_sd_fin[0]
                initial.info['graph'] = atoms2nx(initial)
                initial.info['topoid'] = initial_topo_id
                initial.info['center_atom'] = initial_index
                initial.info['eventid'] = _e
                initial_configs.append(initial)
                
                saddle = init_sd_fin[1]
                saddle.info['graph'] = atoms2nx(saddle)
                saddle.info['center_atom'] = initial_index
                saddle.info['topoid'] = saddle_topo_id
                saddle.info['eventid'] = _e
                saddle_configs.append(saddle)

        with open('initial.pickle', 'wb') as handle:
            pickle.dump(initial_configs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('saddle.pickle', 'wb') as handle:
            pickle.dump(saddle_configs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif mode == 'check':
        with open('initial.pickle', 'rb') as handle:
            initial_configs = pickle.load(handle)
        breakpoint()
    elif mode == 'count':
        df = pd.read_table('all_events_topo', delim_whitespace=True, names=[1,2,3,'init', 'saddle', 'final'])
        initial_count = len(df['init'].value_counts())
        saddle_count = len(df['saddle'].value_counts())
        print(initial_count, saddle_count)
    elif mode == 'dm':
        with open('initial.pickle', 'rb') as handle:
            initial_configs = pickle.load(handle)
        selected = random.sample(initial_configs, 1000)

        numselected = len(selected)

        sql = '''CREATE TABLE IF NOT EXISTS MCS (
        id TEXT PRIMARY KEY, 
        topo_i INTEGER NOT NULL, 
        topo_j INTEGER NOT NULL, 
        sub_i BLOB NOT NULL, 
        sub_j BLOB NOT NULL, 
        distance REAL NOT NULL);
        '''
        conn = sqlite3.connect('mcs.db')
        conn.execute("PRAGMA journal_mode=WAL")
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        conn.close()
        # parworker(0, 1, selected)
        d = Parallel(n_jobs=128)(delayed(parworker)(ii, jj, selected) for ii in range(numselected) for jj in range(ii+1,numselected))



    # for ii in range(len(selected)):
    #     G = selected[ii].info['graph']
    #     for jj in range(ii+1, len(selected)):
    #         H = selected[jj].info['graph']
    #         G_sub, H_sub, solution_size = calculate_mcs(G, H)
    #         d = distance(selected[ii], selected[jj], G_sub, H_sub)
    #         print(ii, jj, d)

