from kart_helpers import match_subgraph, mcs_xyt
import pickle
import numpy as np
import random

def compute(selected):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sig_term = -2
    sig_cont = 0
    comm_tags = {'a0_m2s': 0,
                 'a1_m2s': 1,
                 'result_s2m': 4,
                 'signal_m2s': 5}

    if rank == 0:
        ngraphs = len(selected)
        pool_size = comm.Get_size()
        nworkers = pool_size - 1
        data = np.zeros((ngraphs, ngraphs), dtype='f')
        data_states = -np.ones((ngraphs, ngraphs), dtype='i') # -1: not yet visited, 0: running, 1: finished
        worker_state = np.zeros(nworkers, dtype='b') # False: idle, True: working
        worker_indices = [[-1, -1]] * nworkers # -1: idle, integer: working on this index
        worker_requests = [object()] * nworkers
        while True:
            # find indices not yet worked on and dispatch
            idx_unfinished = np.argwhere(data_states != 1)
            if len(idx_unfinished) == 0:
                # all worker nodes are always receiving after done.
                # You need to send a terminating signal to each worker
                for worker_idx in range(nworkers):
                    worker_rank = worker_idx + 1
                    print('trying to kill ', worker_rank)
                    comm.send(sig_term, worker_rank, tag = comm_tags['signal_m2s'])
                # then exit the master querying loop
                break

            idx_torun = np.argwhere(data_states == -1)
            for idx_job in idx_torun:
                a0 = selected[idx_job[0]]
                a1 = selected[idx_job[1]]
                
                idle_workers = np.argwhere(worker_state == 0)
                if len(idle_workers) > 0:
                    worker_idx = idle_workers[0][0]
                    worker_rank = worker_idx + 1
                    comm.send(sig_cont, dest=worker_rank, tag = comm_tags['signal_m2s'])
                    data_states[idx_job[0], idx_job[1]] = 0
                    comm.send(a0, dest=worker_rank, tag = comm_tags['a0_m2s'])
                    comm.send(a1, dest=worker_rank, tag = comm_tags['a1_m2s'])
                    # print('send 1', worker_rank)
                    worker_indices[worker_idx] = idx_job
                    worker_state[worker_idx] = True
                    worker_requests[worker_idx] = comm.irecv(source=worker_rank, tag = comm_tags['result_s2m'])
                else:
                    break

            for worker_idx in range(nworkers):
                req = worker_requests[worker_idx]
                if isinstance(req, Request):
                    status = req.test()
                    if status[0]:
                        # job completed
                        req.wait()
                        # print('irecv 2')
                        data_idx = worker_indices[worker_idx]
                        data[data_idx[0], data_idx[1]] = status[1]
                        data_states[data_idx[0], data_idx[1]] = 1
                        worker_state[worker_idx] = 0
                        worker_indices[worker_idx] = -1
                        worker_requests[worker_idx] = object()
        return data
    
    else:
        while True:
            signal = comm.recv(source=0, tag = comm_tags['signal_m2s'])
            if signal == sig_term:
                break

            a0 = comm.recv(source=0, tag = comm_tags['a0_m2s'])
            a1 = comm.recv(source=0, tag = comm_tags['a1_m2s'])
            g0 = a0.info['graph']
            g1 = a1.info['graph']
            center0 = a0.info['center_atom']
            center1 = a1.info['center_atom']
            # print('recv 1', rank)
            g0_sub, g1_sub, solution_size = mcs_xyt(g0, g1, center0, center1, rank)
            d = distance(a0, a1, g0_sub, g1_sub, rank)
            req = comm.isend(d, dest=0, tag = comm_tags['result_s2m'])
            req.wait()
            # print('isend 2', after)
            print(f'rank {rank} working')
            

def distance(G_atoms, H_atoms, G_sub, H_sub, rank):
    # from mcs we can extract the non-matched atoms
    # those atoms should serve to similarity lower / distance higher
    # and their influence should depend on their distance to the center atom
    
    G = G_atoms.info['graph']
    H = H_atoms.info['graph']

    if len(G_sub) == 0:
        unmatched_G = G
    else:
        unmatched_G = match_subgraph(G_sub, G, identifier=rank, rematch=False)

    if len(H_sub) == 0:
        unmatched_H = H
    else:
        unmatched_H = match_subgraph(H_sub, H, identifier=rank, rematch=False)
        
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

if __name__ == '__main__':
    random.seed(114514)
    
    with open('initial.pickle', 'rb') as handle:
        initial_configs = pickle.load(handle)
    selected = random.sample(initial_configs, 100)
    mode = 'partest'

    if mode == 'test':
        idx_job = [0, 2]
        g0 = selected[idx_job[0]].info['graph']
        g1 = selected[idx_job[1]].info['graph']
                    
        center0 = selected[idx_job[0]].info['center_atom']
        center1 = selected[idx_job[1]].info['center_atom']
        symbol0 = selected[idx_job[0]][center0].symbol
        symbol1 = selected[idx_job[1]][center1].symbol
        if symbol0 == symbol1:
            g0_sub, g1_sub, solution_size = mcs_xyt(g0, g1, center0, center1)
        else:
            print(symbol0, symbol1)
    else:
        from mpi4py import MPI
        from mpi4py.MPI import Request
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            data = compute(selected)
            print(data)
            with open('initial_distances.pickle', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            compute(selected)
            

