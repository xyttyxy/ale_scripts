# Collection of scripts to manipulate graphs using networkx and C++ programs
# Author: Yantao Xia
# Date: 02/2023

import networkx as nx
import pickle
from ase.io import read
from ase.visualize import view
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import subprocess
import random
import os
import re


# Paths to the separately compiled C/C++ programs

glasgow_subgraph_solver = os.getenv('GLASGOW_ROOT')+'/glasgow_subgraph_solver'
glasgow_common_subgraph_solver = os.getenv('GLASGOW_ROOT')+'/glasgow_common_subgraph_solver'
mcsplit_solver = os.getenv('MCSPLIT_ROOT')+'/mcsplit-mcis-sparse'
mcsxyt_solver = os.getenv('MCSXYT_ROOT')+'/mcsp_xyt'


# networkx-based slow toy methods


def nx_modular_product(G, H):
    r"""Returns the modular product of G and H
    
    The modular product $P$ of the graphs $G$ and $H$ has a node set that is
    the cartesian product of the nodes sets, $V(P)=V(G) \times V(H)$.
    $P$ has an edge $((u,v),(x,y))$ if $u$ and $x$ are adjacent in $G$ and $v$
    and $y$ are adjacent in $H$, or $u$ and $x$ are not adjacent in $G$ and
    $u$ and $x$ are not adjacent in $H$. 

    Parameters
    ----------
    G, H: graphs
     Networkx graphs

    Returns:
    ----------
    P: graph
     The modular product of G and H. This function will not work on multigraphs
    or directed graphs.

    Notes:
    ----------
    Node attributes in P are two-tuple of the G and H node attributes.
    Missing attributes are assigned None.
    
    """
    from networkx.algorithms.operators import product as nxp
    breakpoint()
    GH = nxp._init_product_graph(G, H)
    GH.add_nodes_from(nxp._node_product(G, H))

    GH.add_edges_from(nxp._directed_edges_cross_edges(G, H))
    GH.add_edges_from(nxp._undirected_edges_cross_edges(G, H))
    cG = nx.complement(G)
    cH = nx.complement(H)

    GH.add_edges_from(nxp._directed_edges_cross_edges(cG, cH))
    GH.add_edges_from(nxp._undirected_edges_cross_edges(cG, cH))

    return GH


def nx_get_mcs(G, H, which_graph='first'):
    r"""Returns the maximum common node-induced subgraph (MCS) of $G$ and $H$

    The MCS of graphs $G$ and $H$ is an induced subgraph of both $G$ and $H$
    that has as many nodes as possible. This is approximated by calculating
    the maximum clique of the modular product of $G$ and $H$.

    Parameters
    ----------
    G, H: graphs
     Networkx graphs

    which_graph: string
     'first' or 'second'. which one of $G$ or $H$ to use in reconstructing
    the subgraph.

    Returns:
    S: graph
     maximum common subgraph of G or H, depending on choice of which_graph
    """
    from networkx.algorithms.approximation import max_clique
    P = modular_product(G, H)
    maxclique = max_clique(P)
    if which_graph == 'first':
        nodelist = [elm[0] for elm in maxclique]
        retval = G.subgraph(nodelist)
    elif which_graph == 'second':
        nodelist = [elm[1] for elm in maxclique]
        retval = H.subgraph(nodelist)
        
    return retval


def nx_common_edges(G, H):
    r"""Returns the number of common edges after MCS is calculated.

    This is pure-python based and slow, not practical at all. 

    Parameters
    ----------
    G, H: graphs

    Returns: 
    ----------
    The calculated 'distance metric' as the geometric mean of the fractions of common edges
    """
    import numpy as np
    mcs = nx_get_mcs(G, H)
    num_edges_mcs = len(mcs.edges)
    num_edges_G = len(G.edges)
    num_edges_H = len(H.edges)

    metric = np.sqrt(((num_edges_mcs / num_edges_G)**2 + (num_edges_mcs / num_edges_H)**2)/2)
    print(metric)
    

# general utilities


def network_plot_3D(G):
    r"""Plot a graph using 3D position node attributes 
    
    This method allows interactive visualization of connectivity graph, 
    implemented using plotly. The atomic positions must be specified 
    as node attributes under the dictionary key 'pos'. 
    
    Parameters:
    ----------
    G: the input graph, complete with position
    
    Returns:
    ----------
    None
    """
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = G.nodes[edge[0]]['pos']
        x1, y1, z1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)
        
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                              line=dict(width=1, color='#888'),
                              hoverinfo='none',
                              mode='lines')

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z,
                              mode='markers',
                              hoverinfo='text',
                              marker=dict(
                                  showscale=True,
                                  colorscale='YlGnBu',
                                  reversescale=True,
                                  color=[],
                                  size=2,
                                  colorbar=dict(
                                      thickness=15,
                                      title='Node Connections',
                                      xanchor='left',
                                      titleside='right'
                                  ),
                                  line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
                    )
    fig.update_scenes(camera_projection_type='orthographic')
    fig.show()


def get_full_config_graph(full_config='after_opt.vasp', cluster_path='clusters'):
    r""" Construct whole-configuration connectivity network
    
    This script converts the atom-centered cluster connectivity matrices
    written by a modified K-ART program into networkx graphs. The resulting 
    graph is a connectivity matrix spanning the full configuration. 
    See <kart>/src/lib/Find_topos.f90

    Parameters:
    ----------
    full_config: path to a ase-readable file containing a single atomic 
    configuration corresponding to the whole system

    cluster_path: path to the output path of the modified K-ART program. 
    The cluster files are expected to be named index.path where index is 
    the atomic index, starting at 1. Note this is Fortran convention,
    ASE starts at 0. 
    
    Returns:
    --------
    graphs: a single graph corresponding to connectivity of the whole structure.
    """

    G = nx.Graph()
    atoms = read(full_config)
    pos = {a.index: a.position for a in atoms}
    for a in atoms:
        G.add_node(a.index)
        with open(f'{clusters}/{a.index+1}.dat') as f_cluster:
            lines = f_cluster.readlines()

        lines = [[int(e) for e in l.rstrip().split()] for l in lines]
        [[G.add_edge(line[0]-1, l-1) for l in line[1:]] for line in lines]
        
    nx.set_node_attributes(G, pos, 'pos')        
    return G


# kart output parsers


def get_per_atom_graph(full_config='after_opt.vasp', cluster_path='clusters'):
    r""" Convert connectivity graph from K-ART program to networkx objects
    
    This script converts the atom-centered cluster connectivity matrices
    written by a modified K-ART program into networkx graphs.
    See <kart>/src/lib/Find_topos.f90

    Parameters:
    ----------
    full_config: path to a ase-readable file containing a single atomic 
    configuration corresponding to the whole system

    cluster_path: path to the output path of the modified K-ART program. 
    The cluster files are expected to be named index.path where index is 
    the atomic index, starting at 1. Note this is Fortran convention,
    ASE starts at 0. 
    
    Returns:
    --------
    graphs: a list of networkx graphs. The nodes
    """
    atoms = read(full_config)

    graphs = []
    for a in atoms:
        G = nx.Graph()
        with open(f'{cluster_path}/{a.index+1}.dat') as f_cluster:
            lines = f_cluster.readlines()
            
        lines = [[int(e) for e in l.rstrip().split()] for l in lines]

        nodes = []
        for line in lines:
            nodes += line
        nodes = list(set(nodes))

        # atoms.translate([5,5,0])
        # atoms.wrap()

        pos = {atoms[n-1].index: atoms[n-1].position for n in nodes}
        [G.add_node(n-1) for n in nodes]
        
        for line in lines:
            for l in line[1:]:
                G.add_edge(line[0]-1, l-1)

        nx.set_node_attributes(G, pos, 'pos')
        graphs.append(G)

    return graphs
    

# graph file writers


def write_dimacs(G, filename, atoms=None):
    r""" Write DIMACS graph file. 
    
    DIMACS file format is specified at: http://prolland.free.fr/works/research/dsat/dimacs.html
    This script writes a subset of the specs expected by <mcsplit-si>/sparsegraph.c
    
    Parameters:
    ----------
    G: the input networkx graph 

    filename: output filename
    
    atoms: optional, for determining node labels when node 'pos' 
    attribute not present
    """
    mapping = {node: idx for idx, node in enumerate(G.nodes)}
    labels = {'Cu': 0, 'O': 1}
    with open(filename, "w") as f:
        # write the header
        f.write("p edge {} {}\n".format(G.number_of_nodes(), G.number_of_edges()))
        # now write all edges
        for u, v in G.edges():
            f.write(f"e {mapping[u]+1} {mapping[v]+1}\n")
        for n in G.nodes():
            if atoms:
                symbol = atoms[n].symbol
            else:
                symbol = G.nodes[n]['symbol']
            f.write(f"n {mapping[n]+1} {labels[symbol]}\n")


def write_lad(G, filename, atoms=None):
    r""" Write LAD graph file
    
    LAD format examples: https://perso.liris.cnrs.fr/christine.solnon/SIP.html
    
    Parameters:
    ----------
    G: the input networkx graph 

    filename: output filename
    
    atoms: optional, for determining node labels when node 'pos' 
    attribute not present
    """
    # lad file: example
    # actual comment is c and must be on its own line
    # for illustration I use ! for in-line comment

    # the example graph looks like this:
    #    A
    #   / \
    #  B - C
    # the lad file looks like this:
    #
    # 3 ! number of vertices
    # Cu 2 1 2 # comment, number of edges, edge vertices
    # Cu 2 0 2
    # Cu 2 0 1
    
    mapping = {node: idx for idx, node in enumerate(G.nodes)}
    labels = {'Cu': 0, 'O': 1}
    with open(filename, "w") as f:
        # write the header
        f.write(f'{G.number_of_nodes()}\n')
        for n in G.nodes():
            edges = G.edges(n)
            
            if atoms:
                symbol = atoms[n].symbol
            else:
                symbol = G.nodes[n]['symbol']
                
            f.write(f'{labels[symbol]} {len(edges)} ')
            for u ,v in edges:
                f.write(f'{mapping[v]} ')
            f.write(f'\n')
            

def write_gfd(G, filename, atoms=None):
    r""" Write GFD graph file
    
    GFD format specification: https://github.com/InfOmics/RI
    
    Parameters:
    ----------
    G: the input networkx graph 

    filename: output filename
    
    atoms: optional, for determining node labels when node 'pos' 
    attribute not present
    """
    
    mapping = {node: idx for idx, node in enumerate(G.nodes)}
    labels = {'Cu': 0, 'O': 1}
    with open(filename, "w") as f:
        # write the header
        f.write(f'#{filename}\n')
        f.write(f'{G.number_of_nodes()}\n')
        for n in G.nodes():
            if atoms:
                symbol = atoms[n].symbol
            else:
                symbol = G.nodes[n]['symbol']
            f.write(f'{labels[symbol]}\n')
        f.write(f'{G.number_of_edges()}\n')
        for u, v in G.edges():
            f.write(f'{mapping[u]} {mapping[v]}\n')

def write_gfd_with_center(G, filename, center_atom, atoms=None):
    r""" Write GFD graph file that specifies center atom
    
    GFD format specification: https://github.com/InfOmics/RI
    As an extension to the specs here we use a elementwise unique label 
    for the 'center atom'. This is specific to the mcs-based distance
    metric.
    
    Parameters:
    ----------
    G: the input networkx graph 

    filename: output filename

    center_atom: the index of the center atom. This index must correspond 
    to the node label. 
    
    atoms: optional, for determining node labels when node 'pos' 
    attribute not present
    """

    mapping = {node: idx for idx, node in enumerate(G.nodes)}
    labels = {'Cu': 2, 'O': 3, 'center-Cu': 0, 'center-O': 1}
    with open(filename, "w") as f:
        # write the header
        f.write(f'#{filename}\n')
        f.write(f'{G.number_of_nodes()}\n')
        for n in G.nodes():
            if atoms:
                symbol = atoms[n].symbol
            else:
                symbol = G.nodes[n]['symbol']
            if n == center_atom:
                symbol = 'center-'+symbol
                    
            f.write(f'{labels[symbol]}\n')
        f.write(f'{G.number_of_edges()}\n')
        for u, v in G.edges():
            f.write(f'{mapping[u]} {mapping[v]}\n')

def set_topos():
    atoms = read('after_opt.vasp')
    topo_ids = pd.read_table('all_topos', delim_whitespace=True, names=['Assigning', 'topoId', 'id', 'to', 'atom', 'index'])
    unique_ids = list(set(topo_ids['id'].tolist()))
    markers = [unique_ids.index(elm) for elm in topo_ids['id'].tolist()]
    atoms.set_tags(markers)
    return atoms

def parse_xyt(stdout, G, H):
    solution_size = 0
    mapping_linum = -1
    try:
        for line_idx, line in enumerate(stdout):
            if 'Solution size' in line:
                solution_size = int(line.split()[-1])
                mapping_linum = line_idx+1
    except ValueError:
        G_sub = nx.Graph()
        H_sub = nx.Graph()
        return G_sub, H_sub, solution_size

    if solution_size == 0:
        G_sub = nx.Graph()
        H_sub = nx.Graph()
        return G_sub, H_sub, solution_size

    mapping_line = stdout[mapping_linum]
    mapping = mapping_line.split(') (')
    mapping = [re.sub(r'[)|(]', '', elm).split('->') for elm in mapping]
    
    def build_subgraph(mapping, original_graph, index):
        try:
            nodes = [int(elm[index]) for elm in mapping]
        except ValueError:
            [print(m) for m in mapping]
        
        return original_graph.subgraph([list(original_graph.nodes)[idx] for idx in nodes])

    G_sub = build_subgraph(mapping, G, 0)
    H_sub = build_subgraph(mapping, H, 1)
    
    return G_sub, H_sub, solution_size


def mcs_xyt(g0, g1, center0, center1, rank):
    identifier = rank
    f0 = f'temp_g0_{identifier}.gfd'
    f1 = f'temp_g1_{identifier}.gfd'
    write_gfd_with_center(g0, f0, center0)
    write_gfd_with_center(g1, f1, center1)

    proc = subprocess.run([mcsxyt_solver, 
                           '--connected',
                           '-g', # gfd input format
                           'min_max', # heuristic
                           f0, f1], capture_output=True)
    # proc = subprocess.run([mcsplit_solver, 
    #                        '--vertex-labelled-only',
    #                        '-g', # gfd input format
    #                        'A', # heuristic
    #                        f0, f1], capture_output=True)

    stdout = proc.stdout.decode().split('\n')
    os.remove(f0)
    os.remove(f1)
    return parse_xyt(stdout, g0, g1)

def parse_glasgow(stdout, G, H):
    status = any(['status = true' in line for line in stdout])
    assert status, 'glasgow solver failed'
        
    for line in stdout:
        if 'size = ' in line:
            solution_size = int(line.split('=')[1])
        elif 'mapping = ' in line:
            mapping = line.split('=')[1].split(') (')
            mapping = [re.sub(r'[)|(]', '', elm).split('->') for elm in mapping]
            
            def build_subgraph(mapping, original_graph, index):
                nodes = [int(elm[index]) for elm in mapping]
                return original_graph.subgraph([list(original_graph.nodes)[idx] for idx in nodes])
            
            G_sub = build_subgraph(mapping, G, 0)
            H_sub = build_subgraph(mapping, H, 1)

    return G_sub, H_sub, solution_size

        
def parse_mcsplit(stdout, G, H):
    try:
        solution_size = int(stdout[2].split()[-1])
    except ValueError:
        solution_size = 0
        G_sub = nx.Graph()
        H_sub = nx.Graph()
        return G_sub, H_sub, solution_size
        
    mapping = stdout[-1].split(',')[:-1]
    mapping = [elm.split('-') for elm in mapping]
    
    def build_subgraph(mapping, original_graph, index):
        nodes = [int(elm[index]) for elm in mapping]
        return original_graph.subgraph(nodes)
            
    G_sub = build_subgraph(mapping, G, 0)
    H_sub = build_subgraph(mapping, H, 1)
    return G_sub, H_sub, solution_size
        

def mcs_constraint_programming(G, H, atoms=None):
    identifier = f'{random.getrandbits(32)}'
    f1 = f'temp_G_{identifier}'
    f2 = f'temp_H_{identifier}'
    write_lad(G, f1, atoms)
    write_lad(H, f2, atoms)

    proc = subprocess.run([glasgow_common_subgraph_solver,
                           '--connected',
                           '--timeout', '30',
                           '--format', 'vertexlabelledlad',
                           f'temp_G_{identifier}', f'temp_H_{identifier}'], capture_output=True)
    stdout = proc.stdout.decode().split('\n')

    os.remove(f1)
    os.remove(f2)
    G_sub, H_sub, solution_size = parse_glasgow(stdout, G, H)

    return G_sub, H_sub, solution_size


def mcs_max_clique(G, H, atoms):
    identifier = f'{random.getrandbits(32)}'
    f1 = f'temp_G_{identifier}'
    f2 = f'temp_H_{identifier}'
    write_lad(G, f1, atoms)
    write_lad(H, f2, atoms)
    
    proc = subprocess.run([glasgow_common_subgraph_solver,
                           '--connected',
                           '--clique',
                           '--timeout', '30',
                           '--format', 'vertexlabelledlad',
                           f'temp_G_{identifier}', f'temp_H_{identifier}'], capture_output=True)
    stdout = proc.stdout.decode().split('\n')
    
    os.remove(f1)
    os.remove(f2)
    G_sub, H_sub, solution_size = parse_glasgow(stdout, G, H)
    return G_sub, H_sub, solution_size


def mcs_mcsplit(G, H, atoms):
    identifier = f'{random.getrandbits(32)}'
    f1 = f'temp_G_{identifier}'
    f2 = f'temp_H_{identifier}'
    write_gfd(G, f1, atoms)
    write_gfd(H, f2, atoms)
    proc = subprocess.run([mcsplit_solver, 
                           '--timeout', '30', # seconds
                           '--vertex-labelled-only', # undirected, vertex labelled graph
                           '-g', # gfd input format
                           'A', # heuristic
                           f1, f2], capture_output=True)
    stdout = proc.stdout.decode().split('\n')
    os.remove(f1)
    os.remove(f2)
    return parse_mcsplit(stdout, G, H)


def calculate_mcs(G, H, atoms = None):
    try:
        G_sub, H_sub, solution_size = mcs_constraint_programming(G, H, atoms)
    except AssertionError:
        # failed constrained programming
        try: 
            G_sub, H_sub, solution_size = mcs_max_clique(G, H, atoms)
        except AssertionError:
            # failed max clique
            # fall back to 
            G_sub, H_sub, solution_size = mcs_mcsplit(G, H, atoms)
            
    return G_sub, H_sub, solution_size
    

def match_subgraph(pattern, target, atoms=None, identifier=None, rematch=True):
    if rematch:
        if not identifier:
            identifier = f'{random.getrandbits(32)}'
        write_lad(pattern, f'temp_pattern_{identifier}', atoms)
        write_lad(target, f'temp_target_{identifier}', atoms)

        proc = subprocess.run([glasgow_subgraph_solver,
                               '--induced',
                               '--format', 'vertexlabelledlad',
                               f'temp_pattern_{identifier}', f'temp_target_{identifier}'], capture_output=True)
        stdout = proc.stdout.decode().split('\n')

        try:
            matched_pattern, target_subgraph, solution_size = parse_glasgow(stdout, pattern, target)
        except AssertionError:
            # because pattern is obtained as a maximum common subgraph to target and some other graph, pattern is guaranteed to be in target
            print('common subgraph not found to be subgraph isomorphic to target, aborting...', stdout)
            exit()

        os.remove(f'temp_pattern_{identifier}')
        os.remove(f'temp_target_{identifier}')

        # obtain the part of target not matched to pattern
        # note: nx.difference does not do this. It just differences the edges

        target_antisubgraph = target.subgraph([n for n in target.nodes if n not in target_subgraph.nodes])
        return target_antisubgraph
    else:
        target_antisubgraph = target.subgraph([n for n in target.nodes if n not in pattern.nodes])
        return target_antisubgraph

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

def distance_wholestructure(ia, ja, atoms, graphs, subgraphs):
    # from mcs we can extract the non-matched atoms
    # those atoms should serve to similarity lower / distance higher
    # and their influence should depend on their distance to the center atom
    G = graphs[ia]
    H = graphs[ja]
    
    key = '-'.join(sorted([str(atoms[ia].tag), str(atoms[ja].tag)]))
    # key is always sorted
    # mcs has 2 isomorphic subgraphs, each corresponding to ia, ja
    # should not really matter
    mcs = subgraphs[key]
    if ia <= ja:
        G_sub = mcs[0]
        H_sub = mcs[1]
    else:
        G_sub = mcs[1]
        H_sub = mcs[0]

    unmatched_G = match_subgraph(G_sub, G, atoms)
    unmatched_H = match_subgraph(H_sub, H, atoms)
    # get_sub = lambda graph, mcs, idx : graph.subgraph([sorted(graph.nodes())[l] for l in [int(edge.split('-')[idx]) for edge in mcs]])
    # G_sub = get_sub(G, mcs, 0)
    # H_sub = get_sub(H, mcs, 1)

    G_nosub = unmatched_G.nodes
    H_nosub = unmatched_H.nodes
    topo_rad = 4.8
    dist_G = np.array([atoms.get_distance(n, ia, mic=True) for n in G_nosub])/topo_rad
    dist_H = np.array([atoms.get_distance(n, ja, mic=True) for n in H_nosub])/topo_rad

    dist = np.concatenate((dist_G, dist_H))
    p = 2
    norm = (np.sum(dist**p))**(1/p)/len(dist)
    d = 1 - norm
    if d < 0 or np.isnan(d):
        breakpoint()
    return d
    

def par_worker(ia, ja, graphs, atoms):
    if atoms[ia].tag == atoms[ja].tag:
        d = 0
    else:
        d = distance(graphs[ia], graphs[ja], atoms, ia, ja)
    print(ia, ja, d)
    return d


def atoms2nx(atoms, cutoff_in=None):
    G = nx.Graph()
    dm = atoms.get_all_distances(mic=True)
    natoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    nodes = [(p, dict(symbol=q)) for p, q in zip(range(natoms), symbols)]
    G.add_nodes_from(nodes)
    cutoffs = {'Cu': {'O': 2.3, 'Cu': 2.8},
               'O': {'Cu': 2.3, 'O': 2.0}}
    pos = atoms.get_positions()
    pos = {a.index: a.position for a in atoms}
    nx.set_node_attributes(G, pos, 'pos')
    for ii in range(natoms):
        for jj in range(ii+1, natoms):
            cutoff = cutoffs[atoms[ii].symbol][atoms[jj].symbol]
            if dm[ii][jj] < cutoff:
                G.add_edge(ii, jj)
    return G


def find_atom_id_for_event(eventid, filename='sortieproc.0'):
    with open(filename, 'r') as sortiefile:
        for line in sortiefile:
            if re.search(f'eventid : + {eventid}$', line):
                for i in range(11):
                    nextline = next(sortiefile, '')
                    if 'atm_lbl' in nextline:
                        break

                nextline = next(sortiefile, '')
                atom_id = int(nextline.split()[0])
                return atom_id

            
def find_index_in_cluster(filename, index):
    match_str = f'{index}'.rjust(14, ' ')
    with open(filename, 'r') as eventfile:
        for idx, line in enumerate(eventfile):
            if match_str in line:
                return idx - 18

            
def find_topo_ids(filename):
    with open(filename, 'r') as eventfile:
        lines = eventfile.readlines()
    return int(lines[1].split()[-3]), int(lines[1].split()[-2])


def atoms_changed_topo(filename):
    atoms_changed = []
    with open(filename, 'r') as sortiefile:
        for line in sortiefile:
            if 'Topology has changed' in line and ' 0-->' not in line:
                nextline = next(sortiefile, '')
                atom_idx = int(nextline.split()[-1])-1
                atoms_changed.append(atom_idx)
    return atoms_changed
                
if __name__ == '__main__':    
    G = nx.Graph()
    H = nx.Graph()
    G.add_node("a1", a1=True)
    G.add_node("a2", a1=True)
    G.add_node("a3", a1=True)
    G.add_edge("a1", "a2")
    G.add_edge("a2", "a3")

    H.add_node("b1", a2="Spam")
    H.add_node("b2", a2="Spam")
    H.add_node("b3", a2="Spam")
    H.add_edge("b1", "b2")
    H.add_edge("b2", "b3")

    common_edges(G, H)
    # nx.draw(get_mcs(G, H), with_labels=True)
    # plt.show()

