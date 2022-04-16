from pymatgen.io.lammps.outputs import parse_lammps_dumps, LammpsDump
import mmap
from collections import deque
from ase.calculators.singlepoint import SinglePointCalculator
from ase.quaternions import Quaternions
from ase.db import connect
from ase import Atoms, units
import numpy as np 
from ase.parallel import paropen
from ase.utils import basestring


def dump2atoms(a, impact_type, z_filt = False):
    """ converts pymatgen.io.lammps.outputs.LammpsDump to ase.Atoms """
    data = a.data
    lat = a.box.to_lattice()
    columns = data.columns.to_list()
    if 'xs' not in columns and 'x' in columns:
        coords = np.array(data[['x','y','z']])
    else:
        coords = np.array(data[['xs','ys','zs']]) * np.array([lat.a,lat.b,lat.c])
    types = data['type']
    symbols = np.array(['Cu' if t == 1 else impact_type for t in types])

    if z_filt:
        min_O = min(coords[:, 2][symbols == 'O'])
        z_filt = coords[:, 2] > (min_O - 20)

        at = Atoms(symbols = symbols[z_filt], positions = coords[z_filt, :], cell = [lat.a, lat.b, lat.c], pbc=True)
        if 'vx' in columns:
            vel = data[['vx','vy','vz']].to_numpy(float)
            at.set_velocities(vel[z_filt, :])
    else:
        at = Atoms(symbols = symbols, positions = coords, cell = [lat.a, lat.b, lat.c], pbc=True)
        if 'vx' in columns:
            vel = data[['vx','vy','vz']].to_numpy(float)
            at.set_velocities(vel)

    return at


def dump2db_func(i, impact_type='O', z_filt=False):
    """converts big lammpsdump file to ase.db
    Note: lammps file is usually only accessible sequentially.
    Converting it to sqlite database allows random access.
    However, the same functionality is achieved using memory-mapped
    files now, so this time-consuming conversion is no longer necessary. """
    dumpfile = 'dump.{:}.high_freq'.format(i)
    dbfile = 'run_{:}.db'.format(i)
    timesteps = np.loadtxt('dump.{:}.timesteps'.format(i))
    dump = parse_lammps_dumps(dumpfile)
    i = 0
    with connect(dbfile, append=False) as db:
        for a in dump:
            id = db.reserve(timestep=timesteps[i])
            if id is None:
                continue
            at = dump2atoms(a, impact_type, z_filt)
            db.write(at, timestep=timesteps[i], id=id)
            i += 1


def get_dump_image_indices(filename):
    """ Finds the starting byte indices of each image in a lammpsdump file """
    i = int(filename.split('.')[1])
    byte_indices = np.empty(2, dtype=np.uint64)
    with open('dump.{:}.high_freq'.format(i), 'r+') as dumpfile:
        m = mmap.mmap(dumpfile.fileno(), 0)
        last_byte_idx = 0
        step = 0
        while True:
            byte_idx = m.find(b'ITEM: TIMESTEP', last_byte_idx)
            if byte_idx == -1:
                break
            # point file pointer to matched byte index
            m.seek(byte_idx)

            # timestep is line after match
            m.readline()
            timestep = int(m.readline().rstrip())
            step += 1

            # append byte index
            byte_indices = np.vstack((byte_indices, np.array([byte_idx, timestep], dtype=np.uint64)))
            last_byte_idx = byte_idx+1 # to skip last found
            
    return byte_indices[1:, :]


def get_dump_timesteps(filename, dump_freq = 100):
    """ Get time steps in a dump file. Reads 1st, last, interpolate between """
    i = int(filename.split('.')[1])
    with open('dump.{:}.high_freq'.format(i), 'r+') as dumpfile:
        m = mmap.mmap(dumpfile.fileno(), 0)
        byte_idx = m.find(b'ITEM: TIMESTEP')
        m.seek(byte_idx)
        m.readline()
        init_step = int(m.readline().rstrip())
        
        byte_idx = m.rfind(b'ITEM: TIMESTEP')
        m.seek(byte_idx)
        m.readline()
        final_step = int(m.readline().rstrip())
    timesteps = np.arange(init_step, final_step, dump_freq)
    np.savetxt('dump.{:}.timesteps'.format(i), timesteps, fmt='%d')


def read_dump(filename, find_step):
    """ random-access to structures in huge lammpsdump using mmap """
    i = int(filename.split('.')[1])
    with open(filename.format(i), 'r+') as dumpfile:
        # memory-mapped file for very fast I/O
        m = mmap.mmap(dumpfile.fileno(), 0)

        byte_indices = np.loadtxt('dump.{:}.byteidx'.format(i), dtype=np.uint64)
        assert (find_step <= byte_indices[-1,1]).any() and (find_step >= byte_indices[0,1]).any(), 'problem: step specified not in range of dump'

        byteidx_linum = int(np.where(byte_indices[:,1] == find_step)[0].item())

        index4step = byte_indices[byteidx_linum, 0]

        if byte_indices.shape[0] > byteidx_linum+1:
            # index must be size-1
            index4step_next = byte_indices[byteidx_linum+1, 0]
        else:
            # reading last image in dump
            index4step_next = np.uint64(m.size())

        diffbyte = index4step_next - index4step
        byte_idx = m.find(b'ITEM: TIMESTEP', index4step)

        if byte_idx < 0:
            # problem: byte sequence not found
            assert byte_idx == index4step, 'problem: index specified in key file not found'
        else:
            m.seek(byte_idx)

        bytes = m.read(diffbyte).decode('utf-8')
        df = LammpsDump.from_string(bytes)
        at = dump2atoms(df, 'O')
        return at


def read_lammps_dump(fileobj, index=-1, order=True, atomsobj=Atoms, atom_types=None, return_all=False):
    """Method which reads a LAMMPS dump file.

    ase.io.read implementation for lammps dump file
    This version is modified by Yantao Xia  (19 Sept 24)
    Further modifided 20 Aug 25
    order: Order the particles according to their id. Might be faster to
    switch it off.

    atom_types: list of type definitions to convert lammps types to real atoms
    """
    if isinstance(fileobj, basestring):
        f = paropen(fileobj)
    else:
        f = fileobj

    # load everything into memory
    lines = deque(f.readlines())

    natoms = 0
    images = []

    while len(lines) > natoms:
        line = lines.popleft()

        if 'ITEM: TIMESTEP' in line:
            lo = []
            hi = []
            tilt = []
            id = []
            types = []
            positions = []
            scaled_positions = []
            velocities = []
            forces = []
            quaternions = []
            charges = []

        if 'ITEM: NUMBER OF ATOMS' in line:
            line = lines.popleft()
            natoms = int(line.split()[0])
            #print(natoms)
            
        if 'ITEM: BOX BOUNDS' in line:
            # save labels behind "ITEM: BOX BOUNDS" in
            # triclinic case (>=lammps-7Jul09)
            tilt_items = line.split()[3:]
            for i in range(3):
                line = lines.popleft()
                fields = line.split()
                lo.append(float(fields[0]))
                hi.append(float(fields[1]))
                if (len(fields) >= 3):
                    tilt.append(float(fields[2]))

            # determine cell tilt (triclinic case!)
            if (len(tilt) >= 3):
                # for >=lammps-7Jul09 use labels behind
                # "ITEM: BOX BOUNDS" to assign tilt (vector) elements ...
                if (len(tilt_items) >= 3):
                    xy = tilt[tilt_items.index('xy')]
                    xz = tilt[tilt_items.index('xz')]
                    yz = tilt[tilt_items.index('yz')]
                # ... otherwise assume default order in 3rd column
                # (if the latter was present)
                else:
                    xy = tilt[0]
                    xz = tilt[1]
                    yz = tilt[2]
            else:
                xy = xz = yz = 0
            xhilo = (hi[0] - lo[0]) - (xy**2)**0.5 - (xz**2)**0.5
            yhilo = (hi[1] - lo[1]) - (yz**2)**0.5
            zhilo = (hi[2] - lo[2])
            if xy < 0:
                if xz < 0:
                    celldispx = lo[0] - xy - xz
                else:
                    celldispx = lo[0] - xy
            else:
                celldispx = lo[0]
            celldispy = lo[1]
            celldispz = lo[2]

            cell = [[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]]
            celldisp = [[celldispx, celldispy, celldispz]]

        def add_quantity(fields, var, labels):
            for label in labels:
                if label not in atom_attributes:
                    return
            var.append([float(fields[atom_attributes[label]])
                        for label in labels])
                
        if 'ITEM: ATOMS' in line:
            # (reliably) identify values by labels behind
            # "ITEM: ATOMS" - requires >=lammps-7Jul09
            # create corresponding index dictionary before
            # iterating over atoms to (hopefully) speed up lookups...
            atom_attributes = {}
            for (i, x) in enumerate(line.split()[2:]):
                atom_attributes[x] = i

            # updating 
            for n in range(natoms):
                line = lines.popleft()
                fields = line.split()

                id.append(int(fields[atom_attributes['id']]))
                #types.append(int(fields[atom_attributes['type']]))
                type = int(fields[atom_attributes['type']])
                types.append(atom_types[type-1])
                
                add_quantity(fields, positions, ['x', 'y', 'z'])
                add_quantity(fields, charges, ['q'])
                add_quantity(fields, scaled_positions, ['xs', 'ys', 'zs'])
                add_quantity(fields, velocities, ['vx', 'vy', 'vz'])
                add_quantity(fields, forces, ['fx', 'fy', 'fz'])
                add_quantity(fields, quaternions, ['c_q[1]', 'c_q[2]',
                                                   'c_q[3]', 'c_q[4]'])

            if order:
                def reorder(inlist):
                    if not len(inlist):
                        return inlist
                    outlist = [None] * len(id)
                    for i, v in zip(id, inlist):
                        outlist[i - 1] = v
                    return outlist
                types = reorder(types)
                positions = reorder(positions)
                scaled_positions = reorder(scaled_positions)
                velocities = reorder(velocities)
                forces = reorder(forces)
                quaternions = reorder(quaternions)

            if len(quaternions):
                images.append(Quaternions(symbols=types,
                                          positions=positions,
                                          cell=cell, celldisp=celldisp,
                                          quaternions=quaternions))
            elif len(positions):
                images.append(atomsobj(
                    symbols=types, positions=positions,
                    celldisp=celldisp, cell=cell))
            elif len(scaled_positions):
                images.append(atomsobj(
                    symbols=types, scaled_positions=scaled_positions,
                    celldisp=celldisp, cell=cell))

            if len(velocities):
                images[-1].set_velocities(velocities)
            if len(forces):
                # y.x.: unit conversion for 'units real'
                temp = np.array(forces) * 4.3363e-2
                calculator = SinglePointCalculator(images[-1],
                                                   energy=0.0, forces=temp.tolist())
                images[-1].set_calculator(calculator)
            
    if return_all:
        return images
    else:
        return images[index]


def read_lammps_dump_pymatgen(fileobj, every_n = 1, index=-1, order=True, atomsobj=Atoms, impact_type='O', return_all=False):
    """This is a wrapper around pymatgen generator, to get the list of atoms
    Note: on a very large dump file this will eat the RAM very quickly. Use with caution. """
    LD = parse_lammps_dumps(fileobj)
    # breakpoint()
    atoms_view = []
    i = 0
    for a in LD:
        i = i + 1
        if i % every_n == 0:
            data = a.data
            lat = a.box.to_lattice()
            columns = data.columns.to_list()
            if 'xs' not in columns and 'x' in columns:
                coords = np.array(data[['x','y','z']])
            else:
                coords = np.array(data[['xs','ys','zs']]) * np.array([lat.a,lat.b,lat.c])
            types = data['type']
            symbols = []
            for t in types:
                if t == 1:
                    symbols.append('Cu')
                elif t == 2:
                    symbols.append(impact_type)
                elif t == 3:
                    symbols.append('C')
                elif t == 4:
                    symbols.append('H')

            at = Atoms(symbols = symbols, positions = coords, cell = [lat.a, lat.b, lat.c], pbc=True)
            if 'vx' in columns:
                vel = data[['vx','vy','vz']].to_numpy(float) / units.fs
                # breakpoint()
                at.set_velocities(vel)
            atoms_view.append(at)

    return atoms_view
