from ase.calculators.vasp import Vasp

def get_base_calc():
    calc = Vasp(command=vasp_command,
                # functional and basis set
                encut=400,xc='pbe', 
                # smearing near E_fermi
                ismear=0,sigma=1e-2, 
                # dipole correction
                ldipol=True,idipol=3,dipol=com, 
                # DFT+U
                ldau=True,ldautype=2,ldaul=[2,0],ldauu=[7,0],ldauj=[0,0],lmaxmix=4,lasph=True, 
                # SCF convergence
                nelm=400,nelmdl=5,ediff=1e-6,algo='VeryFast', 
                # parallization & numerics
                ncore=4,kpar=8,lreal='Auto',prec='Normal', 
                # magnetization,magmom set in atoms object
                ispin=2, 
                # I/O
                istart=1,lwave=False,lcharg=False,
                # kpoints
                gamma=False,
                # logistics
                label=atoms.info['label'])
    
    return calc


def geo_opt(atoms, mode="vasp", opt_levels=None, fmax=0.02):
    calc = get_base_calc()
    if not opt_levels:
        # for bulks.
        # other systems: pass in argument
        opt_levels = {
            1: {"kpts": [3, 3, 3]},
            2: {"kpts": [5, 5, 5]},
            3: {"kpts": [7, 7, 7]},
        }

    levels = opt_levels.keys()
    if mode == 'vasp':
        write("CONTCAR", atoms)
        for level in levels:
            level_settings = opt_levels[level]
            # default settings when using built-in optimizer
            set_vasp_key(calc, 'ibrion', 2)
            set_vasp_key(calc, 'ediffg', -1e-2)
            set_vasp_key(calc, 'nsw', 200)
            set_vasp_key(calc, 'nelm', 200)
            # user-supplied overrides
            for key in level_settings.keys():
                set_vasp_key(calc, key, level_settings[key])

            atoms_tmp = read("CONTCAR")
            atoms_tmp.calc = calc
            atoms_tmp.get_potential_energy()
            calc.reset()
            atoms_tmp = read("OUTCAR", index=-1)
            shutil.copyfile("CONTCAR", f"opt{level}.vasp")
            shutil.copyfile("vasprun.xml", f"opt{level}.xml")
            shutil.copyfile("OUTCAR", f"opt{level}.OUTCAR")
    elif mode == 'ase':
        atoms_tmp = atoms.copy()
        from ase.optimize import BFGS
        # this atoms_tmp is updated when optimizer runs
        for level in levels:
            # default settings when using ase optimizer
            set_vasp_key(calc, 'ibrion', -1)
            set_vasp_key(calc, 'nsw', 0)
            # user-supplied overrides
            level_settings = opt_levels[level]
            for key in level_settings.keys():
                if key in ['nsw', 'ibrion', 'ediffg']:
                    continue
                set_vasp_key(calc, key, level_settings[key])

            atoms_tmp.calc = calc
            opt = BFGS(atoms_tmp,
                       trajectory = f"opt{level}.traj",
                       logfile = f"opt{level}.log")
            opt.run(fmax=fmax)
            calc.reset()
            shutil.copyfile("vasprun.xml", f"opt{level}.xml")
            shutil.copyfile("OUTCAR", f"opt{level}.OUTCAR")
            
    return atoms_tmp
