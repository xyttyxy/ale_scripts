# autoads job manager helper functions
# xyttyxy@ucla.edu
# 2020/02/19

from ase.io import write
import os, subprocess, shutil
from ase.calculators.vasp import Vasp2
from ase.calculators import calculator
from myase.utils import autoads, pymail
submit_command = '/u/home/x/xyttyxy/scripts/job_scripts/ase/universal/submit.sh -p vasp_std -pe shared 8 -l h_data=4G,h_rt=24:00:00,exclusive -r filename=POSCAR'
check_sge = 'qstat -u xyttyxy'
calc_log = '/u/home/x/xyttyxy/scripts/job_scripts/ase/universal/calc.log'

# qstat parser, to check if it is still running or waiting
def job_status(jobid):
    jobs = subprocess.run(check_sge, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
    jobs = jobs.split('\n')[2:-1]
    for j in jobs:
        if jobid == int((j.split()[0])):
            return (j.split()[4])
    return 'nf'

# read job ID by checking log files in working folder and centralized $CALC_LOG
def get_jobid(path = '.'):
    # first check the queue
    qstat = subprocess.run('grep -w ' + os.getcwd() + ' ' + calc_log, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')[0:-1]
    qstat = [j.split('\t')[0] for j in qstat]
    qstat = [int(j) for j in qstat]
    if qstat:
        jobid = max(qstat)
    else:
        files = next(os.walk(path))[2]
        log = [int(f[0:8]) for f in files if '.out' in f and f != 'vasp.out']
        if log:
            jobid = max(log)
        else:
            alert_str = os.getcwd() + ': no calculations submitted'
            print(alert_str)
            pymail.send_mail(subject = alert_str)
    return jobid

def build_acc(findir, acc_calc, prefix):
    jobid = get_jobid('./acc')
    if job_status(jobid) in ['qw', 'r'] or os.path.exists('OUTCAR'):
        return
    atoms = autoads.accurate_atoms(findir, prefix)
    write('./acc/POSCAR', atoms)
    subprocess.run('ln '+acc_calc+' ./acc/calc.py', shell=True)
    os.chdir('./acc')
    subprocess.run(submit_command, shell=True)

def iter_check(path, prefix, acc_calc, geo_check=False, mail=True):
    restore_path = os.getcwd()

    os.chdir(path)
    dirs = next(os.walk('.'))[1]
    for p in ['fin', 'un', 'nf', 'acc']:
        try:
            assert(p in dirs)
        except AssertionError:
            if not os.path.exists(p):
                os.mkdir(p)

    dirs = [d for d in dirs if prefix in d]

    if (len(dirs) == 0):
        alert_str = 'Iteration finished at' + path
        if mail:
            pymail.send_mail(subject = alert_str)
        print(alert_str)
        build_acc(os.getcwd() + '/fin/', acc_calc, prefix)
        os.chdir(restore_path)
        return True

    pwd = os.getcwd()
    # calculation not finished
    for d in dirs:
        assert(prefix in d)
        os.chdir(d)
        jobid = get_jobid()
        status = job_status(jobid)
        if status == 'qw':
            print(path + '/'+d+': still waiting')
            os.chdir('..')
            continue
        else:
            try:
                calc = Vasp2(restart=True)
            except calculator.ReadError:
                alert_str = os.getcwd()+': '+status+'/'+str(jobid)+' calculation not found'
                print(alert_str)
                if mail:
                    pymail.send_mail(subject = alert_str)
                os.chdir('..')
                continue
            except IndexError:
                if status == 'r':
                    os.chdir('..')
                    continue
                alert_str = path + '/' +d+ ': started calculation failed before 1st ionic step'
                print(alert_str)
                if mail:
                    pymail.send_mail(subject = alert_str)
                os.chdir('..')
                continue
            except FileNotFoundError:
                alert_str = path + '/' +d+ ': possible file operation error'
                print(alert_str)
                os.chdir('..')
                continue
            atoms = calc.get_atoms()
            if geo_check:
                if autoads.recombined(atoms, 1.4):
                    if status == 'r':
                        # kill first if still running
                        subprocess.run('qdel '+str(jobid), shell=True)
                    # move to un/
                    os.chdir('..')
                    os.rename(d, 'un/'+d)
                    continue
            if calc.converged and status == 'nf':
                # normal termination
                # move to fin/
                os.chdir('..')
                os.rename(d, 'fin/'+d)
                continue
            elif not calc.converged and status == 'nf':
                # abnormal termination
                alert_str = path +'/'+ d + ': convergence failed after 24 hrs'
                print(alert_str)
                if mail:
                    pymail.send_mail(subject = alert_str)
                os.chdir('..')
                if os.path.exists('nf/'+d):
                    shutil.rmtree('nf/'+d)
                shutil.move(d, 'nf/')
                continue
            # still running, nothing to do, go back up
            if status == 'r' and not calc.converged:
                os.chdir('..')
                continue
            # If execution reach here we have problem
            print('Unexpected')
        assert(pwd == os.getcwd())

    os.chdir(restore_path)
    return False
