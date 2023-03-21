#!/bin/bash

#SBATCH --output="%x.%j.out"
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --partition=RM
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=48:00:00

source ~/.bashrc
module purge
module load lammps/intel20.4_openmpi_29Sep2021

conda activate ale

# number of parallel cores to use per search folder
# this should be approximately the number of searches per line in search_list.txt

export spar=12
echo "K-ART Workflow script started"
export ACT_NBRE_KMC_STEPS=5

# cont.conf is used as the working image for K-ART.
# before.conf is never overwritten, cont.conf will be. 
cp before.conf cont.conf

for step in $(seq 1 $ACT_NBRE_KMC_STEPS); do
    # writes search list
    mpirun -np 1 ./topoid.sh
    echo "topology search finished, written search_list"
    # splits the search list
    # the argument to split.py controls how many topologies to assign to each job.
    # the total number of lines of search_list.txt divided by this number multiplied by spar should approx. the total number of cores available
    # here we have 43 lines, 10 folders, each folder uses 12 cores, 8*12=120 cores and total=128, ok.

    # however, size of search_list will change so it is better to adapt this as we go.
    # if search_list is short, we'd better split it into many folders, each with few cores. 
    IFS=',' read -r -a search_folders <<< $(python3 split.py 4)

    # # launch one event search in each search folder
    for index in "${!search_folders[@]}"; do
	cd "${search_folders[index]}";
	# clean up
	rm -r EVENTS_DIR SPEC_EVENTS_DIR sortieproc.*
	# only necessary files
	cp ../common.sh ../events.sh ../in.lammps ../minimize.lammps ../this_conf ../4layer.data ../cont.conf ../Topologies ./
	mpirun --bind-to none -np ${spar} ./events.sh &> kart.out &
	pids[${index}]=$!
	cd ..;
    done

    for pid in ${pids[*]}; do
	wait $pid
	echo $pid "finished"
    done
    echo "event search finished"

    mkdir -p EVENTS_DIR
    mkdir -p SPEC_EVENTS_DIR

    # gather topos.list
    python3 gather_topoid.py
    cp topos.list topos.list.bak.$step
    mv topos.list.gathered topos.list

    # gather generic event and delete search folder
    for index in "${!search_folders[@]}"; do
	rsync -ar "${search_folders[index]}"/EVENTS_DIR/ ./EVENTS_DIR/
	cat "${search_folders[index]}"/events.uft >> ./events.uft
	rm -r "${search_folders[index]}"
    done
    
    # launch KMC using the collected results
    echo "combined events, run a single KMC step"
    mpirun -np ${SLURM_NTASKS} ./single.sh

    # save KMC step
    mkdir -p KART_${step}
    cp cont.conf art_detail.dat Diffusion.dat Energies.dat events.uft Gen_ev.dat KMC-log.dat MinBarr.dat MPI_efficiency.dat restart.dat selec_ev.dat topos.list topo_stat.dat KART_${step}/

    # take last image in allconf (xyz-format) to cont.conf (kart-format)
    python3 allconf2contconf.py $step
done
