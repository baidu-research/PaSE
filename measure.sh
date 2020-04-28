#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <graph> <n_procs> <algo>"
  exit -1
fi

source ~/python_venv/dag-scheduler/bin/activate
batch_size=64

sbatch -p1080Ti --exclusive -N1 -o "time_$1_n$2_algo$3.txt" \
  --job-name=time_measurement \
  --wrap="srun python /mnt/home/venmugil/codes/dag-scheduler/scheduler.py -p$2 -b${batch_size} -g$1 --measure --algo=$3"
