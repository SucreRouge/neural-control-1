#!/bin/sh
#BSUB -q mpi
#BSUB -R span[hosts=1]
#BSUB -n 4
#BSUB -W 16:00
source /usr/users/eschult/tf-venv/bin/activate
python copter-demo.py
