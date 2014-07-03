#!/bin/bash
mpic++ --std=c++11 -O2 main.cpp -o main.o
qsub -l nodes=1:ppn=1 cluster_run_1000.sh
qsub -l nodes=1:ppn=2 cluster_run_1000.sh
qsub -l nodes=1:ppn=4 cluster_run_1000.sh
qsub -l nodes=1:ppn=8 cluster_run_1000.sh