#!/bin/bash
mpic++ --std=c++11 main.cpp -o main.o && mpirun -np 1 ./main.o
