#!/bin/bash
mpic++ --std=c++11 -O2 main.cpp -o main.cur.o && /usr/bin/time -a -f "%E real,%U user,%S sys" mpirun -np 4 ./main.cur.o -ls 100