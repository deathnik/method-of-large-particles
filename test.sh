#!/bin/bash
echo > log
mpic++ --std=c++11 -O2 main.cpp -o main.o
for n in 100 200 400 800 1600 3200
do
for i in 1 2 4 8 16 32 64
do
   echo "time mpirun -np $i ./main -ls $n" >> log
   /usr/bin/time -a -o log -f "%E real,%U user,%S sys" mpirun -np $i ./main.o -ls $n
done
done
