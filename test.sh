#!/bin/bash
echo > log
mpic++ main.cpp -o main.o
for n in 10 20 40 80 160 320
do
for i in 4 8 12 16 20 32 40 60
do
   echo "time mpirun -np $i ./main -ls $n" >> log
   /usr/bin/time -a -o log -f "%E real,%U user,%S sys" mpirun -np $i ./main.o -ls $n
done
done
