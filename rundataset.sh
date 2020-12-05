#!/bin/bash

dataset=Adirondack
epochs=(13 14 15 16 17 18 19 20)
p1s=( 0.1 0.2 0.3 0.5 1.0 )
p2s=( 0.2 0.4 0.6 1.0 2.0 )

for i in "${!p1s[@]}"
do 
    for epoch in "${epochs[@]}"
    do
        evalcommand="python3 main.py $epoch $dataset ${p1s[$i]} ${p2s[$i]}"
        eval $evalcommand
        #printf "%s\t%s\n" "$i" "${foo[$i]}"
        #print "$evalcommand"
    done
done
