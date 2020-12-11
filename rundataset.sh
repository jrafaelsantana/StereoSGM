#!/bin/bash

datasets=( ArtL Jadeplant Motorcycle MotorcycleE Piano PianoL Pipes Playroom Playtable PlaytableP Recycle Shelves Teddy Vintage )
epochs=(15)
p1s=( 9.0 )
p2s=( 18.0 )

for dataset in "${datasets[@]}"
do
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
done
