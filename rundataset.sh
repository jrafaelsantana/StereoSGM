#!/bin/bash

epoch=18
datasets=( ArtL Jadeplant Motorcycle MotorcycleE Piano PianoL Pipes Playroom Playtable PlaytableP Shelves Teddy Vintage )

for dataset in "${datasets[@]}"
do
    evalcommand="python3 main.py $epoch $dataset"
    eval $evalcommand
done
