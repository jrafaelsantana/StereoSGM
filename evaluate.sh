#!/bin/bash

treshold=1.0
datasets=( Adirondack ArtL Jadeplant Motorcycle MotorcycleE Piano PianoL Pipes Playroom Playtable PlaytableP Recycle Shelves Teddy Vintage )

for dataset in "${datasets[@]}"
do
    evalcommand="./runeval Q $dataset $treshold DUPLAJANELAJUNCAO"
    eval $evalcommand
done