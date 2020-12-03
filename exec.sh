#!/bin/bash

for epoch in $(seq 20 30)
do
    evalcommand="python3 main.py $epoch"
    eval $evalcommand
done
