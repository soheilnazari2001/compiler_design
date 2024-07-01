#!/usr/bin/env bash

python3 genparser.py

types=${1:-T S R}

for type in $types;
do
    for test in tests/"$type"*/;
    do
        ./run-single-test.sh "$test"
    done
done
