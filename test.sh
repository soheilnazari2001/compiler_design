#!/bin/bash
for test in testcases/T*; do
    echo "RUNNING TEST $test"
    cp "$test/input.txt" input.txt
    python3 main.py
    diff "$test/lexical_errors.txt" lexical_errors.txt
    diff "$test/tokens.txt" tokens.txt
    echo "=================="
done
rm input.txt lexical_errors.txt tokens.txt symbol_table.txt