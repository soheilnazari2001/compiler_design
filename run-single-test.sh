#!/usr/bin/env bash

test="$1"

cp "$test/input.txt" input.txt
python3 compiler.py
interpreter/tester_linux.out > actual.txt 2> /dev/null
sed -i '/^Total/d' actual.txt
diff actual.txt "$test/expected.txt"