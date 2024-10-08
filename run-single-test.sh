#!/usr/bin/env bash

NO_OUTPUT_GENERATED="The code has not been generated."

test="$1"

cp "$test/input.txt" input.txt
python3 compiler.py >> /dev/null 2>&1
if grep -q "$NO_OUTPUT_GENERATED" output.txt;
then
    if ! diff output.txt "$test/output.txt" >> /dev/null 2>&1;
    then
        echo "$test failed. semantic errors detected. 'output.txt' is different."
    fi
    if ! diff semantic_errors.txt "$test/semantic_errors.txt" >> /dev/null 2>&1;
    then
        echo "$test failed. semantic errors detected. 'semantic_errors.txt' is different."
    fi
else
    interpreter/tester_linux.out output.txt > actual.txt 2> /dev/null
    sed -i '/^Total/d' actual.txt
    if ! diff actual.txt "$test/expected.txt" >> /dev/null 2>&1;
    then
        echo "$test failed. code generated. 'actual.txt' is different than 'expected.txt'."
    fi
    if ! diff semantic_errors.txt "$test/semantic_errors.txt" >> /dev/null 2>&1;
    then
        echo "$test failed. code generated. 'semantic_errors.txt' is different."
    fi
fi