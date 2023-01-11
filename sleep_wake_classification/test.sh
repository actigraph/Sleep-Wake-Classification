#!/bin/bash

# PARAM: test or paper, depending which results you want to compare against
for f in $1_results/*csv; do diff $f outputs/PSGonly_10minpadded_Newcastle_2015/$(basename $f); done