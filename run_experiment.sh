#!/bin/bash

IFS=","
while read -ra LINE; do
python3 $1 -p $2 - -s $2
done
