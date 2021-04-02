#!/bin/bash

SCRIPT="$1"
N_TASKS=100
MEM=100g

spark-submit $SCRIPT --num-executors $N_TASKS --executor-memory $MEMs