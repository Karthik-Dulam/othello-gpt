#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <n> <data_dir> <num_jobs>"
    exit 1
fi

# Assign arguments to variables
n=$1
data_dir=$2
num_jobs=$3

# Calculate the number of iterations per job
iterations_per_job=$((n / num_jobs))

# Run the processes in parallel with different seeds
for i in $(seq 1 $num_jobs); do
    python othello.py -n $iterations_per_job --dir $data_dir --seed $i &
done

# Wait for all background jobs to complete
wait

echo "All jobs completed."