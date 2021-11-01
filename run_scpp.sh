#!/bin/sh
# Sample shallscript to run the SCPP algorithm.
cd `dirname $0`

# Filename
INPUT="data/retail_transaction.csv"

# Output directory
OUTDIR="out/tmp/"

# Set hyperparameters
GAMMA=1
BETA=1
A=1
B=1
TOL=100

# Set Dataset information
TIME_COL="date"
ITEM_COL="item_id"
USER_COL="user_id"
N_SAMPLE=1000
SAMPLING_RATE="D"

python3 main.py --input_fn $INPUT \
                --output_dir $OUTDIR \
                --gamma $GAMMA \
                --beta $BETA \
                --a $A \
                --b $B \
                --tol $TOL \
                --time_col $TIME_COL \
                --item_col $ITEM_COL \
                --user_col $USER_COL \
                --n_sample $N_SAMPLE \
                --sampling_rate $SAMPLING_RATE
