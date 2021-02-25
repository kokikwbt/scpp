#!/bin/sh

cd `dirname $0`

INPUT="dat/scpp_trans_1.csv"
OUTDIR="out/tmp/"

GAMMA=1
BETA=1
A=1
B=1
DATETIME_COL="date"
N_SAMPLE=1000

python3 main.py --input_filename $INPUT \
                --output_dir $OUTDIR \
                --gamma $GAMMA \
                --beta $BETA \
                --a $A \
                --b $B \
                --encode_timestamp $DATETIME_COL \
                --sample_events $N_SAMPLE \
                --replace_outdir \
                --save_params_only \
                --save_train_hist