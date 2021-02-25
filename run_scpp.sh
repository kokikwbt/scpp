#!/bin/sh

cd `dirname $0`

INPUT="dat/scpp_trans_1.csv"
OUTDIR="out/tmp/"

GAMMA=1
BETA=1
A=1
B=1

python3 main.py $INPUT
                --output_dir $OUTDIR \
                --gamma $GAMMA \
                --beta $BETA \
                --a $A \
                --b $B \
                --replace_outdir \
                --save_params_only \
                --save_train_hist \