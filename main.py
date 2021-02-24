import argparse
import os
import shutil
import pandas as pd
import scpp


parser = argparse.ArgumentParser()

parser.add_argument('input_filename', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--a', type=float, default=1)
parser.add_argument('--b', type=float, default=1)

# Options

parser.add_argument('--replace_outdir', action='store_true')

args = parser.parse_args()


if args.replace_outdir == True:
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
else:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


data = pd.read_csv(args.input_filename)
data = data.sample(2000).sort_values('date_id').reset_index()


model = scpp.SCPP()
model.fit(data)
model.save(args.output_dir)