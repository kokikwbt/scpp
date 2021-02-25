import argparse
import pandas as pd
import scpp
import util


parser = argparse.ArgumentParser()

parser.add_argument('--input_filename', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='out/tmp/')
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--a', type=float, default=1)
parser.add_argument('--b', type=float, default=1)

# Learning setting

parser.add_argument('--freq', type=str, default='D')
parser.add_argument('--tol', type=float, default=1e+3)
parser.add_argument('--max_iter', type=int, default=10)
parser.add_argument('--sample_events', type=int, default=0)

# Options

parser.add_argument('--encode_timestamp', type=str, default=None)
parser.add_argument('--encode_item', action='store_true')
parser.add_argument('--encode_user', action='store_true')
parser.add_argument('--replace_outdir', action='store_true')
parser.add_argument('--save_train_hist', action='store_true')
parser.add_argument('--save_params_only', action='store_true')

args = parser.parse_args()

if args.input_filename is None:
    raise ValueError("Specify your input filename")

util.prepare_workspace(args.output_dir,
                       replace=args.replace_outdir)

# Data preparaion

data = pd.read_csv(args.input_filename)

if args.sample_events > 0:
    data = util.sample_events(data, args.sample_events)

if args.encode_timestamp is not None:
    data = util.encode_timestamp(
        data,
        datetime_col=args.encode_timestamp,
        freq=args.freq)

if args.encode_item == True:
    data = util.encode_attribute(data, 'item')
if args.encode_user == True:
    data = util.encode_attribute(data, 'user')

print()
print("INPUT")
print("=====")
print(data.head())

# Fit SCPP model

model = scpp.SCPP()

model.fit(data,
          max_iter=args.max_iter,
          tol=args.tol)

model.save(args.output_dir,
           args.save_params_only,
           args.save_train_hist)
