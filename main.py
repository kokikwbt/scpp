import argparse
import pandas as pd
import scpp
import util


parser = argparse.ArgumentParser()

parser.add_argument('input_filename', type=str)
parser.add_argument('--output_dir', type=str, default='out/tmp/')
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--a', type=float, default=1)
parser.add_argument('--b', type=float, default=1)

# Options

parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--tol', type=float, defalut=10)
parser.add_argument('--replace_outdir', action='store_true')
parser.add_argument('--save_params_only', action='store_true')
parser.add_argument('--save_train_hist', action='store_true')

args = parser.parse_args()

util.prepare_workspace(args.output_dir,
                       replace=args.replace_outdir)

data = pd.read_csv(args.input_filename)
data = data.sample(2000).sort_values('date_id').reset_index()

model = scpp.SCPP()

model.fit(data,
          max_iter=args.max_iter,
          tol=args.tol)

model.save(args.output_dir,
           args.save_params_only,
           args.save_train_hist)