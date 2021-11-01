import argparse
import pandas as pd
import scpp
import util


parser = argparse.ArgumentParser()

parser.add_argument('--input_fn', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='out/tmp/')
parser.add_argument('--replace_outdir', action='store_true')
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--a', type=float, default=1)
parser.add_argument('--b', type=float, default=1)

# Learning setting
parser.add_argument('--tol', type=float, default=100)
parser.add_argument('--max_iter', type=int, default=20)
parser.add_argument('--verbose', type=bool, default=True)

# Options
parser.add_argument('--time_col', type=str, default='date')
parser.add_argument('--item_col', type=str, default='item')
parser.add_argument('--user_col', type=str, default='user')
parser.add_argument('--n_sample', type=int, default=100000)
parser.add_argument('--sampling_rate', type=str, default='D')

# DEMO
parser.add_argument('--retail', action='store_true')

config = parser.parse_args()


# Data preparaion
if config.retail:
    data = pd.read_csv('data/retail_transaction.csv')
    data = util.sample_events(data, 1000)
    data = util.encode_timestamp(data, 'date', 'D')

else:
    if config.input_fn is None:
        raise ValueError("Specify your input filename")

    data = pd.read_csv(config.input_fn)

    if config.n_sample > 0:
        data = util.sample_events(data, config.n_sample)

    data = util.encode_timestamp(data,
        datetime_col=config.time_col,
        freq=config.sampling_rate)

    data = util.encode_attribute(data, col=config.item_col)
    data = util.encode_attribute(data, col=config.user_col)

print()
print(" INPUT ")
print("=======")
print(data.head())
print()

util.prepare_workspace(config.output_dir, replace=config.replace_outdir)

model = scpp.SCPP()
model.fit(data,
          gamma=config.gamma,
          beta=config.beta,
          a=config.a,
          b=config.b,
          max_iter=config.max_iter,
          tol=config.tol,
          verbose=config.verbose)

model.save(config.output_dir)
