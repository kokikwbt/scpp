# SCPP: Shared Cascade Poisson Process

**Unofficial** Python implemetation of:  
[Discovering latent influence in online social activities via shared cascade poisson processes](https://dl.acm.org/doi/10.1145/2487575.2487624),
Iwata, T., Shah, A. and Ghahramani, Z. In KDD. 2013.


## DEMO

- `python3 main.py --retail`  
    to process: https://www.kaggle.com/regivm/retailtransactiondata


## Command line options

- `--input_fn`: csv file name
- `--output_dir`: name of an output directory (default: 'out/tmp')
- `--gamma`: initial value of the decay parameter (default: 1)
- `--beta`: initial value of the Dirichlet paramter (default: 1)
- `--a`: hyperparameter for Gamma distribution (default: 1)
- `--b`: hyperparameter for Gamma distribution (default: 1)
- `--tol`: tolerance for early stopping (default: 100)
- `--max_iter`: maximum number of the EM iteration (default: 20)
- `--time_col`: column name of timestamps (default: 'date')
- `--item_col`: column name of items (default: 'item')
- `--user_col`: column name of users (default: 'user')
- `--n_sample`: number of samples/events from original records (default: 100000)
- `--sampling_rate`: sampling rate when encoding timestamps (default: 'D')


## Reference

- Minka, T., 2000.
    [Estimating a Dirichlet distribution](https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf).
