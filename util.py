import os
import shutil
import pandas as pd


def prepare_workspace(output_dir, replace=False):
    if replace == True:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
