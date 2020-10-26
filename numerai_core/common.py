#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import random as rn
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import numerapi
import wandb
from wandb.lightgbm import wandb_callback

# Initialize Numerai's API
NAPI = numerapi.NumerAPI(verbosity="info")
# Weights and Biases requires you to add your WandB API key for logging in automatically. Because this is a secret key we will use [Kaggle User Secrets](https://www.kaggle.com/product-feedback/114053) to obfuscate the API key.
# Obfuscated WANDB API Key
# from kaggle_secrets import UserSecretsClient
# WANDB_KEY = '7d8786321b64e818153da23692d69d6ad4387b2e'#UserSecretsClient().get_secret("WANDB_API_KEY")
# wandb.login(key=WANDB_KEY)
# Data directory
DIR = "../tools/working"
# Set seed for reproducability
seed = 1234
rn.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# Surpress Pandas warnings
pd.set_option('chained_assignment', None)

# ## Data Processing
def download_current_data(directory: str):
    """
    Downloads the data for the current round
    :param directory: The path to the directory where the data needs to be saved
    """
    current_round = NAPI.get_current_round()
    if os.path.isdir(f'{directory}/numerai_dataset_{current_round}/'):
        print(f"You already have the newest data! Current round is: {current_round}")
    else:
        print(f"Downloading new data for round: {current_round}!")
        NAPI.download_current_dataset(dest_path=directory, unzip=True)
        df_test = pd.read_csv(
            "https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz")
        df_test.to_csv(f'{directory}/numerai_dataset_{current_round}/latest_numerai_tournament_data.csv',index=False)

def load_data(directory: str, reduce_memory: bool = True) -> tuple:
    """
    Get data for current round
    :param directory: The path to the directory where the data needs to be saved
    :return: A tuple containing the datasets
    """
    print('Loading the data')
    full_path = f'{directory}/numerai_dataset_{NAPI.get_current_round()}/'
    train_path = full_path + 'numerai_training_data.csv'
    test_path = full_path + 'numerai_tournament_data.csv'
    test_path1 = full_path + 'latest_numerai_tournament_data.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test1 = pd.read_csv(test_path1)
    if test.shape[0] != test1.shape[0]:
        print('same issue still happends !! ',test.shape[0],test1.shape[0])
    test = test if test.shape[0] > test1.shape[0] else test1 # using the larger rows

    # Reduce all features to 32-bit floats
    if reduce_memory:
        num_features = [f for f in train.columns if f.startswith("feature")]
        train[num_features] = train[num_features].astype(np.float32)
        test[num_features] = test[num_features].astype(np.float32)

    val = test[test['data_type'] == 'validation']
    return train, val, test