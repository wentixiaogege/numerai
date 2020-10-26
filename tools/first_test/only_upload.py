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

# Initialize Numerai's API
NAPI = numerapi.NumerAPI(verbosity="info")
# Weights and Biases requires you to add your WandB API key for logging in automatically. Because this is a secret key we will use [Kaggle User Secrets](https://www.kaggle.com/product-feedback/114053) to obfuscate the API key.
# Obfuscated WANDB API Key
# from kaggle_secrets import UserSecretsClient
# WANDB_KEY = '7d8786321b64e818153da23692d69d6ad4387b2e'#UserSecretsClient().get_secret("WANDB_API_KEY")
# wandb.login(key=WANDB_KEY)
# Data directory
DIR = "../working"
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
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Reduce all features to 32-bit floats
    if reduce_memory:
        num_features = [f for f in train.columns if f.startswith("feature")]
        train[num_features] = train[num_features].astype(np.float32)
        test[num_features] = test[num_features].astype(np.float32)

    val = test[test['data_type'] == 'validation']
    return train, val, test

# # Download, unzip and load data
download_current_data(DIR)
# train, val, test = load_data(DIR, reduce_memory=True)

# ## Submission

# You can use this code to upload your predictions directly to Numerai. You will need a public and private API key that you can create from your Numerai account settings.

# Set API Keys for submitting to Numerai
# MODEL_ID = "e3d4d737-d88c-4886-a63e-37e233e5cffd"#wentixiaogege
# MODEL_ID = "018a4fd7-293b-4f67-b1ca-29f77dc848c4" #test1_ljj
MODEL_ID = "4faf6b8a-d41a-45fb-8f4b-9a6b73d93545" #test1_2jj

PUBLIC_ID = "X7YDCLC43O22WJEH2CPFKFRRMBRBUOZ3"
SECRET_KEY = "TVUDVMUTF6PY3UDWIM52ZWPN7YCSHIYGPDEISQSKZG2J5TPFWUPLXEJCHNC3WZ4S"
# Get your API keys and model_id from https://numer.ai/submit
# other users key not working
# PUBLIC_ID = "CYATEL5QQBU6APNFLCV7HEE7PV6SC7V6"
# SECRET_KEY = "Y22BTSUGU4JEFGQB3RZNEESSULKA3HQJPAW3KI6BIXH2AMNMCTC44IFWTOQIO2UW"
# MODEL_ID = "3c77ba09-cfa2-4b18-b789-918340c84c82"
# Initialize API with API Keys
napi = numerapi.NumerAPI(
                         public_id=PUBLIC_ID,
                         secret_key=SECRET_KEY,
                         verbosity="info",show_progress_bars=True)
# Upload predictions for current round
# test.loc[:, "prediction_kazutsugi"] = lgbm_model.predict(test[feature_list], num_iteration=lgbm_model.best_iteration)
# test[['id', "prediction_kazutsugi"]].to_csv("submission.csv", index=False)



# Add your API keys and uncomment the line of code below to automatically upload your predictions to Numerai.

# Upload predictions to Numerai
# tournament=napi.get_current_round()
# print(tournament)
# tournament
napi.upload_predictions("submission.csv",model_id=MODEL_ID)
print("Submission File:")
# test[['id', "prediction_kazutsugi"]].head(2)

# That's all! Note that there is still a lot to be improved and that a good model requires more rigorous evaluation. However, I hope this introduction got you excited about starting with Numerai!

# **If you like this Kaggle notebook, feel free to give an upvote and leave a comment! I will try to implement your suggestions in this kernel!**
