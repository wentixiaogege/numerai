```

```

### How to get started with Numerai（如何开始使用Numerai）
#####  The hardest data science tournament on the planet?*（地球上最艰苦的数据科学比赛？）
# <img src="https://miro.medium.com/max/4000/1*g5PtFpII33P5EeHxFZN9YA.png" style="zoom: 15%;" /><img src="https://camo.githubusercontent.com/55d9a214447683aae34c1c84b29fc401201d751b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67" style="zoom:35%;" />
github：【辛苦star！！】

### NumerAI简介

**背景：**前些日子跟好友交流，聊到numerai作为一个公司，不仅是创意和方案都非常有意思，所以决定开始调研一下；机器学习，区块链，同态加密。热点全部链接上了！！

**一句话介绍：** Numerai is building a blockchain based hedge fund using crowdsourced predictions， 中文意思就是说：Numerai正在利用众包预测建立一个基于区块链的对冲基金！！

**基本概念讲解：**

​    对冲基金：https://www.zhihu.com/question/19902214

​    众包预测：https://zhuanlan.zhihu.com/p/56636272，意思就是一群人，一群模型预测

**基本流程：**注册账号-》Settings 找到api key 【注意key是注册的时候在页面最底部出现额】-》运行下文基本逻辑代码 -》提交观察；



**创始人想法：**是利用人工智能【机器学习，区块链，同态加密】、众包预测、虚拟货币的概念来包装自己的对冲基金，先来个名声大噪再说。

**交流方式：**交流既有 [Slack](https://link.zhihu.com/?target=https%3A//numerai.slack.com/) 又有 [Numerai Network](https://link.zhihu.com/?target=https%3A//forum.numer.ai/) ，方便不同习惯的用户。

**私有货币很神奇：**这个系统比较有趣的地方在于，为了解决用户增长的问题，以及在一定程度上解决回测和实盘效果不一致的问题，Numerai还开发了自己的数字货币Numeraire，数据科学家可以在提交结果的时候用Numeraire下注，最终根据实盘结果决定 Numeraire 筹码的增减。这个过程也使 Numeraire 本身越来越有经济价值。( [https://medium.com/numerai/a-new-cryptocurrency-for-coordinating-artificial-intelligence-on-numerai-9251a131419a](https://link.zhihu.com/?target=https%3A//medium.com/numerai/a-new-cryptocurrency-for-coordinating-artificial-intelligence-on-numerai-9251a131419a) )



**提交时间：**

Every Saturday at `18:00 UTC`, a new `round` begins and new `tournament_data` is released. Submit your predictions to Numerai to enter the tournament.The submission deadline is `Monday 14:30 UTC`. Late submissions will not be eligible for payouts.

**国内：换算一下就是每周日2：AM开始，然后周一晚上22：30 结束**

不是北京时间，比北京时间慢了8个小时！！！！

![image-20201010153615483](C:\Users\lijingjie\AppData\Roaming\Typora\typora-user-images\image-20201010153615483.png)

### Weights&Biases模块

可帮助您跟踪机器学习项目。记录运行中的超参数和输出指标，然后可视化和比较结果并快速共享。

**我理解就是配置一下，远程记录，并且做好了不错的展示工具；**

![img](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-MA9nX3-ka1BJ3SSgsbv%2F-MA9nYwfD5Kb-oiPwEtt%2FWorkflow%20June%202020%20v1.png?alt=media&token=e8ddf3ac-4a84-4269-bfd7-f519ffc792ee)

## Numerai + Weights&Biases新手入门实例代码

运行代码：

```
#!/usr/bin/env python
# coding: utf-8
## How to get started with Numerai
## *The hardest data science tournament on the planet?*
# 如何开始使用Numerai
# 地球上最艰苦的数据科学比赛？

# ![](https://miro.medium.com/max/4000/1*g5PtFpII33P5EeHxFZN9YA.png)
# 
# ![](https://camo.githubusercontent.com/55d9a214447683aae34c1c84b29fc401201d751b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67)

# This notebook accompanies the [Weights and Biases Gallery Report](https://app.wandb.ai/gallery) on getting started with [Numerai](https://numer.ai). We will go through the whole process from loading the data to submitting your predictions to Numerai. [Weights and Biases](https://www.wandb.com/) will be used for experiment tracking and hyperparameter optimization.

### Preparation
#### Install Numerai's API
#### pip install numerapi
#### Get the latest version of Weights and Biases
####pip install wandb --upgrade

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
WANDB_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXX'#UserSecretsClient().get_secret("WANDB_API_KEY")
wandb.login(key=WANDB_KEY)
# Data directory
DIR = "working"
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
def load_data(directory: str, reduce_memory: bool=True) -> tuple:
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
def get_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features by calculating statistical moments for each group.

    :param df: Pandas DataFrame containing all features
    """
    for group in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]:
        cols = [col for col in df.columns if group in col]
        df[f"feature_{group}_mean"] = df[cols].mean(axis=1)
        df[f"feature_{group}_std"] = df[cols].std(axis=1)
        df[f"feature_{group}_skew"] = df[cols].skew(axis=1)
    return df
def _train():
    # Configure and train model
    wandb.init(name="LightGBM_sweep")
    lgbm_config = {"num_leaves": wandb.config.num_leaves, "max_depth": wandb.config.max_depth,
                   "learning_rate": wandb.config.learning_rate,
                   "bagging_freq": wandb.config.bagging_freq, "bagging_fraction": wandb.config.bagging_fraction,
                   "feature_fraction": wandb.config.feature_fraction,
                   "metric": 'mse', "random_state": seed}
    lgbm_model = lgb.train(lgbm_config, train_set=dtrain, num_boost_round=750, valid_sets=watchlist,
                           callbacks=[wandb_callback()], verbose_eval=100, early_stopping_rounds=50)

    # Create predictions for evaluation
    val_preds = lgbm_model.predict(val[feature_list], num_iteration=lgbm_model.best_iteration)
    val.loc[:, "prediction_kazutsugi"] = val_preds
    # W&B log metrics
    spearman, payout, numerai_sharpe, mae = evaluate(val)
    wandb.log(
        {"Spearman": spearman, "Payout": payout, "Numerai Sharpe Ratio": numerai_sharpe, "Mean Absolute Error": mae})
### Metrics
# In this experiment we will monitor the Spearman correlation (main metric), the Sharpe ratio, payout and Mean Absolute Error (MAE).
def sharpe_ratio(corrs: pd.Series) -> np.float32:
    """
    Calculate the Sharpe ratio for Numerai by using grouped per-era data

    :param corrs: A Pandas Series containing the Spearman correlations for each era
    :return: A float denoting the Sharpe ratio of your predictions.
    """
    return corrs.mean() / corrs.std()
def evaluate(df: pd.DataFrame) -> tuple:
    """
    Evaluate and display relevant metrics for Numerai

    :param df: A Pandas DataFrame containing the columns "era", "target_kazutsugi" and a column for predictions
    :param pred_col: The column where the predictions are stored
    :return: A tuple of float containing the metrics
    """

    def _score(sub_df: pd.DataFrame) -> np.float32:
        """Calculates Spearman correlation"""
        return spearmanr(sub_df["target_kazutsugi"], sub_df["prediction_kazutsugi"])[0]

    # Calculate metrics
    corrs = df.groupby("era").apply(_score)
    payout_raw = (corrs / 0.2).clip(-1, 1)
    spearman = round(corrs.mean(), 4)
    payout = round(payout_raw.mean(), 4)
    numerai_sharpe = round(sharpe_ratio(corrs), 4)
    mae = mean_absolute_error(df["target_kazutsugi"], df["prediction_kazutsugi"]).round(4)

    # Display metrics
    print(f"Spearman Correlation: {spearman}")
    print(f"Average Payout: {payout}")
    print(f"Sharpe Ratio: {numerai_sharpe}")
    print(f"Mean Absolute Error (MAE): {mae}")
    return spearman, payout, numerai_sharpe, mae


# Download, unzip and load data
download_current_data(DIR)
train, val, test = load_data(DIR, reduce_memory=True)

### Exploratory Data Analysis (EDA)
# The Numerai data has 310 obfuscated numerical features that can hold values of 0.0, 0.25, 0.5, 0.75, 1.00. The features are divided into 6 groups ("intelligence", "wisdom", "charisma", "dexterity", "strength" and "constitution"). The meaning of the groups is unclear, but we can use the fact that features are within the same group.

print("Training data:")
print(train.head(2))
print("Test data:")
print(test.head(2))
print("Training set info:")
print(train.info())
print("Test set info:")
print(test.info())

# When we group by the eras it can be seen that the era sizes change over time. This can be taken into account when creating features using the eras.
# Extract era numbers
train["erano"] = train.era.str.slice(3).astype(int)
plt.figure(figsize=[14, 6])
train.groupby(train['erano'])["target_kazutsugi"].size().plot(title="Era sizes", figsize=(14, 8));
plt.show()

# Most of the features have similar standard deviations, but some have very low variability. Consider standardizing the features or removing these low variability features when experimenting with for example neural networks.

feats = [f for f in train.columns if "feature" in f]
plt.figure(figsize=(15, 5))
sns.distplot(pd.DataFrame(train[feats].std()), bins=100)
sns.distplot(pd.DataFrame(val[feats].std()), bins=100)
sns.distplot(pd.DataFrame(test[feats].std()), bins=100)
plt.legend(["Train", "Val", "Test"], fontsize=20)
plt.title("Standard deviations over all features in the data", weight='bold', fontsize=20);
plt.show()


# ## Feature Engineering
# The features have a remarkably low correlation to the target variable. Even the most correlated features only have around 1.5% correlation with the target. Engineering useful features out of feature + era groups is key for creating good Numerai models.
# 
# Additionally, the importance of features may change over time and by selecting a limited number of features we risk having a high "feature exposure". Feature exposure can be quantified as the standard deviation of all your predictions' correlations  with each feature. You can mitigate this risk by using dimensionality reduction techniques like Principal Component Analysis (PCA) to integrate almost all features into your model.
# 
# One example of creating features out of the groups is to calculate statistical moments (mean, standard deviation, skewness) of every group.
# Add group statistics features
train = get_group_stats(train)
val = get_group_stats(val)
test = get_group_stats(test)

# ## Feature Selection

# The features have a remarkably low correlation to the target variable. Even the most correlated features only have around 1.5% correlation with the target. Engineering useful features out of feature and era groupings is key for creating good Numerai models.
# 
# Also, the importance of features may change over time. By selecting a limited number of features we risk having a high "feature exposure". Feature exposure can be quantified as the standard deviation of all your predictions' correlations  with each feature. You can mitigate this risk by using dimensionality reduction techniques like Principal Component Analysis (PCA) to integrate almost all features into your model. In this starter example we take the 150 features that are most correlated to the target variable.

# Calculate correlations with target
full_corr = train.corr()
corr_with_target = full_corr["target_kazutsugi"].T.apply(abs).sort_values(ascending=False)

# Select features with highest correlation to the target variable
features = corr_with_target[:150]
features.drop("target_kazutsugi", inplace=True)
print("Top 10 Features according to correlation with target:")
print(features[:10])

# Create list of most correlated features
feature_list = features.index.tolist()
### Modeling (using Weights and Biases)

# To get a first good model for Numerai we will train a [LightGBM](https://lightgbm.readthedocs.io/en/latest) model and use Weights and Biases to do a hyperparameter sweep. In this example it will be a grid search over some of the most important hyperparameters for LightGBM. First, we define the configuration of the sweep.

# Configuration for hyperparameter sweep
sweep_config = {
   'method': 'grid',
   'metric': {
          'name': 'mse',
          'goal': 'minimize'   
        },
   'parameters': {
       "num_leaves": {'values': [30, 40, 50]}, 
       "max_depth": {'values': [4, 5, 6, 7]}, 
       "learning_rate": {'values': [0.1, 0.05, 0.01]},
       "bagging_freq": {'values': [7]}, 
       "bagging_fraction": {'values': [0.6, 0.7, 0.8]}, 
       "feature_fraction": {'values': [0.85, 0.75, 0.65]},
   }
}
sweep_id = wandb.sweep(sweep_config, project="numerai_tutorial")


# After that we define a function (_train) using wandb.config attributes so Weights and Biases can perform the grid search. We then log all the results and start the agent.


# Prepare data for LightGBM
dtrain = lgb.Dataset(train[feature_list], label=train["target_kazutsugi"])
dvalid = lgb.Dataset(val[feature_list], label=val["target_kazutsugi"])
watchlist = [dtrain, dvalid]
# Run hyperparameter sweep (grid search)
wandb.agent(sweep_id, function=_train)
# Now the grid search is finished we select the hyperparameters that lead to the highest Sharpe ratio.
# Train model with best configuration
wandb.init(project="numerai_tutorial", name="LightGBM")
best_config = {"num_leaves": 50, "max_depth": 6, "learning_rate": 0.1,
               "bagging_freq": 7, "bagging_fraction": 0.6, "feature_fraction": 0.75,
               "metric": 'mse', "random_state": seed}
lgbm_model = lgb.train(best_config, train_set=dtrain, num_boost_round=750, valid_sets=watchlist, 
                       callbacks=[wandb_callback()], verbose_eval=100, early_stopping_rounds=50)
    
# Create final predictions from best model
train.loc[:, "prediction_kazutsugi"] = lgbm_model.predict(train[feature_list], num_iteration=lgbm_model.best_iteration)
val.loc[:, "prediction_kazutsugi"] = lgbm_model.predict(val[feature_list], num_iteration=lgbm_model.best_iteration)

### Evaluation
# Without much feature engineering it is already possible to get a reasonable score on Numerai. Sharpe ratio is one of the best indications of performance on Numerai because it takes into account the variability across eras.


# Evaluate Model
print("--- Final Training Scores ---")
spearman, payout, numerai_sharpe, mae = evaluate(train)
print("\n--- Final Validation Scores ---")
spearman, payout, numerai_sharpe, mae = evaluate(val)

# Calculate feature exposure
all_features = [col for col in train.columns if 'feature' in col]
feature_spearman_val = [spearmanr(val["prediction_kazutsugi"], val[f])[0] for f in all_features]
feature_exposure_val = np.std(feature_spearman_val).round(4)
print(f"Feature exposure on validation set: {feature_exposure_val}")


# ## Submission

# You can use this code to upload your predictions directly to Numerai. You will need a public and private API key that you can create from your Numerai account settings.

# Set API Keys for submitting to Numerai
PUBLIC_ID = "wentixiaogege@163.com"
SECRET_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Initialize API with API Keys
napi = numerapi.NumerAPI(public_id=PUBLIC_ID, 
                          secret_key=SECRET_KEY, 
                          verbosity="info")
# Upload predictions for current round
test.loc[:, "prediction_kazutsugi"] = lgbm_model.predict(test[feature_list], num_iteration=lgbm_model.best_iteration)
test[['id', "prediction_kazutsugi"]].to_csv("submission.csv", index=False)


# Add your API keys and uncomment the line of code below to automatically upload your predictions to Numerai.

# Upload predictions to Numerai
napi.upload_predictions("submission.csv", tournament=napi.get_current_round())

print("Submission File:")
test[['id', "prediction_kazutsugi"]].head(2)

# That's all! Note that there is still a lot to be improved and that a good model requires more rigorous evaluation. However, I hope this introduction got you excited about starting with Numerai!

# **If you like this Kaggle notebook, feel free to give an upvote and leave a comment! I will try to implement your suggestions in this kernel!**

```

```
运行结果：

D:\Program_Files\Anaconda3\python.exe D:/PycharmProjects/numerai/how-to-get-started-with-numerai.py
2020-10-10 11:20:21,522 INFO wandb: setting login settings: {'api_key': 'XXXXXXXXXX'}
wandb: Currently logged in as: wentixiaogege (use `wandb login --relogin` to force relogin)
wandb: Appending key for api.wandb.ai to your netrc file: C:\Users\XXXXXXXX/.netrc
Downloading new data for round: 232!
working\numerai_dataset_232.zip: 392MB [06:33, 997kB/s]                            
2020-10-10 11:26:58,692 INFO numerapi.base_api: unzipping file...
Loading the data
Training data:
                 id   era  ... feature_wisdom46  target_kazutsugi
0  n000315175b67977  era1  ...             0.75              0.75
1  n0014af834a96cdd  era1  ...             1.00              0.25

[2 rows x 314 columns]
Test data:
                 id     era  ... feature_wisdom46  target_kazutsugi
0  n0003aa52cab36c2  era121  ...              0.0              0.00
1  n000920ed083903f  era121  ...              0.5              0.25

[2 rows x 314 columns]
Training set info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 501808 entries, 0 to 501807
Columns: 314 entries, id to target_kazutsugi
dtypes: float32(310), float64(1), object(3)
memory usage: 608.7+ MB
None
Test set info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1653463 entries, 0 to 1653462
Columns: 314 entries, id to target_kazutsugi
dtypes: float32(310), float64(1), object(3)
memory usage: 2.0+ GB
None

```



有一个日本哥们公开了他的key

https://github.com/Kei-Sanada/Numerai/blob/master/Making_your_first_submission_on_Numerai_20200927.ipynb

[本文章目前只是简要介绍，如果要知道如何挣到钱，请持续关注!!]()

### 附录

numerai 链接：https://docs.numer.ai/tournament/new-users

知乎：https://www.zhihu.com/question/50275557

kaggle：https://www.kaggle.com/carlolepelaars/how-to-get-started-with-numerai

NMR货币：https://coinmarket.no/price/numeraire/

Weigths&Biases: https://docs.wandb.com/

![image-20201010114555943](C:\Users\lijingjie\AppData\Roaming\Typora\typora-user-images\image-20201010114555943.png)