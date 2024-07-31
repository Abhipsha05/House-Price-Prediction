import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import calendar
from pandas.api.types import CategoricalDtype
import pickle
from pandas.api.types import CategoricalDtype

from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('A:\\Projects\\House Price Prediction\\train.csv')
df_test = pd.read_csv('A:\\Projects\\House Price Prediction\\test.csv')

print(df_train.shape)
print(df_test.shape)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
df_train.head()
df_test.head()

# Integrating Data
df = pd.concat([df_train, df_test])
print("Shape of Integrated Data / DF: ", df.shape)
df.head(5)
df.tail(5)
df.info()
int_features = df.select_dtypes(include=["int64"]).columns
print("Total number of integer features: ", int_features.shape[0])
print("List of integer features: ", int_features.tolist())
float_features = df.select_dtypes(include=["float64"]).columns
print("Total number of float features: ", float_features.shape[0])
print("List of float features: ", float_features.tolist())
category_features = df.select_dtypes(include=["object"]).columns
print("Total number of categorical features: ", category_features.shape[0])
print("List of categorical features: ", category_features.tolist())

df.describe()

# Handling missing values
# plt.figure(figsize=(16,9))
# sns.heatmap(df.isnull())
# null value detection
# Most null value features : Alley, FireplaceQu, PoolQC, Fence, MiscFeature
# plt.savefig("A:\\Projects\\House Price Prediction\\heatmap_df_of_null.png")

# Set index as ID column
df = df.set_index("Id")
df_train.set_index("Id", inplace=True)
df_test.set_index("Id", inplace=True)

null_count = df.isnull().sum()
null_count
null_percent = df.isnull().sum() / df.shape[0] * 100
null_percent

miss_value_50_perc = null_percent[null_percent > 50]
miss_value_50_perc
# Won't drop, replace it with constants
df["Alley"].value_counts()
miss_value_20_50_perc = null_percent[(null_percent > 20) & (null_percent < 50)]
miss_value_20_50_perc
miss_value_5_20_perc = null_percent[(null_percent > 5) & (null_percent < 20)]
miss_value_5_20_perc
# sns.heatmap(df[miss_value_5_20_perc.keys()].isnull())

# Handling missing values
missing_value_features = null_percent[null_percent > 0]
print("Total number of missing value features = ", len(missing_value_features))
missing_value_features
missing_categorical_features = missing_value_features[missing_value_features.keys().isin(category_features)]
print("Total number of missing categorical value features = ", len(missing_categorical_features))
missing_categorical_features
missing_integer_features = missing_value_features[missing_value_features.keys().isin(int_features)]
print("Total number of missing integer value features = ", len(missing_integer_features))
missing_integer_features
missing_float_features = missing_value_features[missing_value_features.keys().isin(float_features)]
print("Total number of missing float value features = ", len(missing_float_features))
missing_float_features

# Handling MSZoning
df["MSZoning"].value_counts()
# sns.countplot(x = "MSZoning", data = df)
# Backup of original data
df_mvi = df.copy()
df_mvi.shape
mszoning_mode = df["MSZoning"].mode()[0]
df_mvi["MSZoning"].replace(np.nan, mszoning_mode, inplace=True)
df_mvi["MSZoning"].isnull().sum()

# def oldNewCountPlot(df, df_new, feature):
#     plt.figure(figsize=(14, 6))
#     plt.subplot(121)
#     sns.countplot(x=feature, data=df)
#     plt.title("Old Data Distribution")
#     plt.subplot(122)
#     sns.countplot(x=feature, data=df_new)
#     plt.title("New Data Distribution")
#     plt.tight_layout()
#     plt.show()
# oldNewCountPlot(df, df_mvi, "MSZoning")

# Handling Alley
df["Alley"].value_counts()
# sns.countplot(x = "Alley", data = df)
alley_const = "NA"
df_mvi["Alley"].replace(np.nan, alley_const, inplace=True)
df_mvi["Alley"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "Alley")

# Handling LotFrontage
def boxHistPlot(series, figsize=(16, 5)):
    plt.figure(figsize=figsize)
    plt.subplot(121)
    series.plot.box()
    plt.subplot(122)
    series.plot.hist()
    plt.show()

# LotFrontage
df["LotFrontage"].describe()
lotfrontage_median = df["LotFrontage"].median()
df_mvi["LotFrontage"].replace(np.nan, lotfrontage_median, inplace=True)
df_mvi["LotFrontage"].isnull().sum()
# boxHistPlot(df["LotFrontage"])
# boxHistPlot(df_mvi["LotFrontage"])

# Handling Utilities
df["Utilities"].value_counts()
df.drop(columns="Utilities", inplace=True)

# Handling Exterior
df["Exterior1st"].value_counts()
# sns.countplot(x = "Exterior1st", data = df)
exterior1st_mode = df["Exterior1st"].mode()[0]
df_mvi["Exterior1st"].replace(np.nan, exterior1st_mode, inplace=True)
df_mvi["Exterior1st"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "Exterior1st")

df["Exterior2nd"].value_counts()
# sns.countplot(x = "Exterior2nd", data = df)
exterior2nd_mode = df["Exterior2nd"].mode()[0]
df_mvi["Exterior2nd"].replace(np.nan, exterior2nd_mode, inplace=True)
df_mvi["Exterior2nd"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "Exterior2nd")

# Handling MasVnrType and MasVnrArea
df["MasVnrType"].value_counts()
# sns.countplot(x = "MasVnrType", data = df)
masvnrtype_mode = df["MasVnrType"].mode()[0]
df_mvi["MasVnrType"].replace(np.nan, masvnrtype_mode, inplace=True)
df_mvi["MasVnrType"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "MasVnrType")

df["MasVnrArea"].describe()
masvnrarea_median = df["MasVnrArea"].median()
df_mvi["MasVnrArea"].replace(np.nan, masvnrarea_median, inplace=True)
df_mvi["MasVnrArea"].isnull().sum()
# boxHistPlot(df["MasVnrArea"])
# boxHistPlot(df_mvi["MasVnrArea"])

# Handling BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
df["BsmtQual"].value_counts()
# sns.countplot(x = "BsmtQual", data = df)
bsmtqual_mode = df["BsmtQual"].mode()[0]
df_mvi["BsmtQual"].replace(np.nan, bsmtqual_mode, inplace=True)
df_mvi["BsmtQual"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "BsmtQual")

df["BsmtCond"].value_counts()
# sns.countplot(x = "BsmtCond", data = df)
bsmtcond_mode = df["BsmtCond"].mode()[0]
df_mvi["BsmtCond"].replace(np.nan, bsmtcond_mode, inplace=True)
df_mvi["BsmtCond"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "BsmtCond")

df["BsmtExposure"].value_counts()
# sns.countplot(x = "BsmtExposure", data = df)
bsmt_exposure_mode = df["BsmtExposure"].mode()[0]
df_mvi["BsmtExposure"].replace(np.nan, bsmt_exposure_mode, inplace=True)
df_mvi["BsmtExposure"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "BsmtExposure")

df["BsmtFinType1"].value_counts()
# sns.countplot(x = "BsmtFinType1", data = df)
bsmtfintype1_mode = df["BsmtFinType1"].mode()[0]
df_mvi["BsmtFinType1"].replace(np.nan, bsmtfintype1_mode, inplace=True)
df_mvi["BsmtFinType1"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "BsmtFinType1")

df["BsmtFinType2"].value_counts()
# sns.countplot(x = "BsmtFinType2", data = df)
bsmtfintype2_mode = df["BsmtFinType2"].mode()[0]
df_mvi["BsmtFinType2"].replace(np.nan, bsmtfintype2_mode, inplace=True)
df_mvi["BsmtFinType2"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "BsmtFinType2")

# Handling Electrical
df["Electrical"].value_counts()
# sns.countplot(x = "Electrical", data = df)
electrical_mode = df["Electrical"].mode()[0]
df_mvi["Electrical"].replace(np.nan, electrical_mode, inplace=True)
df_mvi["Electrical"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "Electrical")

# Handling KitchenQual
df["KitchenQual"].value_counts()
# sns.countplot(x = "KitchenQual", data = df)
kitchenqual_mode = df["KitchenQual"].mode()[0]
df_mvi["KitchenQual"].replace(np.nan, kitchenqual_mode, inplace=True)
df_mvi["KitchenQual"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "KitchenQual")

# Handling Functional
df["Functional"].value_counts()
# sns.countplot(x = "Functional", data = df)
functional_mode = df["Functional"].mode()[0]
df_mvi["Functional"].replace(np.nan, functional_mode, inplace=True)
df_mvi["Functional"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "Functional")

# Handling GarageType
df["GarageType"].value_counts()
# sns.countplot(x = "GarageType", data = df)
garagetype_mode = df["GarageType"].mode()[0]
df_mvi["GarageType"].replace(np.nan, garagetype_mode, inplace=True)
df_mvi["GarageType"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "GarageType")

# Handling GarageYrBlt
df["GarageYrBlt"].describe()
garageyrblt_median = df["GarageYrBlt"].median()
df_mvi["GarageYrBlt"].replace(np.nan, garageyrblt_median, inplace=True)
df_mvi["GarageYrBlt"].isnull().sum()
# boxHistPlot(df["GarageYrBlt"])
# boxHistPlot(df_mvi["GarageYrBlt"])

# Handling GarageFinish
df["GarageFinish"].value_counts()
# sns.countplot(x = "GarageFinish", data = df)
garagefinish_mode = df["GarageFinish"].mode()[0]
df_mvi["GarageFinish"].replace(np.nan, garagefinish_mode, inplace=True)
df_mvi["GarageFinish"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "GarageFinish")

# Handling GarageQual
df["GarageQual"].value_counts()
# sns.countplot(x = "GarageQual", data = df)
garagequal_mode = df["GarageQual"].mode()[0]
df_mvi["GarageQual"].replace(np.nan, garagequal_mode, inplace=True)
df_mvi["GarageQual"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "GarageQual")

# Handling GarageCond
df["GarageCond"].value_counts()
# sns.countplot(x = "GarageCond", data = df)
garagecond_mode = df["GarageCond"].mode()[0]
df_mvi["GarageCond"].replace(np.nan, garagecond_mode, inplace=True)
df_mvi["GarageCond"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "GarageCond")

# Handling SaleType
df["SaleType"].value_counts()
# sns.countplot(x = "SaleType", data = df)
saletype_mode = df["SaleType"].mode()[0]
df_mvi["SaleType"].replace(np.nan, saletype_mode, inplace=True)
df_mvi["SaleType"].isnull().sum()
# oldNewCountPlot(df, df_mvi, "SaleType")

# ### Feature Transformation

# Converting Numerical features to Categorical features
num_conv = ["MSSubClass", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "MoSold", "YrSold"]
for feat in num_conv:
    print(f"{feat}: data type = {df_mvi[feat].dtype}")
df_mvi[num_conv].head()

df_mvi["MoSold"].unique()

df_mvi["MoSold"] = df_mvi["MoSold"].apply(lambda x: calendar.month_abbr[x])
print(df_mvi["MoSold"].unique())

for feat in num_conv:
    df_mvi[feat]= df_mvi[feat].astype(str)   

df_mvi.shape


# Converting Categorical features to numerical features
#Ordinal Encoding
ordinal_var = ['ExterQual',
               'ExterCond',
               'BsmtQual',
               'BsmtCond',
               'BsmtExposure',
               'BsmtFinType1',
               'BsmtFinType2',
               'BsmtFinSF1',
               'HeatingQC', 
               'KitchenQual',
               'Functional',
               'FireplaceQu',
               'GarageQual',
               'GarageCond',
               'GarageFinish',
               'PavedDrive',
               'PoolQC',
               'Utilities']
print("Total no. of features to convert into ordinal numerical format :", len(ordinal_var))
df_mvi["ExterQual"] = df_mvi["ExterQual"].astype(CategoricalDtype(categories=["Po", "Fa", "TA","Gd","Ex"], ordered = True)).cat.codes
df_mvi["BsmtExposure"]= df_mvi["BsmtExposure"].astype(CategoricalDtype(categories=["NA", "No", "Mn","Av","Gd"], ordered = True)).cat.codes
df_mvi["ExterCond"] = df_mvi["ExterCond"].astype(CategoricalDtype(categories=["Po", "Fa", "TA","Gd","Ex"], ordered = True)).cat.codes
df_mvi["BsmtQual"] = df_mvi["BsmtQual"].astype(CategoricalDtype(categories=["NA", "Po", "Fa","TA","Gd", "Ex"], ordered = True)).cat.codes
df_mvi["BsmtCond"] = df_mvi["BsmtCond"].astype(CategoricalDtype(categories=["NA", "Po", "Fa","TA","Gd", "Ex"], ordered = True)).cat.codes
df_mvi["BsmtFinType1"] = df_mvi["BsmtFinType1"].astype(CategoricalDtype(categories=["NA", "Unf", "LwQ","Rec","BLQ", "ALQ","GLQ"], ordered = True)).cat.codes
df_mvi["BsmtFinType2"] = df_mvi["BsmtFinType2"].astype(CategoricalDtype(categories=["NA", "Unf", "LwQ","Rec","BLQ", "ALQ","GLQ"], ordered = True)).cat.codes
df_mvi["HeatingQC"] = df_mvi["HeatingQC"].astype(CategoricalDtype(categories=["Po", "Fa", "TA","Gd","Ex"], ordered = True)).cat.codes
df_mvi["KitchenQual"] = df_mvi["KitchenQual"].astype(CategoricalDtype(categories=["Po", "Fa", "TA","Gd","Ex"], ordered = True)).cat.codes
df_mvi["FireplaceQu"] = df_mvi["FireplaceQu"].astype(CategoricalDtype(categories=["NA", "Po", "Fa","TA","Gd", "Ex"], ordered = True)).cat.codes
df_mvi["GarageQual"] = df_mvi["GarageQual"].astype(CategoricalDtype(categories=["NA", "Po", "Fa","TA","Gd", "Ex"], ordered = True)).cat.codes
df_mvi["GarageCond"] = df_mvi["GarageCond"].astype(CategoricalDtype(categories=["NA", "Po", "Fa","TA","Gd", "Ex"], ordered = True)).cat.codes
df_mvi["PoolQC"] = df_mvi["PoolQC"].astype(CategoricalDtype(categories=["NA", "Fa","TA","Gd", "Ex"], ordered = True)).cat.codes
df_mvi["Functional"] = df_mvi["Functional"].astype(CategoricalDtype(categories=["Sal", "Sev", "Maj2","Maj1","Mod", "Min2", "Min1", "Typ"], ordered = True)).cat.codes
df_mvi["GarageFinish"] = df_mvi["GarageFinish"].astype(CategoricalDtype(categories=["NA", "Unf", "RFn","Fin"], ordered = True)).cat.codes
df_mvi["PavedDrive"] = df_mvi["PavedDrive"].astype(CategoricalDtype(categories=["N", "P", "Y"], ordered = True)).cat.codes
df_mvi["Utilities"] = df_mvi["Utilities"].astype(CategoricalDtype(categories=["ELO", "NASeWa", "NASeWr","AllPub"], ordered = True)).cat.codes
df_mvi.info()
plt.figure(figsize=(16,9))
#sns.heatmap(df_mvi.isnull())
#Nominal Encoding
df_encod = df_mvi.copy()
object_features = df_encod.select_dtypes(include="object").columns.tolist()
print("Object features before encoding:", object_features)
print("Total object data type features:", len(object_features))
df_encod = pd.get_dummies(df_encod, columns=object_features, drop_first=True)
print("Shape after encoding:", df_encod.shape)  # Should be (2919, 1118)
print(df_encod.columns)
df_encod.select_dtypes(include="object").columns.tolist()

# ### Split data for Training and Testing

df_encod.shape
# Check for missing values in df_encod
missing_values_encod = df_encod.isna().sum()
print(missing_values_encod[missing_values_encod > 0])
#Handling the remaining (if any) missing values
numerical_columns = df_encod.select_dtypes(include=['float64', 'int64']).columns
df_encod[numerical_columns] = df_encod[numerical_columns].fillna(df_encod[numerical_columns].mean())
# Check for missing values in df_encod
missing_values_encod = df_encod.isna().sum()
# Print columns in df_encod with missing values
print(missing_values_encod[missing_values_encod > 0])

len_train = df_train.shape[0]
len_train
df_encod.shape

x_train = df_encod[:len_train].drop("SalePrice", axis=1)
y_train = df_encod["SalePrice"][:len_train]
x_test = df_encod[len_train:].drop("SalePrice", axis=1)

print("Shape of x_train data: ", x_train.shape)
print("Shape of y_train data: ", y_train.shape)
print("Shape of x_test data: ", x_test.shape)

assert (x_train.index == df_train.index).all(), "Indices of x_train and df_train do not match"
assert (x_test.index == df_test.index).all(), "Indices of x_test and df_test do not match"

x_train = df_encod.loc[df_train.index]

# ### Feature Scaling

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, x_train, y_train):
    cv = KFold(n_splits=7, shuffle=True, random_state=45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, x_train, y_train, cv=cv, scoring=r2)
    score = [r2_val_score.mean()]
    return score

score = test_model(gbr, x_train, y_train)
print("Score of model:", score)
gbr.fit(x_train, y_train)

def predict_saleprice(house_id):
    if house_id not in df_encod.index:
        raise ValueError("ID not found in the test data")

    house_features = df_encod.loc[house_id].values.reshape(1, -1)
    saleprice = gbr.predict(house_features)
    return saleprice[0]

house_id = 14
predicted_price = predict_saleprice(house_id)
print(f"The predicted SalePrice for house ID {house_id} is: {predicted_price}")
house_id = 2
predicted_price = predict_saleprice(house_id)
print(f"The predicted SalePrice for house ID {house_id} is: {predicted_price}")
house_id = 100
predicted_price = predict_saleprice(house_id)
print(f"The predicted SalePrice for house ID {house_id} is: {predicted_price}")
house_id = 1339
predicted_price = predict_saleprice(house_id)
print(f"The predicted SalePrice for house ID {house_id} is: {predicted_price}")

import json
import pickle
import os

columns = {
    'data_columns': [col.lower() for col in df_encod.columns]
}
folder_path = 'A:\\Projects\House Price Prediction'
os.makedirs(folder_path, exist_ok=True)

columns_json_path = os.path.join(folder_path, 'columns.json')
with open(columns_json_path, 'w') as json_file:
    json.dump(columns, json_file)
print(f"JSON file created at: {columns_json_path}")

pickle_path = os.path.join(folder_path, 'HOUSE_PRICE_PREDICTION.pickle')
with open(pickle_path, 'wb') as f:
    pickle.dump(gbr, f)
print(f"Pickle file created at: {pickle_path}")



