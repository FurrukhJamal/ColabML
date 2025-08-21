import sys 
# version 3.7 or higher 
# print(sys.version_info)
assert sys.version_info >= (3, 7)  ,  "invalid version" 

from packaging import version
import sklearn
print(sklearn.__version__)

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
# print(version.parse(sklearn.__version__))

from pathlib import Path 
import pandas as pd 
import tarfile 
import urllib.request 

s = pd.Series([2, 4, -5 ,10])
print(s)


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents = True, exist_ok = True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz" 
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path = "datasets")
        return pd.read_csv(Path("datasets/housing/housing.csv"))
    else:
        return pd.read_csv(Path("datasets/housing/housing.csv"))
        
housing = load_housing_data()

print(housing.head())
print()
print(housing.info())

print(f"housing[\"ocean_proximity\"].value_counts() : {housing["ocean_proximity"].value_counts()}")

print(housing.describe())

import matplotlib.pyplot as plt 
housing.hist(bins = 50, figsize = (12, 8 ))
plt.show()


import numpy as np 
# the first part as test and the later as train
def shuffle_and_split_data(data, test_ratio):
    shuffle_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set , test_set = shuffle_and_split_data(housing, 0.2)
print(f"len(train_set) : {len(train_set)}")
print(f"len(test_set) : {len(test_set)}")


# hasing of the index you can achieve by np.random.seed(42) too
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# adds an index column 
housing_with_id = housing.reset_index()
# print(housing_with_id)
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")


# or making longitutde lat into ids 
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
# using skiit learn method
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size= 0.2, random_state= 42)

# dividing the median income into 5 categories
housing["income_cat"] = pd.cut(housing["median_income"], 
                               bins = [0, 1.5, 3.0, 4.5, 6,  np.inf],
                               labels = [1, 2 , 3, 4, 5])

housing["income_cat"].value_counts().sort_index().plot.bar(rot = 0, grid = True)
plt.xlabel("income Category")
plt.ylabel("number of districts")
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit 
splitter = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    # print(f"train_index : {train_index}") # gives an list of indecies of size 16516 
    # print(f"test_index : {test_index}")
    # print(f"len(train_index) : {len(train_index)}")
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# getting the 1st split from the stratified sampling 
strat_train_set, strat_test_set = strat_splits[0]

print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))

# getting a single split using train_test_split()
strat_train_set, strat_test_set = train_test_split(housing, test_size = 0.2, random_state = 42, stratify = housing["income_cat"])

print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))

def get_income_cat_porportions(data):
    return data["income_cat"].value_counts()/len(data)

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
compare_props = pd.DataFrame({
    "Overall %" : get_income_cat_porportions(housing),
    "random %" : get_income_cat_porportions(test_set), 
    "Stratified %" :get_income_cat_porportions(strat_test_set)
}).sort_index()

compare_props.index.name = "Income Category"
compare_props["rand error %"] = compare_props["random %"]/compare_props["Overall %"] - 1
compare_props["strat error %"] = compare_props["Stratified %"]/compare_props["Overall %"] - 1

(compare_props * 100).round(2)
print(compare_props)

# deleting the column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", inplace = True, axis = 1)


# making a copy of training set to explore it and have safe copy of orginal 
housing = strat_train_set.copy()

# scattering using longitude lat 
housing.plot(grid = True, kind = "scatter", x = "longitude" , y = "latitude", alpha = 0.2)
plt.show()

housing.plot(kind = "scatter", x = "longitude", y = "latitude", s = housing["population"]/ 100 , c = "median_house_value", cmap = "jet", colorbar = True, legend = True, sharex = True, figsize = (10, 7))
plt.show()
# standard correlation coefficient 1 for positive relation -1 for negative
corr_matrix = housing.corr(numeric_only = True)
print(corr_matrix["median_house_value"].sort_values(ascending = False))

from pandas.plotting import scatter_matrix 

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age" ]
scatter_matrix(housing[attributes], figsize = (12, 8))
plt.show()

housing.plot(kind = "scatter", x= "median_income", y= "median_house_value", grid = True, alpha = 0.1)
plt.show()

# adding some more data 
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only = True)
print(corr_matrix["median_house_value"].sort_values(ascending = False))

# preparing to feed the data in ML algorithm
housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()

# housing.info()

# filling the missing total_bedrooms
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median, inplace = True)

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy="median")

# getting only numeric values 
housing_num = housing.select_dtypes(include = [np.number])

# filling the imputer instance so it can fill the missing values with median of that column in the following steps
imputer.fit(housing_num)

# all the medians stored in
imputer.statistics_

# they are same as 
housing_num.median().values 

# filling the values with the mean
X = imputer.transform(housing_num)

# scikit returns the np array so adding the column and row names back
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index  = housing_num.index)

# to convert text field in house["ocean_proximity"] into a numeric field
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))
from sklearn.preprocessing import OrdinalEncoder 
ordinal_encoder = OrdinalEncoder() 
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# dispaly a list of categories
print(ordinal_encoder.categories_)

# converting the same text field in housing[ocean_proximity] to one hot 
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# you can convert it into nparray 
housing_cat_1hot.toarray()

# scaling the values, normalization
from sklearn.preprocessing import MinMaxScaler 
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# standardizing the value... subtracting the mean than dividing by std 
from sklearn.preprocessing import StandardScaler 
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# page 122 book
from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing[["housing_median_age"]],
[[35]], gamma=0.1)
# end


from sklearn.linear_model import LinearRegression 
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]  #pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# print(f"actual label : {housing_labels} predictions : {predictions}")
print("some_new_data")
print(some_new_data)

from sklearn.compose import TransformedTargetRegressor 
model = TransformedTargetRegressor(LinearRegression(), transformer = StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
print("predictions")
print(predictions)


# transforming population to log 
from sklearn.preprocessing import FunctionTransformer 
log_transformer = FunctionTransformer(np.log, inverse_func = np.exp)
log_pop = log_transformer.transform(housing[["population"]])

from sklearn.pipeline import Pipeline 
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy = "median")),
    ("standardize", StandardScaler())
])

from sklearn.pipeline import make_pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy="median"),StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housing_num)
print(housing_num_prepared[:2].round(2))

