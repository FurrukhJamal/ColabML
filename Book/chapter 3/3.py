import sys 
assert(sys.version_info) >= (3, 7)
from packaging import version
import sklearn 

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

from pathlib import Path
import tarfile 
import pandas as pd
import urllib.request

def load_data():
    tarfilePath = Path("datasets/titanic.tgz")
    if not tarfilePath.is_file():
        Path("datasets").mkdir(parents= True, exist_ok=True)
        url = "https://homl.info/titanic.tgz"
        urllib.request.urlretrieve(url, tarfilePath)
        with tarfile.open(tarfilePath) as f:
            f.extractall(path = "datasets")
        return [pd.read_csv(Path("datasets/titanic")/filename) for filename in ["train.csv", "test.csv"]]
    else:
        return [pd.read_csv(Path("datasets/titanic")/filename) for filename in ["train.csv", "test.csv"]]

    

if __name__ == "__main__":
    trainCSV, testCSV = load_data()
    print(trainCSV.head())

    trainCSV = trainCSV.set_index("PassengerId")
    testCSV = testCSV.set_index("PassengerId")

    print(trainCSV.info())
    print(f"trainCSV.index.names : {trainCSV.index.names}")
    print(f"testCSV.columns : {testCSV.columns}")

    medianFemaleAge = trainCSV[trainCSV["Sex"] == "female"]["Age"].median()
    print(f"medianFemaleAge : {medianFemaleAge}")

    print(trainCSV.describe())

    print(trainCSV["Survived"].value_counts())

    # looking at the categorical values of sex, pClass and Embarked
    print(trainCSV["Sex"].value_counts())
    print(trainCSV["Pclass"].value_counts())
    print(trainCSV["Embarked"].value_counts())

    from sklearn.pipeline import Pipeline 
    from sklearn.impute import SimpleImputer 
    from sklearn.preprocessing import StandardScaler

    # for numerical attributes
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # for categorical attributes 
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    
    cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),
        ("imputer", SimpleImputer(strategy="most_frequent")), 
        ("cat_encoder", OneHotEncoder(sparse_output = False))
    ])

    from sklearn.compose import ColumnTransformer 
    num_attribs = ["Age", "SibSp", "Parch", "Fare"]
    cat_attribs = ["Pclass", "Sex", "Embarked"]

    preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs), 
        ("cat", cat_pipeline, cat_attribs)
    ])

    xTrain = preprocess_pipeline.fit_transform(trainCSV)
    print(f"\n categorized and standarized training data")
    print(xTrain)

    # getting the labels 
    yTrain = trainCSV["Survived"]

    from sklearn.ensemble import RandomForestClassifier 
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_clf.fit(xTrain, yTrain) 

    xTest = preprocess_pipeline.transform(testCSV)
    yPred = forest_clf.predict(xTest)

    from sklearn.model_selection import cross_val_score
    forest_scores = cross_val_score(forest_clf, xTrain, yTrain, cv =10)

    print(f"forest_training cross val score : {forest_scores.mean()}")

    # scoring via support vector machine 
    from sklearn.svm import SVC 
    svm_clf = SVC(gamma="auto")
    svm_scores = cross_val_score(svm_clf, xTrain, yTrain, cv =10)

    print(f"svm score : {svm_scores.mean()}")

    # checking score on testing values
    # xTest = preprocess_pipeline.fit_transform(xTest)
    # print(testCSV.head())
    # yTest = testCSV["Survived"]
    
    
    # print(f"svm score on testing case : {svm_clf.score(xTest, yTest)}" ) 

    

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot([1]*10, svm_scores, ".")
    plt.plot([2]*10, forest_scores, ".")
    plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
    plt.ylabel("Accuracy")
    plt.show()






    

