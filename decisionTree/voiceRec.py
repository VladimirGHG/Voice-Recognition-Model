import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

voice_data = pd.read_csv("decisionTree/voice.csv")

X = voice_data.drop("label", axis=1)
y = voice_data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=10)
clf.fit(X_train, y_train)

joblib.dump(clf, "voice_gender_model.pkl")
