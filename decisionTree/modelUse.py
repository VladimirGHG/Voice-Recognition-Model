from voiceExtractor import extract_features
import joblib

clf = joblib.load("voice_gender_model.pkl")

features = extract_features("decisionTree/recording1.wav")

prediction = clf.predict(features)[0]
print("Prediction:", prediction)
