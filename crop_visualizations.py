import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = [[N, P, K, temperature, humidity, ph, rainfall ]]
    prediction = model.predict(input_data)
    crop = le.inverse_transform(prediction)[0]
    return crop


# can not directly load the data set as a frame
array = pd.read_csv("E:/Datasets/Crop_recommendation.csv")
df = pd.DataFrame(array)
print(df.duplicated())
df = df.drop_duplicates()

# figsize is used for resizing the figures and grid is used to remove or add the grids
df.hist(figsize=(12,8))
plt.title("g1")
df.hist(figsize = (12, 8), sharey = True)
plt.tight_layout()
plt.title("g2")
plt.show()

plt.figure(figsize=(10, 6))
# df.select_dtyeps(include = "number")  #is used to remove all the string type of attributes it only includes numerical type attributes and correlation heatmap is drawn
# annot = True is allowing the annotation on the graph
# cmap = "YlGnBu" is the color map that uses colores in the graph
sns.heatmap(df.select_dtypes(include = "number").corr(), annot = True, cmap = "YlGnBu")
plt.title("Correlation heatmap")
plt.show()

# Data preprocessing
x = df.drop('label', axis = 1)
y = df['label'] # setting the y to label
le = LabelEncoder() # Encoding the label attributes
y_encoded = le.fit_transform(y) # converts the labels into numerical values like 0,1,2,3,4,...

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size = 0.2, random_state = 42)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

crop = recommend_crop(90, 42, 43, 22.5, 80, 6.5, 120)
print("Recommended Crop:", crop)
