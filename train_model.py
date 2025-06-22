from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs 
import os 
import joblib 

# Generate the sample data 
X,_= make_blobs(n_samples=300, centers=3, random_state=42)
model=KMeans(n_clusters=3, random_state=42)

model.fit(X)

#Saving the model 
os.makedirs("model",exist_ok=True)
joblib.dump(model,"model\kmeans.plk")
print("KMeans model trained and saved ! ")
