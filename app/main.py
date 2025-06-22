from fastapi import FastAPI , HTTPException 
from app.schema import ClusterInput 
import joblib 
import os 


app = FastAPI(title="KMeans CLustering API")

#loading the trained model 
model=joblib.load("model\kmeans.plk")

@app.get("/")
def root():
    return{"message": "KMeans Clustering is ready "}

@app.post("/predict")
def predict_cluster(input_data: ClusterInput):
    if len(input_data.data) != 2:
        raise HTTPException(status_code=400, detail="Input must be a 2D point [x,y]")
    
    cluster= model.predict([input_data.data])[0]
    return {"cluster_label": int(cluster)}

