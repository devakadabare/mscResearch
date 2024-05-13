from fastapi import FastAPI, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from prophet import Prophet
import pandas as pd
import os
import pickle
from typing import Optional,List
from pydantic import BaseModel
from numpy import int64

app = FastAPI()

# Path to the directory where models are stored
PRODUCT_MODEL_DIR = "D:/Research/Code/Web API/models/productWise/"
PRODUCTWISE_USER_MODEL_DIR = "D:/Research/Code/Web API/models/productWiseUser/"

# Dictionary to hold the loaded models
models = {}
user_models = {}

class UserPredictionRequest(BaseModel):
    start_date: str
    end_date: str
    products: List[str]

# Function to remove "model_" prefix from product names
def remove_model_prefix(data):
    for key, products in data.items():
        for product in products:
            product[0] = product[0].replace("model_", "")
    return data

# Function to load a model for a given product
def load_model(product_name):
    model_path = os.path.join(PRODUCT_MODEL_DIR, f"{product_name}.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No model found for {product_name}")


# Function to load a user model for a given product
def load_user_model(product_name):
    model_path = os.path.join(PRODUCTWISE_USER_MODEL_DIR, f"{product_name}.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No model found for {product_name}")

# Load all models at app startup
@app.on_event("startup")
def load_models():
    global models
    # List all product model files in the models directory
    for filename in os.listdir(PRODUCT_MODEL_DIR):
        if filename.endswith(".pkl"):
            product_name = filename[:-4]  # Remove the .pkl extension to get the product name
            models[product_name] = load_model(product_name)

@app.on_event("startup")
def load_user_models():
    global user_models
    # List all product user model files in the models directory
    for filename in os.listdir(PRODUCTWISE_USER_MODEL_DIR):
        if filename.endswith(".pkl"):
            product_name = filename[:-4]  # Remove the _model.pkl extension to get the product name
            user_models[product_name] = load_user_model(product_name)

@app.get("/predictions/products")
async def get_predictions(start_date: str, end_date: str, top_n: int = 5):
    
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])

    results = {}
    for product, model in models.items():
        forecast = model.predict(future)
        total_sales = forecast['yhat'].sum()
        results[product] = total_sales

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    most_selling = sorted_results[:top_n]
    least_selling = sorted_results[-top_n:]

    final_result = jsonable_encoder({"most_selling": most_selling, "least_selling": least_selling}, custom_encoder={int64: int})
    final_result = remove_model_prefix(final_result)
    
    return final_result

# @app.post("/predictions/users")
# async def get_user_predictions(request: PredictionRequest):
#     start_date = pd.to_datetime(request.start_date)
#     end_date = pd.to_datetime(request.end_date)
#     products = request.products
#     print("Inside user predictions", products, start_date, end_date)
#     try:
#         start_date = pd.to_datetime(start_date)
#         end_date = pd.to_datetime(end_date)
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

#     future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
    
#     results = {}
#     for product in products:
#         new_product_name = product + "_model"
#         if new_product_name in user_models:
#             modelName = new_product_name
#             model = user_models[modelName]
#             if(model == None):
#                 print("Model not found")

#             forecast = model.predict(future)
#             total_sales = forecast['yhat'].sum()
#             results[product] = {"total_sales": total_sales}
#         else:
#             results[product] = {"error": "No model available"}

#     return {"predictions": results}

def convert_numpy(obj):
    if isinstance(obj, int64):
        return int(obj)  # or use obj.item() to convert numpy types to native Python types
    raise TypeError

@app.post("/predictions/users")
async def get_user_predictions(request: UserPredictionRequest):
    try:
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")
    
    future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
    
    results = {}
    for product in request.products:
        new_product_name = product + "_model"
        if new_product_name in user_models:
            model_dict = user_models[new_product_name]
            product_results = {}
            for user_id, model in model_dict.items():
                forecast = model.predict(future)
                forecast['will_buy'] = forecast['yhat'].apply(lambda x: int(x > 0.5))  # Adjust threshold as needed
                if forecast['will_buy'].sum() > 0:
                    product_results[user_id] = forecast[forecast['will_buy'] == 1]['ds'].tolist()
            results[product] = product_results
        else:
            results[product] = {"error": "No model available"}
    final_result = jsonable_encoder({"predictions": results}, custom_encoder={int64: int})
    final_result = converted_data = {product: list(details.keys()) for product, details in final_result['predictions'].items()}
    return jsonable_encoder(final_result, custom_encoder={int64: int})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
