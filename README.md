
# California Housing Price Predictor (Gradio + scikit-learn)

**What it does:** Predicts median California house value from 8 features (income, rooms, location, etc.) using a RandomForest in a scikit-learn Pipeline. Includes single input and (optional) batch CSV predictions.

**Live demo:**  
[![Hugging Face Space](https://img.shields.io/badge/Spaces-Live-blue)](https://rohithvarmasuraparaju-cali-housing.hf.space/)

## Stack
- Python, scikit-learn, pandas, numpy
- RandomForestRegressor + Pipeline (StandardScaler)
- Gradio UI (sliders for inputs; optional CSV upload)
- Saved model: `models/rf_cali_housing.joblib`

## Dataset
California Housing (1990 U.S. Census), available via `sklearn.datasets.fetch_california_housing`.  
Pace & Barry (1997), *Sparse Spatial Autoregressions*, *Statistics & Probability Letters*.

## How to use
- **Single prediction:** adjust sliders → **Predict** → view USD output.  
- **Batch prediction:** upload CSV with columns:  
  `MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude` → download results.

## Summary
End-to-end ML regression predicting California median house value from 8 features. Trained a RandomForest in a scikit-learn Pipeline, evaluated with MAE/RMSE/R², and explained with feature importances. Includes saved model and a quick prediction demo.

