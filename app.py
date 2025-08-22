import tempfile
import pandas as pd
import gradio as gr

FEATURES = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']

def predict_csv(file):
    df = pd.read_csv(file.name)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise gr.Error(f"Missing required columns: {missing}")
    preds_100k = model.predict(df[FEATURES])
    df_out = df.copy()
    df_out["PredictedPriceUSD"] = (preds_100k * 100000).round(0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df_out.to_csv(tmp.name, index=False)
    return df_out, tmp.name

with gr.Blocks(title="California Housing Price Predictor") as demo:
    gr.Markdown("### Predict California median house value\nRandomForestRegressor in a scikit-learn Pipeline.")

    with gr.Tab("Single prediction"):
        with gr.Row():
            medinc     = gr.Slider(0.5, 15, value=5.0, step=0.1, label="Median Income (×$10k)")
            house_age  = gr.Slider(1, 60, value=25, step=1, label="House Age (years)")
            ave_rooms  = gr.Slider(1.0, 10.0, value=6.0, step=0.1, label="Avg Rooms")
            ave_bedrms = gr.Slider(0.5, 5.0, value=1.0, step=0.1, label="Avg Bedrooms")
        with gr.Row():
            population = gr.Slider(50, 50000, value=1000, step=50, label="Population")
            ave_occup  = gr.Slider(0.5, 10.0, value=3.0, step=0.1, label="Avg Occupancy")
            latitude   = gr.Slider(32.0, 42.0, value=34.2, step=0.1, label="Latitude")
            longitude  = gr.Slider(-125.0, -114.0, value=-118.3, step=0.1, label="Longitude")
        out_single = gr.Textbox(label="Predicted Median House Value")
        gr.Button("Predict", variant="primary").click(
            predict, [medinc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude], out_single
        )

    with gr.Tab("Batch CSV"):
        gr.Markdown("Upload a CSV with columns: " + ", ".join(FEATURES))
        csv_in = gr.File(file_types=[".csv"], label="Upload CSV")
        df_out = gr.Dataframe(label="Preview predictions")
        file_out = gr.File(label="Download predictions.csv")
        gr.Button("Run batch").click(predict_csv, inputs=[csv_in], outputs=[df_out, file_out])

demo.queue().launch()
