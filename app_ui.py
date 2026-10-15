import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, date

# Configure page layouts and titles
st.set_page_config(page_title="Aedes Breeding Ground Predictor", layout="centered")
st.title("🦟 Aedes Breeding Ground Predictor")
st.write("Enter target parameters below to predict the next breeding location coordinates using the trained Random Forest model.")

# Load the saved model artifact
MODEL_PATH = 'aedes_random_forest_model.joblib'
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' was not found. Please upload it alongside this script.")
else:
    # Load components safely
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    label_encoders = artifact['label_encoders']
    translation_dict = artifact['translation_dict']
    categorical_cols = artifact['categorical_cols']
    target_col = artifact['target_col']
    feature_cols = artifact['feature_cols']

    # Setup UI Form inputs
    st.subheader("Observation Features")
    
    pembiakan_classes = list(label_encoders['pembiakan'].classes_)
    pembiakan_val = st.selectbox("Breeding Type (pembiakan)", options=pembiakan_classes)
    bekas_diperiksa_val = st.number_input("Containers Examined (bekas diperiksa)", min_value=1, max_value=500, value=15)
    container_index_val = st.number_input("Container Index", min_value=0.0, max_value=100.0, value=8.0)

    # 1. Parse trained date strings into date objects
    tarikh_classes = list(label_encoders['Tarikh'].classes_)
    trained_dates_map = {}
    for d_str in tarikh_classes:
        try:
            # Training format is Day/Month/Year (e.g., 1/10/2024)
            d_obj = datetime.strptime(d_str, "%d/%m/%Y").date()
            trained_dates_map[d_obj] = d_str
        except ValueError:
            continue

    # 2. Realtime date picker defaulting to current date
    selected_date = st.date_input("Date (Tarikh)", value=date.today())

    # 3. Match format or find nearest date
    formatted_date_str = f"{selected_date.day}/{selected_date.month}/{selected_date.year}"

    if formatted_date_str in tarikh_classes:
        tarikh_val = formatted_date_str
    else:
        # Map to the closest historical date in the training dataset
        closest_date = min(trained_dates_map.keys(), key=lambda d: abs(d - selected_date))
        tarikh_val = trained_dates_map[closest_date]
        st.caption(f"ℹ️ Selected date not in training set. Mapped to nearest reference date: `{closest_date.strftime('%d/%m/%Y')}`")

    if st.button("🎯 Predict Coordinates"):
        try:
            # Assemble into dataframe
            raw_data = {
                'pembiakan': pembiakan_val,
                'bekas diperiksa': float(bekas_diperiksa_val),
                'Container index': float(container_index_val),
                'Tarikh': tarikh_val
            }
            new_df = pd.DataFrame([raw_data])
            
            # Map translation
            new_df_translated = new_df.rename(columns=translation_dict)
            
            # Encode features
            for col in categorical_cols:
                if col != target_col and col in new_df_translated.columns:
                    le = label_encoders[col]
                    val = str(new_df_translated.loc[0, col])
                    new_df_translated[col] = le.transform([val])
            
            # Align columns
            new_X = new_df_translated[feature_cols]
            
            # Perform prediction and inverse-map location
            predicted_class_encoded = int(round(model.predict(new_X)[0]))
            location_encoder = label_encoders[target_col]
            num_classes = len(location_encoder.classes_)
            predicted_class_encoded = max(0, min(predicted_class_encoded, num_classes - 1))
            predicted_coords = location_encoder.inverse_transform([predicted_class_encoded])[0]
            
            # Show results
            st.success(f"### Predicted Coordinates: `{predicted_coords}`")
            st.metric(label="Predicted Encoded Location Class", value=predicted_class_encoded)
            
            try:
                lat, lon = map(float, predicted_coords.split(','))
                map_df = pd.DataFrame([{'lat': lat, 'lon': lon}])
                st.map(map_df)
            except Exception:
                pass
                
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
