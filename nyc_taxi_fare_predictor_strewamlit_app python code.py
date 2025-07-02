

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import pytz # For timezone conversion
from sklearn.preprocessing import OneHotEncoder

# Set Streamlit page configuration
st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="wide")

# --- Configuration and Model/Preprocessor Loading ---
try:
    model = joblib.load('best_taxi_fare_prediction_model_random_forest_regressor.pkl')
    ohe = joblib.load('one_hot_encoder.pkl') # Load the *fitted* OneHotEncoder
    model_feature_names = joblib.load('model_feature_names.pkl') # Load the exact feature names and order
    st.success("Model and preprocessors loaded successfully!")
except FileNotFoundError as e:
    st.error(f"Error loading required files: {e}. Ensure 'best_taxi_fare_prediction_model_random_forest_regressor.pkl', 'one_hot_encoder.pkl', and 'model_feature_names.pkl' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error during loading: {e}")
    st.stop()

# Defining the columns that were log-transformed during training

LOG_TRANSFORMED_FEATURES = [
    'trip_distance_km',
    'trip_duration_minutes',
    'extra',
    'tip_amount',
    'tolls_amount',
]

# Defining original categorical columns for the OHE transformation

ORIGINAL_CATEGORICAL_COLS = [
    'VendorID', 'RatecodeID', 'payment_type', 'store_and_fwd_flag',
    'passenger_count',
    'day_name', 'am_pm', 'is_weekend', 'is_night', 'month', 'year', 'day_of_week', 'hour_edt'
]

# Defining capping bounds - these should ideally be saved from training data.

CAPPING_BOUNDS = {
    'fare_amount': {'lower': 0, 'upper': 60}, #  target
    'trip_distance_km': {'lower': 0, 'upper': 20},
    'trip_duration_minutes': {'lower': 0, 'upper': 60},
    'extra': {'lower': 0, 'upper': 5},
    'tip_amount': {'lower': 0, 'upper': 10},
    'tolls_amount': {'lower': 0, 'upper': 10},
    'total_amount': {'lower': 0, 'upper': 70}, #  target
    'fare_per_km': {'lower': 0, 'upper': 10}, #  target
    'fare_per_minute': {'lower': 0, 'upper': 5}, #  target
    'mta_tax': {'lower': 0, 'upper': 0.5},
    'improvement_surcharge': {'lower': 0, 'upper': 0.3},
    'congestion_surcharge': {'lower': 0, 'upper': 2.75},
    'airport_fee': {'lower': 0, 'upper': 4.5}
}

# --- Feature Engineering Functions ---
def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

def preprocess_input(input_data):
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])

   
    df['VendorID'] = df.get('VendorID', 2) 
    df['RatecodeID'] = df.get('RatecodeID', 1) 
    df['payment_type'] = df.get('payment_type', 1) 
    df['store_and_fwd_flag'] = df.get('store_and_fwd_flag', 'N') 
    df['extra'] = df.get('extra', 0.0) 
    df['mta_tax'] = df.get('mta_tax', 0.5) 
    df['tip_amount'] = df.get('tip_amount', 0.0) 
    df['tolls_amount'] = df.get('tolls_amount', 0.0) 
    df['improvement_surcharge'] = df.get('improvement_surcharge', 0.3) 
    df['congestion_surcharge'] = df.get('congestion_surcharge', 0.0) 
    df['airport_fee'] = df.get('airport_fee', 0.0) 


    df['passenger_count'] = df['passenger_count'].astype(int)

    # Convert datetime columns from string to datetime objects
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    df['pickup_datetime'] = df['tpep_pickup_datetime']
    df['dropoff_datetime'] = df['tpep_dropoff_datetime']

    # Calculate Trip Distance
    df['trip_distance_km'] = haversine_distance(
        df['pickup_longitude'],
        df['pickup_latitude'],
        df['dropoff_longitude'],
        df['dropoff_latitude']
    )

    # Convert Timezone from UTC to EDT (Eastern Daylight Time)
    ny_tz = pytz.timezone('America/New_York')
    # Localize to UTC first, then convert. Assuming input datetimes are naive and represent UTC.
    df['pickup_datetime_edt'] = df['pickup_datetime'].dt.tz_localize(pytz.utc).dt.tz_convert(ny_tz)
    df['dropoff_datetime_edt'] = df['dropoff_datetime'].dt.tz_localize(pytz.utc).dt.tz_convert(ny_tz)

    # Extract Time-Based Features from EDT datetime
    df['hour_edt'] = df['pickup_datetime_edt'].dt.hour
    df['day_name'] = df['pickup_datetime_edt'].dt.day_name()
    df['day_of_week'] = df['pickup_datetime_edt'].dt.dayofweek
    df['month'] = df['pickup_datetime_edt'].dt.month
    df['year'] = df['pickup_datetime_edt'].dt.year

    # Create Indicator Columns
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['am_pm'] = df['hour_edt'].apply(lambda x: 'pm' if x >= 12 else 'am')
    df['is_night'] = df['hour_edt'].apply(lambda x: 1 if (x >= 22 or x < 5) else 0)

    # Calculate Trip Duration in minutes
    df['trip_duration_minutes'] = (df['dropoff_datetime_edt'] - df['pickup_datetime_edt']).dt.total_seconds() / 60
    df['trip_duration_minutes'] = df['trip_duration_minutes'].fillna(0)
    df['trip_duration_minutes'] = df['trip_duration_minutes'].apply(lambda x: max(x, 0.1)) # Ensure duration is not zero or negative

    # Applying Outlier Capping to numerical features
    numerical_features_to_cap = [
        'trip_distance_km', 'trip_duration_minutes', 'extra', 'tip_amount', 'tolls_amount',
        'mta_tax', 'improvement_surcharge', 'congestion_surcharge', 'airport_fee'
    ]
    for col in numerical_features_to_cap:
        if col in df.columns and col in CAPPING_BOUNDS:
            df[col] = np.where(df[col] < CAPPING_BOUNDS[col]['lower'], CAPPING_BOUNDS[col]['lower'], df[col])
            df[col] = np.where(df[col] > CAPPING_BOUNDS[col]['upper'], CAPPING_BOUNDS[col]['upper'], df[col])

    # Applying Log Transformation to identified features (numerical input features)
    for col in LOG_TRANSFORMED_FEATURES:
        if col in df.columns:
            df[col + '_log'] = np.log1p(df[col])
            df = df.drop(columns=[col], errors='ignore') 

 
    for col in ORIGINAL_CATEGORICAL_COLS:
        if col in df.columns and col not in ['VendorID', 'RatecodeID', 'passenger_count', 'payment_type']:
             df[col] = df[col].astype(str)
  
        elif col in df.columns and col in ['VendorID', 'RatecodeID', 'passenger_count', 'payment_type']:
            df[col] = df[col].astype(int) 

    # Performimg One-Hot Encoding
  
    try:
        encoded_features = ohe.transform(df[ORIGINAL_CATEGORICAL_COLS])
        encoded_feature_names = ohe.get_feature_names_out(ORIGINAL_CATEGORICAL_COLS)
        df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    except ValueError as ve:
        st.error(f"Error during One-Hot Encoding: {ve}. This often happens if the input data "
                 f"contains categories not seen during training, or if the columns for OHE "
                 f"do not match the expected ones. Missing columns: {set(ORIGINAL_CATEGORICAL_COLS) - set(df.columns)}")
        st.stop()


    # Drop original categorical columns and concatenate encoded ones
    df_processed = df.drop(columns=ORIGINAL_CATEGORICAL_COLS, errors='ignore')
    df_processed = pd.concat([df_processed, df_encoded], axis=1)

    # Drop original datetime objects and other intermediate columns
    columns_to_drop_final = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'pickup_datetime', 'dropoff_datetime',
        'pickup_datetime_edt', 'dropoff_datetime_edt',
    ]
    df_final = df_processed.drop(columns=columns_to_drop_final, errors='ignore')

    # Reindex to ensure column order and completeness (CRITICAL STEP)
 
    processed_df_reindexed = df_final.reindex(columns=model_feature_names, fill_value=0)

    # Final check for NaNs - fill any remaining NaNs with 0 (or appropriate value)
    processed_df_reindexed.fillna(0, inplace=True)

    return processed_df_reindexed


# --- Streamlit UI ---
st.title("🚕 NYC Taxi Fare Predictor")
st.markdown("Enter trip details to get an estimated fare amount.")

# Create columns for better layout
col1, col2 = st.columns(2) # Reduced to two columns since fewer inputs

with col1:
    st.header("Pickup Details")
    pickup_date = st.date_input("Pickup Date", datetime.now().date())
    pickup_time = st.time_input("Pickup Time", datetime.now().time())
    pickup_latitude = st.number_input("Pickup Latitude", value=40.7580, format="%.4f")
    st.caption("Example: 40.7580 (Times Square)")
    pickup_longitude = st.number_input("Pickup Longitude", value=-73.9855, format="%.4f")
    st.caption("Example: -73.9855 (Times Square)")

with col2:
    st.header("Dropoff Details")
    dropoff_date = st.date_input("Dropoff Date", datetime.now().date())
    dropoff_time = st.time_input("Dropoff Time", (datetime.now() + pd.Timedelta(minutes=15)).time())
    dropoff_latitude = st.number_input("Dropoff Latitude", value=40.7128, format="%.4f")
    st.caption("Example: 40.7128 (Lower Manhattan)")
    dropoff_longitude = st.number_input("Dropoff Longitude", value=-74.0060, format="%.4f")
    st.caption("Example: -74.0060 (Lower Manhattan)")

    st.header("Trip Information")
    passenger_count = st.slider("Passenger Count", min_value=1, max_value=6, value=1)


# Combine date and time for datetime objects
pickup_dt_str = f"{pickup_date} {pickup_time}"
dropoff_dt_str = f"{dropoff_date} {dropoff_time}"


if st.button("Predict Fare", help="Click to get the estimated taxi fare."):
    with st.spinner("Calculating prediction..."):
        #  simplified input_data dictionary
        # Only includes inputs gathered directly from the UI.
        input_data = {
            'tpep_pickup_datetime': pickup_dt_str,
            'tpep_dropoff_datetime': dropoff_dt_str,
            'pickup_longitude': pickup_longitude,
            'pickup_latitude': pickup_latitude,
            'dropoff_longitude': dropoff_longitude,
            'dropoff_latitude': dropoff_latitude,
            'passenger_count': passenger_count,
        }

        # Preprocess the input
        try:
            processed_input_df = preprocess_input(input_data)

            # Final check that the preprocessed dataframe has the exact columns and order
            if not processed_input_df.columns.equals(pd.Index(model_feature_names)):
                st.error("Feature mismatch: Processed input columns do not match model's expected features.")
                st.write("Expected:", model_feature_names)
                st.write("Got:", processed_input_df.columns.tolist())
              
                st.stop() 

            # Make prediction
            predicted_log_fare = model.predict(processed_input_df)[0]

            # Inverse transform if the target was log-transformed
            predicted_fare = np.expm1(predicted_log_fare)

            st.success(f"### Estimated Taxi Fare: **${predicted_fare:.2f}**")
            st.info("Note: This is an estimate based on the provided inputs and the trained model. "
                      "Actual fare may vary.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e) 