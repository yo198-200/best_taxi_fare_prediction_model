import pandas as pd
import os 

file_path = r"C:\Users\LENOVO\Downloads\taxi_fare.csv"

print(f"Attempting to load file from: {file_path}")
df = pd.read_csv(file_path)
print("\nFile loaded successfully!")
print("First 5 rows of the DataFrame:")
print(df.head())
print(f"\nDataFrame shape: {df.shape}")
print("\n--- Dataset Information ---")
df.info()

    # Check for missing values
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

    # Check for duplicate rows
print("\n--- Duplicate Rows Check ---")
duplicate_rows = df.duplicated().sum()
print(f"There are {duplicate_rows} duplicate rows in the dataset.")    


import numpy as np


# --- Feature Engineering ---

print("Starting feature engineering...")

# 1. Convert 'pickup_datetime' to a datetime object
# The `errors='coerce'` argument will turn any problematic parsing into NaT (Not a Time),
# which can then be handled as a missing value.
df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
print("Converted 'tpep_pickup_datetime' to datetime objects.")
df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
print("Converted 'tpep_dropoff_datetime' to datetime objects.")

# 2. Calculate Trip Distance using the Haversine Formula
def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
    return c * r

df['trip_distance_km'] = haversine_distance(
    df['pickup_longitude'],
    df['pickup_latitude'],
    df['dropoff_longitude'],
    df['dropoff_latitude']
)
print("Created 'trip_distance_km' column.")

# 3. Convert Timezone from UTC to EDT (Eastern Daylight Time)
# New York City primarily uses EDT (UTC-4).
df['pickup_datetime_loc'] = df['pickup_datetime'].dt.tz_localize('America/New_York')
df['pickup_datetime_edt'] = df['pickup_datetime_loc'].dt.tz_convert('America/New_York')
df['dropoff_datetime_loc'] = df['dropoff_datetime'].dt.tz_localize('America/New_York')
df['dropoff_datetime_edt'] = df['dropoff_datetime_loc'].dt.tz_convert('America/New_York')
print("Converted timezone to 'America/New_York' (EDT).")


# 4. Extract Time-Based Features from EDT datetime
df['hour_edt'] = df['pickup_datetime_edt'].dt.hour
df['day_name'] = df['pickup_datetime_edt'].dt.day_name()
df['day_of_week'] = df['pickup_datetime_edt'].dt.dayofweek # Monday=0, Sunday=6
df['month'] = df['pickup_datetime_edt'].dt.month
df['year'] = df['pickup_datetime_edt'].dt.year

print("Extracted hour, day, month, and year from EDT time.")

# 5. Create Indicator Columns (Binary Flags)

# a. is_weekend: 1 if Saturday or Sunday, else 0
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# b. am_pm: 'am' or 'pm'
df['am_pm'] = df['hour_edt'].apply(lambda x: 'pm' if x >= 12 else 'am')

# c. is_night: 1 for trips between 10 PM and 5 AM, else 0
df['is_night'] = df['hour_edt'].apply(lambda x: 1 if (x >= 22 or x < 5) else 0)

print("Created indicator columns: 'is_weekend', 'am_pm', 'is_night'.")

# --- Display Results ---
print("\n--- Feature Engineering Complete! ---")
print("Here's a look at the DataFrame with the new features:\n")

# Display the first 5 rows with key old and new columns to verify
columns_to_show = [
    'pickup_datetime_edt',
    'dropoff_datetime_edt',
    'fare_amount',
    'trip_distance_km',
    'hour_edt',
    'day_name',
    'is_weekend',
    'am_pm',
    'is_night'
]
print(df[columns_to_show].head())

# Display the data types to confirm changes
print("\n--- Updated DataFrame Info ---")
df.info()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


print("--- Starting Exploratory Data Analysis (EDA) ---")


# Calculate Trip Duration in minutes
# Handle potential NaT values from datetime conversion and negative durations
df['trip_duration_minutes'] = (df['dropoff_datetime_edt'] - df['pickup_datetime_edt']).dt.total_seconds() / 60

# Impute NaN durations (e.g., if dropoff_datetime_edt was NaT) with 0 or a more robust method
df['trip_duration_minutes'] = df['trip_duration_minutes'].fillna(0)

# Filter out rows with non-positive trip durations (e.g., 0 or negative)
initial_rows = len(df)
df = df[df['trip_duration_minutes'] > 0].copy() # Use .copy() to avoid SettingWithCopyWarning
print(f"Removed {initial_rows - len(df)} rows with non-positive trip durations.")


# Calculate Fare per Kilometer and Fare per Minute (handle division by zero)
df['fare_per_km'] = np.where(df['trip_distance_km'] > 0, df['fare_amount'] / df['trip_distance_km'], 0)
df['fare_per_minute'] = np.where(df['trip_duration_minutes'] > 0, df['fare_amount'] / df['trip_duration_minutes'], 0)

print("Calculated 'trip_duration_minutes', 'fare_per_km', and 'fare_per_minute'.")

if pd.api.types.is_categorical_dtype(df['passenger_count']):
    df['passenger_count'] = df['passenger_count'].astype(int)

# Convert other relevant columns to categorical type for better plotting and memory usage
df['VendorID'] = df['VendorID'].astype('category')
df['RatecodeID'] = df['RatecodeID'].astype('category')
df['payment_type'] = df['payment_type'].astype('category')
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].astype('category')

print("Converted 'VendorID', 'RatecodeID', 'payment_type', 'store_and_fwd_flag' to categorical type.")
print("\nUpdated DataFrame Info (post re-processing):")
df.info() 


plt.style.use('seaborn-v0_8-darkgrid') 
sns.set_palette('viridis') 

# --- 2. Univariate Analysis ---
print("\n--- 2. Univariate Analysis ---")

# Fare Amount Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['fare_amount'], bins=50, kde=True)
plt.title('Distribution of Fare Amount', fontsize=16)
plt.xlabel('Fare Amount ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xlim(df['fare_amount'].quantile(0.001), df['fare_amount'].quantile(0.99))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Trip Distance Distribution (Kilometers)
plt.figure(figsize=(10, 6))
sns.histplot(df['trip_distance_km'], bins=50, kde=True)
plt.title('Distribution of Trip Distance (km)', fontsize=16)
plt.xlabel('Trip Distance (km)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xlim(df['trip_distance_km'].quantile(0.001), df['trip_distance_km'].quantile(0.99))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Trip Duration Distribution (Minutes)
plt.figure(figsize=(10, 6))
sns.histplot(df['trip_duration_minutes'], bins=50, kde=True)
plt.title('Distribution of Trip Duration (minutes)', fontsize=16)
plt.xlabel('Trip Duration (minutes)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xlim(df['trip_duration_minutes'].quantile(0.001), df['trip_duration_minutes'].quantile(0.99))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Passenger Count Distribution - FIXED
plt.figure(figsize=(8, 5))
# Filter passenger_count before passing to seaborn
df_filtered_passengers = df[df['passenger_count'] > 0].copy()
sns.countplot(x='passenger_count', data=df_filtered_passengers) # Use the filtered DataFrame
plt.title('Distribution of Passenger Count (Excluding 0)', fontsize=16)
plt.xlabel('Passenger Count', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Pickup Hour Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='hour_edt', data=df)
plt.title('Distribution of Trips by Pickup Hour (EDT)', fontsize=16)
plt.xlabel('Pickup Hour of Day (EDT)', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.xticks(np.arange(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Pickup Day of Week Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='day_name', data=df, order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
plt.title('Distribution of Trips by Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- 3. Bivariate Analysis ---
print("\n--- 3. Bivariate Analysis ---")

# Fare vs. Distance
plt.figure(figsize=(12, 7))
sample_df = df.sample(n=min(50000, len(df)), random_state=42)
sns.scatterplot(x='trip_distance_km', y='fare_amount', data=sample_df,
                alpha=0.3, s=10)
plt.title('Fare Amount vs. Trip Distance (km)', fontsize=16)
plt.xlabel('Trip Distance (km)', fontsize=12)
plt.ylabel('Fare Amount ($)', fontsize=12)
plt.xlim(df['trip_distance_km'].quantile(0.001), df['trip_distance_km'].quantile(0.99))
plt.ylim(df['fare_amount'].quantile(0.001), df['fare_amount'].quantile(0.99))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Fare vs. Passenger Count - FIXED
plt.figure(figsize=(10, 6))
sns.boxplot(x='passenger_count', y='fare_amount', data=df_filtered_passengers) # Use the filtered DataFrame
plt.title('Fare Amount vs. Passenger Count (Excluding 0 Passengers)', fontsize=16)
plt.xlabel('Passenger Count', fontsize=12)
plt.ylabel('Fare Amount ($)', fontsize=12)
plt.ylim(df['fare_amount'].quantile(0.01), df['fare_amount'].quantile(0.99))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Fare vs. Payment Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='payment_type', y='fare_amount', data=df)
plt.title('Fare Amount vs. Payment Type', fontsize=16)
plt.xlabel('Payment Type', fontsize=12)
plt.ylabel('Fare Amount ($)', fontsize=12)
plt.ylim(df['fare_amount'].quantile(0.01), df['fare_amount'].quantile(0.99))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# --- 4. Outlier Detection (Visual & Statistical) ---
print("\n--- 4. Outlier Detection ---")

def detect_outliers_iqr(df, column):
    if not pd.api.types.is_numeric_dtype(df[column]) or df[column].empty:
        print(f"Skipping outlier detection for non-numeric or empty column: {column}")
        return pd.DataFrame()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"\nColumn: {column}")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
    print(f"  Number of outliers: {len(outliers)}")
    print(f"  Percentage of outliers: {len(outliers)/len(df)*100:.2f}%")
    return outliers

outliers_fare = detect_outliers_iqr(df, 'fare_amount')
outliers_distance = detect_outliers_iqr(df, 'trip_distance_km')
outliers_duration = detect_outliers_iqr(df, 'trip_duration_minutes')
outliers_extra = detect_outliers_iqr(df, 'extra')
outliers_tip = detect_outliers_iqr(df, 'tip_amount')
outliers_tolls = detect_outliers_iqr(df, 'tolls_amount')
outliers_total = detect_outliers_iqr(df, 'total_amount')


plt.figure(figsize=(20, 8))

plt.subplot(2, 3, 1)
sns.boxplot(y=df['fare_amount'])
plt.title('Box Plot of Fare Amount', fontsize=14)
plt.ylabel('Fare Amount ($)', fontsize=12)

plt.subplot(2, 3, 2)
sns.boxplot(y=df['trip_distance_km'])
plt.title('Box Plot of Trip Distance (km)', fontsize=14)
plt.ylabel('Trip Distance (km)', fontsize=12)

plt.subplot(2, 3, 3)
sns.boxplot(y=df['trip_duration_minutes'])
plt.title('Box Plot of Trip Duration (minutes)', fontsize=14)
plt.ylabel('Trip Duration (minutes)', fontsize=12)

plt.subplot(2, 3, 4)
sns.boxplot(y=df['tip_amount'])
plt.title('Box Plot of Tip Amount', fontsize=14)
plt.ylabel('Tip Amount ($)', fontsize=12)

plt.subplot(2, 3, 5)
sns.boxplot(y=df['tolls_amount'])
plt.title('Box Plot of Tolls Amount', fontsize=14)
plt.ylabel('Tolls Amount ($)', fontsize=12)

plt.subplot(2, 3, 6)
sns.boxplot(y=df['total_amount'])
plt.title('Box Plot of Total Amount', fontsize=14)
plt.ylabel('Total Amount ($)', fontsize=12)

plt.tight_layout()
plt.show()


# Analyzing fare variations across different times of the day (Mean fare)
plt.figure(figsize=(12, 7))
sns.lineplot(x='hour_edt', y='fare_amount', data=df.groupby('hour_edt')['fare_amount'].mean().reset_index(),
             marker='o')
plt.title('Average Fare Amount by Pickup Hour (EDT)', fontsize=16)
plt.xlabel('Pickup Hour (EDT)', fontsize=12)
plt.ylabel('Average Fare Amount ($)', fontsize=12)
plt.xticks(np.arange(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Analyzing fare variations weekdays vs. weekends
plt.figure(figsize=(8, 6))
sns.boxplot(x='is_weekend', y='fare_amount', data=df)
plt.title('Fare Amount Variation: Weekday (0) vs. Weekend (1)', fontsize=16)
plt.xlabel('Is Weekend', fontsize=12)
plt.ylabel('Fare Amount ($)', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Weekday', 'Weekend'])
plt.ylim(df['fare_amount'].quantile(0.01), df['fare_amount'].quantile(0.99))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Analyzing fare variations across months (limited by your data, will show only March)
plt.figure(figsize=(10, 6))
sns.boxplot(x='month', y='fare_amount', data=df)
plt.title('Fare Amount Variation by Month (Limited to March 2016 Data)', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Fare Amount ($)', fontsize=12)
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.ylim(df['fare_amount'].quantile(0.01), df['fare_amount'].quantile(0.99))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
print("Note: 'Fare Amount Variation by Month' plot likely only shows data for March 2016 as per your dataset's range.")


# Visualizing trip counts by pickup hour and pickup day to identify peak demand periods.
plt.figure(figsize=(14, 8))
sns.countplot(data=df, x='hour_edt', hue='day_name',
              order=sorted(df['hour_edt'].unique()))
plt.title('Trip Counts by Pickup Hour and Day of Week', fontsize=16)
plt.xlabel('Pickup Hour (EDT)', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.xticks(np.arange(0, 24))
plt.legend(title='Day of Week', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print("Note: 'Trip Counts by Pickup Hour and Day of Week' plot might only show data for certain weekdays based on your dataset's 'day_of_week' range.")


# Exploring how fare per km and fare per minute behave across different time periods or trip lengths.

# Average Fare per Km by Hour of Day
plt.figure(figsize=(12, 7))
sns.lineplot(x='hour_edt', y='fare_per_km', data=df.groupby('hour_edt')['fare_per_km'].mean().reset_index(),
             marker='o')
plt.title('Average Fare Per Km by Pickup Hour (EDT)', fontsize=16)
plt.xlabel('Pickup Hour (EDT)', fontsize=12)
plt.ylabel('Average Fare Per Km ($/km)', fontsize=12)
plt.xticks(np.arange(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Average Fare per Minute by Hour of Day
plt.figure(figsize=(12, 7))
sns.lineplot(x='hour_edt', y='fare_per_minute', data=df.groupby('hour_edt')['fare_per_minute'].mean().reset_index(),
             marker='o')
plt.title('Average Fare Per Minute by Pickup Hour (EDT)', fontsize=16)
plt.xlabel('Pickup Hour (EDT)', fontsize=12)
plt.ylabel('Average Fare Per Minute ($/min)', fontsize=12)
plt.xticks(np.arange(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Investigating the impact of night rides and weekend trips on fare amounts.
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='is_night', y='fare_amount', data=df)
plt.title('Fare Amount: Day/Evening (0) vs. Night Ride (1)', fontsize=16)
plt.xlabel('Is Night Ride', fontsize=12)
plt.ylabel('Fare Amount ($)', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Day/Evening (5 AM - 10 PM)', 'Night (10 PM - 5 AM)'])
plt.ylim(df['fare_amount'].quantile(0.01), df['fare_amount'].quantile(0.99))
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
sns.boxplot(x='is_weekend', y='fare_amount', data=df)
plt.title('Fare Amount: Weekday (0) vs. Weekend (1)', fontsize=16)
plt.xlabel('Is Weekend', fontsize=12)
plt.ylabel('Fare Amount ($)', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Weekday', 'Weekend'])
plt.ylim(df['fare_amount'].quantile(0.01), df['fare_amount'].quantile(0.99))
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# --- Correlation Heatmap (for numerical features) ---
print("\n--- Correlation Heatmap (Numerical Features) ---")
# Select only numerical columns for correlation matrix
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# Exclude identifier or binary indicator columns if they clutter the heatmap too much
cols_to_correlate = [col for col in numerical_cols if col not in ['VendorID', 'year', 'is_weekend', 'is_night', 'day_of_week', 'month', 'hour_edt']]

correlation_matrix = df[cols_to_correlate].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.show()


print("\n--- EDA Complete ---")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew 
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import joblib

warnings.filterwarnings('ignore') 

df_transformed = df.copy()

print("--- Starting Data Transformation and Feature Selection ---")

# --- Data Transformation ---

# 1. Handle Outliers using IQR-based Capping
print("\n--- 1. Handling Outliers (IQR-based Capping) ---")

#  columns for outlier capping (numerical columns identified as having outliers)
outlier_cols = [
    'fare_amount', 'trip_distance_km', 'trip_duration_minutes',
    'extra', 'tip_amount', 'tolls_amount', 'total_amount',
    'fare_per_km', 'fare_per_minute' #  include the derived fare metrics
]



for col in outlier_cols:
    if col in df_transformed.columns:
        Q1 = df_transformed[col].quantile(0.25)
        Q3 = df_transformed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Capping values
        df_transformed[col] = np.where(df_transformed[col] < lower_bound, lower_bound, df_transformed[col])
        df_transformed[col] = np.where(df_transformed[col] > upper_bound, upper_bound, df_transformed[col])
        print(f"Capped outliers for column: {col}. Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
    else:
        print(f"Column '{col}' not found for outlier capping.")



# 2. Fix Skewness in Continuous Variables
print("\n--- 2. Fixing Skewness in Continuous Variables ---")

# Identify continuous numerical columns that might be skewed (excluding IDs, binary flags, coords)
skew_check_cols = [
    'fare_amount', 'trip_distance_km', 'trip_duration_minutes',
    'extra', 'tip_amount', 'tolls_amount', 'total_amount',
    'fare_per_km', 'fare_per_minute'
]

transformed_skew_cols = []

for col in skew_check_cols:
    if col in df_transformed.columns:
        original_skew = skew(df_transformed[col].dropna())
        print(f"Original skewness for '{col}': {original_skew:.2f}")

        
        if original_skew > 0.75: # A common threshold for significant right-skew
            df_transformed[col + '_log'] = np.log1p(df_transformed[col])
            transformed_skew = skew(df_transformed[col + '_log'].dropna())
            print(f"  Applied log1p transformation to '{col}'. New skewness: {transformed_skew:.2f}")
            transformed_skew_cols.append(col + '_log')
        elif original_skew < -0.75:
            
            print(f"  '{col}' is left-skewed, but no specific transformation applied in this general example.")
        else:
            print(f"  Skewness for '{col}' is within acceptable range, no transformation applied.")
    else:
        print(f"Column '{col}' not found for skewness check.")

# Updating the list of numerical features to include the new log-transformed ones for correlation/selection
numerical_features_for_model = [col for col in df_transformed.select_dtypes(include=np.number).columns if col not in skew_check_cols]
numerical_features_for_model.extend(transformed_skew_cols)

print(f"\nFeatures considered numerical for modeling after skewness: {numerical_features_for_model}")


# 3. Encode Categorical Variables
print("\n--- 3. Encoding Categorical Variables ---")

categorical_cols = [
    'VendorID', 'RatecodeID', 'payment_type', 'store_and_fwd_flag',
    'passenger_count',
    'day_name', 'am_pm', 'is_weekend', 'is_night', 'month', 'year', 'day_of_week', 'hour_edt' # Features derived from datetime
]

# Filter out columns that might not exist or are already handled
categorical_cols = [col for col in categorical_cols if col in df_transformed.columns]


# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the categorical columns
encoded_features = one_hot_encoder.fit_transform(df_transformed[categorical_cols])

# Create a DataFrame from encoded features with proper column names
encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_transformed.index)

# Drop original categorical columns and concatenate the encoded ones
df_processed = df_transformed.drop(columns=categorical_cols)
df_processed = pd.concat([df_processed, df_encoded], axis=1)

print(f"Encoded {len(categorical_cols)} categorical columns. New DataFrame shape: {df_processed.shape}")
print("Sample of encoded columns:")
print(df_encoded.head())

# Drop original datetime objects and original versions of skewed columns
columns_to_drop_after_processing = [
    'tpep_pickup_datetime', 'tpep_dropoff_datetime',
    'pickup_datetime', 'dropoff_datetime',
    'pickup_datetime_loc', 'dropoff_datetime_loc',
    'pickup_datetime_edt', 'dropoff_datetime_edt'
]
# Remove the original skew_check_cols if their _log transformed versions are used
columns_to_drop_after_processing.extend([col for col in skew_check_cols if col + '_log' in df_processed.columns])

df_final = df_processed.drop(columns=columns_to_drop_after_processing, errors='ignore')

print(f"\nFinal DataFrame shape after transformation: {df_final.shape}")
print(df_final.head())



print("\n--- Starting Feature Selection ---")

# Define target variable
TARGET_COL = 'fare_amount_log' # Using the log-transformed fare amount as the target
if TARGET_COL not in df_final.columns:
    print(f"Error: Target column '{TARGET_COL}' not found. Please check transformations.")
    # Fallback to original fare_amount if log not found or for testing
    if 'fare_amount' in df_final.columns:
        TARGET_COL = 'fare_amount'
        print(f"Using '{TARGET_COL}' as target instead.")
    else:
        raise ValueError("Neither 'fare_amount_log' nor 'fare_amount' found in DataFrame.")

X = df_final.drop(columns=[col for col in ['fare_amount', 'fare_amount_log', 'total_amount'] if col in df_final.columns], errors='ignore') # Exclude original and log fare, total amount (as it's highly correlated)
y = df_final[TARGET_COL]

print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

# Ensure all feature columns are numeric after encoding
X = X.select_dtypes(include=np.number)

# Handle potential NaN or inf values in X or y that might arise from transformations
X.replace([np.inf, -np.inf], np.nan, inplace=True)
y.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN in X or y for feature selection, typically handled in model pipeline
# For selection, we'll keep it simple by dropping them here
initial_rows_fs = X.shape[0]
X_fs_temp = pd.concat([X, y], axis=1).dropna()
X = X_fs_temp.drop(columns=TARGET_COL)
y = X_fs_temp[TARGET_COL]
print(f"Removed {initial_rows_fs - X.shape[0]} rows with NaN/inf for feature selection.")


# Split data for model-based feature importance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- 4.1. Feature Selection: Correlation Analysis (for numerical features) ---")
# Re-evaluate correlation after transformations and encoding.
# Using the updated numerical_features_for_model list.
# Note: For one-hot encoded features, pairwise correlation isn't as informative as with continuous data.
# Focus on correlation between original numerical features and target.

# Concatenate target back for correlation matrix calculation
df_for_corr = X_train.copy()
df_for_corr[TARGET_COL] = y_train

# Calculate correlation with the target variable
correlations_with_target = df_for_corr.corr()[TARGET_COL].sort_values(ascending=False)
print("Top 20 Features Correlated with Target ('fare_amount_log' or 'fare_amount'):")
print(correlations_with_target.head(20))


original_numerical_cols = [col for col in X_train.columns if col in numerical_features_for_model and '_log' not in col]

all_numerical_for_corr_matrix = [col for col in X_train.columns if col in numerical_features_for_model or col.endswith('_log')]


if all_numerical_for_corr_matrix:
    corr_matrix_internal = df_for_corr[all_numerical_for_corr_matrix].corr().abs()
    #  upper triangle of correlation matrix
    upper_tri = corr_matrix_internal.where(np.triu(np.ones(corr_matrix_internal.shape), k=1).astype(bool))
    #  features with correlation greater than 0.9 (threshold can be adjusted)
    to_drop_highly_correlated = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    print(f"\nFeatures with high inter-correlation (above 0.9) to consider for removal: {to_drop_highly_correlated}")
else:
    print("\nNo numerical features identified for internal correlation check.")


print("\n--- 4.2. Feature Selection: Random Forest Feature Importance ---")

# Train a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf_model.fit(X_train, y_train)

# feature importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

print("Top 30 Features by Random Forest Importance:")
print(feature_importances_sorted.head(30))

# Visualize feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances_sorted.head(30), y=feature_importances_sorted.head(30).index, palette='viridis')
plt.title('Top 30 Feature Importances from Random Forest Regressor', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()

# top K features
#  features with importance > 0.01 (threshold can be tuned)
selected_features_rf = feature_importances_sorted[feature_importances_sorted > 0.005].index.tolist()
print(f"\nNumber of features selected by Random Forest (importance > 0.005): {len(selected_features_rf)}")
print(f"Selected features: {selected_features_rf}")



print("\n--- Data Transformation and Feature Selection Complete! ---")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

#  Prepare Data for Modeling ---
print("--- Preparing Data for Model Building ---")



TARGET_COL = 'fare_amount_log'


columns_to_drop_from_X = [
    'fare_amount', 
    'fare_amount_log', 
    'total_amount', 
    'total_amount_log', #
    'tpep_pickup_datetime', 'tpep_dropoff_datetime', 
    'pickup_datetime', 'dropoff_datetime', 
    'pickup_datetime_loc', 'dropoff_datetime_loc',
    'pickup_datetime_edt', 'dropoff_datetime_edt' 
]

numerical_original_cols_transformed = [
    'trip_distance_km', 'trip_duration_minutes', 'tip_amount', 'fare_per_minute' 
]
for col in numerical_original_cols_transformed:
    if col + '_log' in df_final.columns and col in df_final.columns:
        columns_to_drop_from_X.append(col)
       

# Filter out columns that do not exist in df_final to prevent errors
columns_to_drop_from_X = [col for col in columns_to_drop_from_X if col in df_final.columns]

X = df_final.drop(columns=columns_to_drop_from_X, errors='ignore')
y = df_final[TARGET_COL]

# Ensure all X columns are numeric (important after OneHotEncoding)
X = X.select_dtypes(include=np.number)


X.replace([np.inf, -np.inf], np.nan, inplace=True)
y.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN in X or y
initial_rows = X.shape[0]
combined_df = pd.concat([X, y], axis=1).dropna()
X = combined_df.drop(columns=TARGET_COL)
y = combined_df[TARGET_COL]
print(f"Removed {initial_rows - X.shape[0]} rows with NaN/Inf values for modeling.")


print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
print(f"Target variable being predicted: '{TARGET_COL}'")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


#  Build and Compare Multiple Regression Models ---
print("\n--- 1. Building and Comparing Regression Models ---")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=42),
    "Lasso Regression": Lasso(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42, n_estimators=100)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    #  linear models, scaling features can be beneficial
    if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # Tree-based models are less sensitive to feature scaling
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {"R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae}

    print(f"  {name} - R2: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

#  all results in a DataFrame
results_df = pd.DataFrame.from_dict(results, orient='index')
print("\n--- Model Comparison Results ---")
print(results_df.sort_values(by="R2", ascending=False))

#  best performing model based on R2 (and visually confirm with others)
best_model_name = results_df['R2'].idxmax()
print(f"\nInitial Best Performing Model: {best_model_name}")


# --- 2. Hyperparameter Tuning (using RandomizedSearchCV for efficiency) ---
print(f"\n--- 2. Hyperparameter Tuning for {best_model_name} ---")

# Define model and parameter grid based on the best initial model
tuned_model = models[best_model_name]

param_dist = {}
if best_model_name == "Random Forest Regressor":
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', 0.8], # 'auto' is deprecated, 'sqrt' is common
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == "Gradient Boosting Regressor":
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 8],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == "Ridge Regression":
    param_dist = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    }
elif best_model_name == "Lasso Regression":
    param_dist = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0] # Lasso alpha is sensitive, start small
    }
# Linear Regression usually doesn't have hyperparameters to tune in this context

if param_dist:
    # Use RandomizedSearchCV for faster tuning, especially with larger datasets/grids
    # n_iter controls how many random combinations are tried
    random_search = RandomizedSearchCV(estimator=tuned_model, param_distributions=param_dist,
                                       n_iter=20, cv=3, verbose=1, random_state=42, n_jobs=-1, scoring='r2')

    #  linear models, scale X_train before tuning
    if best_model_name in ["Ridge Regression", "Lasso Regression"]:
        scaler = StandardScaler()
        X_train_scaled_tuning = scaler.fit_transform(X_train)
        random_search.fit(X_train_scaled_tuning, y_train)
    else:
        random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_score = random_search.best_score_
    final_best_model = random_search.best_estimator_

    print(f"\nBest hyperparameters for {best_model_name}: {best_params}")
    print(f"Best cross-validation R2 score: {best_score:.4f}")

    
    print(f"\nEvaluating Tuned {best_model_name} on Test Set...")
    if best_model_name in ["Ridge Regression", "Lasso Regression"]:
        X_test_scaled_tuning = scaler.transform(X_test)
        y_pred_tuned = final_best_model.predict(X_test_scaled_tuning)
    else:
        y_pred_tuned = final_best_model.predict(X_test)

    r2_tuned = r2_score(y_test, y_pred_tuned)
    mse_tuned = mean_squared_error(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mse_tuned)
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

    print(f"  Tuned {best_model_name} - R2: {r2_tuned:.4f}, MSE: {mse_tuned:.4f}, RMSE: {rmse_tuned:.4f}, MAE: {mae_tuned:.4f}")
else:
    print(f"No hyperparameters to tune for {best_model_name} or tuning not configured.")
    final_best_model = tuned_model # If no tuning, the initial model is the final best
    # Re-evaluate the initial best model on test set without tuning, to ensure we have its test metrics
    if best_model_name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]: # Check if it's a linear model requiring scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        final_best_model.fit(X_train_scaled, y_train) # Retrain with original data
        y_pred_tuned = final_best_model.predict(X_test_scaled)
    else:
        final_best_model.fit(X_train, y_train) # Retrain with original data
        y_pred_tuned = final_best_model.predict(X_test)

    r2_tuned = r2_score(y_test, y_pred_tuned)
    mse_tuned = mean_squared_error(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mse_tuned)
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)


# --- 3. Finalize Best Model and Save ---
print("\n--- 3. Finalizing Best Model and Saving ---")


best_model_for_saving = final_best_model

print(f"Chosen best model for saving: {best_model_name} (Tuned)")


model_filename = f'best_taxi_fare_prediction_model_{best_model_name.replace(" ", "_").lower()}.pkl'
joblib.dump(best_model_for_saving, model_filename)
print(f"Best model saved as '{model_filename}'")

#  Save the fitted OneHotEncoder
one_hot_encoder_filename = 'one_hot_encoder.pkl'
joblib.dump(one_hot_encoder, one_hot_encoder_filename)
print(f"Fitted OneHotEncoder saved as '{one_hot_encoder_filename}'")

#  Save the list of feature names to maintain column order for prediction
model_feature_names_filename = 'model_feature_names.pkl'
joblib.dump(X_train.columns.tolist(), model_feature_names_filename)
print(f"Model feature names saved as '{model_feature_names_filename}'")

print("\n--- Model Building Complete! ---")




