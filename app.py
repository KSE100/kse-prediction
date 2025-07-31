
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from psx import stocks
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Configuration ---
STOCK_TICKER = "UBL"
DATA_FILE = os.path.join("data", "KSE30_raw_data.csv") # Save raw data in the data directory
PROCESSED_DATA_FILE = os.path.join("data", "KSE30_processed_data.csv") # Save processed data in the data directory
PREDICTIONS_FILE = os.path.join("data", "predictions.csv") # Save predictions file in the data directory


# --- Helper Functions ---

@st.cache_data(ttl=timedelta(hours=12)) # Cache raw data fetching
def fetch_raw_data(ticker, data_file):
    """Fetches raw historical data and saves it."""
    # st.write(f"Attempting to fetch raw data for {ticker}...") # Suppressed
    try:
        two_years_ago = date.today() - timedelta(days=730) # Approx 2 years
        raw_data = stocks(ticker, start=two_years_ago, end=date.today())

        if raw_data.empty:
            st.error(f"Could not fetch raw historical data for {ticker}.")
            return pd.DataFrame()

        # st.success(f"Successfully fetched {len(raw_data)} days of raw historical data.") # Suppressed

        # Save raw data
        os.makedirs(os.path.dirname(data_file), exist_ok=True) # Ensure data directory exists
        raw_data.to_csv(data_file)
        # st.write(f"Raw data saved to {data_file}") # Suppressed

        return raw_data

    except Exception as e:
        st.error(f"An error occurred during raw data fetching: {e}")
        return pd.DataFrame()

def process_data(raw_data):
    """Adds technical indicators and features to the raw data."""
    # st.write("Processing raw data...") # Suppressed
    if raw_data.empty:
        st.warning("No raw data to process.")
        return pd.DataFrame(), pd.DataFrame() # Return two empty dataframes

    processed_data = raw_data.copy()

    try:
        # Calculate Technical Indicators (RSI and Moving Averages)
        processed_data['MA_20'] = processed_data['Close'].rolling(window=20).mean()
        processed_data['MA_50'] = processed_data['Close'].rolling(window=50).mean()

        delta = processed_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs = avg_gain / avg_loss
            processed_data['RSI'] = 100 - (100 / (1 + rs))

        # st.write("Calculated technical indicators.") # Suppressed

        # Feature Engineering
        processed_data['Close_Lag1'] = processed_data['Close'].shift(1)
        processed_data['Close_Lag2'] = processed_data['Close'].shift(2)
        processed_data['Close_Lag3'] = processed_data['Close'].shift(3)
        processed_data['Volume_MA_5'] = processed_data['Volume'].rolling(window=5).mean()

        # New features for classification
        processed_data['MA_5'] = processed_data['Close'].rolling(window=5).mean()
        processed_data['MA_5_vs_20'] = processed_data['MA_5'] - processed_data['MA_20']
        processed_data['Volatility_14'] = processed_data['Close'].rolling(window=14).std()
        processed_data['Volume_Change_Lag1'] = processed_data['Volume'].pct_change().shift(1)
        processed_data['Price_Change_1d'] = processed_data['Close'].diff(1).shift(1)
        processed_data['Price_Change_3d'] = processed_data['Close'].diff(3).shift(1)


        # st.write("Engineered features.") # Suppressed

        # Create target variable for classification (next day's price direction)
        # This compares current day's close with next day's close
        processed_data['Target'] = processed_data['Close'].shift(-1) # Keep for potential future use or debugging
        price_difference = processed_data['Target'] - processed_data['Close']
        processed_data['Price_Direction'] = np.where(price_difference > 0, 'Up', 'Down')

        # st.write("Created target variable.") # Suppressed

        # Drop rows with NaN values introduced by lagging and rolling windows (including the last row where Target is NaN)
        processed_data.dropna(inplace=True)
        # st.write(f"Processed data shape after dropping NaNs: {processed_data.shape}") # Suppressed

        if processed_data.empty:
             st.warning("Processed data is empty after dropping NaNs.")
             return pd.DataFrame(), pd.DataFrame()

        # Separate latest day's data - this is the last row where Price_Direction could be calculated
        latest_day_data = processed_data.tail(1)
        historical_data_processed = processed_data.iloc[:-1]

        # st.write(f"Separated latest day data. Historical data shape: {historical_data_processed.shape}, Latest day data shape: {latest_day_data.shape}") # Suppressed

        # Optionally save processed data
        # processed_data.to_csv(PROCESSED_DATA_FILE)
        # st.write(f"Processed data saved to {PROCESSED_DATA_FILE}")

        return historical_data_processed, latest_day_data

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return pd.DataFrame(), pd.DataFrame()


def train_classification_model(data):
    """Trains a classification model to predict price direction."""
    # st.write("Training classification model...") # Suppressed
    if data.empty:
        st.warning("No data available to train the model.")
        return None, None, None, None, None, None, None

    # Define features (X) and target (y) for classification
    # Exclude the 'Target' column when training
    features_classification = data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Price_Direction'], axis=1)
    target_classification = data['Price_Direction']

    # Use the last row for prediction, the rest for training
    # Ensure there's enough data for training
    if len(features_classification) < 1: # Need at least one data point for features
         st.warning("Not enough data to train the model.")
         return None, None, None, None, None, None, None


    X_train_classification = features_classification
    y_train_classification = target_classification


    # Initialize and train the Random Forest Classifier model
    model_classification = RandomForestClassifier(random_state=42)
    model_classification.fit(X_train_classification, y_train_classification)

    # st.write("✅ Classification Model training complete.") # Suppressed

    # Evaluate the model on the training data (optional, but good for debugging)
    y_train_pred = model_classification.predict(X_train_classification)
    train_accuracy = accuracy_score(y_train_classification, y_train_pred)


    return model_classification, train_accuracy, model_classification.classes_.tolist()


def make_prediction(model, X_latest, model_classes):
    """Makes a prediction and gets confidence score for the latest data."""
    # st.write("Making prediction...") # Suppressed
    if model is None or X_latest.empty:
        st.warning("Model not trained or no data for prediction.")
        return None, None, None

    # Predict the direction
    predicted_direction = model.predict(X_latest)[0]

    # Get prediction probabilities
    y_prob_classification = model.predict_proba(X_latest)

    # Get confidence score for the predicted class
    # Find the index of the predicted_direction in the model's classes
    predicted_class_index = model_classes.index(predicted_direction)
    confidence_score = y_prob_classification[0, predicted_class_index]

    # st.write(f"🔮 Prediction made: {predicted_direction} with confidence {confidence_score:.2f}") # Suppressed

    return predicted_direction, confidence_score, X_latest.index[0] # Return the date of the prediction


# --- Streamlit App ---

st.title(f"Stock Price Direction Prediction Dashboard ({STOCK_TICKER})")

st.write("Click the button below to fetch the latest data, update the model, and get a prediction for the next trading day's price direction.")

# Fetch raw data initially
raw_data = fetch_raw_data(STOCK_TICKER, DATA_FILE)

# Process data initially and separate historical and latest day
historical_data_processed, latest_day_data = process_data(raw_data)

# Display the "Run Analysis and Get Prediction" button first
if st.button("Run Analysis and Get Prediction"):
    st.write("Running analysis...") # Keep this to indicate processing started

    if not historical_data_processed.empty and not latest_day_data.empty:
        # Combine historical and latest day data for training purposes
        data_for_training = pd.concat([historical_data_processed, latest_day_data])

        # 3. Train Classification Model
        model_classification, train_accuracy, model_classes = train_classification_model(data_for_training)

        if model_classification is not None:
            # Prepare latest day features for prediction
            # Ensure 'Target' and 'Price_Direction' are dropped as they are not features
            X_latest = latest_day_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Price_Direction'], axis=1, errors='ignore')


            # 4. Make Prediction for the next day
            predicted_direction_tomorrow, confidence_score_tomorrow, date_of_latest_data = make_prediction(model_classification, X_latest, model_classes)
            predicted_date = date_of_latest_data + timedelta(days=1) # The date the prediction is for

            # Display prediction for the NEXT trading day in a table
            st.subheader(f"Prediction for {predicted_date.date()}:")
            if predicted_direction_tomorrow is not None:
                prediction_summary = {
                    'Metric': ['Predicted Direction', 'Confidence Score'],
                    'Value': [predicted_direction_tomorrow, f'{confidence_score_tomorrow:.2%}'] # Format confidence as percentage
                }
                prediction_df = pd.DataFrame(prediction_summary)
                st.dataframe(prediction_df.set_index('Metric'))


            # --- Removed "Model Performance on Previous Day's Prediction" section as requested ---
            # st.subheader(f"Model Performance for Previous Day's Prediction (for {evaluation_date}):")
            # ... (removed code)


            # Optionally display training accuracy for the main model
            # st.write(f"Accuracy on training data: {train_accuracy:.2f}") # Need to pass train_accuracy from train_classification_model


        else:
            st.warning("Model could not be trained.")

    else:
        st.error("Insufficient data to run analysis and prediction.")

# Display "Summary for Today" section after the button and prediction results
st.subheader("Summary for Today:")
if not latest_day_data.empty:
    # Calculate LDCP (Last Day Closing Price) - which is the Close_Lag1 in latest_day_data
    ldcp = latest_day_data['Close_Lag1'].iloc[0]
    # Current price is the Close price of the latest day
    current_price = latest_day_data['Close'].iloc[0]
    # Change is Current - LDCP
    change = current_price - ldcp
    volume = latest_day_data['Volume'].iloc[0]

    # Restructure the data for the table with metrics as columns
    latest_day_summary_dict = {
        'LDCP': [ldcp],
        'Open': [latest_day_data['Open'].iloc[0]],
        'High': [latest_day_data['High'].iloc[0]],
        'Low': [latest_day_data['Low'].iloc[0]],
        'Current': [current_price],
        'Change': [change],
        'Volume': [volume]
    }
    latest_day_df = pd.DataFrame(latest_day_summary_dict)

    # Format numerical columns
    for col in ['LDCP', 'Open', 'High', 'Low', 'Current', 'Change']:
        latest_day_df[col] = latest_day_df[col].apply(lambda x: f'{x:.2f}')
    latest_day_df['Volume'] = latest_day_df['Volume'].apply(lambda x: f'{float(x):,.0f}')


    st.dataframe(latest_day_df) # Display the restructured dataframe

else:
    st.write("No data available for the latest day.")


# Display historical stock data and charts as the last section
st.subheader("Historical Stock Data:")
if not historical_data_processed.empty:
    st.line_chart(historical_data_processed['Close'])
    # Display historical data excluding the last day with specified columns, latest entries first
    # Remove time part from the index for display
    historical_data_display = historical_data_processed[['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'MA_50', 'RSI']].copy()
    historical_data_display.index = historical_data_display.index.date # Convert index to date objects for display
    st.dataframe(historical_data_display.astype(float).applymap('{:.2f}'.format).sort_index(ascending=False))
else:
    st.write("No sufficient historical stock data available to display.")


# The historical predictions section is removed as per the new plan.
# st.subheader("All Recorded Predictions and Outcomes:")
# ... (removed code)

