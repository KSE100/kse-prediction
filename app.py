
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
DATA_FILE = os.path.join("data", "KSE30_raw_data.csv") # Save raw data in the data directory relative to app.py
PREDICTIONS_FILE = os.path.join("data", "predictions.csv") # Save predictions file in the data directory relative to app.py


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

        # Save raw data - Use an absolute path here based on where the script is expected to run
        # When deployed, Streamlit Sharing places the app in the root, so relative paths like "data/..." work.
        # For Colab execution of this cell, need to consider the current directory.
        # However, the functions are written assuming relative paths when run within the Streamlit app context.
        # Let's keep the relative path logic within functions and rely on Streamlit's handling.
        # Ensuring the directory exists is key. The os.makedirs call above handles absolute creation.
        # The to_csv inside the function will write relative to the current working directory of the Streamlit app.

        # Ensure the directory exists relative to the app's run location (handled by os.makedirs above)
        # Saving with relative path assumes the script is run from the project_dir
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
        return None, None

    # Define features (X) and target (y) for classification
    # Exclude the 'Target' column when training
    features_classification = data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Price_Direction'], axis=1)
    target_classification = data['Price_Direction']

    # Use the last row for prediction, the rest for training
    # Ensure there's enough data for training
    if len(features_classification) < 1: # Need at least one data point for features
         st.warning("Not enough data to train the model.")
         return None, None


    X_train_classification = features_classification
    y_train_classification = target_classification


    # Initialize and train the Random Forest Classifier model
    model_classification = RandomForestClassifier(random_state=42)
    model_classification.fit(X_train_classification, y_train_classification)

    # st.write("✅ Classification Model training complete.") # Suppressed

    # Return the trained model and the classes it learned
    return model_classification, model_classification.classes_.tolist()


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

    return predicted_direction, confidence_score, X_latest.index[0]


def load_predictions():
    """Loads historical predictions from a CSV file."""
    try:
        # Ensure the data directory exists before trying to read
        os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)

        if os.path.exists(PREDICTIONS_FILE):
            # Attempt to read the file with robust error handling
            try:
                # Specify dtypes to avoid parsing issues
                predictions_df = pd.read_csv(PREDICTIONS_FILE, index_col='Date', parse_dates=True, dtype={'Predicted_Direction': str, 'Confidence_Score': float, 'Actual_Outcome': str})
                # Ensure index is datetime just in case
                predictions_df.index = pd.to_datetime(predictions_df.index)
                # Ensure required columns exist, add if missing with appropriate NA types
                if 'Predicted_Direction' not in predictions_df.columns: predictions_df['Predicted_Direction'] = pd.Series(dtype='string')
                if 'Confidence_Score' not in predictions_df.columns: predictions_df['Confidence_Score'] = pd.Series(dtype='float64')
                if 'Actual_Outcome' not in predictions_df.columns: predictions_df['Actual_Outcome'] = pd.Series(dtype='string')


                # st.write(f"Loaded {len(predictions_df)} historical predictions from {PREDICTIONS_FILE}") # Suppressed
                return predictions_df
            except Exception as e:
                st.warning(f"Could not read {PREDICTIONS_FILE} due to error: {e}. Returning empty predictions DataFrame.")
                # Return an empty DataFrame with the expected structure and DatetimeIndex
                return pd.DataFrame(columns=['Predicted_Direction', 'Confidence_Score', 'Actual_Outcome']).astype({'Predicted_Direction': 'string', 'Confidence_Score': 'float64', 'Actual_Outcome': 'string'}).set_index(pd.to_datetime([]).rename('Date'))
        else:
            # st.write("No historical predictions file found. Starting with empty predictions DataFrame.") # Suppressed
            # Return an empty DataFrame with the expected structure and DatetimeIndex
            return pd.DataFrame(columns=['Predicted_Direction', 'Confidence_Score', 'Actual_Outcome']).astype({'Predicted_Direction': 'string', 'Confidence_Score': 'float64', 'Actual_Outcome': 'string'}).set_index(pd.to_datetime([]).rename('Date'))
    except Exception as e:
        st.error(f"An unexpected error occurred while loading predictions: {e}")
        # Return an empty DataFrame even for unexpected errors
        return pd.DataFrame(columns=['Predicted_Direction', 'Confidence_Score', 'Actual_Outcome']).astype({'Predicted_Direction': 'string', 'Confidence_Score': 'float64', 'Actual_Outcome': 'string'}).set_index(pd.to_datetime([]).rename('Date'))


def store_prediction(prediction_date, predicted_direction, confidence_score):
    """Stores the prediction in a CSV file."""
    try:
        # Load existing predictions or create an empty DataFrame
        predictions_df = load_predictions() # Use the robust load function

        # Add the new prediction
        # Check if a prediction for this date already exists
        if prediction_date in predictions_df.index:
             # st.write(f"Prediction for {prediction_date.date()} already exists. Skipping storage of new prediction.") # Suppressed
             pass # Suppress message
        else:
            new_prediction = pd.DataFrame([{
                'Predicted_Direction': predicted_direction,
                'Confidence_Score': confidence_score,
                'Actual_Outcome': pd.NA # Actual outcome is unknown at prediction time
            }], index=[prediction_date])
            # Ensure the new prediction DataFrame also has a DatetimeIndex
            new_prediction.index = pd.to_datetime(new_prediction.index)

            # Ensure columns match before concat, using proper NA types
            new_prediction = new_prediction.reindex(columns=predictions_df.columns).astype({'Predicted_Direction': 'string', 'Confidence_Score': 'float64', 'Actual_Outcome': 'string'})


            predictions_df = pd.concat([predictions_df, new_prediction])
            predictions_df.sort_index(inplace=True)
            # Use error handling for saving
            try:
                predictions_df.to_csv(PREDICTIONS_FILE)
                st.write(f"Prediction for {prediction_date.date()} stored.") # Keep this message as it's an action confirmation
            except Exception as e:
                 st.error(f"Could not save prediction to {PREDICTIONS_FILE}: {e}")


    except Exception as e:
        st.error(f"An error occurred while storing the prediction: {e}")


def update_actual_outcomes(historical_data_processed):
    """Updates the actual outcomes for historical predictions."""
    # st.write("Attempting to update historical prediction outcomes...") # Suppressed
    try:
        # Load existing predictions
        predictions_df = load_predictions()

        if predictions_df.empty:
             # st.write("No historical predictions to update.") # Suppressed
             return predictions_df # Return empty if no predictions loaded

        # Ensure historical_data_processed index and relevant columns are ready for comparison
        historical_data_processed.index = pd.to_datetime(historical_data_processed.index)
        if 'Price_Direction' not in historical_data_processed.columns:
             st.warning("Price_Direction column not found in historical_data_processed. Cannot update outcomes.")
             return predictions_df # Cannot update without Price_Direction


        updated_count = 0
        # Iterate through predictions and find corresponding actual outcomes in historical data
        for index, row in predictions_df.iterrows():
            # Check if Actual_Outcome is NA (using pd.isna for robustness)
            if pd.isna(row['Actual_Outcome']):
                # The actual outcome for a prediction made for date X is the direction on date X
                prediction_date = index # The date the prediction was for

                # Check if the data for the prediction date exists in the historical_data_processed
                # Note: historical_data_processed contains data points where Price_Direction could be calculated.
                if prediction_date in historical_data_processed.index:
                    actual_direction = historical_data_processed.loc[prediction_date, 'Price_Direction']
                    predictions_df.loc[index, 'Actual_Outcome'] = actual_direction
                    updated_count += 1

        if updated_count > 0:
             # Use error handling for saving
            try:
                predictions_df.to_csv(PREDICTIONS_FILE)
                st.write(f"Updated {updated_count} historical prediction outcomes.") # Keep this message as it's an action confirmation
            except Exception as e:
                st.error(f"Could not save updated predictions to {PREDICTIONS_FILE}: {e}")


        return predictions_df # Return updated predictions dataframe

    except Exception as e:
        st.error(f"An error occurred while updating actual outcomes: {e}")
        # Return the predictions_df as it was before the attempt to update
        return predictions_df


# --- Streamlit App ---

st.title(f"Stock Price Direction Prediction Dashboard ({STOCK_TICKER})")

st.write("Click the button below to fetch the latest data, update the model, and get a prediction for the next trading day's price direction.")

# Fetch raw data initially
raw_data = fetch_raw_data(STOCK_TICKER, DATA_FILE)

# Process data initially and separate historical and latest day
historical_data_processed, latest_day_data = process_data(raw_data)

# Update actual outcomes for historical predictions when the app loads
# Pass historical_data_processed to the update function
historical_predictions_df = update_actual_outcomes(historical_data_processed)

# Display the "Run Analysis and Get Prediction" button first
if st.button("Run Analysis and Get Prediction"):
    st.write("Running analysis...") # Keep this to indicate processing started

    if not historical_data_processed.empty and not latest_day_data.empty:
        # 3. Train Classification Model
        model_classification, model_classes = train_classification_model(historical_data_processed)

        if model_classification is not None:
            # Prepare latest day features for prediction
            # Ensure 'Target' and 'Price_Direction' are dropped as they are not features
            X_latest = latest_day_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Price_Direction'], axis=1, errors='ignore')


            # 4. Make Prediction for the next day
            predicted_direction_tomorrow, confidence_score_tomorrow, date_of_latest_data = make_prediction(model_classification, X_latest, model_classes)

            if predicted_direction_tomorrow is not None:
                # The prediction date is the day AFTER the latest data point
                predicted_date = date_of_latest_data + timedelta(days=1)

                # Display prediction for the NEXT trading day in a table
                st.subheader(f"Prediction for {predicted_date.date()}:")
                prediction_summary_dict = {
                    'Predicted Direction': [predicted_direction_tomorrow],
                    'Confidence Score': [f'{confidence_score_tomorrow:.2%}'] # Format confidence as percentage
                }
                prediction_df = pd.DataFrame(prediction_summary_dict)
                st.dataframe(prediction_df)

                # 5. Store the new prediction
                # Store the prediction for the NEXT day
                store_prediction(predicted_date, predicted_direction_tomorrow, confidence_score_tomorrow)


            # Display model performance metrics for the PREVIOUS day in a table
            # Use latest_day_data for actual outcome and predict on its features (X_latest)
            if 'Price_Direction' in latest_day_data.columns and not latest_day_data.empty:
                 # Ensure latest_day_data has a valid date index to display
                 if not latest_day_data.index.empty:
                    latest_data_date = latest_day_data.index[0]
                    st.subheader(f"Model Performance on Previous Day ({latest_data_date.date()}):")
                    latest_actual_direction = latest_day_data['Price_Direction'].iloc[0]

                    # Predict again on the cleaned X_latest data used for prediction
                    # Need to re-clean X_latest just for this performance calculation if make_prediction drops NaNs
                    X_latest_cleaned_for_performance = X_latest.dropna(inplace=False)
                    if not X_latest_cleaned_for_performance.empty:
                         latest_predicted_direction = model_classification.predict(X_latest_cleaned_for_performance)[0]
                         performance_summary = {
                            'Metric': ['Actual Direction', 'Predicted Direction', 'Accuracy'],
                            'Value': [latest_actual_direction, latest_predicted_direction, f'{accuracy_score([latest_actual_direction], [latest_predicted_direction]):.2f}']
                        }
                         performance_df = pd.DataFrame(performance_summary)
                         st.dataframe(performance_df.set_index('Metric'))
                    else:
                         st.write("Could not calculate performance on previous day due to missing data.")


            # Optionally display training accuracy
            # st.write(f"Accuracy on training data: {train_accuracy:.2f}") # Need to pass train_accuracy from train_classification_model


        else:
            st.warning("Model could not be trained.")

    else:
        st.error("Insufficient data to run analysis and prediction.")

# Display "Summary for Today" section after the button and prediction results
st.subheader("Summary for Today:")
if not latest_day_data.empty:
    # Calculate LDCP (Last Day Closing Price) - which is the Close_Lag1 in latest_day_data
    # Check if Close_Lag1 exists and is not NaN
    ldcp = latest_day_data['Close_Lag1'].iloc[0] if 'Close_Lag1' in latest_day_data.columns and not pd.isna(latest_day_data['Close_Lag1'].iloc[0]) else "N/A"
    # Current price is the Close price of the latest day
    current_price = latest_day_data['Close'].iloc[0] if 'Close' in latest_day_data.columns and not pd.isna(latest_day_data['Close'].iloc[0]) else "N/A"
    # Change is Current - LDCP - only calculate if both are valid numbers
    change = current_price - ldcp if isinstance(current_price, (int, float)) and isinstance(ldcp, (int, float)) else "N/A"
    volume = latest_day_data['Volume'].iloc[0] if 'Volume' in latest_day_data.columns and not pd.isna(latest_day_data['Volume'].iloc[0]) else "N/A"
    latest_day_open = latest_day_data['Open'].iloc[0] if 'Open' in latest_day_data.columns and not pd.isna(latest_day_data['Open'].iloc[0]) else "N/A"
    latest_day_high = latest_day_data['High'].iloc[0] if 'High' in latest_day_data.columns and not pd.isna(latest_day_data['High'].iloc[0]) else "N/A"
    latest_day_low = latest_day_data['Low'].iloc[0] if 'Low' in latest_day_data.columns and not pd.isna(latest_day_data['Low'].iloc[0]) else "N/A"


    latest_day_summary_dict = {
        'LDCP': [ldcp],
        'Open': [latest_day_open],
        'High': [latest_day_high],
        'Low': [latest_day_low],
        'Current': [current_price],
        'Change': [change],
        'Volume': [volume]
    }
    latest_day_df = pd.DataFrame(latest_day_summary_dict)

    # Format numerical columns, handling "N/A"
    for col in ['LDCP', 'Open', 'High', 'Low', 'Current', 'Change']:
        latest_day_df[col] = latest_day_df[col].apply(lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x)
    # Format Volume differently if needed, e.g., without decimals or with commas, handling "N/A"
    latest_day_df.loc[latest_day_df.index, 'Volume'] = latest_day_df['Volume'].apply(lambda x: f'{float(x):,.0f}' if isinstance(x, (int, float)) else x)

    st.dataframe(latest_day_df) # Display the restructured dataframe

else:
    st.write("No data available for the latest day.")


# Display historical stock data and charts as the last section
st.subheader("Historical Stock Data:")
# Use the original raw_data for the chart and display
if not raw_data.empty:
    st.line_chart(raw_data['Close'])
    # Display historical data excluding the last day with specified columns, latest entries first
    # Remove time part from the index for display
    if not historical_data_processed.empty: # Use historical_data_processed for the table as it has features
        historical_data_display = historical_data_processed[['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'MA_50', 'RSI']].copy()
        historical_data_display.index = historical_data_display.index.date # Convert index to date objects for display
        # Ensure columns are numeric for formatting, coerce errors to handle potential non-numeric values introduced by processing
        historical_data_display = historical_data_display.apply(pd.to_numeric, errors='coerce')
        st.dataframe(historical_data_display.applymap('{:.2f}'.format).sort_index(ascending=False))
    else:
         st.write("No sufficient historical stock data available for the table after processing.")
else:
     st.write("No raw historical stock data file found.")

# Display historical predictions and outcomes
st.subheader("Historical Predictions and Outcomes:")
# Reload predictions to ensure the latest updates are shown
historical_predictions_df = load_predictions()

if not historical_predictions_df.empty:
    # Calculate accuracy of historical predictions with known outcomes
    evaluated_predictions = historical_predictions_df.dropna(subset=['Actual_Outcome'])
    if not evaluated_predictions.empty:
        # Ensure 'Actual_Outcome' and 'Predicted_Direction' are strings for comparison
        historical_accuracy = accuracy_score(evaluated_predictions['Actual_Outcome'].astype(str), evaluated_predictions['Predicted_Direction'].astype(str))
        st.write(f"Historical Prediction Accuracy (evaluated outcomes): {historical_accuracy:.2%}") # Display as percentage

    # Display the historical predictions table, latest entries first
    st.dataframe(historical_predictions_df.sort_index(ascending=False))
else:
    st.write("No recorded predictions found.")

