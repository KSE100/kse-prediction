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
import calendar # Import calendar for day name

# --- Configuration ---
STOCK_TICKER = "UBL"
# DATA_FILE = os.path.join("data", "KSE30_raw_data.csv") # No longer used for main data storage
# Update paths to be relative within the new app directory structure
PREDICTIONS_FILE = os.path.join("data", "predictions.csv") # Save predictions file in the data directory relative to app.py
LOCAL_HISTORICAL_DATA_FILE = os.path.join("data", "local_historical_data.csv") # File for persistent historical data


# --- Helper Functions ---

# Reduce cache TTL to 1 hour to fetch more recent data more frequently
@st.cache_data(ttl=timedelta(hours=1)) # MODIFIED TTL HERE
def fetch_raw_data_from_psx(ticker, start_date, end_date):
    """Fetches raw historical data from PSX for a given date range."""
    # st.write(f"Attempting to fetch raw data for {ticker} from {start_date} to {end_date} from PSX...") # Suppressed
    try:
        # CORRECTED the psx.stocks() call to use positional arguments for dates
        raw_data = stocks(ticker, start_date, end_date) # CORRECTED CALL HERE

        if raw_data.empty:
            # st.warning(f"Could not fetch raw historical data for {ticker} from {start_date} to {end_date} from PSX.") # Suppressed
            return pd.DataFrame()

        # st.success(f"Successfully fetched {len(raw_data)} days of raw historical data from PSX.") # Suppressed

        return raw_data

    except Exception as e:
        st.error(f"An error occurred during raw data fetching from PSX: {e}")
        return pd.DataFrame()

# New function to load and update local historical data
def load_and_update_historical_data(ticker):
    """Loads local historical data and updates it with new data from PSX."""
    # Paths are relative to the app.py location, so this should create data dir inside /content/my_stock_app
    os.makedirs(os.path.dirname(LOCAL_HISTORICAL_DATA_FILE), exist_ok=True)
    local_data = pd.DataFrame()
    last_local_date = None

    # 1. Load existing local data if available
    if os.path.exists(LOCAL_HISTORICAL_DATA_FILE):
        try:
            # Assume Date is the index
            local_data = pd.read_csv(LOCAL_HISTORICAL_DATA_FILE, index_col='Date', parse_dates=['Date'])
            local_data.index = pd.to_datetime(local_data.index)
            if not local_data.empty:
                last_local_date = local_data.index.max().date()
                # st.write(f"Loaded {len(local_data)} days of historical data from local file. Last date: {last_local_date}") # Suppressed
            else:
                 # st.write("Local historical data file is empty.") # Suppressed
                 pass # Suppress message
        except Exception as e:
            st.warning(f"Could not load local historical data from {LOCAL_HISTORICAL_DATA_FILE}: {e}. Starting fresh.")
            local_data = pd.DataFrame() # Start with empty if loading fails

    # 2. Determine start date for fetching new data from PSX
    if last_local_date:
        # Fetch from the day after the last local date
        fetch_start_date = last_local_date + timedelta(days=1)
        # st.write(f"Fetching new data from PSX starting from: {fetch_start_date}") # Suppressed
    else:
        # If no local data, fetch for the last ~2 years
        fetch_start_date = date.today() - timedelta(days=730)
        # st.write(f"No local data found. Fetching ~2 years of data from PSX starting from: {fetch_start_date}") # Suppressed


    # 3. Fetch new data from PSX up to today
    # Ensure fetch_start_date is not in the future
    fetch_end_date = date.today()
    if fetch_start_date <= fetch_end_date:
         new_data = fetch_raw_data_from_psx(ticker, fetch_start_date, fetch_end_date)

         if not new_data.empty:
            # 4. Append new data to local data
            # Ensure indices are datetime for proper concatenation
            new_data.index = pd.to_datetime(new_data.index)
            local_data = pd.concat([local_data, new_data[~new_data.index.isin(local_data.index)]]) # Avoid duplicate dates
            local_data.sort_index(inplace=True) # Keep data sorted by date
            # 5. Save the updated data back to the local file
            try:
                local_data.to_csv(LOCAL_HISTORICAL_DATA_FILE, index=True, index_label='Date')
                # st.write(f"Updated historical data saved to {LOCAL_HISTORICAL_DATA_FILE}") # Suppressed
            except Exception as e:
                st.error(f"An error occurred while saving updated historical data: {e}")
         else:
              # st.write("No new data fetched from PSX.") # Suppressed
              pass # Suppress message
    else:
         # st.write("Fetch start date is in the future. No new data to fetch.") # Suppressed
         pass # Suppress message


    return local_data # Return the combined local historical data


# @st.cache_data # Consider caching processed data as well if processing is slow
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

        # --- Modified logic to separate latest day data BEFORE dropping NaNs ---
        if processed_data.empty:
             st.warning("Processed data is empty before separating latest day.")
             return pd.DataFrame(), pd.DataFrame()

        # Separate latest day's data (including potential NaNs from lagging/rolling)
        latest_day_data = processed_data.tail(1)
        historical_data_for_training = processed_data.iloc[:-1].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Drop rows with NaN values introduced by lagging and rolling windows from historical data used for training
        historical_data_processed = historical_data_for_training.dropna(inplace=False) # Use inplace=False to return new DataFrame
        # st.write(f"Historical data shape after dropping NaNs: {historical_data_processed.shape}") # Suppressed

        if historical_data_processed.empty:
             st.warning("Historical data is empty after dropping NaNs.")
             # If no historical data, also return empty latest_day_data as model training won't happen
             return pd.DataFrame(), pd.DataFrame()


        # st.write(f"Separated latest day data. Historical data shape: {historical_data_processed.shape}, Latest day data shape: {latest_day_data.shape}") # Suppressed

        # Optionally save processed data
        # processed_data.to_csv(PROCESSED_DATA_FILE)
        # st.write(f"Processed data saved to {PROCESSED_DATA_FILE}")

        return historical_data_processed, latest_day_data

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return pd.DataFrame(), pd.DataFrame()


def train_classification_model(historical_data):
    """Trains a classification model to predict price direction."""
    # st.write("Training classification model...") # Suppressed
    if historical_data.empty:
        st.warning("No data available to train the model.")
        return None, None

    # Define features (X) and target (y) for classification
    # Exclude the 'Target' column when training
    features_classification = historical_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Price_Direction'], axis=1)
    target_classification = historical_data['Price_Direction']

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

    # Handle potential NaNs in X_latest by dropping the row if any feature is NaN
    X_latest_cleaned = X_latest.dropna(inplace=False)

    if X_latest_cleaned.empty:
        st.warning("Latest data contains NaNs in features required for prediction.")
        return None, None, None


    # Predict the direction
    predicted_direction = model.predict(X_latest_cleaned)[0]

    # Get prediction probabilities
    y_prob_classification = model.predict_proba(X_latest_cleaned)

    # Get confidence score for the predicted class
    # Find the index of the predicted_direction in the model's classes
    predicted_class_index = model_classes.index(predicted_direction)
    confidence_score = y_prob_classification[0, predicted_class_index]

    return predicted_direction, confidence_score, X_latest_cleaned.index[0] # Return the date of the predicted data


def load_predictions():
    """Loads historical predictions from a CSV file."""
    # Ensure the data directory exists relative to the app location
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)

    # Define the expected columns for the predictions DataFrame
    expected_columns = ['Predicted_Direction', 'Confidence_Score', 'Actual_Outcome']
    index_column_name = 'Date'

    # Define an empty DataFrame with the expected structure and index name
    empty_predictions_df = pd.DataFrame(columns=expected_columns).astype({
        'Predicted_Direction': 'string',
        'Confidence_Score': 'float64',
        'Actual_Outcome': 'string'
    }).set_index(pd.to_datetime([]).rename(index_column_name))


    if os.path.exists(PREDICTIONS_FILE):
        try:
            # Try reading the file assuming the 'Date' column exists and is the index
            predictions_df = pd.read_csv(PREDICTIONS_FILE, index_col=index_column_name, parse_dates=[index_column_name])

            # Ensure index is datetime
            predictions_df.index = pd.to_datetime(predictions_df.index)

            # Ensure required columns exist, add if missing with appropriate NA types
            for col in expected_columns:
                if col not in predictions_df.columns:
                    predictions_df[col] = pd.Series(dtype='string') # Use string dtype for object-like columns

            # Ensure dtypes are correct after loading
            predictions_df = predictions_df.astype({
                'Predicted_Direction': 'string',
                'Confidence_Score': 'float64',
                'Actual_Outcome': 'string'
            })

            # Explicitly handle potential empty strings or whitespace as NA before dropping
            for col in ['Predicted_Direction', 'Actual_Outcome']:
                if col in predictions_df.columns: # Add check if column exists
                    predictions_df.loc[:, col] = predictions_df[col].replace(r'^\s*$', pd.NA, regex=True)

            # Drop rows where Predicted_Direction is NA
            predictions_df.dropna(subset=['Predicted_Direction'], inplace=True)


            # Check if the loaded DataFrame is empty (e.g., file was created but empty)
            if predictions_df.empty:
                # st.write(f"Loaded empty predictions file from {PREDICTIONS_FILE}. Starting with empty DataFrame.") # Suppressed
                return empty_predictions_df # Return the correctly structured empty DataFrame
            else:
                 # st.write(f"Loaded {len(predictions_df)} historical predictions from {PREDICTIONS_FILE}\") # Suppressed
                 return predictions_df

        except Exception as e:
            # If reading with index_col fails (e.g., 'Date' column missing or file malformed)
            st.warning(f"Could not read {PREDICTIONS_FILE} correctly due to error: {e}. Attempting to create a fresh file.")
            # Return the correctly structured empty DataFrame and attempt to save it
            try:
                empty_predictions_df.to_csv(PREDICTIONS_FILE, index=True, index_label=index_column_name)
                # st.write(f"Created a new empty predictions file at {PREDICTIONS_FILE}") # Suppressed
            except Exception as save_e:
                 st.error(f"Could not save a new empty predictions file to {PREDICTIONS_FILE}: {save_e}")

            return empty_predictions_df # Always return the correctly structured empty DataFrame on error

    else:
        # If the file does not exist
        # st.write("No historical predictions file found. Creating a new one.") # Suppressed
        # Create an empty DataFrame with the expected structure and DatetimeIndex
        try:
            empty_predictions_df.to_csv(PREDICTIONS_FILE, index=True, index_label=index_column_name)
            # st.write(f"Created a new empty predictions file at {PREDICTIONS_FILE}\") # Suppressed
        except Exception as save_e:
             st.error(f"Could not save a new empty predictions file to {PREDICTIONS_FILE}: {save_e}")

        return empty_predictions_df # Return the correctly structured empty DataFrame


def store_prediction(prediction_date, predicted_direction, confidence_score):
    """Stores the prediction in a CSV file."""
    try:
        # Load existing predictions or create an empty DataFrame
        predictions_df = load_predictions() # Use the robust load function

        # Convert prediction_date to datetime if it's not already
        prediction_date = pd.to_datetime(prediction_date)


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
                # Drop any blank rows before saving, after ensuring empty strings are NA
                for col in ['Predicted_Direction', 'Actual_Outcome']:
                     if col in predictions_df.columns: # Add check if column exists
                        predictions_df.loc[:, col] = predictions_df[col].replace(r'^\s*$', pd.NA, regex=True)
                # Drop rows where Predicted_Direction is NA
                predictions_df.dropna(subset=['Predicted_Direction'], inplace=True)

                # Save with index=True to ensure the 'Date' index is written as a column
                predictions_df.to_csv(PREDICTIONS_FILE, index=True, index_label='Date')
                st.write(f"Prediction for {prediction_date.date()} stored.\") # Keep this message as it's an action confirmation
            except Exception as e:
                 st.error(f"Could not save prediction to {PREDICTIONS_FILE}: {e}")


    except Exception as e:
        st.error(f"An error occurred while storing the prediction: {e}")


def update_actual_outcomes(historical_data_processed):
    """Updates the actual outcomes for historical predictions."""
    # st.write("Attempting to update historical prediction outcomes...") # Suppressed
    try:
        # Load existing predictions
        predictions_df = load_predictions() # This will now use the updated dropping logic

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
             # Use error handling for saving, after ensuring empty strings are NA
            try:
                for col in ['Predicted_Direction', 'Actual_Outcome']:
                     if col in predictions_df.columns: # Add check if column exists
                        predictions_df.loc[:, col] = predictions_df[col].replace(r'^\s*$', pd.NA, regex=True)
                # Drop rows where Predicted_Direction is NA
                predictions_df.dropna(subset=['Predicted_Direction'], inplace=True)

                # Save with index=True to ensure the 'Date' index is written as a column
                predictions_df.to_csv(PREDICTIONS_FILE, index=True, index_label='Date')
                # st.write(f"Updated {updated_count} historical prediction outcomes.\") # Suppressed
            except Exception as e:
                st.error(f"Could not save updated predictions to {PREDICTIONS_FILE}: {e}")


        return predictions_df # Return updated predictions dataframe

    except Exception as e:
        st.error(f"An error occurred while updating actual outcomes: {e}")
        # Return the predictions_df as it was before the attempt to update
        return predictions_df


# --- Streamlit App ---

st.title(f"Stock Price Direction Prediction Dashboard ({STOCK_TICKER})")

st.write("Click the button below to fetch the latest data, update the model, and get a prediction for the next trading day\'s price direction.")

# Initialize session state for latest_day_data and historical_data_processed
# Fetch and process data initially only if not in session state
# This block is outside the button, so data is loaded/processed on each app rerun
# The fetch_raw_data_from_psx function now has a reduced TTL cache
raw_data = load_and_update_historical_data(STOCK_TICKER) # Use the new function to load/update local data
historical_data_processed, latest_day_data = process_data(raw_data)

# Store in session state for potential use in other parts of the app if needed,
# though the primary data source is now the local file via raw_data
st.session_state['latest_day_data'] = latest_day_data
st.session_state['historical_data_processed'] = historical_data_processed


# Update actual outcomes for historical predictions when the app loads or reruns
# Pass historical_data_processed which now includes the latest data from PSX
historical_predictions_df = update_actual_outcomes(historical_data_processed)


# Display the \"Run Analysis and Get Prediction\" button first
if st.button("Run Analysis and Get Prediction"):
    st.write("Running analysis...") # Keep this to indicate processing started

    # Use historical_data_processed from the updated local file for training
    if not historical_data_processed.empty and not latest_day_data.empty:
        # 3. Train Classification Model
        model_classification, model_classes = train_classification_model(historical_data_processed)

        if model_classification is not None:
            # Prepare latest day features for prediction from the latest_day_data
            # Ensure 'Target' and 'Price_Direction' are dropped as they are not features
            X_latest = latest_day_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Price_Direction'], axis=1, errors='ignore')


            # 4. Make Prediction for the next day
            predicted_direction_tomorrow, confidence_score_tomorrow, date_of_latest_data = make_prediction(model_classification, X_latest, model_classes)

            if predicted_direction_tomorrow is not None:
                # The prediction date is the day AFTER the latest data point
                predicted_date = date_of_latest_data + timedelta(days=1)

                # Adjust prediction date to skip weekends
                while predicted_date.weekday() >= 5: # Monday is 0, Sunday is 6
                     predicted_date += timedelta(days=1)


                # --- Display prediction for the NEXT trading day using HTML table in markdown ---
                # Format the date for the title
                day_name = calendar.day_name[predicted_date.weekday()]
                day_with_suffix = str(predicted_date.day) + ('th' if 11<=predicted_date.day<=13 else {1:'st', 2:'nd', 3:'rd'}.get(predicted_date.day%10, 'th'))
                month_name = calendar.month_name[predicted_date.month]
                prediction_title = f"Prediction for {day_name}, {day_with_suffix} {month_name}, {predicted_date.year}"

                st.subheader(prediction_title)

                # Prepare data for HTML table, ensuring confidence score is formatted as percentage string
                predicted_direction_display = predicted_direction_tomorrow
                confidence_score_display = f'{confidence_score_tomorrow:.2%}' # Format confidence as percentage string

                # Construct HTML table string with centering and border styling - Removed background-color
                prediction_html = f"""
                <table style="width:100%; text-align: center; border-collapse: collapse;">
                  <thead>
                    <tr>
                      <th style="border: 1px solid #dddddd; padding: 8px;">Predicted Direction</th>
                      <th style="border: 1px solid #dddddd; padding: 8px;">Confidence Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style="border: 1px solid #dddddd; padding: 8px;">{predicted_direction_display}</td>
                      <td style="border: 1px solid #dddddd; padding: 8px;">{confidence_score_display}</td>
                    </tr>
                  </tbody>
                </table>
                """
                st.markdown(prediction_html, unsafe_allow_html=True)


                # 5. Store the new prediction
                # Store the prediction for the NEXT day
                store_prediction(predicted_date, predicted_direction_tomorrow, confidence_score_tomorrow)


            # --- REMOVED: Display model performance metrics for the PREVIOUS day ---


            # Optionally display training accuracy
            # st.write(f"Accuracy on training data: {train_accuracy:.2f}") # Need to pass train_accuracy from train_classification_model


        else:
            st.warning("Model could not be trained.")

    else:
        st.error("Insufficient data to run analysis and prediction.")

# Display \"Summary for Today\" section after the button and prediction results
# Clarify label to indicate it's the latest available data from the source
st.subheader("Summary for Last Available Trading Day:") # MODIFIED LABEL HERE
# Use latest_day_data from the processed data (which comes from the updated local file)
if not latest_day_data.empty:
    latest_day_data_for_display = latest_day_data

    # Calculate and format the summary data points
    # Check if required columns exist before accessing them
    # Corrected LDCP calculation to use the 'Close' value from the second to last row of raw_data
    # Ensure raw_data has at least two rows before accessing index -2
    ldcp = raw_data['Close'].iloc[-2] if len(raw_data) >= 2 and 'Close' in raw_data.columns and not pd.isna(raw_data['Close'].iloc[-2]) else "N/A"

    current_price = latest_day_data_for_display['Close'].iloc[0] if 'Close' in latest_day_data_for_display.columns and not latest_day_data_for_display['Close'].empty and not pd.isna(latest_day_data_for_display['Close'].iloc[0]) else "N/A"
    change = current_price - ldcp if isinstance(current_price, (int, float)) and isinstance(ldcp, (int, float)) else "N/A"
    volume = latest_day_data_for_display['Volume'].iloc[0] if 'Volume' in latest_day_data_for_display.columns and not latest_day_data_for_display['Volume'].empty and not pd.isna(latest_day_data_for_display['Volume'].iloc[0]) else "N/A"
    latest_day_open = latest_day_data_for_display['Open'].iloc[0] if 'Open' in latest_day_data_for_display.columns and not latest_day_data_for_display['Open'].empty and not pd.isna(latest_day_data_for_display['Open'].iloc[0]) else "N/A"
    latest_day_high = latest_day_data_for_display['High'].iloc[0] if 'High' in latest_day_data_for_display.columns and not latest_day_data_for_display['High'].empty and not pd.isna(latest_day_data_for_display['High'].iloc[0]) else "N/A"
    latest_day_low = latest_day_data_for_display['Low'].iloc[0] if 'Low' in latest_day_data_for_display.columns and not latest_day_data_for_display['Low'].empty and not pd.isna(latest_day_data_for_display['Low'].iloc[0]) else "N/A"


    # Format numerical values to 2 decimal places, handle "N/A", return as strings for consistent type in DataFrame
    ldcp_str = f'{ldcp:.2f}' if isinstance(ldcp, (int, float)) else "N/A"
    open_str = f'{latest_day_open:.2f}' if isinstance(latest_day_open, (int, float)) else "N/A"
    high_str = f'{latest_day_high:.2f}' if isinstance(latest_day_high, (int, float)) else "N/A"
    low_str = f'{latest_day_low:.2f}' if isinstance(latest_day_low, (int, float)) else "N/A"
    current_str = f'{current_price:.2f}' if isinstance(current_price, (int, float)) else "N/A"
    change_str = f'{change:.2f}' if isinstance(change, (int, float)) else "N/A"
    volume_str = f'{float(volume):,.0f}' if isinstance(volume, (int, float)) else "N/A" # Format volume with commas

    # Construct HTML table string with centering and border styling - Removed background-color
    summary_html = f"""
    <table style="width:100%; text-align: center; border-collapse: collapse;">
      <thead>
        <tr>
          <th style="border: 1px solid #dddddd; padding: 8px;">LDCP</th>
          <th style="border: 1px solid #dddddd; padding: 8px;">Open</th>
          <th style="border: 1px solid #dddddd; padding: 8px;">High</th>
          <th style="border: 1px solid #dddddd; padding: 8px;">Low</th>
          <th style="border: 1px solid #dddddd; padding: 8px;">Current</th>
          <th style="border: 1px solid #dddddd; padding: 8px;">Change</th>
          <th style="border: 1px solid #dddddd; padding: 8px;">Volume</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="border: 1px solid #dddddd; padding: 8px;">{ldcp_str}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{open_str}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{high_str}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{low_str}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{current_str}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{change_str}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{volume_str}</td>
        </tr>
      </tbody>
    </table>
    """
    st.markdown(summary_html, unsafe_allow_html=True)


else:
    st.write("No data available for the latest day.")


# Display historical stock data and charts as the last section
st.subheader("Historical Stock Data:")
# Use the combined local historical data for the chart and display
if not raw_data.empty: # raw_data now holds the combined local historical data
    st.line_chart(raw_data['Close'])
    # Display historical data with specified columns, latest entries first
    # Use raw_data for the table to ensure the latest day is included
    if not raw_data.empty:
        historical_data_display = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy() # Display basic historical data
        historical_data_display.index = historical_data_display.index.date # Convert index to date objects for display
        # Ensure columns are numeric for formatting, coerce errors to handle potential non-numeric values
        historical_data_display = historical_data_display.apply(pd.to_numeric, errors='coerce')
        # Display historical data using st.dataframe with use_container_width and height
        st.dataframe(historical_data_display.applymap('{:.2f}'.format).sort_index(ascending=False), use_container_width=True, height=300)
    else:
         st.write("No sufficient historical stock data available for the table after processing.")
else:
     st.write("No raw historical stock data file found.")

# Display historical predictions and outcomes
st.subheader("Historical Predictions and Outcomes:")
# Reload predictions to ensure the latest updates are shown
historical_predictions_df = load_predictions() # This will now use the updated dropping logic

if not historical_predictions_df.empty:
    # Calculate accuracy of historical predictions with known outcomes
    evaluated_predictions = historical_predictions_df.dropna(subset=['Actual_Outcome'])
    if not evaluated_predictions.empty:
        # Ensure 'Actual_Outcome' and 'Predicted_Direction' are strings for comparison
        historical_accuracy = accuracy_score(evaluated_predictions['Actual_Outcome'].astype(str), evaluated_predictions['Predicted_Direction'].astype(str))
        st.write(f"Historical Prediction Accuracy (evaluated outcomes): {historical_accuracy:.2%}") # Display as percentage

    # --- Display historical predictions using st.dataframe ---
    # Prepare data for st.dataframe, formatting confidence score as percentage string
    historical_predictions_display = historical_predictions_df.copy()

    # Refined cleaning: Ensure empty strings or whitespace are treated as NA before dropping/formatting
    for col in ['Predicted_Direction', 'Actual_Outcome']:
        if col in historical_predictions_display.columns:
            historical_predictions_display.loc[:, col] = historical_predictions_display[col].replace(r'^\s*$', pd.NA, regex=True)

    # Drop rows where Predicted_Direction is NA - This is now done in load_predictions

    # Apply formatting only to non-NA confidence scores
    historical_predictions_display['Confidence Score'] = historical_predictions_display['Confidence_Score'].apply(lambda x: f'{x:.2%}' if pd.notna(x) else "N/A") # Renaming here

    # Ensure Actual_Outcome is string for consistent display, replacing NA with "N/A"
    historical_predictions_display['Actual Outcome'] = historical_predictions_display['Actual_Outcome'].astype(str).replace('NA', 'N/A') # Renaming here

    # Rename 'Predicted_Direction' column for display
    historical_predictions_display = historical_predictions_display.rename(columns={'Predicted_Direction': 'Predicted Direction'})
    # 'Confidence_Score' and 'Actual_Outcome' are already renamed in the previous two lines

    # Drop the original 'Actual_Outcome' column if it still exists and is not the renamed one
    # Check for both original and new name existence before dropping
    if 'Actual_Outcome' in historical_predictions_display.columns and 'Actual Outcome' in historical_predictions_display.columns and 'Actual_Outcome' != 'Actual Outcome':
         historical_predictions_display = historical_predictions_display.drop(columns=['Actual_Outcome'])
    # Check for original Confidence_Score column existence if it wasn't renamed in the apply step
    if 'Confidence_Score' in historical_predictions_display.columns and 'Confidence Score' in historical_predictions_display.columns and 'Confidence_Score' != 'Confidence Score':
         historical_predictions_display = historical_predictions_display.drop(columns=['Confidence_Score'])


    # Reorder columns for display
    # Ensure columns exist after cleaning and renaming
    display_columns = []
    if 'Predicted Direction' in historical_predictions_display.columns: display_columns.append('Predicted Direction')
    if 'Confidence Score' in historical_predictions_display.columns: display_columns.append('Confidence Score')
    if 'Actual Outcome' in historical_predictions_display.columns: display_columns.append('Actual Outcome')

    if display_columns:
        historical_predictions_display = historical_predictions_display[display_columns]
    else:
        st.warning("Could not find required columns for historical predictions display after processing.")
        historical_predictions_display = pd.DataFrame() # Ensure it's an empty DataFrame if columns are missing


    # Display using st.dataframe with use_container_width, sorting by date descending and setting height
    if not historical_predictions_display.empty:
        st.dataframe(historical_predictions_display.sort_index(ascending=False), use_container_width=True, height=300) # Set a larger height for historical data
    else:
        st.write("No historical predictions to display after cleaning.")


else:
    st.write("No recorded predictions found.")
