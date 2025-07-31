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
    # Ensure the data directory exists before trying to read or write
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

            # Check if the loaded DataFrame is empty (e.g., file was created but empty)
            if predictions_df.empty:
                # st.write(f"Loaded empty predictions file from {PREDICTIONS_FILE}. Starting with empty DataFrame.") # Suppressed
                return empty_predictions_df # Return the correctly structured empty DataFrame
            else:
                 # st.write(f"Loaded {len(predictions_df)} historical predictions from {PREDICTIONS_FILE}") # Suppressed
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
            # st.write(f"Created a new empty predictions file at {PREDICTIONS_FILE}") # Suppressed
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
                # Save with index=True to ensure the 'Date' index is written as a column
                predictions_df.to_csv(PREDICTIONS_FILE, index=True, index_label='Date')
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
                # Save with index=True to ensure the 'Date' index is written as a column
                predictions_df.to_csv(PREDICTIONS_FILE, index=True, index_label='Date')
                # st.write(f"Updated {updated_count} historical prediction outcomes.") # Suppressed
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

# Initialize session state for latest_day_data if it doesn't exist
if 'latest_day_data' not in st.session_state:
    # Fetch and process data initially only if not in session state
    raw_data = fetch_raw_data(STOCK_TICKER, DATA_FILE)
    historical_data_processed, latest_day_data = process_data(raw_data)
    st.session_state['latest_day_data'] = latest_day_data
    st.session_state['historical_data_processed'] = historical_data_processed
else:
    # Retrieve data from session state on subsequent reruns
    latest_day_data = st.session_state['latest_day_data']
    historical_data_processed = st.session_state['historical_data_processed']
    # Re-fetch raw data on rerun to ensure the chart and raw data table are up-to-date
    # This will use the cache if not expired
    raw_data = fetch_raw_data(STOCK_TICKER, DATA_FILE)


# Update actual outcomes for historical predictions when the app loads or reruns
# Pass historical_data_processed from session state
historical_predictions_df = update_actual_outcomes(historical_data_processed)


# Display the \"Run Analysis and Get Prediction\" button first
if st.button("Run Analysis and Get Prediction"):
    st.write("Running analysis...") # Keep this to indicate processing started

    # Use historical_data_processed from session state for training
    if 'historical_data_processed' in st.session_state and not st.session_state['historical_data_processed'].empty and 'latest_day_data' in st.session_state and not st.session_state['latest_day_data'].empty:
        # 3. Train Classification Model
        model_classification, model_classes = train_classification_model(st.session_state['historical_data_processed'])

        if model_classification is not None:
            # Prepare latest day features for prediction from session state
            # Ensure 'Target' and 'Price_Direction' are dropped as they are not features
            X_latest = st.session_state['latest_day_data'].drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Price_Direction'], axis=1, errors='ignore')


            # 4. Make Prediction for the next day
            predicted_direction_tomorrow, confidence_score_tomorrow, date_of_latest_data = make_prediction(model_classification, X_latest, model_classes)

            if predicted_direction_tomorrow is not None:
                # The prediction date is the day AFTER the latest data point
                predicted_date = date_of_latest_data + timedelta(days=1)

                # --- Display prediction for the NEXT trading day using st.columns ---
                # Format the date for the title
                # Get the day name (e.g., Friday)
                day_name = calendar.day_name[predicted_date.weekday()]
                # Get the day with suffix (e.g., 1st)
                day_with_suffix = str(predicted_date.day) + ('th' if 11<=predicted_date.day<=13 else {1:'st', 2:'nd', 3:'rd'}.get(predicted_date.day%10, 'th'))
                # Get the month name (e.g., August)
                month_name = calendar.month_name[predicted_date.month]
                # Format the full title
                prediction_title = f"Prediction for {day_name}, {day_with_suffix} {month_name}, {predicted_date.year}"

                st.subheader(prediction_title)

                # Display prediction details using columns for alignment
                pred_col1, pred_col2 = st.columns(2)

                with pred_col1:
                    st.markdown("<div style='text-align: center;'>**Predicted Direction**</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center;'>{predicted_direction_tomorrow}</div>", unsafe_allow_html=True)

                with pred_col2:
                    st.markdown("<div style='text-align: center;'>**Confidence Score**</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center;'>{confidence_score_tomorrow:.2%}</div>", unsafe_allow_html=True)


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
st.subheader("Summary for Today:")
# Use latest_day_data from session state for displaying the summary
if 'latest_day_data' in st.session_state and not st.session_state['latest_day_data'].empty:
    latest_day_data_for_display = st.session_state['latest_day_data']

    # Calculate and format the summary data points
    ldcp = latest_day_data_for_display['Close_Lag1'].iloc[0] if 'Close_Lag1' in latest_day_data_for_display.columns and not pd.isna(latest_day_data_for_display['Close_Lag1'].iloc[0]) else "N/A"
    current_price = latest_day_data_for_display['Close'].iloc[0] if 'Close' in latest_day_data_for_display.columns and not pd.isna(latest_day_data_for_display['Close'].iloc[0]) else "N/A"
    change = current_price - ldcp if isinstance(current_price, (int, float)) and isinstance(ldcp, (int, float)) else "N/A"
    volume = latest_day_data_for_display['Volume'].iloc[0] if 'Volume' in latest_day_data_for_display.columns and not pd.isna(latest_day_data_for_display['Volume'].iloc[0]) else "N/A"
    latest_day_open = latest_day_data_for_display['Open'].iloc[0] if 'Open' in latest_day_data_for_display.columns and not pd.isna(latest_day_data_for_display['Open'].iloc[0]) else "N/A"
    latest_day_high = latest_day_data_for_display['High'].iloc[0] if 'High' in latest_day_data_for_display.columns and not pd.isna(latest_day_data_for_display['High'].iloc[0]) else "N/A"
    latest_day_low = latest_day_data_for_display['Low'].iloc[0] if 'Low' in latest_day_data_for_display.columns and not pd.isna(latest_day_data_for_display['Low'].iloc[0]) else "N/A"

    # Format numerical values to 2 decimal places, handle "N/A"
    ldcp_str = f'{ldcp:.2f}' if isinstance(ldcp, (int, float)) else "N/A"
    open_str = f'{latest_day_open:.2f}' if isinstance(latest_day_open, (int, float)) else "N/A"
    high_str = f'{latest_day_high:.2f}' if isinstance(latest_day_high, (int, float)) else "N/A"
    low_str = f'{latest_day_low:.2f}' if isinstance(latest_day_low, (int, float)) else "N/A"
    current_str = f'{current_price:.2f}' if isinstance(current_price, (int, float)) else "N/A"
    change_str = f'{change:.2f}' if isinstance(change, (int, float)) else "N/A"
    volume_str = f'{float(volume):,.0f}' if isinstance(volume, (int, float)) else "N/A" # Format volume with commas

    # Display using st.columns for better layout control
    # Use relative widths for columns to try and keep them from stretching unevenly
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1.5]) # Adjusted widths

    with col1:
        st.markdown("<div style='text-align: center;'>**LDCP**</div>", unsafe_allow_html=True)
        st.write(f"<div style='text-align: center;'>{ldcp_str}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='text-align: center;'>**Open**</div>", unsafe_allow_html=True)
        st.write(f"<div style='text-align: center;'>{open_str}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div style='text-align: center;'>**High**</div>", unsafe_allow_html=True)
        st.write(f"<div style='text-align: center;'>{high_str}</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div style='text-align: center;'>**Low**</div>", unsafe_allow_html=True)
        st.write(f"<div style='text-align: center;'>{low_str}</div>", unsafe_allow_html=True)
    with col5:
        st.markdown("<div style='text-align: center;'>**Current**</div>", unsafe_allow_html=True)
        st.write(f"<div style='text-align: center;'>{current_str}</div>", unsafe_allow_html=True)
    with col6:
        st.markdown("<div style='text-align: center;'>**Change**</div>", unsafe_allow_html=True)
        st.write(f"<div style='text-align: center;'>{change_str}</div>", unsafe_allow_html=True)
    with col7:
        st.markdown("<div style='text-align: center;'>**Volume**</div>", unsafe_allow_html=True)
        st.write(f"<div style='text-align: center;'>{volume_str}</div>", unsafe_allow_html=True)

else:
    st.write("No data available for the latest day.")


# Display historical stock data and charts as the last section
st.subheader("Historical Stock Data:")
# Use the original raw_data for the chart and display
if not raw_data.empty:
    st.line_chart(raw_data['Close'])
    # Display historical data excluding the last day with specified columns, latest entries first
    # Remove time part from the index for display
    # Use historical_data_processed from session state for the table
    if 'historical_data_processed' in st.session_state and not st.session_state['historical_data_processed'].empty:
        historical_data_display = st.session_state['historical_data_processed'][['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'MA_50', 'RSI']].copy()
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

    # --- Display historical predictions using st.columns for formatting and alignment ---
    # Define column headers and widths
    hist_pred_cols = st.columns([1, 2, 2, 2]) # Date, Predicted Direction, Confidence Score, Actual Outcome

    # Display headers
    with hist_pred_cols[0]:
         st.markdown("<div style='text-align: center;'>**Date**</div>", unsafe_allow_html=True)
    with hist_pred_cols[1]:
         st.markdown("<div style='text-align: center;'>**Predicted Direction**</div>", unsafe_allow_html=True)
    with hist_pred_cols[2]:
         st.markdown("<div style='text-align: center;'>**Confidence Score**</div>", unsafe_allow_html=True)
    with hist_pred_cols[3]:
         st.markdown("<div style='text-align: center;'>**Actual Outcome**</div>", unsafe_allow_html=True)

    # Display rows, sorting by date descending
    for index, row in historical_predictions_df.sort_index(ascending=False).iterrows():
        date_str = index.strftime('%Y-%m-%d') # Format date
        predicted_direction = row['Predicted_Direction'] if pd.notna(row['Predicted_Direction']) else "N/A"
        confidence_score = row['Confidence_Score'] if pd.notna(row['Confidence_Score']) else pd.NA # Keep as number/NA for formatting
        actual_outcome = row['Actual_Outcome'] if pd.notna(row['Actual_Outcome']) else "N/A"

        # Format confidence score as percentage, handle NA
        confidence_str = f'{confidence_score:.2%}' if pd.notna(confidence_score) else "N/A"


        row_cols = st.columns([1, 2, 2, 2]) # Match header column widths

        with row_cols[0]:
             st.markdown(f"<div style='text-align: center;'>{date_str}</div>", unsafe_allow_html=True)
        with row_cols[1]:
             st.markdown(f"<div style='text-align: center;'>{predicted_direction}</div>", unsafe_allow_html=True)
        with row_cols[2]:
             st.markdown(f"<div style='text-align: center;'>{confidence_str}</div>", unsafe_allow_html=True)
        with row_cols[3]:
             st.markdown(f"<div style='text-align: center;'>{actual_outcome}</div>", unsafe_allow_html=True)

else:
    st.write("No recorded predictions found.")
