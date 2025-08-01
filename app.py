import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from psx import stocks
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import calendar

# --- Configuration ---
STOCK_TICKER = "UBL"
PREDICTIONS_FILE = os.path.join("data", "predictions.csv")
LOCAL_HISTORICAL_DATA_FILE = os.path.join("data", "local_historical_data.csv")


# --- Helper Functions ---
@st.cache_data(ttl=timedelta(hours=1))
def fetch_raw_data_from_psx(ticker, start_date, end_date):
    """Fetches raw historical data from PSX for a given date range."""
    try:
        raw_data = stocks(ticker, start_date, end_date)
        if raw_data.empty:
            return pd.DataFrame()
        return raw_data
    except Exception as e:
        st.error(f"An error occurred during raw data fetching from PSX: {e}")
        return pd.DataFrame()

def load_and_update_historical_data(ticker):
    """Loads local historical data and updates it with new data from PSX."""
    os.makedirs(os.path.dirname(LOCAL_HISTORICAL_DATA_FILE), exist_ok=True)
    local_data = pd.DataFrame()
    last_local_date = None

    if os.path.exists(LOCAL_HISTORICAL_DATA_FILE):
        try:
            local_data = pd.read_csv(LOCAL_HISTORICAL_DATA_FILE, index_col='Date', parse_dates=['Date'])
            local_data.index = pd.to_datetime(local_data.index)
            if not local_data.empty:
                last_local_date = local_data.index.max().date()
        except Exception as e:
            st.warning(f"Could not load local historical data from {LOCAL_HISTORICAL_DATA_FILE}: {e}. Starting fresh.")
            local_data = pd.DataFrame()

    if last_local_date:
        fetch_start_date = last_local_date + timedelta(days=1)
    else:
        fetch_start_date = date.today() - timedelta(days=730)

    # Fetch data up to yesterday
    fetch_end_date = date.today() - timedelta(days=1)
    if fetch_start_date <= fetch_end_date:
        new_data = fetch_raw_data_from_psx(ticker, fetch_start_date, fetch_end_date)
        if not new_data.empty:
            new_data.index = pd.to_datetime(new_data.index)
            combined_data = pd.concat([local_data, new_data[~new_data.index.isin(local_data.index)]])
            combined_data.sort_index(inplace=True)
            try:
                combined_data.to_csv(LOCAL_HISTORICAL_DATA_FILE, index=True, index_label='Date')
            except Exception as e:
                st.error(f"An error occurred while saving updated historical data: {e}")
            return combined_data
    return local_data

def process_data(raw_data):
    """Adds technical indicators and features to the raw data."""
    if raw_data.empty:
        st.warning("No raw data to process.")
        return pd.DataFrame(), pd.DataFrame()

    processed_data = raw_data.copy()
    try:
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
        processed_data['Close_Lag1'] = processed_data['Close'].shift(1)
        processed_data['Close_Lag2'] = processed_data['Close'].shift(2)
        processed_data['Close_Lag3'] = processed_data['Close'].shift(3)
        processed_data['Volume_MA_5'] = processed_data['Volume'].rolling(window=5).mean()
        processed_data['MA_5'] = processed_data['Close'].rolling(window=5).mean()
        processed_data['MA_5_vs_20'] = processed_data['MA_5'] - processed_data['MA_20']
        processed_data['Volatility_14'] = processed_data['Close'].rolling(window=14).std()
        processed_data['Volume_Change_Lag1'] = processed_data['Volume'].pct_change().shift(1)
        processed_data['Price_Change_1d'] = processed_data['Close'].diff(1).shift(1)
        processed_data['Price_Change_3d'] = processed_data['Close'].diff(3).shift(1)
        processed_data['Target'] = processed_data['Close'].shift(-1)
        price_difference = processed_data['Target'] - processed_data['Close']
        processed_data['Price_Direction'] = np.where(price_difference > 0, 'Up', 'Down')

        if len(processed_data) < 2:
             st.warning("Not enough data to separate latest day for prediction.")
             return processed_data.dropna(inplace=False), pd.DataFrame()

        latest_day_data = processed_data.tail(1)
        historical_data_for_training = processed_data.iloc[:-1].copy()
        historical_data_processed = historical_data_for_training.dropna(inplace=False)

        if historical_data_processed.empty:
             st.warning("Historical data is empty after dropping NaNs.")
             return pd.DataFrame(), pd.DataFrame()

        return historical_data_processed, latest_day_data
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return pd.DataFrame(), pd.DataFrame()

def train_classification_model(historical_data):
    """Trains a classification model to predict price direction."""
    if historical_data.empty:
        st.warning("No data available to train the model.")
        return None, None
    features_classification = historical_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Price_Direction'], axis=1)
    target_classification = historical_data['Price_Direction']
    if len(features_classification) < 1:
         st.warning("Not enough data to train the model.")
         return None, None
    model_classification = RandomForestClassifier(random_state=42)
    model_classification.fit(features_classification, target_classification)
    return model_classification, model_classification.classes_.tolist()

def make_prediction(model, X_latest, model_classes):
    """Makes a prediction and gets confidence score for the latest data."""
    if model is None or X_latest.empty:
        st.warning("Model not trained or no data for prediction.")
        return None, None, None
    X_latest_cleaned = X_latest.dropna(inplace=False)
    if X_latest_cleaned.empty:
        st.warning("Latest data contains NaNs in features required for prediction.")
        return None, None, None
    predicted_direction = model.predict(X_latest_cleaned)[0]
    y_prob_classification = model.predict_proba(X_latest_cleaned)
    predicted_class_index = model_classes.index(predicted_direction)
    confidence_score = y_prob_classification[0, predicted_class_index]
    return predicted_direction, confidence_score, X_latest_cleaned.index[0]

def load_predictions():
    """Loads historical predictions from a CSV file."""
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    expected_columns = ['Predicted_Direction', 'Confidence_Score', 'Actual_Outcome']
    index_column_name = 'Date'
    empty_predictions_df = pd.DataFrame(columns=expected_columns).astype({
        'Predicted_Direction': 'string', 'Confidence_Score': 'float64', 'Actual_Outcome': 'string'
    }).set_index(pd.to_datetime([]).rename(index_column_name))
    if os.path.exists(PREDICTIONS_FILE):
        try:
            predictions_df = pd.read_csv(PREDICTIONS_FILE, index_col=index_column_name, parse_dates=[index_column_name])
            predictions_df.index = pd.to_datetime(predictions_df.index)
            for col in expected_columns:
                if col not in predictions_df.columns:
                    predictions_df[col] = pd.Series(dtype='string')
            predictions_df = predictions_df.astype({'Predicted_Direction': 'string', 'Confidence_Score': 'float64', 'Actual_Outcome': 'string'})
            for col in ['Predicted_Direction', 'Actual_Outcome']:
                if col in predictions_df.columns:
                    predictions_df.loc[:, col] = predictions_df[col].replace(r'^\s*$', pd.NA, regex=True)
            predictions_df.dropna(subset=['Predicted_Direction'], inplace=True)
            if predictions_df.empty:
                return empty_predictions_df
            else:
                 return predictions_df
        except Exception as e:
            st.warning(f"Could not read {PREDICTIONS_FILE} correctly due to error: {e}. Attempting to create a fresh file.")
            try:
                empty_predictions_df.to_csv(PREDICTIONS_FILE, index=True, index_label=index_column_name)
            except Exception as save_e:
                 st.error(f"Could not save a new empty predictions file to {PREDICTIONS_FILE}: {save_e}")
            return empty_predictions_df
    else:
        try:
            empty_predictions_df.to_csv(PREDICTIONS_FILE, index=True, index_label=index_column_name)
        except Exception as save_e:
             st.error(f"Could not save a new empty predictions file to {PREDICTIONS_FILE}: {save_e}")
        return empty_predictions_df

def store_prediction(prediction_date, predicted_direction, confidence_score):
    """Stores the prediction in a CSV file."""
    try:
        predictions_df = load_predictions()
        prediction_date = pd.to_datetime(prediction_date)
        if prediction_date not in predictions_df.index:
            new_prediction = pd.DataFrame([{'Predicted_Direction': predicted_direction, 'Confidence_Score': confidence_score, 'Actual_Outcome': pd.NA}], index=[prediction_date])
            new_prediction.index = pd.to_datetime(new_prediction.index)
            new_prediction = new_prediction.reindex(columns=predictions_df.columns).astype({'Predicted_Direction': 'string', 'Confidence_Score': 'float64', 'Actual_Outcome': 'string'})
            predictions_df = pd.concat([predictions_df, new_prediction])
            predictions_df.sort_index(inplace=True)
            try:
                for col in ['Predicted_Direction', 'Actual_Outcome']:
                     if col in predictions_df.columns:
                        predictions_df.loc[:, col] = predictions_df[col].replace(r'^\s*$', pd.NA, regex=True)
                predictions_df.dropna(subset=['Predicted_Direction'], inplace=True)
                predictions_df.to_csv(PREDICTIONS_FILE, index=True, index_label='Date')
                st.write(f"Prediction for {prediction_date.date()} stored.")
            except Exception as e:
                 st.error(f"Could not save prediction to {PREDICTIONS_FILE}: {e}")
    except Exception as e:
        st.error(f"An error occurred while storing the prediction: {e}")

def update_actual_outcomes(historical_data_processed):
    """Updates the actual outcomes for historical predictions."""
    try:
        predictions_df = load_predictions()
        if predictions_df.empty:
             return predictions_df
        historical_data_processed.index = pd.to_datetime(historical_data_processed.index)
        if 'Price_Direction' not in historical_data_processed.columns:
             st.warning("Price_Direction column not found in historical_data_processed. Cannot update outcomes.")
             return predictions_df
        updated_count = 0
        for index, row in predictions_df.iterrows():
            if pd.isna(row['Actual_Outcome']):
                prediction_date = index
                if prediction_date in historical_data_processed.index:
                    actual_direction = historical_data_processed.loc[prediction_date, 'Price_Direction']
                    predictions_df.loc[index, 'Actual_Outcome'] = actual_direction
                    updated_count += 1
        if updated_count > 0:
            try:
                for col in ['Predicted_Direction', 'Actual_Outcome']:
                     if col in predictions_df.columns:
                        predictions_df.loc[:, col] = predictions_df[col].replace(r'^\s*$', pd.NA, regex=True)
                predictions_df.dropna(subset=['Predicted_Direction'], inplace=True)
                predictions_df.to_csv(PREDICTIONS_FILE, index=True, index_label='Date')
            except Exception as e:
                st.error(f"Could not save updated predictions to {PREDICTIONS_FILE}: {e}")
        return predictions_df
    except Exception as e:
        st.error(f"An error occurred while updating actual outcomes: {e}")
        return predictions_df

# --- Streamlit App ---
st.title(f"Stock Price Direction Prediction Dashboard ({STOCK_TICKER})")
st.write("Click the button below to fetch the latest data, update the model, and get a prediction for the next trading day's price direction.")

raw_data = load_and_update_historical_data(STOCK_TICKER)

# Use raw_data directly for processing as it now contains data up to yesterday
historical_data_processed, latest_day_data = process_data(raw_data)

st.session_state['latest_day_data'] = latest_day_data
st.session_state['historical_data_processed'] = historical_data_processed
st.session_state['raw_data'] = raw_data # Store raw_data for the chart/table display


historical_predictions_df = update_actual_outcomes(historical_data_processed)

if st.button("Run Analysis and Get Prediction"):
    st.write("Running analysis...")
    if not historical_data_processed.empty and not latest_day_data.empty:
        model_classification, model_classes = train_classification_model(historical_data_processed)
        if model_classification is not None:
            X_latest = latest_day_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Price_Direction'], axis=1, errors='ignore')
            predicted_direction_tomorrow, confidence_score_tomorrow, date_of_latest_data = make_prediction(model_classification, X_latest, model_classes)
            if predicted_direction_tomorrow is not None:
                predicted_date = date_of_latest_data + timedelta(days=1)
                while predicted_date.weekday() >= 5:
                     predicted_date += timedelta(days=1)
                day_name = calendar.day_name[predicted_date.weekday()]
                day_with_suffix = str(predicted_date.day) + ('th' if 11<=predicted_date.day<=13 else {1:'st', 2:'nd', 3:'rd'}.get(predicted_date.day%10, 'th'))
                month_name = calendar.month_name[predicted_date.month]
                prediction_title = f"Prediction for {day_name}, {day_with_suffix} {month_name}, {predicted_date.year}"
                st.subheader(prediction_title)
                predicted_direction_display = predicted_direction_tomorrow
                confidence_score_display = f'{confidence_score_tomorrow:.2%}'
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
                store_prediction(predicted_date, predicted_direction_tomorrow, confidence_score_tomorrow)
        else:
            st.warning("Model could not be trained.")
    else:
        st.error("Insufficient data to run analysis and prediction.")

st.subheader("Summary for Last Available Trading Day:")
# Use raw_data which now contains data up to yesterday for summary
if not raw_data.empty:
    latest_day_data_for_display = raw_data.tail(1) # Get the last complete day's data (yesterday)

    if not latest_day_data_for_display.empty:
        # Calculate and format the summary data points
        # LDCP is the closing price of the second to last day
        ldcp = raw_data['Close'].iloc[-2] if len(raw_data) >= 2 and 'Close' in raw_data.columns and not pd.isna(raw_data['Close'].iloc[-2]) else "N/A"

        # Current price is the closing price of the last complete day (yesterday's close)
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
        st.write("No data available for the latest complete day.")

else:
    st.write("No data available for the latest day.")


# Display historical stock data and charts as the last section
st.subheader("Historical Stock Data:")
# Use raw_data (data up to yesterday) for the chart and table display
if not raw_data.empty:
    st.line_chart(raw_data['Close'])
    # Display historical data with specified columns, latest entries first
    # Use raw_data for the table
    if not raw_data.empty:
        # Display historical data with specified columns, excluding technical indicators for simplicity
        historical_data_display = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        historical_data_display.index = historical_data_display.index.date # Convert index to date objects for display
        # Ensure columns are numeric for formatting, coerce errors to handle potential non-numeric values
        historical_data_display = historical_data_display.apply(pd.to_numeric, errors='coerce')
        # Display historical data using st.dataframe with use_container_width and height
        st.dataframe(historical_data_display.applymap('{:.2f}'.format).sort_index(ascending=False), use_container_width=True, height=300)
    else:
         st.write("No sufficient historical stock data available for the table.")
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
