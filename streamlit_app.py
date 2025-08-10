import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import io
import math
import time
import altair as alt
from PIL import Image
import requests
from io import BytesIO

# Optional: page config
st.set_page_config(page_title="Deep Learning Playground", layout="wide")

# App title with custom styling
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
    color: #4a8cff;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🔮 Deep Learning Playground</p>', unsafe_allow_html=True)
st.caption("An interactive tool for exploring LSTM and Transformer models")

# Sidebar for navigation and settings
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Settings")
    model_type = st.radio("Select Model", ["LSTM", "Transformer"], index=0)

    # default values to expose to outer scope
    model_choice = "DistilBERT (fast)"
    custom_model_name = ""
    max_length = 256
    sentiment_threshold = 0.75

    if model_type == "LSTM":
        st.subheader("LSTM Parameters")
        epochs = st.slider("Epochs", 1, 200, 50, help="Number of training iterations")
        batch_size = st.slider("Batch Size", 8, 128, 32, help="Number of samples per gradient update")
        look_back = st.slider("Look Back Period", 1, 20, 5, help="Number of previous time steps to use")
        lstm_units = st.slider("LSTM Units", 16, 256, 64, step=16)
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)
    else:
        st.subheader("Transformer Options")
        model_choice = st.selectbox("Model", 
                                  ["DistilBERT (fast)", "RoBERTa (accurate)", "Custom Model"]) 
        if model_choice == "Custom Model":
            custom_model_name = st.text_input("Enter HuggingFace model path", value="distilbert-base-uncased-finetuned-sst-2-english")
        max_length = st.slider("Max Text Length", 64, 512, 256)
        sentiment_threshold = st.slider("Sentiment Threshold", 0.5, 0.95, 0.75, step=0.05)

# File upload with drag and drop
with st.expander("📁 Upload Data", expanded=True):
    uploaded_file = st.file_uploader("Drag and drop CSV or Text file here", 
                                   type=["csv", "txt"], 
                                   help="Upload your dataset or use demo data below")
    use_demo_data = st.checkbox("Use demo data", help="Try the app with sample data")

# Initialize data
data = None
text_data = None
target_column = None
feature_columns = None

# File reading with progress
if uploaded_file is not None:
    with st.spinner(f"Loading {uploaded_file.name}..."):
        progress_bar = st.progress(0)

        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
            progress_bar.progress(50)
            time.sleep(0.2)
            progress_bar.progress(100)

            st.success("✅ Data loaded successfully!")

            # Data explorer
            with st.expander("🔍 Data Explorer"):
                st.write(f"Shape: {data.shape[0]} rows, {data.shape[1]} columns")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Preview")
                    st.dataframe(data.head())

                with col2:
                    st.write("### Statistics")
                    st.dataframe(data.describe())

                # Interactive correlation plot for numeric data
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) > 1:
                    st.write("### Correlation Heatmap")
                    corr = data[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)

            if model_type == "LSTM" and len(data.columns) > 1:
                target_column = st.selectbox("Select Target Column", data.columns)
                feature_columns = st.multiselect("Select Feature Columns", 
                                               [col for col in data.columns if col != target_column],
                                               default=[col for col in data.columns if col != target_column])

        elif uploaded_file.name.endswith('.txt'):
            raw = uploaded_file.read()
            try:
                text_data = raw.decode("utf-8")
            except Exception:
                # already bytes->str fallback
                text_data = raw if isinstance(raw, str) else raw.decode('latin-1')
            progress_bar.progress(100)
            st.success("✅ Text loaded successfully!")

            # Text analysis
            with st.expander("📝 Text Analysis"):
                st.write(f"Character count: {len(text_data)}")
                st.write(f"Word count: {len(text_data.split())}")
                st.write(f"Line count: {len(text_data.splitlines())}")

                # Word cloud visualization
                st.write("### Word Cloud")
                if text_data.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                else:
                    st.write("No text to generate word cloud.")

# Demo data with more options
if use_demo_data and uploaded_file is None:
    with st.spinner("Loading demo data..."):
        np.random.seed(42)

        if model_type == "LSTM":
            # More sophisticated demo data
            dates = pd.date_range(start="2020-01-01", periods=365)
            base_value = 100 + np.cumsum(np.random.randn(365) * 0.5)
            seasonal = 10 * np.sin(np.linspace(0, 10*np.pi, 365))
            noise = np.random.randn(365) * 2
            values = base_value + seasonal + noise

            # Additional meaningful features
            feature1 = np.random.rand(365) * 20 + values/10  # Correlated feature
            feature2 = np.random.rand(365) * 50  # Random noise

            data = pd.DataFrame({
                "Date": dates,
                "Value": values,
                "Temperature": feature1,
                "Market_Index": feature2
            })

            target_column = "Value"
            feature_columns = ["Temperature", "Market_Index"]

            st.write("## Demo Time Series Data")
            fig = px.line(data, x="Date", y="Value", title="Sample Time Series")
            st.plotly_chart(fig, use_container_width=True)

        else:
            # More diverse sample texts
            sample_texts = [
                "I'm absolutely thrilled with this product! It exceeded all my expectations.",
                "This service is terrible. I've never been so disappointed in my life.",
                "The item works as described, but the shipping took longer than expected.",
                "Incredible value for the price. Highly recommended to all my friends!",
                "Meh, it's okay I guess. Not terrible but not great either.",
                "The customer support team was extremely helpful and resolved my issue quickly.",
                "Poor quality materials. Broke after just one week of normal use.",
                "Best purchase I've made this year! Worth every penny.",
                "The interface is confusing and the instructions are unclear.",
                "Five stars! Perfect in every way."
            ]
            data = pd.DataFrame({
                "Text": sample_texts,
                "Source": ["Product Review", "Service Complaint", "Neutral Feedback", 
                          "Product Review", "Neutral Feedback", "Service Review",
                          "Product Complaint", "Product Review", "Service Complaint",
                          "Product Review"]
            })

            st.write("## Demo Text Data")
            st.dataframe(data)

# Enhanced metric calculation (NaN-safe)
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"RMSE": float('nan'), "MAE": float('nan'), "MAPE": 'N/A', "R² Score": float('nan'), "Explained Variance": float('nan')}

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    mse = mean_squared_error(y_true_masked, y_pred_masked)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_masked, y_pred_masked)
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / np.maximum(np.abs(y_true_masked), 1e-8))) * 100
    r2 = r2_score(y_true_masked, y_pred_masked)
    explained_variance = max(0, 1 - (np.var(y_true_masked - y_pred_masked) / np.var(y_true_masked))) if np.var(y_true_masked) != 0 else 0
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": f"{mape:.2f}%",
        "R² Score": r2,
        "Explained Variance": explained_variance
    }

# Enhanced LSTM model with more flexibility
def create_lstm_model(input_shape, units=64, dropout=0.2):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(max(1, units//2)),
        Dropout(max(0.0, dropout/2)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# LSTM training with callbacks and visualization
def train_lstm(data, target_column, feature_columns=None, look_back=5, lstm_units=64, dropout_rate=0.2, epochs=50, batch_size=32):
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column]

    # Ensure there is at least one feature
    if not feature_columns:
        st.error("No feature columns found. Please include at least one feature besides the target.")
        st.stop()

    # Keep only numeric columns used
    df = data[[target_column] + feature_columns].copy()
    df = df.dropna()

    if len(df) <= look_back:
        st.error(f"Not enough data to train LSTM (need > {look_back} rows after dropping NaNs).")
        st.stop()

    target_values = df[target_column].values
    feature_values = df[feature_columns].values if feature_columns else None
    X, y = [], []

    for i in range(len(target_values) - look_back):
        if feature_values is not None:
            X.append(feature_values[i:i+look_back])
        else:
            X.append(target_values[i:i+look_back])
        y.append(target_values[i+look_back])

    X = np.array(X)
    y = np.array(y)

    if feature_values is not None:
        X = X.reshape((X.shape[0], X.shape[1], len(feature_columns)))
    else:
        X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = create_lstm_model((X.shape[1], X.shape[2]), units=lstm_units, dropout=dropout_rate)

    # Training visualization
    st.write("### Model Architecture")
    stringio = io.StringIO()
    model.summary(print_fn=lambda x: stringio.write(x + "\n"))
    st.text(stringio.getvalue())

    # Create placeholders for dynamic updates
    loss_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Custom training loop for visualization
    history = {'loss': [], 'val_loss': []}
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    with st.spinner("Training in progress..."):
        for epoch in range(epochs):
            history_epoch = model.fit(
                X_train, y_train,
                epochs=1,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )

            # Update history
            history['loss'].append(history_epoch.history['loss'][0])
            history['val_loss'].append(history_epoch.history['val_loss'][0])

            # Update the loss display
            loss_placeholder.write(f"Epoch {epoch+1}/{epochs} - Loss: {history['loss'][-1]:.4f} - Val Loss: {history['val_loss'][-1]:.4f}")

            # Update the training chart
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history['loss'], label='Training Loss')
            ax.plot(history['val_loss'], label='Validation Loss')
            ax.set_title('Training Progress')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            chart_placeholder.pyplot(fig)
            plt.close()

            # Early stopping check: Keras sets early_stopping.stopped_epoch > 0 when triggered
            if getattr(early_stopping, 'stopped_epoch', 0) > 0:
                st.warning(f"Early stopping triggered at epoch {epoch+1}")
                break

    y_pred = model.predict(X_test).flatten()
    return y_test, y_pred, model, history

# Sentiment pipeline with caching and explicit inputs
@st.cache_resource
def get_sentiment_pipeline(model_choice="DistilBERT (fast)", custom_model_name=""):
    try:
        if model_choice == "DistilBERT (fast)":
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        elif model_choice == "RoBERTa (accurate)":
            model_name = "siebert/sentiment-roberta-large-english"
        elif model_choice == "Custom Model":
            model_name = custom_model_name if custom_model_name else "distilbert-base-uncased-finetuned-sst-2-english"
        else:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# analyze_sentiment now takes pipeline and params
def analyze_sentiment(text, sentiment_pipeline, max_length=256, sentiment_threshold=0.75):
    try:
        # Some pipelines accept a single string, some prefer batches; using single here is fine
        result = sentiment_pipeline(text[:max_length], truncation=True)
        res = result[0]
        # Normalize labels
        label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
        label_raw = res.get('label', '').upper()
        label = label_map.get(label_raw, label_raw)
        score = res.get('score', 0.0)
        sentiment = 'POSITIVE' if label == 'POSITIVE' and score > sentiment_threshold else (
                    'NEGATIVE' if label == 'NEGATIVE' and score > sentiment_threshold else 'NEUTRAL')
        return {
            'label': label,
            'score': score,
            'sentiment': sentiment
        }
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        return {'label': 'ERROR', 'score': 0, 'sentiment': 'ERROR'}

# Batch sentiment analysis using pipeline batching for speed
def batch_sentiment_analysis(df, text_column='Text', sentiment_pipeline=None, max_length=256, sentiment_threshold=0.75):
    if sentiment_pipeline is None:
        st.error("Sentiment pipeline not available")
        return df

    texts = df[text_column].astype(str).tolist()
    sentiments = []
    scores = []
    detailed = []

    with st.spinner("Analyzing sentiments..."):
        progress_bar = st.progress(0)
        # Batch call — many pipelines accept lists
        try:
            results = sentiment_pipeline(texts, truncation=True)
        except Exception:
            # fallback to per-item if batch fails for model
            results = [sentiment_pipeline(t[:max_length], truncation=True)[0] for t in texts]

        for i, res in enumerate(results):
            label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
            label_raw = res.get('label', '').upper()
            label = label_map.get(label_raw, label_raw)
            score = res.get('score', 0.0)
            sentiment = 'POSITIVE' if label == 'POSITIVE' and score > sentiment_threshold else (
                        'NEGATIVE' if label == 'NEGATIVE' and score > sentiment_threshold else 'NEUTRAL')
            sentiments.append(label)
            scores.append(score)
            detailed.append(sentiment)
            progress_bar.progress(int((i + 1) / len(texts) * 100))

    df = df.copy()
    df['Sentiment'] = sentiments
    df['Confidence'] = scores
    df['Detailed_Sentiment'] = detailed
    return df

# Enhanced visualization functions
def plot_results(y_test, y_pred):
    fig = px.line(title="Actual vs Predicted Values")
    fig.add_scatter(y=y_test, name='Actual')
    fig.add_scatter(y=y_pred, name='Predicted')
    fig.update_layout(
        xaxis_title="Time Steps",
        yaxis_title="Value",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residual plot
    residuals = y_test - y_pred
    fig = px.scatter(x=y_pred, y=residuals, 
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    title="Residual Analysis")
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_distribution(df):
    # Sentiment distribution pie chart
    fig1 = px.pie(df, names='Detailed_Sentiment', title='Sentiment Distribution')

    # Confidence distribution by sentiment
    fig2 = px.box(df, x='Detailed_Sentiment', y='Confidence', 
                 title='Confidence Distribution')

    # Time series of sentiment if date column exists
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        fig3 = px.line(df, x='Date', y='Confidence', color='Detailed_Sentiment',
                      title='Sentiment Over Time')
        st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

# Advanced export options — directly present download buttons
def get_export_options(df):
    st.write("### Export Options")
    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download CSV", data=csv, file_name='results.csv', mime='text/csv')

    with col2:
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        st.download_button("📊 Download Excel", data=excel_buffer, file_name='results.xlsx', 
                         mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    with col3:
        html = df.to_html(index=False)
        st.download_button("📈 Download HTML", data=html, file_name='report.html', mime='text/html')

# Main analysis execution
if st.button("🚀 Run Analysis", type="primary"):
    if data is not None and model_type == "LSTM":
        try:
            if target_column is None:
                if len(data.columns) == 2:
                    target_column = data.columns[1]
                    feature_columns = [data.columns[0]]
                else:
                    st.error("Please select a target column")
                    st.stop()

            st.write("## 📈 LSTM Time Series Analysis")

            y_test, y_pred, model, history = train_lstm(
                data, target_column, feature_columns, look_back=look_back, 
                lstm_units=lstm_units, dropout_rate=dropout_rate, epochs=epochs, batch_size=batch_size
            )

            st.write("### 🔍 Model Evaluation")
            metrics = calculate_metrics(y_test, y_pred)

            # Display metrics in columns
            cols = st.columns(len(metrics))
            for col, (name, value) in zip(cols, metrics.items()):
                col.metric(label=name, value=value)

            # Plot results
            st.write("### 📊 Results Visualization")
            plot_results(y_test, y_pred)

            # Feature importance (simplified and consistent with input dim)
            if feature_columns:
                st.write("### ⚖️ Feature Importance")
                # Extract weights from first LSTM layer and aggregate across gates
                try:
                    weights = model.layers[0].get_weights()[0]  # shape: (input_dim, units*4)
                    importance = np.mean(np.abs(weights), axis=1)
                    # if number of features mismatches, try to align
                    if len(importance) != len(feature_columns):
                        importance = importance[:len(feature_columns)]
                    fig = px.bar(x=feature_columns, y=importance, 
                                labels={'x': 'Feature', 'y': 'Relative Importance'})
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.write("Could not compute feature importance:", e)

            # Export options
            results_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred,
                "Residual": y_test - y_pred
            })
            get_export_options(results_df)

        except Exception as e:
            st.error(f"Error in LSTM analysis: {str(e)}")
            st.exception(e)

    elif model_type == "Transformer":
        try:
            st.write("## 💬 Sentiment Analysis")
            sentiment_pipeline = get_sentiment_pipeline(model_choice=model_choice, custom_model_name=custom_model_name)

            if data is not None and 'Text' in data.columns:
                results_df = batch_sentiment_analysis(data, text_column='Text', sentiment_pipeline=sentiment_pipeline, max_length=max_length, sentiment_threshold=sentiment_threshold)

                st.write("### 🔍 Analysis Results")
                st.dataframe(results_df.style.background_gradient(
                    subset=['Confidence'], cmap='RdYlGn'
                ))

                st.write("### 📊 Visualizations")
                plot_sentiment_distribution(results_df)

                # Word clouds by sentiment
                st.write("### ☁️ Word Clouds by Sentiment")
                sentiment_types = results_df['Detailed_Sentiment'].unique()

                for sentiment in sentiment_types:
                    if sentiment != 'N/A':
                        st.write(f"#### {sentiment} Sentiment")
                        text = ' '.join(results_df[results_df['Detailed_Sentiment'] == sentiment]['Text'].astype(str))
                        if text.strip():
                            wordcloud = WordCloud(width=800, height=400, 
                                                background_color='white').generate(text)
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis("off")
                            st.pyplot(plt)
                        else:
                            st.write("No text available for this sentiment category.")

                # Export options
                get_export_options(results_df)

            elif text_data is not None:
                result = analyze_sentiment(text_data, sentiment_pipeline, max_length=max_length, sentiment_threshold=sentiment_threshold)
                st.write("## 📝 Text Analysis Result")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", result['sentiment'])
                with col2:
                    st.metric("Confidence", f"{result['score']:.2%}")

                # Highlight sentiment in text
                st.write("### ✍️ Analyzed Text Preview")
                preview_length = min(500, len(text_data))
                preview = text_data[:preview_length]

                if result['sentiment'] == 'POSITIVE':
                    st.success(preview + "..." if len(text_data) > preview_length else preview)
                elif result['sentiment'] == 'NEGATIVE':
                    st.error(preview + "..." if len(text_data) > preview_length else preview)
                else:
                    st.info(preview + "..." if len(text_data) > preview_length else preview)

            else:
                st.warning("Please upload a file with text data for sentiment analysis")

        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            st.exception(e)

    else:
        st.warning("Please upload data or select demo data to proceed")

# Add footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 12px;
    color: #888;
    text-align: center;
}
</style>
<div class="footer">
    Made with ❤️ using Streamlit | Deep Learning Playground v2.0
</div>
""", unsafe_allow_html=True)
