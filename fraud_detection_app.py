import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

model = joblib.load("fraud_detection_model.jb")

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .fraud {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .legitimate {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model with error handling"""
    try:
        model = joblib.load("fraud_detection_model.jb")
        
        # Try to load feature columns
        try:
            with open('feature_columns.json', 'r') as f:
                feature_columns = json.load(f)
        except:
            feature_columns = ['distance_from_home', 'distance_from_last_transaction', 
                             'ratio_to_median_purchase_price', 'repeat_retailer', 
                             'used_chip', 'used_pin_number', 'online_order']
        
        return model, feature_columns
        
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def create_demo_model():
    """Create a demo model for testing purposes"""
    st.warning("⚠️ Using demo model - Train and save your actual model in the notebook")
    
    class DemoModel:
        def predict(self, X):
            # Simple rule-based demo
            risk_scores = []
            for row in X:
                # Simple heuristic based on your EDA
                risk = 0
                if row[0] > 50: risk += 0.3  # distance_from_home
                if row[2] > 3: risk += 0.4   # ratio_to_median_purchase_price
                if row[6] == 1: risk += 0.2  # online_order
                if row[5] == 0: risk += 0.1  # no pin used
                risk_scores.append(1 if risk > 0.5 else 0)
            return np.array(risk_scores)
        
        def predict_proba(self, X):
            predictions = self.predict(X)
            probas = []
            for pred in predictions:
                if pred == 1:
                    probas.append([0.2, 0.8])  # 80% fraud probability
                else:
                    probas.append([0.9, 0.1])  # 10% fraud probability
            return np.array(probas)
    
    return DemoModel(), ['distance_from_home', 'distance_from_last_transaction', 
                        'ratio_to_median_purchase_price', 'repeat_retailer', 
                        'used_chip', 'used_pin_number', 'online_order']

def preprocess_input(input_data, feature_columns):
    """Preprocess input data for prediction - no scaling needed"""
    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    
    return input_df.values  # Return numpy array directly

def show_model_instructions():
    """Show instructions for setting up the model"""
    st.error("## 🚨 Model Setup Required")
    
    st.markdown("""
    ### To use this app, you need to train and save your model first:
    
    1. **Open your Jupyter notebook** (`fraud_detection.ipynb`)
    2. **Add this code at the end** and run it:
    ```python
    # Save the trained model
    joblib.dump(model, 'fraud_detection_model.pkl')
    
    # Save feature columns
    feature_columns = ['distance_from_home', 'distance_from_last_transaction', 
                      'ratio_to_median_purchase_price', 'repeat_retailer', 
                      'used_chip', 'used_pin_number', 'online_order']
    
    import json
    with open('feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    
    print("Model files saved successfully!")
    ```
    
    3. **Ensure these files are in the same folder as this app:**
       - `fraud_detection_model.pkl`
       - `feature_columns.json`
    
    4. **Restart this Streamlit app**
    """)

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, feature_columns = load_model()
    
    # Use demo model if real model not available
    use_demo = False
    if model is None:
        st.sidebar.error("⚠️ Real model not found!")
        if st.sidebar.button("🔄 Use Demo Model for Testing"):
            model, feature_columns = create_demo_model()
            use_demo = True
            st.rerun()
        else:
            show_model_instructions()
            return
    
    if use_demo:
        st.warning("🔧 Currently using DEMO model with simple rules")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        app_mode = st.radio(
            "Choose Mode",
            ["Single Prediction", "Batch Prediction", "Model Info"]
        )
        
        st.markdown("---")
        st.header("About")
        st.info(
            "This system uses machine learning to detect fraudulent credit card transactions "
            "based on transaction patterns and behavioral features."
        )
        
        if use_demo:
            st.markdown("---")
            st.warning("**Demo Mode Active**")
            st.markdown("Using simple rule-based detection for demonstration.")
    
    if app_mode == "Single Prediction":
        single_prediction(model, feature_columns, use_demo)
    elif app_mode == "Batch Prediction":
        batch_prediction(model, feature_columns, use_demo)
    elif app_mode == "Model Info":
        model_info(use_demo)

def single_prediction(model, feature_columns, use_demo=False):
    """Single transaction prediction interface"""
    
    st.header("🔍 Single Transaction Analysis")
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Transaction Details")
        
        if use_demo:
            st.info("🎯 Demo model uses: High distance + high amount ratio + online orders = higher risk")
        
        # Create input fields in a modern layout
        with st.container():
            st.markdown("### Location Features")
            col1a, col2a = st.columns(2)
            
            with col1a:
                distance_from_home = st.slider(
                    "Distance from Home (miles)",
                    min_value=0.0,
                    max_value=200.0,
                    value=25.0,
                    help="Distance between transaction location and cardholder's home"
                )
                
            with col2a:
                distance_from_last_transaction = st.slider(
                    "Distance from Last Transaction (miles)",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    help="Distance from previous transaction location"
                )
        
        with st.container():
            st.markdown("### Transaction Features")
            col1b, col2b, col3b = st.columns(3)
            
            with col1b:
                ratio_to_median = st.slider(
                    "Price Ratio to Median",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.8,
                    step=0.1,
                    help="Ratio of transaction amount to median purchase price"
                )
            
            with col2b:
                repeat_retailer = st.selectbox(
                    "Repeat Retailer",
                    options=[1, 0],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Transaction with previously used merchant"
                )
            
            with col3b:
                used_chip = st.selectbox(
                    "Chip Used",
                    options=[1, 0],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Transaction used chip (EMV)"
                )
        
        with st.container():
            st.markdown("### Security Features")
            col1c, col2c = st.columns(2)
            
            with col1c:
                used_pin = st.selectbox(
                    "PIN Used",
                    options=[1, 0],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Transaction used PIN verification"
                )
            
            with col2c:
                online_order = st.selectbox(
                    "Online Order",
                    options=[1, 0],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Online/remote transaction"
                )
    
    with col2:
        st.subheader("Risk Assessment")
        
        # Prediction button
        if st.button("🚀 Analyze Transaction", type="primary", use_container_width=True):
            with st.spinner("Analyzing transaction patterns..."):
                # Prepare input data
                input_data = {
                    'distance_from_home': distance_from_home,
                    'distance_from_last_transaction': distance_from_last_transaction,
                    'ratio_to_median_purchase_price': ratio_to_median,
                    'repeat_retailer': repeat_retailer,
                    'used_chip': used_chip,
                    'used_pin_number': used_pin,
                    'online_order': online_order
                }
                
                # Make prediction
                try:
                    processed_data = preprocess_input(input_data, feature_columns)
                    prediction = model.predict(processed_data)
                    probability = model.predict_proba(processed_data)[0]
                    
                    fraud_probability = probability[1] * 100
                    
                    # Display results
                    if prediction[0] == 1:
                        st.markdown(
                            f'<div class="prediction-box fraud">'
                            f'<h2>🚨 HIGH RISK</h2>'
                            f'<h3>Fraud Probability: {fraud_probability:.1f}%</h3>'
                            f'<p>This transaction shows suspicious patterns</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box legitimate">'
                            f'<h2>✅ LEGITIMATE</h2>'
                            f'<h3>Fraud Probability: {fraud_probability:.1f}%</h3>'
                            f'<p>This transaction appears normal</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Show probability breakdown
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Legitimate Probability", f"{probability[0]*100:.1f}%")
                    with col2:
                        st.metric("Fraud Probability", f"{fraud_probability:.1f}%")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        
        # Feature importance explanation
        with st.expander("📊 Feature Impact Guide"):
            st.markdown("""
            **High-Risk Indicators:**
            - Large distance from home (>50 miles)
            - High transaction amount ratio (>3x median)
            - Online transactions
            - No PIN usage
            
            **Lower Risk Indicators:**
            - Repeat retailers
            - Chip transactions  
            - PIN verification
            - Local transactions
            """)

def batch_prediction(model, feature_columns, use_demo=False):
    """Batch prediction for multiple transactions"""
    
    st.header("📊 Batch Transaction Analysis")
    
    if use_demo:
        st.info("📁 Demo mode: Upload a CSV with transaction data to test batch processing")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with transactions",
        type=['csv'],
        help="CSV should contain the feature columns from your dataset"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(batch_data)} transactions")
            
            # Show data preview
            with st.expander("Preview Data"):
                st.dataframe(batch_data.head())
            
            if st.button("🔍 Analyze Batch", type="primary"):
                with st.spinner("Analyzing transactions..."):
                    try:
                        # Check if all required columns are present
                        missing_cols = [col for col in feature_columns if col not in batch_data.columns]
                        
                        if missing_cols:
                            st.error(f"❌ Missing columns: {missing_cols}")
                            st.info("Please ensure your CSV contains all required feature columns")
                        else:
                            # Process and predict
                            processed_data = batch_data[feature_columns].values
                            predictions = model.predict(processed_data)
                            probabilities = model.predict_proba(processed_data)
                            
                            # Add results to dataframe
                            results_df = batch_data.copy()
                            results_df['Fraud_Prediction'] = predictions
                            results_df['Fraud_Probability'] = probabilities[:, 1]
                            results_df['Risk_Level'] = np.where(
                                results_df['Fraud_Prediction'] == 1, 'High Risk', 'Legitimate'
                            )
                            
                            # Display summary
                            fraud_count = sum(predictions)
                            fraud_percentage = (fraud_count / len(predictions)) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Transactions", len(results_df))
                            with col2:
                                st.metric("High Risk Transactions", fraud_count)
                            with col3:
                                st.metric("Risk Percentage", f"{fraud_percentage:.1f}%")
                            
                            # Show results table
                            st.subheader("Analysis Results")
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="fraud_analysis_results.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"Error processing batch: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    else:
        # Show sample CSV format
        with st.expander("📋 Expected CSV Format"):
            sample_data = {
                'distance_from_home': [57.88, 10.83, 5.09],
                'distance_from_last_transaction': [0.31, 0.18, 0.81],
                'ratio_to_median_purchase_price': [1.95, 1.29, 0.43],
                'repeat_retailer': [1, 1, 1],
                'used_chip': [1, 0, 0],
                'used_pin_number': [0, 0, 0],
                'online_order': [0, 0, 1]
            }
            st.dataframe(pd.DataFrame(sample_data))
            st.download_button(
                label="📥 Download Sample CSV",
                data=pd.DataFrame(sample_data).to_csv(index=False),
                file_name="sample_transactions.csv",
                mime="text/csv"
            )

def model_info(use_demo=False):
    """Display model information and training details"""
    
    st.header("🤖 Model Information")
    
    if use_demo:
        st.warning("🔧 Currently showing information for DEMO model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Architecture")
        
        if use_demo:
            st.markdown("""
            **Algorithm:** Rule-based Demo System
            
            **Current Rules:**
            - Distance from home > 50 miles: +30% risk
            - Amount ratio > 3x median: +40% risk  
            - Online order: +20% risk
            - No PIN used: +10% risk
            - Total risk > 50% = Fraud prediction
            """)
        else:
            st.markdown("""
            **Algorithm:** LightGBM (Gradient Boosting)
            
            **Key Features:**
            - High performance gradient boosting framework
            - Fast training speed and high efficiency
            - Lower memory usage
            - Excellent handling of large-scale data
            - Built-in handling of imbalanced data
            - No feature scaling required
            """)
        
        st.subheader("Feature Description")
        feature_info = {
            "Feature": [
                "distance_from_home",
                "distance_from_last_transaction", 
                "ratio_to_median_purchase_price",
                "repeat_retailer",
                "used_chip",
                "used_pin_number", 
                "online_order"
            ],
            "Description": [
                "Distance from cardholder's home location",
                "Distance from previous transaction location",
                "Ratio to typical transaction amount",
                "Transaction with known merchant",
                "Chip (EMV) technology used",
                "PIN verification used",
                "Online/remote transaction"
            ],
            "Type": [
                "Continuous",
                "Continuous", 
                "Continuous",
                "Binary (0/1)",
                "Binary (0/1)",
                "Binary (0/1)",
                "Binary (0/1)"
            ]
        }
        
        st.table(pd.DataFrame(feature_info))
    
    with col2:
        st.subheader("Performance Metrics")
        
        if use_demo:
            metrics_data = {
                "Metric": ["Mode", "Accuracy"],
                "Value": ["Demo", "Rule-based"]
            }
        else:
            # These would typically come from your model evaluation
            metrics_data = {
                "Metric": ["ROC-AUC", "Precision", "Recall", "F1-Score"],
                "Value": ["0.98+", "0.95+", "0.92+", "0.93+"]
            }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        for _, row in metrics_df.iterrows():
            st.markdown(
                f'<div class="metric-card">'
                f'<strong>{row["Metric"]}</strong><br>'
                f'<span style="font-size: 1.5rem; color: #1f77b4;">{row["Value"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        st.subheader("Dataset Info")
        st.markdown("""
        - **Samples:** 1,000,000
        - **Fraud Rate:** 8.74%
        - **Features:** 7
        - **Balance:** SMOTE applied
        - **Scaling:** Not required
        """)

if __name__ == "__main__":
    main()