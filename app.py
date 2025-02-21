import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics


st.set_page_config(
    page_title="Feature Scoring Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_synthetic_data(num_records, feature_configs):
    """Generate synthetic data based on feature configurations."""
    data = {'ID': list(range(1, num_records + 1))}
    
    for feature, config in feature_configs.items():
        if config['enable']:
            min_val, max_val = config['range']
            data[feature] = np.random.uniform(min_val, max_val, num_records).round(2)
    
    return pd.DataFrame(data)

def calculate_scores(df, feature_configs):
    """Calculate weighted scores based on selected features."""
    
    
    enabled_features = [f for f, config in feature_configs.items() if config['enable']]
    
    if not enabled_features:
        st.error("Please select at least one feature to calculate scores.")
        return None
    
    
    num_features = len(enabled_features)
    weight = round(1.0 / num_features, 2)
    
    
    total_weight = weight * num_features
    if total_weight != 1.0:
        st.error(f"The weights do not sum to 1.0 (current sum: {total_weight}). Please adjust feature selection.")
        return None
    
   
    df['Score'] = 0
    for feature in enabled_features:
        # Normalize feature values to 0-1 range
        min_val, max_val = feature_configs[feature]['range']
        range_size = max_val - min_val
        if range_size > 0:
            normalized = (df[feature] - min_val) / range_size
            df['Score'] += normalized * weight
        else:
            st.warning(f"Range for {feature} is zero. This feature will not contribute to the score.")
    
    df['Score'] = df['Score'].round(4)
    return df

def main():
    st.title("Feature Scoring Application")
    
    
    st.sidebar.header("Data Generation Settings")
    
    num_records = st.sidebar.slider("Number of Records", 5, 1000, 100)
    
   
    st.sidebar.header("Feature Selection")
    
    feature_configs = {
        'Electricity': {'enable': st.sidebar.checkbox("Electricity", value=True), 
                       'range': st.sidebar.slider("Electricity Range (kWh)", 0, 10000, (100, 5000))},
        'Water': {'enable': st.sidebar.checkbox("Water", value=True), 
                 'range': st.sidebar.slider("Water Range (litres)", 0, 10000, (100, 3000))},
        'Gas': {'enable': st.sidebar.checkbox("Gas", value=False), 
               'range': st.sidebar.slider("Gas Range (kgs)", 0, 1000, (10, 500))},
        'Waste': {'enable': st.sidebar.checkbox("Waste", value=False), 
                 'range': st.sidebar.slider("Waste Range (kgs)", 0, 5000, (50, 1000))}
    }
    
    
    if st.sidebar.button("Generate Data"):
       
        df = generate_synthetic_data(num_records, feature_configs)
        
        
        scored_df = calculate_scores(df, feature_configs)
        
        if scored_df is not None:
            
            leaderboard = scored_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
            
            
            st.header("Leaderboard")
            st.dataframe(leaderboard, height=400)
            
           
            csv = leaderboard.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="generated_data.csv",
                mime="text/csv"
            )
            
            
            st.header("Score Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Score", f"{leaderboard['Score'].mean():.4f}")
            with col2:
                st.metric("Median Score", f"{leaderboard['Score'].median():.4f}")
            with col3:
                st.metric("Standard Deviation", f"{leaderboard['Score'].std():.4f}")
            
            
            st.header("Score Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(leaderboard['Score'], bins=20, alpha=0.7)
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Scores')
            st.pyplot(fig)
            
    
    st.sidebar.header("Or Upload Your Own Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.header("Uploaded Data")
            st.dataframe(user_df, height=300)
            
            # Check if required features exist
            missing_features = [f for f, config in feature_configs.items() 
                               if config['enable'] and f not in user_df.columns]
            
            if missing_features:
                st.error(f"Missing required features in uploaded data: {', '.join(missing_features)}")
            else:
                # Calculate scores for uploaded data
                scored_user_df = calculate_scores(user_df.copy(), feature_configs)
                
                if scored_user_df is not None:
                    # Sort by score in descending order
                    user_leaderboard = scored_user_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
                    
                    # Display leaderboard
                    st.header("Leaderboard (Uploaded Data)")
                    st.dataframe(user_leaderboard, height=400)
                    
                    # Download button
                    csv = user_leaderboard.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="user_leaderboard.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

if __name__ == "__main__":
    main()