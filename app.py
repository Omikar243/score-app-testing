import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
import json
from io import BytesIO
from fpdf import FPDF


# Page configuration and styling
st.set_page_config(page_title="Resource Sustainability Dashboard", layout="wide")
st.markdown("""
    <style>
    .dataframe-container {
        display: block;
        max-height: 500px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }
    table { width: 100%; border-collapse: collapse; }
    table, th, td { border: 1px solid #ddd; }
    th, td { padding: 8px; text-align: center; }
    th {
        background-color: #f2f2f2;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    </style>
""", unsafe_allow_html=True)


INDIAN_STATES_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", 
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", 
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", 
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]


def normalize_series(series, new_min=1, new_max=1000, inverse=True):
    if series.max() == series.min():
        return pd.Series(new_min if inverse else new_max, index=series.index)
    if inverse:
        norm = (series.max() - series) / (series.max() - series.min())
    else:
        norm = (series - series.min()) / (series.max() - series.min())
    return norm * (new_max - new_min) + new_min

def generate_feature_data(n, dist_type, params):
    if dist_type == "Uniform":
        low, high = int(params['min']), int(params['max'])
        data = np.random.randint(low, high + 1, size=n)
    elif dist_type == "Normal":
        data = np.random.normal(params['mean'], params['std'], size=n)
    elif dist_type == "Poisson":
        data = np.random.poisson(params['lambda'], size=n)
    elif dist_type == "Exponential":
        data = np.random.exponential(params['scale'], size=n)
    elif dist_type == "Binomial":
        data = np.random.binomial(int(params['n_trials']), params['p'], size=n)
    elif dist_type == "Lognormal":
        data = np.random.lognormal(params['mean'], params['sigma'], size=n)
    else:
        low, high = int(params['min']), int(params['max'])
        data = np.random.randint(low, high + 1, size=n)
    data = np.clip(data, params['min'], params['max'])
    return np.round(data).astype(int)

def generate_synthetic_data(n, feature_settings, electricity_data=None):
    """Generate synthetic data but use real electricity data where available"""
    data = {}
    

    for feat, (dtype, params) in feature_settings.items():
        data[feat] = generate_feature_data(n, dtype, params)
    
    
    df = pd.DataFrame(data)
    df.insert(0, "ID", range(1, n + 1))
    
    
    if electricity_data is not None:
       
        if len(electricity_data) < n:
            sampled_data = electricity_data.sample(n=n, replace=True)
        else:
            sampled_data = electricity_data.sample(n=n, replace=False)
        
       
        df["Electricity"] = sampled_data["qty_usage_in_1month"].values
        df["MPCE"] = sampled_data["mpce"].values
        df["State_UT"] = sampled_data["state_name"].values
    else:
       
        df["State_UT"] = np.random.choice(INDIAN_STATES_UTS, size=n)
    
    return df

def compute_feature_stats(df, by_state=False):
    stats = {}
    
    if by_state:
     
        for state in df["State_UT"].unique():
            state_df = df[df["State_UT"] == state]
            stats[state] = {}
            for col in df.columns:
                if col not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Sustainability_Score",
                              "Company_Name", "Sector classification", "Environment Score", 
                              "ESG Rating", "Category", "Date of Rating", "Total Employees", "State_UT"]:
                    if pd.api.types.is_numeric_dtype(state_df[col]):
                        stats[state][col] = {
                            'mean': float(state_df[col].mean()),
                            'median': float(state_df[col].median()),
                            'std': float(state_df[col].std()),
                            'min': float(state_df[col].min()),
                            'max': float(state_df[col].max())
                        }
    else:
        
        for col in df.columns:
            if col not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Sustainability_Score",
                          "Company_Name", "Sector classification", "Environment Score", 
                          "ESG Rating", "Category", "Date of Rating", "Total Employees", "State_UT"]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats[col] = {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
    return stats

def compute_weighted_score(raw_df, weights, inverse=True, state_specific=False):
    """
    Scoring methodology with state-specific comparisons:
    1. For each feature x_i, compute the z-score: z_i = (x_i - mean) / std.
       - If state_specific is True, use state-specific means and standard deviations
    2. Cap each z_i between -1 and 1.
    3. Compute S1 = sum (w_i * z_i).
    4. Transform S1 to final score S using tanh transformation.
    5. Rank individuals based on S.
    """
    df = raw_df.copy()
    norm_vals = {}
    z_vals = {}
    
    
    weight_columns = [col for col in df.columns if any(wcol in col for wcol in weights.keys())]
    
    
    for col in df.columns:
        weight_key = None
        
        for key in weights.keys():
            if key in col or key.replace('_', '') in col:
                weight_key = key
                break
        
        if weight_key and pd.api.types.is_numeric_dtype(df[col]) and col not in ["ID", "Weighted_Score", "Rank", "Z_Score", 
                           "Sustainability_Score", "Weighted_Z_Score"]:
            
            if state_specific and "State_UT" in df.columns:
                
                z_vals[col] = pd.Series(np.nan, index=df.index)
                norm_vals[col] = pd.Series(np.nan, index=df.index)
                
                for state in df["State_UT"].unique():
                    state_mask = df["State_UT"] == state
                    state_df = df[state_mask]
                    
                   
                    if len(state_df) > 0 and not state_df[col].isna().all():
                        
                        if ("baseline_values_by_state" in st.session_state and 
                            state in st.session_state.baseline_values_by_state and 
                            col in st.session_state.baseline_values_by_state[state]):
                            base_mean, base_std = st.session_state.baseline_values_by_state[state][col]
                            mean_val, std_val = base_mean, base_std
                        else:
                            mean_val = state_df[col].mean()
                            std_val = state_df[col].std(ddof=1) if len(state_df) > 1 else 1.0
                        
                        
                        if pd.isna(std_val) or std_val == 0:
                            z = pd.Series(0, index=state_df.index)
                        else:
                            z = (state_df[col] - mean_val) / std_val
                        
                        
                        if inverse:
                            z = -z
                        
                        
                        z = np.clip(z, -1, 1)
                        z_vals[col].loc[state_mask] = z.values
                        
                        
                        cmin, cmax = state_df[col].min(), state_df[col].max()
                        if cmax == cmin:
                            norm = pd.Series(1 if inverse else 1000, index=state_df.index)
                        else:
                            if inverse:
                                norm = ((cmax - state_df[col]) / (cmax - cmin)) * 999 + 1
                            else:
                                norm = ((state_df[col] - cmin) / (cmax - cmin)) * 999 + 1
                        norm_vals[col].loc[state_mask] = norm.values
                
                
                df[f"{col}_z_score"] = z_vals[col]
                df[f"{col}_normalized"] = norm_vals[col]
                
            else:
                
                if "baseline_values" in st.session_state and col in st.session_state.baseline_values:
                    base_mean, base_std = st.session_state.baseline_values[col]
                    mean_val, std_val = base_mean, base_std
                else:
                    mean_val = df[col].mean()
                    std_val = df[col].std(ddof=1) if len(df) > 1 else 1.0
                
                
                if pd.isna(std_val) or std_val == 0:
                    z = pd.Series(0, index=df.index)
                else:
                    z = (df[col] - mean_val) / std_val
                
                
                if inverse:
                    z = -z
                    
               
                z = np.clip(z, -1, 1)
                
                z_vals[col] = z
                df[f"{col}_z_score"] = z
                
                
                cmin, cmax = df[col].min(), df[col].max()
                if cmax == cmin:
                    norm = pd.Series(1 if inverse else 1000, index=df.index)
                else:
                    if inverse:
                        norm = ((cmax - df[col]) / (cmax - cmin)) * 999 + 1
                    else:
                        norm = ((df[col] - cmin) / (cmax - cmin)) * 999 + 1
                norm_vals[col] = norm
                df[f"{col}_normalized"] = norm


    weighted_z_sum = pd.Series(0, index=df.index)
    weighted_norm_sum = pd.Series(0, index=df.index)
    
    
    column_to_weight = {}
    for col in z_vals.keys():
        for weight_key in weights.keys():
            if weight_key in col or weight_key.replace('_', '') in col:
                column_to_weight[col] = weight_key
                break
    
   
    for col, weight_key in column_to_weight.items():
        if weight_key in weights:
            weight = weights[weight_key]
            weighted_z_sum += z_vals[col] * weight
            weighted_norm_sum += norm_vals[col] * weight
    
    
    if weighted_z_sum.sum() == 0 and not column_to_weight:
        for weight_key, weight in weights.items():
            if weight_key in z_vals:
                weighted_z_sum += z_vals[weight_key] * weight
            if weight_key in norm_vals:
                weighted_norm_sum += norm_vals[weight_key] * weight
    
    df['Z_Score'] = weighted_z_sum
    df['Weighted_Z_Score'] = weighted_z_sum  
    
    
    df['Sustainability_Score'] = 500 * (1 - np.tanh(df['Z_Score']/2.5))
    
    
    df['Weighted_Score'] = weighted_norm_sum
    
    
    df['Z_Score'] = df['Z_Score'].fillna(0.5)
    df['Weighted_Z_Score'] = df['Weighted_Z_Score'].fillna(0.5)
    df['Sustainability_Score'] = df['Sustainability_Score'].fillna(500)
    df['Weighted_Score'] = df['Weighted_Score'].fillna(df['Weighted_Score'].mean())
    

    df['Rank'] = df['Sustainability_Score'].rank(method='min', ascending=False).astype(int)
    df.sort_values('Rank', inplace=True)

    
    if 'Electricity' in raw_df.columns and 'MPCE' in raw_df.columns:
        
        mpce_safe = raw_df['MPCE'].replace(0, np.nan)
        ratio = raw_df['Electricity'] / mpce_safe
        if ratio.isna().all():
            df['electricity_z_score'] = 0  
        else:
            mean_ratio = ratio.mean()
            std_ratio = ratio.std(ddof=1) if len(ratio.dropna()) > 1 else 1.0
            
            if pd.isna(std_ratio) or std_ratio == 0:
                df['electricity_z_score'] = 0
            else:
                df['electricity_z_score'] = (ratio - mean_ratio) / std_ratio
                df['electricity_z_score'] = df['electricity_z_score'].fillna(0)  

    return df, column_to_weight  

def feature_distribution_ui(feature_name, default_min=1, default_max=1000):
    dist_type = st.selectbox(
        f"Distribution for {feature_name}",
        ["Uniform","Normal","Poisson","Exponential","Binomial","Lognormal"],
        key=f"dist_{feature_name}"
    )
    params = {}
    if dist_type == "Uniform":
        cols = st.columns(2)
        with cols[0]:
            params['min'] = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        with cols[1]:
            params['max'] = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Normal":
        cols = st.columns(4)
        with cols[0]:
            params['mean'] = st.number_input(f"{feature_name} Mean", value=(default_min+default_max)//2, key=f"{feature_name}_mean")
        with cols[1]:
            params['std'] = st.number_input(f"{feature_name} Std Dev", value=(default_max-default_min)//4, key=f"{feature_name}_std")
        with cols[2]:
            params['min'] = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        with cols[3]:
            params['max'] = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Poisson":
        cols = st.columns(3)
        with cols[0]:
            params['lambda'] = st.number_input(f"{feature_name} Lambda", value=(default_min+default_max)//2, key=f"{feature_name}_lambda")
        with cols[1]:
            params['min'] = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        with cols[2]:
            params['max'] = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Exponential":
        cols = st.columns(3)
        with cols[0]:
            params['scale'] = st.number_input(f"{feature_name} Scale", value=(default_max-default_min)//2, key=f"{feature_name}_scale")
        with cols[1]:
            params['min'] = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        with cols[2]:
            params['max'] = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Binomial":
        cols = st.columns(4)
        with cols[0]:
            params['n_trials'] = st.number_input(f"{feature_name} Trials", value=10, key=f"{feature_name}_n_trials")
        with cols[1]:
            params['p'] = st.number_input(f"{feature_name} p", min_value=0.0, max_value=1.0, value=0.5, key=f"{feature_name}_p")
        with cols[2]:
            params['min'] = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        with cols[3]:
            params['max'] = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Lognormal":
        cols = st.columns(4)
        with cols[0]:
            params['mean'] = st.number_input(f"{feature_name} Mean(log)", value=1.0, key=f"{feature_name}_mean")
        with cols[1]:
            params['sigma'] = st.number_input(f"{feature_name} Sigma(log)", value=0.5, key=f"{feature_name}_sigma")
        with cols[2]:
            params['min'] = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        with cols[3]:
            params['max'] = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    return dist_type, params

def load_company_data():
    """
    Preload company data from a fixed CSV path.
    """
    if 'company_data' not in st.session_state:
        try:
            df = pd.read_csv("employees_2024_25_updated.csv")
        except FileNotFoundError:
            st.error(
                "Could not find company_data.csv in the app directory.\n"
                "Please download the template and populate it as needed."
            )
            return None

        needed = [
            "Company_Name",
            "Sector_classification",
            "Environment_Score",
            "ESG_Rating",
            "Category",
            "Total_Employees"
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            st.error(f"Missing columns in company_data.csv: {', '.join(missing)}")
            return None

        st.session_state.company_data = df

    return st.session_state.company_data

def load_real_electricity_data():
    """Load electricity consumption data from file path"""
    try:
        
        file_path = "electricity_data.xlsx"
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            file_path = "electricity_data_with_mpce.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                st.error("No electricity data file found. Please make sure electricity_data.csv or electricity_data.xlsx exists.")
                return None
        
        
        needed = ["hhid", "state_name", "sector", "qty_usage_in_1month", "mpce"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            st.error(f"Missing columns in electricity data: {', '.join(missing)}")
            return None
        
        
        unique_states = df["state_name"].unique()
        print(f"Unique state names in data file: {unique_states}")
        

        unrecognized_states = [s for s in unique_states if s not in INDIAN_STATES_UTS]
        if unrecognized_states:
            print(f"Warning: Unrecognized state names: {unrecognized_states}")
            
            state_name_mapping = {
                # Add mappings for any misspelled or differently formatted names
                # Example: "Delhi (NCT)": "Delhi",
                # "Andaman & Nicobar": "Andaman and Nicobar Islands",
            }
            df["state_name"] = df["state_name"].replace(state_name_mapping)
        
       
        state_stats = {}
        for state in df["state_name"].unique():
            state_df = df[df["state_name"] == state]
            if len(state_df) > 0:
                state_stats[state] = {
                    "Electricity": (
                        state_df["qty_usage_in_1month"].mean(),
                        state_df["qty_usage_in_1month"].std()
                    ),
                    "MPCE": (
                        state_df["mpce"].mean(),
                        state_df["mpce"].std()
                    )
                }
        
        st.session_state.baseline_values_by_state = state_stats
        
       
        st.session_state.electricity_data = df
        return df
    
    except Exception as e:
        st.error(f"Error loading electricity data: {str(e)}")
        return None

def render_sortable_table(df):
    st.dataframe(df, use_container_width=True)


st.title("Resource Sustainability Consumption Dashboard")
st.markdown("Analyze your resource consumption and sustainability score")


st.markdown("## Distribution Settings")


n = st.number_input("Number of Records", min_value=1, value=100, step=1)


tab1, tab2, tab3 = st.tabs(["Water", "Public Transport", "Private Transport"])

with tab1:
    st.markdown("### Water Consumption")
    dist_water, params_water = feature_distribution_ui("Water (in Litres)", default_min=200, default_max=600)

with tab2:
    st.markdown("### Public Transport (in Kms)")
    include_public = st.checkbox("Include Public Transport", value=True)
    if include_public:
        public_dist, public_params = feature_distribution_ui("Public_Transport", default_min=0, default_max=500)
    else:
        public_dist, public_params = None, None

with tab3:
    st.markdown("### Private Transport (in Kms)")
    include_private = st.checkbox("Include Private Transport", value=True)
    if include_private:
        private_dist, private_params = feature_distribution_ui("Private_Transport", default_min=0, default_max=300)
    else:
        private_dist, private_params = None, None


feature_settings = {
    "Water": (dist_water, params_water)
}
if include_public and public_dist is not None:
    feature_settings["Public_Transport"] = (public_dist, public_params)
if include_private and private_dist is not None:
    feature_settings["Private_Transport"] = (private_dist, private_params)


if st.button("Generate Analysis Data", type="primary"):
    if 'electricity_data' not in st.session_state:
        electricity_data = load_real_electricity_data()
    else:
        electricity_data = st.session_state.electricity_data
    
    if electricity_data is not None:
        if 'full_electricity_data' not in st.session_state:
            st.session_state.full_electricity_data = electricity_data
        
        raw = generate_synthetic_data(n, feature_settings, st.session_state.full_electricity_data)
        
        st.session_state.synth_data_raw = raw
        st.session_state.feature_settings = feature_settings
        
        st.session_state.feature_constraints = {
            feat: (p['min'], p['max']) for feat, (_, p) in feature_settings.items()
        }
        
        full_data = st.session_state.full_electricity_data
        st.session_state.feature_constraints["Electricity"] = (
            full_data["qty_usage_in_1month"].min(),
            full_data["qty_usage_in_1month"].max()
        )
        
        st.session_state.feature_constraints["MPCE"] = (
            full_data["mpce"].min(),
            full_data["mpce"].max()
        )
        
        st.session_state.feature_stats = compute_feature_stats(raw)
        st.session_state.feature_stats_by_state = compute_feature_stats(raw, by_state=True)
        
        st.success("Analysis data generated!")
    else:
        st.error("No electricity data available. Please check your data file.")

if "synth_data_raw" in st.session_state:
    st.markdown("## Synthetic Data Preview")
    
    synthetic_cols = ["ID"]
    if "Water" in st.session_state.synth_data_raw.columns:
        synthetic_cols.append("Water")
    if "Public_Transport" in st.session_state.synth_data_raw.columns:
        synthetic_cols.append("Public_Transport")
    if "Private_Transport" in st.session_state.synth_data_raw.columns:
        synthetic_cols.append("Private_Transport")
    
    if len(synthetic_cols) > 1:  # More than just ID
        render_sortable_table(st.session_state.synth_data_raw[synthetic_cols])
    else:
        st.info("No synthetic data features selected for preview.")

    

st.markdown("---")

st.markdown("## Company Data")
company_data = load_company_data()
if company_data is not None:
    st.success(f"Loaded {len(company_data)} companies")
    st.dataframe(company_data, use_container_width=True)
else:
    st.info("Please upload company data CSV")
    template = pd.DataFrame({
        "Company_Name":["A","B","C"],
        "Sector_classification":["Tech","Manu","Health"],
        "Environment_Score":[70,60,80],
        "ESG_Rating":["A","B","A-"],
        "Category":["Leader","Average","Leader"],
        "Total_Employees":[100,200,150]
    })
    st.download_button("Download Company Template",
                       data=template.to_csv(index=False).encode('utf-8'),
                       file_name="company_template.csv",
                       mime="text/csv")

st.markdown("## Electricity Consumption Data")
if 'electricity_data' not in st.session_state:
    electricity_data = load_real_electricity_data()
    if electricity_data is not None:
        st.success(f"Loaded {len(electricity_data)} electricity records from data file")
else:
    electricity_data = st.session_state.electricity_data

if electricity_data is not None and 'full_electricity_data' not in st.session_state:
    st.session_state.full_electricity_data = electricity_data

if electricity_data is not None:
    states_count = electricity_data['state_name'].nunique()
    unique_states = sorted(electricity_data['state_name'].unique().tolist())
    
    states = [s for s in unique_states if s in INDIAN_STATES_UTS[:28]]  # First 28 are states
    uts = [s for s in unique_states if s in INDIAN_STATES_UTS[28:]]  # Rest are UTs
    
    st.success(f"Loaded electricity data with {len(electricity_data)} records across {len(states)} states and {len(uts)} union territories")
    
    state_options = sorted(electricity_data['state_name'].unique().tolist())
    selected_state = st.selectbox("Select State/UT", state_options, key="dashboard_state")
    
    sector_options = [('Rural', 1), ('Urban', 2)]
    selected_sector_name, selected_sector = sector_options[0]
    col1, col2 = st.columns(2)
    with col1:
        selected_sector_name = st.radio("Select Sector", ['Rural', 'Urban'], key="dashboard_sector")
        selected_sector = 1 if selected_sector_name == 'Rural' else 2
    
    filtered_data = electricity_data[
        (electricity_data['state_name'] == selected_state) & 
        (electricity_data['sector'] == selected_sector)
    ]
    
    with col2:
        if not filtered_data.empty:
            avg_electricity = filtered_data['qty_usage_in_1month'].mean()
            avg_mpce = filtered_data['mpce'].mean()
            st.metric("Average Electricity", f"{avg_electricity:.2f} kWh")
            st.metric("Average MPCE", f"₹{avg_mpce:.2f}")
    
    st.markdown(f"## Electricity Distribution - {selected_state}, {selected_sector_name}")
if not filtered_data.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_data["qty_usage_in_1month"], bins=30, kde=True, ax=ax)
    ax.set_title(f"Electricity Consumption Distribution - {selected_state}, {selected_sector_name}")
    ax.set_xlabel("Electricity (kWh/month)")
    st.pyplot(fig)
else:
    st.warning(f"No data available for {selected_state} ({selected_sector_name})")

# New addition: Overall electricity consumption comparison across states
st.markdown("## Overall Electricity Consumption Data")
sector_for_comparison = st.radio("Select Sector for Comparison", ['Rural', 'Urban', 'Both'], key="comparison_sector")

# Calculate average electricity consumption by state and sector
if sector_for_comparison == 'Both':
    # Group by state and calculate mean for both sectors combined
    state_avg = electricity_data.groupby('state_name')['qty_usage_in_1month'].mean().reset_index()
    chart_title = "Average Electricity Consumption by State (Rural & Urban)"
else:
    # Filter by selected sector and then group by state
    sector_value = 1 if sector_for_comparison == 'Rural' else 2
    filtered_comparison = electricity_data[electricity_data['sector'] == sector_value]
    state_avg = filtered_comparison.groupby('state_name')['qty_usage_in_1month'].mean().reset_index()
    chart_title = f"Average Electricity Consumption by State ({sector_for_comparison})"

# Sort by average consumption for better visualization
state_avg = state_avg.sort_values('qty_usage_in_1month', ascending=False)

# Create the comparison chart
fig = px.bar(state_avg, x='state_name', y='qty_usage_in_1month',
             title=chart_title,
             labels={
                 'state_name': 'State/UT',
                 'qty_usage_in_1month': 'Average Electricity (kWh/month)'
             })

# Customize layout
fig.update_layout(
    xaxis_tickangle=90,
    xaxis_title='State/UT',
    yaxis_title='Average Electricity (kWh/month)'
)

# Add value labels on top of bars
fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')

st.plotly_chart(fig)

# New addition: Total electricity usage by sectors across India
st.markdown("## Total Electricity Usage by Sectors and States Across India")

# Tab layout for sector and state views
sector_state_tab1, sector_state_tab2 = st.tabs(["By Sector", "By State"])

# Calculate the sum of electricity usage by sector
sector_total = electricity_data.groupby('sector')['qty_usage_in_1month'].sum().reset_index()
# Map sector codes to names
sector_total['sector_name'] = sector_total['sector'].map({1: 'Rural', 2: 'Urban'})

# Create the pie chart for sector distribution
fig_pie = px.pie(
    sector_total, 
    values='qty_usage_in_1month', 
    names='sector_name',
    title="Total Electricity Usage Distribution by Sector",
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Set2
)

# Update to show only percentage in labels (removed value)
fig_pie.update_traces(
    textinfo='percent', 
    textposition='inside',
    hovertemplate='<b>%{label}</b><br>Total Electricity: %{value:.1f} kWh<br>Percentage: %{percent}'
)

# Show the total sum in the center of the donut
total_usage = sector_total['qty_usage_in_1month'].sum()
fig_pie.update_layout(
    annotations=[dict(
        text=f'Total<br>{total_usage:.1f} kWh',
        x=0.5, y=0.5,
        font_size=14,
        showarrow=False
    )]
)

st.plotly_chart(fig_pie)

with sector_state_tab2:
    # Calculate the sum of electricity usage by state
    state_total = electricity_data.groupby('state_name')['qty_usage_in_1month'].sum().reset_index()
    
    # Sort by total usage for better visualization
    state_total = state_total.sort_values('qty_usage_in_1month', ascending=False)
    
    # Create a treemap for state distribution
    fig_state = px.treemap(
        state_total,
        path=['state_name'],
        values='qty_usage_in_1month',
        title="Total Electricity Usage Distribution by State",
        color='qty_usage_in_1month',
        color_continuous_scale='Viridis',
        hover_data={'qty_usage_in_1month': ':.1f'}
    )
    
    # Customize layout
    fig_state.update_layout(
        margin=dict(t=50, l=25, r=25, b=25)
    )
    
    # Add percentage to labels
    fig_state.update_traces(
        texttemplate='%{label}<br>%{value:.1f} kWh<br>%{percentRoot:.1%}',
        hovertemplate='<b>%{label}</b><br>Total: %{value:.1f} kWh<br>Percentage: %{percentRoot:.1%}'
    )
    
    st.plotly_chart(fig_state)
    
    # Also show a horizontal bar chart for clearer comparison
    st.subheader("State-wise Electricity Usage - Bar Chart")
    
    # Create a bar chart with all states/UTs
    fig_bar = px.bar(
        state_total,  # Use all states
        y='state_name',
        x='qty_usage_in_1month',
        orientation='h',
        title="State-wise Total Electricity Usage",
        labels={
            'state_name': 'State/UT',
            'qty_usage_in_1month': 'Total Electricity Usage (kWh)'
        },
        color='qty_usage_in_1month',
        color_continuous_scale='Viridis'
    )
    
    # Add value labels
    fig_bar.update_traces(
        texttemplate='%{x:.1f}', 
        textposition='outside'
    )
    
    # Customize layout for scrollable chart - adjust height based on number of states
    num_states = len(state_total)
    chart_height = max(500, 20 * num_states)  # Minimum 500px height, or 20px per state
    
    fig_bar.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Total Electricity Usage (kWh)',
        yaxis_title='State/UT',
        height=chart_height,
        margin=dict(l=200, r=100, t=50, b=50)  # Increase left margin for state names
    )
    
    # Make the container scrollable
    st.plotly_chart(fig_bar, use_container_width=True)


st.markdown("---\n## Weighted Score Settings")
    
st.markdown("### 1. Electricity Consumption")
elec_total = st.number_input("Electricity Total Weight (Target: 0.25)", value=0.25, step=0.01, format="%.3f", key="wt_elec_total")

    
    
elec_col1, elec_col2 = st.columns(2)

with elec_col1:
    st.markdown("**Location Based (0.125)**") 
    location_col1, location_col2 = st.columns(2)

    with location_col1:
        w_elec_state = st.selectbox("State/UT", sorted(INDIAN_STATES_UTS), key="selected_state_ut")

    with location_col2:
        w_elec_sector = st.selectbox("Area Type", ["Rural", "Urban"], key="selected_area_type")

    w_elec_location = st.number_input("Location Based Weight", value=0.125, step=0.01, format="%.3f", key="wt_elec_location")

with elec_col2:
    st.markdown("**Economic Based (0.125)**")
    w_elec_mpce = st.number_input("MPCE Weight", value=0.125, step=0.01, format="%.3f", key="wt_elec_mpce")

elec_total = float(w_elec_location) + float(w_elec_mpce)

    

st.markdown("### 2. Water Consumption (Total weight: 0.25)")
w_water = st.number_input("Water Weight", value=0.25, step=0.01, format="%.3f", key="wt_water")

st.markdown("### 3. Commute")
w_commute_total = st.number_input("Total Commute Weight (Target: 0.25)", value=0.25, step=0.01, format="%.3f", key="wt_commute_total")

comm_col1, comm_col2 = st.columns(2)

with comm_col1:
    w_public = st.number_input("Public Transport Weight", value=0.125, step=0.01, format="%.3f", key="wt_public")

with comm_col2:
    w_private = st.number_input("Private Transport Weight", value=0.125, step=0.01, format="%.3f", key="wt_private")

commute_total = w_public + w_private


st.markdown("### 4. Company Environmental Score (Total weight: 0.25)")
w_company = st.number_input("Company Environmental Score Weight", value=0.25, step=0.01, format="%.3f", key="wt_company")

weights = {
    "Electricity_State": float(w_elec_location),   
    "Electricity_Sector": 0.0,                    
    "Electricity_MPCE": float(w_elec_mpce),       
    "Water": float(w_water),
    "Public_Transport": float(w_public),
    "Private_Transport": float(w_private),
    "Company": float(w_company)         
}

total = sum(weights.values())
st.markdown(f"**Total Weight:** {total:.3f}")

use_state_specific = st.checkbox("Use State/UT Specific Scoring", value=True)

if abs(total - 1.0) > 1e-3:
    st.error("Total weights must sum exactly to 1!")
else:
    if st.button("Generate Weighted Score"):
        company_data = load_company_data()
        
        if 'synth_data_raw' in st.session_state and company_data is not None:
            merged_data = st.session_state.synth_data_raw.copy()
            if 'Environment_Score' not in merged_data.columns and company_data is not None:
                company_scores = company_data['Environment_Score'].to_list()
                merged_data['Environment_Score'] = np.random.choice(company_scores, size=len(merged_data))
        else:
            if 'synth_data_raw' in st.session_state:
                merged_data = st.session_state.synth_data_raw
            else:
                st.error("No synthetic data available. Please generate analysis data first.")
                merged_data = None
        
        if merged_data is not None:
            # Modified to only use water, public and private transport weights
            # Create a new dictionary with only the weights we want to use
            filtered_weights = {
                "Public_Transport": w_public,
                "Private_Transport": w_private,
                "Water": w_water
            }
            
            # Pass only the filtered weights to the compute_weighted_score function
            scored, _ = compute_weighted_score(
                merged_data, filtered_weights, inverse=True, state_specific=False
            )
            
            st.session_state.scored_data = scored
            st.session_state.weights = filtered_weights
            
            st.session_state.sub_weights = {
                "commute": {
                    "public": w_public,
                    "private": w_private
                },
                "water": {
                    "water": w_water
                }
            }
            st.success("Weighted score and ranking generated using only water and transport data!")
        else:
            st.error("No data available for scoring. Please generate analysis data first.")

if "scored_data" in st.session_state:
    st.markdown("### Ranked Individuals Based on Sustainability Score")
    scored = st.session_state.scored_data
    core = ["ID", "Weighted_Score", "Z_Score", "Sustainability_Score", "Rank"]
    
    # Exclude electricity, state, and MPCE from the display columns
    exclude_cols = [
        "Electricity", "State_UT", "MPCE", 
        "Electricity_normalized", "Electricity_z_score",
        "State_Electricity_Factor", "Sector_Electricity_Factor", "MPCE_Electricity_Factor"
    ]
    
    extras = [
        col for col in (scored.columns if scored is not None else [])
        if col not in core
        and col not in exclude_cols
        and col != "Environment_Score"  
        and not col.endswith("_normalized")  
        and not col.endswith("_z_score")
        and col not in ["Sector_classification", "ESG_Rating", "Total_Employees"]
    ]
    
    display_cols = list(dict.fromkeys(core + extras))

    if scored is not None:
        render_sortable_table(scored[display_cols].reset_index(drop=True))
    else:
        st.warning("No scored data available to display.")

if "synth_data_raw" in st.session_state:
    if st.button("Show Distribution Graphs"):
        # Only show relevant columns (water, public transport, private transport)
        relevant_cols = ["Water", "Public_Transport", "Private_Transport"]
        for col in relevant_cols:
            if col in st.session_state.synth_data_raw.columns:
                fig, ax = plt.subplots()
                sns.histplot(st.session_state.synth_data_raw[col], bins=30, kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
        
        if "scored_data" in st.session_state:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            sns.histplot(st.session_state.scored_data["Sustainability_Score"], 
                        bins=30, kde=True, ax=ax, color='blue', label='Sustainability Score')
            ax.set_title("Distribution of Sustainability Score")
            
            ax2 = ax.twinx()
            sns.histplot(st.session_state.scored_data["Weighted_Score"], 
                        bins=30, kde=True, ax=ax2, color='red', alpha=0.5, label='Legacy Weighted Score')
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            st.pyplot(fig)
            
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.scored_data["Z_Score"], bins=30, kde=True, ax=ax)
            ax.set_title("Distribution of Z-Score")
            ax.axvline(0, color='red', linestyle='--', label='Mean (0)')
            ax.axvline(-1, color='green', linestyle='--', label='Lower Cap (-1)')
            ax.axvline(1, color='green', linestyle='--', label='Upper Cap (1)')
            ax.legend()
            st.pyplot(fig)
        
        if company_data is not None:
            # The company data visualizations remain unchanged
            st.subheader("Company Environmental Performance Comparison")
            
            company_scores = company_data.sort_values("Environment_Score", ascending=False)
            top_n = min(20, len(company_scores))  # Top 20 or all if less than 20
            top_companies = company_scores.head(top_n)
            
            fig = px.bar(top_companies,
                x="Environment_Score", 
                y="Company_Name",
                title="Top Companies by Environmental Score",
                orientation='h',  # Horizontal bars
                color="Environment_Score",
                color_continuous_scale="viridis")
            
            fig.update_layout(
            xaxis_title="Environment Score",
            yaxis_title="Company",
            yaxis=dict(autorange="reversed"),  
            height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if "Sector_classification" in company_data.columns:
                sector_avg = company_data.groupby("Sector_classification")["Environment_Score"].agg(
                    ['mean', 'count', 'std']
                ).sort_values('mean', ascending=False)

                fig = px.bar(
                    sector_avg.reset_index(),
                    x='Sector_classification', 
                    y='mean',
                    error_y='std',
                    title="Average Environmental Score by Sector",
                    labels={
                        'Sector_classification': 'Sector',
                        'mean': 'Average Environment Score',
                        'count': 'Number of Companies'
                    }
                )

                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=600,
                    showlegend=False
                )

                for i in range(len(sector_avg)):
                    fig.add_annotation(
                        x=sector_avg.index[i],
                        y=sector_avg['mean'][i],
                        text=f"n={int(sector_avg['count'][i])}",
                        showarrow=False,
                        yshift=10
                    )

                # Display plot
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.box(company_data, 
                           x="Sector_classification", 
                           y="Environment_Score",
                           title="Distribution of Environmental Scores by Sector",
                           color="Sector_classification")
                
                fig.update_layout(
                    xaxis_title="Sector",
                    yaxis_title="Environment Score", 
                    xaxis_tickangle=45,
                    showlegend=False,
                    height=600)
                
                st.plotly_chart(fig)
                
                # Scatter plot with plotly
                if "Total_Employees" in company_data.columns:
                    fig = px.scatter(company_data,
                                   x="Total_Employees",
                                   y="Environment_Score",
                                   color="Sector_classification", 
                                   title="Environment Score vs Company Size",
                                   opacity=0.6)
                    
                    fig.update_layout(
                        xaxis_title="Number of Employees",
                        yaxis_title="Environment Score",
                        height=600,
                        legend=dict(
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        ))
                    
                    st.plotly_chart(fig)

    if "scored_data" in st.session_state:
        st.markdown("---\n## Data Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Feature Correlations")
            numeric_cols = [col for col in st.session_state.synth_data_raw.columns
                            if st.session_state.synth_data_raw[col].dtype in [np.int64, np.float64]]
            corr = st.session_state.synth_data_raw[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.markdown("### Sustainability Score vs Features")
            synth_features = [
                c for c in st.session_state.synth_data_raw.columns
                if c not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Sustainability_Score",
                        "Company_Name", "Sector_classification", "State_UT",
                        "Total_Employees"]
            ]
            
            features = synth_features
            feature_to_plot = st.selectbox("Select Feature", features)
            fig, ax = plt.subplots()
            
            x_data = st.session_state.synth_data_raw[feature_to_plot]
                
            ax.scatter(x_data, st.session_state.scored_data["Sustainability_Score"])
            ax.set_xlabel(feature_to_plot)
            ax.set_ylabel("Sustainability Score")
            ax.set_title(f"{feature_to_plot} vs Sustainability Score")
            st.pyplot(fig)
        
        
            
        
        if "Electricity" in st.session_state.synth_data_raw.columns and "MPCE" in st.session_state.synth_data_raw.columns:
            st.markdown("---\n### Electricity-MPCE Analysis")
            fig, ax = plt.subplots()
            ratio = st.session_state.synth_data_raw["Electricity"] / st.session_state.synth_data_raw["MPCE"]
            ax.scatter(st.session_state.synth_data_raw["MPCE"], ratio,
                    c=st.session_state.scored_data["Sustainability_Score"], cmap='viridis')
            ax.set_xlabel("Monthly Per Capita Expenditure (MPCE)")
            ax.set_ylabel("Electricity/MPCE Ratio")
            ax.set_title("Affordability Analysis")
            fig.colorbar(ax.collections[0], label="Sustainability Score")
            st.pyplot(fig)
            
            # Affordability quartiles
            ratio_quartiles = pd.qcut(ratio, 4, labels=["Excellent", "Good", "Fair", "Poor"])
            st.markdown("### Affordability Quartiles")
            affordability_df = pd.DataFrame({
                "MPCE": st.session_state.synth_data_raw["MPCE"],
                "Electricity": st.session_state.synth_data_raw["Electricity"],
                "Ratio": ratio,
                "Quartile": ratio_quartiles,
                "Sustainability_Score": st.session_state.scored_data["Sustainability_Score"],
                "Water": st.session_state.synth_data_raw["Water"],
                "Public_Transport": st.session_state.synth_data_raw["Public_Transport"],
                "Private_Transport": st.session_state.synth_data_raw["Private_Transport"]
            })
            st.dataframe(affordability_df.groupby("Quartile").agg({
                "MPCE": "mean",
                "Electricity": "mean",
                "Ratio": "mean",
                "Sustainability_Score": "mean",
                "Water": "mean",
                "Public_Transport": "mean",
                "Private_Transport": "mean"
            }))

st.markdown("---\n# Test New Customer")
test_mode = st.radio("Input Mode", ["Manual Entry", "CSV Upload"], key="test_mode")

if "synth_data_raw" not in st.session_state:
    st.error("Please generate analysis data first.")
    features = []
else:
    features = [
        c for c in st.session_state.synth_data_raw.columns
        if c not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Sustainability_Score",
                    "Company_Name", "Sector_classification", "State_UT",
                    "Environment_Score", "ESG_Rating",
                    "Category", "Total_Employees"]
    ]
if test_mode == "CSV Upload":
        st.markdown("### Upload Test Data")   
        # Download template based on features from scored data
        if "synth_data_raw" in st.session_state:
            required_cols = ["Electricity", "MPCE","Water", "Public_Transport", "Private_Transport", "Company_Name", "Sector_classification", "Environment_Score", "ESG_Rating", "Total_Employees"]

            additional_cols = [c for c in st.session_state.synth_data_raw.columns 
                             if c not in ["ID", "Weighted_Score", "Z_Score", "Sustainability_Score", "Rank",
                                        "Company_Name", "Sector_classification", "Environment_Score", 
                                        "ESG_Rating", "Total_Employees"]]
            
            template_cols = list(dict.fromkeys(required_cols + additional_cols))
            tmpl = pd.DataFrame(columns=template_cols).to_csv(index=False).encode('utf-8')
            st.download_button("Download Test CSV Template",
                             data=tmpl, file_name="test_template.csv", mime="text/csv")
            
            up_test = st.file_uploader("Upload Test Data", type="csv", key="test_uploader")
            if up_test:
                test_df = pd.read_csv(up_test)
                st.dataframe(test_df)
                
                if st.button("Process Test Batch"):
                    if 'feature_stats' not in st.session_state or 'weights' not in st.session_state:
                        st.error("Please generate scores first.")
                    else:
                        test_with_scores = []
                        for _, row in test_df.iterrows():
                            new_customer = {f: row[f] for f in features if f in row}
                            
                            z_vals = {}
                            for feat, val in new_customer.items():
                                if feat in st.session_state.feature_stats:
                                    stats = st.session_state.feature_stats[feat]
                                    z = (stats['mean'] - val)/stats['std']  # Using inverse scoring
                                    z = np.clip(z, -1, 1)
                                    z_vals[feat] = z
                            
                            z_score = sum(z_vals[f] * st.session_state.weights.get(f, 0) for f in features if f in z_vals)
                            
                            sust_score = 500 * (1 - np.tanh(z_score/2.5))
                            
                            norm_vals = {}
                            for feat, val in new_customer.items():
                                if feat in st.session_state.feature_constraints:
                                    cmin, cmax = st.session_state.feature_constraints[feat]
                                    if cmax > cmin:
                                        norm_vals[feat] = ((cmax - val)/(cmax - cmin))*999 + 1
                                    else:
                                        norm_vals[feat] = 500
                            
                            weighted_score = sum(norm_vals[f] * st.session_state.weights.get(f, 0) for f in features if f in norm_vals)
                            
                            sust_rank = (st.session_state.scored_data["Sustainability_Score"] > sust_score).sum() + 1
                            trad_rank = (st.session_state.scored_data["Weighted_Score"] > weighted_score).sum() + 1
                            
                            result = row.to_dict()
                            result.update({
                                "Z_Score": z_score,
                                "Sustainability_Score": sust_score,
                                "Sustainability_Rank": sust_rank,
                                "Weighted_Score": weighted_score,
                                "Legacy_Rank": trad_rank
                            })
                            test_with_scores.append(result)
                        
                        results_df = pd.DataFrame(test_with_scores)
                        st.markdown("### Test Results")
                        st.dataframe(results_df)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(data=results_df, x="Sustainability_Score", bins=20, kde=True)
                        ax.axvline(results_df["Sustainability_Score"].mean(), color='red', linestyle='--', label='Mean')
                        ax.set_title("Distribution of Test Sustainability Scores")
                        ax.legend()
                        st.pyplot(fig)
                        
                        st.markdown("### Feature Importance Analysis")
                        
                        feature_importance = {}
                        for feat in features:
                            if feat in results_df.columns:
                                z_scores = []
                                for val in results_df[feat]:
                                    if feat in st.session_state.feature_stats:
                                        stats = st.session_state.feature_stats[feat]
                                        z = (stats['mean'] - val)/stats['std']
                                        z = np.clip(z, -1, 1)
                                        z_scores.append(abs(z))
                                if z_scores:
                                    feature_importance[feat] = np.mean(z_scores) * st.session_state.weights.get(feat, 0)
                        
                        if feature_importance:
                            fig, ax = plt.subplots(figsize=(10, 10))
                            total = sum(feature_importance.values())
                            sizes = [v/total for v in feature_importance.values()]
                            plt.pie(sizes, labels=feature_importance.keys(), autopct='%1.1f%%')
                            plt.title('Average Feature Contribution')
                            st.pyplot(fig)
                            
                            importance_df = pd.DataFrame([
                                {"Feature": k, "Importance": v, "Percentage": f"{(v/total)*100:.2f}%"}
                                for k, v in feature_importance.items()
                            ]).sort_values("Importance", ascending=False)
                            st.dataframe(importance_df)

else:
    st.markdown("### Enter Customer Details")
# CRISIL ESG - Environmental Scoring Only
    # Get sectors from company data if available
    st.markdown("## CRISIL ESG - Environmental Scoring Only")

    if company_data is not None:
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Employee Range Analysis", "Company-Only Analysis"],
            key="crisil_analysis_type"
        )
        
        if analysis_type == "Employee Range Analysis":
            # Option 1: Employee Range Analysis
            # 1) Industry dropdown
            industry_opts = company_data['Sector_classification'].dropna().unique()
            industry = st.selectbox("Select Industry Sector", industry_opts, key="employee_range_industry")

            df_sector = company_data[company_data['Sector_classification'] == industry]
            
            # 2) Employee Range slider
            min_emp = int(df_sector['Total_Employees'].min()) if not df_sector['Total_Employees'].isna().all() else 0
            max_emp = int(df_sector['Total_Employees'].max()) if not df_sector['Total_Employees'].isna().all() else 10000
            
            # Ensure min and max values are different
            if min_emp == max_emp:
                min_emp = max(0, min_emp - 100)
                max_emp = max_emp + 100
                
            emp_range = st.slider(
                "Select Employee Range", 
                min_value=min_emp,
                max_value=max_emp,
                value=(min_emp, max_emp),
                step=10,
                key="employee_range_slider"
            )
            
            df_filtered = df_sector[
                (df_sector['Total_Employees'] >= emp_range[0]) & 
                (df_sector['Total_Employees'] <= emp_range[1])
            ]
            
            if len(df_filtered) > 0:
                baseline_mean = df_filtered['Environment_Score'].mean()
                baseline_std = df_filtered['Environment_Score'].std(ddof=1)
                
                # Display filtered companies
                st.markdown(f"### Companies in {industry} sector with {emp_range[0]}-{emp_range[1]} employees")
                st.markdown(f"**Baseline Environment Score:** {baseline_mean:.2f} (std: {baseline_std:.2f})")
                
                df_results = df_filtered.copy()
                df_results['Env_Z_Score'] = (df_results['Environment_Score'] - baseline_mean) / baseline_std
                
                # Normalize scores to 0-100 scale
                min_z = df_results['Env_Z_Score'].min()
                max_z = df_results['Env_Z_Score'].max()
                df_results['Normalized_Score'] = ((df_results['Env_Z_Score'] - min_z) / (max_z - min_z)) * 100
                
                # Display results table
                st.dataframe(
                    df_results[[
                        'Company_Name',
                        'Total_Employees',
                        'Environment_Score',
                        'Env_Z_Score',
                        'Normalized_Score'
                    ]].sort_values('Environment_Score', ascending=False).reset_index(drop=True)
                )
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.histplot(df_results['Environment_Score'], kde=True, ax=ax)
                
                ax.axvline(baseline_mean, color='red', linestyle='--', label=f'Mean ({baseline_mean:.2f})')
                
                ax.axvline(baseline_mean + baseline_std, color='green', linestyle=':', 
                        label=f'+1 Std Dev ({baseline_mean + baseline_std:.2f})')
                ax.axvline(baseline_mean - baseline_std, color='orange', linestyle=':', 
                        label=f'-1 Std Dev ({baseline_mean - baseline_std:.2f})')
                
                ax.set_xlabel('Environment Score')
                ax.set_title(f'Distribution of Environment Scores for {industry} ({emp_range[0]}-{emp_range[1]} employees)')
                ax.legend()
                
                st.pyplot(fig)
                
                st.markdown("### Compare Your Company")
                
                compare_method = st.radio("Comparison Method", 
                                        ["Select from list", "Enter custom score"],
                                        key="employee_range_compare_method")
                
                if compare_method == "Select from list":
                    selected_company = st.selectbox("Select Your Company", 
                                                ["(None)"] + df_results['Company_Name'].tolist(),
                                                key="employee_range_comparison_company")
                    
                    if selected_company != "(None)":
                        company_data_row = df_results[df_results['Company_Name'] == selected_company].iloc[0]
                        company_score = company_data_row['Environment_Score']
                        company_z = company_data_row['Env_Z_Score']
                        company_norm = company_data_row['Normalized_Score']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Environment Score", f"{company_score:.2f}")
                        with col2:
                            st.metric("Z-Score", f"{company_z:.2f}", 
                                    f"{company_z:.2f} SD from mean")
                        with col3:
                            st.metric("Normalized Score", f"{company_norm:.2f}/100")
                            
                        better_than = (df_results['Env_Z_Score'] < company_z).mean() * 100
                        st.success(f"**{selected_company}** performs better than **{better_than:.1f}%** of companies in this segment (based on Z-Score)")
                else:
                    custom_score = st.number_input("Enter Your Company's Environment Score", 
                                                min_value=0.0, max_value=100.0, value=50.0,
                                                key="employee_range_custom_score")
                    
                    custom_z = (custom_score - baseline_mean) / baseline_std
                    
                    custom_norm = ((custom_z - min_z) / (max_z - min_z)) * 100 if max_z != min_z else 50
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Environment Score", f"{custom_score:.2f}")
                    with col2:
                        st.metric("Z-Score", f"{custom_z:.2f}", 
                                f"{custom_z:.2f} SD from mean")
                    with col3:
                        st.metric("Normalized Score", f"{custom_norm:.2f}/100")
                        
                    # Position in distribution based on Z-score
                    better_than = (df_results['Env_Z_Score'] < custom_z).mean() * 100
                    st.success(f"Your company performs better than **{better_than:.1f}%** of companies in this segment (based on Z-Score)")
            else:
                st.warning(f"No companies found in {industry} sector with {emp_range[0]}-{emp_range[1]} employees")
        
        elif analysis_type == "Company-Only Analysis":
            # Option 2: Company-Only Analysis
            # Calculate baseline statistics for the entire dataset
            overall_mean = company_data['Environment_Score'].mean()
            overall_std = company_data['Environment_Score'].std(ddof=1)
            
            st.markdown("### Compare Against Entire Dataset")
            st.markdown(f"**Overall Environment Score Baseline:** {overall_mean:.2f} (std: {overall_std:.2f})")
            
            # Company selection
            all_companies = company_data['Company_Name'].dropna().tolist()
            selected_company = st.selectbox("Select Your Company", ["(None)"] + all_companies, key="company_only_company")
            
            if selected_company != "(None)":
                # Get company data
                company_row = company_data[company_data['Company_Name'] == selected_company].iloc[0]
                company_score = company_row['Environment_Score']
                company_sector = company_row['Sector_classification']
                company_employees = company_row['Total_Employees']
                
                company_z = (company_score - overall_mean) / overall_std
                
                percentile = (company_data['Environment_Score'] < company_score).mean() * 100
                
                # Display company info
                st.markdown(f"### {selected_company}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sector", f"{company_sector}")
                with col2:
                    st.metric("Employees", f"{company_employees}")
                with col3:
                    st.metric("Environment Score", f"{company_score:.2f}")
                
                # Display comparison metrics (now focusing on Z-score)
                col1, col2, col3 = st.columns(3)
                with col1:
                    z_description = "above" if company_z > 0 else "below"
                    st.metric("Z-Score", 
                            f"{company_z:.2f}", 
                            f"{abs(company_z):.2f} SD {z_description} mean")
                with col2:
                    perf_description = "outperforms" if company_z > 0 else "underperforms"
                    st.metric("Performance", f"{perf_description} by {abs(company_z):.2f} SD")
                with col3:
                    st.metric("Percentile", f"{percentile:.1f}%")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.histplot(company_data['Environment_Score'], kde=True, ax=ax)
                
                ax.axvline(overall_mean, color='red', linestyle='--', label=f'Mean ({overall_mean:.2f})')
                
                ax.axvline(company_score, color='blue', linestyle='-', 
                        label=f'{selected_company} ({company_score:.2f}, Z={company_z:.2f})')
                
                ax.set_xlabel('Environment Score')
                ax.set_title(f'Distribution of Environment Scores Across All Companies')
                ax.legend()
                
                st.pyplot(fig)
                
                sector_data = company_data[company_data['Sector_classification'] == company_sector]
                sector_mean = sector_data['Environment_Score'].mean()
                sector_std = sector_data['Environment_Score'].std(ddof=1)
                sector_z = (company_score - sector_mean) / sector_std
                sector_percentile = (sector_data['Environment_Score'] < company_score).mean() * 100
                
                st.markdown(f"### Comparison with {company_sector} Sector")
                st.markdown(f"**Sector Environment Score:** {sector_mean:.2f} (std: {sector_std:.2f})")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    z_description = "above" if sector_z > 0 else "below"
                    st.metric("Sector Z-Score", 
                            f"{sector_z:.2f}", 
                            f"{abs(sector_z):.2f} SD {z_description} sector mean")
                with col2:
                    perf_description = "outperforms" if sector_z > 0 else "underperforms"
                    st.metric("Sector Performance", f"{perf_description} by {abs(sector_z):.2f} SD")
                with col3:
                    st.metric("Sector Percentile", f"{sector_percentile:.1f}%")
                
                st.markdown(f"### All Performers in {company_sector} (Highest to Lowest)")
                
                all_sector_companies = sector_data.copy()
                all_sector_companies['Sector_Z_Score'] = (all_sector_companies['Environment_Score'] - sector_mean) / sector_std
                all_sector_companies = all_sector_companies.sort_values('Environment_Score', ascending=False)
                
                display_df = all_sector_companies[[
                    'Company_Name',
                    'Environment_Score',
                    'Sector_Z_Score',
                    'ESG_Rating',
                    'Total_Employees'
                ]].reset_index(drop=True)
                
                selected_idx = display_df.index[display_df['Company_Name'] == selected_company].tolist()
                
                styled_df = display_df.style.apply(
                    lambda x: ['background-color: lightskyblue' if i in selected_idx else '' 
                            for i in range(len(display_df))],
                    axis=0
                )
                
                st.dataframe(styled_df)
            else:
                custom_score = st.number_input("Enter Your Company's Environment Score", 
                                            min_value=0.0, max_value=100.0, value=50.0,
                                            key="company_only_custom_score")
                
                custom_z = (custom_score - overall_mean) / overall_std
                
                percentile = (company_data['Environment_Score'] < custom_score).mean() * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    z_description = "above" if custom_z > 0 else "below"
                    st.metric("Z-Score", 
                            f"{custom_z:.2f}", 
                            f"{abs(custom_z):.2f} SD {z_description} mean")
                with col2:
                    perf_description = "outperforms" if custom_z > 0 else "underperforms"
                    st.metric("Performance", f"{perf_description} by {abs(custom_z):.2f} SD")
                with col3:
                    st.metric("Percentile", f"{percentile:.1f}%")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.histplot(company_data['Environment_Score'], kde=True, ax=ax)
                
                ax.axvline(overall_mean, color='red', linestyle='--', label=f'Mean ({overall_mean:.2f})')
                
                ax.axvline(custom_score, color='blue', linestyle='-', 
                        label=f'Your Company ({custom_score:.2f}, Z={custom_z:.2f})')
                
                ax.set_xlabel('Environment Score')
                ax.set_title(f'Distribution of Environment Scores Across All Companies')
                ax.legend()
                
                st.pyplot(fig)
                
                selected_sector = st.selectbox("Compare with Sector (Optional)", 
                                        ["(None)"] + sorted([str(x) for x in company_data['Sector_classification'].dropna().unique()]),
                                        key="company_only_sector")
                
                if selected_sector != "(None)":
                    sector_data = company_data[company_data['Sector_classification'] == selected_sector]
                    sector_mean = sector_data['Environment_Score'].mean()
                    sector_std = sector_data['Environment_Score'].std(ddof=1)
                    sector_z = (custom_score - sector_mean) / sector_std
                    sector_percentile = (sector_data['Environment_Score'] < custom_score).mean() * 100
                    
                    st.markdown(f"### Comparison with {selected_sector} Sector")
                    st.markdown(f"**Sector Environment Score:** {sector_mean:.2f} (std: {sector_std:.2f})")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        z_description = "above" if sector_z > 0 else "below"
                        st.metric("Sector Z-Score", 
                                f"{sector_z:.2f}", 
                                f"{abs(sector_z):.2f} SD {z_description} sector mean")
                    with col2:
                        perf_description = "outperforms" if sector_z > 0 else "underperforms"
                        st.metric("Sector Performance", f"{perf_description} by {abs(sector_z):.2f} SD")
                    with col3:
                        st.metric("Sector Percentile", f"{sector_percentile:.1f}%")
                        
                    all_sector_companies = sector_data.copy()
                    all_sector_companies['Sector_Z_Score'] = (all_sector_companies['Environment_Score'] - sector_mean) / sector_std
                    all_sector_companies = all_sector_companies.sort_values('Environment_Score', ascending=False)
                    
                    st.markdown(f"### All Performers in {selected_sector} (Highest to Lowest)")
                    
                    display_df = all_sector_companies[[
                        'Company_Name',
                        'Environment_Score',
                        'Sector_Z_Score',
                        'ESG_Rating',
                        'Total_Employees'
                    ]].reset_index(drop=True)
                    
                    st.dataframe(display_df)
    else:
        st.error("Company data unavailable. Please add `employees_2024_25_updated.csv` or download the template in the sidebar.")

    st.markdown("---")
    st.markdown("## Your Personal Electricity Analysis")
    st.markdown("")  # Add some spacing
    col1, col2 = st.columns(2)

    with col1:
        user_state = st.selectbox("Select Your State/UT", 
                                options=sorted(INDIAN_STATES_UTS),
                                key="personal_state")
        st.session_state.user_state = user_state
        
        user_sector_name = st.radio("Select Your Sector", 
                                ['Rural', 'Urban'],
                                key="personal_sector")
        user_sector = 1 if user_sector_name == 'Rural' else 2

        mpce_ranges = ["₹1-1,000", "₹1,000-5,000", "₹5,000-10,000", "₹10,000-25,000", "₹25,000+"]
        mpce_range_values = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 25000), (25000, float('inf'))]
        
        user_mpce_range_index = st.selectbox(
            "Select Your Monthly Per Capita Expenditure (MPCE) Range",
            options=range(len(mpce_ranges)),
            format_func=lambda x: mpce_ranges[x],
            key="personal_mpce_range"
        )
        
        st.session_state.user_mpce_range = mpce_range_values[user_mpce_range_index]
        st.session_state.user_mpce_range_name = mpce_ranges[user_mpce_range_index]

    with col2:
        st.markdown("Enter your personal electricity consumption and compare it with the dataset")
        user_electricity = st.number_input("Your Monthly Electricity Usage (kWh)", 
                                        min_value=0.0, 
                                        value=0.0, 
                                        step=10.0,
                                        key="personal_electricity")
        
        st.session_state.user_electricity = user_electricity

        if user_electricity > 0:
            st.metric("Your Electricity Consumption", f"{user_electricity:.2f} kWh")
            
            user_cost = st.number_input("Enter your monthly electricity cost (₹)", 
                                        min_value=0.0, 
                                        step=100.0)
            
            if electricity_data is not None:
                state_sector_data = electricity_data[
                    (electricity_data['state_name'] == user_state) & 
                    (electricity_data['sector'] == user_sector)
                ]
                
                if not state_sector_data.empty:
                    state_avg = state_sector_data['qty_usage_in_1month'].mean()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(state_sector_data['qty_usage_in_1month'], bins=15, kde=True, ax=ax)
                    ax.axvline(user_electricity, color='red', linestyle='--', label='Your Usage')
                    ax.axvline(state_avg, color='green', linestyle='--', label=f'{user_state} Average')
                    ax.set_title(f"Your Electricity Usage Compared to {user_state}, {user_sector_name}")
                    ax.set_xlabel("Monthly Consumption (kWh)")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning(f"No data available for {user_state} ({user_sector_name}). Showing overall statistics instead.")
                    
                    if 'full_electricity_data' in st.session_state:
                        overall_avg = st.session_state.full_electricity_data['qty_usage_in_1month'].mean()
                        st.markdown(f"Overall average electricity consumption: **{overall_avg:.2f} kWh**")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(st.session_state.full_electricity_data['qty_usage_in_1month'], 
                                    bins=15, kde=True, ax=ax)
                        ax.axvline(user_electricity, color='red', linestyle='--', label='Your Usage')
                        ax.axvline(overall_avg, color='green', linestyle='--', label='Overall Average')
                        ax.set_title("Your Electricity Usage Compared to Overall Average")
                        ax.set_xlabel("Monthly Consumption (kWh)")
                        ax.legend()
                        st.pyplot(fig)
            
            # Calculate and display sustainability scores with demographic factors
            if 'feature_stats' in st.session_state and 'Electricity' in st.session_state.feature_stats:
                # FEATURE 1: State and Rural/Urban based calculations
                # Check if we have state-sector specific data
                feature1_score = 0
                feature1_z_score = 0
                
                # Try to get state-sector specific stats
                if user_state in st.session_state.feature_stats_by_state and 'Electricity' in st.session_state.feature_stats_by_state[user_state]:
                    state_stats = st.session_state.feature_stats_by_state[user_state]['Electricity']
                    
                    # Create or access sector-specific stats for this state
                    if 'sector_stats' not in st.session_state:
                        st.session_state.sector_stats = {}
                    
                    # Generate key for this state-sector combination
                    state_sector_key = f"{user_state}_{user_sector_name}"
                    
                    # If we don't have this combination in our stats, calculate it
                    if state_sector_key not in st.session_state.sector_stats and not state_sector_data.empty:
                        sector_mean = state_sector_data['qty_usage_in_1month'].mean()
                        sector_std = state_sector_data['qty_usage_in_1month'].std()
                        
                        # Store these stats
                        st.session_state.sector_stats[state_sector_key] = {
                            'mean': sector_mean,
                            'std': sector_std
                        }
                    
                    # Use state-sector specific stats if available, otherwise use state stats
                    if state_sector_key in st.session_state.sector_stats:
                        sector_stats = st.session_state.sector_stats[state_sector_key]
                        feature1_mean = sector_stats['mean']
                        feature1_std = sector_stats['std']
                    else:
                        # Fall back to state stats if no sector stats
                        feature1_mean = state_stats['mean']
                        feature1_std = state_stats['std']
                    
                    # Calculate z-score (negative because inverse scoring - lower consumption is better)
                    feature1_z_score = (feature1_mean - user_electricity) / feature1_std if feature1_std > 0 else 0
                    feature1_z_score = np.clip(feature1_z_score, -3, 3)  # Cap between -3 and 3
                    
                    # Scale to 0-0.5 range for the first feature (50% weight)
                    feature1_score = 0.25 * (1 - np.tanh(feature1_z_score/2.5)) + 0.25
                else:
                    # Fall back to overall stats if no state-specific stats
                    stats = st.session_state.feature_stats['Electricity']
                    feature1_mean = stats['mean']
                    feature1_std = stats['std']
                    
                    # Calculate z-score
                    feature1_z_score = (feature1_mean - user_electricity) / feature1_std if feature1_std > 0 else 0
                    feature1_z_score = np.clip(feature1_z_score, -3, 3)
                    
                    # Scale to 0-0.5 range for the first feature
                    feature1_score = 0.25 * (1 - np.tanh(feature1_z_score/2.5)) + 0.25
                
                # FEATURE 2: MPCE Range based calculations
                feature2_score = 0
                feature2_z_score = 0
                
                # Calculate or retrieve MPCE range statistics
                if 'mpce_stats' not in st.session_state:
                    # Initialize MPCE range statistics if this is the first run
                    st.session_state.mpce_stats = {}
                    
                    # Check if we have full data to calculate MPCE statistics
                    if 'full_electricity_data' in st.session_state and 'mpce' in st.session_state.full_electricity_data.columns:
                        full_data = st.session_state.full_electricity_data
                        
                        # Calculate stats for each MPCE range
                        for i, (lower, upper) in enumerate(mpce_range_values):
                            range_data = full_data[(full_data['mpce'] >= lower) & (full_data['mpce'] < upper)]
                            
                            if not range_data.empty:
                                mean_usage = range_data['qty_usage_in_1month'].mean()
                                std_usage = range_data['qty_usage_in_1month'].std()
                                
                                st.session_state.mpce_stats[i] = {
                                    'range_name': mpce_ranges[i],
                                    'mean': mean_usage,
                                    'std': std_usage
                                }
                            else:
                                # If no data for this range, use overall stats
                                st.session_state.mpce_stats[i] = {
                                    'range_name': mpce_ranges[i],
                                    'mean': st.session_state.feature_stats['Electricity']['mean'],
                                    'std': st.session_state.feature_stats['Electricity']['std']
                                }
                
                # Use MPCE range statistics for feature 2
                if user_mpce_range_index in st.session_state.mpce_stats:
                    mpce_stats = st.session_state.mpce_stats[user_mpce_range_index]
                    feature2_mean = mpce_stats['mean']
                    feature2_std = mpce_stats['std']
                    
                    # Calculate z-score
                    feature2_z_score = (feature2_mean - user_electricity) / feature2_std if feature2_std > 0 else 0
                    feature2_z_score = np.clip(feature2_z_score, -3, 3)
                    
                    # Scale to 0-0.5 range for the second feature (50% weight)
                    feature2_score = 0.25 * (1 - np.tanh(feature2_z_score/2.5)) + 0.25
                else:
                    # Fall back to overall stats
                    stats = st.session_state.feature_stats['Electricity']
                    feature2_mean = stats['mean']
                    feature2_std = stats['std']
                    
                    # Calculate z-score
                    feature2_z_score = (feature2_mean - user_electricity) / feature2_std if feature2_std > 0 else 0
                    feature2_z_score = np.clip(feature2_z_score, -3, 3)
                    
                    # Scale to 0-0.5 range for the second feature
                    feature2_score = 0.25 * (1 - np.tanh(feature2_z_score/2.5)) + 0.25
                
                # Calculate total sustainability score (both features sum to 1)
                total_sust_score = (feature1_score + feature2_score) * 1000
                
                # Calculate combined z-score (average of both z-scores)
                combined_z_score = (feature1_z_score + feature2_z_score) / 2
                
                # Calculate percentile rank if we have scored data
                percentile = 0
                if 'scored_data' in st.session_state:
                    percentile = (st.session_state.scored_data['Sustainability_Score'] < total_sust_score).mean() * 100
                
                # Display the scores in columns
                st.markdown("### Your Sustainability Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    comparison = "better than average" if user_electricity < feature1_mean else "worse than average"
                    diff_pct = abs(user_electricity - feature1_mean) / feature1_mean * 100
                    st.metric(f"Compared to {user_state} ({user_sector_name})", 
                            f"{diff_pct:.1f}% {comparison}", 
                            f"{feature1_mean:.1f} kWh avg")
                    
                    st.metric("Location-Based Score (50%)", 
                            f"{feature1_score*1000:.1f}/500",
                            f"Z-score: {feature1_z_score:.2f}")
                
                with col2:
                    mpce_comparison = "better than average" if user_electricity < feature2_mean else "worse than average"
                    mpce_diff_pct = abs(user_electricity - feature2_mean) / feature2_mean * 100
                    st.metric(f"Compared to MPCE {st.session_state.user_mpce_range_name}", 
                            f"{mpce_diff_pct:.1f}% {mpce_comparison}", 
                            f"{feature2_mean:.1f} kWh avg")
                    
                    st.metric("MPCE-Based Score (50%)", 
                            f"{feature2_score*1000:.1f}/500",
                            f"Z-score: {feature2_z_score:.2f}")
                
                with col3:
                    st.metric("Total Sustainability Score", 
                            f"{total_sust_score:.1f}/1000",
                            f"Combined Z-score: {combined_z_score:.2f}")
                    
                    st.metric("Your Percentile", 
                            f"{percentile:.1f}%",
                            f"Better than {percentile:.1f}% of users")
                
                # Create visualization showing both features
                st.markdown("### Feature Breakdown")
                
                # Feature contribution chart
                feature_data = pd.DataFrame({
                    'Feature': ['Location-Based (State/Sector)', 'MPCE-Based'],
                    'Score': [feature1_score*1000, feature2_score*1000],
                    'Max_Score': [500, 500]
                })
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh(feature_data['Feature'], feature_data['Score'], color=['#4287f5', '#42f5a7'])
                
                # Add score labels
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                        f"{feature_data['Score'][i]:.1f}/500", 
                        va='center')
                
                # Add a vertical line at the total score
                ax.axvline(x=total_sust_score, color='red', linestyle='--', 
                        label=f'Total Score: {total_sust_score:.1f}/1000')
                
                ax.set_xlim(0, 1000)
                ax.set_title('Your Sustainability Score Breakdown')
                ax.set_xlabel('Score')
                ax.legend()
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Additional state visualization 
                if 'feature_stats_by_state' in st.session_state:
                    st.markdown("### Your Usage Compared to State Averages")
                    
                    # Get state averages
                    state_data = []
                    for state, stats in st.session_state.feature_stats_by_state.items():
                        if 'Electricity' in stats:
                            state_data.append({
                                'State_UT': state,
                                'Average_Electricity': stats['Electricity']['mean'],
                                'Is_Your_State': state == user_state
                            })
                    
                    state_df = pd.DataFrame(state_data)
                    state_df = state_df.sort_values('Average_Electricity')
                    
                    # Create bar chart with user's state highlighted
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.bar(state_df['State_UT'], state_df['Average_Electricity'], 
                                color=state_df['Is_Your_State'].map({True: 'red', False: 'blue'}))
                    
                    # Add user's consumption as a horizontal line
                    ax.axhline(y=user_electricity, color='green', linestyle='--', 
                            label=f'Your Usage ({user_electricity:.1f} kWh)')
                    
                    # Highlight user's state's average
                    user_state_avg = state_df[state_df['Is_Your_State']]['Average_Electricity'].values[0] \
                                    if len(state_df[state_df['Is_Your_State']]) > 0 else 0
                    ax.scatter(user_state, user_state_avg, s=100, color='red', zorder=3,
                            label=f'{user_state} Avg ({user_state_avg:.1f} kWh)')
                    
                    # Format chart
                    plt.xticks(rotation=90)
                    plt.xlabel('State/UT')
                    plt.ylabel('Average Electricity Consumption (kWh)')
                    plt.title('Your Electricity Usage vs. State Averages')
                    plt.legend()
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                
                # MPCE range comparison visualization
                if 'mpce_stats' in st.session_state:
                    st.markdown("### Your Usage Compared to MPCE Ranges")
                    
                    # Prepare MPCE data for visualization
                    mpce_data = []
                    for idx, stats in st.session_state.mpce_stats.items():
                        mpce_data.append({
                            'MPCE_Range': stats['range_name'],
                            'Average_Electricity': stats['mean'],
                            'Is_Your_Range': idx == user_mpce_range_index
                        })
                    
                    mpce_df = pd.DataFrame(mpce_data)
                    
                    # Create bar chart with user's MPCE range highlighted
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(mpce_df['MPCE_Range'], mpce_df['Average_Electricity'], 
                                color=mpce_df['Is_Your_Range'].map({True: 'red', False: 'blue'}))
                    
                    # Add user's consumption as a horizontal line
                    ax.axhline(y=user_electricity, color='green', linestyle='--', 
                            label=f'Your Usage ({user_electricity:.1f} kWh)')
                    
                    # Highlight user's MPCE range average
                    user_mpce_avg = mpce_df[mpce_df['Is_Your_Range']]['Average_Electricity'].values[0] \
                                    if len(mpce_df[mpce_df['Is_Your_Range']]) > 0 else 0
                    ax.scatter(st.session_state.user_mpce_range_name, user_mpce_avg, s=100, color='red', zorder=3,
                            label=f'Your MPCE Range Avg ({user_mpce_avg:.1f} kWh)')
                    
                    # Format chart
                    plt.xticks(rotation=45)
                    plt.xlabel('MPCE Range')
                    plt.ylabel('Average Electricity Consumption (kWh)')
                    plt.title('Your Electricity Usage vs. MPCE Range Averages')
                    plt.legend()
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            
        
        
        st.markdown("---")
        st.markdown("## Resource Consumption")
        
        st.markdown("Water  (in Litres)")
        water_units = st.number_input("Water", min_value=0.0)
            
        water_min, water_max = 0.0, 1000.0
        if 'feature_constraints' in st.session_state and 'Water' in st.session_state.feature_constraints:
                water_min, water_max = st.session_state.feature_constraints['Water']
            
        
        
    st.title("Transport Carbon Footprint Calculator")
st.markdown("""
This helps you calculate your monthly carbon footprint based on your daily commute and transportation choices.
Compare different transportation modes and receive personalized sustainability recommendations.
""")

# Initialize session state variables if they don't exist
if 'calculated' not in st.session_state:
    st.session_state.calculated = False
if 'total_emissions' not in st.session_state:
    st.session_state.total_emissions = 0
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'emissions_data' not in st.session_state:
    st.session_state.emissions_data = {}

# Define emission factors (approximate values)
emission_factors = {
    # Two wheelers
    "two_wheeler": {
        "Scooter": {
            "petrol": {"min": 0.03, "max": 0.06},
            "diesel": {"min": 0.04, "max": 0.07},
            "electric": {"min": 0.01, "max": 0.02}
        },
        "Motorcycle": {
            "petrol": {"min": 0.05, "max": 0.09},
            "diesel": {"min": 0.06, "max": 0.10},
            "electric": {"min": 0.01, "max": 0.02}
        }
    },
    
    # Three wheelers
    "three_wheeler": {
        "petrol": {"min": 0.07, "max": 0.12},
        "diesel": {"min": 0.08, "max": 0.13},
        "electric": {"min": 0.02, "max": 0.03},
        "cng": {"min": 0.05, "max": 0.09}
    },
    
    # Four wheelers
    "four_wheeler": {
        "small": {
            "petrol": {"base": 0.12, "uplift": 1.1},
            "diesel": {"base": 0.14, "uplift": 1.1},
            "cng": {"base": 0.10, "uplift": 1.05},
            "electric": {"base": 0.05, "uplift": 1.0}
        },
        "hatchback": {
            "petrol": {"base": 0.15, "uplift": 1.1},
            "diesel": {"base": 0.17, "uplift": 1.1},
            "cng": {"base": 0.12, "uplift": 1.05},
            "electric": {"base": 0.06, "uplift": 1.0}
        },
        "premium_hatchback": {
            "petrol": {"base": 0.18, "uplift": 1.15},
            "diesel": {"base": 0.20, "uplift": 1.15},
            "cng": {"base": 0.14, "uplift": 1.10},
            "electric": {"base": 0.07, "uplift": 1.0}
        },
        "compact_suv": {
            "petrol": {"base": 0.21, "uplift": 1.2},
            "diesel": {"base": 0.23, "uplift": 1.2},
            "cng": {"base": 0.16, "uplift": 1.15},
            "electric": {"base": 0.08, "uplift": 1.0}
        },
        "sedan": {
            "petrol": {"base": 0.20, "uplift": 1.2},
            "diesel": {"base": 0.22, "uplift": 1.2},
            "cng": {"base": 0.16, "uplift": 1.15},
            "electric": {"base": 0.08, "uplift": 1.0}
        },
        "suv": {
            "petrol": {"base": 0.25, "uplift": 1.25},
            "diesel": {"base": 0.28, "uplift": 1.25},
            "cng": {"base": 0.20, "uplift": 1.2},
            "electric": {"base": 0.10, "uplift": 1.0}
        },
        "hybrid": {
            "petrol": {"base": 0.14, "uplift": 1.05},
            "diesel": {"base": 0.16, "uplift": 1.05},
            "electric": {"base": 0.07, "uplift": 1.0}
        }
    },
    
    # Public transport
    "public_transport": {
        "taxi": {
            "small": {
                "petrol": {"base": 0.12, "uplift": 1.1},
                "diesel": {"base": 0.14, "uplift": 1.1},
                "cng": {"base": 0.10, "uplift": 1.05},
                "electric": {"base": 0.05, "uplift": 1.0}
            },
            "hatchback": {
                "petrol": {"base": 0.15, "uplift": 1.1},
                "diesel": {"base": 0.17, "uplift": 1.1},
                "cng": {"base": 0.12, "uplift": 1.05},
                "electric": {"base": 0.06, "uplift": 1.0}
            },
            "sedan": {
                "petrol": {"base": 0.20, "uplift": 1.2},
                "diesel": {"base": 0.22, "uplift": 1.2},
                "cng": {"base": 0.16, "uplift": 1.15},
                "electric": {"base": 0.08, "uplift": 1.0}
            },
            "suv": {
                "petrol": {"base": 0.25, "uplift": 1.25},
                "diesel": {"base": 0.28, "uplift": 1.25},
                "cng": {"base": 0.20, "uplift": 1.2},
                "electric": {"base": 0.10, "uplift": 1.0}
            }
        },
        "bus": {
            "electric": 0.025,
            "petrol": 0.05,
            "diesel": 0.045,
            "cng": 0.035
        },
        "metro": 0.015
    }
}

# Create input form in the main area
st.header("Your Commute Details")

# Universal commute details
col1, col2, col3 = st.columns(3)
with col1:
    distance = st.number_input("Daily one-way distance (km)", min_value=0.1, value=10.0, step=0.5)
with col2:
    days_per_week = st.number_input("Commuting days per week", min_value=1, max_value=7, value=5, step=1)
with col3:
    weeks_per_month = st.number_input("Commuting weeks per month", min_value=1, max_value=5, value=4, step=1)

# Calculate total monthly distance
total_monthly_km = distance * 2 * days_per_week * weeks_per_month
st.metric("Total monthly commute distance", f"{total_monthly_km:.1f} km")

# Transport category selection
st.header("Select Your Transport Category")
transport_category = st.selectbox(
    "Transport Category",
    ["Private Transport", "Public Transport", "Both Private and Public"]
)

# Define variable to store emission factors
emission_factor = 0
people_count = 1
vehicle_type = ""
vehicle_name = ""

# Dynamic form based on transport category
if transport_category == "Private Transport" or transport_category == "Both Private and Public":
    st.subheader("Private Transport Details")
    col1, col2 = st.columns([1, 2])
    with col1:
        private_vehicle_type = st.selectbox(
            "Vehicle Type",
            ["Two Wheeler", "Three Wheeler", "Four Wheeler"],
            key="private_vehicle"
        )

    # Dynamic form based on private vehicle type
    if private_vehicle_type == "Two Wheeler":
        col1, col2, col3 = st.columns(3)
        with col1:
            category = st.selectbox("Category", ["Scooter", "Motorcycle"])
        with col2:
            engine_cc = st.number_input("Engine (cc)", 50, 1500, 150)
        with col3:
            fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "electric"])
        
        # Calculate emission factor based on engine size
        if engine_cc <= 150:
            emission_factor = emission_factors["two_wheeler"][category][fuel_type]["min"]
        else:
            # Linear interpolation based on engine size
            min_ef = emission_factors["two_wheeler"][category][fuel_type]["min"]
            max_ef = emission_factors["two_wheeler"][category][fuel_type]["max"]
            ratio = min(1.0, (engine_cc - 150) / 1350)  # Normalize to 0-1
            emission_factor = min_ef + ratio * (max_ef - min_ef)
        
        col1, col2 = st.columns(2)
        with col1:
            rideshare = st.checkbox("Rideshare")
        with col2:
            if rideshare:
                people_count = st.slider("Number of people sharing", 1, 2, 1)
            else:
                people_count = 1
        
        vehicle_type = "Two Wheeler"
        vehicle_name = f"{category} ({fuel_type}, {engine_cc}cc)"
        if rideshare:
            vehicle_name += f" with {people_count} people"

    elif private_vehicle_type == "Three Wheeler":
        col1, col2 = st.columns(2)
        with col1:
            engine_cc = st.slider("Engine (cc)", 50, 1000, 200)
        with col2:
            fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "electric", "cng"])
        
        # Calculate emission factor based on engine size
        min_ef = emission_factors["three_wheeler"][fuel_type]["min"]
        max_ef = emission_factors["three_wheeler"][fuel_type]["max"]
        ratio = min(1.0, (engine_cc - 50) / 950)  # Normalize to 0-1
        emission_factor = min_ef + ratio * (max_ef - min_ef)
        
        col1, col2 = st.columns(2)
        with col1:
            rideshare = st.checkbox("Rideshare")
        with col2:
            if rideshare:
                people_count = st.slider("Number of people sharing", 1, 3, 1)
            else:
                people_count = 1
        
        vehicle_type = "Three Wheeler"
        vehicle_name = f"Three Wheeler ({fuel_type}, {engine_cc}cc)"
        if rideshare:
            vehicle_name += f" with {people_count} people"

    elif private_vehicle_type == "Four Wheeler":
        col1, col2 = st.columns(2)
        with col1:
            car_type = st.selectbox(
                "Car Type", 
                ["small", "hatchback", "premium_hatchback", "compact_suv", "sedan", "suv", "hybrid"]
            )
        with col2:
            engine_cc = st.slider("Engine (cc)", 600, 4000, 1200)
        
        fuel_options = ["petrol", "diesel", "cng", "electric"]
        if car_type == "hybrid":
            fuel_options = ["petrol", "diesel", "electric"]
        
        col1, col2 = st.columns(2)
        with col1:
            fuel_type = st.selectbox("Fuel Type", fuel_options)
        
        # Calculate emission factor with uplift
        base_ef = emission_factors["four_wheeler"][car_type][fuel_type]["base"]
        uplift = emission_factors["four_wheeler"][car_type][fuel_type]["uplift"]
        
        # Adjust for engine size (larger engines emit more)
        if fuel_type != "electric":
            engine_factor = 1.0 + min(1.0, (engine_cc - 600) / 3400) * 0.5  # Up to 50% more for largest engines
        else:
            engine_factor = 1.0  # Electric doesn't scale with engine size in the same way
            
        emission_factor = base_ef * uplift * engine_factor
        
        col1, col2 = st.columns(2)
        with col1:
            rideshare = st.checkbox("Rideshare")
        with col2:
            if rideshare:
                people_count = st.slider("Number of people sharing", 1, 5, 1)
            else:
                people_count = 1
        
        vehicle_type = "Four Wheeler"
        vehicle_name = f"{car_type.replace('_', ' ').title()} ({fuel_type}, {engine_cc}cc)"
        if rideshare:
            vehicle_name += f" with {people_count} people"

if transport_category == "Public Transport" or transport_category == "Both Private and Public":
    st.subheader("Public Transport Details")
    col1, col2 = st.columns(2)
    with col1:
        transport_mode = st.selectbox("Mode", ["Taxi", "Bus", "Metro"], key="public_mode")
    
    if transport_mode == "Taxi":
        col1, col2 = st.columns(2)
        with col1:
            car_type = st.selectbox(
                "Car Type", 
                ["small", "hatchback", "sedan", "suv"],
                key="taxi_type"
            )
        with col2:
            fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "cng", "electric"], key="taxi_fuel")
        
        base_ef = emission_factors["public_transport"]["taxi"][car_type][fuel_type]["base"]
        uplift = emission_factors["public_transport"]["taxi"][car_type][fuel_type]["uplift"]
        public_emission_factor = base_ef * uplift
        
        public_people_count = st.slider("Number of people sharing", 1, 4, 1, key="taxi_people")
        
        # Only update main variables if only using public transport
        if transport_category == "Public Transport":
            emission_factor = public_emission_factor
            people_count = public_people_count
            vehicle_type = "Public Transport"
            vehicle_name = f"Taxi - {car_type.replace('_', ' ').title()} ({fuel_type})"
            if public_people_count > 1:
                vehicle_name += f" with {public_people_count} people"
    
    elif transport_mode == "Bus":
        public_fuel_type = st.selectbox("Fuel Type", ["diesel", "cng", "electric", "petrol"], key="bus_fuel")
        public_emission_factor = emission_factors["public_transport"]["bus"][public_fuel_type]
        # For buses, we assume a certain average occupancy already factored into emission factor
        public_people_count = 1
        
        # Only update main variables if only using public transport
        if transport_category == "Public Transport":
            emission_factor = public_emission_factor
            people_count = public_people_count
            vehicle_type = "Public Transport"
            vehicle_name = f"Bus ({public_fuel_type})"
    
    else:  # Metro
        public_emission_factor = emission_factors["public_transport"]["metro"]
        public_people_count = 1  # Already factored into emission factor
        
        # Only update main variables if only using public transport
        if transport_category == "Public Transport":
            emission_factor = public_emission_factor
            people_count = public_people_count
            vehicle_type = "Public Transport"
            vehicle_name = "Metro"

# Handle "Both" case by calculating combined emissions
if transport_category == "Both Private and Public":
    # Here we need to ask for usage ratio
    st.subheader("Usage Distribution")
    private_trips = st.number_input("Number of trips per day using private transport", min_value=0, max_value=10, value=2, step=1)
    total_trips = st.number_input("Total number of trips per day", min_value=1, max_value=10, value=4, step=1)
    private_ratio = private_trips / total_trips if total_trips > 0 else 0
    public_ratio = 1 - private_ratio
    
    # Calculate combined emission factor
    if private_ratio > 0 and public_ratio > 0:
        # Create a combined name for private transport
        if private_vehicle_type == "Two Wheeler":
            private_part = f"{category} ({fuel_type}, {engine_cc}cc)"
        elif private_vehicle_type == "Three Wheeler":
            private_part = f"Three Wheeler ({fuel_type}, {engine_cc}cc)"
        elif private_vehicle_type == "Four Wheeler":
            private_part = f"{car_type.replace('_', ' ').title()} ({fuel_type}, {engine_cc}cc)"
        
        # Create a combined name for public transport
        if transport_mode == "Taxi":
            public_part = f"Taxi - {car_type.replace('_', ' ').title()} ({fuel_type})"
        elif transport_mode == "Bus":
            public_part = f"Bus ({public_fuel_type})"
        else:  # Metro
            public_part = "Metro"
        
        # Calculate combined emission factor with proper division by people count
        combined_emission_factor = (emission_factor / people_count) * private_ratio + (public_emission_factor / public_people_count) * public_ratio
        emission_factor = combined_emission_factor
        people_count = 1  # Already factored in above
        
        vehicle_type = "Combined Transport"
        vehicle_name = f"{private_part} ({private_ratio*100:.0f}%) & {public_part} ({public_ratio*100:.0f}%)"
    
    


if st.button("Evaluate Customer"):
    # Create new_customer dictionary with collected values
    new_customer = {
        'Water': water_units,
        'Public_Transport': total_monthly_km if transport_category in ["Public Transport", "Both Private and Public"] else 0,
        'Private_Transport': total_monthly_km if transport_category in ["Private Transport", "Both Private and Public"] else 0
    }

    # First, calculate carbon footprint similar to "Calculate Carbon Footprint" button logic
    if 'total_monthly_km' in locals() and 'emission_factor' in locals():
        # Calculate total monthly emissions
        total_emissions = total_monthly_km * emission_factor
        
        # Store in session state
        st.session_state.calculated = True
        st.session_state.total_emissions = total_emissions
        
        # Generate alternative scenarios for comparison
        alternatives = {}
        
        # Add current vehicle
        alternatives[vehicle_name] = total_emissions
        
        # Generate alternatives for comparison
        # Add public transport options
        alternatives["Bus (Diesel)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["diesel"])
        alternatives["Bus (CNG)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["cng"])
        alternatives["Bus (Electric)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["electric"])
        alternatives["Metro"] = (total_monthly_km * emission_factors["public_transport"]["metro"])
        
        # Add car sharing options if not already selected
        if not (vehicle_type == "Four Wheeler" and "rideshare" in locals() and rideshare and people_count >= 3):
            alternatives["Car Pooling (4 people)"] = (total_monthly_km * emission_factors["four_wheeler"]["sedan"]["petrol"]["base"] * 
                                                    emission_factors["four_wheeler"]["sedan"]["petrol"]["uplift"]) / 4
        
        # Add electric vehicle options if not already selected
        if not (vehicle_type == "Four Wheeler" and "fuel_type" in locals() and fuel_type == "electric"):
            alternatives["Electric Car"] = (total_monthly_km * emission_factors["four_wheeler"]["sedan"]["electric"]["base"] * 
                                          emission_factors["four_wheeler"]["sedan"]["electric"]["uplift"])
        
        if not (vehicle_type == "Two Wheeler" and "fuel_type" in locals() and fuel_type == "electric"):
            alternatives["Electric Scooter"] = (total_monthly_km * emission_factors["two_wheeler"]["Scooter"]["electric"]["min"])
        
        st.session_state.emissions_data = alternatives
        
        # Generate recommendations
        recommendations = []
        
        # Basic recommendation based on emissions
        if total_emissions > 100:
            recommendations.append("Your carbon footprint from commuting is quite high. Consider switching to more sustainable transport options.")
        elif total_emissions > 50:
            recommendations.append("Your carbon footprint is moderate. There's room for improvement by considering more sustainable options.")
        else:
            recommendations.append("Your carbon footprint is relatively low, but you can still make improvements.")
        
        # Specific recommendations
        if vehicle_type == "Four Wheeler" and people_count == 1:
            recommendations.append("Consider carpooling to reduce emissions. Sharing your ride with 3 other people could reduce your emissions by up to 75%.")
        
        if "fuel_type" in locals() and fuel_type in ["petrol", "diesel"] and vehicle_type != "Public Transport":
            recommendations.append("Consider switching to an electric vehicle to significantly reduce your carbon footprint.")
        
        # Compare with public transit options for personal vehicles
        if vehicle_type in ["Four Wheeler", "Two Wheeler"]:
            bus_emissions = total_monthly_km * emission_factors["public_transport"]["bus"]["electric"]
            metro_emissions = total_monthly_km * emission_factors["public_transport"]["metro"]
            
            if total_emissions > 2 * bus_emissions:
                recommendations.append(f"Using an electric bus could reduce your emissions by approximately {(total_emissions - bus_emissions) / total_emissions * 100:.1f}%.")
            
            if total_emissions > 2 * metro_emissions:
                recommendations.append(f"Using metro could reduce your emissions by approximately {(total_emissions - metro_emissions) / total_emissions * 100:.1f}%.")
        
        st.session_state.recommendations = recommendations
    
    # Now continue with the existing customer evaluation logic
    if 'feature_stats' not in st.session_state or 'weights' not in st.session_state:
        st.error("Please generate scores first.")
    else:
        inverse_scoring = True
        
        z_vals = {}
        for feat, val in new_customer.items():
            if feat in features:  
                stats = st.session_state.feature_stats.get(feat)
                if stats:
                    z = (stats['mean'] - val)/stats['std'] if inverse_scoring else (val - stats['mean'])/stats['std']
                    z = np.clip(z, -1, 1)  
                    z_vals[feat] = z
        
        if "Environment_Score" in new_customer:
            if "Environment_Score" in features:
                stats = st.session_state.feature_stats.get("Environment_Score")
                if stats:
                    z = (stats['mean'] - new_customer["Environment_Score"])/stats['std'] if inverse_scoring else (new_customer["Environment_Score"] - stats['mean'])/stats['std']
                    z = np.clip(z, -1, 1)  
                    z_vals["Environment_Score"] = z
            else:
                if 'company_data' in locals() or 'company_data' in globals():
                    if "Environment_Score" in company_data.columns:
                        mean = company_data["Environment_Score"].mean()
                        std = company_data["Environment_Score"].std()
                        z = (mean - new_customer["Environment_Score"])/std if inverse_scoring else (new_customer["Environment_Score"] - mean)/std
                        z = np.clip(z, -1, 1)  # Cap z-value between -1 and 1
                        z_vals["Environment_Score"] = z
        
        if "user_electricity" in st.session_state:
            electricity_value = st.session_state.user_electricity
            z_vals["Electricity"] = 0
            
        weights = st.session_state.weights
        z_score = sum(z_vals[f] * weights.get(f, 0) for f in features if f in z_vals)
        
        if "Environment_Score" in z_vals:
            z_score += z_vals["Environment_Score"] * weights.get("Environment_Score", 0)
        
        sust_score = 500 * (1 - np.tanh(z_score/2.5))
        
        norm_vals = {}
        for feat, val in new_customer.items():
            if feat in features:  # Only process numeric features for scoring
                cmin, cmax = st.session_state.feature_constraints.get(feat, (val, val))
                if cmax == cmin:
                    norm_vals[feat] = 1
                else:
                    if inverse_scoring:
                        norm_vals[feat] = ((cmax - val)/(cmax - cmin))*999 + 1
                    else:
                        norm_vals[feat] = ((val - cmin)/(cmax - cmin))*999 + 1
        
        if "Environment_Score" in new_customer:
            if "Environment_Score" in features:
                cmin, cmax = st.session_state.feature_constraints.get("Environment_Score", (new_customer["Environment_Score"], new_customer["Environment_Score"]))
                if cmax == cmin:
                    norm_vals["Environment_Score"] = 1
                else:
                    if inverse_scoring:
                        norm_vals["Environment_Score"] = ((cmax - new_customer["Environment_Score"])/(cmax - cmin))*999 + 1
                    else:
                        norm_vals["Environment_Score"] = ((new_customer["Environment_Score"] - cmin)/(cmax - cmin))*999 + 1
        
        weighted_score = sum(norm_vals[f] * weights.get(f, 0) for f in features if f in norm_vals)
        
        if "Environment_Score" in norm_vals:
            weighted_score += norm_vals["Environment_Score"] * weights.get("Environment_Score", 0)
        
        existing_sust = st.session_state.scored_data["Sustainability_Score"]
        existing_trad = st.session_state.scored_data["Weighted_Score"]
        sust_rank = (existing_sust > sust_score).sum() + 1
        trad_rank = (existing_trad > weighted_score).sum() + 1
        
        st.markdown("---")
        st.markdown("### Customer Score Results")
        
        col1, col2 = st.columns(2)
        with col1:
            if "user_electricity" in st.session_state:
                st.metric("Personal Electricity Usage", f"{st.session_state.user_electricity:.2f} kWh")
            if "Environment_Score" in new_customer:
                st.metric("Company Environment Score", f"{new_customer['Environment_Score']:.2f}")
            
            st.metric("Water Usage", f"{new_customer['Water']:.2f}")
            st.metric("Public Transport", f"{new_customer['Public_Transport']:.2f}")
            st.metric("Private Transport", f"{new_customer['Private_Transport']:.2f}")
            
            st.metric("Z-Score", f"{z_score:.2f}")
            st.metric("Sustainability Score", f"{sust_score:.2f}")
            st.metric("Sustainability Rank", f"{sust_rank}")
            st.metric("Legacy Weighted Score", f"{weighted_score:.2f}")
            st.metric("Legacy Rank", f"{trad_rank}")

            better_than = (existing_sust < sust_score).mean() * 100
            st.success(f"This customer performs better than **{better_than:.1f}%** of customers in the dataset (based on Z-Score)")

            z_description = "above" if z_score > 0 else "below"
            st.info(f"Performance: **{abs(z_score):.2f} SD {z_description} mean**")
        
        with col2:
            all_features_weights = {
                feat: abs(weights.get(feat, 0))
                for feat in z_vals.keys()
            }
            all_features_weights.setdefault('Environment_Score', 0)
            if all_features_weights['Environment_Score'] == 0:
                all_features_weights['Environment_Score'] = 0.15
            
            if "user_electricity" in st.session_state:
                all_features_weights.setdefault('Electricity', 0)
                if all_features_weights['Electricity'] == 0:
                    all_features_weights['Electricity'] = 0.15
            
            all_features_weights = {
                f: w for f, w in all_features_weights.items()
                if w > 0
            }
            total = sum(all_features_weights.values())
            contribs = {f: w / total for f, w in all_features_weights.items()}
            explode = [0.1 if f in ['Environment_Score', 'Electricity'] else 0 for f in contribs]
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(
                list(contribs.values()),
                labels=list(contribs.keys()),
                autopct='%1.1f%%',
                startangle=90,
                explode=explode
            )
            ax.set_title('Feature Weightage for Customer')
            st.pyplot(fig)
            
            if "Environment_Score" in contribs:
                st.info(f"Environment Score contributes {contribs['Environment_Score']*100:.1f}%")
            if "Electricity" in contribs:
                st.info(f"Electricity Usage contributes {contribs['Electricity']*100:.1f}%")
        
        # Display carbon footprint results if calculation has been done
        if st.session_state.calculated:
            st.divider()
            st.header("Carbon Footprint Results")
            
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display the total emissions with a metric and color coding
                total_kg = st.session_state.total_emissions
                total_tonnes = total_kg / 1000
                
                if total_kg > 100:
                    emissions_color = "red"
                elif total_kg > 50:
                    emissions_color = "orange"
                else:
                    emissions_color = "green"
                
                st.metric(
                    "Monthly CO₂ Emissions",
                    f"{total_kg:.1f} kg CO₂e",
                )
                
                st.markdown(f"<div style='color:{emissions_color}; font-size:18px;'><strong>Sustainability Rating:</strong> {['Low', 'Moderate', 'High'][int(min(2, total_kg/50))]}</div>", unsafe_allow_html=True)
                
                # Context comparison
                avg_emissions = 200  # Example average emissions for commuting per person per month
                if total_kg < avg_emissions:
                    st.success(f"Your emissions are {(1 - total_kg/avg_emissions) * 100:.1f}% lower than the average commuter.")
                else:
                    st.warning(f"Your emissions are {(total_kg/avg_emissions - 1) * 100:.1f}% higher than the average commuter.")
            
            with col2:
                # Create a gauge chart for visual impact
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = total_kg,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Monthly CO₂ Emissions (kg)"},
                    gauge = {
                        'axis': {'range': [None, 300], 'tickwidth': 1},
                        'bar': {'color': emissions_color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "yellow"},
                            {'range': [100, 300], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': avg_emissions
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            # Show comparison chart of alternatives
            st.subheader("Comparison with Alternative Transport Options")
            
            # Create dataframe for plotting
            df = pd.DataFrame({
                'Transport Mode': list(st.session_state.emissions_data.keys()),
                'Monthly CO₂ Emissions (kg)': list(st.session_state.emissions_data.values())
            })
            
            # Sort by emissions for better visualization
            df = df.sort_values('Monthly CO₂ Emissions (kg)')
            
            # Create the comparison bar chart
            fig = px.bar(
                df, 
                y='Transport Mode', 
                x='Monthly CO₂ Emissions (kg)',
                orientation='h',
                color='Monthly CO₂ Emissions (kg)',
                color_continuous_scale='RdYlGn_r'
            )
            
            # Highlight the user's current transport mode
            current_mode = df['Transport Mode'].iloc[-1]  # Current mode should be the first one
            
            fig.update_layout(height=400, width=800)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recommendations
            st.header("Sustainability Recommendations")
            for i, rec in enumerate(st.session_state.recommendations):
                st.markdown(f"**{i+1}. {rec}**")
                
        st.markdown("---")
        st.markdown("### Score Distribution Analysis")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(existing_sust, kde=True, ax=ax)

        mean_sust = existing_sust.mean()
        std_sust = existing_sust.std()
        ax.axvline(mean_sust, color='red', linestyle='--', 
                label=f'Mean ({mean_sust:.2f})')

        ax.axvline(mean_sust + std_sust, color='green', linestyle=':', 
                label=f'+1 Std Dev ({mean_sust + std_sust:.2f})')
        ax.axvline(mean_sust - std_sust, color='orange', linestyle=':', 
                label=f'-1 Std Dev ({mean_sust - std_sust:.2f})')

        ax.axvline(sust_score, color='blue', linestyle='-', 
                label=f'This Customer ({sust_score:.2f}, Z={z_score:.2f})')

        ax.set_xlabel('Sustainability Score')
        ax.set_title('Distribution of Sustainability Scores')
        ax.legend()
        st.pyplot(fig)


        st.markdown("---")
        st.markdown("### Customer Input Weightage Analysis")

        core_features = ['Water', 'MPCE', 'Public_Transport', 'Private_Transport', 'Electricity', 'Environment_Score']
        input_data = {}
        input_weight = {}

        for feat in core_features:
            input_data[feat] = new_customer.get(feat, 0)  # Default to 0 if not present
            input_weight[feat] = 0.01  # Minimal weight to ensure visibility in chart

        z_vals = {}
        if 'feature_stats' in st.session_state:
            for feat, val in new_customer.items():
                if feat in features:  # Only process numeric features for scoring
                    stats = st.session_state.feature_stats.get(feat)
                    if stats and stats['std'] > 0:
                        z = (stats['mean'] - val)/stats['std'] if inverse_scoring else (val - stats['mean'])/stats['std']
                        z = np.clip(z, -1, 1)  # Cap z-value between -1 and 1
                        z_vals[feat] = z

        for feat in features:
            if feat in new_customer:
                input_data[feat] = new_customer[feat]
                
                weight = weights.get(feat, 0)
                if weight != 0 and feat in z_vals:
                    input_weight[feat] = abs(z_vals.get(feat, 0) * weight)

        if "user_electricity" in st.session_state:
            input_data["Electricity"] = st.session_state.user_electricity
            if st.session_state.user_electricity > 0:
                electricity_weight = max(0.125, st.session_state.user_electricity / 1000)
                elec_weight_factors = weights.get("Electricity_State", 0) + weights.get("Electricity_MPCE", 0)
                
                if elec_weight_factors > 0:
                    input_weight["Electricity"] = electricity_weight * elec_weight_factors
                else:
                    input_weight["Electricity"] = electricity_weight

        if "MPCE" in new_customer and new_customer["MPCE"] is not None:
            input_data["MPCE"] = new_customer["MPCE"]
            mpce_weight = 0.15  # Base weight to ensure visibility
            
            if "MPCE" in z_vals:
                mpce_weight = max(0.15, abs(z_vals["MPCE"] * (weights.get("MPCE", 0) or 0.15)))
            else:
                if new_customer["MPCE"] > 0:
                    mpce_weight = max(0.15, 0.1 + (new_customer["MPCE"] / 10000))
                    
            input_weight["MPCE"] = mpce_weight

        if "Environment_Score" in new_customer and new_customer["Environment_Score"] is not None:
            input_data["Environment_Score"] = new_customer["Environment_Score"]
            
            env_weight = 0.15  # Higher base value to ensure visibility
            
            if "Environment_Score" in z_vals:
                env_weight = max(0.15, abs(z_vals["Environment_Score"] * (weights.get("Environment_Score", 0) or 0.15)))
            else:
                if 'company_data' in locals() or 'company_data' in globals():
                    if "Environment_Score" in company_data.columns:
                        mean = company_data["Environment_Score"].mean()
                        std = company_data["Environment_Score"].std()
                        if std > 0:  # Avoid division by zero
                            env_z_score = (mean - new_customer["Environment_Score"])/std if inverse_scoring else (new_customer["Environment_Score"] - mean)/std
                            env_z_score = np.clip(env_z_score, -1, 1)
                            env_weight = max(0.15, abs(env_z_score * (weights.get("Environment_Score", 0) or 0.15)))
            
            input_weight["Environment_Score"] = env_weight

        input_weight = {k: max(v, 0.01) for k, v in input_weight.items()}

        for feat in core_features:
            if feat in input_data:
                input_weight[feat] = max(input_weight.get(feat, 0), 0.01)

        if input_weight:
            total_weight = sum(input_weight.values())
            if total_weight > 0:  # Prevent division by zero
                weighted_inputs = {k: v/total_weight for k, v in input_weight.items()}
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    sorted_inputs = sorted(weighted_inputs.items(), key=lambda x: x[1], reverse=True)
                    input_labels = [f[0] for f in sorted_inputs]
                    input_sizes = [f[1] for f in sorted_inputs]
                    
                    if len(input_labels) < len(core_features):
                        for feat in core_features:
                            if feat not in input_labels and feat in input_data:
                                input_labels.append(feat)
                                input_sizes.append(0.005)  # Small but visible slice
                    
                    if len(input_labels) > 10:
                        other_weight = sum(input_sizes[10:])
                        input_labels = input_labels[:10] + ["Others"]
                        input_sizes = input_sizes[:10] + [other_weight]
                    
                    fig = px.pie(
                        values=input_sizes,
                        names=input_labels,
                        title='Customer Input Weightage Distribution'
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(t=40, l=0, r=0, b=0),
                        height=400
                    )
                    st.plotly_chart(fig)
                
                with col2:
                    input_table = []
                    
                    for feat in sorted(input_data.keys()):
                        if feat in features or feat in core_features:  # Include core features regardless of being in features
                            weight_val = input_weight.get(feat, 0)
                            contrib_pct = (weight_val/total_weight)*100 if total_weight > 0 else 0.00
                            input_table.append({
                                "Input Field": feat,
                                "Value": input_data[feat],
                                "Weightage Score": weight_val,
                                "Contribution %": f"{contrib_pct:.2f}%"
                            })
                    
                    input_df = pd.DataFrame(input_table)
                    st.write("Individual Input Weightage Scores")
                    st.dataframe(input_df)
                           
        
    
    
st.title("Data Downloads and Group Upload Interface")

# -- Download Buttons for Existing Tables --
if "synth_data_raw" in st.session_state:
    synth_df = st.session_state.synth_data_raw[[col for col in st.session_state.synth_data_raw.columns if col in ['ID','Water','Public_Transport','Private_Transport']]]
    csv = synth_df.to_csv(index=False)
    st.download_button(
        label="Download Synthetic Data Preview CSV",
        data=csv,
        file_name="synthetic_data_preview.csv",
        mime="text/csv"
    )

if "scored_data" in st.session_state:
    scored_df = st.session_state.scored_data.copy()
    cols = ['ID', 'Weighted_Score', 'Z_Score', 'Sustainability_Score', 'Rank', 'Environment_Score']
    extra = [c for c in scored_df.columns if c not in cols and not c.endswith("_normalized") and not c.endswith("_z_score")]
    export_df = scored_df[cols + extra]
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="Download Ranked Individuals CSV",
        data=csv,
        file_name="ranked_individuals.csv",
        mime="text/csv"
    )

if "synth_data_raw" in st.session_state and "scored_data" in st.session_state:
    raw = st.session_state.synth_data_raw
    scored = st.session_state.scored_data
    ratio = raw['Electricity'] / raw['MPCE']
    quartiles = pd.qcut(ratio, 4, labels=['Excellent','Good','Fair','Poor'])
    aff_df = pd.DataFrame({
        'ID': raw['ID'],
        'MPCE': raw['MPCE'],
        'Electricity': raw['Electricity'],
        'Ratio': ratio,
        'Quartile': quartiles,
        'Sustainability_Score': scored['Sustainability_Score']
    })
    st.dataframe(aff_df)
    csv = aff_df.to_csv(index=False)
    st.download_button(
        label="Download Affordability Quartiles CSV",
        data=csv,
        file_name="affordability_quartiles.csv",
        mime="text/csv"
    )

if "synth_data_raw" in st.session_state:
    template_cols = [c for c in st.session_state.synth_data_raw.columns if c not in ['Weighted_Score','Z_Score','Sustainability_Score','Rank']]
    group_template = pd.DataFrame(columns=template_cols)
    template_csv = group_template.to_csv(index=False)
    st.download_button(
        label="Download Group Data Template",
        data=template_csv,
        file_name="group_data_template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload Group Data CSV for Scoring and Ranking", type=['csv'])
    if uploaded is not None:
        group_df = pd.read_csv(uploaded)
        if 'weights' in st.session_state:
            scored_group, _ = compute_weighted_score(group_df, st.session_state.weights, inverse=True, state_specific=st.session_state.get('use_state_specific', False))
            cols = ['ID', 'Weighted_Score', 'Z_Score', 'Sustainability_Score', 'Rank']
            display = scored_group[cols]
            st.dataframe(display)
            csv = display.to_csv(index=False)
            st.download_button(
                label="Download Scored Group Data",
                data=csv,
                file_name="scored_group_data.csv",
                mime="text/csv"
            )
        else:
            st.error("Please generate and configure weights before uploading group data.")

def generate_pdf(scored_df, customer_df):
    """Generate a PDF report for the sustainability analysis"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Sustainability Analysis Report', 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Customer Sustainability Profile', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    for col in customer_df.columns:
        if col in ['Electricity', 'Water', 'Public_Transport', 'Private_Transport', 'Sustainability_Score']:
            value = customer_df[col].iloc[0]
            pdf.cell(0, 10, f'{col}: {value:.2f}', 0, 1)
    
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Comparison with Dataset', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    for col in ['Electricity', 'Water', 'Public_Transport', 'Private_Transport']:
        if col in scored_df.columns and col in customer_df.columns:
            value = customer_df[col].iloc[0]
            percentile = (scored_df[col] < value).mean() * 100
            pdf.cell(0, 10, f'{col} Percentile: {percentile:.1f}%', 0, 1)
    
    return pdf.output(dest='S').encode('latin-1')
def add_pdf_section():
    """Add a PDF generation section to the Streamlit app"""
    st.header("Generate Sustainability Report (Not Fully Implemented)")
  
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Generate a comprehensive PDF report with your sustainability assessment.")
    
    with st.expander("Debug Session State", expanded=False):
        st.write("Session State Keys:", list(st.session_state.keys()))
        st.write("Required Keys Check:")
        st.write("- scored_data present:", "scored_data" in st.session_state)
        st.write("- user_electricity present:", "user_electricity" in st.session_state)
        st.write("- water_units present:", "water_units" in st.session_state)
        if "water_units" in st.session_state:
            st.write("- water_units value:", st.session_state.get("water_units", 0))
    
    with col2:
        has_required_data = (
            ("scored_data" in st.session_state or st.session_state.get("scored_data", pd.DataFrame()) is not None) and
            ("user_electricity" in st.session_state or st.session_state.get("user_electricity", 0) > 0)
        )
        
        test_customer = {
            'Environment_Score': st.session_state.get('company_data', {}).get('Environment_Score', [0]).iloc[0] 
                if isinstance(st.session_state.get('company_data', {}), pd.DataFrame) else 0,
            'Electricity': st.session_state.get('user_electricity', 0),
            'Water': st.session_state.get('water_units', 0),
            'Public_Transport': st.session_state.get('public_distance', 0),
            'Private_Transport': st.session_state.get('private_distance', 0),
            'MPCE': st.session_state.get('user_mpce_range', [0])[0] if isinstance(st.session_state.get('user_mpce_range', []), list) else 0
        }
        
        if 'Sustainability_Score' in st.session_state:
            test_customer['Sustainability_Score'] = st.session_state.Sustainability_Score
        
        if st.button("Generate PDF Report", key="gen_pdf"):
            with st.spinner("Generating your sustainability report..."):
                try:
                    scored_df = st.session_state.get('scored_data', pd.DataFrame())
                    if scored_df is None or scored_df.empty:
                        scored_df = pd.DataFrame([test_customer])
                    
                    pdf_file = generate_pdf(
                        scored_df, 
                        pd.DataFrame([test_customer])
                    )
                    
                    if pdf_file:
                        st.success("PDF successfully generated!")
                        st.download_button(
                            label="Download Sustainability Report",
                            data=pdf_file,
                            file_name="sustainability_report.pdf",
                            mime="application/pdf",
                            key="download_pdf"
                        )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.error("Debug info: " + str({k: type(v) for k, v in st.session_state.items()}))
        else:
            if not has_required_data:
                st.warning("Please enter all required data in the Test New Customer section first")


if __name__ == "__main__":
    st.markdown("""
    <style>
    .css-1v3fvcr {
        background-color: #f8f9fa;
    }
    .css-1kyxreq {
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)
    add_pdf_section()
