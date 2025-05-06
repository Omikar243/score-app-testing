import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
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

# List of Indian states and union territories
INDIAN_STATES_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", 
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", 
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", 
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

# Utility functions
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
    
    # Generate synthetic data for specified features
    for feat, (dtype, params) in feature_settings.items():
        data[feat] = generate_feature_data(n, dtype, params)
    
    # Create DataFrame with an ID column
    df = pd.DataFrame(data)
    df.insert(0, "ID", range(1, n + 1))
    
    # Sample from real electricity data if provided
    if electricity_data is not None:
        # Sample rows with replacement if needed
        if len(electricity_data) < n:
            sampled_data = electricity_data.sample(n=n, replace=True)
        else:
            sampled_data = electricity_data.sample(n=n, replace=False)
        
        # Add electricity, MPCE and State_UT from the real data
        df["Electricity"] = sampled_data["qty_usage_in_1month"].values
        df["MPCE"] = sampled_data["mpce"].values
        df["State_UT"] = sampled_data["state_name"].values
    else:
        # Fallback to random state assignments if no real data
        df["State_UT"] = np.random.choice(INDIAN_STATES_UTS, size=n)
    
    return df

def compute_feature_stats(df, by_state=False):
    stats = {}
    
    if by_state:
        # Compute stats by state
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
        # Overall stats
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
    dist_water, params_water = feature_distribution_ui("Water", default_min=200, default_max=600)

with tab2:
    st.markdown("### Public Transport")
    include_public = st.checkbox("Include Public Transport", value=True)
    if include_public:
        public_dist, public_params = feature_distribution_ui("Public_Transport", default_min=0, default_max=500)
    else:
        public_dist, public_params = None, None

with tab3:
    st.markdown("### Private Transport")
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


    st.markdown("---\n## Weighted Score Settings")
    
    st.markdown("### 1. Electricity Consumption (Total weight: 0.25)")
    elec_col1, elec_col2 = st.columns(2)
    
    with elec_col1:
        st.markdown("**Location Based (0.125)**") 
        location_col1, location_col2 = st.columns(2)

        with location_col1:
            w_elec_state = st.selectbox("State/UT", sorted(INDIAN_STATES_UTS), key="selected_state_ut")

        with location_col2:
            w_elec_sector = st.selectbox("Area Type", ["Rural", "Urban"], key="selected_area_type")

        w_elec_location = 0.125  # Fixed weight for location based analysis
    
    with elec_col2:
        st.markdown("**Economic Based (0.125)**")
        w_elec_mpce = st.number_input("MPCE Weight", value=0.125, step=0.01, format="%.3f", key="wt_elec_mpce")
    
    elec_total = float(w_elec_location) + float(w_elec_mpce)
    if elec_total > 0.25:
        st.error("Electricity weights sum exceeds 0.25!")

    st.markdown("### 2. Water Consumption (Total weight: 0.25)")
    w_water = st.number_input("Water Weight", value=0.25, step=0.01, format="%.3f", key="wt_water")
    
    st.markdown("### 3. Commute (Total weight: 0.25)")
    comm_col1, comm_col2 = st.columns(2)
    
    with comm_col1:
        w_public = st.number_input("Public Transport Weight", value=0.125, step=0.01, format="%.3f", key="wt_public")
    
    with comm_col2:
        w_private = st.number_input("Private Transport Weight", value=0.125, step=0.01, format="%.3f", key="wt_private")
    
    commute_total = w_public + w_private
    if commute_total > 0.25:
        st.error("Commute weights sum exceeds 0.25!")

    st.markdown("### 4. Company Environmental Score (Total weight: 0.25)")
    w_company = st.number_input("Company Environmental Score Weight", value=0.25, step=0.01, format="%.3f", key="wt_company")
    
    weights = {
        "Electricity_State": float(w_elec_location),   # State-based electricity weight
        "Electricity_Sector": 0.0,                     # Rural/Urban electricity weight
        "Electricity_MPCE": float(w_elec_mpce),       # Economic-based electricity weight
        "Water": float(w_water),
        "Public_Transport": float(w_public),
        "Private_Transport": float(w_private),
        "Environment_Score": float(w_company)         # Company environmental score weight
    }
    
    total = sum(weights.values())
    st.markdown(f"**Total Weight:** {total:.3f}")
    
    use_state_specific = st.checkbox("Use State/UT Specific Scoring", value=True)
    
    if abs(total - 1.0) > 1e-3:
        st.error("Total weights must sum exactly to 1!")
    else:
        if st.button("Generate Weighted Score"):
            if 'user_electricity' not in st.session_state:
                st.session_state.user_electricity = 0

            company_data = load_company_data()
            
            if 'synth_data_raw' in st.session_state and company_data is not None:
                merged_data = st.session_state.synth_data_raw.copy()
                # Add random company environmental scores if no company data
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
                scored, _ = compute_weighted_score(
                    merged_data, weights, inverse=True, state_specific=use_state_specific
                )
                st.session_state.scored_data = scored
                st.session_state.weights = weights
                
                # Store sub-weights for detailed analysis
                st.session_state.sub_weights = {
                    "electricity": {
                        "state": w_elec_state,
                        "sector": w_elec_sector,
                        "mpce": w_elec_mpce
                    },
                    "commute": {
                        "public": w_public,
                        "private": w_private
                    },
                    "water": {
                        "water": w_water
                    },
                    "company": {
                        "env_score": w_company
                    }
                }
                st.success("Weighted score and ranking generated!")
            else:
                st.error("No data available for scoring. Please generate analysis data first.")

    if "scored_data" in st.session_state:
        st.markdown("### Ranked Individuals Based on Sustainability Score")
        scored = st.session_state.scored_data
        core = ["ID", "Weighted_Score", "Z_Score", "Sustainability_Score", "Rank", "Environment_Score"]
        extras = [
            col for col in (scored.columns if scored is not None else [])
            if col not in core
            and not col.endswith("_normalized")  
            and not col.endswith("_z_score")
            and col not in ["Sector_classification", "ESG_Rating", "Total_Employees"]
        ]
        display_cols = list(dict.fromkeys(core + extras))

        if scored is not None:
            render_sortable_table(scored[display_cols].reset_index(drop=True))
        else:
            st.warning("No scored data available to display.")
    



# Show distribution graphs and insights
if "synth_data_raw" in st.session_state:
    if st.button("Show Distribution Graphs"):
        for col in st.session_state.synth_data_raw.columns:
            if col not in ["ID", "MPCE", "Company_Name", "Sector classification",
                           "Environment Score", "ESG Rating", "Category",
                           "Date of Rating", "Total Employees", "State_UT"]:
                fig, ax = plt.subplots()
                sns.histplot(st.session_state.synth_data_raw[col], bins=30, kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                if col == 'Electricity':
                    ue = st.session_state.user_electricity
                    ax.axvline(ue, color='red', linestyle='--', label='Your Usage')
                    ax.legend()
                st.pyplot(fig)
        
        if "scored_data" in st.session_state:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot both scores for comparison
            sns.histplot(st.session_state.scored_data["Sustainability_Score"], 
                        bins=30, kde=True, ax=ax, color='blue', label='Sustainability Score')
            ax.set_title("Distribution of Sustainability Score")
            
            # Add secondary y-axis for Weighted Score
            ax2 = ax.twinx()
            sns.histplot(st.session_state.scored_data["Weighted_Score"], 
                        bins=30, kde=True, ax=ax2, color='red', alpha=0.5, label='Legacy Weighted Score')
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            st.pyplot(fig)
            
            # Also show Z-Score distribution
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.scored_data["Z_Score"], bins=30, kde=True, ax=ax)
            ax.set_title("Distribution of Z-Score")
            ax.axvline(0, color='red', linestyle='--', label='Mean (0)')
            ax.axvline(-1, color='green', linestyle='--', label='Lower Cap (-1)')
            ax.axvline(1, color='green', linestyle='--', label='Upper Cap (1)')
            ax.legend()
            st.pyplot(fig)
        
        # Add company comparison graphs
        if company_data is not None:
            st.subheader("Company Environmental Performance Comparison")
            
            # Top companies by Environment Score
            company_scores = company_data.sort_values("Environment_Score", ascending=False)
            top_n = min(20, len(company_scores))  # Top 20 or all if less than 20
            top_companies = company_scores.head(top_n)
            
            # Create plotly bar chart
            fig = px.bar(top_companies,
                x="Environment_Score", 
                y="Company_Name",
                title="Top Companies by Environmental Score",
                orientation='h',  # Horizontal bars
                color="Environment_Score",
                color_continuous_scale="viridis")
            
            # Update layout
            fig.update_layout(
            xaxis_title="Environment Score",
            yaxis_title="Company",
            yaxis=dict(autorange="reversed"),  # Reverse y-axis to show highest score at top
            height=600
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Sector comparison
            if "Sector_classification" in company_data.columns:
                # Calculate sector averages
                sector_avg = company_data.groupby("Sector_classification")["Environment_Score"].agg(
                    ['mean', 'count', 'std']
                ).sort_values('mean', ascending=False)

                # Create plotly figure
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

                # Update layout for better readability
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=600,
                    showlegend=False
                )

                # Add count annotations on top of bars
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
                
                # Boxplot showing score distribution by sector
                # Box plot with plotly
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

    # Add insights section
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
            features = [
                c for c in st.session_state.synth_data_raw.columns
                if c not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Sustainability_Score",
                        "Company_Name", "Sector_classification", "State_UT",
                        "Environment_Score", "Total_Employees"]
            ]
            feature_to_plot = st.selectbox("Select Feature", features)
            fig, ax = plt.subplots()
            ax.scatter(st.session_state.synth_data_raw[feature_to_plot],
                    st.session_state.scored_data["Sustainability_Score"])
            ax.set_xlabel(feature_to_plot)
            ax.set_ylabel("Sustainability Score")
            ax.set_title(f"{feature_to_plot} vs Sustainability Score")
            st.pyplot(fig)
        
        
            
        
        # Optional: Add sector analysis if sector data is available
        if "Sector_classification" in st.session_state.scored_data.columns and "Environment_Score" in st.session_state.scored_data.columns:
            st.markdown("---\n### Environmental Performance by Sector")
            
            # Group by sector and calculate statistics
            sector_analysis = st.session_state.scored_data.groupby("Sector_classification").agg({
                "Environment_Score": ["mean", "min", "max", "count"]
            }).reset_index()
            sector_analysis.columns = ["Sector", "Mean Score", "Min Score", "Max Score", "Company Count"]
            sector_analysis = sector_analysis.sort_values(by="Mean Score", ascending=False)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x="Mean Score", y="Sector", data=sector_analysis, ax=ax)
            ax.set_title("Average Environmental Score by Sector")
            ax.set_xlabel("Mean Environmental Score")
            ax.set_ylabel("Sector")
            st.pyplot(fig)
            
            # Show detailed sector data
            with st.expander("View Detailed Sector Analysis"):
                st.dataframe(sector_analysis, use_container_width=True)
        
        # Optional: Add more advanced analytics
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
                "Sustainability_Score": st.session_state.scored_data["Sustainability_Score"]
            })
            st.dataframe(affordability_df.groupby("Quartile").agg({
                "MPCE": "mean",
                "Electricity": "mean",
                "Ratio": "mean",
                "Sustainability_Score": "mean"
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

if test_mode == "Manual Entry":
    st.markdown("### Enter Customer Details")
# CRISIL ESG - Environmental Scoring Only
    # Get sectors from company data if available
    st.markdown("## CRISIL ESG - Environmental Scoring Only")

    if company_data is not None:
        # Add analysis type selection
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
            
            # Filter companies by employee range
            df_filtered = df_sector[
                (df_sector['Total_Employees'] >= emp_range[0]) & 
                (df_sector['Total_Employees'] <= emp_range[1])
            ]
            
            if len(df_filtered) > 0:
                # Calculate baseline statistics
                baseline_mean = df_filtered['Environment_Score'].mean()
                baseline_std = df_filtered['Environment_Score'].std(ddof=1)
                
                # Display filtered companies
                st.markdown(f"### Companies in {industry} sector with {emp_range[0]}-{emp_range[1]} employees")
                st.markdown(f"**Baseline Environment Score:** {baseline_mean:.2f} (std: {baseline_std:.2f})")
                
                # Calculate Z-scores
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
                
                # Create histogram with KDE
                sns.histplot(df_results['Environment_Score'], kde=True, ax=ax)
                
                # Add vertical line for mean
                ax.axvline(baseline_mean, color='red', linestyle='--', label=f'Mean ({baseline_mean:.2f})')
                
                # Add one standard deviation markers
                ax.axvline(baseline_mean + baseline_std, color='green', linestyle=':', 
                        label=f'+1 Std Dev ({baseline_mean + baseline_std:.2f})')
                ax.axvline(baseline_mean - baseline_std, color='orange', linestyle=':', 
                        label=f'-1 Std Dev ({baseline_mean - baseline_std:.2f})')
                
                ax.set_xlabel('Environment Score')
                ax.set_title(f'Distribution of Environment Scores for {industry} ({emp_range[0]}-{emp_range[1]} employees)')
                ax.legend()
                
                st.pyplot(fig)
                
                # Company comparison section
                st.markdown("### Compare Your Company")
                
                # Option to select a company or input a custom score
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
                        
                        # Display company comparison (now focusing on Z-score)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Environment Score", f"{company_score:.2f}")
                        with col2:
                            st.metric("Z-Score", f"{company_z:.2f}", 
                                    f"{company_z:.2f} SD from mean")
                        with col3:
                            st.metric("Normalized Score", f"{company_norm:.2f}/100")
                            
                        # Position in distribution
                        better_than = (df_results['Env_Z_Score'] < company_z).mean() * 100
                        st.success(f"**{selected_company}** performs better than **{better_than:.1f}%** of companies in this segment (based on Z-Score)")
                else:
                    custom_score = st.number_input("Enter Your Company's Environment Score", 
                                                min_value=0.0, max_value=100.0, value=50.0,
                                                key="employee_range_custom_score")
                    
                    # Calculate metrics
                    custom_z = (custom_score - baseline_mean) / baseline_std
                    
                    # Normalize
                    custom_norm = ((custom_z - min_z) / (max_z - min_z)) * 100 if max_z != min_z else 50
                    
                    # Display comparison (now focusing on Z-score)
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
                
                # Calculate Z-score
                company_z = (company_score - overall_mean) / overall_std
                
                # Calculate percentile based on Z-score
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
                
                # Create histogram with KDE
                sns.histplot(company_data['Environment_Score'], kde=True, ax=ax)
                
                # Add vertical line for mean
                ax.axvline(overall_mean, color='red', linestyle='--', label=f'Mean ({overall_mean:.2f})')
                
                # Add company score marker
                ax.axvline(company_score, color='blue', linestyle='-', 
                        label=f'{selected_company} ({company_score:.2f}, Z={company_z:.2f})')
                
                ax.set_xlabel('Environment Score')
                ax.set_title(f'Distribution of Environment Scores Across All Companies')
                ax.legend()
                
                st.pyplot(fig)
                
                # Compare with sector
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
                
                # Show all performers in sector sorted from top to bottom
                st.markdown(f"### All Performers in {company_sector} (Highest to Lowest)")
                
                # Calculate Z-scores for all companies in this sector
                all_sector_companies = sector_data.copy()
                all_sector_companies['Sector_Z_Score'] = (all_sector_companies['Environment_Score'] - sector_mean) / sector_std
                all_sector_companies = all_sector_companies.sort_values('Environment_Score', ascending=False)
                
                # Highlight the selected company in the dataframe
                # Reset index before styling to ensure row indices match
                display_df = all_sector_companies[[
                    'Company_Name',
                    'Environment_Score',
                    'Sector_Z_Score',
                    'ESG_Rating',
                    'Total_Employees'
                ]].reset_index(drop=True)
                
                # Find the row index of the selected company after reset_index
                selected_idx = display_df.index[display_df['Company_Name'] == selected_company].tolist()
                
                # Apply styling to highlight the selected company
                styled_df = display_df.style.apply(
                    lambda x: ['background-color: lightskyblue' if i in selected_idx else '' 
                            for i in range(len(display_df))],
                    axis=0
                )
                
                st.dataframe(styled_df)
            else:
                # Enter custom score option
                custom_score = st.number_input("Enter Your Company's Environment Score", 
                                            min_value=0.0, max_value=100.0, value=50.0,
                                            key="company_only_custom_score")
                
                # Calculate Z-score
                custom_z = (custom_score - overall_mean) / overall_std
                
                # Calculate percentile based on Z-score
                percentile = (company_data['Environment_Score'] < custom_score).mean() * 100
                
                # Display comparison metrics (now focusing on Z-score)
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
                
                # Create histogram with KDE
                sns.histplot(company_data['Environment_Score'], kde=True, ax=ax)
                
                # Add vertical line for mean
                ax.axvline(overall_mean, color='red', linestyle='--', label=f'Mean ({overall_mean:.2f})')
                
                # Add custom score marker
                ax.axvline(custom_score, color='blue', linestyle='-', 
                        label=f'Your Company ({custom_score:.2f}, Z={custom_z:.2f})')
                
                ax.set_xlabel('Environment Score')
                ax.set_title(f'Distribution of Environment Scores Across All Companies')
                ax.legend()
                
                st.pyplot(fig)
                
                # Sector selection for comparison
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
                        
                    # Calculate Z-scores for all companies in this sector
                    all_sector_companies = sector_data.copy()
                    all_sector_companies['Sector_Z_Score'] = (all_sector_companies['Environment_Score'] - sector_mean) / sector_std
                    all_sector_companies = all_sector_companies.sort_values('Environment_Score', ascending=False)
                    
                    # Show all performers in the selected sector sorted from top to bottom
                    st.markdown(f"### All Performers in {selected_sector} (Highest to Lowest)")
                    
                    # Display all companies in this sector
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

    # Personal electricity benchmark
    st.markdown("---")
    st.markdown("## Your Personal Electricity Analysis")
    st.markdown("")  # Add some spacing
    col1, col2 = st.columns(2)

    with col1:
        # User state selection
        user_state = st.selectbox("Select Your State/UT", 
                                options=sorted(INDIAN_STATES_UTS),
                                key="personal_state")
        # Store the state in session state
        st.session_state.user_state = user_state
        
        # User sector selection
        user_sector_name = st.radio("Select Your Sector", 
                                ['Rural', 'Urban'],
                                key="personal_sector")
        user_sector = 1 if user_sector_name == 'Rural' else 2

        # MPCE Range Selection
        mpce_ranges = ["₹1-1,000", "₹1,000-5,000", "₹5,000-10,000", "₹10,000-25,000", "₹25,000+"]
        mpce_range_values = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 25000), (25000, float('inf'))]
        
        user_mpce_range_index = st.selectbox(
            "Select Your Monthly Per Capita Expenditure (MPCE) Range",
            options=range(len(mpce_ranges)),
            format_func=lambda x: mpce_ranges[x],
            key="personal_mpce_range"
        )
        
        # Store MPCE range in session state
        st.session_state.user_mpce_range = mpce_range_values[user_mpce_range_index]
        st.session_state.user_mpce_range_name = mpce_ranges[user_mpce_range_index]

    with col2:
        #User input for electricity consumption
        st.markdown("Enter your personal electricity consumption and compare it with the dataset")
        user_electricity = st.number_input("Your Monthly Electricity Usage (kWh)", 
                                        min_value=0.0, 
                                        value=0.0, 
                                        step=10.0,
                                        key="personal_electricity")
        
        # Store the value in session state so it's accessible elsewhere
        st.session_state.user_electricity = user_electricity

        # Calculate cost using data patterns if we have user input
        if user_electricity > 0:
            # Display user electricity consumption
            st.metric("Your Electricity Consumption", f"{user_electricity:.2f} kWh")
            
            # Add cost input regardless of whether we have state data
            user_cost = st.number_input("Enter your monthly electricity cost (₹)", 
                                        min_value=0.0, 
                                        step=100.0)
            
            # Now check if we have electricity data for the selected state/sector
            if electricity_data is not None:
                # Filter data for user's state and sector
                state_sector_data = electricity_data[
                    (electricity_data['state_name'] == user_state) & 
                    (electricity_data['sector'] == user_sector)
                ]
                
                if not state_sector_data.empty:
                    # Calculate state average consumption
                    state_avg = state_sector_data['qty_usage_in_1month'].mean()
                    
                    # Show where user falls in the distribution
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
                    
                    # Use overall statistics if available
                    if 'full_electricity_data' in st.session_state:
                        overall_avg = st.session_state.full_electricity_data['qty_usage_in_1month'].mean()
                        st.markdown(f"Overall average electricity consumption: **{overall_avg:.2f} kWh**")
                        
                        # Show overall distribution
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
        # Resource consumption section
        st.markdown("## Resource Consumption")
        
            # Water section
        st.markdown("Water")
        water_units = st.number_input("Water Units", min_value=0.0)
            
            # Get water min/max constraints from feature constraints
        water_min, water_max = 0.0, 1000.0
        if 'feature_constraints' in st.session_state and 'Water' in st.session_state.feature_constraints:
                water_min, water_max = st.session_state.feature_constraints['Water']
            
        
        
        # Transportation section
        st.markdown("### Transportation")
        col1, col2 = st.columns(2)
        
        with col1:
            # Public transport
            st.subheader("Public Transport")
            public_distance = st.number_input("Distance (km)", min_value=0.0)
            public_transport_type = st.selectbox("Type", ["Bus", "Train", "Metro", "Other"])
            
            # Get public transport min/max constraints
            public_min, public_max = 0.0, 500.0
            if 'feature_constraints' in st.session_state and 'Public_Transport' in st.session_state.feature_constraints:
                public_min, public_max = st.session_state.feature_constraints['Public_Transport']
        
        with col2:
            # Private transport
            st.subheader("Private Transport")
            private_distance = st.number_input("Distance (km)", min_value=0.0, key="private_distance")
            private_transport_type = st.selectbox("Type", ["Car", "Motorcycle", "Auto", "Other"])
            private_transport_fuel = st.selectbox("Vehicle Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
            
            # Get private transport min/max constraints
            private_min, private_max = 0.0, 300.0
            if 'feature_constraints' in st.session_state and 'Private_Transport' in st.session_state.feature_constraints:
                private_min, private_max = st.session_state.feature_constraints['Private_Transport']
        
        # Create a dictionary to store all customer data
        new_customer = {
            # Basic info
            
            
            # Resource consumption
            "Water": water_units,
            
            # Transportation
            "Public_Transport": public_distance,
            "Public_Transport_Type": public_transport_type,
            "Private_Transport": private_distance,
            "Private_Transport_Type": private_transport_type,
            "Private_Transport_fuel": private_transport_fuel
    }
    
    
else:
        # Download template based on features from scored data
        if "synth_data_raw" in st.session_state:
            # Define required columns that must be in the template
            
            # Get additional columns from synthetic data, excluding special columns
            # Define required columns that must be in the template
            required_cols = ["Electricity", "MPCE","Water", "Public_Transport", "Private_Transport", "Company_Name", "Sector_classification", "Environment_Score", "ESG_Rating", "Total_Employees"]

            # Get additional columns from synthetic data, excluding special columns 
            additional_cols = [c for c in st.session_state.synth_data_raw.columns 
                             if c not in ["ID", "Weighted_Score", "Z_Score", "Sustainability_Score", "Rank",
                                        "Company_Name", "Sector_classification", "Environment_Score", 
                                        "ESG_Rating", "Total_Employees"]]
            
            # Combine required and additional columns, ensuring no duplicates
            template_cols = list(dict.fromkeys(required_cols + additional_cols))
            tmpl = pd.DataFrame(columns=template_cols).to_csv(index=False).encode('utf-8')
            st.download_button("Download Test CSV Template",
                             data=tmpl, file_name="test_template.csv", mime="text/csv")
            
            # Upload and process test data
            up_test = st.file_uploader("Upload Test Data", type="csv", key="test_uploader")
            if up_test:
                test_df = pd.read_csv(up_test)
                st.dataframe(test_df)
                
                if st.button("Process Test Batch"):
                    if 'feature_stats' not in st.session_state or 'weights' not in st.session_state:
                        st.error("Please generate scores first.")
                    else:
                        # Process each test customer
                        test_with_scores = []
                        for _, row in test_df.iterrows():
                            # Extract features present in the test data
                            new_customer = {f: row[f] for f in features if f in row}
                            
                            # Calculate z-scores for each feature
                            z_vals = {}
                            for feat, val in new_customer.items():
                                if feat in st.session_state.feature_stats:
                                    stats = st.session_state.feature_stats[feat]
                                    z = (stats['mean'] - val)/stats['std']  # Using inverse scoring
                                    z = np.clip(z, -1, 1)
                                    z_vals[feat] = z
                            
                            # Calculate weighted z-score
                            z_score = sum(z_vals[f] * st.session_state.weights.get(f, 0) for f in features if f in z_vals)
                            
                            # Calculate sustainability score using tanh transformation
                            sust_score = 500 * (1 - np.tanh(z_score/2.5))
                            
                            # Calculate traditional normalized score
                            norm_vals = {}
                            for feat, val in new_customer.items():
                                if feat in st.session_state.feature_constraints:
                                    cmin, cmax = st.session_state.feature_constraints[feat]
                                    if cmax > cmin:
                                        norm_vals[feat] = ((cmax - val)/(cmax - cmin))*999 + 1
                                    else:
                                        norm_vals[feat] = 500
                            
                            weighted_score = sum(norm_vals[f] * st.session_state.weights.get(f, 0) for f in features if f in norm_vals)
                            
                            # Calculate ranks relative to existing data
                            sust_rank = (st.session_state.scored_data["Sustainability_Score"] > sust_score).sum() + 1
                            trad_rank = (st.session_state.scored_data["Weighted_Score"] > weighted_score).sum() + 1
                            
                            # Store results
                            result = row.to_dict()
                            result.update({
                                "Z_Score": z_score,
                                "Sustainability_Score": sust_score,
                                "Sustainability_Rank": sust_rank,
                                "Weighted_Score": weighted_score,
                                "Legacy_Rank": trad_rank
                            })
                            test_with_scores.append(result)
                        
                        # Create and display results
                        results_df = pd.DataFrame(test_with_scores)
                        st.markdown("### Test Results")
                        st.dataframe(results_df)
                        
                        # Show score distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(data=results_df, x="Sustainability_Score", bins=20, kde=True)
                        ax.axvline(results_df["Sustainability_Score"].mean(), color='red', linestyle='--', label='Mean')
                        ax.set_title("Distribution of Test Sustainability Scores")
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Add feature importance analysis
                        st.markdown("### Feature Importance Analysis")
                        
                        # Calculate average feature contribution across test set
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
                            # Create pie chart
                            fig, ax = plt.subplots(figsize=(10, 10))
                            total = sum(feature_importance.values())
                            sizes = [v/total for v in feature_importance.values()]
                            plt.pie(sizes, labels=feature_importance.keys(), autopct='%1.1f%%')
                            plt.title('Average Feature Contribution')
                            st.pyplot(fig)
                            
                            # Show feature importance table
                            importance_df = pd.DataFrame([
                                {"Feature": k, "Importance": v, "Percentage": f"{(v/total)*100:.2f}%"}
                                for k, v in feature_importance.items()
                            ]).sort_values("Importance", ascending=False)
                            st.dataframe(importance_df)

if st.button("Evaluate Customer"):
        if 'feature_stats' not in st.session_state or 'weights' not in st.session_state:
            st.error("Please generate scores first.")
        else:
            # Define inverse_scoring value
            inverse_scoring = True
            
            # Calculate z-scores and sustainability score
            z_vals = {}
            for feat, val in new_customer.items():
                if feat in features:  # Only process numeric features for scoring
                    stats = st.session_state.feature_stats.get(feat)
                    if stats:
                        z = (stats['mean'] - val)/stats['std'] if inverse_scoring else (val - stats['mean'])/stats['std']
                        z = np.clip(z, -1, 1)  # Cap z-value between -1 and 1
                        z_vals[feat] = z
            
            # Calculate weighted z-score
            weights = st.session_state.weights
            z_score = sum(z_vals[f] * weights.get(f, 0) for f in features if f in z_vals)
            
            # Calculate sustainability score
            sust_score = 500 * (1 - np.tanh(z_score/2.5))
            
            # Also calculate traditional normalization for comparison
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
            
            weighted_score = sum(norm_vals[f] * weights.get(f, 0) for f in features if f in norm_vals)
            
            # Calculate ranks
            existing_sust = st.session_state.scored_data["Sustainability_Score"]
            existing_trad = st.session_state.scored_data["Weighted_Score"]
            sust_rank = (existing_sust > sust_score).sum() + 1
            trad_rank = (existing_trad > weighted_score).sum() + 1
            
            # Display customer score section
            st.markdown("---")
            st.markdown("### Customer Score Results")
            
            col1, col2 = st.columns(2)
            with col1:
                # Add electricity and company score metrics first
                if "user_electricity" in st.session_state:
                    st.metric("Personal Electricity Usage", f"{st.session_state.user_electricity:.2f} kWh")
                if "Environment_Score" in new_customer:
                    st.metric("Company Environment Score", f"{new_customer['Environment_Score']:.2f}")
                
                # Add resource consumption metrics
                st.metric("Water Usage", f"{new_customer['Water']:.2f}")
                st.metric("Public Transport", f"{new_customer['Public_Transport']:.2f}")
                st.metric("Private Transport", f"{new_customer['Private_Transport']:.2f}")
                
                # Original metrics
                st.metric("Z-Score", f"{z_score:.2f}")
                st.metric("Sustainability Score", f"{sust_score:.2f}")
                st.metric("Sustainability Rank", f"{sust_rank}")
                st.metric("Legacy Weighted Score", f"{weighted_score:.2f}")
                st.metric("Legacy Rank", f"{trad_rank}")

                # Add position in distribution (similar to CRISIL analysis)
                better_than = (existing_sust < sust_score).mean() * 100
                st.success(f"This customer performs better than **{better_than:.1f}%** of customers in the dataset (based on Z-Score)")

                # Add descriptive performance metric (from CRISIL ESG)
                z_description = "above" if z_score > 0 else "below"
                st.info(f"Performance: **{abs(z_score):.2f} SD {z_description} mean**")
            
            with col2:
                # Create pie chart for score distribution
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # FIX 1: Ensure all features with z-scores are included in pie chart
                all_features_weights = {}
                for f in features:
                    if f in z_vals:
                        # Use actual weight or small default to ensure visibility
                        all_features_weights[f] = max(abs(weights.get(f, 0)), 0.01)
                
                # Add electricity weights if available
                if "user_electricity" in st.session_state:
                    all_features_weights["Electricity"] = weights.get("Electricity_State", 0) + weights.get("Electricity_MPCE", 0)

                weights_sum = sum(all_features_weights.values())
                if weights_sum > 0:  # Prevent division by zero
                    contributions = {f: w/weights_sum for f, w in all_features_weights.items()}
                    
                    # Sort by contribution value
                    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
                    labels = [f[0] for f in sorted_contrib]
                    sizes = [f[1] for f in sorted_contrib]
                    
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Feature Weightage for Customer')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("No feature weights available for visualization")
            
            # Show feature contributions
            st.markdown("---")
            st.markdown("### Feature Z-Score Contributions")
            
            # FIX: Ensure all features with z-scores are included in contributions
            z_contrib = {f: z_vals[f] * weights.get(f, 0) for f in features if f in z_vals}
            
            # Add electricity contribution if available
            if "user_electricity" in st.session_state:
                elec_z_score = (st.session_state.feature_stats['Electricity']['mean'] - st.session_state.user_electricity) / st.session_state.feature_stats['Electricity']['std']
                elec_z_score = np.clip(elec_z_score, -1, 1)
                z_contrib["Electricity"] = elec_z_score * weights.get("Electricity_State", 0) + elec_z_score * weights.get("Electricity_MPCE", 0)
            
            # Add company environment score contribution if available
            if "Environment_Score" in new_customer:
                company_z_score = (new_customer["Environment_Score"] - company_data["Environment_Score"].mean()) / company_data["Environment_Score"].std()
                company_z_score = np.clip(company_z_score, -1, 1)
                z_contrib["Environment_Score"] = company_z_score * weights.get("Environment_Score", 0)
            
            if z_contrib:
                sorted_z_contrib = sorted(z_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_names = [f for f, _ in sorted_z_contrib]
                contributions = [c for _, c in sorted_z_contrib]
                bar_colors = ['red' if c < 0 else 'green' for c in contributions]
                
                ax.bar(feature_names, contributions, color=bar_colors)
                ax.set_title('Feature Contributions to Z-Score')
                ax.set_ylabel('Contribution')
                ax.set_xlabel('Feature')
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show numerical values with better/worse comparison like CRISIL ESG
                for f, c in sorted_z_contrib:
                    total_abs = sum(abs(v) for v in z_contrib.values())
                    if total_abs > 0:
                        pct = (abs(c) / total_abs * 100)
                        direction = "Better" if (c > 0 and not inverse_scoring) or (c < 0 and inverse_scoring) else "Worse"
                        st.write(f"{f}: {c:.4f} ({pct:.2f}%) - {direction} than average")
            else:
                st.warning("No feature contributions available for visualization")
            
            # Add histogram distribution visualization (from CRISIL ESG code)
            st.markdown("---")
            st.markdown("### Score Distribution Analysis")
            
            # Create histogram with KDE for sustainability scores
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(existing_sust, kde=True, ax=ax)
            
            # Add vertical line for mean
            mean_sust = existing_sust.mean()
            std_sust = existing_sust.std()
            ax.axvline(mean_sust, color='red', linestyle='--', 
                      label=f'Mean ({mean_sust:.2f})')
            
            # Add one standard deviation markers
            ax.axvline(mean_sust + std_sust, color='green', linestyle=':', 
                      label=f'+1 Std Dev ({mean_sust + std_sust:.2f})')
            ax.axvline(mean_sust - std_sust, color='orange', linestyle=':', 
                      label=f'-1 Std Dev ({mean_sust - std_sust:.2f})')
            
            # Add customer score marker
            ax.axvline(sust_score, color='blue', linestyle='-', 
                      label=f'This Customer ({sust_score:.2f}, Z={z_score:.2f})')
            
            ax.set_xlabel('Sustainability Score')
            ax.set_title('Distribution of Sustainability Scores')
            ax.legend()
            st.pyplot(fig)
            
            # Add pie chart for customer input weightage
            st.markdown("---")
            st.markdown("### Customer Input Weightage Analysis")
            
            # FIX 2: Ensure all core features are included in input weightage analysis
            core_features = ['Water', 'MPCE', 'Public_Transport', 'Private_Transport', 'Electricity', 'Environment_Score']
            input_data = {}
            input_weight = {}
            
            # Ensure all core features are included with at least minimal weight
            for feat in features:
                if feat in new_customer:
                    # Get the raw value
                    input_data[feat] = new_customer[feat]
                    
                    # Calculate importance
                    weight = weights.get(feat, 0)
                    if weight != 0 and feat in z_vals:
                        input_weight[feat] = abs(z_vals.get(feat, 0) * weight)
                    elif feat in core_features:  # Ensure core features are included
                        input_weight[feat] = 0.01  # Minimal weight to show in chart
            
            # Add electricity if available
            if "user_electricity" in st.session_state:
                input_data["Electricity"] = st.session_state.user_electricity
                if "Electricity" in z_vals:
                    input_weight["Electricity"] = abs(z_vals["Electricity"] * (weights.get("Electricity_State", 0) + weights.get("Electricity_MPCE", 0)))
                else:
                    input_weight["Electricity"] = 0.125  # Minimal weight to show in chart

            # Add company environment score if available
            if "Environment_Score" in new_customer:
                input_data["Environment_Score"] = new_customer["Environment_Score"]
                if "Environment_Score" in z_vals:
                    input_weight["Environment_Score"] = abs(z_vals["Environment_Score"] * weights.get("Environment_Score", 0))
                else:
                    input_weight["Environment_Score"] = 0.125  # Minimal weight to show in chart

            # Create pie chart showing weightage of each input field
            if input_weight:
                total_weight = sum(input_weight.values())
                if total_weight > 0:  # Prevent division by zero
                    weighted_inputs = {k: v/total_weight for k, v in input_weight.items()}
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create pie chart for input weightage
                        fig, ax = plt.subplots(figsize=(8, 8))
                        
                        # Sort by weightage for better visualization
                        sorted_inputs = sorted(weighted_inputs.items(), key=lambda x: x[1], reverse=True)
                        input_labels = [f[0] for f in sorted_inputs]
                        input_sizes = [f[1] for f in sorted_inputs]
                        
                        # Only show top 10 inputs in pie chart to avoid overcrowding
                        if len(input_labels) > 10:
                            other_weight = sum(input_sizes[10:])
                            input_labels = input_labels[:10] + ["Others"]
                            input_sizes = input_sizes[:10] + [other_weight]
                        
                        ax.pie(input_sizes, labels=input_labels, autopct='%1.1f%%', startangle=90)
                        ax.set_title('Customer Input Weightage Distribution')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        # Display table with input values and their weightage scores
                        input_table = []
                        
                        # FIX: Include all core features in the table regardless of weight
                        for feat in sorted(input_data.keys()):
                            if feat in features:  # Only include features used in scoring
                                input_table.append({
                                    "Input Field": feat,
                                    "Value": input_data[feat],
                                    "Weightage Score": input_weight.get(feat, 0),
                                    "Contribution %": f"{(input_weight.get(feat, 0)/total_weight)*100:.2f}%" if total_weight > 0 else "0.00%"
                                })
                        
                        input_df = pd.DataFrame(input_table)
                        st.write("Individual Input Weightage Scores")
                        st.dataframe(input_df)
                else:
                    st.warning("No input weights available for visualization")
            else:
                st.warning("No input data available for weightage analysis")
                
            
            
        
    
    
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
    # Only relevant columns
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

# Affordability Quartiles
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
    # Show table
    st.dataframe(aff_df)
    csv = aff_df.to_csv(index=False)
    st.download_button(
        label="Download Affordability Quartiles CSV",
        data=csv,
        file_name="affordability_quartiles.csv",
        mime="text/csv"
    )

# -- Group Data Template Download & Upload --
# Define template based on synth_data_raw columns
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

    # File uploader for group data
    uploaded = st.file_uploader("Upload Group Data CSV for Scoring and Ranking", type=['csv'])
    if uploaded is not None:
        group_df = pd.read_csv(uploaded)
        # Merge and score
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

# -- Individual Customer Report PDF --
def generate_pdf(scored_df, customer_df):
    """Generate a PDF report for the sustainability analysis"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Sustainability Analysis Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Customer Data Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Customer Sustainability Profile', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    # Add each metric
    for col in customer_df.columns:
        if col in ['Electricity', 'Water', 'Public_Transport', 'Private_Transport', 'Sustainability_Score']:
            value = customer_df[col].iloc[0]
            pdf.cell(0, 10, f'{col}: {value:.2f}', 0, 1)
    
    # Add comparison with dataset
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Comparison with Dataset', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    # Calculate percentiles
    for col in ['Electricity', 'Water', 'Public_Transport', 'Private_Transport']:
        if col in scored_df.columns and col in customer_df.columns:
            value = customer_df[col].iloc[0]
            percentile = (scored_df[col] < value).mean() * 100
            pdf.cell(0, 10, f'{col} Percentile: {percentile:.1f}%', 0, 1)
    
      # Convert to bytes
    return pdf.output(dest='S').encode('latin-1')
def add_pdf_section():
    """Add a PDF generation section to the Streamlit app"""
    st.header("Generate Sustainability Report (Not Fully Implemented)")
  
    # Create columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Generate a comprehensive PDF report with your sustainability assessment.")
    
    # Add a debug expander to check session state (can be removed in production)
    with st.expander("Debug Session State", expanded=False):
        st.write("Session State Keys:", list(st.session_state.keys()))
        st.write("Required Keys Check:")
        st.write("- scored_data present:", "scored_data" in st.session_state)
        st.write("- user_electricity present:", "user_electricity" in st.session_state)
        st.write("- water_units present:", "water_units" in st.session_state)
        if "water_units" in st.session_state:
            st.write("- water_units value:", st.session_state.get("water_units", 0))
    
    with col2:
        # Modified condition to ensure the button is available once basic data is entered
        # This is less strict than the original conditions but ensures users can generate PDFs
        has_required_data = (
            # Check for essential data but fall back to empty dataframes if not present
            ("scored_data" in st.session_state or st.session_state.get("scored_data", pd.DataFrame()) is not None) and
            ("user_electricity" in st.session_state or st.session_state.get("user_electricity", 0) > 0)
        )
        
        # Create a test customer record with all the input data
        # Use get() with default values to prevent KeyErrors
        test_customer = {
            'Environment_Score': st.session_state.get('company_data', {}).get('Environment_Score', [0]).iloc[0] 
                if isinstance(st.session_state.get('company_data', {}), pd.DataFrame) else 0,
            'Electricity': st.session_state.get('user_electricity', 0),
            'Water': st.session_state.get('water_units', 0),
            'Public_Transport': st.session_state.get('public_distance', 0),
            'Private_Transport': st.session_state.get('private_distance', 0),
            'MPCE': st.session_state.get('user_mpce_range', [0])[0] if isinstance(st.session_state.get('user_mpce_range', []), list) else 0
        }
        
        # Add Sustainability Score if it exists
        if 'Sustainability_Score' in st.session_state:
            test_customer['Sustainability_Score'] = st.session_state.Sustainability_Score
        
        # Generate PDF button - show regardless of conditions
        if st.button("Generate PDF Report", key="gen_pdf"):
            with st.spinner("Generating your sustainability report..."):
                try:
                    # Create fallback data if scored_data is missing
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
            # Only show this warning if we detect missing essential data
            if not has_required_data:
                st.warning("Please enter all required data in the Test New Customer section first")


if __name__ == "__main__":
    # Set styling for better appearance
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
