import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
                    stats[state][col] = {
                        'mean': state_df[col].mean(),
                        'median': state_df[col].median(),
                        'std': state_df[col].std(),
                        'min': state_df[col].min(),
                        'max': state_df[col].max()
                    }
    else:
        # Overall stats
        for col in df.columns:
            if col not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Sustainability_Score",
                          "Company_Name", "Sector classification", "Environment Score", 
                          "ESG Rating", "Category", "Date of Rating", "Total Employees", "State_UT"]:
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
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
    norm_vals, z_vals = {}, {}
    
    for col in df.columns:
        if col not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Sustainability_Score",
                      "Company_Name", "Sector classification", "Environment Score", "ESG Rating",
                      "Category", "Date of Rating", "Total Employees", "State_UT"]:
            
            if state_specific and "State_UT" in df.columns:
                # Compute z-scores state by state
                for state in df["State_UT"].unique():
                    state_mask = df["State_UT"] == state
                    state_df = df[state_mask]
                    
                    # Check if baseline values are provided for this state
                    if ("baseline_values_by_state" in st.session_state and 
                        state in st.session_state.baseline_values_by_state and 
                        col in st.session_state.baseline_values_by_state[state]):
                        base_mean, base_std = st.session_state.baseline_values_by_state[state][col]
                        mean_val, std_val = base_mean, base_std
                    else:
                        mean_val, std_val = state_df[col].mean(), state_df[col].std(ddof=1)
                    
                    # Calculate z-score
                    if std_val == 0:
                        z = pd.Series(0, index=state_df.index)
                    else:
                        z = (state_df[col] - mean_val) / std_val
                    
                    # Apply inverse if needed (higher consumption = worse sustainability)
                    if inverse:
                        z = -z
                    
                    # Cap z-values between -1 and 1
                    z = np.clip(z, -1, 1)
                    
                    # Assign the z-scores to the dataframe
                    if col not in z_vals:
                        z_vals[col] = pd.Series(np.nan, index=df.index)
                    z_vals[col].loc[state_mask] = z.values
                    
                    # Also calculate traditional normalization
                    if col not in norm_vals:
                        norm_vals[col] = pd.Series(np.nan, index=df.index)
                    
                    # Handle state-level min/max for normalization
                    cmin, cmax = state_df[col].min(), state_df[col].max()
                    if cmax == cmin:
                        norm = pd.Series(1 if inverse else 1000, index=state_df.index)
                    else:
                        if inverse:
                            norm = ((cmax - state_df[col]) / (cmax - cmin)) * 999 + 1
                        else:
                            norm = ((state_df[col] - cmin) / (cmax - cmin)) * 999 + 1
                    norm_vals[col].loc[state_mask] = norm.values
                    
                # Add the z-score columns to the dataframe
                df[f"{col}_z_score"] = z_vals[col]
                df[f"{col}_normalized"] = norm_vals[col]
            else:
                # Original non-state-specific calculation
                # Check if baseline values are provided
                if "baseline_values" in st.session_state and col in st.session_state.baseline_values:
                    base_mean, base_std = st.session_state.baseline_values[col]
                    mean_val, std_val = base_mean, base_std
                else:
                    mean_val, std_val = df[col].mean(), df[col].std(ddof=1)
                    
                # Calculate z-score
                if std_val == 0:
                    z = 0
                else:
                    z = (df[col] - mean_val) / std_val
                
                # Apply inverse if needed (higher consumption = worse sustainability)
                if inverse:
                    z = -z
                    
                # Cap z-values between -1 and 1
                z = np.clip(z, -1, 1)
                
                z_vals[col] = z
                df[f"{col}_z_score"] = z
                
                # Also calculate traditional normalization (for backward compatibility)
                norm = normalize_series(df[col], 1, 1000, inverse)
                norm_vals[col] = norm
                df[f"{col}_normalized"] = norm

    # Calculate weighted z-score sum (S1)
    if state_specific and "State_UT" in df.columns:
        # For state-specific calculation, handle z_vals as Series
        weighted_z_sum = sum(z_vals[f] * w for f, w in weights.items() if f in z_vals)
    else:
        weighted_z_sum = sum(z_vals[f] * w for f, w in weights.items() if f in z_vals)
        
    df['Z_Score'] = weighted_z_sum
    df['Weighted_Z_Score'] = weighted_z_sum  # Added for compatibility
    
    # Transform S1 to final score S:
    # When S1 = 0, S = 500
    # When S1 = +5, S ≈ 0
    # When S1 = -5, S ≈ 1000
    # Using a sigmoid-like transformation: S = 500 * (1 - tanh(S1/2.5))
    df['Sustainability_Score'] = 500 * (1 - np.tanh(df['Z_Score']/2.5))
    
    # Also keep the original weighted score for comparison
    if state_specific:
        weighted_score = sum(norm_vals[f] * w for f, w in weights.items() if f in norm_vals)
    else:
        weighted_score = sum(norm_vals[f] * w for f, w in weights.items() if f in norm_vals)
    df['Weighted_Score'] = weighted_score
    
    # Rank based on Sustainability Score (lower z-score = better sustainability)
    df['Rank'] = df['Sustainability_Score'].fillna(-1).rank(method='min', ascending=False).astype(int)
    df.sort_values('Rank', inplace=True)

    # electricity_z_score based on Electricity/MPCE ratio
    if 'Electricity' in raw_df.columns and 'MPCE' in raw_df.columns:
        # Handle potential divide by zero
        mpce_safe = raw_df['MPCE'].replace(0, np.nan)
        ratio = raw_df['Electricity'] / mpce_safe
        if ratio.isna().all():
            df['electricity_z_score'] = 0  # Set default if all NaN
        else:
            df['electricity_z_score'] = (ratio - ratio.mean()) / ratio.std(ddof=1)
            df['electricity_z_score'] = df['electricity_z_score'].fillna(0)  # Replace NaN with 0

    return df, None

def feature_distribution_ui(feature_name, default_min=1, default_max=1000):
    st.markdown(f"#### {feature_name} Settings")
    dist_type = st.selectbox(
        f"Distribution for {feature_name}",
        ["Uniform","Normal","Poisson","Exponential","Binomial","Lognormal"],
        key=f"dist_{feature_name}"
    )
    params = {}
    if dist_type == "Uniform":
        params['min'] = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        params['max'] = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Normal":
        params['mean'] = st.number_input(f"{feature_name} Mean", value=(default_min+default_max)//2, key=f"{feature_name}_mean")
        params['std']  = st.number_input(f"{feature_name} Std Dev", value=(default_max-default_min)//4, key=f"{feature_name}_std")
        params['min']  = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        params['max']  = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Poisson":
        params['lambda'] = st.number_input(f"{feature_name} Lambda", value=(default_min+default_max)//2, key=f"{feature_name}_lambda")
        params['min']    = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        params['max']    = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Exponential":
        params['scale'] = st.number_input(f"{feature_name} Scale", value=(default_max-default_min)//2, key=f"{feature_name}_scale")
        params['min']   = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        params['max']   = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Binomial":
        params['n_trials'] = st.number_input(f"{feature_name} Trials", value=10, key=f"{feature_name}_n_trials")
        params['p']        = st.number_input(f"{feature_name} p", min_value=0.0, max_value=1.0, value=0.5, key=f"{feature_name}_p")
        params['min']      = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        params['max']      = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Lognormal":
        params['mean']  = st.number_input(f"{feature_name} Mean(log)", value=1.0, key=f"{feature_name}_mean")
        params['sigma'] = st.number_input(f"{feature_name} Sigma(log)", value=0.5, key=f"{feature_name}_sigma")
        params['min']   = st.number_input(f"{feature_name} Min", value=default_min, key=f"{feature_name}_min")
        params['max']   = st.number_input(f"{feature_name} Max", value=default_max, key=f"{feature_name}_max")
    return dist_type, params

def load_company_data():
    if 'company_data' not in st.session_state:
        st.session_state.company_data = None
    uploaded = st.file_uploader("Upload Company Data CSV", type="csv", key="company_data_uploader")
    if uploaded:
        df = pd.read_csv(uploaded)
        needed = ["Company_Name","Sector classification","Environment Score",
                  "ESG Rating","Category","Date of Rating","Total Employees"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return None
        st.session_state.company_data = df
        return df
    return st.session_state.company_data

def load_real_electricity_data():
    """Load electricity consumption data from file path"""
    try:
        # Try to load from file in current directory first
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
        
        # Check if required columns exist
        needed = ["hhid", "state_name", "sector", "qty_usage_in_1month", "mpce"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            st.error(f"Missing columns in electricity data: {', '.join(missing)}")
            return None
        
        # Ensure state_name is in INDIAN_STATES_UTS list
        df["state_name"] = df["state_name"].apply(lambda x: x if x in INDIAN_STATES_UTS else INDIAN_STATES_UTS[0])
        
        # Calculate baseline statistics by state
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
        
        # Set dataset in session state
        st.session_state.electricity_data = df
        return df
    
    except Exception as e:
        st.error(f"Error loading electricity data: {str(e)}")
        return None

def render_sortable_table(df):
    st.dataframe(df, use_container_width=True)

# App title
st.title("Resource Sustainability Consumption Dashboard")
st.markdown("Analyze your resource consumption and sustainability score")

# Sidebar configuration
st.sidebar.markdown("## Data Configuration")

# Load real electricity data
if 'electricity_data' not in st.session_state:
    electricity_data = load_real_electricity_data()
    if electricity_data is not None:
        st.sidebar.success(f"Loaded {len(electricity_data)} electricity records from data file")
else:
    electricity_data = st.session_state.electricity_data

# Show electricity data info if available
if electricity_data is not None:
    # Display summary of loaded data
    st.success(f"Loaded electricity data with {len(electricity_data)} records across {electricity_data['state_name'].nunique()} states")
    
    # State and sector filters for the dashboard
    state_options = sorted(electricity_data['state_name'].unique().tolist())
    selected_state = st.selectbox("Select State", state_options, key="dashboard_state")
    
    sector_options = [('Rural', 1), ('Urban', 2)]
    selected_sector_name, selected_sector = sector_options[0]
    col1, col2 = st.columns(2)
    with col1:
        selected_sector_name = st.radio("Select Sector", ['Rural', 'Urban'], key="dashboard_sector")
        selected_sector = 1 if selected_sector_name == 'Rural' else 2
    
    # Filter electricity data based on selections
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
    
    # Display distribution of electricity usage for selected state/sector
    st.markdown(f"## Electricity Distribution - {selected_state}, {selected_sector_name}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_data["qty_usage_in_1month"], bins=30, kde=True, ax=ax)
    ax.set_title(f"Electricity Consumption Distribution - {selected_state}, {selected_sector_name}")
    ax.set_xlabel("Electricity (kWh/month)")
    st.pyplot(fig)

# Sidebar: Corporate data
st.sidebar.markdown("## Corporate Data Configuration")
company_data = load_company_data()
if company_data is not None:
    st.sidebar.success(f"Loaded {len(company_data)} companies")
    st.sidebar.dataframe(company_data.head(3))
else:
    st.sidebar.info("Please upload company data CSV")
    template = pd.DataFrame({
        "Company_Name":["A","B","C"],
        "Sector classification":["Tech","Manu","Health"],
        "Environment Score":[70,60,80],
        "ESG Rating":["A","B","A-"],
        "Category":["Leader","Average","Leader"],
        "Date of Rating":["2024-03-01","2024-02-15","2024-04-10"],
        "Total Employees":[100,200,150]
    })
    st.sidebar.download_button("Download Company Template",
                               data=template.to_csv(index=False).encode('utf-8'),
                               file_name="company_template.csv",
                               mime="text/csv")

# Generate synthetic data for other resources
st.sidebar.markdown("## Generate Data for Analysis")
n = st.sidebar.number_input("Number of Records", min_value=1, value=100, step=1)

# Only configure parameters for Water and Transport (not Electricity or MPCE)
st.sidebar.markdown("### Water Consumption")
dist_water, params_water = feature_distribution_ui("Water", default_min=200, default_max=600)

st.sidebar.markdown("### Commute")
include_public = st.sidebar.checkbox("Include Public Transport", value=True)
if include_public:
    public_dist, public_params = feature_distribution_ui("Public_Transport", default_min=0, default_max=500)

include_private = st.sidebar.checkbox("Include Private Transport", value=True)
if include_private:
    private_dist, private_params = feature_distribution_ui("Private_Transport", default_min=0, default_max=300)

st.sidebar.markdown("### Corporate Details")
include_corp = st.sidebar.checkbox("Include Corporate Details", value=True)

# Build feature settings but only for Water and Transport
feature_settings = {
    "Water": (dist_water, params_water)
}
if include_public:
    feature_settings["Public_Transport"] = (public_dist, public_params)
if include_private:
    feature_settings["Private_Transport"] = (private_dist, private_params)

if st.sidebar.button("Generate Analysis Data"):
    if electricity_data is not None:
        # Generate synthetic data but use real electricity data
        raw = generate_synthetic_data(n, feature_settings, electricity_data)
        
        # Add corporate data if requested
        if include_corp and company_data is not None:
            samp = company_data.sample(n=n, replace=True).reset_index(drop=True)
            for col in ["Company_Name", "Sector classification", "Environment Score",
                        "ESG Rating", "Category", "Date of Rating", "Total Employees"]:
                raw[col] = samp[col]
        
        st.session_state.synth_data_raw = raw
        st.session_state.feature_settings = feature_settings
        
        # Set feature constraints for all columns
        st.session_state.feature_constraints = {
            feat: (p['min'], p['max']) for feat, (_, p) in feature_settings.items()
        }
        
        # Add constraints for electricity and MPCE from real data
        st.session_state.feature_constraints["Electricity"] = (
            electricity_data["qty_usage_in_1month"].min(),
            electricity_data["qty_usage_in_1month"].max()
        )
        
        st.session_state.feature_constraints["MPCE"] = (
            electricity_data["mpce"].min(),
            electricity_data["mpce"].max()
        )
        
        # Compute statistics
        st.session_state.feature_stats = compute_feature_stats(raw)
        st.session_state.feature_stats_by_state = compute_feature_stats(raw, by_state=True)
        
        st.sidebar.success("Analysis data generated!")
    else:
        st.sidebar.error("No electricity data available. Please check your data file.")

# Main app logic
if "synth_data_raw" in st.session_state:
    st.markdown("## Data Preview")
    render_sortable_table(st.session_state.synth_data_raw)

    st.markdown("---\n## Weighted Score Settings")
    features = [
        c for c in st.session_state.synth_data_raw.columns
        if c not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Sustainability_Score",
                     "Company_Name", "Sector classification", "State_UT",
                     "Environment Score", "ESG Rating", "Category",
                     "Date of Rating", "Total Employees"]
    ]
    default_w = round(1/len(features), 4)
    weights = {}
    for f in features:
        weights[f] = st.number_input(
            f"Weight: {f}", value=default_w, step=0.01, format="%.4f", key=f"wt_{f}"
        )
    total = sum(weights.values())
    st.markdown(f"**Total Weight:** {total:.2f}")
    
    # Option to use state-specific scoring
    use_state_specific = st.checkbox("Use State/UT Specific Scoring", value=True)
    
    if abs(total - 1.0) > 1e-3:
        st.error("Weights must sum exactly to 1!")
    else:
        if st.button("Generate Weighted Score"):
            if 'user_electricity' not in st.session_state:
                st.session_state.user_electricity = 0
            scored, _ = compute_weighted_score(
                st.session_state.synth_data_raw, weights, inverse=True, state_specific=use_state_specific
            )
            st.session_state.scored_data = scored
            st.session_state.weights = weights
            st.success("Weighted score and ranking generated!")

    if "scored_data" in st.session_state:
        st.markdown("### Ranked Individuals Based on Sustainability Score")
        scored = st.session_state.scored_data
        # Build an ordered list and drop duplicates
        core = ["ID", "Weighted_Score", "Z_Score", "Sustainability_Score", "Rank"]
        extras = [
            col for col in scored.columns
            if col not in core
            and not col.endswith("_normalized")
            and not col.endswith("_z_score")
        ]
        display_cols = list(dict.fromkeys(core + extras))

        render_sortable_table(scored[display_cols].reset_index(drop=True))
    
    # Personal electricity benchmark
    st.markdown("## Your Personal Electricity Analysis")
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
    
    with col2:
        #User input for electricity consumption
        user_electricity = st.number_input("Your Monthly Electricity Usage (kWh)", 
                                           min_value=0.0, 
                                           value=0.0, 
                                           step=10.0,
                                           key="personal_electricity")
        
        # Store the value in session state so it's accessible elsewhere
        st.session_state.user_electricity = user_electricity

        # Calculate cost using data patterns if we have user input and data
        if user_electricity > 0 and electricity_data is not None:
            # Filter data for user's state and sector
            state_sector_data = electricity_data[
                (electricity_data['state_name'] == user_state) & 
                (electricity_data['sector'] == user_sector)
            ]
            
            if not state_sector_data.empty:
                # Display user electricity consumption
                st.metric("Your Electricity Consumption", f"{user_electricity:.2f} kWh")
                
                # Calculate average rate from the data (Rs/kWh)
                df_rates = pd.DataFrame({
                    'consumption': [user_electricity],  # Use entered electricity value
                    'cost': st.number_input("Enter your monthly electricity cost (₹)", min_value=0.0, step=100.0)  # Add cost input
                })
                
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
                
                # Calculate and display sustainability scores if we have feature stats
                if 'feature_stats' in st.session_state and 'Electricity' in st.session_state.feature_stats:
                    stats = st.session_state.feature_stats['Electricity']
                    mean_usage = stats['mean']
                    std_usage = stats['std']
                    
                    # Calculate z-score (negative because inverse scoring - lower consumption is better)
                    z_score = (mean_usage - user_electricity) / std_usage if std_usage > 0 else 0
                    z_score = np.clip(z_score, -1, 1)  # Cap between -1 and 1
                    
                    # Calculate sustainability score (500 = average, higher is better)
                    sust_score = 500 * (1 - np.tanh(z_score/2.5))
                    
                    # Calculate percentile rank if we have scored data
                    percentile = 0
                    if 'scored_data' in st.session_state:
                        percentile = (st.session_state.scored_data['Sustainability_Score'] < sust_score).mean() * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        comparison = "better than average" if user_electricity < mean_usage else "worse than average"
                        diff_pct = abs(user_electricity - mean_usage) / mean_usage * 100
                        st.metric("Comparison to Average", 
                                  f"{diff_pct:.1f}% {comparison}", 
                                  f"{mean_usage:.1f} kWh avg overall")
                    
                    with col2:
                        st.metric("Your Sustainability Score", 
                                  f"{sust_score:.1f}/1000",
                                  f"Z-score: {z_score:.2f}")
                    
                    with col3:
                        st.metric("Your Percentile", 
                                  f"{percentile:.1f}%",
                                  f"Better than {percentile:.1f}% of users")
                
                # Add state comparison visualization
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
                    
                    # Calculate and display user's percentile within their state
                    if user_state in st.session_state.feature_stats_by_state and 'Electricity' in st.session_state.feature_stats_by_state[user_state]:
                        state_stats = st.session_state.feature_stats_by_state[user_state]['Electricity']
                        state_mean = state_stats['mean']
                        state_std = state_stats['std']
                        
                        state_specific_z = (state_mean - user_electricity) / state_std if state_std > 0 else 0
                        state_specific_z = np.clip(state_specific_z, -1, 1)
                        
                        state_sust_score = 500 * (1 - np.tanh(state_specific_z/2.5))
                        
                        st.metric(
                            f"Your Score Compared to {user_state} Average",
                            f"{state_sust_score:.1f}/1000",
                            f"{'Better' if user_electricity < state_mean else 'Worse'} than average by {abs(user_electricity - state_mean):.1f} kWh"
                        )

# Show distribution graphs and insights
if "synth_data_raw" in st.session_state:
    if st.checkbox("Show Distribution Graphs"):
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
                           "Company_Name", "Sector classification", "State_UT",
                           "Environment Score", "ESG Rating", "Category",
                           "Date of Rating", "Total Employees"]
            ]
            feature_to_plot = st.selectbox("Select Feature", features)
            fig, ax = plt.subplots()
            ax.scatter(st.session_state.synth_data_raw[feature_to_plot], 
                      st.session_state.scored_data["Sustainability_Score"])
            ax.set_xlabel(feature_to_plot)
            ax.set_ylabel("Sustainability Score")
            ax.set_title(f"{feature_to_plot} vs Sustainability Score")
            st.pyplot(fig)
        
        # Optional: Add more advanced analytics
        if "Electricity" in st.session_state.synth_data_raw.columns and "MPCE" in st.session_state.synth_data_raw.columns:
            st.markdown("### Electricity-MPCE Analysis")
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

    st.markdown("---\n## Test New Customer")
    test_mode = st.radio("Input Mode", ["Manual Entry", "CSV Upload"], key="test_mode")
    features = [
        c for c in st.session_state.synth_data_raw.columns
        if c not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Sustainability_Score",
                     "Company_Name", "Sector classification", "State_UT",
                     "Environment Score", "ESG Rating",
                     "Category", "Date of Rating", "Total Employees"]
    ]

    # Function to export feature details (referenced but missing in both files)
    def export_feature_details():
        if 'feature_stats' in st.session_state:
            stats_df = pd.DataFrame()
            for feat, stat_dict in st.session_state.feature_stats.items():
                stats_df[feat] = pd.Series(stat_dict)
            stats_df = stats_df.T
            
            st.markdown("### Feature Statistics")
            st.dataframe(stats_df)
            
            # Add download button
            st.download_button(
                "Download Feature Statistics",
                data=stats_df.to_csv().encode('utf-8'),
                file_name="feature_stats.csv",
                mime="text/csv"
            )

    export_feature_details()

    if test_mode == "Manual Entry":
        st.markdown("### Enter Customer Feature Values")
        new_customer = {}
        for feat in features:
            if 'feature_constraints' in st.session_state and feat in st.session_state.feature_constraints:
                cmin, cmax = st.session_state.feature_constraints[feat]
            else:
                cmin, cmax = st.session_state.synth_data_raw[feat].min(), st.session_state.synth_data_raw[feat].max()
            
            new_customer[feat] = st.number_input(
                f"{feat} (Allowed: {cmin}–{cmax})",
                min_value=float(cmin), max_value=float(cmax), value=float(cmin), key=f"test_{feat}"
            )
        
        if company_data is not None and "Company_Name" in st.session_state.synth_data_raw.columns:
            sel = st.selectbox("Select Company", company_data["Company_Name"].unique())
            row = company_data[company_data["Company_Name"] == sel].iloc[0]
            st.info(
                f"Sector: {row['Sector classification']}, "
                f"Environment Score: {row['Environment Score']}, "
                f"ESG Rating: {row['ESG Rating']}, "
                f"Category: {row['Category']}, "
                f"Date of Rating: {row['Date of Rating']}, "
                f"Employees: {row['Total Employees']}"
            )
        
        if st.button("Evaluate Customer"):
            if 'feature_stats' not in st.session_state or 'weights' not in st.session_state:
                st.error("Please generate scores first.")
            else:
                # Define inverse_scoring value (referenced but not defined in second file)
                inverse_scoring = True
                
                # Calculate z-scores and sustainability score
                z_vals = {}
                for feat, val in new_customer.items():
                    stats = st.session_state.feature_stats.get(feat)
                    if stats:
                        z = (stats['mean'] - val)/stats['std'] if inverse_scoring else (val - stats['mean'])/stats['std']
                        z = np.clip(z, -1, 1)  # Cap z-value between -1 and 1
                        z_vals[feat] = z
                
                # Calculate weighted z-score
                weights = st.session_state.weights
                z_score = sum(z_vals[f] * weights.get(f, 0) for f in features)
                
                # Calculate sustainability score
                sust_score = 500 * (1 - np.tanh(z_score/2.5))
                
                # Also calculate traditional normalization for comparison
                norm_vals = {}
                for feat, val in new_customer.items():
                    cmin, cmax = st.session_state.feature_constraints.get(feat, (val, val))
                    if cmax == cmin:
                        norm_vals[feat] = 1
                    else:
                        if inverse_scoring:
                            norm_vals[feat] = ((cmax - val)/(cmax - cmin))*999 + 1
                        else:
                            norm_vals[feat] = ((val - cmin)/(cmax - cmin))*999 + 1
                
                weighted_score = sum(norm_vals[f] * weights.get(f, 0) for f in features)
                
                # Calculate ranks
                existing_sust = st.session_state.scored_data["Sustainability_Score"]
                existing_trad = st.session_state.scored_data["Weighted_Score"]
                sust_rank = (existing_sust > sust_score).sum() + 1
                trad_rank = (existing_trad > weighted_score).sum() + 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Z-Score", f"{z_score:.2f}")
                with col2:
                    st.metric("Sustainability Score", f"{sust_score:.2f}")
                    st.metric("Sustainability Rank", f"{sust_rank}")
                with col3:
                    st.metric("Legacy Weighted Score", f"{weighted_score:.2f}")
                    st.metric("Legacy Rank", f"{trad_rank}")
                
                # Show feature contributions
                st.markdown("**Feature Z-Score Contributions:**")
                z_contrib = {f: z_vals[f] * weights.get(f, 0) for f in features}
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
                
                # Show numerical values
                for f, c in sorted_z_contrib:
                    pct = (abs(c) / sum(abs(v) for v in z_contrib.values()) * 100)
                    direction = "Better" if (c > 0 and not inverse_scoring) or (c < 0 and inverse_scoring) else "Worse"
                    st.write(f"{f}: {c:.4f} ({pct:.2f}%) - {direction} than average")
    else:
        tmpl = pd.DataFrame(columns=features).to_csv(index=False).encode('utf-8')
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
                    # Define inverse_scoring value (referenced but not defined in second file)
                    inverse_scoring = True
                    
                    test_with_scores = []
                    for _, row in test_df.iterrows():
                        # Extract features
                        new_customer = {f: row[f] for f in features if f in row}
                        
                        # Calculate z-scores
                        z_vals = {}
                        for feat, val in new_customer.items():
                            stats = st.session_state.feature_stats.get(feat)
                            if stats:
                                z = (stats['mean'] - val)/stats['std'] if inverse_scoring else (val - stats['mean'])/stats['std']
                                z = np.clip(z, -1, 1)  # Cap z-value between -1 and 1
                                z_vals[feat] = z
                        
                        # Calculate weighted z-score
                        weights = st.session_state.weights
                        z_score = sum(z_vals[f] * weights.get(f, 0) for f in features)
                        
                        # Calculate sustainability score
                        sust_score = 500 * (1 - np.tanh(z_score/2.5))
                        
                        # Calculate traditional normalization
                        norm_vals = {}
                        for feat, val in new_customer.items():
                            cmin, cmax = st.session_state.feature_constraints.get(feat, (val, val))
                            if cmax == cmin:
                                norm_vals[feat] = 1
                            else:
                                if inverse_scoring:
                                    norm_vals[feat] = ((cmax - val)/(cmax - cmin))*999 + 1
                                else:
                                    norm_vals[feat] = ((val - cmin)/(cmax - cmin))*999 + 1
                        
                        weighted_score = sum(norm_vals[f] * weights.get(f, 0) for f in features)
                        
                        # Calculate ranks
                        existing_sust = st.session_state.scored_data["Sustainability_Score"]
                        existing_trad = st.session_state.scored_data["Weighted_Score"]
                        sust_rank = (existing_sust > sust_score).sum() + 1
                        trad_rank = (existing_trad > weighted_score).sum() + 1
                        
                        # Add scores to the row
                        result = row.to_dict()
                        result.update({
                            "Z_Score": z_score,
                            "Sustainability_Score": sust_score,
                            "Sustainability_Rank": sust_rank,
                            "Weighted_Score": weighted_score,
                            "Legacy_Rank": trad_rank
                        })
                        test_with_scores.append(result)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(test_with_scores)
                    st.markdown("### Test Results")
                    st.dataframe(results_df)
                    
                    # Create download button for results
                    st.download_button("Download Test Results",
                                       data=results_df.to_csv(index=False).encode('utf-8'),
                                       file_name="test_results.csv",
                                       mime="text/csv")
                    
                    # Show distribution of scores
                    fig, ax = plt.subplots()
                    sns.histplot(results_df["Sustainability_Score"], bins=20, kde=True, ax=ax)
                    ax.set_title("Distribution of Test Sustainability Scores")
                    st.pyplot(fig)

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
