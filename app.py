import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Resource Sustainability Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .dataframe-container {
        display: block;
        max-height: 500px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    table, th, td {
        border: 1px solid #ddd;
    }
    th, td {
        padding: 8px;
        text-align: center;
    }
    th {
        background-color: #f2f2f2;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def normalize_series(series, new_min=1, new_max=1000, inverse=True):
    """Normalize series with option to inverse (higher values = lower score)"""
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
        data = np.random.randint(low, high+1, size=n)
    elif dist_type == "Normal":
        mean, std = params['mean'], params['std']
        data = np.random.normal(mean, std, size=n)
    elif dist_type == "Poisson":
        lam = params['lambda']
        data = np.random.poisson(lam, size=n)
    elif dist_type == "Exponential":
        scale = params['scale']
        data = np.random.exponential(scale, size=n)
    elif dist_type == "Binomial":
        n_trials = int(params.get('n_trials', 10))
        p = params.get('p', 0.5)
        data = np.random.binomial(n_trials, p, size=n)
    elif dist_type == "Lognormal":
        mean, sigma = params['mean'], params['sigma']
        data = np.random.lognormal(mean, sigma, size=n)
    else:
        low, high = int(params['min']), int(params['max'])
        data = np.random.randint(low, high+1, size=n)
    data = np.clip(data, params['min'], params['max'])
    return np.round(data).astype(int)

def generate_synthetic_data(n, feature_settings):
    data = {}
    for feature, (dist_type, params) in feature_settings.items():
        data[feature] = generate_feature_data(n, dist_type, params)
    df = pd.DataFrame(data)
    df.insert(0, "ID", range(1, n+1))
    return df

def compute_feature_stats(df):
    """Compute statistics for each feature for baseline comparison"""
    feature_stats = {}
    for col in df.columns:
        if col not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Company_Name", "Sector classification", "Environment Score", 
                       "ESG Rating", "Category", "Date of Rating", "Total Employees"]:
            feature_stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
    return feature_stats

def compute_weighted_score(raw_df, weights, inverse=True):
    """Compute weighted score with option for inverse relationship"""
    df = raw_df.copy()
    norm_values = {}
    z_scores = {}
    for col in df.columns:
        if col not in ["ID", "Weighted_Score", "Rank", "Z_Score", "Company_Name", "Sector classification", "Environment Score", 
                       "ESG Rating", "Category", "Date of Rating", "Total Employees"]:
            norm_values[col] = normalize_series(df[col], 1, 1000, inverse=inverse)
            z = (df[col] - df[col].mean()) / df[col].std()
            if inverse:
                z = -z
            z_scores[col] = z
            df[f"{col}_normalized"] = norm_values[col]
            df[f"{col}_z_score"] = z

    weighted_sum = 0
    for key, weight in weights.items():
        if key in norm_values:
            contribution = norm_values[key] * weight
            df[f"{key}_contribution"] = contribution
            weighted_sum += contribution

    df['Weighted_Score'] = weighted_sum
    df['Z_Score'] = sum(z_scores[k] * weights.get(k, 0) for k in z_scores)
    df['Rank'] = df['Weighted_Score'].rank(method='min', ascending=False).astype(int)
    df.sort_values('Rank', inplace=True)
    return df, None

def feature_distribution_ui(feature_name, default_min=1, default_max=1000):
    st.markdown(f"#### {feature_name} Settings")
    dist_type = st.selectbox(
        f"Select Distribution for {feature_name}",
        options=["Uniform", "Normal", "Poisson", "Exponential", "Binomial", "Lognormal"],
        key=f"dist_{feature_name}"
    )
    params = {}
    if dist_type == "Uniform":
        params['min'] = st.number_input(f"{feature_name} Minimum", value=default_min, key=f"{feature_name}_min")
        params['max'] = st.number_input(f"{feature_name} Maximum", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Normal":
        params['mean'] = st.number_input(f"{feature_name} Mean", value=(default_min+default_max)//2, key=f"{feature_name}_mean")
        params['std'] = st.number_input(f"{feature_name} Std Deviation", value=100, key=f"{feature_name}_std")
        params['min'] = st.number_input(f"{feature_name} Minimum", value=default_min, key=f"{feature_name}_min")
        params['max'] = st.number_input(f"{feature_name} Maximum", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Poisson":
        params['lambda'] = st.number_input(f"{feature_name} Lambda", value=(default_min+default_max)//2, key=f"{feature_name}_lambda")
        params['min'] = st.number_input(f"{feature_name} Minimum", value=default_min, key=f"{feature_name}_min")
        params['max'] = st.number_input(f"{feature_name} Maximum", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Exponential":
        params['scale'] = st.number_input(f"{feature_name} Scale", value=100, key=f"{feature_name}_scale")
        params['min'] = st.number_input(f"{feature_name} Minimum", value=default_min, key=f"{feature_name}_min")
        params['max'] = st.number_input(f"{feature_name} Maximum", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Binomial":
        params['n_trials'] = st.number_input(f"{feature_name} Number of Trials", value=10, key=f"{feature_name}_n_trials")
        params['p'] = st.number_input(f"{feature_name} Probability", min_value=0.0, max_value=1.0, value=0.5, key=f"{feature_name}_p")
        params['min'] = st.number_input(f"{feature_name} Minimum", value=default_min, key=f"{feature_name}_min")
        params['max'] = st.number_input(f"{feature_name} Maximum", value=default_max, key=f"{feature_name}_max")
    elif dist_type == "Lognormal":
        params['mean'] = st.number_input(f"{feature_name} Mean", value=4, key=f"{feature_name}_mean")
        params['sigma'] = st.number_input(f"{feature_name} Sigma", value=0.5, key=f"{feature_name}_sigma")
        params['min'] = st.number_input(f"{feature_name} Minimum", value=default_min, key=f"{feature_name}_min")
        params['max'] = st.number_input(f"{feature_name} Maximum", value=default_max, key=f"{feature_name}_max")
    return dist_type, params

def load_company_data():
    """Load company data from uploaded CSV"""
    if 'company_data' not in st.session_state:
        st.session_state.company_data = None
    
    uploaded_file = st.file_uploader("Upload Company Data CSV", type="csv", key="company_data_uploader")
    if uploaded_file is not None:
        try:
            company_data = pd.read_csv(uploaded_file)
            required_columns = ["Company_Name", "Sector classification", "Environment Score", 
                                "ESG Rating", "Category", "Date of Rating", "Total Employees"]
            missing_columns = [col for col in required_columns if col not in company_data.columns]
            if missing_columns:
                st.error(f"Company data is missing required columns: {', '.join(missing_columns)}")
                return None
            st.session_state.company_data = company_data
            return company_data
        except Exception as e:
            st.error(f"Error loading company data: {e}")
            return None
    return st.session_state.company_data

def export_feature_details():
    """
    Export feature details including distribution settings and synthetic data
    """
    if "feature_settings" in st.session_state:
        distribution_details = []
        for feature, (dist_type, params) in st.session_state.feature_settings.items():
            details = {
                'Feature': feature,
                'Distribution Type': dist_type,
                **params
            }
            distribution_details.append(details)
        
        distribution_df = pd.DataFrame(distribution_details)
        
        export_options = st.radio("Select Export Type", 
                                  ["Feature Distribution Settings", 
                                   "Synthetic Data", 
                                   "Weighted Scored Data",
                                   "Full Dataset with Details"])
        
        if export_options == "Feature Distribution Settings":
            csv = distribution_df.to_csv(index=False)
            st.download_button(
                label="Download Feature Distribution Settings",
                data=csv,
                file_name="feature_distribution_settings.csv",
                mime="text/csv"
            )
        
        elif export_options == "Synthetic Data":
            if "synth_data_raw" in st.session_state:
                raw_csv = st.session_state.synth_data_raw.to_csv(index=False)
                st.download_button(
                    label="Download Synthetic Data",
                    data=raw_csv,
                    file_name="synthetic_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No synthetic data generated yet.")
        
        elif export_options == "Weighted Scored Data":
            if "scored_data" in st.session_state:
                scored_csv = st.session_state.scored_data.to_csv(index=False)
                st.download_button(
                    label="Download Weighted Scored Data",
                    data=scored_csv,
                    file_name="weighted_scored_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No weighted scores generated yet.")
        
        elif export_options == "Full Dataset with Details":
            if "synth_data_raw" in st.session_state:
                synthetic_df = st.session_state.synth_data_raw.copy()
                if "scored_data" in st.session_state:
                    synthetic_df['Weighted_Score'] = st.session_state.scored_data['Weighted_Score']
                    synthetic_df['Rank'] = st.session_state.scored_data['Rank']
                full_export_csv = synthetic_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Dataset with Details",
                    data=full_export_csv,
                    file_name="full_dataset_with_details.csv",
                    mime="text/csv"
                )
                st.write("Distribution Details:", distribution_df)
            else:
                st.warning("No synthetic data generated yet.")

def render_sortable_table(df):
    """
    Render a sortable table using standard Streamlit methods
    """
    st.dataframe(df, use_container_width=True)

st.title("Resource Sustainability Consumption Dashboard")
st.markdown("Enhance your synthetic data and rank users based on weighted resource consumption.")

st.sidebar.markdown("## Data Configuration")
data_source = st.sidebar.radio("Select Data Source", ["Generate Synthetic Data", "Upload CSV Data"], key="data_source")

# Load company data for corporate feature
st.sidebar.markdown("## Corporate Data Configuration")
company_data = load_company_data()
if company_data is not None:
    st.sidebar.success(f"Company data loaded with {len(company_data)} companies")
    st.sidebar.dataframe(company_data.head(3))
else:
    st.sidebar.info("Please upload company data CSV with required columns")
    company_template = pd.DataFrame({
        "Company_Name": ["Company A", "Company B", "Company C"],
        "Sector classification": ["Technology", "Manufacturing", "Healthcare"],
        "Environment Score": [75, 60, 82],
        "ESG Rating": ["A", "B+", "A-"],
        "Category": ["Leader", "Average", "Leader"],
        "Date of Rating": ["2024-03-15", "2024-02-20", "2024-04-01"],
        "Total Employees": [500, 1200, 300]
    })
    company_template_csv = company_template.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "Download Company Data Template",
        data=company_template_csv,
        file_name="company_data_template.csv",
        mime="text/csv"
    )

inverse_scoring = True

if data_source == "Generate Synthetic Data":
    n_users = st.sidebar.number_input("Number of Individuals", min_value=1, value=100, step=1)
    st.sidebar.markdown("## Set Feature Distributions")
    st.sidebar.markdown("### Electricity Consumption")
    dist_elec, params_elec = feature_distribution_ui("Electricity", default_min=100, default_max=1000)
    st.sidebar.markdown("### Water Consumption")
    dist_water, params_water = feature_distribution_ui("Water", default_min=200, default_max=600)
    st.sidebar.markdown("### Commute")
    include_public = st.sidebar.checkbox("Include Public Transport", value=True, key="include_public")
    if include_public:
        public_transport_dist, public_transport_params = feature_distribution_ui("Public_Transport", default_min=0, default_max=500)
    include_private = st.sidebar.checkbox("Include Private Transport", value=True, key="include_private")
    if include_private:
        private_transport_dist, private_transport_params = feature_distribution_ui("Private_Transport", default_min=0, default_max=300)
    st.sidebar.markdown("### Corporate Details")
    include_corporate = st.sidebar.checkbox("Include Corporate Details", value=True, key="include_corporate")

    feature_settings = {
        "Electricity": (dist_elec, params_elec),
        "Water": (dist_water, params_water)
    }
    if include_public:
        feature_settings["Public_Transport"] = (public_transport_dist, public_transport_params)
    if include_private:
        feature_settings["Private_Transport"] = (private_transport_dist, private_transport_params)

    if st.sidebar.button("Generate Synthetic Data"):
        raw = generate_synthetic_data(n_users, feature_settings)
        if include_corporate and company_data is not None:
            selected_companies = company_data.sample(n=n_users, replace=True).reset_index(drop=True)
            raw["Company_Name"] = selected_companies["Company_Name"]
            raw["Sector classification"] = selected_companies["Sector classification"]
            raw["Environment Score"] = selected_companies["Environment Score"]
            raw["ESG Rating"] = selected_companies["ESG Rating"]
            raw["Category"] = selected_companies["Category"]
            raw["Date of Rating"] = selected_companies["Date of Rating"]
            raw["Total Employees"] = selected_companies["Total Employees"]
        st.session_state.synth_data_raw = raw
        feature_constraints = {feat: (params['min'], params['max']) for feat, (_, params) in feature_settings.items()}
        st.session_state.feature_constraints = feature_constraints
        st.session_state.feature_stats = compute_feature_stats(raw)
        st.session_state.feature_settings = feature_settings
        st.sidebar.success("Synthetic data generated successfully!")

elif data_source == "Upload CSV Data":
    st.sidebar.markdown("### Upload your CSV file")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv", key="uploaded_csv")
    if st.sidebar.button("Load Data"):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.synth_data_raw = df
            constraints = {col: (df[col].min(), df[col].max()) for col in df.columns if col not in
                           ["ID", "Company_Name", "Sector classification", "Environment Score", 
                            "ESG Rating", "Category", "Date of Rating", "Total Employees"]}
            st.session_state.feature_constraints = constraints
            st.session_state.feature_stats = compute_feature_stats(df)
            st.sidebar.success("Data loaded successfully!")
        else:
            st.sidebar.error("Please upload a CSV file.")
    template_columns = ["ID", "Electricity", "Water"]
    if st.sidebar.checkbox("Include Public Transport in template", value=True):
        template_columns.append("Public_Transport")
    if st.sidebar.checkbox("Include Private Transport in template", value=True):
        template_columns.append("Private_Transport")
    if st.sidebar.checkbox("Include Corporate Details in template", value=True):
        template_columns += ["Company_Name", "Sector classification", "Environment Score", 
                             "ESG Rating", "Category", "Date of Rating", "Total Employees"]
    template_df = pd.DataFrame(columns=template_columns)
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download CSV Template", data=csv_template, file_name="template.csv", mime="text/csv")

if "synth_data_raw" in st.session_state:
    st.markdown("## Data Preview")
    render_sortable_table(st.session_state.synth_data_raw)
    st.markdown("---")
    st.markdown("## Weighted Score Settings")
    features_for_weight = [col for col in st.session_state.synth_data_raw.columns if col not in
                           ["ID", "Company_Name", "Sector classification", "Environment Score", 
                            "ESG Rating", "Category", "Date of Rating", "Total Employees"]]
    default_weight = round(1 / len(features_for_weight), 4)
    weight_inputs = {}
    for feat in features_for_weight:
        weight_inputs[feat] = st.number_input(f"Weight: {feat}", value=default_weight, step=0.01, format="%.4f", key=f"weight_{feat}")
    weights = weight_inputs
    total_weight = sum(weights.values())
    st.markdown(f"**Total Weight:** {total_weight:.2f}")
    if abs(total_weight - 1.0) > 0.001:
        st.error("Weights must sum exactly to 1!")
    else:
        if st.button("Generate Weighted Score"):
            final_df, feature_contributions = compute_weighted_score(st.session_state.synth_data_raw, weights, inverse=inverse_scoring)
            st.session_state.scored_data = final_df
            st.session_state.weights = weights
            st.success("Weighted score and ranking generated!")
            st.markdown("## Ranked Individuals Based on Weighted Score")
            render_sortable_table(st.session_state.scored_data.reset_index(drop=True))
    if st.checkbox("Show Distribution Graphs"):
        for col in st.session_state.synth_data_raw.columns:
            if col not in ["ID", "Company_Name", "Sector classification", "Environment Score", 
                           "ESG Rating", "Category", "Date of Rating", "Total Employees"]:
                fig, ax = plt.subplots()
                sns.histplot(st.session_state.synth_data_raw[col], bins=30, kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
        if "scored_data" in st.session_state:
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.scored_data["Weighted_Score"], bins=30, kde=True, ax=ax)
            ax.set_title("Distribution of Weighted Score")
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("## Export and Analysis Tools")
    export_feature_details()

    st.markdown("## Test New Customer")
    test_mode = st.radio("Select Input Mode for New Customer", ["Manual Entry", "CSV Upload"], key="test_mode")
    features = [col for col in st.session_state.synth_data_raw.columns if col not in
                ["ID", "Weighted_Score", "Rank", "Z_Score", "Company_Name", "Sector classification", "Environment Score", 
                 "ESG Rating", "Category", "Date of Rating", "Total Employees"]]

    if test_mode == "Manual Entry":
        st.markdown("### Enter Customer Feature Values")
        new_customer = {}

        # Dedicated manual inputs for key features
        if 'Electricity' in features:
            min_e, max_e = st.session_state.feature_constraints.get('Electricity', (0, 0))
            new_customer['Electricity'] = st.number_input(
                f"Electricity Usage (kWh/month)  (Allowed: {min_e} – {max_e})",
                value=min_e, min_value=min_e, max_value=max_e, key="manual_Electricity"
            )
        if 'Water' in features:
            min_w, max_w = st.session_state.feature_constraints.get('Water', (0, 0))
            new_customer['Water'] = st.number_input(
                f"Water Usage (Litres/month)       (Allowed: {min_w} – {max_w})",
                value=min_w, min_value=min_w, max_value=max_w, key="manual_Water"
            )
        if 'Public_Transport' in features:
            min_pu, max_pu = st.session_state.feature_constraints.get('Public_Transport', (0, 0))
            new_customer['Public_Transport'] = st.number_input(
                f"Public Transport (Units/month)   (Allowed: {min_pu} – {max_pu})",
                value=min_pu, min_value=min_pu, max_value=max_pu, key="manual_Public_Transport"
            )
        if 'Private_Transport' in features:
            min_pr, max_pr = st.session_state.feature_constraints.get('Private_Transport', (0, 0))
            new_customer['Private_Transport'] = st.number_input(
                f"Private Transport (Units/month)  (Allowed: {min_pr} – {max_pr})",
                value=min_pr, min_value=min_pr, max_value=max_pr, key="manual_Private_Transport"
            )

        # Input fields for any other features
        for feat in features:
            if feat in ['Electricity', 'Water', 'Public_Transport', 'Private_Transport']:
                continue
            if "feature_constraints" in st.session_state and feat in st.session_state.feature_constraints:
                constraint_min, constraint_max = st.session_state.feature_constraints[feat]
            else:
                constraint_min = st.session_state.synth_data_raw[feat].min()
                constraint_max = st.session_state.synth_data_raw[feat].max()
            new_customer[feat] = st.number_input(
                f"{feat} (Allowed: {constraint_min} – {constraint_max})",
                value=constraint_min, min_value=constraint_min, max_value=constraint_max,
                key=f"test_{feat}"
            )

        # Corporate details (unchanged)...
        if company_data is not None and "Company_Name" in st.session_state.synth_data_raw.columns:
            selected_company = st.selectbox(
                "Select Company",
                options=company_data["Company_Name"].unique(),
                index=0
            )
            company_row = company_data[company_data["Company_Name"] == selected_company].iloc[0]
            st.info(
                f"Sector: {company_row['Sector classification']}, "
                f"Environment Score: {company_row['Environment Score']}, "
                f"ESG Rating: {company_row['ESG Rating']}, "
                f"Category: {company_row['Category']}, "
                f"Rating Date: {company_row['Date of Rating']}, "
                f"Employees: {company_row['Total Employees']}"
            )

        if st.button("Evaluate Customer"):
            norm_values = {}
            for feat in features:
                if "feature_constraints" in st.session_state and feat in st.session_state.feature_constraints:
                    cmin, cmax = st.session_state.feature_constraints[feat]
                else:
                    cmin = st.session_state.synth_data_raw[feat].min()
                    cmax = st.session_state.synth_data_raw[feat].max()
                val = new_customer[feat]
                if val < cmin:
                    st.info(f"{feat} value is below the allowed minimum of {cmin}.")
                elif val > cmax:
                    st.info(f"{feat} value is above the allowed maximum of {cmax}.")
                if cmax == cmin:
                    norm = 1 if inverse_scoring else 1000
                else:
                    if inverse_scoring:
                        norm = (cmax - val) / (cmax - cmin) * 999 + 1
                    else:
                        norm = (val - cmin) / (cmax - cmin) * 999 + 1
                norm_values[feat] = norm

            if "weights" not in st.session_state:
                st.error("Please generate weighted scores first.")
            else:
                weights = st.session_state.weights
                score = sum(norm_values[feat] * weights.get(feat, 0) for feat in features)
                if "scored_data" not in st.session_state:
                    st.error("Please generate weighted scores first.")
                existing_scores = st.session_state.scored_data["Weighted_Score"]
                rank = (existing_scores > score).sum() + 1
                st.success(f"Customer Score: {score:.2f} (Estimated Rank: {rank})")

                contributions = {feat: norm_values[feat] * weights.get(feat, 0) for feat in features}
                sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

                st.markdown("**Feature Contributions:**")
                for feat, contrib in sorted_contrib:
                    pct = (contrib / score * 100) if score else 0
                    st.write(f"{feat}: {contrib:.2f} ({pct:.2f}%)")

                fig, ax = plt.subplots()
                ax.pie(list(contributions.values()), labels=list(contributions.keys()),
                       autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

    else:  # CSV Upload mode
        # (unchanged)
        new_user_template_df = pd.DataFrame(columns=features)
        csv_new_user_template = new_user_template_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download New Customer Template", data=csv_new_user_template,
                           file_name="new_customer_template.csv", mime="text/csv")

        uploaded_test = st.file_uploader("Upload CSV for New Customers", type="csv", key="test_csv")
        if st.button("Evaluate Test Customers"):
            if uploaded_test is not None:
                test_df = pd.read_csv(uploaded_test)
                scores, ranks = [], []
                for i, row in test_df.iterrows():
                    norm_values, _scores, _ranks = {}, [], []
                    for feat in features:
                        if "feature_constraints" in st.session_state and feat in st.session_state.feature_constraints:
                            cmin, cmax = st.session_state.feature_constraints[feat]
                        else:
                            cmin = st.session_state.synth_data_raw[feat].min()
                            cmax = st.session_state.synth_data_raw[feat].max()
                        val = row[feat]
                        if val < cmin:
                            st.info(f"{feat} in row {i+1} below minimum {cmin}.")
                        elif val > cmax:
                            st.info(f"{feat} in row {i+1} above maximum {cmax}.")
                        if cmax == cmin:
                            norm = 1 if inverse_scoring else 1000
                        else:
                            if inverse_scoring:
                                norm = (cmax - val) / (cmax - cmin) * 999 + 1
                            else:
                                norm = (val - cmin) / (cmax - cmin) * 999 + 1
                        norm_values[feat] = norm
                    score = sum(norm_values[feat] * st.session_state.weights.get(feat, 0) for feat in features)
                    scores.append(score)
                    existing_scores = st.session_state.scored_data["Weighted_Score"]
                    rank_val = (existing_scores > score).sum() + 1
                    ranks.append(rank_val)
                test_df["Weighted_Score"] = scores
                test_df["Estimated Rank"] = ranks
                st.write("Evaluated Customer Data", test_df)
            else:
                st.error("Please upload a CSV file.")
