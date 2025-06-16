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

# --- Global Configurations and Data Loading Functions (Unchanged from previous version) ---

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

@st.cache_data
def load_company_data():
    """
    Preload company data from a fixed CSV path.
    """
    try:
        df = pd.read_csv("employees_2024_25_updated.csv")
    except FileNotFoundError:
        st.error(
            "Could not find employees_2024_25_updated.csv in the app directory.\n"
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
        st.error(f"Missing columns in employees_2024_25_updated.csv: {', '.join(missing)}")
        return None
    return df

@st.cache_data
def load_real_electricity_data():
    """Load electricity consumption data from file path"""
    try:
        file_path = "electricity_data.xlsx"
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            file_path = "electricity_data_with_mpce_hhsize.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                st.error("No electricity data file found. Please make sure electricity_data.csv or electricity_data.xlsx exists.")
                return None

        needed = ["hhid","hh_size", "state_name", "sector", "qty_usage_in_1month", "mpce"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            st.error(f"Missing columns in electricity data: {', '.join(missing)}")
            return None

        # Clean state names if necessary
        state_name_mapping = {
             "Delhi (NCT)": "Delhi",
             "Andaman & Nicobar": "Andaman and Nicobar Islands",
             "Jammu & Kashmir": "Jammu and Kashmir",
             "Dadra & Nagar Haveli and Daman & Diu": "Dadra and Nagar Haveli and Daman and Du"
        }
        df["state_name"] = df["state_name"].replace(state_name_mapping)

        # Further filter for states that are in INDIAN_STATES_UTS
        df = df[df['state_name'].isin(INDIAN_STATES_UTS)].copy()
        if df.empty:
            st.warning("No valid electricity data records found after state filtering.")
            return None

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
        return df

    except Exception as e:
        st.error(f"Error loading electricity data: {str(e)}")
        return None

# Initialize session state variables if they don't exist
if 'company_data' not in st.session_state:
    st.session_state.company_data = load_company_data()
if 'electricity_data' not in st.session_state:
    st.session_state.electricity_data = load_real_electricity_data()
    if st.session_state.electricity_data is not None:
        st.session_state.full_electricity_data = st.session_state.electricity_data # Store for overall stats

# Default weights (moved from original weighted score settings section)
if 'weights' not in st.session_state:
    st.session_state.weights = {
        "Electricity_State": 0.125,
        "Electricity_MPCE": 0.125,
        "Water": 0.25,
        "Public_Transport": 0.125,
        "Private_Transport": 0.125,
        "Company": 0.25
    }
if 'sub_weights' not in st.session_state:
    st.session_state.sub_weights = {
        "electricity": {"location": 0.125, "mpce": 0.125},
        "commute": {"public": 0.125, "private": 0.125},
        "water": {"water": 0.25},
        "company": {"company": 0.25}
    }

# Initialize other session state variables that hold user inputs or calculated results
if 'scored_data' not in st.session_state:
    st.session_state.scored_data = pd.DataFrame()
if 'user_electricity' not in st.session_state:
    st.session_state.user_electricity = 0.0
if 'user_state' not in st.session_state:
    st.session_state.user_state = INDIAN_STATES_UTS[0]
if 'user_sector_name' not in st.session_state:
    st.session_state.user_sector_name = 'Rural'
if 'user_mpce_range' not in st.session_state:
    st.session_state.user_mpce_range = (0, 1000) # Default to first range
if 'user_mpce_range_name' not in st.session_state:
    st.session_state.user_mpce_range_name = "₹1-1,000"
if 'household_size' not in st.session_state:
    st.session_state.household_size = 1
if 'water_units' not in st.session_state:
    st.session_state.water_units = 0.0
if 'transport_carbon_results' not in st.session_state:
    st.session_state.transport_carbon_results = {}
if 'calculated_single_customer_score' not in st.session_state:
    st.session_state.calculated_single_customer_score = None
if 'calculated_single_customer_zscore' not in st.session_state:
    st.session_state.calculated_single_customer_zscore = None
if 'crisil_esg_score_input' not in st.session_state:
    st.session_state.crisil_esg_score_input = 0.0 # Initialize Crisil ESG score
if 'selected_company_for_comparison' not in st.session_state:
    st.session_state.selected_company_for_comparison = None
if 'custom_esg_score_for_comparison' not in st.session_state:
    st.session_state.custom_esg_score_for_comparison = 50.0 # Default value
if 'esg_comparison_method' not in st.session_state:
    st.session_state.esg_comparison_method = "Select from list"
if 'esg_analysis_type' not in st.session_state:
    st.session_state.esg_analysis_type = "Employee Range Analysis"
if 'selected_esg_industry' not in st.session_state:
    st.session_state.selected_esg_industry = None
if 'selected_esg_emp_size' not in st.session_state:
    st.session_state.selected_esg_emp_size = "Small (<5,000)"
if 'calculated' not in st.session_state: # For transport carbon footprint section
    st.session_state.calculated = False
if 'total_emissions' not in st.session_state: # For transport carbon footprint section
    st.session_state.total_emissions = 0.0
if 'emissions_data' not in st.session_state: # For transport carbon footprint section
    st.session_state.emissions_data = {}
if 'recommendations' not in st.session_state: # For transport carbon footprint section
    st.session_state.recommendations = []


# Helper function for Z-score calculation based on overall data distribution
@st.cache_data
def calculate_feature_stats(electricity_data):
    if electricity_data is None or electricity_data.empty:
        return {}
    return {
        'Electricity': {
            'mean': electricity_data['qty_usage_in_1month'].mean(),
            'std':  electricity_data['qty_usage_in_1month'].std()
        }
        # Add other features if they have baseline distributions for Z-score calc
    }
st.session_state.feature_stats = calculate_feature_stats(st.session_state.electricity_data)

# Emission factors (moved here for global access, and ensures it's defined before any page uses it)
emission_factors = {
    "two_wheeler": {
        "Scooter": {"petrol": {"min": 0.03, "max": 0.06}, "diesel": {"min": 0.04, "max": 0.07}, "electric": {"min": 0.01, "max": 0.02}},
        "Motorcycle": {"petrol": {"min": 0.05, "max": 0.09}, "diesel": {"min": 0.06, "max": 0.10}, "electric": {"min": 0.01, "max": 0.02}}
    },
    "three_wheeler": {"petrol": {"min": 0.07, "max": 0.12}, "diesel": {"min": 0.08, "max": 0.13}, "electric": {"min": 0.02, "max": 0.03}, "cng": {"min": 0.05, "max": 0.09}},
    "four_wheeler": {
        "small": {"petrol": {"base": 0.12, "uplift": 1.1}, "diesel": {"base": 0.14, "uplift": 1.1}, "cng": {"base": 0.10, "uplift": 1.05}, "electric": {"base": 0.05, "uplift": 1.0}},
        "hatchback": {"petrol": {"base": 0.15, "uplift": 1.1}, "diesel": {"base": 0.17, "uplift": 1.1}, "cng": {"base": 0.12, "uplift": 1.05}, "electric": {"base": 0.06, "uplift": 1.0}},
        "premium_hatchback": {"petrol": {"base": 0.18, "uplift": 1.15}, "diesel": {"base": 0.20, "uplift": 1.15}, "cng": {"base": 0.14, "uplift": 1.10}, "electric": {"base": 0.07, "uplift": 1.0}},
        "compact_suv": {"petrol": {"base": 0.21, "uplift": 1.2}, "diesel": {"base": 0.23, "uplift": 1.2}, "cng": {"base": 0.16, "uplift": 1.15}, "electric": {"base": 0.08, "uplift": 1.0}},
        "sedan": {"petrol": {"base": 0.20, "uplift": 1.2}, "diesel": {"base": 0.22, "uplift": 1.2}, "cng": {"base": 0.16, "uplift": 1.15}, "electric": {"base": 0.08, "uplift": 1.0}},
        "suv": {"petrol": {"base": 0.25, "uplift": 1.25}, "diesel": {"base": 0.28, "uplift": 1.25}, "cng": {"base": 0.20, "uplift": 1.2}, "electric": {"base": 0.10, "uplift": 1.0}},
        "hybrid": {"petrol": {"base": 0.14, "uplift": 1.05}, "diesel": {"base": 0.16, "uplift": 1.05}, "electric": {"base": 0.07, "uplift": 1.0}}
    },
    "public_transport": {
        "base_emission": 0.03, # Base emission factor for generic public transport
        "taxi": {
            "small": {"petrol": {"base": 0.12, "uplift": 1.1}, "diesel": {"base": 0.14, "uplift": 1.1}, "cng": {"base": 0.10, "uplift": 1.05}, "electric": {"base": 0.05, "uplift": 1.0}},
            "hatchback": {"petrol": {"base": 0.15, "uplift": 1.1}, "diesel": {"base": 0.17, "uplift": 1.1}, "cng": {"base": 0.12, "uplift": 1.05}, "electric": {"base": 0.06, "uplift": 1.0}},
            "sedan": {"petrol": {"base": 0.20, "uplift": 1.2}, "diesel": {"base": 0.22, "uplift": 1.2}, "cng": {"base": 0.16, "uplift": 1.15}, "electric": {"base": 0.08, "uplift": 1.0}},
            "suv": {"petrol": {"base": 0.25, "uplift": 1.25}, "diesel": {"base": 0.28, "uplift": 1.25}, "cng": {"base": 0.20, "uplift": 1.2}, "electric": {"base": 0.10, "uplift": 1.0}}
        },
        "bus": {"electric": 0.025, "petrol": 0.05, "diesel": 0.045, "cng": 0.035},
        "metro": 0.015
    }
}

def calculate_emission_stats():
    all_values = []
    for cat in emission_factors["two_wheeler"]:
        for fuel in emission_factors["two_wheeler"][cat]:
            min_val, max_val = emission_factors["two_wheeler"][cat][fuel]["min"], emission_factors["two_wheeler"][cat][fuel]["max"]
            all_values.append((min_val + max_val) / 2)
    for fuel in emission_factors["three_wheeler"]:
        min_val, max_val = emission_factors["three_wheeler"][fuel]["min"], emission_factors["three_wheeler"][fuel]["max"]
        all_values.append((min_val + max_val) / 2)
    for car_type in emission_factors["four_wheeler"]:
        for fuel in emission_factors["four_wheeler"][car_type]:
            base, uplift = emission_factors["four_wheeler"][car_type][fuel]["base"], emission_factors["four_wheeler"][car_type][fuel]["uplift"]
            all_values.append(base * uplift)
    for taxi_type in emission_factors["public_transport"]["taxi"]:
        for fuel in emission_factors["public_transport"]["taxi"][taxi_type]:
            base, uplift = emission_factors["public_transport"]["taxi"][taxi_type][fuel]["base"], emission_factors["public_transport"]["taxi"][taxi_type][fuel]["uplift"]
            all_values.append(base * uplift)
    for fuel in emission_factors["public_transport"]["bus"]:
        all_values.append(emission_factors["public_transport"]["bus"][fuel])
    all_values.append(emission_factors["public_transport"]["metro"])
    return np.mean(all_values), np.std(all_values)

emission_mean, emission_std = calculate_emission_stats()

def calculate_z_score_emission(emission_factor_val):
    """Calculate Z-score for given emission factor"""
    if emission_std == 0:
        return 0
    return (emission_factor_val - emission_mean) / emission_std

# --- PDF Generation Function ---
def generate_pdf(scored_df, customer_df, recommendations, emissions_data):
    """Generate a PDF report for the sustainability analysis"""
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Sustainability Analysis Report', 0, 1, 'C')
    pdf.ln(10)

    # Customer Profile
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Customer Sustainability Profile', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)

    # Convert test_customer to a dictionary
    test_customer_dict = customer_df.iloc[0].to_dict()

    for col in ['Electricity', 'Water', 'Public_Transport', 'Private_Transport', 'Crisil_ESG_Score', 'Sustainability_Score', 'MPCE']:
        if col in test_customer_dict:
            value = test_customer_dict[col]
            if isinstance(value, (int, float)):
                pdf.cell(0, 7, f'{col}: {value:.2f}', 0, 1)
            else:
                pdf.cell(0, 7, f'{col}: {value}', 0, 1) # For non-numeric values
    pdf.ln(5)

    # Comparison with Dataset (for relevant columns)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Comparison with Dataset', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)

    if not scored_df.empty:
        for col in ['Electricity', 'Water', 'Crisil_ESG_Score']: # Add other numeric columns you want to compare
            if col in scored_df.columns and col in test_customer_dict:
                value = test_customer_dict[col]
                # Ensure the column in scored_df is numeric before comparison
                if pd.api.types.is_numeric_dtype(scored_df[col]):
                    percentile = (scored_df[col] < value).mean() * 100
                    pdf.cell(0, 7, f'{col} Percentile: {percentile:.1f}%', 0, 1)
                else:
                    pdf.cell(0, 7, f'{col} (Non-numeric): {value}', 0, 1)
    else:
        pdf.cell(0, 7, "No comprehensive dataset for comparison.", 0, 1)

    pdf.ln(5)

    # Transport Carbon Footprint Results
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Transport Carbon Footprint', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    if emissions_data:
        pdf.cell(0, 7, f"Total Monthly CO2 Emissions: {emissions_data.get('total_emissions', 0):.2f} kg CO2e", 0, 1)
        pdf.cell(0, 7, f"Emission Factor: {emissions_data.get('emission_factor', 0):.4f} kg CO2/km", 0, 1)
        pdf.cell(0, 7, f"Z-Score (Emissions): {emissions_data.get('z_score_emission', 0):.2f}", 0, 1)
        pdf.cell(0, 7, f"Transport Category: {emissions_data.get('emission_category', 'N/A')}", 0, 1)
        pdf.ln(3)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 7, 'Sustainability Recommendations:', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        if recommendations:
            for i, rec in enumerate(recommendations):
                pdf.multi_cell(0, 7, f"{i+1}. {rec}")
        else:
            pdf.cell(0, 7, "No specific recommendations generated.", 0, 1)
    else:
        pdf.cell(0, 7, "No transport carbon footprint data available.", 0, 1)
    pdf.ln(5)


    return pdf.output(dest='S').encode('latin-1')

# --- Page 1: Test New Customer ---
def page_test_customer():
    st.header("1. New Customer Assessment")
    st.markdown("Enter demographic details, resource consumption, and transport choices for a new customer.")

    test_mode = st.radio("Input Mode", ["Manual Entry", "CSV Upload"], key="test_mode")

    if test_mode == "Manual Entry":
        # <<< MODIFICATION START >>>
        # The 'with st.form(...)' block has been removed to allow for immediate widget updates.
        # All widgets will now trigger a script rerun on change.

        st.subheader("Demographics")
        col1, col2 = st.columns(2)
        with col1:
            user_state = st.selectbox("State/UT", options=sorted(INDIAN_STATES_UTS), key="input_state")
            st.session_state.user_state = user_state
        with col2:
            user_sector_name = st.radio("Area Type", ['Rural', 'Urban'], key="input_sector")
            st.session_state.user_sector_name = user_sector_name

        mpce_ranges = ["₹1-1,000", "₹1,000-5,000", "₹5,000-10,000", "₹10,000-25,000", "₹25,000+"]
        mpce_range_values = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 25000), (25000, float('inf'))]
        user_mpce_range_index = st.selectbox(
            "Monthly Per Capita Expenditure (MPCE) Range",
            options=range(len(mpce_ranges)),
            format_func=lambda x: mpce_ranges[x],
            key="input_mpce_range"
        )
        st.session_state.user_mpce_range = mpce_range_values[user_mpce_range_index]
        st.session_state.user_mpce_range_name = mpce_ranges[user_mpce_range_index]

        household_size = st.number_input("Number of people in household", min_value=1, value=st.session_state.household_size, step=1, key="input_household_size")
        st.session_state.household_size = household_size

        st.subheader("CRISIL ESG Rating (Environmental Score)")
        
        company_data = st.session_state.company_data
        if company_data is not None:
            st.markdown("### Compare Your Company's ESG Score with Benchmarks")
            analysis_type = st.radio(
                "Select Analysis Type",
                ["Employee Range Analysis", "Company-Only Analysis"],
                key="crisil_analysis_type"
            )

            employee_size = None
            selected_industry = None

            if analysis_type == "Employee Range Analysis":
                industry_opts = sorted(company_data['Sector_classification'].dropna().unique())
                selected_industry = st.selectbox("Select Industry Sector", industry_opts, key="employee_range_industry")

                employee_size = st.selectbox(
                    "Select Employee Size Category",
                    ["Small (<5,000)", "Medium (5,000 to 15,000)", "Large (>15,000)"],
                    key="employee_size_category"
                )

            comparison_method = st.radio("Comparison Method",
                                        ["Select from list", "Enter custom score"],
                                        key="form_esg_comparison_method")

            # This filtering logic now works instantly because it's not inside a form.
            # When 'selected_industry' or 'employee_size' changes, the script reruns,
            # and this block generates a new, filtered list for the company dropdown.
            if comparison_method == "Select from list":
                df_to_filter = company_data.copy()

                if analysis_type == "Employee Range Analysis":
                    if selected_industry:
                        df_to_filter = df_to_filter[df_to_filter['Sector_classification'] == selected_industry]
                    if employee_size:
                        if employee_size == "Small (<5,000)":
                            df_to_filter = df_to_filter[df_to_filter['Total_Employees'] < 5000]
                        elif employee_size == "Medium (5,000 to 15,000)":
                            df_to_filter = df_to_filter[
                                (df_to_filter['Total_Employees'] >= 5000) & (df_to_filter['Total_Employees'] <= 15000)
                            ]
                        else:
                            df_to_filter = df_to_filter[df_to_filter['Total_Employees'] > 15000]
                
                filtered_companies_list = sorted(df_to_filter['Company_Name'].dropna().unique().tolist())

                if not filtered_companies_list and analysis_type == "Employee Range Analysis":
                    st.warning("No companies match the selected Industry and Employee Size filters.")
                    st.selectbox(
                        "Select Your Company", 
                        ["(None)"], 
                        key="select_company_for_comparison",
                        disabled=True
                    )
                else:
                    st.selectbox(
                        "Select Your Company", 
                        ["(None)"] + filtered_companies_list, 
                        key="select_company_for_comparison"
                    )
            else:
                st.number_input("Enter Your Company's Environment Score",
                                            min_value=0.0, max_value=100.0, value=st.session_state.crisil_esg_score_input,
                                            key="enter_custom_esg_score")

        else:
            st.info("Company data unavailable. ESG comparison features require 'employees_2024_25_updated.csv'.")


        st.subheader("Resource Consumption")
        st.markdown("#### Electricity")
        user_electricity = st.number_input("Your Monthly Electricity Usage (kWh)", min_value=0.0, value=st.session_state.user_electricity, step=10.0, key="input_electricity")
        st.session_state.user_electricity = user_electricity
        user_cost = st.number_input("Your Monthly Electricity Cost (₹) (Optional, for calculation)", min_value=0.0, value=0.0, step=100.0, key="input_electricity_cost")

        st.markdown("#### Water (in Litres)")
        usage_type = st.selectbox("Water Usage Type", ["Daily Usage", "Monthly Usage"], key="input_water_usage_type")
        water_units = st.number_input("Water Consumption Value", min_value=0.0, value=st.session_state.water_units, step=1.0, key="input_water_units")
        st.session_state.water_units = water_units

        if usage_type == "Daily Usage":
            monthly_water_total = water_units * 30 * household_size
        else:
            monthly_water_total = water_units * household_size
        per_person_monthly_water = monthly_water_total / household_size if household_size > 0 else 0

        st.subheader("Transport Carbon Footprint Calculator")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            distance = st.number_input("Daily one-way commute distance (km)", min_value=0.1, value=10.0, step=0.5, key="input_distance")
        with col_t2:
            days_per_week = st.number_input("Commuting days per week", min_value=1, max_value=7, value=5, step=1, key="input_days_per_week")
        with col_t3:
            weeks_per_month = st.number_input("Commuting weeks per month", min_value=1, max_value=5, value=4, step=1, key="input_weeks_per_month")
        total_monthly_km = distance * 2 * days_per_week * weeks_per_month
        
        st.session_state.total_monthly_km = total_monthly_km

        transport_category = st.selectbox("Transport Category", ["Private Transport", "Public Transport", "Both Private and Public"], key="input_transport_category")

        # All the transport calculation logic below now also updates instantly
        emission_factor_calc = 0
        people_count_calc = 1
        vehicle_type_calc = ""
        vehicle_name_calc = ""
        private_emission_factor = 0
        public_emission_factor = 0
        private_vehicle_name = ""
        public_vehicle_name = ""
        current_public_people = 1 # Initialize here

        if transport_category == "Private Transport" or transport_category == "Both Private and Public":
            st.markdown("##### Private Transport Details")
            has_multiple_vehicles = st.checkbox("I have multiple private vehicles", key="input_multiple_vehicles")
            if has_multiple_vehicles:
                # ... (rest of multiple vehicle logic is fine)
                num_vehicles = st.number_input("How many private vehicles?", min_value=2, max_value=10, value=2, step=1, key="input_num_private_vehicles")
                total_private_emission_factor_multiple = 0
                temp_private_names = []
                for i in range(num_vehicles):
                    st.markdown(f"**Vehicle {i+1}:**")
                    col_pv1, col_pv2 = st.columns([1, 2])
                    with col_pv1:
                        pv_category = st.selectbox("Vehicle Category", ["Two Wheeler", "Four Wheeler"], key=f"pv_cat_{i}")
                    if pv_category == "Two Wheeler":
                        col_2w1, col_2w2, col_2w3 = st.columns(3)
                        with col_2w1:
                            w2_category = st.selectbox("2W Category", ["Scooter", "Motorcycle"], key=f"2w_cat_val_{i}")
                        with col_2w2:
                            w2_engine_cc = st.number_input("Engine (cc)", 50, 1500, 150, key=f"2w_cc_val_{i}")
                        with col_2w3:
                            w2_fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "electric"], key=f"2w_fuel_val_{i}")
                        if w2_engine_cc <= 150:
                            ef_2w = emission_factors["two_wheeler"][w2_category][w2_fuel_type]["min"]
                        else:
                            min_ef = emission_factors["two_wheeler"][w2_category][w2_fuel_type]["min"]
                            max_ef = emission_factors["two_wheeler"][w2_category][w2_fuel_type]["max"]
                            ratio = min(1.0, (w2_engine_cc - 150) / 1350)
                            ef_2w = min_ef + ratio * (max_ef - min_ef)
                        total_private_emission_factor_multiple += ef_2w
                        temp_private_names.append(f"{w2_category} ({w2_fuel_type}, {w2_engine_cc}cc)")
                    elif pv_category == "Four Wheeler":
                        col_4w1, col_4w2 = st.columns(2)
                        with col_4w1:
                            w4_car_type = st.selectbox("Car Type", ["small", "hatchback", "premium_hatchback", "compact_suv", "sedan", "suv", "hybrid"], key=f"4w_type_val_{i}")
                        with col_4w2:
                            w4_engine_cc = st.slider("Engine (cc)", 600, 4000, 1200, key=f"4w_cc_val_{i}")
                        w4_fuel_options = ["petrol", "diesel", "cng", "electric"]
                        if w4_car_type == "hybrid":
                            w4_fuel_options = ["petrol", "diesel", "electric"]
                        w4_fuel_type = st.selectbox("Fuel Type", w4_fuel_options, key=f"4w_fuel_val_{i}")
                        base_ef = emission_factors["four_wheeler"][w4_car_type][w4_fuel_type]["base"]
                        uplift = emission_factors["four_wheeler"][w4_car_type][w4_fuel_type]["uplift"]
                        engine_factor = 1.0 + min(1.0, (w4_engine_cc - 600) / 3400) * 0.5 if w4_fuel_type != "electric" else 1.0
                        ef_4w = base_ef * uplift * engine_factor
                        total_private_emission_factor_multiple += ef_4w
                        temp_private_names.append(f"{w4_car_type.replace('_', ' ').title()} ({w4_fuel_type}, {w4_engine_cc}cc)")
                private_emission_factor = total_private_emission_factor_multiple
                private_vehicle_name = f"Multiple Vehicles: {', '.join(temp_private_names)}"
                people_count_calc = 1
            else:
                private_vehicle_type = st.selectbox("Private Vehicle Type", ["Two Wheeler", "Three Wheeler", "Four Wheeler"], key="input_private_vehicle")
                current_private_ef = 0
                current_private_name = ""
                current_private_people = 1
                if private_vehicle_type == "Two Wheeler":
                    col_2w1, col_2w2, col_2w3 = st.columns(3)
                    with col_2w1: w2_category = st.selectbox("Category", ["Scooter", "Motorcycle"], key="2w_cat_input")
                    with col_2w2: w2_engine_cc = st.number_input("Engine (cc)", 50, 1500, 150, key="2w_cc_input")
                    with col_2w3: w2_fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "electric"], key="2w_fuel_input")
                    if w2_engine_cc <= 150: current_private_ef = emission_factors["two_wheeler"][w2_category][w2_fuel_type]["min"]
                    else:
                        min_ef, max_ef = emission_factors["two_wheeler"][w2_category][w2_fuel_type]["min"], emission_factors["two_wheeler"][w2_category][w2_fuel_type]["max"]
                        ratio = min(1.0, (w2_engine_cc - 150) / 1350)
                        current_private_ef = min_ef + ratio * (max_ef - min_ef)
                    current_private_name = f"{w2_category} ({w2_fuel_type}, {w2_engine_cc}cc)"
                    if st.checkbox("Rideshare for Two Wheeler", key="2w_rideshare"): current_private_people = st.slider("People sharing 2W", 1, 2, 1, key="2w_people_share")
                elif private_vehicle_type == "Three Wheeler":
                    col_3w1, col_3w2 = st.columns(2)
                    with col_3w1: w3_engine_cc = st.slider("Engine (cc)", 50, 1000, 200, key="3w_cc_input")
                    with col_3w2: w3_fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "electric", "cng"], key="3w_fuel_input")
                    min_ef, max_ef = emission_factors["three_wheeler"][w3_fuel_type]["min"], emission_factors["three_wheeler"][w3_fuel_type]["max"]
                    ratio = min(1.0, (w3_engine_cc - 50) / 950)
                    current_private_ef = min_ef + ratio * (max_ef - min_ef)
                    current_private_name = f"Three Wheeler ({w3_fuel_type}, {w3_engine_cc}cc)"
                    if st.checkbox("Rideshare for Three Wheeler", key="3w_rideshare"): current_private_people = st.slider("People sharing 3W", 1, 3, 1, key="3w_people_share")
                elif private_vehicle_type == "Four Wheeler":
                    col_4w1, col_4w2 = st.columns(2)
                    with col_4w1: w4_car_type = st.selectbox("Car Type", ["small", "hatchback", "premium_hatchback", "compact_suv", "sedan", "suv", "hybrid"], key="4w_type_input")
                    with col_4w2: w4_engine_cc = st.slider("Engine (cc)", 600, 4000, 1200, key="4w_cc_input")
                    w4_fuel_options = ["petrol", "diesel", "cng", "electric"]
                    if w4_car_type == "hybrid": w4_fuel_options = ["petrol", "diesel", "electric"]
                    w4_fuel_type = st.selectbox("Fuel Type", w4_fuel_options, key="4w_fuel_input")
                    base_ef, uplift = emission_factors["four_wheeler"][w4_car_type][w4_fuel_type]["base"], emission_factors["four_wheeler"][w4_car_type][w4_fuel_type]["uplift"]
                    engine_factor = 1.0 + min(1.0, (w4_engine_cc - 600) / 3400) * 0.5 if w4_fuel_type != "electric" else 1.0
                    current_private_ef = base_ef * uplift * engine_factor
                    current_private_name = f"{w4_car_type.replace('_', ' ').title()} ({w4_fuel_type}, {w4_engine_cc}cc)"
                    if st.checkbox("Rideshare for Four Wheeler", key="4w_rideshare"): current_private_people = st.slider("People sharing 4W", 1, 5, 1, key="4w_people_share")
                private_emission_factor = current_private_ef
                private_vehicle_name = current_private_name
                people_count_calc = current_private_people

        if transport_category == "Public Transport" or transport_category == "Both Private and Public":
            st.markdown("##### Public Transport Details")
            public_transport_mode = st.selectbox("Public Transport Mode", ["Taxi", "Bus", "Metro"], key="input_public_mode")
            current_public_ef = 0
            current_public_name = ""
            current_public_people = 1
            if public_transport_mode == "Taxi":
                col_taxi1, col_taxi2 = st.columns(2)
                with col_taxi1: taxi_car_type = st.selectbox("Taxi Car Type", ["small", "hatchback", "sedan", "suv"], key="taxi_car_type")
                with col_taxi2: taxi_fuel_type = st.selectbox("Taxi Fuel Type", ["petrol", "diesel", "cng", "electric"], key="taxi_fuel_type")
                base_ef, uplift = emission_factors["public_transport"]["taxi"][taxi_car_type][taxi_fuel_type]["base"], emission_factors["public_transport"]["taxi"][taxi_car_type][taxi_fuel_type]["uplift"]
                current_public_ef = base_ef * uplift
                current_public_name = f"Taxi - {taxi_car_type.replace('_', ' ').title()} ({taxi_fuel_type})"
                current_public_people = st.slider("People sharing Taxi", 1, 4, 1, key="taxi_people_share")
            elif public_transport_mode == "Bus":
                bus_fuel_type = st.selectbox("Bus Fuel Type", ["diesel", "cng", "electric", "petrol"], key="bus_fuel_type")
                current_public_ef = emission_factors["public_transport"]["bus"][bus_fuel_type]
                current_public_name = f"Bus ({bus_fuel_type})"
                current_public_people = 1
            else:
                current_public_ef = emission_factors["public_transport"]["metro"]
                current_public_name = "Metro"
                current_public_people = 1
            public_emission_factor = current_public_ef
            public_vehicle_name = current_public_name
        
        # Final emission calculation logic remains the same
        private_trips_input = 1
        total_trips_input = 2

        if transport_category == "Private Transport":
            emission_factor_calc = private_emission_factor
            vehicle_type_calc = "Private Transport"
            vehicle_name_calc = private_vehicle_name
        elif transport_category == "Public Transport":
            emission_factor_calc = public_emission_factor
            vehicle_type_calc = "Public Transport"
            vehicle_name_calc = public_vehicle_name
            people_count_calc = current_public_people
        elif transport_category == "Both Private and Public":
            st.markdown("##### Usage Distribution (for Both Private and Public)")
            private_trips_input = st.number_input("Number of trips per day using private transport", min_value=0, max_value=10, value=1, step=1, key="input_private_trips")
            total_trips_input = st.number_input("Total number of trips per day", min_value=1, max_value=10, value=2, step=1, key="input_total_trips")
            private_ratio = private_trips_input / total_trips_input if total_trips_input > 0 else 0
            public_ratio = 1 - private_ratio
            combined_emission_factor = (private_emission_factor / (people_count_calc if people_count_calc > 0 else 1)) * private_ratio + \
                                       (public_emission_factor / (current_public_people if current_public_people > 0 else 1)) * public_ratio
            emission_factor_calc = combined_emission_factor
            vehicle_type_calc = "Combined Transport"
            vehicle_name_calc = f"{private_vehicle_name} ({private_ratio*100:.0f}%) & {public_vehicle_name} ({public_ratio*100:.0f}%)"
            people_count_calc = 1

        # Use a standard st.button instead of st.form_submit_button
        if st.button("Submit Assessment (Proceed to next page for evaluation)"):
            # This logic now runs only when the button is explicitly clicked.
            st.session_state.esg_analysis_type = st.session_state.crisil_analysis_type
            st.session_state.selected_esg_industry = st.session_state.get('employee_range_industry', None)
            st.session_state.selected_esg_emp_size = st.session_state.get('employee_size_category', "Small (<5,000)")
            st.session_state.selected_company_for_comparison = st.session_state.get('select_company_for_comparison', None)
            st.session_state.crisil_esg_score_input = st.session_state.get('enter_custom_esg_score', 0.0)
            st.session_state.esg_comparison_method = st.session_state.form_esg_comparison_method
            
            # This check remains the same
            if st.session_state.crisil_esg_score_input == 0.0 and st.session_state.user_electricity == 0.0 and st.session_state.water_units == 0.0 and st.session_state.total_monthly_km == 0.0:
                st.error("Please enter at least one input for Crisil ESG, Electricity, Water, or Transport to submit the assessment.")
            else:
                st.success("Customer data submitted! You can now navigate to '4. Evaluation Results & Reporting' to see the assessment.")

                private_monthly_km = 0
                public_monthly_km = 0

                if transport_category == "Private Transport":
                    private_monthly_km = total_monthly_km
                elif transport_category == "Public Transport":
                    public_monthly_km = total_monthly_km
                elif transport_category == "Both Private and Public":
                    private_ratio_calculated = private_trips_input / total_trips_input if total_trips_input > 0 else 0
                    public_ratio_calculated = 1 - private_ratio_calculated
                    private_monthly_km = total_monthly_km * private_ratio_calculated
                    public_monthly_km = total_monthly_km * public_ratio_calculated

                st.session_state.transport_carbon_results = {
                    "total_monthly_km": total_monthly_km,
                    "emission_factor": emission_factor_calc,
                    "people_count": people_count_calc,
                    "vehicle_type": vehicle_type_calc,
                    "vehicle_name": vehicle_name_calc
                }

                st.session_state.single_customer_inputs = {
                    'State': user_state,
                    'Rural/Urban': user_sector_name,
                    'MPCE_Range': st.session_state.user_mpce_range_name,
                    'MPCE_Value': st.session_state.user_mpce_range[1],
                    'People in the household': household_size,
                    'Crisil_ESG_Score': st.session_state.crisil_esg_score_input,
                    'Electricity': user_electricity,
                    'Electricity_Cost': user_cost,
                    'Water_Usage_Type': usage_type,
                    'Water_Value': water_units,
                    'Water_Monthly_Total': monthly_water_total,
                    'Water_Per_Person_Monthly': per_person_monthly_water,
                    'Km_per_month': total_monthly_km,
                    'Transport_Emission_Factor': emission_factor_calc,
                    'Transport_People_Count': people_count_calc,
                    'Vehicle_Type': vehicle_type_calc,
                    'Vehicle_Name': vehicle_name_calc,
                    'Private_Monthly_Km': private_monthly_km,
                    'Public_Monthly_Km': public_monthly_km
                }
                
                st.session_state.calculated = True
                st.session_state.total_emissions = (emission_factor_calc * total_monthly_km) / (people_count_calc if people_count_calc > 0 else 1)
                
                alternatives = {vehicle_name_calc: st.session_state.total_emissions}
                alternatives["Bus (Diesel)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["diesel"]) / (people_count_calc if people_count_calc > 0 else 1)
                alternatives["Bus (CNG)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["cng"]) / (people_count_calc if people_count_calc > 0 else 1)
                alternatives["Bus (Electric)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["electric"]) / (people_count_calc if people_count_calc > 0 else 1)
                alternatives["Metro"] = (total_monthly_km * emission_factors["public_transport"]["metro"]) / (people_count_calc if people_count_calc > 0 else 1)
                if not (vehicle_type_calc == "Four Wheeler" and people_count_calc >= 3):
                    alternatives["Car Pooling (4 people)"] = (total_monthly_km * emission_factors["four_wheeler"]["sedan"]["petrol"]["base"] * emission_factors["four_wheeler"]["sedan"]["petrol"]["uplift"]) / 4
                if not ("electric" in vehicle_name_calc.lower()):
                    alternatives["Electric Car"] = (total_monthly_km * emission_factors["four_wheeler"]["sedan"]["electric"]["base"] * emission_factors["four_wheeler"]["sedan"]["electric"]["uplift"]) / (people_count_calc if people_count_calc > 0 else 1)
                if not ("electric scooter" in vehicle_name_calc.lower()):
                    alternatives["Electric Scooter"] = (total_monthly_km * emission_factors["two_wheeler"]["Scooter"]["electric"]["min"]) / (people_count_calc if people_count_calc > 0 else 1)
                st.session_state.emissions_data = alternatives


    else: # CSV Upload mode
        st.markdown("### Upload Test Data for Bulk Evaluation")

        st.markdown("#### Download Template")
        template_file_path = "customer_template (1).xlsx"
        try:
            with open(template_file_path, "rb") as file:
                st.download_button(
                    label="Download Excel Template (with Data Validation)",
                    data=file,
                    file_name="customer_template_with_validation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except FileNotFoundError:
            st.error(f"Error: Template file not found at '{template_file_path}'. Please make sure the file is in the correct directory.")

        # File uploader for test CSV
        up_test = st.file_uploader("Upload Test Data", type=["csv", "xlsx", "xls"], key="bulk_upload_file_page1")
        if up_test:
            try:
                # Check the file extension to use the correct pandas function
                if up_test.name.endswith(('.xlsx', '.xls')):
                    # Use pd.read_excel for Excel files
                    test_df = pd.read_excel(up_test, header=0, dtype=object)
                    st.success("Successfully loaded Excel file.")
                
                elif up_test.name.endswith('.csv'):
                    # Use pd.read_csv for CSV files
                    # This loop is useful for CSVs which might have different encodings
                    test_df = None
                    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    for encoding in encodings_to_try:
                        try:
                            # Reset file pointer to the beginning for each try
                            up_test.seek(0)
                            test_df = pd.read_csv(up_test, encoding=encoding, dtype=object)
                            st.success(f"Successfully loaded CSV with '{encoding}' encoding.")
                            break # Exit the loop if successful
                        except Exception:
                            continue # Try the next encoding if the current one fails
                    
                    if test_df is None:
                        st.error("Could not read the CSV file with any of the supported encodings. Please check the file format.")
                        st.stop()
                
                else:
                    st.error(f"Unsupported file type: {up_test.name}. Please upload a .csv, .xlsx, or .xls file.")
                    st.stop()

                # Clean the dataframe after a successful read
                test_df = test_df.where(pd.notna(test_df), None)

            except Exception as e:
                st.error(f"Error reading or processing the file: {str(e)}")
                st.stop()

            st.markdown("#### Uploaded Test Data Preview")
            st.dataframe(test_df)
            st.session_state.uploaded_bulk_df = test_df # Store for processing on page 3

            if st.button("Process & Evaluate Bulk Data (Proceed to next page for evaluation)", key="process_batch_btn_page1"):
                # Trigger processing and move to Page 3
                # The actual processing logic is on Page 3
                st.success("Bulk data uploaded. Please navigate to '4. Evaluation Results & Reporting' to see the processed results.")
                st.session_state.current_eval_mode = "Bulk CSV Upload" # Indicate mode for page 3


# --- Page 2: Company & Electricity Data Insights ---
def page_data_comparison():
    st.header("2. Company & Electricity Data Insights")
    st.markdown("Explore overall company performance and electricity consumption trends.")

    col_data_insights = st.container()

    with col_data_insights:
        st.subheader("Company Data Overview")
        company_data = st.session_state.company_data
        if company_data is not None:
            st.success(f"Loaded {len(company_data)} companies")
            st.dataframe(company_data, use_container_width=True)

            st.markdown("#### Company Environmental Performance Comparison")
            company_scores = company_data.sort_values("Environment_Score", ascending=False)
            top_n = min(20, len(company_scores))
            top_companies = company_scores.head(top_n)

            fig = px.bar(top_companies,
                        x="Environment_Score", y="Company_Name", title="Top Companies by Environmental Score",
                        orientation='h', color="Environment_Score", color_continuous_scale="viridis")
            fig.update_layout(xaxis_title="Environment Score", yaxis_title="Company", yaxis=dict(autorange="reversed"), height=600)
            st.plotly_chart(fig, use_container_width=True)

            if "Sector_classification" in company_data.columns:
                sector_avg = company_data.groupby("Sector_classification")["Environment_Score"].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
                fig = px.bar(sector_avg.reset_index(), x='Sector_classification', y='mean', error_y='std',
                            title="Average Environmental Score by Sector", labels={'Sector_classification': 'Sector', 'mean': 'Average Environment Score'})
                fig.update_layout(xaxis_tickangle=-45, height=600, showlegend=False)
                for i in range(len(sector_avg)):
                    fig.add_annotation(x=sector_avg.index[i], y=sector_avg['mean'][i], text=f"n={int(sector_avg['count'][i])}", showarrow=False, yshift=10)
                st.plotly_chart(fig, use_container_width=True)

                fig = px.box(company_data, x="Sector_classification", y="Environment_Score",
                           title="Distribution of Environmental Scores by Sector", color="Sector_classification")
                fig.update_layout(xaxis_title="Sector", yaxis_title="Environment Score", xaxis_tickangle=45, showlegend=False, height=600)
                st.plotly_chart(fig)

                if "Total_Employees" in company_data.columns:
                    fig = px.scatter(company_data, x="Total_Employees", y="Environment_Score", color="Sector_classification",
                                   title="Environment Score vs Company Size", opacity=0.6)
                    fig.update_layout(xaxis_title="Number of Employees", yaxis_title="Environment Score", height=600)
                    st.plotly_chart(fig)
        else:
            st.info("Company data not loaded. Please ensure 'employees_2024_25_updated.csv' is in the directory.")
            template = pd.DataFrame({
                "Company_Name":["A","B","C"], "Sector_classification":["Tech","Manu","Health"],
                "Environment_Score":[70,60,80], "ESG_Rating":["A","B","A-"],
                "Category":["Leader","Average","Leader"], "Total_Employees":[100,200,150]
            })
            st.download_button("Download Company Template", data=template.to_csv(index=False).encode('utf-8'),
                               file_name="company_template.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Electricity Consumption Data Analysis")
        electricity_data = st.session_state.electricity_data
        if electricity_data is not None:
            st.success(f"Loaded {len(electricity_data)} electricity records.")
            states_count = electricity_data['state_name'].nunique()
            unique_states = sorted(electricity_data['state_name'].unique().tolist())
            st.success(f"Loaded electricity data across {len(unique_states)} states and union territories")

            state_options = sorted(electricity_data['state_name'].unique().tolist())
            selected_state = st.selectbox("Select State/UT for detailed view", state_options, key="dashboard_state")

            sector_options = ['Rural', 'Urban']
            selected_sector_name = st.radio("Select Sector for detailed view", sector_options, key="dashboard_sector")
            selected_sector = 1 if selected_sector_name == 'Rural' else 2

            filtered_data = electricity_data[
                (electricity_data['state_name'] == selected_state) &
                (electricity_data['sector'] == selected_sector)
            ]

            col_elec1, col_elec2 = st.columns(2)
            with col_elec1:
                if not filtered_data.empty:
                    avg_electricity = filtered_data['qty_usage_in_1month'].mean()
                    avg_mpce = filtered_data['mpce'].mean()
                    st.metric("Average Electricity", f"{avg_electricity:.2f} kWh")
                    st.metric("Average MPCE", f"₹{avg_mpce:.2f}")
                    if 'hh_size' in filtered_data.columns:
                        avg_hh_size = filtered_data['hh_size'].mean()
                        per_capita_electricity = filtered_data['qty_usage_in_1month'] / filtered_data['hh_size']
                        avg_per_capita_electricity = per_capita_electricity.mean()
                        st.metric("Average Household Size", f"{avg_hh_size:.1f} people")
                        st.metric("Per Capita Electricity", f"{avg_per_capita_electricity:.2f} kWh/person")
                else:
                    st.warning(f"No data available for {selected_state} ({selected_sector_name})")

            with col_elec2:
                st.markdown(f"**Electricity Consumption Distribution - {selected_state}, {selected_sector_name}**")
                if not filtered_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(filtered_data["qty_usage_in_1month"], bins=30, kde=True, ax=ax)
                    ax.set_xlabel("Electricity (kWh/month)")
                    st.pyplot(fig)
                else:
                    st.info("No data to display distribution.")

            if not filtered_data.empty and 'hh_size' in filtered_data.columns:
                st.markdown(f"**Household Size Distribution - {selected_state}, {selected_sector_name}**")
                fig_hh, ax_hh = plt.subplots(figsize=(10, 6))
                hh_size_counts = filtered_data['hh_size'].value_counts().sort_index()
                ax_hh.bar(hh_size_counts.index, hh_size_counts.values, alpha=0.7, edgecolor='black')
                ax_hh.set_xlabel("Number of People in Household")
                ax_hh.set_ylabel("Number of Households")
                st.pyplot(fig_hh)

            st.markdown("#### Overall Electricity Consumption Comparison Across States")
            sector_for_comparison = st.radio("Select Sector for Comparison", ['Rural', 'Urban', 'Both'], key="comparison_sector")

            if sector_for_comparison == 'Both':
                state_avg = electricity_data.groupby('state_name')['qty_usage_in_1month'].mean().reset_index()
                if 'hh_size' in electricity_data.columns:
                    state_hh_avg = electricity_data.groupby('state_name')['hh_size'].mean().reset_index()
                    state_avg = state_avg.merge(state_hh_avg, on='state_name')
                    electricity_data['per_capita_usage'] = electricity_data['qty_usage_in_1month'] / electricity_data['hh_size']
                    state_per_capita_avg = electricity_data.groupby('state_name')['per_capita_usage'].mean().reset_index()
                    state_avg = state_avg.merge(state_per_capita_avg, on='state_name')
                chart_title = "Average Electricity Consumption by State (Rural & Urban)"
            else:
                sector_value = 1 if sector_for_comparison == 'Rural' else 2
                filtered_comparison = electricity_data[electricity_data['sector'] == sector_value]
                state_avg = filtered_comparison.groupby('state_name')['qty_usage_in_1month'].mean().reset_index()
                if 'hh_size' in filtered_comparison.columns:
                    state_hh_avg = filtered_comparison.groupby('state_name')['hh_size'].mean().reset_index()
                    state_avg = state_avg.merge(state_hh_avg, on='state_name')
                    filtered_comparison['per_capita_usage'] = filtered_comparison['qty_usage_in_1month'] / filtered_comparison['hh_size']
                    state_per_capita_avg = filtered_comparison.groupby('state_name')['per_capita_usage'].mean().reset_index()
                    state_avg = state_avg.merge(state_per_capita_avg, on='state_name')
                chart_title = f"Average Electricity Consumption by State ({sector_for_comparison})"

            state_avg = state_avg.sort_values('qty_usage_in_1month', ascending=False)
            fig = px.bar(state_avg, x='state_name', y='qty_usage_in_1month', title=chart_title,
                         labels={'state_name': 'State/UT', 'qty_usage_in_1month': 'Average Electricity (kWh/month)'})
            fig.update_layout(xaxis_tickangle=90)
            fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            if 'hh_size' in electricity_data.columns:
                st.markdown("##### Household Size Comparison Across States")
                hh_tab1, hh_tab2 = st.tabs(["Average Household Size", "Per Capita Electricity"])
                with hh_tab1:
                    state_hh_sorted = state_avg.sort_values('hh_size', ascending=False) if 'hh_size' in state_avg.columns else state_avg
                    fig_hh = px.bar(state_hh_sorted, x='state_name', y='hh_size',
                                    title=f"Average Household Size by State ({sector_for_comparison})",
                                    labels={'state_name': 'State/UT', 'hh_size': 'Average Household Size (people)'},
                                    color='hh_size', color_continuous_scale='Blues')
                    fig_hh.update_layout(xaxis_tickangle=90)
                    fig_hh.update_traces(texttemplate='%{y:.1f}', textposition='outside')
                    st.plotly_chart(fig_hh)
                with hh_tab2:
                    if 'per_capita_usage' in state_avg.columns:
                        state_pc_sorted = state_avg.sort_values('per_capita_usage', ascending=False)
                        fig_pc = px.bar(state_pc_sorted, x='state_name', y='per_capita_usage',
                                        title=f"Per Capita Electricity Consumption by State ({sector_for_comparison})",
                                        labels={'state_name': 'State/UT', 'per_capita_usage': 'Per Capita Electricity (kWh/person/month)'},
                                        color='per_capita_usage', color_continuous_scale='Viridis')
                        fig_pc.update_layout(xaxis_tickangle=90)
                        fig_pc.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                        st.plotly_chart(fig_pc)

            st.markdown("##### Total Electricity Usage by Sectors and States Across India")
            sector_state_tab1, sector_state_tab2 = st.tabs(["By Sector", "By State"])
            with sector_state_tab1:
                sector_total = electricity_data.groupby('sector')['qty_usage_in_1month'].sum().reset_index()
                sector_total['sector_name'] = sector_total['sector'].map({1: 'Rural', 2: 'Urban'})
                fig_pie = px.pie(sector_total, values='qty_usage_in_1month', names='sector_name',
                                title="Total Electricity Usage Distribution by Sector", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
                fig_pie.update_traces(textinfo='percent', textposition='inside')
                total_usage = sector_total['qty_usage_in_1month'].sum()
                fig_pie.update_layout(annotations=[dict(text=f'Total<br>{total_usage:.1f} kWh', x=0.5, y=0.5, font_size=14, showarrow=False)])
                st.plotly_chart(fig_pie)
            with sector_state_tab2:
                state_total = electricity_data.groupby('state_name')['qty_usage_in_1month'].sum().reset_index()
                state_total = state_total.sort_values('qty_usage_in_1month', ascending=False)
                fig_treemap = px.treemap(state_total, path=['state_name'], values='qty_usage_in_1month',
                                       title="Total Electricity Usage Distribution by State", color='qty_usage_in_1month',
                                       color_continuous_scale='Viridis')
                st.plotly_chart(fig_treemap)
                fig_bar = px.bar(state_total, y='state_name', x='qty_usage_in_1month', orientation='h',
                                 title="State-wise Total Electricity Usage", labels={'state_name': 'State/UT', 'qty_usage_in_1month': 'Total Electricity Usage (kWh)'},
                                 color='qty_usage_in_1month', color_continuous_scale='Viridis')
                fig_bar.update_traces(texttemplate='%{x:.1f}', textposition='outside')
                num_states = len(state_total)
                chart_height = max(500, 20 * num_states)
                fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title='Total Electricity Usage (kWh)',
                                      yaxis_title='State/UT', height=chart_height, margin=dict(l=200, r=100, t=50, b=50))
                st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.info("Electricity consumption data not loaded. Please ensure 'electricity_data.xlsx' or 'electricity_data_with_mpce_hhsize.csv' is in the directory.")

# --- Page 3: Weighted Score Settings ---
def page_evaluation():
    st.header("3. Weighted Score Settings")
    st.markdown("Adjust the importance of different sustainability factors.")

    st.markdown("**Electricity Consumption**")
    w_elec_location = st.number_input("Location Based (e.g., State/UT) Weight", value=st.session_state.weights["Electricity_State"], step=0.01, format="%.3f", key="wt_elec_location")
    w_elec_mpce = st.number_input("Economic Based (MPCE) Weight", value=st.session_state.weights["Electricity_MPCE"], step=0.01, format="%.3f", key="wt_elec_mpce")
    st.markdown(f"*(Current Total Electricity Weight: {w_elec_location + w_elec_mpce:.3f}, Target: 0.25)*")

    st.markdown("**Water Consumption**")
    w_water = st.number_input("Water Weight", value=st.session_state.weights["Water"], step=0.01, format="%.3f", key="wt_water")
    st.markdown(f"*(Current Total Water Weight: {w_water:.3f}, Target: 0.25)*")

    st.markdown("**Commute**")
    w_public = st.number_input("Public Transport Weight", value=st.session_state.weights["Public_Transport"], step=0.01, format="%.3f", key="wt_public")
    w_private = st.number_input("Private Transport Weight", value=st.session_state.weights["Private_Transport"], step=0.01, format="%.3f", key="wt_private")
    st.markdown(f"*(Current Total Commute Weight: {w_public + w_private:.3f}, Target: 0.25)*")

    st.markdown("**Company Environmental Score**")
    w_company = st.number_input("Company Environmental Score Weight", value=st.session_state.weights["Company"], step=0.01, format="%.3f", key="wt_company")
    st.markdown(f"*(Current Total Company Weight: {w_company:.3f}, Target: 0.25)*")

    current_weights = {
        "Electricity_State": w_elec_location,
        "Electricity_MPCE": w_elec_mpce,
        "Water": w_water,
        "Public_Transport": w_public,
        "Private_Transport": w_private,
        "Company": w_company
    }
    total_current_weight = sum(current_weights.values())
    st.markdown(f"**Overall Total Weight:** {total_current_weight:.3f}")

    if abs(total_current_weight - 1.0) > 1e-3:
        st.error("Overall total weights must sum exactly to 1.0!")
    else:
        if st.button("Apply Weighted Score Settings"):
            st.session_state.weights = current_weights
            st.session_state.sub_weights = {
                "electricity": {"location": w_elec_location, "mpce": w_elec_mpce},
                "commute": {"public": w_public, "private": w_private},
                "water": {"water": w_water},
                "company": {"company": w_company}
            }
            st.success("Weighted score settings applied successfully!")

# --- Page 4: Evaluation Results & Reporting ---
def page_reporting():
    st.header("4. Evaluation Results & Reporting")
    st.markdown("View detailed sustainability assessment for individual customers or process bulk data, and generate reports.")
    
    # Define MPCE ranges and values at the start of the function
    mpce_ranges = ["₹1-1,000", "₹1,000-5,000", "₹5,000-10,000", "₹10,000-25,000", "₹25,000+"]
    mpce_range_values = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 25000), (25000, float('inf'))]
  # Determine which mode to display results for
    evaluation_mode_display = st.session_state.get('current_eval_mode', 'Individual Customer Evaluation') # Default

    if evaluation_mode_display == "Individual Customer Evaluation":
        st.subheader("Individual Customer Assessment Results")

        # <<< START: MODIFIED MULTILEVEL DONUT CHART >>>
        st.markdown("#### Weighted Score Contributions")
        st.markdown("This chart shows the weighting of each factor in the total sustainability score, as configured on Page 3.")

        # Get weights from session state
        weights = st.session_state.weights

        # Data for the multi-level donut chart
        labels = [
            "Electricity", "Water", "Commute", "Company",
            "Location Based", "Economic Based", "Public Transport", "Private Transport"
        ]
        parents = [
            "", "", "", "",
            "Electricity", "Electricity", "Commute", "Commute"
        ]

        # Calculate parent values from their children
        electricity_total = weights["Electricity_State"] + weights["Electricity_MPCE"]
        commute_total = weights["Public_Transport"] + weights["Private_Transport"]

        # The values for all nodes including calculated parent values
        values = [
            electricity_total,  # Electricity parent (sum of children)
            weights["Water"],
            commute_total,      # Commute parent (sum of children)
            weights["Company"],
            weights["Electricity_State"],
            weights["Electricity_MPCE"],
            weights["Public_Transport"],
            weights["Private_Transport"]
        ]

        # Check if all weight values are zero to avoid errors
        if sum(weights.values()) == 0:
            st.warning("All weight values are zero. The chart cannot be displayed until weights are set on Page 3.")
        else:
            try:
                fig = go.Figure(go.Sunburst(
                    labels=labels,
                    parents=parents,
                    values=values,
                    branchvalues="total",
                    hovertemplate='<b>%{label}</b><br>Weight: %{value:.3f}<extra></extra>',
                    insidetextorientation='radial',
                    maxdepth=2,  # Limit depth for better visualization
                ))

                # Update layout for better visibility
                fig.update_layout(
                    margin=dict(t=60, l=20, r=20, b=20),
                    title_text="Weight Distribution for Sustainability Score",
                    title_x=0.5,
                    title_font_size=16,
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="black", size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred while creating the chart: {e}")
                # <<< END: MODIFIED MULTILEVEL DONUT CHART >>>

        if 'single_customer_inputs' not in st.session_state or not st.session_state.single_customer_inputs:
            st.info("Please go to '1. New Customer Assessment' page and submit customer details (Manual Entry mode) first.")
            return

        customer_data = st.session_state.single_customer_inputs

        # Re-calculate values based on stored session state values for display consistency
        # These calculations must use the values retrieved from session state, not the input widgets from Page 1.
        user_electricity = customer_data['Electricity']
        household_size = customer_data['People in the household']
        user_state = customer_data['State']
        user_sector_name = customer_data['Rural/Urban']
        user_sector = 1 if user_sector_name == 'Rural' else 2
        user_mpce_range_name = customer_data['MPCE_Range']
        user_mpce_value = customer_data['MPCE_Value'] # Using upper bound for calculation
        crisil_esg = customer_data['Crisil_ESG_Score']
        monthly_water_total = customer_data['Water_Monthly_Total']
        per_person_monthly_water = customer_data['Water_Per_Person_Monthly']
        total_monthly_km_disp = customer_data['Km_per_month'] # Renamed to avoid conflict with `total_monthly_km` from input form
        emission_factor_disp = customer_data['Transport_Emission_Factor'] # Renamed to avoid conflict with `emission_factor_calc`
        people_count_disp = customer_data['Transport_People_Count'] # Renamed to avoid conflict with `people_count_calc`
        vehicle_name_disp = customer_data['Vehicle_Name'] # Renamed to avoid conflict with `vehicle_name_calc`
        user_cost = customer_data['Electricity_Cost'] # Ensure user_cost is available
        mpce_ranges = ["₹1-1,000", "₹1,000-5,000", "₹5,000-10,000", "₹10,000-25,000", "₹25,000+"]
        user_mpce_range_index = next((i for i, r_name in enumerate(mpce_ranges) if r_name == user_mpce_range_name), 0)


        # Core calculations (replicated from original script)
        safe_hh_size = household_size if household_size > 0 else 1

        # Calculate equivalence scale factor based on household size
        if safe_hh_size == 1:
            equivalence_factor = 1
        elif safe_hh_size == 2:
            equivalence_factor = 1.75
        elif safe_hh_size == 3:
            equivalence_factor = 2.5
        elif safe_hh_size == 4:
            equivalence_factor = 3.25
        else:
            # For 5+ people: equivalence_factor = household_size - 1
            equivalence_factor = safe_hh_size - 1

        individual_equivalent_consumption = user_electricity / equivalence_factor if equivalence_factor > 0 else 0

        # --- FIX 1: Electricity Z-score based on individual_equivalent_consumption ---
        # --- UNIFIED SUSTAINABILITY SCORE CALCULATION ---
        # This logic replicates the calculation from the "Customer Score Results" section
        # to ensure the "Individual Sustainability Score" is consistent.

        # 1. Define the customer data dictionary used in the calculation
        temp_new_customer = {
            'Water': per_person_monthly_water,
            'Electricity': individual_equivalent_consumption,
            'Private_Transport': customer_data.get('Private_Monthly_Km', 0.0),
            'Public_Transport': customer_data.get('Public_Monthly_Km', 0.0),
            'Company Score': crisil_esg
        }

        # 2. Calculate the component scores with hardcoded weights and baselines
        total_z_score = 0

        # Water component (25% weight)
        # Compares against a baseline of 3900L/person/month
        water_z = (temp_new_customer['Water'] - 3900) / 3900
        total_z_score += water_z * 0.25

        # Electricity component (25% weight)
        # Compares against a baseline of 300 kWh for individual equivalent usage
        electricity_z = max(0, (individual_equivalent_consumption - 300) / 300)
        total_z_score += electricity_z * 0.25

        # Private transport component (25% weight)
        # Normalizes based on 1000 km/month
        private_transport_z = temp_new_customer.get('Private_Transport', 0) / 1000
        total_z_score += private_transport_z * 0.25

        # Public transport component (15% weight, gives a bonus for usage)
        public_transport_z = -min(1, temp_new_customer.get('Public_Transport', 0) / 500)
        total_z_score += public_transport_z * 0.15

        # Company score component (10% weight, gives a bonus for higher scores)
        company_z = -(temp_new_customer.get('Company Score', 0) / 100)
        total_z_score += company_z * 0.10

        # 3. Calculate the final sustainability score
        sustainability_score = 500 * (1 - np.tanh(total_z_score / 2.5))
        # --- END OF UNIFIED CALCULATION ---


        st.session_state.calculated_single_customer_score = sustainability_score
        st.session_state.calculated_single_customer_zscore = total_z_score

        # Display results for individual customer
        st.markdown("#### Individual Sustainability Score")
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Total Sustainability Score", f"{sustainability_score:.1f}/500")
        with col_res2:
            st.metric("Combined Z-score", f"{total_z_score:.2f}")
        with col_res3:
            if st.session_state.scored_data is not None and not st.session_state.scored_data.empty:
                percentile = (st.session_state.scored_data['Sustainability_Score'] < sustainability_score).mean() * 100
                st.metric("Your Percentile", f"{percentile:.1f}%")
            else:
                st.info("No batch data available for percentile comparison.")

        st.markdown("#### Feature Breakdown (Z-Scores)")
        transport_z = private_transport_z + public_transport_z
        feature_z_df = pd.DataFrame({
            'Feature': ['Electricity (Individual Equivalent)', 'Water', 'Transport', 'Company ESG'],
            'Z-Score': [electricity_z, water_z, transport_z, company_z] 
        })
        st.dataframe(feature_z_df, use_container_width=True)

        st.markdown("#### Your Electricity Consumption Analysis")
        if st.session_state.electricity_data is not None:
            state_sector_data_filtered = st.session_state.electricity_data[
                (st.session_state.electricity_data['state_name'] == user_state) &
                (st.session_state.electricity_data['sector'] == user_sector)
            ]

            if not state_sector_data_filtered.empty:
                state_avg_elec = state_sector_data_filtered['qty_usage_in_1month'].mean()
                st.metric(f"Your Monthly Electricity Usage ({user_state}, {user_sector_name})", f"{user_electricity:.2f} kWh")
                st.metric(f"Average Monthly Electricity Usage ({user_state}, {user_sector_name})", f"{state_avg_elec:.2f} kWh")

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(state_sector_data_filtered['qty_usage_in_1month'], bins=15, kde=True, ax=ax)
                ax.axvline(user_electricity, color='red', linestyle='--', label='Your Usage')
                ax.axvline(state_avg_elec, color='green', linestyle='--', label=f'{user_state} Average')
                ax.set_title(f"Your Electricity Usage Compared to {user_state}, {user_sector_name}")
                ax.set_xlabel("Monthly Consumption (kWh)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning(f"No specific electricity data available for {user_state} ({user_sector_name}) to compare.")
        else:
            st.info("Electricity data not loaded for comparison.")

        # --- START NEWLY ADDED CODE (Electricity Section) ---
        if user_electricity > 0:
            st.markdown("#### Individual Electricity Consumption")
            ind_col1, ind_col2 = st.columns(2)
            
            with ind_col1:
                st.metric("Total Household Usage", f"{user_electricity:.1f} kWh")
                
            with ind_col2:
                st.metric("Individual Equivalent Usage", f"{individual_equivalent_consumption:.1f} kWh",
                                f"Based on {household_size} people")

            electricity_data = st.session_state.electricity_data # Ensure local variable is set
            if electricity_data is not None and 'hh_size' in electricity_data.columns:
                state_sector_hh_data = electricity_data[
                    (electricity_data['state_name'] == user_state) & 
                    (electricity_data['sector'] == user_sector)
                ]
                
                if not state_sector_hh_data.empty:
                    state_hh_avg = state_sector_hh_data['hh_size'].mean()
                    state_hh_median = state_sector_hh_data['hh_size'].median()
                    
                    st.markdown("#### Household Size Comparison")
                    hh_col1, hh_col2, hh_col3 = st.columns(3)
                    
                    with hh_col1:
                        st.metric("Your Household Size", f"{household_size} people")
                    
                    with hh_col2:
                        comparison = "larger" if household_size > state_hh_avg else "smaller" if household_size < state_hh_avg else "same as"
                        st.metric(f"{user_state} ({user_sector_name}) Average", 
                                    f"{state_hh_avg:.1f} people",
                                    f"Your household is {comparison} average")
                    
                    with hh_col3:
                        percentile_hh = (state_sector_hh_data['hh_size'] <= household_size).mean() * 100
                        st.metric("Your Percentile", 
                                    f"{percentile_hh:.1f}%",
                                    f"Larger than {percentile_hh:.1f}% of households")
                    
                    fig_hh, ax_hh = plt.subplots(figsize=(10, 6))
                    
                    sns.histplot(data=state_sector_hh_data, x='hh_size', bins=range(1, int(state_sector_hh_data['hh_size'].max()) + 2), 
                                kde=True, ax=ax_hh, alpha=0.7)
                    
                    ax_hh.axvline(household_size, color='red', linestyle='-', linewidth=2, label=f'Your Household ({household_size} people)')
                    ax_hh.axvline(state_hh_avg, color='green', linestyle='--', linewidth=2, label=f'State Average ({state_hh_avg:.1f} people)')
                    ax_hh.axvline(state_hh_median, color='orange', linestyle='--', linewidth=2, label=f'State Median ({state_hh_median:.1f} people)')
                    
                    ax_hh.set_title(f"Household Size Distribution in {user_state} ({user_sector_name})")
                    ax_hh.set_xlabel("Household Size (Number of People)")
                    ax_hh.set_ylabel("Frequency")
                    ax_hh.legend()
                    ax_hh.grid(True, alpha=0.3)
                    
                    st.pyplot(fig_hh)
                    
                    st.markdown("#### Household Size Comparison Across States")
                    state_hh_comparison = electricity_data.groupby('state_name')['hh_size'].agg(['mean', 'median']).reset_index()
                    state_hh_comparison = state_hh_comparison.sort_values('mean')
                    state_hh_comparison['is_user_state'] = state_hh_comparison['state_name'] == user_state
                    fig_states, ax_states = plt.subplots(figsize=(14, 8))
                    
                    bars = ax_states.bar(state_hh_comparison['state_name'], 
                                        state_hh_comparison['mean'],
                                        color=state_hh_comparison['is_user_state'].map({True: 'red', False: 'skyblue'}),
                                        alpha=0.8)
                    
                    ax_states.axhline(y=household_size, color='green', linestyle='--', linewidth=2,
                                    label=f'Your Household Size ({household_size} people)')
                    
                    user_state_avg_hh = state_hh_comparison[state_hh_comparison['is_user_state']]['mean'].values[0]
                    ax_states.scatter(user_state, user_state_avg_hh, s=150, color='darkred', zorder=5,
                                    label=f'{user_state} Average ({user_state_avg_hh:.1f} people)')
                    
                    ax_states.set_title('Average Household Size by State/UT')
                    ax_states.set_xlabel('State/UT')
                    ax_states.set_ylabel('Average Household Size (People)')
                    plt.xticks(rotation=90)
                    ax_states.legend()
                    ax_states.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    st.pyplot(fig_states)
                    
                    if len(electricity_data['sector'].unique()) > 1:
                        st.markdown("#### Rural vs Urban Household Size Comparison")
                        
                        sector_hh_comparison = electricity_data.groupby(['state_name', 'sector'])['hh_size'].mean().reset_index()
                        sector_hh_comparison['sector_name'] = sector_hh_comparison['sector'].map({1: 'Rural', 2: 'Urban'})
                        
                        user_state_sectors = sector_hh_comparison[sector_hh_comparison['state_name'] == user_state]
                        
                        if not user_state_sectors.empty:
                            fig_sector, ax_sector = plt.subplots(figsize=(8, 6))
                            
                            bars_sector = ax_sector.bar(user_state_sectors['sector_name'], 
                                                        user_state_sectors['hh_size'],
                                                        color=['lightcoral' if s == user_sector_name else 'lightblue' 
                                                                for s in user_state_sectors['sector_name']],
                                                        alpha=0.8)
                            
                            ax_sector.axhline(y=household_size, color='green', linestyle='--', linewidth=2,
                                            label=f'Your Household Size ({household_size} people)')
                            
                            user_sector_avg = user_state_sectors[user_state_sectors['sector_name'] == user_sector_name]['hh_size'].values
                            if len(user_sector_avg) > 0:
                                ax_sector.scatter(user_sector_name, user_sector_avg[0], s=150, color='darkred', zorder=5,
                                                label=f'Your Sector Average ({user_sector_avg[0]:.1f} people)')
                            
                            ax_sector.set_title(f'Household Size Comparison: Rural vs Urban in {user_state}')
                            ax_sector.set_xlabel('Sector')
                            ax_sector.set_ylabel('Average Household Size (People)')
                            ax_sector.legend()
                            ax_sector.grid(True, alpha=0.3)
                            
                            st.pyplot(fig_sector)
                        else:
                            st.info(f"No household size data available for {user_state} ({user_sector_name})")
            else:
                st.info("Household size data not available in the dataset for comparison")

            if user_electricity > 0:
                st.metric("Your Electricity Consumption", f"{user_electricity:.2f} kWh")
                
                if household_size and household_size > 0:
                    st.metric("Per Person Electricity Consumption", f"{individual_equivalent_consumption:.2f} kWh/person")
                else:
                    st.metric("Per Person Electricity Consumption", "N/A (household size not specified)")
                
                if user_cost > 0:
                    def calculate_electricity_from_cost(cost, household_size_arg):
                        if household_size_arg == 1:
                            return cost / 1
                        elif household_size_arg == 2:
                            return cost / 1.75
                        elif household_size_arg == 3:
                            return cost / 2.5
                        elif household_size_arg == 4:
                            return cost / 3.25
                        else:  # 5 or more people
                            return cost / (household_size_arg - 1)
                    
                    # Auto-calculate user_electricity based on cost and household size
                    safe_hh_size_calc = household_size if household_size and household_size > 0 else 1
                    # Note: user_electricity is already passed as a variable at the top of the function
                    # For this specific block, we'll re-calculate if user_cost > 0 for display purposes.
                    calculated_electricity_from_cost = calculate_electricity_from_cost(user_cost, safe_hh_size_calc)
                    
                    # Display cost-based calculation metrics
                    st.markdown("### Cost-Based Electricity Usage Calculation")
                    
                    if safe_hh_size_calc == 1:
                        divisor = 1
                    elif safe_hh_size_calc == 2:
                        divisor = 1.75
                    elif safe_hh_size_calc == 3:
                        divisor = 2.5
                    elif safe_hh_size_calc == 4:
                        divisor = 3.25
                    else:  # 5 or more people
                        divisor = safe_hh_size_calc - 1
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Monthly Cost", f"₹{user_cost:.2f}")
                    with col2:
                        st.metric("Household Size", f"{safe_hh_size_calc} people")
                    with col3:
                        st.metric("Individual Cost", f"{calculated_electricity_from_cost:.2f} ₹")
        
        electricity_data = st.session_state.electricity_data # Ensure local variable is set
        if electricity_data is not None:
            state_sector_data = electricity_data[
                (electricity_data['state_name'] == user_state) & 
                (electricity_data['sector'] == user_sector)
            ]
            
            if not state_sector_data.empty:
                state_avg = state_sector_data['qty_usage_in_1month'].mean()
                
                if 'hh_size' in state_sector_data.columns:
                    state_avg_per_person = state_sector_data.apply(
                        lambda x: x['qty_usage_in_1month'] / x['hh_size'] if x['hh_size'] > 0 else x['qty_usage_in_1month'],
                        axis=1
                    ).mean()
                else:
                    state_avg_per_person = state_avg
                
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



        st.session_state.feature_stats = {
            'Electricity': {
                'mean': st.session_state.full_electricity_data['qty_usage_in_1month'].mean(),
                'std':  st.session_state.full_electricity_data['qty_usage_in_1month'].std()
            }
        }



             # Initialize baseline values if not present
        if 'baseline_values_by_state' not in st.session_state:
            st.session_state.baseline_values_by_state = {}
        
        if 'feature_stats' in st.session_state and 'Electricity' in st.session_state.feature_stats:
            # Removed feature1_score and feature1_z_score calculations that were based on Electricity_State
            # as per instruction 1.
            
            # Removed feature2_score and feature2_z_score calculations that were based on Electricity_MPCE
            # as per instruction 1.
            
            # Removed total_sust_score, combined_z_score, and percentile metrics that were based on 
            # feature1_score and feature2_score, as per instruction 1, as they are redundant with the
            # main sustainability score calculation at the top of page_evaluation.
            
            # Removed "Your Sustainability Analysis" section with the 3 columns and feature breakdown bar chart
            # as it was based on the old electricity Z-score breakdown.

            # Create stats per state from electricity data
            state_stats = {}
            if 'full_electricity_data' in st.session_state:
                for state in st.session_state.full_electricity_data['state_name'].unique():
                    state_data = st.session_state.full_electricity_data[
                        st.session_state.full_electricity_data['state_name'] == state
                    ]
                    state_stats[state] = {
                        'Electricity': (
                            state_data['qty_usage_in_1month'].mean(),
                            state_data['qty_usage_in_1month'].std()
                        )
                    }

                st.session_state.feature_stats_by_state = {
                    state: {
                        'Electricity': {
                            'mean': vals['Electricity'][0],
                            'std':  vals['Electricity'][1]
                        }
                    }
                    for state, vals in state_stats.items()
                }


            if 'feature_stats_by_state' in st.session_state:
                st.markdown("### Your Usage Compared to State Averages")
                state_data = []
                for state, stats in st.session_state.feature_stats_by_state.items():
                    if 'Electricity' in stats:
                        state_data.append({
                            'State_UT': state,
                            'Average_Electricity': stats['Electricity']['mean'],
                            'Is_Your_State': state == user_state
                        })
                
                state_df = pd.DataFrame(state_data)
                state_df = state_df.sort_values('Average_Electricity', ascending=True)
                if state_df.empty:
                    st.warning("No electricity data available for comparison")
                
                
                fig, ax = plt.subplots(figsize=(12, 8))
                bars = ax.bar(state_df['State_UT'], state_df['Average_Electricity'], 
                                color=state_df['Is_Your_State'].map({True: 'red', False: 'blue'}))
                
                ax.axhline(y=user_electricity, color='green', linestyle='--', 
                                label=f'Your Usage ({user_electricity:.1f} kWh, {individual_equivalent_consumption:.1f} kWh/person)')
                
                user_state_avg = state_df[state_df['Is_Your_State']]['Average_Electricity'].values[0] \
                                    if len(state_df[state_df['Is_Your_State']]) > 0 else 0
                if user_state_avg > 0:
                    ax.scatter(user_state, user_state_avg, s=100, color='red', zorder=3,
                                    label=f'{user_state} Avg ({user_state_avg:.1f} kWh)')
                
                plt.xticks(rotation=90)
                plt.xlabel('State/UT')
                plt.ylabel('Average Electricity Consumption (kWh)')
                plt.title('Your Electricity Usage vs. State Averages')
                plt.legend()
                plt.tight_layout()
                
                st.pyplot(fig)
            

            # after you compute baseline mpce_stats_by_range, e.g. in load_data():
            # Calculate baseline MPCE stats from electricity data if available
            baseline_mpce_stats = {}
            if 'full_electricity_data' in st.session_state and 'mpce' in st.session_state.full_electricity_data.columns:
                for i, (lower, upper) in enumerate(mpce_range_values):
                    range_data = st.session_state.full_electricity_data[
                        (st.session_state.full_electricity_data['mpce'] >= lower) & 
                        (st.session_state.full_electricity_data['mpce'] < upper)
                    ]
                    if not range_data.empty:
                        baseline_mpce_stats[mpce_ranges[i]] = (
                            range_data['qty_usage_in_1month'].mean(),
                            range_data['qty_usage_in_1month'].std()
                        )
                    else:
                        # Use overall stats as fallback if no data for this range
                        baseline_mpce_stats[mpce_ranges[i]] = (
                            st.session_state.feature_stats['Electricity']['mean'],
                            st.session_state.feature_stats['Electricity']['std']
                        )
            
            st.session_state.mpce_stats = {
                idx: {
                    'range_name': range_name,
                    'mean': stats[0],
                    'std': stats[1]
                }
                for idx, (range_name, stats) in enumerate(baseline_mpce_stats.items())
            }


            if 'mpce_stats' in st.session_state:
                st.markdown("### Your Usage Compared to MPCE Ranges")
                
                mpce_data = []
                for idx, stats in st.session_state.mpce_stats.items():
                    mpce_data.append({
                        'MPCE_Range': stats['range_name'],
                        'Average_Electricity': stats['mean'], # Use mean directly from stats, no need to multiply by safe_hh_size
                        'Is_Your_Range': idx == user_mpce_range_index
                    })
                
                mpce_df = pd.DataFrame(mpce_data)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(mpce_df.index, mpce_df['Average_Electricity'], 
                                color=mpce_df['Is_Your_Range'].map({True: 'red', False: 'blue'}))
                
                ax.axhline(y=user_electricity, color='green', linestyle='--', 
                                label=f'Your Usage ({user_electricity:.1f} kWh, {individual_equivalent_consumption:.1f} kWh/person)')
                
                user_mpce_avg = mpce_df[mpce_df['Is_Your_Range']]['Average_Electricity'].values[0] \
                                    if len(mpce_df[mpce_df['Is_Your_Range']]) > 0 else 0
                if user_mpce_avg > 0 and 'user_mpce_range_name' in st.session_state:
                    ax.scatter(st.session_state.user_mpce_range_name, user_mpce_avg, s=100, color='red', zorder=3,
                                    label=f'Your MPCE Range Avg ({user_mpce_avg:.1f} kWh)')
                
                plt.xticks(mpce_df.index, mpce_df['MPCE_Range'], rotation=45) # Set ticks to range names
                plt.xlabel('MPCE Range')
                plt.ylabel('Average Electricity Consumption (kWh)')
                plt.title('Your Electricity Usage vs. MPCE Range Averages')
                plt.legend()
                plt.tight_layout()
                
                st.pyplot(fig)
        # --- END NEWLY ADDED CODE (Electricity Section) ---

        st.markdown("#### Your Water Consumption Analysis")
        avg_water_liters_per_person = 130 # liters per day
        st.metric("Your Water Usage (L/person/month)", f"{per_person_monthly_water:.2f} L")
        st.metric("Average Water Usage (L/person/month)", f"{avg_water_liters_per_person * 30:.2f} L")
        if per_person_monthly_water > avg_water_liters_per_person * 30:
            st.warning("Your water usage is higher than the average.")
        else:
            st.success("Your water usage is at or below the average.")

        st.markdown("#### Your Transport Carbon Footprint")
        if st.session_state.transport_carbon_results:
            # Variables from session state
            total_monthly_km = st.session_state.transport_carbon_results["total_monthly_km"]
            emission_factor = st.session_state.transport_carbon_results["emission_factor"]
            people_count = st.session_state.transport_carbon_results["people_count"]
            vehicle_name = st.session_state.transport_carbon_results["vehicle_name"]
            vehicle_type = customer_data['Vehicle_Type'] # Use vehicle_type from customer_data to get overall transport type
            
            # Recalculate total emissions based on per-person logic if applicable
            # The calculation `(emission_factor_disp * total_monthly_km_disp) / (people_count_disp if people_count_disp > 0 else 1)`
            # was already stored in `st.session_state.total_emissions` upon form submission.
            total_emissions_per_person = st.session_state.total_emissions 

            st.metric("Monthly CO₂ Emissions", f"{total_emissions_per_person:.1f} kg CO₂e")
            st.metric("Emission Factor", f"{emission_factor:.4f} kg CO2/km")
            st.metric("Vehicle Used", vehicle_name)

            z_score_emission = calculate_z_score_emission(emission_factor) # Use the global mean/std for Z-score
            st.metric("Emission Z-Score", f"{z_score_emission:.2f}")

            # Store updated transport_carbon_results for PDF generation
            st.session_state.transport_carbon_results['total_emissions'] = total_emissions_per_person
            st.session_state.transport_carbon_results['z_score_emission'] = z_score_emission
            st.session_state.transport_carbon_results['emission_category'] = "High" if z_score_emission > 1 else ("Average" if z_score_emission > -1 else "Low")
            # Recommendations are now calculated and stored separately below.


            # --- START NEWLY ADDED CODE (Transport Section) ---
            if emission_factor > 0:
                z_score = calculate_z_score_emission(emission_factor) # Use the correct function
                
                st.header("Emission Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Emission Factor", f"{emission_factor:.4f} kg CO2/km")
                
                with col2:
                    st.metric("Z-Score", f"{z_score:.2f}")
                
                with col3:
                    if z_score < -1:
                        emission_category = "Low Emissions"
                        color = "🟢"
                    elif z_score < 1:
                        emission_category = "Average Emissions"
                        color = "🟡"
                    else:
                        emission_category = "High Emissions"
                        color = "🔴"
                    
                    st.metric("Category", f"{color} {emission_category}")

            if st.session_state.get('calculated', False): # Use .get for safety
                total_kg = st.session_state.total_emissions # Per person total emissions
                
                # Alternatives were already calculated and stored in st.session_state.emissions_data
                alternatives = st.session_state.emissions_data
                
                st.info(f"""
                **Z-Score Interpretation:**
                - Your emission factor is {abs(z_score_emission):.2f} standard deviations {'above' if z_score_emission > 0 else 'below'} the average
                - Population mean: {emission_mean:.4f} kg CO2/km
                - Population std dev: {emission_std:.4f} kg CO2/km
                """)
                
                # Removed results_text as it's not being used directly
                
                recommendations = []
                
                # Using total_emissions_per_person (which is st.session_state.total_emissions)
                if total_kg > 100:
                    recommendations.append("Your carbon footprint from commuting is quite high. Consider switching to more sustainable transport options.")
                elif total_kg > 50:
                    recommendations.append("Your carbon footprint is moderate. There's room for improvement by considering more sustainable options.")
                else:
                    recommendations.append("Your carbon footprint is relatively low, but you can still make improvements.")
                
                if vehicle_type == "Private Transport" and people_count == 1:
                    recommendations.append("Consider carpooling to reduce emissions. Sharing your ride with 3 other people could reduce your emissions by up to 75%.")
                
                # NEW: Add recommendation for multiple vehicles
                if vehicle_type == "Combined Transport" and "Multiple Vehicles" in vehicle_name: # Check for the specific naming
                    recommendations.append("You are using multiple private vehicles in your commute, which can significantly increase your carbon footprint. Consider consolidating trips or switching to more eco-friendly options for some legs of your journey.")
                
                # Ensure fuel_type is available if needed for recommendations
                # For `vehicle_name_disp` if it contains fuel type.
                if "petrol" in vehicle_name_disp.lower() or "diesel" in vehicle_name_disp.lower():
                    recommendations.append("Consider switching to an electric vehicle to significantly reduce your carbon footprint.")
                
                # Compare with specific public transport options if higher emissions
                bus_emissions = (total_monthly_km_disp * emission_factors["public_transport"]["bus"]["electric"]) / (people_count_disp if people_count_disp > 0 else 1)
                metro_emissions = (total_monthly_km_disp * emission_factors["public_transport"]["metro"]) / (people_count_disp if people_count_disp > 0 else 1)
                
                if total_kg > 2 * bus_emissions:
                    recommendations.append(f"Using an electric bus could reduce your emissions by approximately {(total_kg - bus_emissions) / total_kg * 100:.1f}%.")
                
                if total_kg > 2 * metro_emissions:
                    recommendations.append(f"Using metro could reduce your emissions by approximately {(total_kg - metro_emissions) / total_kg * 100:.1f}%.")
                
                st.session_state.recommendations = recommendations # Store for PDF generation
                
                
                st.divider()
                st.header("Carbon Footprint Results")
                
                col1_cf, col2_cf = st.columns([1, 1])
                
                with col1_cf:
                    total_kg = st.session_state.total_emissions # Per person total emissions
                    
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
                    
                    sustainability_rating_index = int(min(2, total_kg/50)) if total_kg >= 0 else 0
                    st.markdown(f"<div style='color:{emissions_color}; font-size:18px;'><strong>Sustainability Rating:</strong> {['Low', 'Moderate', 'High'][sustainability_rating_index]}</div>", unsafe_allow_html=True)
                    
                    avg_emissions = 200 # Fixed average for comparison 
                    if avg_emissions > 0: 
                        if total_kg < avg_emissions:
                            st.success(f"Your emissions are {(1 - total_kg/avg_emissions) * 100:.1f}% lower than the average commuter.")
                        else:
                            st.warning(f"Your emissions are {(total_kg/avg_emissions - 1) * 100:.1f}% higher than the average commuter.")
                
                with col2_cf:
                    fig_gauge = go.Figure(go.Indicator(
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
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.subheader("Comparison with Alternative Transport Options")
                
                df_emissions = pd.DataFrame({
                    'Transport Mode': list(st.session_state.emissions_data.keys()),
                    'Monthly CO₂ Emissions (kg)': list(st.session_state.emissions_data.values())
                })
                
                df_emissions = df_emissions.sort_values('Monthly CO₂ Emissions (kg)')
                
                fig_bar_comparison = px.bar(
                    df_emissions, 
                    y='Transport Mode', 
                    x='Monthly CO₂ Emissions (kg)',
                    orientation='h',
                    color='Monthly CO₂ Emissions (kg)',
                    color_continuous_scale='RdYlGn_r'
                )
                
                fig_bar_comparison.update_layout(height=400, width=800)
                st.plotly_chart(fig_bar_comparison, use_container_width=True)
                
                st.header("Sustainability Recommendations")
                for i, rec in enumerate(st.session_state.recommendations):
                    st.markdown(f"**{i+1}. {rec}**")
            # --- END NEWLY ADDED CODE (Transport Section) ---

        else:
            st.info("No transport carbon footprint data available. Please complete the input on the '1. New Customer Assessment' page.")

        # Define placeholder variables for the new snippet to work
        # NOTE: These are approximations based on the available data in page_evaluation
        # and may not perfectly reflect the original intent of the provided snippet's context.
        temp_new_customer = {
            'Water': per_person_monthly_water if 'per_person_monthly_water' in locals() else 0,
            'Electricity': individual_equivalent_consumption if 'individual_equivalent_consumption' in locals() else 0,
            'MPCE': user_mpce_value if 'user_mpce_value' in locals() else 0, # Use user_mpce_value directly
            'Private_Transport': customer_data.get('Private_Monthly_Km', 0.0), # Use stored value
            'Public_Transport': customer_data.get('Public_Monthly_Km', 0.0), # Use stored value
            'Company Score': crisil_esg if 'crisil_esg' in locals() else 0
        }
        # Assuming z_vals and inverse_scoring are from a different context; defining minimally to avoid errors
        z_vals = {}
        inverse_scoring = False

        # Start of the code block provided by the user
        z_score = 0
        
        # Water component 
        water_z = (temp_new_customer['Water'] - 3900) / 3900
        z_score += water_z * 0.25  # 25% weight
        
        # Electricity component 
        electricity_z = max(0, (individual_equivalent_consumption - 300) / 300)  
        z_score += electricity_z * 0.25  # 25% weight
        
        # Private transport component 
        private_transport_z = temp_new_customer.get('Private_Transport', 0) / 1000  
        z_score += private_transport_z * 0.25  # 25% weight
        
        # Public transport component (bonus for usage, negative z for bonus)
        public_transport_z = -min(1, temp_new_customer.get('Public_Transport', 0) / 500) 
        z_score += public_transport_z * 0.15  # 15% weight
        
        company_z = -(temp_new_customer.get('Company Score', 0) / 100)  
        z_score += company_z * 0.10  # 10% weight
        
        sust_score = 500 * (1 - np.tanh(z_score/2.5))

        norm_vals = {}
        for feat, val in temp_new_customer.items():
            if feat in z_vals:  
                cmin, cmax = (0, 1000)  # Default range
                if cmax == cmin:
                    norm_vals[feat] = 1
                else:
                    if inverse_scoring:
                        norm_vals[feat] = ((cmax - val)/(cmax - cmin))*999 + 1
                    else:
                        norm_vals[feat] = ((val - cmin)/(cmax - cmin))*999 + 1

        
        st.markdown("---")
        st.markdown("### Customer Score Results")
        
        col1, col2 = st.columns(2)
        with col1:
            # Re-using existing variables from page_evaluation for these metrics
            # Note: water_units, num_people, usage_type are not available directly in this scope,
            # but per_person_monthly_water is.
            # I will use per_person_monthly_water and derive water_units for plotting.
            if 'per_person_monthly_water' in locals() and household_size > 0:
                monthly_water = per_person_monthly_water * household_size # This would be total water for the household
                per_person_monthly = per_person_monthly_water # Already per person
                
                average_monthly_per_person = 130 * 30 
                
                water_usage_z_score = (per_person_monthly - average_monthly_per_person) / 1000  
                if water_usage_z_score > 0:
                    st.warning(f"High water usage detected! Z-score: {water_usage_z_score:.2f} (Above average)")
                else:
                    st.success(f"Good water usage! Z-score: {water_usage_z_score:.2f} (Below or at average)")
                
                fig_water, ax_water = plt.subplots(figsize=(10, 6))
                categories = ['Your Usage', 'Average Usage']
                values = [per_person_monthly, average_monthly_per_person]
                colors = ['#ff6b6b' if per_person_monthly > average_monthly_per_person else '#51cf66', '#74c0fc']
                bars = ax_water.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                for bar, value_label in zip(bars, values):
                    height = bar.get_height()
                    ax_water.text(bar.get_x() + bar.get_width()/2., height + 50,
                                    f'{value_label:.0f}L', ha='center', va='bottom', fontweight='bold', fontsize=12)
                ax_water.set_ylabel('Water Consumption (Litres/Month/Person)', fontsize=12, fontweight='bold')
                ax_water.set_title('Your Water Usage vs Average Usage', fontsize=14, fontweight='bold', pad=20)
                ax_water.grid(axis='y', alpha=0.3, linestyle='--')
                ax_water.set_ylim(0, max(values) * 1.2 if values else 100) 
                if average_monthly_per_person > 0: 
                    percentage_diff = ((per_person_monthly - average_monthly_per_person) / average_monthly_per_person) * 100
                    if percentage_diff > 0:
                        ax_water.text(0.5, (max(values) * 1.1 if values else 50), f'{percentage_diff:.1f}% above average',
                                        ha='center', va='center', fontsize=11, fontweight='bold',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
                    else:
                        ax_water.text(0.5, (max(values) * 1.1 if values else 50), f'{abs(percentage_diff):.1f}% below average',
                                        ha='center', va='center', fontsize=11, fontweight='bold',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
                plt.tight_layout()
                st.pyplot(fig_water)

            if "individual_equivalent_consumption" in locals():
                st.metric("Personal Electricity Usage", f"{individual_equivalent_consumption:.2f} kWh")
            
            # This part for 'Company Score' seems to re-implement display logic already there.
            # I will attempt to re-use the `display_score` if it's available from the main ESG comparison.
            # If not, I'll use `crisil_esg`.
            display_score = crisil_esg # Default to input ESG score
            score_source = "Input"
            if 'esg_comparison_method' in st.session_state:
                compare_method = st.session_state['esg_comparison_method']
                if compare_method == "Select from list":
                    if 'selected_company_for_comparison' in st.session_state and st.session_state['selected_company_for_comparison'] != "(None)":
                        selected_company = st.session_state['selected_company_for_comparison']
                        if st.session_state.company_data is not None:
                            company_data_row_filtered = st.session_state.company_data[st.session_state.company_data['Company_Name'] == selected_company]
                            if not company_data_row_filtered.empty:
                                company_data_row = company_data_row_filtered.iloc[0]
                                display_score = company_data_row['Environment_Score']
                                score_source = f"from {selected_company}"
                else:  # Custom score method
                    if 'crisil_esg_score_input' in st.session_state:
                        display_score = st.session_state['crisil_esg_score_input']
                        score_source = "Custom Input"
            st.metric("Company Environment Score",f"{display_score:.2f}", help=f"Score {score_source}")

            # mpce_range_values and user_mpce_range_index are defined at the start of page_evaluation.
            st.metric("MPCE", f"{mpce_range_values[user_mpce_range_index][1]:.2f} ₹")
            st.metric("Water Usage", f"{temp_new_customer.get('Water', 0):.2f}") # Use .get for safety
            st.metric("Public Transport", f"{temp_new_customer.get('Public_Transport', 0):.2f}")
            st.metric("Private Transport", f"{temp_new_customer.get('Private_Transport', 0):.2f}")
            
            st.metric("Z-Score", f"{z_score:.2f}") # This z_score is calculated from the snippet.
            st.metric("Sustainability Score", f"{sust_score:.2f}") # This sust_score is calculated from the snippet.

            # Compute rank: count how many existing scores exceed this one, then +1
            if 'scored_data' in st.session_state and 'Sustainability_Score' in st.session_state.scored_data:
                existing = st.session_state.scored_data["Sustainability_Score"]
                sust_rank = (existing > sust_score).sum() + 1
            else:
                sust_rank = 1  # Default rank if no comparison data available

            st.metric("Sustainability Rank", f"{sust_rank}")


            if 'scored_data' in st.session_state and 'Sustainability_Score' in st.session_state.scored_data.columns:
                existing_sust = st.session_state.scored_data['Sustainability_Score']
                better_than = (existing_sust < sust_score).mean() * 100
                st.success(f"This customer performs better than **{better_than:.1f}%** of customers in the dataset (based on Z-Score)")
            else:
                st.info("No comparison data available in the dataset.")

            z_description = "above" if z_score > 0 else "below"
            st.info(f"Performance: **{abs(z_score):.2f} SD {z_description} mean**")
        
        with col2:
            # --- FIX 3: Pie chart components ---
            # Construct the features and their values for the pie chart
            pie_chart_data = {}

            if temp_new_customer['Electricity'] > 0:
                pie_chart_data['Personal Electricity Usage (kWh)'] = temp_new_customer['Electricity']
            if temp_new_customer['Water'] > 0:
                pie_chart_data['Water Usage (L/person/month)'] = temp_new_customer['Water'] # Assuming this is per person value
            if temp_new_customer['Company Score'] > 0:
                pie_chart_data['Company Score'] = temp_new_customer['Company Score']

            # Add transport components only if they are greater than 0
            if temp_new_customer['Private_Transport'] > 0:
                pie_chart_data['Private Transport (Km/month)'] = temp_new_customer['Private_Transport']
            if temp_new_customer['Public_Transport'] > 0:
                pie_chart_data['Public Transport (Km/month)'] = temp_new_customer['Public_Transport']
            
            # Filter out entries with 0 value for the pie chart for cleaner visualization
            active_pie_features = {k: v for k, v in pie_chart_data.items() if v > 0}

            if active_pie_features:  
                total_sum_for_pie = sum(active_pie_features.values())
                if total_sum_for_pie == 0: # Avoid division by zero if all values are 0
                    st.info("No active features with positive values to display for pie chart.")
                    # Removed 'return' to allow remaining code in col2 to execute if any
                else:
                    pie_values = list(active_pie_features.values())
                    pie_labels = list(active_pie_features.keys())

                    explode = [0.1 if i % 2 == 0 else 0 for i in range(len(pie_labels))] # Keep explode if desired
                    
                    fig_pie_weights, ax_pie_weights = plt.subplots(figsize=(6,6))
                    ax_pie_weights.pie(
                        pie_values, # Use actual values for distribution
                        labels=pie_labels,
                        autopct='%1.1f%%',
                        startangle=90,
                        explode=explode
                    )
                    ax_pie_weights.set_title('Contribution of Inputs to Customer Profile') # More descriptive title
                    st.pyplot(fig_pie_weights)
                    
                    for feat, value in active_pie_features.items():
                        percentage_contribution = (value / total_sum_for_pie) * 100
                        st.info(f"{feat} contributes {percentage_contribution:.1f}% (Value: {value:.2f})")
            else:
                st.info("No active features with positive values to display for pie chart.")
        # End of the code block provided by the user


        # ESG Comparison Visualizations (moved here as per request 2)
        company_data = st.session_state.company_data
        if company_data is not None:
            st.subheader("ESG Comparison Benchmarking")
            esg_analysis_type = st.session_state.get('esg_analysis_type', "Employee Range Analysis")
            selected_esg_industry = st.session_state.get('selected_esg_industry')
            selected_esg_emp_size = st.session_state.get('selected_esg_emp_size')
            esg_comparison_method = st.session_state.get('esg_comparison_method', "Select from list")
            selected_company_for_comparison = st.session_state.get('selected_company_for_comparison')
            custom_esg_score_for_comparison = st.session_state.get('custom_esg_score_for_comparison')

            if esg_analysis_type == "Employee Range Analysis":
                if selected_esg_industry and selected_esg_emp_size:
                    df_sector = company_data[company_data['Sector_classification'] == selected_esg_industry]

                    if selected_esg_emp_size == "Small (<5,000)":
                        df_filtered = df_sector[df_sector['Total_Employees'] < 5000]
                        emp_range_text = "less than 5,000"
                    elif selected_esg_emp_size == "Medium (5,000 to 15,000)":
                        df_filtered = df_sector[(df_sector['Total_Employees'] >= 5000) & (df_sector['Total_Employees'] <= 15000)]
                        emp_range_text = "5,000 to 15,000"
                    else:  # Large (>15,000)
                             df_filtered = df_sector[df_sector['Total_Employees'] > 15000]
                             emp_range_text = "more than 15,000"
                    if len(df_filtered) > 0:
                        baseline_mean = df_filtered['Environment_Score'].mean()
                        baseline_std = df_filtered['Environment_Score'].std(ddof=1)

                        st.markdown(f"### Companies in {selected_esg_industry} sector with {emp_range_text} employees")
                        st.markdown(f"**Baseline Environment Score:** {baseline_mean:.2f} (std: {baseline_std:.2f})")

                        df_results = df_filtered.copy()
                        df_results['Env_Z_Score'] = (df_results['Environment_Score'] - baseline_mean) / baseline_std

                        min_z = df_results['Env_Z_Score'].min()
                        max_z = df_results['Env_Z_Score'].max()
                        df_results['Normalized_Score'] = ((df_results['Env_Z_Score'] - min_z) / (max_z - min_z)) * 100 if (max_z - min_z) != 0 else 50 # Handle division by zero

                        st.dataframe(
                            df_results[[
                                'Company_Name',
                                'Total_Employees',
                                'Environment_Score',
                                'Env_Z_Score',
                                'Normalized_Score'
                            ]].sort_values('Environment_Score', ascending=False).reset_index(drop=True)
                        )

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(df_results['Environment_Score'], kde=True, ax=ax)
                        ax.axvline(baseline_mean, color='red', linestyle='--', label=f'Mean ({baseline_mean:.2f})')
                        ax.axvline(baseline_mean + baseline_std, color='green', linestyle=':', label=f'+1 Std Dev ({baseline_mean + baseline_std:.2f})')
                        ax.axvline(baseline_mean - baseline_std, color='orange', linestyle=':', label=f'-1 Std Dev ({baseline_mean - baseline_std:.2f})')
                        ax.set_xlabel('Environment Score')
                        ax.set_title(f'Distribution of Environment Scores for {selected_esg_industry} ({emp_range_text} employees)')
                        ax.legend()
                        st.pyplot(fig)

                        st.markdown("#### Your Company's Comparison")
                        if esg_comparison_method == "Select from list":
                            if selected_company_for_comparison and selected_company_for_comparison != "(None)":
                                company_row_filtered = df_results[df_results['Company_Name'].str.lower() == selected_company_for_comparison.lower()]
                                if not company_row_filtered.empty:
                                    company_data_row = company_row_filtered.iloc[0]
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
                                    st.success(f"**{selected_company_for_comparison}** performs better than **{better_than:.1f}%** of companies in this segment (based on Z-Score)")
                                else:
                                    st.warning(f"Selected company '{selected_company_for_comparison}' not found in the filtered dataset (Industry: {selected_esg_industry}, Employee Size: {selected_esg_emp_size}). Please ensure the company name matches exactly or check for case sensitivity.")
                            
                        else: # Custom score method
                            custom_score = custom_esg_score_for_comparison
                            custom_z = (custom_score - baseline_mean) / baseline_std
                            custom_norm = ((custom_z - min_z) / (max_z - min_z)) * 100 if (max_z - min_z) != 0 else 0  # Fallback to 0 for normalization

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Environment Score", f"{custom_score:.2f}")
                            with col2:
                                st.metric("Z-Score", f"{custom_z:.2f}",
                                        f"{custom_z:.2f} SD from mean")
                            with col3:
                                st.metric("Normalized Score", f"{custom_norm:.2f}/100")

                            better_than = (df_results['Env_Z_Score'] < custom_z).mean() * 100
                            st.success(f"Your company performs better than **{better_than:.1f}%** of companies in this segment (based on Z-Score)")
                    else:
                        st.warning(f"No companies found in {selected_esg_industry} sector with {emp_range_text} employees to show comparison.")

                else:
                    st.info("Please select an Industry Sector and Employee Size Category on page 1 to see ESG comparison data.")

            elif esg_analysis_type == "Company-Only Analysis":
                overall_mean = company_data['Environment_Score'].mean()
                overall_std = company_data['Environment_Score'].std(ddof=1)

                st.markdown("### Compare Against Entire Dataset")
                st.markdown(f"**Overall Environment Score Baseline:** {overall_mean:.2f} (std: {overall_std:.2f})")

                if esg_comparison_method == "Select from list":
                    if selected_company_for_comparison and selected_company_for_comparison != "(None)":
                        company_row_filtered = company_data[company_data['Company_Name'] == selected_company_for_comparison]
                        if not company_row_filtered.empty:
                            company_row = company_row_filtered.iloc[0]
                            company_score = company_row.get('Environment_Score','N/A')
                            company_sector = company_row.get('Sector_classification', 'N/A')
                            company_employees = company_row.get('Total_Employees', 'N/A')

                            company_z = (company_score - overall_mean) / overall_std
                            percentile = (company_data['Environment_Score'] < company_score).mean() * 100
                            st.markdown(f"### {selected_company_for_comparison}")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sector", f"{company_sector}")
                            with col2:
                                st.metric("Employees", f"{company_employees}")
                            with col3:
                                st.metric("Environment Score", f"{company_score:.2f}")

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

                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(company_data['Environment_Score'], kde=True, ax=ax)
                            ax.axvline(overall_mean, color='red', linestyle='--', label=f'Mean ({overall_mean:.2f})')
                            ax.axvline(company_score, color='blue', linestyle='-',
                                    label=f'{selected_company_for_comparison} ({company_score:.2f}, Z={company_z:.2f})')
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

                            selected_idx = display_df.index[display_df['Company_Name'] == selected_company_for_comparison].tolist()

                            styled_df = display_df.style.apply(
                                lambda x: ['background-color: lightskyblue' if i in selected_idx else ''
                                        for i in range(len(display_df))],
                                axis=0
                            )
                            st.dataframe(styled_df)
                        else:
                            st.warning(f"Selected company '{selected_company_for_comparison}' not found in the dataset. Please adjust your selection.")
                    else:
                        st.info("Please select a company on page 1 to see its comparison here.")
                else: # Custom score comparison
                    custom_score = custom_esg_score_for_comparison
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

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(company_data['Environment_Score'], kde=True, ax=ax)
                    ax.axvline(overall_mean, color='red', linestyle='--', label=f'Mean ({overall_mean:.2f})')
                    ax.axvline(custom_score, color='blue', linestyle='-',
                            label=f'Your Company ({custom_score:.2f}, Z={custom_z:.2f})')
                    ax.set_xlabel('Environment Score')
                    ax.set_title(f'Distribution of Environment Scores Across All Companies')
                    ax.legend()
                    st.pyplot(fig)

                    selected_sector_for_comparison = st.selectbox("Compare with Sector (Optional)",
                                            ["(None)"] + sorted([str(x) for x in company_data['Sector_classification'].dropna().unique()]),
                                            key="company_only_sector_eval")

                    if selected_sector_for_comparison != "(None)":
                        sector_data = company_data[company_data['Sector_classification'] == selected_sector_for_comparison]
                        sector_mean = sector_data['Environment_Score'].mean()
                        sector_std = sector_data['Environment_Score'].std(ddof=1)
                        sector_z = (custom_score - sector_mean) / sector_std
                        sector_percentile = (sector_data['Environment_Score'] < custom_score).mean() * 100

                        st.markdown(f"### Comparison with {selected_sector_for_comparison} Sector")
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

                        st.markdown(f"### All Performers in {selected_sector_for_comparison} (Highest to Lowest)")

                        display_df = all_sector_companies[[
                            'Company_Name',
                            'Environment_Score',
                            'Sector_Z_Score',
                            'ESG_Rating',
                            'Total_Employees'
                        ]].reset_index(drop=True)
                        st.dataframe(display_df)
        else:
            st.info("Company data unavailable for ESG comparison visualizations.")

        st.subheader("Generate Sustainability Report (PDF)")
        # PDF generation section
        has_required_data = (
            ('single_customer_inputs' in st.session_state and st.session_state.single_customer_inputs) and
            ('transport_carbon_results' in st.session_state and st.session_state.transport_carbon_results)
        )
        if st.button("Generate PDF Report", key="gen_pdf_eval"):
            if has_required_data:
                with st.spinner("Generating your sustainability report..."):
                    try:
                        scored_df = st.session_state.get('scored_data', pd.DataFrame())
                        if scored_df is None or scored_df.empty:
                            # Create a DataFrame for the current customer for PDF if no bulk data
                            current_customer_df = pd.DataFrame([st.session_state.single_customer_inputs])
                            current_customer_df['Sustainability_Score'] = st.session_state.calculated_single_customer_score
                            current_customer_df['Electricity'] = st.session_state.single_customer_inputs['Electricity']
                            current_customer_df['Water'] = st.session_state.single_customer_inputs['Water_Monthly_Total'] # Use total water for PDF
                            current_customer_df['Crisil_ESG_Score'] = st.session_state.single_customer_inputs['Crisil_ESG_Score']

                            scored_df = current_customer_df # Use this as the basis for the PDF

                        pdf_file = generate_pdf(
                            scored_df,
                            pd.DataFrame([st.session_state.single_customer_inputs]),
                            st.session_state.recommendations, # Use the recommendations from session state
                            st.session_state.transport_carbon_results # Use the full transport results dict
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
                st.warning("Please complete the 'New Customer Assessment' on page 1 before generating a PDF report.")

    else: # Bulk CSV Upload
        st.subheader("Bulk Upload Evaluation Results")
        st.markdown("Results from your uploaded CSV/Excel file.")

        if 'uploaded_bulk_df' not in st.session_state or st.session_state.uploaded_bulk_df.empty:
            st.info("Please go to '1. New Customer Assessment' page and upload a CSV/Excel file in 'CSV Upload' mode first.")
            return

        test_df = st.session_state.uploaded_bulk_df
        st.markdown("#### Uploaded Test Data Preview (from Page 1)")
        test_df = test_df.astype(object)
        test_df = test_df.where(pd.notna(test_df), None)
        st.dataframe(test_df)

        if st.button("Process & Display Bulk Results", key="process_batch_now_page3"):
            with st.spinner("Processing batch data..."):
                # Ensure emission_factors are accessible
                emission_factors_transport = emission_factors

                test_with_scores = []
                mpce_ranges = ["₹1-1,000", "₹1,000-5,000", "₹5,000-10,000", "₹10,000-25,000", "₹25,000+"]
                mpce_range_values = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 25000), (25000, float('inf'))]

                for idx, row in test_df.iterrows():
                    try:
                        # Extract data from row, handle missing values
                        electricity_kwh = float(row.get("Electricity", 0.0)) if row.get("Electricity") is not None else 0.0
                        household_count = int(row.get("People in the household", 1)) if not pd.isna(row.get("People in the household")) else 1
                        water_input_type = str(row.get("Water (Monthly or Daily)", "monthly")).strip().lower()
                        water_value = float(row.get("Water (Monthly or Daily) Value", 0.0)) if not pd.isna(row.get("Water (Monthly or Daily) Value")) else 0.0
                        
                        if water_input_type == "daily":
                            total_water_liters = water_value * 30 * household_count
                        else:
                            total_water_liters = water_value * household_count

                        state_name_raw = str(row.get("State/UT", "")).strip().lower()
                        rural_urban_raw = str(row.get("Rural/Urban", "rural")).strip().lower()
                        user_sector = 1 if rural_urban_raw == 'rural' else 2

                        # Clean state name using the global mapping from load_real_electricity_data
                        state_name_mapping_inverse = {
                            "Delhi": "Delhi (NCT)",
                            "Andaman and Nicobar Islands": "Andaman & Nicobar",
                            "Jammu and Kashmir": "Jammu & Kashmir",
                            "Dadra and Nagar Haveli and Daman and Diu": "Dadra & Nagar Haveli and Daman and Diu",
                            "Puducherry": "Puducherry (UT)"
                        }
                        cleaned_state_name = state_name_raw
                        for k, v in state_name_mapping_inverse.items():
                            if state_name_raw == v: # Check if raw name is in the mapped values
                                cleaned_state_name = k
                                break
                        if cleaned_state_name not in INDIAN_STATES_UTS: # If still not in, use it as is but warn
                            # st.warning(f"State '{state_name_raw}' not recognized, using as is. Ensure it matches `INDIAN_STATES_UTS` or its mapped value.") # Comment out for bulk to avoid too many warnings
                            pass


                        # MPCE value - use the direct value if available, else derive from range
                        mpce_val = float(row.get("MPCE Value", 0.0)) if not pd.isna(row.get("MPCE Value")) else 0.0
                        # FIX: Use "MPCE" which is the correct column name from the screenshot, not "MPCE Range"
                        mpce_range_str = str(row.get("MPCE", "₹1-1,000")).strip()
                        if mpce_val == 0.0: # If MPCE Value column is missing/empty, try to infer from range
                            mpce_index = mpce_ranges.index(mpce_range_str) if mpce_range_str in mpce_ranges else 0
                            mpce_val = mpce_range_values[mpce_index][1] # Use upper bound of selected range for calculation

                        vehicle_type_csv = str(row.get("Vehicle (Drop Down)", row.get("Vehicle", ""))).strip()
                        engine_fuel_csv = str(row.get("Engine (Drop Down)", "")).strip()
                        km_per_month_csv = float(row.get("Km_per_month", 0.0)) if not pd.isna(row.get("Km_per_month")) else 0.0
                        crisil_esg_csv = float(row.get("Crisil_ESG_Score", 0.0)) if not pd.isna(row.get("Crisil_ESG_Score")) else 0.0
                        company_sector_csv = str(row.get("Industry (Sector_classification)", "")) if not pd.isna(row.get("Industry (Sector_classification)")) else ""
                        num_employees_csv = int(row.get("Number of Employees", None)) if not pd.isna(row.get("Number of Employees")) else None

                        # Transport emission calculation (replicate original logic)
                        commute_emission_calc = 0.0
                        chosen_ef = 0.0 # Initialize chosen_ef
                        if vehicle_type_csv.lower() in ["two_wheeler", "2 wheeler", "three_wheeler", "four_wheeler", "public_transport"]:
                            if vehicle_type_csv.lower() == "two_wheeler" or vehicle_type_csv.lower() == "2 wheeler":
                                category_2w = row.get("Two_Wheeler_Category", "Scooter") # Assume column for 2W category in CSV
                                if category_2w in emission_factors_transport["two_wheeler"] and engine_fuel_csv in emission_factors_transport["two_wheeler"][category_2w]:
                                    ef_min = emission_factors_transport["two_wheeler"][category_2w][engine_fuel_csv]["min"]
                                    ef_max = emission_factors_transport["two_wheeler"][category_2w][engine_fuel_csv]["max"]
                                    chosen_ef = (ef_min + ef_max) / 2
                            elif vehicle_type_csv.lower() == "three_wheeler":
                                if engine_fuel_csv in emission_factors_transport["three_wheeler"]:
                                    ef_min = emission_factors_transport["three_wheeler"][engine_fuel_csv]["min"]
                                    ef_max = emission_factors_transport["three_wheeler"][engine_fuel_csv]["max"]
                                    chosen_ef = (ef_min + ef_max) / 2
                            elif vehicle_type_csv.lower() == "four_wheeler":
                                car_type_4w = row.get("Four_Wheeler_Type", "sedan") # For simplicity in bulk
                                if car_type_4w in emission_factors_transport["four_wheeler"] and engine_fuel_csv in emission_factors_transport["four_wheeler"][car_type_4w]:
                                    base = emission_factors_transport["four_wheeler"][car_type_4w][engine_fuel_csv]["base"]
                                    uplift = emission_factors_transport["four_wheeler"][car_type_4w][engine_fuel_csv]["uplift"]
                                    chosen_ef = base * uplift
                            
                            # FIX: Corrected logic for public transport to handle 'metro' correctly
                            elif vehicle_type_csv.lower() == "public_transport":
                                public_mode_csv = row.get("Public_Transport_Mode", "metro") # Default to metro for simplicity
                                
                                if public_mode_csv == "taxi":
                                    taxi_type_csv = row.get("Taxi_Car_Type", "sedan")
                                    taxi_fuel_csv = row.get("Taxi_Fuel_Type", "petrol")
                                    if taxi_type_csv in emission_factors_transport["public_transport"]["taxi"] and taxi_fuel_csv in emission_factors_transport["public_transport"]["taxi"][taxi_type_csv]:
                                        base = emission_factors_transport["public_transport"]["taxi"][taxi_type_csv][taxi_fuel_csv]["base"]
                                        uplift = emission_factors_transport["public_transport"]["taxi"][taxi_type_csv][taxi_fuel_csv]["uplift"]
                                        chosen_ef = base * uplift
                                    else:
                                        chosen_ef = 0.05 # Fallback
                                elif public_mode_csv == "metro":
                                    # Handle metro directly as it has a float value, not a dict
                                    chosen_ef = emission_factors_transport["public_transport"]["metro"]
                                elif public_mode_csv == "bus":
                                    # Handle bus, which has a dict of fuel types
                                    bus_details = emission_factors_transport["public_transport"]["bus"]
                                    # Use .get() on the dictionary, with a fallback
                                    chosen_ef = bus_details.get(engine_fuel_csv, bus_details["diesel"])
                                else:
                                    # Fallback for any other public transport mode
                                    chosen_ef = emission_factors_transport["public_transport"]["base_emission"]
                        
                        commute_emission_calc = chosen_ef

                        # --- Z-Score and Final Score Calculation (Replicated as is) ---
                        # Electricity Z-score (Location based)
                        feature1_mean_batch = 0
                        feature1_std_batch = 0
                        state_sector_data_batch = None
                        if st.session_state.electricity_data is not None:
                             state_sector_data_batch = st.session_state.electricity_data[
                                (st.session_state.electricity_data['state_name'] == cleaned_state_name) &
                                (st.session_state.electricity_data['sector'] == user_sector)
                            ]
                        if state_sector_data_batch is not None and not state_sector_data_batch.empty:
                            feature1_mean_batch = state_sector_data_batch['qty_usage_in_1month'].mean()
                            feature1_std_batch = state_sector_data_batch['qty_usage_in_1month'].std()
                        else: # Fallback to overall mean if specific state/sector data not found
                            if st.session_state.feature_stats and 'Electricity' in st.session_state.feature_stats:
                                feature1_mean_batch = st.session_state.feature_stats['Electricity']['mean']
                                feature1_std_batch = st.session_state.feature_stats['Electricity']['std']

                        feature1_z_score_batch = (electricity_kwh - feature1_mean_batch) / feature1_std_batch if feature1_std_batch > 0 else 0
                        feature1_z_score_batch = np.clip(feature1_z_score_batch, -3, 3) # Cap between -3 and 3

                        # Electricity Z-score (MPCE based)
                        feature2_mean_batch = 0
                        feature2_std_batch = 0
                        current_mpce_range_batch = next((r for r_name, r in zip(mpce_ranges, mpce_range_values) if r_name == mpce_range_str), (0, float('inf')))

                        if st.session_state.electricity_data is not None:
                            range_data_batch = st.session_state.electricity_data[(st.session_state.electricity_data['mpce'] >= current_mpce_range_batch[0]) & (st.session_state.electricity_data['mpce'] < current_mpce_range_batch[1])]
                            if not range_data_batch.empty:
                                feature2_mean_batch = range_data_batch['qty_usage_in_1month'].mean()
                                feature2_std_batch = range_data_batch['qty_usage_in_1month'].std()
                            else:
                                if st.session_state.feature_stats and 'Electricity' in st.session_state.feature_stats:
                                    feature2_mean_batch = st.session_state.feature_stats['Electricity']['mean']
                                    feature2_std_batch = st.session_state.feature_stats['Electricity']['std']

                        feature2_z_score_batch = (electricity_kwh - feature2_mean_batch) / feature2_std_batch if feature2_std_batch > 0 else 0
                        feature2_z_score_batch = np.clip(feature2_z_score_batch, -3, 3)


                        # Water Z-score
                        average_monthly_per_person_water_batch = 130 * 30 # Baseline: 130 liters/day * 30 days
                        water_z_score_batch = (total_water_liters / (household_count if household_count > 0 else 1) - average_monthly_per_person_water_batch) / average_monthly_per_person_water_batch if average_monthly_per_person_water_batch > 0 else 0
                        water_z_score_batch = np.clip(water_z_score_batch, -3, 3)

                        # Transport Z-score
                        # Ensure monthly_co2_emissions_batch is calculated correctly based on km_per_month_csv * commute_emission_calc
                        monthly_co2_emissions_batch = (commute_emission_calc * km_per_month_csv) / (1 if km_per_month_csv > 0 else 1) # assuming commute_emission_calc is per km per person, so no /people_count
                        avg_commute_emissions_batch = 50 # Example baseline (from original)
                        commute_z_score_batch = (monthly_co2_emissions_batch - avg_commute_emissions_batch) / avg_commute_emissions_batch if avg_commute_emissions_batch > 0 else 0
                        commute_z_score_batch = np.clip(commute_z_score_batch, -3, 3)


                        # Company ESG Z-score
                        crisil_z_score_batch = 0
                        if st.session_state.company_data is not None and not st.session_state.company_data.empty and 'Environment_Score' in st.session_state.company_data.columns:
                            comp_mean_batch = st.session_state.company_data['Environment_Score'].mean()
                            comp_std_batch = st.session_state.company_data['Environment_Score'].std()
                            crisil_z_score_batch = -(crisil_esg_csv - comp_mean_batch) / comp_std_batch if comp_std_batch > 0 else 0
                        else:
                            crisil_z_score_batch = -(crisil_esg_csv - 70) / 15 # Fixed fallback
                        crisil_z_score_batch = np.clip(crisil_z_score_batch, -3, 3)


                        # Calculate combined Z-score based on weights
                        weights_bulk = st.session_state.weights # Use the weights from session state
                        total_z_score_batch = (
                            crisil_z_score_batch * weights_bulk["Company"] +
                            water_z_score_batch * weights_bulk["Water"] +
                            ((feature1_z_score_batch * weights_bulk["Electricity_State"]) + (feature2_z_score_batch * weights_bulk["Electricity_MPCE"])) +
                            (commute_z_score_batch * (weights_bulk["Public_Transport"] + weights_bulk["Private_Transport"]))
                        )

                        # Calculate Sustainability Score
                        sustainability_score_batch = 500 * (1 - np.tanh(total_z_score_batch / 2.5))


                        result = row.to_dict()
                        result.update({
                            "Electricity_Z": feature1_z_score_batch, # Using location based Z-score for simplicity in bulk display
                            "Water_Z": water_z_score_batch,
                            "Commute_Z": commute_z_score_batch,
                            "Company_Z": crisil_z_score_batch,
                            "Z_Total": total_z_score_batch,
                            "Sustainability_Score": sustainability_score_batch
                        })
                        test_with_scores.append(result)

                    except Exception as e:
                        st.warning(f"Error processing row {idx + 1}: {str(e)}. Skipping this row.")
                        continue

                if not test_with_scores:
                    st.error("No valid data could be processed from the uploaded file. Please check your file format and data content.")
                    st.stop()

                results_df = pd.DataFrame(test_with_scores)
                st.session_state.scored_data = results_df # Store processed bulk data

                st.markdown("### Bulk Test Results")
                st.dataframe(results_df)

                st.markdown("### Comprehensive Metrics for Bulk Data")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    avg_score = results_df['Sustainability_Score'].mean()
                    st.metric("Average Score", f"{avg_score:.1f}")

                with col2:
                    max_score = results_df['Sustainability_Score'].max()
                    st.metric("Highest Score", f"{max_score:.1f}")

                with col3:
                    min_score = results_df['Sustainability_Score'].min()
                    st.metric("Lowest Score", f"{min_score:.1f}")

                with col4:
                    total_customers = len(results_df)
                    st.metric("Total Customers", total_customers)

                # Additional metrics from original code
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    median_score = results_df['Sustainability_Score'].median()
                    st.metric("Median Score", f"{median_score:.1f}")
                with col6:
                    std_score = results_df['Sustainability_Score'].std()
                    st.metric("Standard Deviation", f"{std_score:.1f}")
                with col7:
                    excellent_count = len(results_df[results_df['Sustainability_Score'] >= 400])
                    excellent_pct = (excellent_count / total_customers * 100) if total_customers > 0 else 0
                    st.metric("Excellent Scores", f"{excellent_count} ({excellent_pct:.1f}%)")
                with col8:
                    poor_count = len(results_df[results_df['Sustainability_Score'] < 200])
                    poor_pct = (poor_count / total_customers * 100) if total_customers > 0 else 0
                    st.metric("Poor Scores", f"{poor_count} ({poor_pct:.1f}%)")

                # More metrics for bulk analysis
                col9, col10, col11, col12 = st.columns(4)
                with col9:
                    avg_electricity = results_df['Electricity'].mean()
                    st.metric("Avg Electricity (kWh)", f"{avg_electricity:.1f}")
                with col10:
                    if 'Number of Employees' in results_df.columns:
                        avg_employees = results_df['Number of Employees'].mean()
                        st.metric("Avg Employees", f"{avg_employees:.0f}")
                    else:
                        st.info("No 'Number of Employees' data in bulk upload.")
                with col11:
                    avg_esg = results_df['Crisil_ESG_Score'].mean()
                    st.metric("Avg ESG Score", f"{avg_esg:.1f}")
                with col12:
                    if 'Km_per_month' in results_df.columns:
                        avg_km = results_df['Km_per_month'].mean()
                        st.metric("Avg Km/Month", f"{avg_km:.0f}")
                    else:
                        st.info("No 'Km_per_month' data in bulk upload.")

                col13, col14, col15, col16 = st.columns(4)
                with col13:
                    if 'Engine (Drop Down)' in results_df.columns:
                        electric_vehicles = len(results_df[results_df['Engine (Drop Down)'] == 'electric'])
                        electric_pct = (electric_vehicles / total_customers * 100) if total_customers > 0 else 0
                        st.metric("Electric Vehicles", f"{electric_vehicles} ({electric_pct:.1f}%)")
                    else:
                        st.info("No 'Engine (Drop Down)' data in bulk upload.")
                with col14:
                    if 'Rural/Urban' in results_df.columns:
                        urban_customers = len(results_df[results_df['Rural/Urban'] == 'urban'])
                        urban_pct = (urban_customers / total_customers * 100) if total_customers > 0 else 0
                        st.metric("Urban Customers", f"{urban_customers} ({urban_pct:.1f}%)")
                    else:
                        st.info("No 'Rural/Urban' data in bulk upload.")
                with col15:
                    high_esg = len(results_df[results_df['Crisil_ESG_Score'] >= 80])
                    high_esg_pct = (high_esg / total_customers * 100) if total_customers > 0 else 0
                    st.metric("High ESG (≥80)", f"{high_esg} ({high_esg_pct:.1f}%)")
                with col16:
                    if 'Vehicle (Drop Down)' in results_df.columns:
                        public_transport_users = len(results_df[results_df['Vehicle (Drop Down)'] == 'public_transport'])
                        public_pct = (public_transport_users / total_customers * 100) if total_customers > 0 else 0
                        st.metric("Public Transport", f"{public_transport_users} ({public_pct:.1f}%)")
                    else:
                        st.info("No 'Vehicle (Drop Down)' data in bulk upload.")

                st.markdown("### Z-Score Analytics (Bulk)")
                z_col1, z_col2, z_col3, z_col4 = st.columns(4)
                with z_col1:
                    avg_elec_z = results_df['Electricity_Z'].mean()
                    st.metric("Avg Electricity Z", f"{avg_elec_z:.3f}")
                with z_col2:
                    avg_water_z = results_df['Water_Z'].mean()
                    st.metric("Avg Water Z", f"{avg_water_z:.3f}")
                with z_col3:
                    avg_commute_z = results_df['Commute_Z'].mean()
                    st.metric("Avg Commute Z", f"{avg_commute_z:.3f}")
                with z_col4:
                    avg_company_z = results_df['Company_Z'].mean()
                    st.metric("Avg Company Z", f"{avg_company_z:.3f}")
        else:
            st.info("Click 'Process & Display Bulk Results' to view the evaluation for the uploaded data.")


# --- Main Application Logic ---
def main():
    st.sidebar.title("Navigation")
    # Using a session state variable to keep track of the current page, and a default.
    # This helps ensure that if a form submission leads to a 'success' message, the user
    # stays on the same page or can navigate directly to the results.
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "1. New Customer Assessment"

    page_options = [
        "1. New Customer Assessment",
        "2. Company & Electricity Data Insights",
        "3. Weighted Score Settings",
        "4. Evaluation Results & Reporting"
    ]
    page_selection = st.sidebar.radio("Go to", page_options, index=page_options.index(st.session_state.current_page))

    # Update session state if sidebar radio changes
    if page_selection != st.session_state.current_page:
        st.session_state.current_page = page_selection

    st.title("Resource Sustainability Dashboard")
    st.markdown("Analyze your resource consumption and sustainability score")
    st.markdown("---")

    if st.session_state.current_page == "1. New Customer Assessment":
        page_test_customer()
    elif st.session_state.current_page == "2. Company & Electricity Data Insights":
        page_data_comparison()
    elif st.session_state.current_page == "3. Weighted Score Settings":
        page_evaluation()
    elif st.session_state.current_page == "4. Evaluation Results & Reporting":
        page_reporting()

if __name__ == "__main__":
    main()
