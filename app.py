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

# --- Global Configurations and Data Loading Functions ---

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
    table { width: 100%; border-collapse: collapse; }\
    table, th, td { padding: 8px; text-align: center; }
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
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadeep", "Puducherry"
]

@st.cache_data
def load_company_data():
    """
    Preload company data from a fixed CSV path and apply sector classification mapping.
    """
    try:
        # Try different encodings to handle special characters
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv("employees_2024_25_updated.csv", encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            st.error("Could not read the CSV file with any of the supported encodings.")
            return None
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

    # Define the mapping for sector classifications based on user's 24 rules
    sector_mapping = {
        # 1. Agri Commodities
        'Agri Commodities': 'Agri Commodities',
        'Agri Commodity': 'Agri Commodities',

        # 2. Auto Parts
        'Auto Components & Equipments': 'Auto Parts',
        'Auto Dealer': 'Auto Parts',
        'Auto OEM': 'Auto Parts',
        'Auto components and equipments (Batteries)': 'Auto Parts',

        # 3. Bank
        'Bank': 'Bank',
        'Banks': 'Bank',

        # 4. Cement
        'Cement': 'Cement',
        'Cement & Cement Products (Diversified)': 'Cement',

        # 5. Chemicals
        'Chemical (Carbon Black)': 'Chemicals',
        'Chemicals (Carbon Black)': 'Chemicals',
        'Chemicals (Dyes & Pigments)': 'Chemicals',
        'Chemicals (Dyes and Pigments)': 'Chemicals',
        'Chemicals (Explosives)': 'Chemicals',
        'Chemicals (Petrochemicals)': 'Chemicals',
        'Chemicals - Fertilizers': 'Chemicals',

        'Civil Construction': 'Civil Construction', 


        # 6. Compressors, Pumps & Diesel Engines
        'Compressors, Pumps & Diesel Engine': 'Compressors, Pumps & Diesel Engines',
        'Compressors, Pumps & Diesel Engines': 'Compressors, Pumps & Diesel Engines',

        # 7. Consumer Durables
        'Consumer Durables - Consumer Electronics': 'Consumer Durables',
        'Consumer Durables - Footwear': 'Consumer Durables',
        'Consumer Durables - Furniture And Home Furnishing': 'Consumer Durables',
        'Consumer Durables - Gems, Jewellery And Watches': 'Consumer Durables',
        'Consumer Durables - Gems, Jewellery and Watches': 'Consumer Durables',
        'Consumer Durables - gems, jewellery and watches': 'Consumer Durables',
        'Consumer Durables - Glassware': 'Consumer Durables',
        'Consumer Durables - Household Appliances': 'Consumer Durables',
        'Consumer Durables - Houseware': 'Consumer Durables',
        'Consumer Durables - Leather And Leather Products': 'Consumer Durables',
        'Consumer Durables - Plastic Products': 'Consumer Durables',
        'Consumer Durables - PlasticProducts': 'Consumer Durables',
        # Added to consolidate related terms under Consumer Durables as per user's implied request to fix repetitions
        'Consumer Electronics': 'Consumer Durables',
        'Household appliances': 'Consumer Durables',
        'Household products (Batteries)': 'Consumer Durables',
        'Consumer': 'Consumer Durables',
        'Consumer Durables': 'Consumer Durables',
        'Consumer Durables - Gems, Jewellery And': 'Consumer Durables',  # Consolidating 'Consumer Durables - Gems, Jewellery And' to 'Consumer Durables'
        'Consumer Durables - Plastic': 'Consumer Durables',  # Consolidating 'Consumer Durables - Plastic' to 'Consumer Durables'

        # 8. Diversified
        'Diversified': 'Diversified',
        'Diversified Commercial Services': 'Diversified',
        'Diversified FMCG': 'Diversified',

        # 9. Financial Institution
        'Financial Institution': 'Financial Institution',
        'Financial Services (Capital Market)': 'Financial Institution',
        'Financial Services (capital markets)': 'Financial Institution',
        'Financial Services (fintech)': 'Financial Institution',

        # 10. Holding Company
        'Holding': 'Holding Company',
        'Holding company': 'Holding Company',

        # 11. IT
        'IT': 'IT',
        'IT - BPO and KPO Services': 'IT',
        'IT - Financial Technology': 'IT',
        'IT - Software': 'IT',
        'IT Computers - Software & Consulting': 'IT',
        'IT – Computers - Software & Consulting': 'IT', # Added to consolidate
        'IT – Financial Technology': 'IT', # Added to consolidate
        'IT – Other telecom services': 'IT', # Added to consolidate
        'IT Other telecom services': 'IT',
        'Information Technology': 'IT',
        'Information Technology (Satellite Communication Services)': 'IT',

        # 12. Industrials
        'Industrial': 'Industrials',
        'Industrial Products (Auto Components & Equipments)': 'Industrials',
        'Industrial Products-Plastic & Packaging': 'Industrials',
        'Industrial gases': 'Industrials',
        'Industrial manufacturing': 'Industrials',
        'Industrial products': 'Industrials',
        'Industrial Products':'Industrials',
        'Industrial products (Packaging)': 'Industrials',
        'Industrials': 'Industrials',

        # 13. Media & Entertainment
        'Media & Entertainment': 'Media & Entertainment',
        'Media Entertainment & Publication': 'Media & Entertainment',
        'Media Film Production Distribution & Exhibition': 'Media & Entertainment',
        'Media & Entertainment': 'Media & Entertainment',
        'Media – Film Production Distribution & Exhibition': 'Media & Entertainment',

        # 14. Metals
        'Metals - Ferrous': 'Metals',
        'Metals - Non-Ferrous': 'Metals',
        'Metals Ferrous (Ferro & Silica Manganese)': 'Metals',
        'Metals Ferrous (Ferro & silica manganese)': 'Metals',
        'Metals Ferrous (Iron & Steel)': 'Metals',
        'Metals Ferrous (Iron & Steel Products)': 'Metals',
        'Metals Ferrous (Iron & Steel products)': 'Metals',
        'Metals Ferrous (Iron and steel)': 'Metals',
        'Metals Ferrous (Pig Iron)': 'Metals',
        'Metals Linked products (Iron & Steel Products)': 'Metals',

        # 15. Minerals and Mining
        'Minerals & Mining': 'Minerals and Mining',
        'Minerals and mining': 'Minerals and Mining',
        'Mining': 'Minerals and Mining',
        'Mining (coal)': 'Minerals and Mining',
        'Non - Ferrous Metals': 'Minerals and Mining', # Explicitly requested in point 15

        # 16. Oil & Gas
        'Oil & Gas - Exploration & Production': 'Oil & Gas',
        'Oil & Gas - Gas': 'Oil & Gas',
        'Oil & Gas - Refining & Marketing': 'Oil & Gas',
        'Oil and Gas - gas': 'Oil & Gas',
        'Oil and Gas - Gas': 'Oil & Gas',

        # 17. Plastic Products
        'Plastic Products Industrial (Pipes)': 'Plastic Products',
        'Plastic products- Industrial': 'Plastic Products',
        'Plastic Products': 'Plastic Products',
        'Plastic Products – Industrial (Pipes)': 'Plastic Products',

        # 18. Ports & Services
        'Port & Port services': 'Ports & Services',
        'Ports & Port Services': 'Ports & Services',

        # 19. Power
        'Power - Thermal': 'Power',
        'Power Renewable (Hydro, Nuclear)': 'Power',
        'Power T&D': 'Power',
        'Power Thermal': 'Power',
        'Power Trading': 'Power',
        'Power generation (Renewables)': 'Power',
        'Power- Thermal': 'Power',
        'Power- Thermal (Integrated Power Utilities)': 'Power',
        'Power- Thermal (Power Generation)': 'Power',

        # 20. Retail
        'Retailing (Diversified Retail)': 'Retail',
        'Retailing (Pharmacy Retail)': 'Retail',

        # 21. Road Assets - Toll Annuity Hybrid-Annuity
        'Road Assets - Toll Annuity Hybrid-Annuity': 'Road Assets - Toll Annuity Hybrid-Annuity',
        'Road Assets Toll, Annuity, Hybrid-Annuity': 'Road Assets - Toll Annuity Hybrid-Annuity',
        'Road Assets–Toll, Annuity, Hybrid-Annuity': 'Road Assets - Toll Annuity Hybrid-Annuity',

        # 22. Ship Building & Allied Services
        'Ship Building & Allied Services': 'Ship Building & Allied Services',
        'Shipbuilding & Allied Services': 'Ship Building & Allied Services',

        # 23. Telecommunications
        'Telecom': 'Telecommunications',
        'Telecom - Cellular & Fixed line services': 'Telecommunications',
        'Telecom - Infrastructure': 'Telecommunications',
        'Telecommunication - Cellular & Fixed line services': 'Telecommunications',
        'Telecommunication - Equipment & Accessories': 'Telecommunications',
        'Telecommunication - Infrastructure': 'Telecommunications',
        'Telecommunication - Services': 'Telecommunications',
        'Telecommunication Infrastructure': 'Telecommunications',
        'Telecommunications': 'Telecommunications',
        'Telecommunications - Equipment & Accessories': 'Telecommunications',
        'Telecommunications - Services': 'Telecommunications',
        'Telecommunication -Services': 'Telecommunications',
        'Telecommunication – Infrastructure': 'Telecommunications',
        'Equipment & Accessories': 'Telecommunications', # Added to consolidate

        'Textile': 'Textiles', # Consolidate 'Textile' to 'Textiles'

        # 24. Tour and Travel Related Services
        'Tour and travel-related services': 'Tour and Travel Related Services',
        'Tour, Travel Related Services': 'Tour and Travel Related Services',
    }

    # Apply the mapping to the 'Sector_classification' column
    # Use .get() with a default to handle any classifications not explicitly in the map,
    # keeping them as is.
    df['Sector_classification'] = df['Sector_classification'].apply(
        lambda x: sector_mapping.get(x.strip(), x) if isinstance(x, str) else x
    )
    
    return df

@st.cache_data
def load_real_electricity_data():
    """Load electricity consumption data from the new 'electricity_data_with_mpce_hhsize.csv' file."""
    try:
        file_path = "electricity_data_with_mpce_hhsize.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            st.error("No electricity data file found. Please make sure 'electricity_data_with_mpce_hhsize.csv' exists in the app directory.")
            return None

       
        needed = ['state_name', 'cons_total_qty', 'hh_size']
        missing = [c for c in needed if c not in df.columns]
        if missing:
            st.error(f"Missing columns in 'electricity_data_with_mpce_hhsize.csv': {', '.join(missing)}")
            return None

        # No renaming needed as the columns are already in the desired format:
        # 'state_name' for state names
        # 'cons_total_qty' for electricity usage per month
        # 'hh_size' for household size

        # Ensure data types are correct
        df['cons_total_qty'] = pd.to_numeric(df['cons_total_qty'], errors='coerce')
        df['hh_size'] = pd.to_numeric(df['hh_size'], errors='coerce')
        df.dropna(subset=['state_name', 'cons_total_qty', 'hh_size'], inplace=True)

        # Calculate per_person_qty_usage for the baseline
        # Handle division by zero for hh_size
        df['per_person_qty_usage'] = df.apply(
            lambda row: row['cons_total_qty'] / row['hh_size'] if row['hh_size'] > 0 else row['cons_total_qty'],
            axis=1
        )

        # Correcting state names in the DataFrame to match INDIAN_STATES_UTS for proper filtering
        df['state_name'] = df['state_name'].replace({
            'Tamilnadu': 'Tamil Nadu',
            'Lakshadweep (U.T.)': 'Lakshadweph',
            'Chandigarh(U.T.)': 'Chandigarh',
            'Dadra & Nagar Haveli and Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
            'Jammu & Kashmir': 'Jammu and Kashmir',
            'Ladakh (U.T.)': 'Ladakh',
            'Puducherry (U.T.)': 'Puducherry',
            'Uttrakhand': 'Uttarakhand',
            'A and N Islands (U.T.)': 'Andaman and Nicobar Islands',
        })

        # Further filter for states that are in INDIAN_STATES_UTS
        df = df[df['state_name'].isin(INDIAN_STATES_UTS)].copy()
        if df.empty:
            st.warning("No valid electricity data records found after state filtering.")
            return None

        state_stats = {}
        for state in df["state_name"].unique():
            state_df = df[df["state_name"] == state]
            if len(state_df) > 0: # Check if state_df is not empty after filtering
                state_stats[state] = {
                    "Electricity": (
                        state_df["per_person_qty_usage"].mean(), # Use per_person_qty_usage for mean
                        state_df["per_person_qty_usage"].std()   # Use per_person_qty_usage for std
                    ),
                    "HH_Size": (
                        state_df["hh_size"].mean(),
                        state_df["hh_size"].std()
                    )
                }
        st.session_state.baseline_values_by_state = state_stats
        return df

    except Exception as e:
        st.error(f"Error loading or processing 'electricity_data_with_mpce_hhsize.csv': {str(e)}")
        return None


if 'company_data' not in st.session_state:
    st.session_state.company_data = load_company_data()
if 'electricity_data' not in st.session_state:
    st.session_state.electricity_data = load_real_electricity_data()
    if st.session_state.electricity_data is not None:
        st.session_state.full_electricity_data = st.session_state.electricity_data # Store for overall stats

# Ensure baseline_values_by_state is always initialized
if 'baseline_values_by_state' not in st.session_state:
    st.session_state.baseline_values_by_state = {}

# Ensure weights are always initialized with all expected keys
if 'weights' not in st.session_state:
    st.session_state.weights = {} # Initialize as an empty dict if not present at all

# Set default values for each weight, adding them if missing
st.session_state.weights.setdefault("Electricity", 0.2885)
st.session_state.weights.setdefault("Water", 0.2885)
st.session_state.weights.setdefault("Commute", 0.1923)
st.session_state.weights.setdefault("Company", 0.2308)

if 'sub_weights' not in st.session_state:
    st.session_state.sub_weights = {
        "electricity": {"location": 0.125, "mpce": 0.125}, # These sub-weights might become redundant if only "Electricity" is used
        "commute": {"public": 0.125, "private": 0.125}, # These sub-weights might become redundant if only "Commute" is used
        "water": {"water": 0.25},
        "company": {"company": 0.25}
    }

if 'scored_data' not in st.session_state:
    st.session_state.scored_data = pd.DataFrame()
if 'user_electricity' not in st.session_state:
    st.session_state.user_electricity = 0.0
if 'user_state' not in st.session_state:
    st.session_state.user_state = INDIAN_STATES_UTS[0]
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
    st.session_state.crisil_esg_score_input = 0.0 
if 'selected_company_for_comparison' not in st.session_state:
    st.session_state.selected_company_for_comparison = None
if 'custom_esg_score_for_comparison' not in st.session_state:
    st.session_state.custom_esg_score_for_comparison = 50.0 
if 'esg_comparison_method' not in st.session_state:
    st.session_state.esg_comparison_method = "Select from list"
if 'esg_analysis_type' not in st.session_state:
    st.session_state.esg_analysis_type = "Employee Range Analysis"
if 'selected_esg_industry' not in st.session_state:
    st.session_state.selected_esg_industry = None
if 'selected_esg_emp_size' not in st.session_state:
    st.session_state.selected_esg_emp_size = "Small (<5,000)"
if 'calculated' not in st.session_state: 
    st.session_state.calculated = False
if 'total_emissions' not in st.session_state: 
    st.session_state.total_emissions = 0.0
if 'emissions_data' not in st.session_state: 
    st.session_state.emissions_data = {}
if 'recommendations' not in st.session_state: 
    st.session_state.recommendations = []
# Initialize new session state variables for transport inputs
if 'distance_input' not in st.session_state:
    st.session_state.distance_input = 10.0
if 'days_per_week_input' not in st.session_state:
    st.session_state.days_per_week_input = 5
if 'weeks_per_month_input' not in st.session_state:
    st.session_state.weeks_per_month_input = 4
if 'private_trips_per_day_input' not in st.session_state: # New
    st.session_state.private_trips_per_day_input = 1
if 'public_trips_per_day_input' not in st.session_state: # New
    st.session_state.public_trips_per_day_input = 1


@st.cache_data
def calculate_feature_stats(electricity_data):
    if electricity_data is None or electricity_data.empty:
        return {}
    return {
        'Electricity': {
            'mean': electricity_data['cons_total_qty'].mean(),
            'std':  electricity_data['cons_total_qty'].std()
        }
    }
st.session_state.feature_stats = calculate_feature_stats(st.session_state.electricity_data)

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
        "base_emission": 0.03, 
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

# Remove global emission_mean and emission_std calculation as it will be calculated dynamically for transport_z
# def calculate_emission_stats():
#     all_values = []
#     for cat in emission_factors["two_wheeler"]:
#         for fuel in emission_factors["two_wheeler"][cat]:
#             min_val, max_val = emission_factors["two_wheeler"][cat][fuel]["min"], emission_factors["two_wheeler"][cat][fuel]["max"]
#             all_values.append((min_val + max_val) / 2)
#     for fuel in emission_factors["three_wheeler"]:
#         min_val, max_val = emission_factors["three_wheeler"][fuel]["min"], emission_factors["three_wheeler"][fuel]["max"]
#         all_values.append((min_val + max_val) / 2)
#     for car_type in emission_factors["four_wheeler"]:
#         for fuel in emission_factors["four_wheeler"][car_type]:
#             base, uplift = emission_factors["four_wheeler"][car_type][fuel]["base"], emission_factors["four_wheeler"][car_type][fuel]["uplift"]
#             all_values.append(base * uplift)
#     for taxi_type in emission_factors["public_transport"]["taxi"]:
#         for fuel in emission_factors["public_transport"]["taxi"][taxi_type]:
#             base, uplift = emission_factors["public_transport"]["taxi"][taxi_type][fuel]["base"], emission_factors["public_transport"]["taxi"][taxi_type][fuel]["uplift"]
#             all_values.append(base * uplift)
#     for fuel in emission_factors["public_transport"]["bus"]:
#         all_values.append(emission_factors["public_transport"]["bus"][fuel])
#     all_values.append(emission_factors["public_transport"]["metro"])
#     return np.mean(all_values), np.std(all_values)

# emission_mean, emission_std = calculate_emission_stats()

# Remove calculate_z_score_emission as it's no longer used
# def calculate_z_score_emission(emission_factor_val):
#     """Calculate Z-score for given emission factor"""
#     if emission_std == 0:
#         return 0
#     return (emission_factor_val - emission_mean) / emission_std

# --- Unified Sustainability Score Calculation Function ---
def calculate_sustainability_score(customer_data, company_data_df, electricity_data_df, weights): # Removed emission_mean, emission_std from arguments
    """
    Calculates the overall sustainability score and individual Z-scores for a customer.

    Args:
        customer_data (dict): Dictionary containing customer inputs.
        company_data_df (pd.DataFrame): DataFrame with company ESG data.
        electricity_data_df (pd.DataFrame): DataFrame with electricity consumption data.
        weights (dict): Dictionary of weights for each sustainability factor.

    Returns:
        dict: A dictionary containing individual Z-scores, total Z-score, and sustainability score.
    """
    st.write(f"DEBUG: calculate_sustainability_score received customer_data: {customer_data}")

    # Extract customer inputs
    user_electricity = customer_data.get('Electricity', 0.0)
    household_size = customer_data.get('People in the household', 1)
    crisil_esg = customer_data.get('Crisil_ESG_Score', 0.0)
    per_person_monthly_water = customer_data.get('Water_Per_Person_Monthly', 0.0)
    
    # Transport inputs (now derived from total_monthly_km and category)
    private_monthly_km = customer_data.get('Private_Monthly_Km', 0.0)
    public_monthly_km = customer_data.get('Public_Monthly_Km', 0.0)
    total_monthly_km = customer_data.get('Km_per_month', 0.0) # Total monthly km for bonus calculation
    emission_factor = customer_data.get('Transport_Emission_Factor', 0.0)
    people_count_transport = customer_data.get('Transport_People_Count', 1)
    
    # Ensure household_size is at least 1 to avoid division by zero
    safe_household_size = household_size if household_size > 0 else 1
    
    # Calculate individual equivalent electricity consumption
    individual_equivalent_consumption = user_electricity / safe_household_size
    st.write(f"DEBUG: Individual Equivalent Electricity Consumption: {individual_equivalent_consumption}")


    # --- Z-score Calculation for each factor ---

    # 1. Electricity Z-score
    electricity_z = 0
    electricity_baseline = 0.0
    electricity_std = 0.0 # Default fallback if data issues
    if electricity_data_df is not None and not electricity_data_df.empty and 'per_person_qty_usage' in electricity_data_df.columns:
        user_state_name = customer_data.get('State') # Get the user's selected state
        state_data = electricity_data_df[electricity_data_df['state_name'] == user_state_name]
        
        if not state_data.empty:
            electricity_baseline = state_data['per_person_qty_usage'].mean()
            electricity_std = state_data['per_person_qty_usage'].std()
        else:
            # Fallback to overall mean/std if no data for selected state
            electricity_baseline = electricity_data_df['per_person_qty_usage'].mean()
            electricity_std = electricity_data_df['per_person_qty_usage'].std()

        if electricity_std > 0:
            electricity_z = (individual_equivalent_consumption - electricity_baseline) / electricity_std
        else:
            electricity_z = max(0, (individual_equivalent_consumption - electricity_baseline) / 50) # Fallback if std is 0
    else:
        # Fallback if electricity data is not loaded or invalid
        electricity_z = max(0, (individual_equivalent_consumption - 50) / 50)
    st.write(f"DEBUG: Electricity Z-score: {electricity_z}")
    st.write(f"DEBUG: Electricity Baseline used for Z-score: {electricity_baseline:.4f} kWh")
    st.write(f"DEBUG: Electricity Std Dev used for Z-score: {electricity_std:.4f} kWh")


    # 2. Water Z-score
    # Baseline for water consumption (e.g., 130 liters/day * 30 days = 3900 liters/month)
    average_monthly_per_person_water = 130 * 30
    water_z = 0
    if average_monthly_per_person_water > 0:
        water_z = (per_person_monthly_water - average_monthly_per_person_water) / average_monthly_per_person_water
    st.write(f"DEBUG: Per Person Monthly Water: {per_person_monthly_water}, Average Monthly Per Person Water: {average_monthly_per_person_water}, Water Z-score: {water_z}")

    # 3. Transport Z-score (using provided snippet logic)
    transport_z = 0
    
    # Calculate bus emission mean and std for the given distance
    bus_emission_factors_values = list(emission_factors["public_transport"]["bus"].values())
    avg_bus_emission_per_km = np.mean(bus_emission_factors_values)
    std_bus_emission_per_km = np.std(bus_emission_factors_values)

    # Scale mean and std by the total monthly distance
    bus_emission_mean_for_distance = avg_bus_emission_per_km * total_monthly_km
    bus_emission_std_for_distance = std_bus_emission_per_km * total_monthly_km

    # Ensure std_for_distance is not zero to avoid division by zero
    if bus_emission_std_for_distance == 0:
        # If std is zero (e.g., all bus emissions are the same or total_monthly_km is zero)
        # use a small arbitrary value to prevent division by zero.
        # This fallback uses a small fraction of the mean if mean is not zero, otherwise a fixed small value.
        if bus_emission_mean_for_distance > 0:
            bus_emission_std_for_distance = bus_emission_mean_for_distance * 0.01 
        else:
            bus_emission_std_for_distance = 0.005 

    # The emission factor is already per km, so we need to consider the total monthly km
    # and the number of people sharing the transport to get an individual equivalent emission.
    individual_emission_factor_per_km = emission_factor / (people_count_transport if people_count_transport > 0 else 1)
    
    # Calculate total individual emission for the month
    individual_total_monthly_emission = individual_emission_factor_per_km * total_monthly_km

    if bus_emission_std_for_distance > 0:
        transport_z = (individual_total_monthly_emission - bus_emission_mean_for_distance) / bus_emission_std_for_distance
    else:
        # Fallback if bus_emission_std_for_distance is 0, use a simple comparison to a baseline
        transport_z = max(0, (individual_total_monthly_emission - bus_emission_mean_for_distance) / 0.05) # Small arbitrary std for fallback
        
    # FIX: Removed the inversion here. A negative Z-score means better emissions,
    # which will correctly contribute to a more negative total_z_score and higher sustainability score.
    # transport_z = -transport_z

    # --- Add bonus for public transportation usage ---
    # The goal is to make the overall sustainability score better for higher public transport usage.
    # A better sustainability score implies a more negative 'total_z_score'.
    # Since 'transport_z' is now negative for good emissions, subtracting a positive bonus
    # will make it even more negative, further improving the overall score.
    
    if total_monthly_km > 0:
        public_transport_ratio = public_monthly_km / total_monthly_km
        
        # Define a threshold: if more than 50% of commute is public transport, apply a bonus.
        public_transport_threshold = 0.5 
        # Maximum Z-score reduction (bonus) to apply.
        # This value will be subtracted from the current 'transport_z'.
        max_bonus_z_reduction = 0.5 

        if public_transport_ratio > public_transport_threshold:
            # Scale the bonus based on how much the ratio exceeds the threshold.
            # The 'excess_ratio' goes from 0 (at threshold) to 1 (at 100% public transport).
            excess_ratio = min(1.0, (public_transport_ratio - public_transport_threshold) / (1.0 - public_transport_threshold))
            
            # Calculate the actual bonus reduction (a positive value to be subtracted from transport_z)
            bonus_reduction = excess_ratio * max_bonus_z_reduction
            
            # Apply the bonus by subtracting it from transport_z
            transport_z -= bonus_reduction
    st.write(f"DEBUG: Private Monthly Km: {private_monthly_km}, Public Monthly Km: {public_monthly_km}, Transport Z-score: {transport_z}")
    
    # 4. Company ESG Z-score
    m = 0.1 # Multiplier for crisil_esg
    company_z = crisil_esg * m
    # Invert company Z-score so higher ESG scores result in a better (more negative) Z-score
    # This inversion is no longer needed as per the new formula company_z = crisil_esg * m
    company_z = -company_z 
    st.write(f"DEBUG: Company ESG Score: {crisil_esg}, Company Z-score: {company_z}")

    # --- Weighted Total Z-score ---
    total_z_score = (
        electricity_z * weights["Electricity"] +
        water_z * weights["Water"] +
        transport_z * weights["Commute"] +
        company_z * weights["Company"]
    )
    st.write(f"DEBUG: Total Z-score (before tanh): {total_z_score}")

    # --- Sustainability Score Calculation ---
    # The tanh function scales the total Z-score to a range, and 500 * (1 - ...) maps it to 0-1000
    sustainability_score = 500 * (1 - np.tanh(total_z_score / 2.5))
    sustainability_score = np.clip(sustainability_score, 0, 1000) # Ensure score is within 0-1000
    st.write(f"DEBUG: Sustainability Score: {sustainability_score}")

    return {
        'electricity_z': electricity_z,
        'water_z': water_z,
        'transport_z': transport_z, # Return the calculated transport Z-score
        'company_z': company_z,
        'total_z_score': total_z_score,
        'sustainability_score': sustainability_score
    }


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

    # Removed 'Rural/Urban' and 'MPCE' from PDF generation
    for col in ['Electricity', 'Water', 'Private_Monthly_Km', 'Public_Monthly_Km', 'Crisil_ESG_Score', 'Sustainability_Score']:
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


    return pdf.output(dest='S')

# --- Page 1: Test New Customer ---
def page_test_customer():
    st.header("1. New Customer Assessment")
    st.markdown("Enter demographic details, resource consumption, and transport choices for a new customer.")

    test_mode = st.radio("Input Mode", ["Manual Entry", "CSV Upload"], key="test_mode")

    if test_mode == "Manual Entry":
        st.subheader("Demographics")
        col1, col2 = st.columns(2)
        with col1:
            user_state = st.selectbox("State/UT", options=sorted(INDIAN_STATES_UTS), key="input_state")
            st.session_state.user_state = user_state
        
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
                # Use the mapped sector classifications for the selectbox
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
            distance = st.number_input("Daily one-way commute distance (km)", min_value=0.1, value=st.session_state.distance_input, step=0.5, key="input_distance")
            st.session_state.distance_input = distance
        with col_t2:
            days_per_week = st.number_input("Commuting days per week", min_value=1, max_value=7, value=st.session_state.days_per_week_input, step=1, key="input_days_per_week")
            st.session_state.days_per_week_input = days_per_week
        with col_t3:
            weeks_per_month = st.number_input("Commuting weeks per month", min_value=1, max_value=5, value=st.session_state.weeks_per_month_input, step=1, key="input_weeks_per_month")
            st.session_state.weeks_per_month_input = weeks_per_month
        total_monthly_km = distance * 2 * days_per_week * weeks_per_month
        
        st.session_state.total_monthly_km = total_monthly_km

        transport_category = st.selectbox("Transport Category", ["Private Transport", "Public Transport", "Both Private and Public"], key="input_transport_category")

        emission_factor_calc = 0
        people_count_calc = 1
        vehicle_type_calc = ""
        vehicle_name_calc = ""
        private_emission_factor = 0
        public_emission_factor = 0
        private_vehicle_name = ""
        public_vehicle_name = ""
        current_public_people = 1

        if transport_category == "Private Transport" or transport_category == "Both Private and Public":
            st.markdown("##### Private Transport Details")
            has_multiple_vehicles = st.checkbox("I have multiple private vehicles", key="input_multiple_vehicles")
            if has_multiple_vehicles:
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
                            w2_engine_cc = st.number_input("Engine (cc)", 50, 1500, 150, key=f"2w_cc_input_{i}")
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
                            w4_engine_cc = st.slider("Engine (cc)", 600, 4000, 1200, key=f"4w_cc_input_{i}")
                        w4_fuel_options = ["petrol", "diesel", "cng", "electric"]
                        if w4_car_type == "hybrid":
                            w4_fuel_options = ["petrol", "diesel", "electric"]
                        w4_fuel_type = st.selectbox("Fuel Type", w4_fuel_options, key=f"4w_fuel_input_{i}")
                        base_ef = emission_factors["four_wheeler"][w4_car_type][w4_fuel_type]["base"]
                        uplift = emission_factors["four_wheeler"][w4_car_type][w4_fuel_type]["uplift"]
                        engine_factor = 1.0 + min(1.0, (w4_engine_cc - 600) / 3400) * 0.5 if w4_fuel_type != "electric" else 1.0
                        ef_4w = base_ef * uplift * engine_factor
                        total_private_emission_factor_multiple += ef_4w
                        temp_private_names.append(f"{w4_car_type.replace('_', ' ').title()} ({w4_fuel_type}, {w4_engine_cc}cc)")
                private_emission_factor = total_private_emission_factor_multiple / num_vehicles if num_vehicles > 0 else 0 
                private_vehicle_name = f"Multiple Vehicles: {', '.join(temp_private_names)}"
                people_count_calc = 1 # Assuming average 1 person per vehicle for multiple vehicle emission factor calc
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
        
        # Calculate private_monthly_km and public_monthly_km based on transport_category
        private_monthly_km = 0.0
        public_monthly_km = 0.0

        if transport_category == "Private Transport":
            emission_factor_calc = private_emission_factor
            vehicle_type_calc = "Private Transport"
            vehicle_name_calc = private_vehicle_name
            private_monthly_km = total_monthly_km
            public_monthly_km = 0.0
        elif transport_category == "Public Transport":
            emission_factor_calc = public_emission_factor
            vehicle_type_calc = "Public Transport"
            vehicle_name_calc = public_vehicle_name
            people_count_calc = current_public_people
            private_monthly_km = 0.0
            public_monthly_km = total_monthly_km
        elif transport_category == "Both Private and Public":
            st.markdown("##### Usage Distribution (for Both Private and Public)")
            # Updated inputs for private and public trips per day
            private_trips_per_day = st.number_input("Number of private trips per day", min_value=0, max_value=10, value=st.session_state.private_trips_per_day_input, step=1, key="input_private_trips_per_day")
            public_trips_per_day = st.number_input("Number of public trips per day", min_value=0, max_value=10, value=st.session_state.public_trips_per_day_input, step=1, key="input_public_trips_per_day")
            
            st.session_state.private_trips_per_day_input = private_trips_per_day
            st.session_state.public_trips_per_day_input = public_trips_per_day

            total_trips_per_day = private_trips_per_day + public_trips_per_day
            
            if total_trips_per_day > 0:
                private_ratio = private_trips_per_day / total_trips_per_day
                public_ratio = public_trips_per_day / total_trips_per_day
            else:
                private_ratio = 0.0
                public_ratio = 0.0
            
            private_monthly_km = total_monthly_km * private_ratio
            public_monthly_km = total_monthly_km * public_ratio

            combined_emission_factor = (private_emission_factor / (people_count_calc if people_count_calc > 0 else 1)) * private_ratio + \
                                       (public_emission_factor / (current_public_people if current_public_people > 0 else 1)) * public_ratio
            emission_factor_calc = combined_emission_factor
            vehicle_type_calc = "Combined Transport"
            vehicle_name_calc = f"{private_vehicle_name} ({private_ratio*100:.0f}%) & {public_vehicle_name} ({public_ratio*100:.0f}%)"
            people_count_calc = 1 # For combined, we consider the overall emission factor per person

        if st.button("Submit Assessment (Proceed to next page for evaluation)"):
            
            final_esg_score = 0.0
            compare_method = st.session_state.get('form_esg_comparison_method')
            
            # Initialize employee_size_value and selected_company_name for CSV download
            employee_size_value_for_csv = None
            selected_company_name_for_csv = None

            if compare_method == "Enter custom score":
                final_esg_score = st.session_state.get('enter_custom_esg_score', 0.0)
            
            elif compare_method == "Select from list":
                selected_company_name = st.session_state.get('select_company_for_comparison')

                if selected_company_name and selected_company_name != "(None)":
                    company_df = st.session_state.get('company_data')
                    if company_df is not None:
                        company_row = company_df[company_df['Company_Name'] == selected_company_name]
                        if not company_row.empty:
                            final_esg_score = company_row.iloc[0]['Environment_Score']
                            selected_company_name_for_csv = selected_company_name # Save selected company name

                else: # No specific company selected, use average for Employee Range Analysis if applicable
                    company_df = st.session_state.get('company_data')
                    analysis_type = st.session_state.get('crisil_analysis_type')

                    if company_df is not None and analysis_type == "Employee Range Analysis":
                        selected_industry = st.session_state.get('employee_range_industry')
                        employee_size_cat = st.session_state.get('employee_size_category')

                        df_to_filter = company_df.copy()

                        if selected_industry:
                            df_to_filter = df_to_filter[df_to_filter['Sector_classification'] == selected_industry]
                        
                        if employee_size_cat:
                            if employee_size_cat == "Small (<5,000)":
                                df_to_filter = df_to_filter[df_to_filter['Total_Employees'] < 5000]
                                employee_size_value_for_csv = 5000
                            elif employee_size_cat == "Medium (5,000 to 15,000)":
                                df_to_filter = df_to_filter[(df_to_filter['Total_Employees'] >= 5000) & (df_to_filter['Total_Employees'] <= 15000)]
                                employee_size_value_for_csv = 15000
                            else: # Large
                                df_to_filter = df_to_filter[df_to_filter['Total_Employees'] > 15000]
                                employee_size_value_for_csv = 15000 # As per user's request for >15000

                        if not df_to_filter.empty:
                            average_score = df_to_filter['Environment_Score'].mean()
                            final_esg_score = average_score
                            st.info(f"No specific company was selected. Using the average Environment Score ({average_score:.2f}) of the {len(df_to_filter)} companies in the selected filter as the benchmark.")

            st.session_state.crisil_esg_score_input = final_esg_score
            
            st.session_state.esg_analysis_type = st.session_state.crisil_analysis_type
            st.session_state.selected_esg_industry = st.session_state.get('employee_range_industry', None)
            st.session_state.selected_esg_emp_size = st.session_state.get('employee_size_category', "Small (<5,000)")
            st.session_state.selected_company_for_comparison = st.session_state.get('select_company_for_comparison', None)
            st.session_state.esg_comparison_method = st.session_state.form_esg_comparison_method
            
            if st.session_state.crisil_esg_score_input == 0.0 and st.session_state.user_electricity == 0.0 and st.session_state.water_units == 0.0 and total_monthly_km == 0.0:
                st.error("Please enter at least one input for Crisil ESG, Electricity, Water, or Transport to submit the assessment.")
            else:
                st.success("Customer data submitted! You can now navigate to '3. Evaluation Results & Reporting' to see the assessment.")

                st.session_state.transport_carbon_results = {
                    "total_monthly_km": total_monthly_km,
                    "emission_factor": emission_factor_calc,
                    "people_count": people_count_calc,
                    "vehicle_type": vehicle_type_calc,
                    "vehicle_name": vehicle_name_calc
                }

                single_customer_inputs_to_save = {
                    'State': user_state,
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
                    'Private_Monthly_Km': private_monthly_km, # Store calculated private km
                    'Public_Monthly_Km': public_monthly_km, # Store calculated public km
                    'Private_Trips_Per_Day': st.session_state.private_trips_per_day_input if transport_category == "Both Private and Public" else 0, # New
                    'Public_Trips_Per_Day': st.session_state.public_trips_per_day_input if transport_category == "Both Private and Public" else 0  # New
                }

                # Add employee size or selected company based on analysis type
                if st.session_state.esg_analysis_type == "Employee Range Analysis":
                    single_customer_inputs_to_save['Total_Employees'] = employee_size_value_for_csv
                elif st.session_state.esg_analysis_type == "Company-Only Analysis":
                    single_customer_inputs_to_save['Selected_Company'] = selected_company_name_for_csv
                
                st.session_state.single_customer_inputs = single_customer_inputs_to_save
                st.write(f"DEBUG: Manual Input - st.session_state.single_customer_inputs: {st.session_state.single_customer_inputs}")

                submitted_data_df = pd.DataFrame([st.session_state.single_customer_inputs])
                csv_data = submitted_data_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Download Submitted Data as CSV",
                    data=csv_data,
                    file_name='customer_assessment_submission.csv',
                    mime='text/csv',
                    key='download_submitted_csv'
                )

                st.session_state.calculated = True
                st.session_state.total_emissions = (emission_factor_calc * total_monthly_km) / (people_count_calc if people_count_calc > 0 else 1)
                
                alternatives = {vehicle_name_calc: st.session_state.total_emissions}
                # Add alternatives based on total_monthly_km
                if total_monthly_km > 0:
                    alternatives["Bus (Diesel)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["diesel"]) / (people_count_calc if people_count_calc > 0 else 1)
                    alternatives["Bus (CNG)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["cng"]) / (people_count_calc if people_count_calc > 0 else 1)
                    alternatives["Bus (Electric)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["electric"]) / (people_count_calc if people_count_calc > 0 else 1)
                    alternatives["Metro"] = (total_monthly_km * emission_factors["public_transport"]["metro"]) / (people_count_calc if people_count_calc > 0 else 1)
                    
                    # Carpooling alternative if private transport is used
                    if private_monthly_km > 0 and not (vehicle_type_calc == "Four Wheeler" and people_count_calc >= 3):
                        # Assuming a sedan petrol car for carpooling example
                        carpool_ef = emission_factors["four_wheeler"]["sedan"]["petrol"]["base"] * emission_factors["four_wheeler"]["sedan"]["petrol"]["uplift"]
                        alternatives["Car Pooling (4 people)"] = (private_monthly_km * carpool_ef) / 4
                    
                    # Electric car alternative if private transport is used and not already electric
                    if private_monthly_km > 0 and not ("electric" in vehicle_name_calc.lower()):
                        # Assuming a sedan electric car for alternative
                        electric_car_ef = emission_factors["four_wheeler"]["sedan"]["electric"]["base"] * emission_factors["four_wheeler"]["sedan"]["electric"]["uplift"]
                        alternatives["Electric Car"] = (private_monthly_km * electric_car_ef) / (people_count_calc if people_count_calc > 0 else 1)
                    
                    # Electric scooter alternative if private transport is used and not already electric
                    if private_monthly_km > 0 and not ("electric scooter" in vehicle_name_calc.lower()):
                        electric_scooter_ef = emission_factors["two_wheeler"]["Scooter"]["electric"]["min"]
                        alternatives["Electric Scooter"] = (private_monthly_km * electric_scooter_ef) / (people_count_calc if people_count_calc > 0 else 1)

                st.session_state.emissions_data = alternatives


    else: 
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

        up_test = st.file_uploader("Upload Test Data", type=["csv", "xlsx", "xls"], key="bulk_upload_file_page1")
        if up_test:
            try:
                if up_test.name.endswith(('.xlsx', '.xls')):
                    test_df = pd.read_excel(up_test, header=0, dtype=object)
                    st.success("Successfully loaded Excel file.")
                
                elif up_test.name.endswith('.csv'):
                    test_df = None
                    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    for encoding in encodings_to_try:
                        try:
                            up_test.seek(0)
                            test_df = pd.read_csv(up_test, encoding=encoding, dtype=object)
                            st.success(f"Successfully loaded CSV with '{encoding}' encoding.")
                            break
                        except Exception:
                            continue
                    
                    if test_df is None:
                        st.error("Could not read the CSV file with any of the supported encodings. Please check the file format.")
                        st.stop()
                
                else:
                    st.error(f"Unsupported file type: {up_test.name}. Please upload a .csv, .xlsx, or .xls file.")
                    st.stop()

                test_df = test_df.where(pd.notna(test_df), None)

            except Exception as e:
                st.error(f"Error reading or processing the file: {str(e)}")
                st.stop()

            st.markdown("#### Uploaded Test Data Preview")
            st.dataframe(test_df)
            st.session_state.uploaded_bulk_df = test_df

            if st.button("Process & Evaluate Bulk Data (Proceed to next page for evaluation)", key="process_batch_btn_page1"):
                st.success("Bulk data uploaded. Please navigate to '3. Evaluation Results & Reporting' to see the processed results.")
                st.session_state.current_eval_mode = "Bulk CSV Upload"


def page_evaluation():
    st.header("2. Weighted Score Settings")
    st.markdown("Adjust the importance of different sustainability factors.")

    st.markdown("**Electricity Consumption**")
    w_elec = st.number_input("Electricity Weight", value=st.session_state.weights["Electricity"], step=0.001, format="%.4f", key="wt_elec")
    st.markdown(f"*(Current Electricity Weight: {w_elec:.4f}, Target: 0.2885)*")

    st.markdown("**Water Consumption**")
    w_water = st.number_input("Water Weight", value=st.session_state.weights["Water"], step=0.001, format="%.4f", key="wt_water")
    st.markdown(f"*(Current Water Weight: {w_water:.4f}, Target: 0.2885)*")

    st.markdown("**Commute**")
    w_commute = st.number_input("Commute Weight", value=st.session_state.weights["Commute"], step=0.001, format="%.4f", key="wt_commute")
    st.markdown(f"*(Current Commute Weight: {w_commute:.4f}, Target: 0.1923)*")

    st.markdown("**Company Environmental Score**")
    w_company = st.number_input("Company Environmental Score Weight", value=st.session_state.weights["Company"], step=0.001, format="%.4f", key="wt_company")
    st.markdown(f"*(Current Company Weight: {w_company:.4f}, Target: 0.2308)*")

    current_weights = {
        "Electricity": w_elec,
        "Water": w_water,
        "Commute": w_commute,
        "Company": w_company
    }
    total_current_weight = sum(current_weights.values())
    st.markdown(f"**Overall Total Weight:** {total_current_weight:.4f}")

    if abs(total_current_weight - 1.0) > 1e-3:
        st.error("Overall total weights must sum exactly to 1.0!")
    else:
        if st.button("Apply Weighted Score Settings"):
            st.session_state.weights = current_weights
            st.session_state.sub_weights = {
                "electricity": {"total": w_elec},
                "commute": {"total": w_commute},
                "water": {"water": w_water},
                "company": {"company": w_company}
            }
            st.success("Weighted score settings applied successfully!")

def page_reporting():
    st.header("3. Evaluation Results & Reporting")
    st.markdown("View detailed sustainability assessment for individual customers or process bulk data, and generate reports.")
    
    evaluation_mode_display = st.session_state.get('current_eval_mode', 'Individual Customer Evaluation') 

    if evaluation_mode_display == "Individual Customer Evaluation":
        st.subheader("Individual Customer Assessment Results")

        st.markdown("#### Weighted Score Contributions")
        st.markdown("This chart shows the weighting of each factor in the total sustainability score, as configured on Page 3.")

        weights = st.session_state.weights

        labels = [
            "Electricity", "Water", "Commute", "Company"
        ]
        parents = [
            "", "", "", ""
        ]

        values = [
            weights["Electricity"],
            weights["Water"],
            weights["Commute"],
            weights["Company"]
        ]

        if sum(weights.values()) == 0:
            st.warning("All weight values are zero. The chart cannot be displayed until weights are set on Page 3.")
        else:
            try:
                fig = go.Figure(go.Sunburst(
                    labels=labels,
                    parents=parents,
                    values=values,
                    branchvalues="total",
                    hovertemplate='<b>%{label}</b><br>Weight: %{value:.4f}<extra></extra>', # Updated format to 4 decimal places
                    insidetextorientation='radial',
                    maxdepth=1, # Changed maxdepth to 1 as there are no sub-categories for now
                ))

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

        if 'single_customer_inputs' not in st.session_state or not st.session_state.single_customer_inputs:
            st.info("Please go to '1. New Customer Assessment' page and submit customer details (Manual Entry mode) first.")
            return

        customer_data = st.session_state.single_customer_inputs

        # Call the unified sustainability score calculation function
        sustainability_results = calculate_sustainability_score(
            customer_data,
            st.session_state.company_data,
            st.session_state.electricity_data,
            st.session_state.weights
            # Removed emission_mean, emission_std from arguments
        )
        st.write(f"DEBUG: Manual Input - Sustainability Results: {sustainability_results}")


        sustainability_score = sustainability_results['sustainability_score']
        total_z_score = sustainability_results['total_z_score']
        electricity_z = sustainability_results['electricity_z']
        water_z = sustainability_results['water_z']
        transport_z = sustainability_results['transport_z'] # This is now the combined transport Z-score
        company_z = sustainability_results['company_z']

        user_electricity = customer_data['Electricity']
        household_size = customer_data['People in the household']
        user_state = customer_data['State']
        crisil_esg = customer_data['Crisil_ESG_Score']
        monthly_water_total = customer_data['Water_Monthly_Total']
        per_person_monthly_water = customer_data['Water_Per_Person_Monthly']
        total_monthly_km_disp = customer_data['Km_per_month'] # This is the sum of private and public km
        emission_factor_disp = customer_data['Transport_Emission_Factor'] # This is the calculated overall emission factor for display
        people_count_disp = customer_data['Transport_People_Count']
        vehicle_name_disp = customer_data['Vehicle_Name']
        user_cost = customer_data['Electricity_Cost']
        private_monthly_km_disp = customer_data['Private_Monthly_Km']
        public_monthly_km_disp = customer_data['Public_Monthly_Km']


        safe_hh_size = household_size if household_size > 0 else 1
        equivalence_factor = safe_hh_size
        individual_equivalent_consumption = user_electricity / equivalence_factor if equivalence_factor > 0 else 0

        st.session_state.calculated_single_customer_score = sustainability_score
        st.session_state.calculated_single_customer_zscore = total_z_score

        st.markdown("#### Individual Sustainability Score")
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Total Sustainability Score", f"{sustainability_score:.1f}/1000")
        with col_res2:
            st.metric("Combined Z-score", f"{total_z_score:.2f}")
        with col_res3:
            if st.session_state.scored_data is not None and not st.session_state.scored_data.empty:
                percentile = (st.session_state.scored_data['Sustainability_Score'] < sustainability_score).mean() * 100
                st.metric("Your Percentile", f"{percentile:.1f}%")
            # Removed the else block that displayed "No batch data available for percentile comparison."

        st.markdown("#### Feature Breakdown (Z-Scores)")
        feature_z_df = pd.DataFrame({
            'Feature': ['Electricity (Individual Equivalent)', 'Water', 'Transport', 'Company ESG'],
            'Z-Score': [electricity_z, water_z, transport_z, company_z] 
        })
        st.dataframe(feature_z_df, use_container_width=True)

        # New section for percentage contribution to score
        st.markdown("#### Percentage Contribution to Total Score")
        
        # Calculate weighted Z-scores for each component
        elec_weighted_z = electricity_z * weights["Electricity"]
        water_weighted_z = water_z * weights["Water"]
        transport_weighted_z = transport_z * weights["Commute"]
        company_weighted_z = company_z * weights["Company"]

        # Sum of absolute weighted Z-scores to normalize contributions
        total_absolute_weighted_z = (
            abs(elec_weighted_z) + 
            abs(water_weighted_z) + 
            abs(transport_weighted_z) + 
            abs(company_weighted_z)
        )

        if total_absolute_weighted_z > 0:
            elec_contrib_pct = (abs(elec_weighted_z) / total_absolute_weighted_z) * 100
            water_contrib_pct = (abs(water_weighted_z) / total_absolute_weighted_z) * 100
            transport_contrib_pct = (abs(transport_weighted_z) / total_absolute_weighted_z) * 100
            company_contrib_pct = (abs(company_weighted_z) / total_absolute_weighted_z) * 100
        else:
            elec_contrib_pct = water_contrib_pct = transport_contrib_pct = company_contrib_pct = 0.0
            st.info("No significant contributions to calculate percentages.")

        col_contrib1, col_contrib2, col_contrib3, col_contrib4 = st.columns(4)
        with col_contrib1:
            st.metric("Electricity Contribution", f"{elec_contrib_pct:.1f}%")
        with col_contrib2:
            st.metric("Water Contribution", f"{water_contrib_pct:.1f}%")
        with col_contrib3:
            st.metric("Transport Contribution", f"{transport_contrib_pct:.1f}%")
        with col_contrib4:
            st.metric("Company ESG Contribution", f"{company_contrib_pct:.1f}%")


        st.markdown("#### Your Electricity Consumption Analysis")
        if st.session_state.electricity_data is not None:
            electricity_data = st.session_state.electricity_data
            state_electricity_data = electricity_data[electricity_data['state_name'] == user_state]

            if not state_electricity_data.empty:
                state_avg_elec = state_electricity_data['per_person_qty_usage'].mean() # Use mean for the state
                state_std_elec = state_electricity_data['per_person_qty_usage'].std() # Get std for the state
                num_observations = len(state_electricity_data) # Get number of observations
                st.metric(f"Your Monthly Individual Electricity Usage ({user_state})", f"{individual_equivalent_consumption:.2f} kWh")
                st.metric(f"Average Monthly Individual Electricity Usage ({user_state})", f"{state_avg_elec:.2f} kWh")
                st.metric(f"Standard Deviation for {user_state}", f"{state_std_elec:.2f} kWh") # Display standard deviation
                st.metric(f"Number of Observations in {user_state}", f"{num_observations}") # Display observations

                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plotting distribution of averages from all states for context
                sns.histplot(electricity_data['per_person_qty_usage'], bins=15, kde=True, ax=ax, label="Distribution of State Averages")
                ax.axvline(individual_equivalent_consumption, color='red', linestyle='--', label='Your Usage')
                ax.axvline(state_avg_elec, color='green', linestyle='--', label=f'{user_state} Average')
                ax.set_title(f"Your Individual Electricity Usage Compared to {user_state} Average")
                ax.set_xlabel("Monthly Individual Consumption (kWh)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning(f"No specific electricity data available for {user_state} to compare.")
        else:
            st.info("Electricity data not loaded for comparison.")

        if user_electricity > 0:
            st.markdown("#### Individual Electricity Consumption")
            ind_col1, ind_col2 = st.columns(2)
            
            with ind_col1:
                st.metric("Total Household Usage", f"{user_electricity:.1f} kWh")
                
            with ind_col2:
                st.metric("Individual Equivalent Usage", f"{individual_equivalent_consumption:.1f} kWh",
                                f"Based on {household_size} people")

            electricity_data = st.session_state.electricity_data if 'electricity_data' in st.session_state else None
            if electricity_data is not None and 'hh_size' in electricity_data.columns:
                state_hh_data = electricity_data[electricity_data['state_name'] == user_state]
                
                if not state_hh_data.empty:
                    state_hh_avg = state_hh_data['hh_size'].mean() # Use mean for the state
                    
                    st.markdown("#### Household Size Comparison")
                    hh_col1, hh_col2 = st.columns(2)
                    
                    with hh_col1:
                        st.metric("Your Household Size", f"{household_size} people")
                    
                    with hh_col2:
                        comparison = "larger" if household_size > state_hh_avg else "smaller" if household_size < state_hh_avg else "same as"
                        st.metric(f"Average HH Size in {user_state}", 
                                    f"{state_hh_avg:.1f} people",
                                    f"Your household is {comparison} average")
                    
                    fig_states, ax_states = plt.subplots(figsize=(14, 8))
                    
                    state_hh_comparison = electricity_data.groupby('state_name')['hh_size'].mean().reset_index().sort_values('hh_size')
                    state_hh_comparison['is_user_state'] = state_hh_comparison['state_name'] == user_state
                    
                    bars = ax_states.bar(state_hh_comparison['state_name'], 
                                        state_hh_comparison['hh_size'],
                                        color=state_hh_comparison['is_user_state'].map({True: 'red', False: 'skyblue'}),
                                        alpha=0.8)
                    
                    ax_states.axhline(y=household_size, color='green', linestyle='--', linewidth=2,
                                    label=f'Your Household Size ({household_size} people)')
                    
                    user_state_avg_hh = state_hh_comparison[state_hh_comparison['is_user_state']]['hh_size'].values[0]
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
                    
            else:
                st.info("Household size data not available in the dataset for comparison")

            if user_cost > 0:
                def calculate_electricity_from_cost(cost, household_size_arg):
                    divisor = household_size_arg
                    return cost / divisor
                
                safe_hh_size_calc = household_size if household_size and household_size > 0 else 1
                calculated_electricity_from_cost = calculate_electricity_from_cost(user_cost, safe_hh_size_calc)
                
                st.markdown("### Cost-Based Electricity Usage Calculation")
                
                divisor = safe_hh_size_calc

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Monthly Cost", f"₹{user_cost:.2f}")
                with col2:
                    st.metric("Household Size", f"{safe_hh_size_calc} people")
                with col3:
                    st.metric("Individual Cost", f"{calculated_electricity_from_cost:.2f} ₹")
        
        st.markdown("#### Your Water Consumption Analysis")
        avg_water_liters_per_person = 130 
        st.metric("Your Water Usage (L/person/month)", f"{per_person_monthly_water:.2f} L")
        st.metric("Average Water Usage (L/person/month)", f"{avg_water_liters_per_person * 30:.2f} L")
        if per_person_monthly_water > avg_water_liters_per_person * 30:
            st.warning("Your water usage is higher than the average.")
        else:
            st.success("Your water usage is at or below the average.")

        st.markdown("#### Your Transport Carbon Footprint")
        if st.session_state.transport_carbon_results:
            total_monthly_km = st.session_state.transport_carbon_results["total_monthly_km"]
            emission_factor = st.session_state.transport_carbon_results["emission_factor"]
            people_count = st.session_state.transport_carbon_results["people_count"]
            vehicle_name = st.session_state.transport_carbon_results["vehicle_name"]
            vehicle_type = customer_data['Vehicle_Type']
            
            total_emissions_per_person = st.session_state.total_emissions 

            st.metric("Monthly CO₂ Emissions", f"{total_emissions_per_person:.1f} kg CO₂e")
            st.metric("Emission Factor", f"{emission_factor:.4f} kg CO2/km")
            st.metric("Vehicle Used", vehicle_name)

            # Use the calculated transport_z from calculate_sustainability_score
            st.metric("Transport Z-Score", f"{transport_z:.2f}")

            st.session_state.transport_carbon_results['total_emissions'] = total_emissions_per_person
            st.session_state.transport_carbon_results['z_score_emission'] = transport_z # Use the calculated Z-score
            st.session_state.transport_carbon_results['emission_category'] = "High" if transport_z > 0.5 else ("Average" if transport_z > -0.5 else "Low")
            
            if total_monthly_km > 0: # Check if any transport is used
                
                st.header("Emission Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Monthly Km", f"{total_monthly_km:.1f} km")
                
                with col2:
                    st.metric("Transport Z-Score", f"{transport_z:.2f}")
                
                with col3:
                    if transport_z < -0.5: # More negative is better
                        emission_category = "Low Emissions"
                        color = "🟢"
                    elif transport_z < 0.5:
                        emission_category = "Average Emissions"
                        color = "🟡"
                    else:
                        emission_category = "High Emissions"
                        color = "🔴"
                    
                    st.metric("Category", f"{color} {emission_category}")

            if st.session_state.get('calculated', False):
                total_kg = st.session_state.total_emissions
                
                alternatives = st.session_state.emissions_data
                
                st.info(f"""
                **Z-Score Interpretation for Transport:**
                - A more negative Z-score indicates lower emissions relative to the baseline (global mean).
                - Your transport Z-score is {abs(transport_z):.2f} standard deviations {'above' if transport_z > 0 else 'below'} the average.
                - A positive Z-score indicates higher emissions than average, while a negative Z-score indicates lower emissions.
                """)
                
                recommendations = []
                
                if total_kg > 100:
                    recommendations.append("Your carbon footprint from commuting is quite high. Consider switching to more sustainable transport options.")
                elif total_kg > 50:
                    recommendations.append("Your carbon footprint is moderate. There's room for improvement by considering more sustainable options.")
                else:
                    recommendations.append("Your carbon footprint is relatively low, but you can still make improvements.")
                
                if private_monthly_km_disp > 0 and public_monthly_km_disp == 0:
                    recommendations.append("You are relying solely on private transport. Consider incorporating public transport for some of your commutes to reduce emissions.")
                
                if private_monthly_km_disp > 0 and people_count == 1:
                    recommendations.append("Consider carpooling for your private transport to reduce emissions. Sharing your ride with others can significantly lower your per-person footprint.")
                
                if "petrol" in vehicle_name_disp.lower() or "diesel" in vehicle_name_disp.lower():
                    recommendations.append("If using private transport, consider switching to an electric vehicle to significantly reduce your carbon footprint.")
                
                if total_monthly_km > 0: # Only suggest public transport alternatives if there's actual travel
                    bus_emissions = (total_monthly_km * emission_factors["public_transport"]["bus"]["electric"]) / (people_count_disp if people_count_disp > 0 else 1)
                    metro_emissions = (total_monthly_km * emission_factors["public_transport"]["metro"]) / (people_count_disp if people_count_disp > 0 else 1)
                    
                    if total_kg > 2 * bus_emissions:
                        recommendations.append(f"Using an electric bus for your commute could reduce your emissions by approximately {(total_kg - bus_emissions) / total_kg * 100:.1f}%.")
                    
                    if total_kg > 2 * metro_emissions:
                        recommendations.append(f"Using metro for your commute could reduce your emissions by approximately {(total_kg - metro_emissions) / total_kg * 100:.1f}%.")
                
                st.session_state.recommendations = recommendations
                
                
                st.divider()
                st.header("Carbon Footprint Results")
                
                col1_cf, col2_cf = st.columns([1, 1])
                
                with col1_cf:
                    total_kg = st.session_state.total_emissions
                    
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
                    
                    avg_emissions = 200 
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

        else:
            st.info("No transport carbon footprint data available. Please complete the input on the '1. New Customer Assessment' page.")

        # Display customer score results using the calculated values
        st.markdown("---")
        st.markdown("### Customer Score Results")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'per_person_monthly_water' in locals() and household_size > 0:
                monthly_water = per_person_monthly_water * household_size
                per_person_monthly = per_person_monthly_water
                
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
            
            display_score = crisil_esg
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
                else:
                    if 'crisil_esg_score_input' in st.session_state:
                        display_score = st.session_state['crisil_esg_score_input']
                        score_source = "Custom Input"
            st.metric("Company Environment Score",f"{display_score:.2f}", help=f"Score {score_source}")

            st.metric("Water Usage", f"{per_person_monthly_water:.2f}") # Use the actual per person water
            st.metric("Public Transport (Km/month)", f"{customer_data.get('Public_Monthly_Km', 0):.2f}")
            st.metric("Private Transport (Km/month)", f"{customer_data.get('Private_Monthly_Km', 0):.2f}")
            
            st.metric("Z-Score", f"{total_z_score:.2f}")
            st.metric("Sustainability Score", f"{sustainability_score:.2f}")

            if 'scored_data' in st.session_state and 'Sustainability_Score' in st.session_state.scored_data:
                existing = st.session_state.scored_data["Sustainability_Score"]
                sust_rank = (existing > sustainability_score).sum() + 1
            else:
                sust_rank = 1

            st.metric("Sustainability Rank", f"{sust_rank}")


            if 'scored_data' in st.session_state and 'Sustainability_Score' in st.session_state.scored_data.columns:
                existing_sust = st.session_state.scored_data['Sustainability_Score']
                better_than = (existing_sust < sustainability_score).mean() * 100
                st.success(f"This customer performs better than **{better_than:.1f}%** of customers in the dataset (based on Z-Score)")
            else:
                st.info("No comparison data available in the dataset.")

            z_description = "above" if total_z_score > 0 else "below"
            st.info(f"Performance: **{abs(total_z_score):.2f} SD {z_description} mean**")
        
        with col2:
            pie_chart_data = {}

            if individual_equivalent_consumption > 0:
                pie_chart_data['Personal Electricity Usage (kWh)'] = individual_equivalent_consumption
            if per_person_monthly_water > 0:
                pie_chart_data['Water Usage (L/person/month)'] = per_person_monthly_water
            if crisil_esg > 0:
                pie_chart_data['Company Score'] = crisil_esg

            if customer_data.get('Private_Monthly_Km', 0) > 0:
                pie_chart_data['Private Transport (Km/month)'] = customer_data.get('Private_Monthly_Km', 0)
            if customer_data.get('Public_Monthly_Km', 0) > 0:
                pie_chart_data['Public Transport (Km/month)'] = customer_data.get('Public_Monthly_Km', 0)
            
            active_pie_features = {k: v for k, v in pie_chart_data.items() if v > 0}

            if active_pie_features:  
                total_sum_for_pie = sum(active_pie_features.values())
                if total_sum_for_pie == 0:
                    st.info("No active features with positive values to display for pie chart.")
                else:
                    pie_values = list(active_pie_features.values())
                    pie_labels = list(active_pie_features.keys())

                    explode = [0.1 if i % 2 == 0 else 0 for i in range(len(pie_labels))]
                    
                    fig_pie_weights, ax_pie_weights = plt.subplots(figsize=(6,6))
                    ax_pie_weights.pie(
                        pie_values,
                        labels=pie_labels,
                        autopct='%1.1f%%',
                        startangle=90,
                        explode=explode
                    )
                    ax_pie_weights.set_title('Contribution of Inputs to Customer Profile')
                    st.pyplot(fig_pie_weights)
                    
                    for feat, value in active_pie_features.items():
                        percentage_contribution = (value / total_sum_for_pie) * 100
                        st.info(f"{feat} contributes {percentage_contribution:.1f}% (Value: {value:.2f})")
            else:
                st.info("No active features with positive values to display for pie chart.")
        
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
                    else:
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
                        df_results['Normalized_Score'] = ((df_results['Env_Z_Score'] - min_z) / (max_z - min_z)) * 100 if (max_z - min_z) != 0 else 50

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
                                    company_z_display = (company_score - baseline_mean) / baseline_std if baseline_std > 0 else 0
                                    company_norm = company_data_row['Normalized_Score']

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Environment Score", f"{company_score:.2f}")
                                    with col2:
                                        st.metric("Z-Score", f"{company_z_display:.2f}",
                                                f"{company_z_display:.2f} SD from mean")
                                    with col3:
                                        st.metric("Normalized Score", f"{company_norm:.2f}/100")

                                    better_than = (df_results['Env_Z_Score'] < company_z_display).mean() * 100
                                    st.success(f"**{selected_company_for_comparison}** performs better than **{better_than:.1f}%** of companies in this segment (based on Z-Score)")
                                else:
                                    st.warning(f"Selected company '{selected_company_for_comparison}' not found in the dataset (Industry: {selected_esg_industry}, Employee Size: {selected_esg_emp_size}). Please ensure the company name matches exactly or check for case sensitivity.")
                            
                            else:
                                st.markdown("#### Benchmark Group Analysis")
                                st.info(f"No specific company selected. Displaying aggregate metrics for the **{len(df_filtered)}** companies in the **{selected_esg_industry}** sector with **{emp_range_text}** employees.")

                                median_score = df_filtered['Environment_Score'].median()
                                max_score = df_filtered['Environment_Score'].max()
                                min_score = df_filtered['Environment_Score'].min()

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Number of Companies", f"{len(df_filtered)}")
                                with col2:
                                    st.metric("Average Score", f"{baseline_mean:.2f}")
                                with col3:
                                    st.metric("Median Score", f"{median_score:.2f}")

                                col4, col5, col6 = st.columns(3)
                                with col4:
                                    st.metric("Standard Deviation", f"{baseline_std:.2f}")
                                with col5:
                                    st.metric("Highest Score", f"{max_score:.2f}")
                                with col6:
                                    st.metric("Lowest Score", f"{min_score:.2f}")

                                st.markdown("##### Score Distribution Box Plot")
                                fig_box = px.box(df_filtered, y="Environment_Score", 
                                               title=f"Score Distribution for {selected_esg_industry} ({emp_range_text} employees)",
                                               points="all", hover_data=['Company_Name'])
                                fig_box.update_layout(yaxis_title="Environment Score")
                                st.plotly_chart(fig_box, use_container_width=True)

                        else: 
                            custom_score = st.session_state.get('crisil_esg_score_input', 0.0)
                            
                            if baseline_std > 0:
                                custom_z_display = (custom_score - baseline_mean) / baseline_std
                            else:
                                custom_z_display = 0

                            custom_norm = ((custom_z_display - min_z) / (max_z - min_z)) * 100 if (max_z - min_z) != 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Environment Score", f"{custom_score:.2f}")
                            with col2:
                                st.metric("Z-Score", f"{custom_z_display:.2f}",
                                        f"{custom_z_display:.2f} SD from mean")
                            with col3:
                                st.metric("Normalized Score", f"{custom_norm:.2f}/100")

                            better_than = (df_results['Env_Z_Score'] < custom_z_display).mean() * 100
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

                            company_z_display = (company_score - overall_mean) / overall_std if overall_std > 0 else 0
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
                                z_description = "above" if company_z_display > 0 else "below"
                                st.metric("Z-Score",
                                        f"{company_z_display:.2f}",
                                        f"{abs(company_z_display):.2f} SD from mean")
                            with col2:
                                perf_description = "outperforms" if company_z_display > 0 else "underperforms"
                                st.metric("Performance", f"{perf_description} by {abs(company_z_display):.2f} SD")
                            with col3:
                                st.metric("Percentile", f"{percentile:.1f}%")

                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(company_data['Environment_Score'], kde=True, ax=ax)
                            ax.axvline(overall_mean, color='red', linestyle='--', label=f'Mean ({overall_mean:.2f})')
                            ax.axvline(company_score, color='blue', linestyle='-',
                                    label=f'{selected_company_for_comparison} ({company_score:.2f}, Z={company_z_display:.2f})')
                            ax.set_xlabel('Environment Score')
                            ax.set_title(f'Distribution of Environment Scores Across All Companies')
                            ax.legend()
                            st.pyplot(fig)

                            sector_data = company_data[company_data['Sector_classification'] == company_sector]
                            sector_mean = sector_data['Environment_Score'].mean()
                            sector_std = sector_data['Environment_Score'].std(ddof=1)
                            sector_z_display = (company_score - sector_mean) / sector_std if sector_std > 0 else 0
                            sector_percentile = (sector_data['Environment_Score'] < company_score).mean() * 100

                            st.markdown(f"### Comparison with {company_sector} Sector")
                            st.markdown(f"**Sector Environment Score:** {sector_mean:.2f} (std: {sector_std:.2f})")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                z_description = "above" if sector_z_display > 0 else "below"
                                st.metric("Sector Z-Score",
                                        f"{sector_z_display:.2f}",
                                        f"{abs(sector_z_display):.2f} SD {z_description} sector mean")
                            with col2:
                                perf_description = "outperforms" if sector_z_display > 0 else "underperforms"
                                st.metric("Sector Performance", f"{perf_description} by {abs(sector_z_display):.2f} SD")
                            with col3:
                                st.metric("Sector Percentile", f"{sector_percentile:.1f}%")

                            st.markdown(f"### All Performers in {company_sector} (Highest to Lowest)")

                            all_sector_companies = sector_data.copy()
                            all_sector_companies['Sector_Z_Score'] = (all_sector_companies['Environment_Score'] - sector_mean) / sector_std if sector_std > 0 else 0
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
                else:
                    custom_score = st.session_state.get('crisil_esg_score_input')

                    if isinstance(custom_score, (int, float)):
                        
                        if overall_std > 0:
                            custom_z_display = (custom_score - overall_mean) / overall_std
                        else:
                            custom_z_display = 0.0
                        
                        percentile = (company_data['Environment_Score'] < custom_score).mean() * 100

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            z_description = "above" if custom_z_display > 0 else "below"
                            st.metric("Z-Score",
                                    f"{custom_z_display:.2f}",
                                    f"{abs(custom_z_display):.2f} SD {z_description} mean")
                        with col2:
                            perf_description = "outperforms" if custom_z_display > 0 else "underperforms"
                            st.metric("Performance", f"{perf_description} by {abs(custom_z_display):.2f} SD")
                        with col3:
                            st.metric("Percentile", f"{percentile:.1f}%")

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(company_data['Environment_Score'], kde=True, ax=ax)
                        ax.axvline(overall_mean, color='red', linestyle='--', label=f'Mean ({overall_mean:.2f})')
                        
                        ax.axvline(custom_score, color='blue', linestyle='-',
                                label=f'Your Company ({custom_score:.2f}, Z={custom_z_display:.2f})')
                        
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
                        sector_z_display = (custom_score - sector_mean) / sector_std if sector_std > 0 else 0
                        sector_percentile = (sector_data['Environment_Score'] < custom_score).mean() * 100

                        st.markdown(f"### Comparison with {selected_sector_for_comparison} Sector")
                        st.markdown(f"**Sector Environment Score:** {sector_mean:.2f} (std: {sector_std:.2f})")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            z_description = "above" if sector_z_display > 0 else "below"
                            st.metric("Sector Z-Score",
                                    f"{sector_z_display:.2f}",
                                    f"{abs(sector_z_display):.2f} SD {z_description} sector mean")
                        with col2:
                            perf_description = "outperforms" if sector_z_display > 0 else "underperforms"
                            st.metric("Sector Performance", f"{perf_description} by {abs(sector_z_display):.2f} SD")
                        with col3:
                            st.metric("Sector Percentile", f"{sector_percentile:.1f}%")

                        all_sector_companies = sector_data.copy()
                        all_sector_companies['Sector_Z_Score'] = (all_sector_companies['Environment_Score'] - sector_mean) / sector_std if sector_std > 0 else 0
                        all_sector_companies = all_sector_companies.sort_values('Environment_Score', ascending=False)

                        st.markdown(f"### All Performers in {selected_sector_for_comparison} (Highest to Lowest)")

                        display_df = all_sector_companies[[
                            'Company_Name',
                            'Environment_Score',
                            'Sector_Z_Score',
                            'ESG_Rating',
                            'Total_Employees'
                        ]].reset_index(drop=True)

                            # Highlight the custom score if it falls within the displayed companies
                        if custom_score is not None:
                            # Create a dummy row for custom score to find its rank
                            dummy_row = pd.DataFrame([{
                                'Company_Name': 'Your Custom Score',
                                'Environment_Score': custom_score,
                                'Sector_Z_Score': sector_z_display,
                                'ESG_Rating': 'N/A',
                                'Total_Employees': 'N/A'
                            }])
                            display_df = pd.concat([display_df, dummy_row], ignore_index=True)
                            display_df = display_df.sort_values('Environment_Score', ascending=False).reset_index(drop=True)
                            selected_idx = display_df.index[display_df['Company_Name'] == 'Your Custom Score'].tolist()

                            styled_df = display_df.style.apply(
                                lambda x: ['background-color: lightskyblue' if i in selected_idx else ''
                                        for i in range(len(display_df))],
                                axis=0
                            )
                            st.dataframe(styled_df)
                        else:
                            st.dataframe(display_df)
        else:
            st.info("Company data unavailable for ESG comparison visualizations.")

        st.subheader("Generate Sustainability Report (PDF)")
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
                            current_customer_df = pd.DataFrame([st.session_state.single_customer_inputs])
                            current_customer_df['Sustainability_Score'] = st.session_state.calculated_single_customer_score
                            current_customer_df['Electricity'] = st.session_state.single_customer_inputs['Electricity']
                            current_customer_df['Water'] = st.session_state.single_customer_inputs['Water_Monthly_Total']
                            current_customer_df['Crisil_ESG_Score'] = st.session_state.single_customer_inputs['Crisil_ESG_Score']
                            current_customer_df['Private_Monthly_Km'] = st.session_state.single_customer_inputs['Private_Monthly_Km']
                            current_customer_df['Public_Monthly_Km'] = st.session_state.single_customer_inputs['Public_Monthly_Km']

                            scored_df = current_customer_df

                        pdf_file = generate_pdf(
                            scored_df,
                            pd.DataFrame([st.session_state.single_customer_inputs]),
                            st.session_state.recommendations,
                            st.session_state.transport_carbon_results
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
                emission_factors_transport = emission_factors

                test_with_scores = []
                # MPCE related variables are kept for bulk parts of the script if needed
                mpce_ranges = ["₹1-1,000", "₹1,000-5,000", "₹5,000-10,000", "₹10,000-25,000", "₹25,000+"]
                mpce_range_values = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 25000), (25000, float('inf'))]

                global_electricity_mean = 0
                global_electricity_std = 1 
                if st.session_state.electricity_data is not None and not st.session_state.electricity_data.empty:
                    global_electricity_mean = st.session_state.electricity_data['cons_total_qty'].mean()
                    global_electricity_std = st.session_state.electricity_data['cons_total_qty'].std()


                for idx, row in test_df.iterrows():
                    st.write(f"--- DEBUG: Processing Bulk Row {idx} ---")
                    st.write(f"DEBUG: Raw row data: {row.to_dict()}")
                    try:
                        total_electricity_kwh = float(row.get("Electricity", 0.0)) if row.get("Electricity") is not None else 0.0
                        household_count = int(row.get("People in the household", 1)) if not pd.isna(row.get("People in the household")) else 1
                        user_state_csv = str(row.get("State", INDIAN_STATES_UTS[0])).strip() # Get state from CSV
                        
                        divisor = household_count 
                        
                        individual_electricity_kwh = total_electricity_kwh / divisor if divisor > 0 else total_electricity_kwh
                        
                        # FIX: Corrected column names for Water_Usage_Type and Water_Value
                        water_input_type = str(row.get("Water_Usage_Type", "monthly")).strip().lower()
                        water_value_raw = row.get("Water_Value")
                        
                        try:
                            water_value = float(water_value_raw)
                        except (ValueError, TypeError):
                            water_value = 0.0 # Default to 0 if conversion fails (e.g., empty string, None, non-numeric)

                        st.write(f"DEBUG: Row {idx} - Raw Water Value: {water_value_raw}, Parsed Water Value: {water_value}")
                        st.write(f"DEBUG: Row {idx} - Water Input Type: {water_input_type}, Household Count: {household_count}")

                        # FIX: Corrected water calculation for 'daily' type in bulk processing
                        if water_input_type == "daily usage": # Ensure this matches the exact string from CSV if applicable
                            total_water_liters = water_value * 30 * household_count # Added * 30
                        else:
                            total_water_liters = water_value * household_count
                        
                        st.write(f"DEBUG: Row {idx} - Total Water Liters: {total_water_liters}")

                        per_person_monthly_water_calc = total_water_liters / (household_count if household_count > 0 else 1)
                        st.write(f"DEBUG: Row {idx} - Per Person Monthly Water Calc: {per_person_monthly_water_calc}")


                        # Re-introduce logic to calculate total_monthly_km from CSV
                        distance_csv = float(row.get("Km_per_month", 0.0)) if not pd.isna(row.get("Km_per_month")) else 0.0
                        # The CSV template has Km_per_month directly, not distance, days, weeks
                        total_monthly_km_csv = distance_csv # Use Km_per_month directly from CSV

                        transport_category_csv = str(row.get("Vehicle_Type", "Private Transport")).strip() # Use Vehicle_Type from CSV
                        
                        # Ensure Private_Monthly_Km and Public_Monthly_Km are correctly read as floats
                        private_monthly_km_csv = pd.to_numeric(row.get("Private_Monthly_Km", 0.0), errors='coerce')
                        public_monthly_km_csv = pd.to_numeric(row.get("Public_Monthly_Km", 0.0), errors='coerce') # FIX: Read Public_Monthly_Km

                        # New: Read private and public trips per day from CSV for bulk processing
                        private_trips_per_day_csv = int(row.get("Private_Trips_Per_Day", 0)) if not pd.isna(row.get("Private_Trips_Per_Day")) else 0
                        public_trips_per_day_csv = int(row.get("Public_Trips_Per_Day", 0)) if not pd.isna(row.get("Public_Trips_Per_Day")) else 0

                        # Replace NaN with 0.0 if coercion failed
                        private_monthly_km_csv = private_monthly_km_csv if not pd.isna(private_monthly_km_csv) else 0.0
                        public_monthly_km_csv = public_monthly_km_csv if not pd.isna(public_monthly_km_csv) else 0.0

                        crisil_esg_csv = float(row.get("Crisil_ESG_Score", 0.0)) if not pd.isna(row.get("Crisil_ESG_Score")) else 0.0
                        company_sector_csv = str(row.get("Sector_classification", "")) if not pd.isna(row.get("Sector_classification")) else "" # Use Sector_classification from CSV
                        num_employees_csv = int(row.get("Total_Employees", None)) if not pd.isna(row.get("Total_Employees")) else None # Use Total_Employees from CSV

                        commute_emission_calc = 0.0
                        
                        # Initialize ratios to 0.0 to prevent UndefinedVariable error
                        private_ratio_csv = 0.0
                        public_ratio_csv = 0.0

                        # Logic to calculate emission_factor_calc for bulk, similar to single customer
                        # For bulk, we'll use the Transport_Emission_Factor directly if available, otherwise calculate a default
                        transport_emission_factor_from_csv = float(row.get("Transport_Emission_Factor", 0.0)) if not pd.isna(row.get("Transport_Emission_Factor")) else 0.0
                        if transport_emission_factor_from_csv > 0:
                            commute_emission_calc = transport_emission_factor_from_csv
                        else:
                            # Fallback calculation if Transport_Emission_Factor is not provided in CSV
                            if transport_category_csv == "Private Transport":
                                # Default to a common private vehicle like Scooter (petrol, 150cc)
                                commute_emission_calc = emission_factors["two_wheeler"]["Scooter"]["petrol"]["min"] 
                            elif transport_category_csv == "Public Transport":
                                # Default to Metro as a common public transport
                                commute_emission_calc = emission_factors["public_transport"]["metro"]
                            elif transport_category_csv == "Both Private and Public":
                                # Recalculate ratios for combined transport based on new trip inputs from CSV
                                total_trips_per_day_csv = private_trips_per_day_csv + public_trips_per_day_csv
                                if total_trips_per_day_csv > 0:
                                    private_ratio_csv = private_trips_per_day_csv / total_trips_per_day_csv
                                    public_ratio_csv = public_trips_per_day_csv / total_trips_per_day_csv
                                
                                # Use default emission factors for combined if not specified in CSV
                                private_ef_for_combined_csv = emission_factors["two_wheeler"]["Scooter"]["petrol"]["min"] # Default private
                                public_ef_for_combined_csv = emission_factors["public_transport"]["metro"] # Default public

                                commute_emission_calc = (private_ef_for_combined_csv / (1 if private_trips_per_day_csv > 0 else 1)) * private_ratio_csv + \
                                                        (public_ef_for_combined_csv / (1 if public_trips_per_day_csv > 0 else 1)) * public_ratio_csv
                            else:
                                commute_emission_calc = 0.0 # No transport or unknown category

                        # Prepare customer data dict for the unified function
                        customer_data_for_bulk = {
                            'State': user_state_csv, # Pass the state from the CSV row
                            'Electricity': total_electricity_kwh,
                            'People in the household': household_count,
                            'Crisil_ESG_Score': crisil_esg_csv,
                            'Water_Per_Person_Monthly': per_person_monthly_water_calc,
                            'Km_per_month': total_monthly_km_csv, # Total for display
                            'Transport_Emission_Factor': commute_emission_calc, # Overall for display
                            'Transport_People_Count': int(row.get("Transport_People_Count", 1)) if not pd.isna(row.get("Transport_People_Count")) else 1, 
                            'Private_Monthly_Km': private_monthly_km_csv, # Pass calculated private km
                            'Public_Monthly_Km': public_monthly_km_csv # Pass calculated public km
                        }

                        # Call the unified sustainability score calculation function for each row
                        bulk_sustainability_results = calculate_sustainability_score(
                            customer_data_for_bulk,
                            st.session_state.company_data,
                            st.session_state.electricity_data,
                            st.session_state.weights
                            # Removed emission_mean, emission_std from arguments
                        )
                        st.write(f"DEBUG: Row {idx} - Bulk Sustainability Results: {bulk_sustainability_results}")


                        result = row.to_dict()
                        result.update({
                            "Electricity_Z": bulk_sustainability_results['electricity_z'],
                            "Water_Z": bulk_sustainability_results['water_z'],
                            "Commute_Z": bulk_sustainability_results['transport_z'], # This is now the combined transport Z-score
                            "Company_Z": bulk_sustainability_results['company_z'],
                            "Z_Total": bulk_sustainability_results['total_z_score'],
                            "Sustainability_Score": bulk_sustainability_results['sustainability_score'],
                            "Private_Trips_Per_Day": private_trips_per_day_csv, # New
                            "Public_Trips_Per_Day": public_trips_per_day_csv # New
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
                
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    median_score = results_df['Sustainability_Score'].median()
                    st.metric("Median Score", f"{median_score:.1f}")
                with col6:
                    std_score = results_df['Sustainability_Score'].std()
                    st.metric("Standard Deviation", f"{std_score:.1f}")
                with col7:
                    excellent_count = len(results_df[results_df['Sustainability_Score'] >= 800])
                    excellent_pct = (excellent_count / total_customers * 100) if total_customers > 0 else 0
                    st.metric("Excellent Scores (>=800)", f"{excellent_count} ({excellent_pct:.1f}%)")
                with col8:
                    poor_count = len(results_df[results_df['Sustainability_Score'] < 400])
                    poor_pct = (poor_count / total_customers * 100) if total_customers > 0 else 0
                    st.metric("Poor Scores (<400)", f"{poor_count} ({poor_pct:.1f}%)")

                col9, col10, col11, col12 = st.columns(4)
                with col9:
                    avg_electricity = pd.to_numeric(results_df['Electricity'], errors='coerce').mean()
                    st.metric("Avg Electricity (kWh)", f"{avg_electricity:.1f}")
                with col10:
                    if 'Total_Employees' in results_df.columns: # Use Total_Employees from CSV
                        avg_employees = pd.to_numeric(results_df['Total_Employees'], errors='coerce').mean()
                        st.metric("Avg Employees", f"{avg_employees:.0f}")
                    else:
                        st.info("No 'Total_Employees' data in bulk upload.")
                with col11:
                    avg_esg = pd.to_numeric(results_df['Crisil_ESG_Score'], errors='coerce').mean()
                    st.metric("Avg ESG Score", f"{avg_esg:.1f}")
                with col12:
                    if 'Km_per_month' in results_df.columns:
                        avg_km = pd.to_numeric(results_df['Km_per_month'], errors='coerce').mean()
                        st.metric("Avg Km/Month", f"{avg_km:.0f}")
                    else:
                        st.info("No 'Km_per_month' data in bulk upload.")

                col13, col14, col15, col16 = st.columns(4)
                with col13:
                    if 'Vehicle_Name' in results_df.columns: # Check for Vehicle_Name to infer electric vehicles
                        electric_vehicles_count = len(results_df[results_df['Vehicle_Name'].astype(str).str.contains('electric', case=False, na=False)])
                        electric_pct = (electric_vehicles_count / total_customers * 100) if total_customers > 0 else 0
                        st.metric("Electric Vehicles", f"{electric_vehicles_count} ({electric_pct:.1f}%)")
                    else:
                        st.info("No 'Vehicle_Name' data to infer electric vehicles in bulk upload.")
                with col14:
                    # Removed 'Rural/Urban' data display as the option is removed
                    st.info("Rural/Urban data removed from bulk upload.")
                with col15:
                    esg_numeric = pd.to_numeric(results_df['Crisil_ESG_Score'], errors='coerce')
                    high_esg = len(results_df[esg_numeric >= 80])
                    high_esg_pct = (high_esg / total_customers * 100) if total_customers > 0 else 0
                    st.metric("High ESG (≥80)", f"{high_esg} ({high_esg_pct:.1f}%)")
                with col16:
                    # Updated logic to count public transport users based on Public_Trips_Per_Day
                    if 'Public_Trips_Per_Day' in results_df.columns:
                        public_trips_numeric = pd.to_numeric(results_df['Public_Trips_Per_Day'], errors='coerce')
                        public_transport_users = len(results_df[public_trips_numeric > 0])
                        public_pct = (public_transport_users / total_customers * 100) if total_customers > 0 else 0
                        st.metric("Public Transport Users", f"{public_transport_users} ({public_pct:.1f}%)")
                    else:
                        st.info("No 'Public_Trips_Per_Day' data in bulk upload.")

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

def main():
    st.sidebar.title("Navigation")
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "1. New Customer Assessment"

    page_options = [
        "1. New Customer Assessment",
        "2. Weighted Score Settings",
        "3. Evaluation Results & Reporting"
    ]
    if st.session_state.current_page in page_options:
        default_index = page_options.index(st.session_state.current_page)
    else:
        default_index = 0
        st.session_state.current_page = page_options[0]
    page_selection = st.sidebar.radio("Go to", page_options, index=default_index)

    if page_selection != st.session_state.current_page:
        st.session_state.current_page = page_selection

    st.title("Resource Sustainability Dashboard")
    st.markdown("Analyze your resource consumption and sustainability score")
    st.markdown("---")

    if st.session_state.current_page == "1. New Customer Assessment":
        page_test_customer()
    elif st.session_state.current_page == "2. Weighted Score Settings":
        page_evaluation()
    elif st.session_state.current_page == "3. Evaluation Results & Reporting":
        page_reporting()

if __name__ == "__main__":
    main()
