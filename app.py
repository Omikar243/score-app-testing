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
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import branca.colormap as cm

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
            # Add household size metrics
            if 'hh_size' in filtered_data.columns:
                avg_hh_size = filtered_data['hh_size'].mean()
                per_capita_electricity = filtered_data['qty_usage_in_1month'] / filtered_data['hh_size']
                avg_per_capita_electricity = per_capita_electricity.mean()
                st.metric("Average Electricity", f"{avg_electricity:.2f} kWh")
                st.metric("Average MPCE", f"₹{avg_mpce:.2f}")
                st.metric("Average Household Size", f"{avg_hh_size:.1f} people")
                st.metric("Per Capita Electricity", f"{avg_per_capita_electricity:.2f} kWh/person")
            else:
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

# Add household size distribution graph
if not filtered_data.empty and 'hh_size' in filtered_data.columns:
    st.markdown(f"## Household Size Distribution - {selected_state}, {selected_sector_name}")
    fig_hh, ax_hh = plt.subplots(figsize=(10, 6))
    hh_size_counts = filtered_data['hh_size'].value_counts().sort_index()
    ax_hh.bar(hh_size_counts.index, hh_size_counts.values, alpha=0.7, edgecolor='black')
    ax_hh.set_title(f"Household Size Distribution - {selected_state}, {selected_sector_name}")
    ax_hh.set_xlabel("Number of People in Household")
    ax_hh.set_ylabel("Number of Households")
    ax_hh.grid(True, alpha=0.3)
    st.pyplot(fig_hh)

# New addition: Overall electricity consumption comparison across states
st.markdown("## Overall Electricity Consumption Data")
sector_for_comparison = st.radio("Select Sector for Comparison", ['Rural', 'Urban', 'Both'], key="comparison_sector")

if sector_for_comparison == 'Both':
    # Group by state and calculate mean for both sectors combined
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

fig = px.bar(state_avg, x='state_name', y='qty_usage_in_1month',
             title=chart_title,
             labels={
                 'state_name': 'State/UT',
                 'qty_usage_in_1month': 'Average Electricity (kWh/month)'
             })

fig.update_layout(
    xaxis_tickangle=90,
    xaxis_title='State/UT',
    yaxis_title='Average Electricity (kWh/month)'
)

fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')

st.plotly_chart(fig)

if 'hh_size' in electricity_data.columns:
    st.markdown("## Household Size Comparison Across States")
    
    hh_tab1, hh_tab2 = st.tabs(["Average Household Size", "Per Capita Electricity"])
    
    with hh_tab1:
        state_hh_sorted = state_avg.sort_values('hh_size', ascending=False) if 'hh_size' in state_avg.columns else state_avg
        
        fig_hh = px.bar(state_hh_sorted, x='state_name', y='hh_size',
                        title=f"Average Household Size by State ({sector_for_comparison})",
                        labels={
                            'state_name': 'State/UT',
                            'hh_size': 'Average Household Size (people)'
                        },
                        color='hh_size',
                        color_continuous_scale='Blues')

        fig_hh.update_layout(
            xaxis_tickangle=90,
            xaxis_title='State/UT',
            yaxis_title='Average Household Size (people)'
        )

        fig_hh.update_traces(texttemplate='%{y:.1f}', textposition='outside')

        st.plotly_chart(fig_hh)
    
    with hh_tab2:
        if 'per_capita_usage' in state_avg.columns:
            state_pc_sorted = state_avg.sort_values('per_capita_usage', ascending=False)
            
            fig_pc = px.bar(state_pc_sorted, x='state_name', y='per_capita_usage',
                            title=f"Per Capita Electricity Consumption by State ({sector_for_comparison})",
                            labels={
                                'state_name': 'State/UT',
                                'per_capita_usage': 'Per Capita Electricity (kWh/person/month)'
                            },
                            color='per_capita_usage',
                            color_continuous_scale='Viridis')

            fig_pc.update_layout(
                xaxis_tickangle=90,
                xaxis_title='State/UT',
                yaxis_title='Per Capita Electricity (kWh/person/month)'
            )

            # Add value labels on top of bars
            fig_pc.update_traces(texttemplate='%{y:.2f}', textposition='outside')

            st.plotly_chart(fig_pc)

st.markdown("## Total Electricity Usage by Sectors and States Across India")

sector_state_tab1, sector_state_tab2 = st.tabs(["By Sector", "By State"])

sector_total = electricity_data.groupby('sector')['qty_usage_in_1month'].sum().reset_index()
sector_total['sector_name'] = sector_total['sector'].map({1: 'Rural', 2: 'Urban'})

fig_pie = px.pie(
    sector_total, 
    values='qty_usage_in_1month', 
    names='sector_name',
    title="Total Electricity Usage Distribution by Sector",
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_pie.update_traces(
    textinfo='percent', 
    textposition='inside',
    hovertemplate='<b>%{label}</b><br>Total Electricity: %{value:.1f} kWh<br>Percentage: %{percent}'
)

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
    state_total = electricity_data.groupby('state_name')['qty_usage_in_1month'].sum().reset_index()
    
    state_total = state_total.sort_values('qty_usage_in_1month', ascending=False)
    
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
    
    fig_state.update_traces(
        texttemplate='%{label}<br>%{value:.1f} kWh<br>%{percentRoot:.1%}',
        hovertemplate='<b>%{label}</b><br>Total: %{value:.1f} kWh<br>Percentage: %{percentRoot:.1%}'
    )
    
    st.plotly_chart(fig_state)
    
    st.subheader("State-wise Electricity Usage - Bar Chart")
    
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
    
    fig_bar.update_traces(
        texttemplate='%{x:.1f}', 
        textposition='outside'
    )
    
    num_states = len(state_total)
    chart_height = max(500, 20 * num_states)  
    
    fig_bar.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Total Electricity Usage (kWh)',
        yaxis_title='State/UT',
        height=chart_height,
        margin=dict(l=200, r=100, t=50, b=50)  
    )
    
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
        filtered_weights = {
        "Electricity_State": float(w_elec_location),
        "Electricity_MPCE": float(w_elec_mpce),
        "Electricity": float(w_elec_location) + float(w_elec_mpce),  # Add total
        "Public_Transport": w_public,
        "Private_Transport": w_private,
        "Water": w_water,
        "Company": w_company
    }
        
        st.session_state.weights = filtered_weights
        
        st.session_state.sub_weights = {
            "electricity": {
                "location": float(w_elec_location),
                "mpce": float(w_elec_mpce)
            },
            "commute": {
                "public": w_public,
                "private": w_private
            },
            "water": {
                "water": w_water
            },
            "company": {
                "company": w_company
            }
        }
        st.success("Weighted score generated successfully!")

    st.markdown("---\n# Test New Customer")
test_mode = st.radio("Input Mode", ["Manual Entry", "CSV Upload"], key="test_mode")

features = ["Electricity", "Water (Monthly or Daily)", "Public_Transport", "Private_Transport", "Industry (Sector_classification)", "Number of Employees"]

if test_mode == "CSV Upload":
        st.markdown("### Upload Test Data")   
        
        # Always show template downloads
        st.markdown("#### Download Template")
        
# 1. Define the required columns for the CSV template (exactly matching the manual form):
        required_cols = [
            "Industry (Sector_classification)",
            "Number of Employees",
            "Electricity",
            "People in the household",
            "Water (Monthly or Daily)",
            "Water (Monthly or Daily) Value",
            "State (Drop Down)",
            "Rural/Urban",
            "Vehicle (Drop Down)",
            "Category (Drop Down)",
            "Engine (Drop Down)",
            "Km_per_month",
            "Crisil_ESG_Score"
        ]

        template_data = {
            "Industry (Sector_classification)": [
                "Manufacturing",
                "IT Services",
                "Healthcare",
                "Automotive",
                "FMCG"
            ],
            "Number of Employees": [
                50,
                200,
                75,
                120,
                300
            ],
            "Electricity": [
                350.0,
                1200.0,
                800.0,
                450.0,
                1600.0
            ],  # Monthly kWh
            "People in the household": [
                4,
                2,
                5,
                3,
                6
            ],
            "Water (Monthly or Daily)": [
                "monthly",
                "daily",
                "monthly",
                "daily",
                "monthly"
            ],
            "Water (Monthly or Daily) Value": [
                2000.0,
                150.0,
                4500.0,
                100.0,
                5200.0
            ],  # If monthly: liters/month total; if daily: liters/person/day
            "State (Drop Down)": [
                "Karnataka",
                "Maharashtra",
                "Delhi",
                "Tamil Nadu",
                "Gujarat"
            ],
            "Rural/Urban": [
                "urban",
                "rural",
                "urban",
                "urban",
                "rural"
            ],
            "Vehicle (Drop Down)": [
                "two_wheeler",
                "four_wheeler",
                "three_wheeler",
                "public_transport",
                "four_wheeler"
            ],
            "Category (Drop Down)": [
                "Scooter",      # when two_wheeler
                "sedan",        # when four_wheeler
                "petrol",       # when three_wheeler (will treat 'petrol' as a category under three_wheeler)
                "bus",          # when public_transport
                "compact_suv"   # when four_wheeler
            ],
            "Engine (Drop Down)": [
                "petrol",
                "electric",
                "cng",
                "electric",
                "diesel"
            ],
            "Km_per_month": [
                600.0,
                1500.0,
                500.0,
                2000.0,
                1200.0
            ],
            "Crisil_ESG_Score": [
                70.0,
                85.0,
                60.0,
                90.0,
                75.0
            ]
        }

        # 3. Build a DataFrame from template_data and offer it for download:
        template_df = pd.DataFrame(template_data)
        tmpl_csv = template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Template (with Examples)",
            data=tmpl_csv,
            file_name="customer_template.csv",
            mime="text/csv"
        )

        # 4. File uploader for test CSV
        up_test = st.file_uploader("Upload Test CSV", type="csv")
        if up_test:
            test_df = pd.read_csv(up_test)
            st.markdown("#### Uploaded Test Data")
            st.dataframe(test_df)

            if st.button("Process Test Batch"):
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

                # ────────────────────────────────────────────────────────────────────────
                test_with_scores = []

                for _, row in test_df.iterrows():
                    industry = str(row["Industry (Sector_classification)"]).strip()
                    num_employees = int(row["Number of Employees"]) if not pd.isna(row["Number of Employees"]) else 0

                    electricity_kwh = float(row["Electricity"]) if not pd.isna(row["Electricity"]) else 0.0

                    household_count = int(row["People in the household"]) if not pd.isna(row["People in the household"]) else 1

                    water_input = str(row["Water (Monthly or Daily)"]).strip().lower()
                    water_value  = float(row["Water (Monthly or Daily) Value"]) if not pd.isna(row["Water (Monthly or Daily) Value"]) else 0.0
                    if water_input == "daily":
                        # Assume 'water_value' is liters/person/day
                        total_water_liters = water_value * 30 * household_count
                    else:
                        # Assume 'water_value' is total liters/month
                        total_water_liters = water_value * household_count

                    state_ut = str(row["State (Drop Down)"]).strip().lower()
                    rural_urban = str(row["Rural/Urban"]).strip().lower()

                    vehicle_type = str(row["Vehicle (Drop Down)"]).strip()
                    category    = str(row["Category (Drop Down)"]).strip()
                    engine_fuel = str(row["Engine (Drop Down)"]).strip()
                    km_per_month = float(row["Km_per_month"]) if not pd.isna(row["Km_per_month"]) else 0.0

                    crisil_esg = float(row["Crisil_ESG_Score"]) if not pd.isna(row["Crisil_ESG_Score"]) else 0.0

                    if vehicle_type == "two_wheeler":
                        ef_min = emission_factors["two_wheeler"][category][engine_fuel]["min"]
                        ef_max = emission_factors["two_wheeler"][category][engine_fuel]["max"]
                        chosen_ef = (ef_min + ef_max) / 2
                        commute_emission = chosen_ef * (km_per_month / 1000.0)

                    elif vehicle_type == "three_wheeler":
                        ef_min = emission_factors["three_wheeler"][engine_fuel]["min"]
                        ef_max = emission_factors["three_wheeler"][engine_fuel]["max"]
                        chosen_ef = (ef_min + ef_max) / 2
                        commute_emission = chosen_ef * (km_per_month / 1000.0)

                    elif vehicle_type == "four_wheeler":
                        base   = emission_factors["four_wheeler"][category][engine_fuel]["base"]
                        uplift = emission_factors["four_wheeler"][category][engine_fuel]["uplift"]
                        chosen_ef = base * uplift
                        commute_emission = chosen_ef * (km_per_month / 1000.0)

                    else:  # public_transport
                        if category in emission_factors["public_transport"]:
                            factor = emission_factors["public_transport"][category]
                            if category == "taxi":
                                subcat = "small"  # default to small if not specified
                                sub_factor = emission_factors["public_transport"]["taxi"][subcat][engine_fuel]
                                commute_emission = sub_factor["base"] * sub_factor["uplift"] * (km_per_month / 1000.0)
                            elif category == "bus":
                                commute_emission = factor[engine_fuel] * (km_per_month / 1000.0)
                            elif category == "metro":
                                commute_emission = factor * (km_per_month / 1000.0)
                            else:
                                commute_emission = 0.0

                    elec_z = max(0.0, (electricity_kwh - 300.0) / 300.0)

                    water_z = max(0.0, (total_water_liters - 3900.0) / 3900.0)

                    if vehicle_type in ["two_wheeler", "three_wheeler", "four_wheeler"]:
                        commute_z = commute_emission / 1000.0
                    else:
                        commute_z = -min(1.0, commute_emission / 500.0)

                    company_z = -(crisil_esg / 100.0)

                    # 4) Combine these z‐values with the exact same weights your manual branch uses.
                    #    Example weights (adjust if your manual code differs):
                    #      water_z           * 0.25
                    #      elec_z            * 0.25
                    #      commute_z         * 0.25
                    #      company_z         * 0.10
                    #      [public_transport_z * 0.15]  <- if your manual branch treated public separately
                    z_total = (
                        water_z * 0.25
                        + elec_z * 0.25
                        + commute_z * 0.25
                        + company_z * 0.10
                    )
                    # If your manual branch also had a distinct public_transport_z term:
                    # public_transport_z = -min(1.0, (commute_emission / 500.0))
                    # z_total += public_transport_z * 0.15

                    sust_score = 500.0 * (1.0 - np.tanh(z_total / 2.5))

                    result = row.to_dict()
                    result.update({
                        "Electricity_Z":          elec_z,
                        "Water_Z":                water_z,
                        "Commute_Z":              commute_z,
                        "Company_Z":              company_z,
                        "Z_Total":                z_total,
                        "Sustainability_Score":   sust_score
                    })
                    test_with_scores.append(result)

                results_df = pd.DataFrame(test_with_scores)
                st.markdown("### Test Results")
                st.dataframe(results_df)

                st.markdown("### Comprehensive Metrics")
                
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
                    excellent_count = len(results_df[results_df['Sustainability_Score'] >= 400])
                    excellent_pct = (excellent_count / total_customers * 100) if total_customers > 0 else 0
                    st.metric("Excellent Scores", f"{excellent_count} ({excellent_pct:.1f}%)")
                
                with col8:
                    poor_count = len(results_df[results_df['Sustainability_Score'] < 200])
                    poor_pct = (poor_count / total_customers * 100) if total_customers > 0 else 0
                    st.metric("Poor Scores", f"{poor_count} ({poor_pct:.1f}%)")

                col9, col10, col11, col12 = st.columns(4)
                
                with col9:
                    avg_electricity = results_df['Electricity'].mean()
                    st.metric("Avg Electricity (kWh)", f"{avg_electricity:.1f}")
                
                with col10:
                    avg_employees = results_df['Number of Employees'].mean()
                    st.metric("Avg Employees", f"{avg_employees:.0f}")
                
                with col11:
                    avg_esg = results_df['Crisil_ESG_Score'].mean()
                    st.metric("Avg ESG Score", f"{avg_esg:.1f}")
                
                with col12:
                    avg_km = results_df['Km_per_month'].mean()
                    st.metric("Avg Km/Month", f"{avg_km:.0f}")

                col13, col14, col15, col16 = st.columns(4)
                
                with col13:
                    electric_vehicles = len(results_df[results_df['Engine (Drop Down)'] == 'electric'])
                    electric_pct = (electric_vehicles / total_customers * 100) if total_customers > 0 else 0
                    st.metric("Electric Vehicles", f"{electric_vehicles} ({electric_pct:.1f}%)")
                
                with col14:
                    urban_customers = len(results_df[results_df['Rural/Urban'] == 'urban'])
                    urban_pct = (urban_customers / total_customers * 100) if total_customers > 0 else 0
                    st.metric("Urban Customers", f"{urban_customers} ({urban_pct:.1f}%)")
                
                with col15:
                    high_esg = len(results_df[results_df['Crisil_ESG_Score'] >= 80])
                    high_esg_pct = (high_esg / total_customers * 100) if total_customers > 0 else 0
                    st.metric("High ESG (≥80)", f"{high_esg} ({high_esg_pct:.1f}%)")
                
                with col16:
                    public_transport_users = len(results_df[results_df['Vehicle (Drop Down)'] == 'public_transport'])
                    public_pct = (public_transport_users / total_customers * 100) if total_customers > 0 else 0
                    st.metric("Public Transport", f"{public_transport_users} ({public_pct:.1f}%)")

                st.markdown("### Z-Score Analytics")
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
    st.markdown("### Enter Customer Details")
# CRISIL ESG - Environmental Scoring Only
    st.markdown("## CRISIL ESG - Environmental Scoring Only")

    if company_data is not None:
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Employee Range Analysis", "Company-Only Analysis"],
            key="crisil_analysis_type"
        )
        
        if analysis_type == "Employee Range Analysis":
            industry_opts = company_data['Sector_classification'].dropna().unique()
            industry = st.selectbox("Select Industry Sector", industry_opts, key="employee_range_industry")

            df_sector = company_data[company_data['Sector_classification'] == industry]
            
            employee_size = st.selectbox(
                "Select Employee Size Category",
                ["Small (<5,000)", "Medium (5,000 to 15,000)", "Large (>15,000)"],
                key="employee_size_category"
            )
            
            if employee_size == "Small (<5,000)":
                df_filtered = df_sector[df_sector['Total_Employees'] < 5000]
                emp_range_text = "less than 5,000"
            elif employee_size == "Medium (5,000 to 15,000)":
                df_filtered = df_sector[(df_sector['Total_Employees'] >= 5000) & (df_sector['Total_Employees'] <= 15000)]
                emp_range_text = "5,000 to 15,000"
            else:  # Large (>15,000)
                df_filtered = df_sector[df_sector['Total_Employees'] > 15000]
                emp_range_text = "more than 15,000"
            
            if len(df_filtered) > 0:
                baseline_mean = df_filtered['Environment_Score'].mean()
                baseline_std = df_filtered['Environment_Score'].std(ddof=1)
                
                st.markdown(f"### Companies in {industry} sector with {emp_range_text} employees")
                st.markdown(f"**Baseline Environment Score:** {baseline_mean:.2f} (std: {baseline_std:.2f})")
                
                df_results = df_filtered.copy()
                df_results['Env_Z_Score'] = (df_results['Environment_Score'] - baseline_mean) / baseline_std
                
                min_z = df_results['Env_Z_Score'].min()
                max_z = df_results['Env_Z_Score'].max()
                df_results['Normalized_Score'] = ((df_results['Env_Z_Score'] - min_z) / (max_z - min_z)) * 100
                
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
                
                ax.axvline(baseline_mean + baseline_std, color='green', linestyle=':', 
                        label=f'+1 Std Dev ({baseline_mean + baseline_std:.2f})')
                ax.axvline(baseline_mean - baseline_std, color='orange', linestyle=':', 
                        label=f'-1 Std Dev ({baseline_mean - baseline_std:.2f})')
                
                ax.set_xlabel('Environment Score')
                ax.set_title(f'Distribution of Environment Scores for {industry} ({emp_range_text} employees)')
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
                        
                    better_than = (df_results['Env_Z_Score'] < custom_z).mean() * 100
                    st.success(f"Your company performs better than **{better_than:.1f}%** of companies in this segment (based on Z-Score)")
            else:
                st.warning(f"No companies found in {industry} sector with {emp_range_text} employees")
        
        elif analysis_type == "Company-Only Analysis":
            overall_mean = company_data['Environment_Score'].mean()
            overall_std = company_data['Environment_Score'].std(ddof=1)
            
            st.markdown("### Compare Against Entire Dataset")
            st.markdown(f"**Overall Environment Score Baseline:** {overall_mean:.2f} (std: {overall_std:.2f})")
            
            all_companies = company_data['Company_Name'].dropna().tolist()
            selected_company = st.selectbox("Select Your Company", ["(None)"] + all_companies, key="company_only_company")
            
            if selected_company != "(None)":
                company_row = company_data[company_data['Company_Name'] == selected_company].iloc[0]
                company_score = company_row['Environment_Score']
                company_sector = company_row['Sector_classification']
                company_employees = company_row['Total_Employees']
                
                company_z = (company_score - overall_mean) / overall_std
                
                percentile = (company_data['Environment_Score'] < company_score).mean() * 100
                st.markdown(f"### {selected_company}")
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
    st.markdown("### Your Personal Electricity Analysis")
    st.markdown("")  
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
        electricity_data = st.session_state.get('full_electricity_data', None)

        st.markdown("Enter your personal electricity consumption and compare it with the dataset")
        user_electricity = st.number_input("Your Monthly Electricity Usage (kWh)", 
                                        min_value=0.0, 
                                        value=0.0, 
                                        step=10.0,
                                        key="personal_electricity")
        
        st.session_state.user_electricity = user_electricity

        hh_size = st.number_input("Number of people in household", 
                        min_value=1, 
                        value=st.session_state.get('household_size', 1), 
                        step=1,
                        key="household_size")

        def calculate_equivalent_consumption(electricity_usage, household_size):
            if household_size == 1:
                return electricity_usage / 1
            elif household_size == 2:
                return electricity_usage / 1.75
            elif household_size == 3:
                return electricity_usage / 2.5
            elif household_size == 4:
                return electricity_usage / 3.25
            else:  
                return electricity_usage / (household_size - 1)

        individual_equivalent_consumption = calculate_equivalent_consumption(user_electricity, hh_size)

        if user_electricity > 0:
            st.markdown("#### Individual Electricity Consumption")
            ind_col1, ind_col2 = st.columns(2)
            
            with ind_col1:
                st.metric("Total Household Usage", f"{user_electricity:.1f} kWh")
                
            with ind_col2:
                st.metric("Individual Equivalent Usage", f"{individual_equivalent_consumption:.1f} kWh",
                         f"Based on {hh_size} people")

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
                    st.metric("Your Household Size", f"{hh_size} people")
                
                with hh_col2:
                    comparison = "larger" if hh_size > state_hh_avg else "smaller" if hh_size < state_hh_avg else "same as"
                    st.metric(f"{user_state} ({user_sector_name}) Average", 
                             f"{state_hh_avg:.1f} people",
                             f"Your household is {comparison} average")
                
                with hh_col3:
                    percentile_hh = (state_sector_hh_data['hh_size'] <= hh_size).mean() * 100
                    st.metric("Your Percentile", 
                             f"{percentile_hh:.1f}%",
                             f"Larger than {percentile_hh:.1f}% of households")
                
                fig_hh, ax_hh = plt.subplots(figsize=(10, 6))
                
                sns.histplot(data=state_sector_hh_data, x='hh_size', bins=range(1, int(state_sector_hh_data['hh_size'].max()) + 2), 
                           kde=True, ax=ax_hh, alpha=0.7)
                
                ax_hh.axvline(hh_size, color='red', linestyle='-', linewidth=2, label=f'Your Household ({hh_size} people)')
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
                
                ax_states.axhline(y=hh_size, color='green', linestyle='--', linewidth=2,
                                label=f'Your Household Size ({hh_size} people)')
                
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
                        
                        ax_sector.axhline(y=hh_size, color='green', linestyle='--', linewidth=2,
                                        label=f'Your Household Size ({hh_size} people)')
                        
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
            
            if hh_size and hh_size > 0:
                st.metric("Per Person Electricity Consumption", f"{individual_equivalent_consumption:.2f} kWh/person")
            else:
                st.metric("Per Person Electricity Consumption", "N/A (household size not specified)")
            
            user_cost = st.number_input("Enter your monthly electricity cost (₹)", 
                                                    min_value=0.0, 
                                                    step=100.0)

            if user_cost > 0:
                def calculate_electricity_from_cost(cost, household_size):
                    if household_size == 1:
                        return cost / 1
                    elif household_size == 2:
                        return cost / 1.75
                    elif household_size == 3:
                        return cost / 2.5
                    elif household_size == 4:
                        return cost / 3.25
                    else:  # 5 or more people
                        return cost / (household_size - 1)
                
                # Auto-calculate user_electricity based on cost and household size
                safe_hh_size_calc = hh_size if hh_size and hh_size > 0 else 1
                user_electricity = calculate_electricity_from_cost(user_cost, safe_hh_size_calc)
                
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
                    st.metric("Individual Cost", f"₹{user_electricity:.2f}")
                        
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

            if 'feature_stats' in st.session_state and 'Electricity' in st.session_state.feature_stats:
                feature1_score = 0
                feature1_z_score = 0
                
                # Fix: Add safety check for hh_size
                safe_hh_size = hh_size if hh_size and hh_size > 0 else 1
                
                if user_state in st.session_state.feature_stats_by_state and 'Electricity' in st.session_state.feature_stats_by_state[user_state]:
                    state_stats = st.session_state.feature_stats_by_state[user_state]['Electricity']
                    
                    if 'sector_stats' not in st.session_state:
                        st.session_state.sector_stats = {}
                    
                    state_sector_key = f"{user_state}_{user_sector_name}"
                    
                    if state_sector_key not in st.session_state.sector_stats and not state_sector_data.empty:
                        sector_mean = state_sector_data['qty_usage_in_1month'].mean() / safe_hh_size  # Adjust for household size
                        sector_std = state_sector_data['qty_usage_in_1month'].std() / safe_hh_size   # Adjust for household size
                        
                        st.session_state.sector_stats[state_sector_key] = {
                            'mean': sector_mean,
                            'std': sector_std
                        }
                    
                    if state_sector_key in st.session_state.sector_stats:
                        sector_stats = st.session_state.sector_stats[state_sector_key]
                        feature1_mean = sector_stats['mean'] * safe_hh_size  
                        feature1_std = sector_stats['std'] * safe_hh_size   
                    else:
                        feature1_mean = state_stats['mean']
                        feature1_std = state_stats['std']
                    
                    feature1_z_score = (feature1_mean - user_electricity) / feature1_std if feature1_std > 0 else 0
                    feature1_z_score = np.clip(feature1_z_score, -3, 3)  # Cap between -3 and 3
                    
                    feature1_score = 0.25 * (1 - np.tanh(feature1_z_score/2.5)) + 0.25
                else:
                    stats = st.session_state.feature_stats['Electricity']
                    feature1_mean = stats['mean']
                    feature1_std = stats['std']
                    
                    feature1_z_score = (feature1_mean - user_electricity) / feature1_std if feature1_std > 0 else 0
                    feature1_z_score = np.clip(feature1_z_score, -3, 3)
                    
                    feature1_score = 0.25 * (1 - np.tanh(feature1_z_score/2.5)) + 0.25
                
                feature2_score = 0
                feature2_z_score = 0
                
                if 'mpce_stats' not in st.session_state:
                    st.session_state.mpce_stats = {}
                    
                    if 'full_electricity_data' in st.session_state and 'mpce' in st.session_state.full_electricity_data.columns:
                        full_data = st.session_state.full_electricity_data
                        
                        for i, (lower, upper) in enumerate(mpce_range_values):
                            range_data = full_data[(full_data['mpce'] >= lower) & (full_data['mpce'] < upper)]
                            
                            if not range_data.empty:
                                mean_usage = range_data['qty_usage_in_1month'].mean() / safe_hh_size  
                                std_usage = range_data['qty_usage_in_1month'].std() / safe_hh_size   
                                
                                st.session_state.mpce_stats[i] = {
                                    'range_name': mpce_ranges[i],
                                    'mean': mean_usage,
                                    'std': std_usage
                                }
                            else:
                                st.session_state.mpce_stats[i] = {
                                    'range_name': mpce_ranges[i],
                                    'mean': st.session_state.feature_stats['Electricity']['mean'] / safe_hh_size,
                                    'std': st.session_state.feature_stats['Electricity']['std'] / safe_hh_size
                                }
                
                if user_mpce_range_index in st.session_state.mpce_stats:
                    mpce_stats = st.session_state.mpce_stats[user_mpce_range_index]
                    feature2_mean = mpce_stats['mean'] * safe_hh_size  
                    feature2_std = mpce_stats['std'] * safe_hh_size   
                    
                    feature2_z_score = (feature2_mean - user_electricity) / feature2_std if feature2_std > 0 else 0
                    feature2_z_score = np.clip(feature2_z_score, -3, 3)
                    
                    feature2_score = 0.25 * (1 - np.tanh(feature2_z_score/2.5)) + 0.25
                else:
                    stats = st.session_state.feature_stats['Electricity']
                    feature2_mean = stats['mean']
                    feature2_std = stats['std']
                    
                    feature2_z_score = (feature2_mean - user_electricity) / feature2_std if feature2_std > 0 else 0
                    feature2_z_score = np.clip(feature2_z_score, -3, 3)
                    
                    feature2_score = 0.25 * (1 - np.tanh(feature2_z_score/2.5)) + 0.25
                
                total_sust_score = (feature1_score + feature2_score) * 1000
                
                combined_z_score = (feature1_z_score + feature2_z_score) / 2
                
                percentile = 0
                if 'scored_data' in st.session_state:
                    percentile = (st.session_state.scored_data['Sustainability_Score'] < total_sust_score).mean() * 100
                
                st.markdown("### Your Sustainability Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    comparison = "better than average" if user_electricity < feature1_mean else "worse than average"
                    diff_pct = abs(user_electricity - feature1_mean) / feature1_mean * 100 if feature1_mean > 0 else 0
                    st.metric(f"Compared to {user_state} ({user_sector_name})", 
                            f"{diff_pct:.1f}% {comparison}", 
                            f"{feature1_mean:.1f} kWh avg ({feature1_mean/safe_hh_size:.1f} kWh/person)")
                    
                    st.metric("Location-Based Score (50%)", 
                            f"{feature1_score*1000:.1f}/500",
                            f"Z-score: {feature1_z_score:.2f}")
                
                with col2:
                    mpce_comparison = "better than average" if user_electricity < feature2_mean else "worse than average"
                    mpce_diff_pct = abs(user_electricity - feature2_mean) / feature2_mean * 100 if feature2_mean > 0 else 0
                    st.metric(f"Compared to MPCE {st.session_state.user_mpce_range_name}", 
                            f"{mpce_diff_pct:.1f}% {mpce_comparison}", 
                            f"{feature2_mean:.1f} kWh avg ({feature2_mean/safe_hh_size:.1f} kWh/person)")
                    
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
                
                st.markdown("### Feature Breakdown")
                
                feature_data = pd.DataFrame({
                    'Feature': ['Location-Based (State/Sector)', 'MPCE-Based'],
                    'Score': [feature1_score*1000, feature2_score*1000],
                    'Max_Score': [500, 500]
                })
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh(feature_data['Feature'], feature_data['Score'], color=['#4287f5', '#42f5a7'])
                
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                        f"{feature_data['Score'][i]:.1f}/500", 
                        va='center')
                
                ax.axvline(x=total_sust_score, color='red', linestyle='--', 
                        label=f'Total Score: {total_sust_score:.1f}/1000')
                
                ax.set_xlim(0, 1000)
                ax.set_title('Your Sustainability Score Breakdown')
                ax.set_xlabel('Score')
                ax.legend()
                plt.tight_layout()
                
                st.pyplot(fig)
                
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
                    state_df = state_df.sort_values('Average_Electricity')
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.bar(state_df['State_UT'], state_df['Average_Electricity'], 
                                color=state_df['Is_Your_State'].map({True: 'red', False: 'blue'}))
                    
                    ax.axhline(y=user_electricity, color='green', linestyle='--', 
                            label=f'Your Usage ({user_electricity:.1f} kWh, {user_electricity/safe_hh_size:.1f} kWh/person)')
                    
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
                
                if 'mpce_stats' in st.session_state:
                    st.markdown("### Your Usage Compared to MPCE Ranges")
                    
                    mpce_data = []
                    for idx, stats in st.session_state.mpce_stats.items():
                        mpce_data.append({
                            'MPCE_Range': stats['range_name'],
                            'Average_Electricity': stats['mean'] * safe_hh_size,  
                            'Is_Your_Range': idx == user_mpce_range_index
                        })
                    
                    mpce_df = pd.DataFrame(mpce_data)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(mpce_df['MPCE_Range'], mpce_df['Average_Electricity'], 
                                color=mpce_df['Is_Your_Range'].map({True: 'red', False: 'blue'}))
                    
                    ax.axhline(y=user_electricity, color='green', linestyle='--', 
                            label=f'Your Usage ({user_electricity:.1f} kWh, {user_electricity/safe_hh_size:.1f} kWh/person)')
                    
                    user_mpce_avg = mpce_df[mpce_df['Is_Your_Range']]['Average_Electricity'].values[0] \
                                    if len(mpce_df[mpce_df['Is_Your_Range']]) > 0 else 0
                    if user_mpce_avg > 0 and 'user_mpce_range_name' in st.session_state:
                        ax.scatter(st.session_state.user_mpce_range_name, user_mpce_avg, s=100, color='red', zorder=3,
                                label=f'Your MPCE Range Avg ({user_mpce_avg:.1f} kWh)')
                    
                    plt.xticks(rotation=45)
                    plt.xlabel('MPCE Range')
                    plt.ylabel('Average Electricity Consumption (kWh)')
                    plt.title('Your Electricity Usage vs. MPCE Range Averages')
                    plt.legend()
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            
        
        
    st.markdown("---")
    st.markdown("### Resource Consumption")
        
    st.markdown("Water (in Litres)")

    usage_type = st.selectbox("Usage Type", ["Daily Usage", "Monthly Usage"])

    num_people = st.number_input("Number of People", min_value=1, value=1)

    new_customer = {}
        
    water_units = st.number_input("Water", min_value=0.0)
            
    water_min, water_max = 0.0, 1000.0
    if 'feature_constraints' in st.session_state and 'Water' in st.session_state.feature_constraints:
            water_min, water_max = st.session_state.feature_constraints['Water']

    if usage_type == "Daily Usage":
        monthly_water = water_units * 30
    else:  
        monthly_water = water_units

    per_person_monthly = monthly_water / num_people

    new_customer['Water'] = per_person_monthly
    average_monthly_per_person = 130 * 30

    z_score = (per_person_monthly - average_monthly_per_person) / 1000        
        
        
        
    st.title("Transport Carbon Footprint Calculator")
    st.markdown("""
    This helps you calculate your monthly carbon footprint based on your daily commute and transportation choices.
    Compare different transportation modes and receive personalized sustainability recommendations.
    """)

    if 'calculated' not in st.session_state:
        st.session_state.calculated = False
    if 'total_emissions' not in st.session_state:
        st.session_state.total_emissions = 0
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'emissions_data' not in st.session_state:
        st.session_state.emissions_data = {}

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

    def calculate_emission_stats():
        all_values = []
        
        for category in emission_factors["two_wheeler"]:
            for fuel in emission_factors["two_wheeler"][category]:
                min_val = emission_factors["two_wheeler"][category][fuel]["min"]
                max_val = emission_factors["two_wheeler"][category][fuel]["max"]
                all_values.append((min_val + max_val) / 2)
        
        for fuel in emission_factors["three_wheeler"]:
            min_val = emission_factors["three_wheeler"][fuel]["min"]
            max_val = emission_factors["three_wheeler"][fuel]["max"]
            all_values.append((min_val + max_val) / 2)
        
        for car_type in emission_factors["four_wheeler"]:
            for fuel in emission_factors["four_wheeler"][car_type]:
                base = emission_factors["four_wheeler"][car_type][fuel]["base"]
                uplift = emission_factors["four_wheeler"][car_type][fuel]["uplift"]
                all_values.append(base * uplift)
        
        for car_type in emission_factors["public_transport"]["taxi"]:
            for fuel in emission_factors["public_transport"]["taxi"][car_type]:
                base = emission_factors["public_transport"]["taxi"][car_type][fuel]["base"]
                uplift = emission_factors["public_transport"]["taxi"][car_type][fuel]["uplift"]
                all_values.append(base * uplift)
        
        for fuel in emission_factors["public_transport"]["bus"]:
            all_values.append(emission_factors["public_transport"]["bus"][fuel])
        all_values.append(emission_factors["public_transport"]["metro"])
        
        return np.mean(all_values), np.std(all_values)

    emission_mean, emission_std = calculate_emission_stats()

    def calculate_z_score(emission_factor):
        """Calculate Z-score for given emission factor"""
        if emission_std == 0:
            return 0
        return (emission_factor - emission_mean) / emission_std

    st.header("Your Commute Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        distance = st.number_input("Daily one-way distance (km)", min_value=0.1, value=10.0, step=0.5)
    with col2:
        days_per_week = st.number_input("Commuting days per week", min_value=1, max_value=7, value=5, step=1)
    with col3:
        weeks_per_month = st.number_input("Commuting weeks per month", min_value=1, max_value=5, value=4, step=1)

    total_monthly_km = distance * 2 * days_per_week * weeks_per_month
    st.metric("Total monthly commute distance", f"{total_monthly_km:.1f} km")

    st.header("Select Your Transport Category")
    transport_category = st.selectbox(
        "Transport Category",
        ["Private Transport", "Public Transport", "Both Private and Public"]
    )

    emission_factor = 0
    people_count = 1
    vehicle_type = ""
    vehicle_name = ""

    if transport_category == "Private Transport" or transport_category == "Both Private and Public":
        st.subheader("Private Transport Details")
        col1, col2 = st.columns([1, 2])
        with col1:
            private_vehicle_type = st.selectbox(
                "Vehicle Type",
                ["Two Wheeler", "Three Wheeler", "Four Wheeler"],
                key="private_vehicle"
            )

        if private_vehicle_type == "Two Wheeler":
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.selectbox("Category", ["Scooter", "Motorcycle"])
            with col2:
                engine_cc = st.number_input("Engine (cc)", 50, 1500, 150)
            with col3:
                fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "electric"])
            
            if engine_cc <= 150:
                emission_factor = emission_factors["two_wheeler"][category][fuel_type]["min"]
            else:
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
            
            if transport_category == "Public Transport":
                emission_factor = public_emission_factor
                people_count = public_people_count
                vehicle_type = "Public Transport"
                vehicle_name = f"Bus ({public_fuel_type})"
        
        else:  # Metro
            public_emission_factor = emission_factors["public_transport"]["metro"]
            public_people_count = 1  # Already factored into emission factor
            
            if transport_category == "Public Transport":
                emission_factor = public_emission_factor
                people_count = public_people_count
                vehicle_type = "Public Transport"
                vehicle_name = "Metro"

    if transport_category == "Both Private and Public":
        st.subheader("Usage Distribution")
        private_trips = st.number_input("Number of trips per day using private transport", min_value=0, max_value=10, value=2, step=1)
        total_trips = st.number_input("Total number of trips per day", min_value=1, max_value=10, value=4, step=1)
        private_ratio = private_trips / total_trips if total_trips > 0 else 0
        public_ratio = 1 - private_ratio
        
        if private_ratio > 0 and public_ratio > 0:
            if private_vehicle_type == "Two Wheeler":
                private_part = f"{category} ({fuel_type}, {engine_cc}cc)"
            elif private_vehicle_type == "Three Wheeler":
                private_part = f"Three Wheeler ({fuel_type}, {engine_cc}cc)"
            elif private_vehicle_type == "Four Wheeler":
                private_part = f"{car_type.replace('_', ' ').title()} ({fuel_type}, {engine_cc}cc)"
            
            if transport_mode == "Taxi":
                public_part = f"Taxi - {car_type.replace('_', ' ').title()} ({fuel_type})"
            elif transport_mode == "Bus":
                public_part = f"Bus ({public_fuel_type})"
            else:  
                public_part = "Metro"
            
            combined_emission_factor = (emission_factor / people_count) * private_ratio + (public_emission_factor / public_people_count) * public_ratio
            emission_factor = combined_emission_factor
            people_count = 1  
            
            vehicle_type = "Combined Transport"
            vehicle_name = f"{private_part} ({private_ratio*100:.0f}%) & {public_part} ({public_ratio*100:.0f}%)"

    if emission_factor > 0:
        z_score = calculate_z_score(emission_factor)
        
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
    
    


if st.button("Evaluate Customer"):
    new_customer = {
        'Water': per_person_monthly if 'per_person_monthly' in locals() else 0,
        'Electricity': individual_equivalent_consumption if 'individual_equivalent_consumption' in locals() else 0,
        'MPCE': user_mpce_avg if 'user_mpce_avg' in locals() else 0,
        'Public_Transport': total_monthly_km if 'transport_category' in locals() and transport_category in ["Public Transport", "Both Private and Public"] else 0,
        'Private_Transport': total_monthly_km if 'transport_category' in locals() and transport_category in ["Private Transport", "Both Private and Public"] else 0,
        'Company Score': company_score if 'company_score' in locals() else 0
    }

    if 'total_monthly_km' in locals() and 'emission_factor' in locals():
        total_emissions = total_monthly_km * emission_factor
        
        st.session_state.calculated = True
        st.session_state.total_emissions = total_emissions
        
        alternatives = {}
        
        alternatives[vehicle_name] = total_emissions
        
        alternatives["Bus (Diesel)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["diesel"])
        alternatives["Bus (CNG)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["cng"])
        alternatives["Bus (Electric)"] = (total_monthly_km * emission_factors["public_transport"]["bus"]["electric"])
        alternatives["Metro"] = (total_monthly_km * emission_factors["public_transport"]["metro"])
        
        # Add car sharing options if not already selected
        if not (vehicle_type == "Four Wheeler" and "rideshare" in locals() and rideshare and people_count >= 3):
            alternatives["Car Pooling (4 people)"] = (total_monthly_km * emission_factors["four_wheeler"]["sedan"]["petrol"]["base"] * emission_factors["four_wheeler"]["sedan"]["petrol"]["uplift"]) / 4
        
        # Add electric vehicle options if not already selected
        if not (vehicle_type == "Four Wheeler" and "fuel_type" in locals() and fuel_type == "electric"):
            alternatives["Electric Car"] = (total_monthly_km * emission_factors["four_wheeler"]["sedan"]["electric"]["base"] * emission_factors["four_wheeler"]["sedan"]["electric"]["uplift"])
        
        if not (vehicle_type == "Two Wheeler" and "fuel_type" in locals() and fuel_type == "electric"):
            alternatives["Electric Scooter"] = (total_monthly_km * emission_factors["two_wheeler"]["Scooter"]["electric"]["min"])
        
        st.session_state.emissions_data = alternatives
        
        st.info(f"""
    **Z-Score Interpretation:**
    - Your emission factor is {abs(z_score):.2f} standard deviations {'above' if z_score > 0 else 'below'} the average
    - Population mean: {emission_mean:.4f} kg CO2/km
    - Population std dev: {emission_std:.4f} kg CO2/km
    """)
       
        results_text = f"""Vehicle: {vehicle_name}
    Emission Factor: {emission_factor:.4f} kg CO2/km
    Z-Score: {z_score:.2f}
    Category: {emission_category}
    Interpretation: Your emission factor is {abs(z_score):.2f} standard deviations {'above' if z_score > 0 else 'below'} the average
    Population Mean: {emission_mean:.4f} kg CO2/km
    Population Standard Deviation: {emission_std:.4f} kg CO2/km
    Total Monthly Distance: {total_monthly_km:.1f} km
    Monthly CO2 Emissions: {(emission_factor * total_monthly_km / people_count):.2f} kg"""
        
        recommendations = []
        
        if total_emissions > 100:
            recommendations.append("Your carbon footprint from commuting is quite high. Consider switching to more sustainable transport options.")
        elif total_emissions > 50:
            recommendations.append("Your carbon footprint is moderate. There's room for improvement by considering more sustainable options.")
        else:
            recommendations.append("Your carbon footprint is relatively low, but you can still make improvements.")
        
        if vehicle_type == "Four Wheeler" and people_count == 1:
            recommendations.append("Consider carpooling to reduce emissions. Sharing your ride with 3 other people could reduce your emissions by up to 75%.")
        
        if "fuel_type" in locals() and fuel_type in ["petrol", "diesel"] and vehicle_type != "Public Transport":
            recommendations.append("Consider switching to an electric vehicle to significantly reduce your carbon footprint.")
        
        if vehicle_type in ["Four Wheeler", "Two Wheeler"]:
            bus_emissions = total_monthly_km * emission_factors["public_transport"]["bus"]["electric"]
            metro_emissions = total_monthly_km * emission_factors["public_transport"]["metro"]
            
            if total_emissions > 2 * bus_emissions:
                recommendations.append(f"Using an electric bus could reduce your emissions by approximately {(total_emissions - bus_emissions) / total_emissions * 100:.1f}%.")
            
            if total_emissions > 2 * metro_emissions:
                recommendations.append(f"Using metro could reduce your emissions by approximately {(total_emissions - metro_emissions) / total_emissions * 100:.1f}%.")
        
        st.session_state.recommendations = recommendations

        
    


        inverse_scoring = True
        
        z_vals = {}
        for feat, val in new_customer.items():
            if feat in features:  
                if 'feature_stats' in st.session_state and feat in st.session_state.feature_stats:
                    stats = st.session_state.feature_stats[feat]
                    z = (stats['mean'] - val)/stats['std'] if inverse_scoring else (val - stats['mean'])/stats['std']
                else:
                    z = (val - 0.5) / 0.5  # Simple normalization assuming 0-1 range
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
        
        if "individual_equivalent_consumption" in st.session_state:
            electricity_value = st.session_state.user_electricity
            normalized_electricity = electricity_value / 1000  # Normalize to 0-1 range
            z_electricity = (0.5 - normalized_electricity) / 0.5 if inverse_scoring else (normalized_electricity - 0.5) / 0.5
            z_vals["Electricity"] = np.clip(z_electricity, -1, 1)

        

        z_score = 0
        
        # Water component 
        water_z = max(0, (new_customer['Water'] - 3900) / 3900)  
        z_score += water_z * 0.25  # 25% weight
        
        # Electricity component 
        electricity_z = max(0, (individual_equivalent_consumption - 300) / 300)  
        z_score += electricity_z * 0.25  # 25% weight
        
        # Private transport component 
        private_transport_z = new_customer['Private_Transport'] / 1000  
        z_score += private_transport_z * 0.25  # 25% weight
        
        # Public transport component (bonus for usage, negative z for bonus)
        public_transport_z = -min(1, new_customer['Public_Transport'] / 500) 
        z_score += public_transport_z * 0.15  # 15% weight
        
        company_z = -(new_customer['Company Score'] / 100)  
        z_score += company_z * 0.10  # 10% weight
        
        sust_score = 500 * (1 - np.tanh(z_score/2.5))

        norm_vals = {}
        for feat, val in new_customer.items():
            if feat in z_vals:   
                cmin, cmax = (0, 1000)  # Default range
                if cmax == cmin:
                    norm_vals[feat] = 1
                else:
                    if inverse_scoring:
                        norm_vals[feat] = ((cmax - val)/(cmax - cmin))*999 + 1
                    else:
                        norm_vals[feat] = ((val - cmin)/(cmax - cmin))*999 + 1

        

        sust_rank = 1  # Default rank if no comparison data available
        trad_rank = 1  # Default rank if no comparison data available
        
        st.markdown("---")
        st.markdown("### Customer Score Results")
        
        col1, col2 = st.columns(2)
        with col1:
            if "individual_equivalent_consumption" in locals():
                st.metric("Personal Electricity Usage", f"{individual_equivalent_consumption:.2f} kWh")
            if "Company Score" in new_customer:
                st.metric("Company Environment Score", f"{new_customer['Company Score']:.2f}")
            
            if 'water_units' in locals() and 'num_people' in locals() and 'usage_type' in locals() and num_people > 0:
                if usage_type == "Daily Usage":
                    monthly_water = water_units * 30
                else:  
                    monthly_water = water_units
                
                per_person_monthly = monthly_water / num_people
                
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
            
            st.metric("Water Usage", f"{new_customer.get('Water', 0):.2f}") # Use .get for safety
            st.metric("Public Transport", f"{new_customer.get('Public_Transport', 0):.2f}")
            st.metric("Private Transport", f"{new_customer.get('Private_Transport', 0):.2f}")
            
            weighted_score = sum(z_vals[feat] * weights.get(feat, 0) for feat in z_vals.keys())
            
            st.metric("Z-Score", f"{z_score:.2f}")
            st.metric("Sustainability Score", f"{sust_score:.2f}")
            st.metric("Sustainability Rank", f"{sust_rank}")
            st.metric("Legacy Weighted Score", f"{weighted_score:.2f}")
            st.metric("Legacy Rank", f"{trad_rank}")

            if 'scored_data' in st.session_state and 'Sustainability_Score' in st.session_state.scored_data.columns:
                existing_sust = st.session_state.scored_data['Sustainability_Score']
                better_than = (existing_sust < sust_score).mean() * 100
                st.success(f"This customer performs better than **{better_than:.1f}%** of customers in the dataset (based on Z-Score)")
            else:
                st.info("No comparison data available in the dataset.")

            z_description = "above" if z_score > 0 else "below"
            st.info(f"Performance: **{abs(z_score):.2f} SD {z_description} mean**")
        
        with col2:
            active_features = {feat: val for feat, val in new_customer.items() if val > 0}
            
            if active_features:  
                num_active = len(active_features)
                equal_weight = 1.0 / num_active
                
                contribs = {feat: equal_weight for feat in active_features.keys()}
                
                explode = [0.1 if i % 2 == 0 else 0 for i in range(len(contribs))]
                
                fig_pie_weights, ax_pie_weights = plt.subplots(figsize=(6,6))
                ax_pie_weights.pie(
                    list(contribs.values()),
                    labels=list(contribs.keys()),
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=explode
                )
                ax_pie_weights.set_title('Active Features Distribution for Customer')
                st.pyplot(fig_pie_weights)
                
                for feat, weight in contribs.items():
                    st.info(f"{feat} contributes {weight*100:.1f}% (Value: {new_customer[feat]:.2f})")
            else:
                st.info("No active features with positive values to display for pie chart.")

        if st.session_state.get("calculated", False): # Use .get for safety
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
                
        st.markdown("---")
        st.markdown("### Customer Input Weightage Analysis")
        core_input_features = ['Water', 'MPCE', 'Public_Transport', 'Private_Transport', 'Electricity', 'Company Score']
        input_data = {}
        input_weight = {} 
        for feat in core_input_features:
            input_data[feat] = new_customer.get(feat, 0) 
            input_weight[feat] = 0.01 # Start with a minimal weight for visibility



        if "user_electricity" in st.session_state:
            input_data["Electricity"] = st.session_state.user_electricity
            if "Electricity" in z_vals and "Electricity" in weights:
                 input_weight["Electricity"] = max(input_weight.get("Electricity",0.01), abs(z_vals["Electricity"] * weights["Electricity"]))
            elif st.session_state.user_electricity > 0 : # Fallback if not in z_vals/weights but has value
                 input_weight["Electricity"] = max(input_weight.get("Electricity",0.01), 0.125, st.session_state.user_electricity/1000)


        if "MPCE" in new_customer and new_customer["MPCE"] is not None:
            input_data["MPCE"] = new_customer["MPCE"]
            if "MPCE" in z_vals and "MPCE" in weights:
                input_weight["MPCE"] = max(input_weight.get("MPCE",0.01), abs(z_vals["MPCE"] * weights["MPCE"]))
            elif new_customer["MPCE"] > 0: 
                 input_weight["MPCE"] = max(input_weight.get("MPCE",0.01), 0.15)


        if "Company Score" in new_customer and new_customer["Company Score"] is not None:
            input_data["Company Score"] = new_customer["Company Score"]
            if "Company Score" in z_vals and "Company Score" in weights:
                 input_weight["Company Score"] = max(input_weight.get("Company Score",0.01), abs(z_vals["Company Score"] * weights["Company Score"]))
            elif 'company_data' in locals() or 'company_data' in globals(): 
                if "Company Score" in company_data.columns:
                    mean = company_data["Company Score"].mean()
                    std = company_data["Company Score"].std()
                    if std > 0:
                        env_z_score_temp = (mean - new_customer["Company Score"])/std if inverse_scoring else (new_customer["Company Score"] - mean)/std
                        env_z_score_temp = np.clip(env_z_score_temp, -1, 1)
                        input_weight["Company Score"] = max(input_weight.get("Company Score",0.01), abs(env_z_score_temp * (weights.get("Company Score", 0.15)))) # Use 0.15 default model weight
                    else:
                        input_weight["Company Score"] = max(input_weight.get("Company Score",0.01),0.15)
            else:
                input_weight["Company Score"] = max(input_weight.get("Company Score",0.01),0.15)
        
        final_input_weights_for_pie = {}
        for feat in core_input_features: 
            if feat in input_data: 
                 final_input_weights_for_pie[feat] = max(input_weight.get(feat,0), 0.005) 
        for feat, wt in input_weight.items(): 
            if wt > 0.005 : 
                 final_input_weights_for_pie[feat] = wt
            elif feat in core_input_features : 
                 final_input_weights_for_pie[feat] = max(final_input_weights_for_pie.get(feat,0), wt)


        if final_input_weights_for_pie:
            total_pie_weight = sum(final_input_weights_for_pie.values())
            if total_pie_weight > 0:
                
                col1_input_w, col2_input_w = st.columns(2)
                
                with col1_input_w:
                    sorted_inputs_pie = sorted(final_input_weights_for_pie.items(), key=lambda x: x[1], reverse=True)
                    input_labels_pie = [f[0] for f in sorted_inputs_pie]
                    input_sizes_pie = [f[1] for f in sorted_inputs_pie]
                    
                    if len(input_labels_pie) > 7: 
                        other_weight_pie = sum(input_sizes_pie[6:])
                        input_labels_pie = input_labels_pie[:6] + ["Others"]
                        input_sizes_pie = input_sizes_pie[:6] + [other_weight_pie]

                    fig_pie_inputs = px.pie(
                        values=input_sizes_pie,
                        names=input_labels_pie,
                        title='Customer Input Influence Distribution'
                    )
                    fig_pie_inputs.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie_inputs.update_layout(
                        showlegend=True, 
                        legend_title_text='Features',
                        margin=dict(t=40, l=0, r=0, b=0),
                        height=400
                    )
                    st.plotly_chart(fig_pie_inputs, use_container_width=True)
                
                with col2_input_w:
                    input_table_data = []
                    for feat, weight_val_pie in sorted_inputs_pie:
                        if feat == "Others": 
                            continue
                        contrib_pct_pie = (weight_val_pie / total_pie_weight) * 100 if total_pie_weight > 0 else 0.00
                        input_table_data.append({
                            "Input Field": feat,
                            "Value": input_data.get(feat, "N/A"), 
                            "Influence Score": weight_val_pie, 
                            "Contribution %": f"{contrib_pct_pie:.2f}%"
                        })
                    
                    if input_table_data:
                        input_df_display = pd.DataFrame(input_table_data)
                        st.write("Individual Input Influence Scores")
                        st.dataframe(input_df_display)
                    else:
                        st.info("No input data to display in table.")
            else:
                st.info("No input weights to display for Customer Input Weightage.")
        else:
            st.info("No input weight data available for Customer Input Weightage Analysis.")


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
