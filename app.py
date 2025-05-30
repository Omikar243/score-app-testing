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

# Calculate average electricity consumption by state and sector
if sector_for_comparison == 'Both':
    # Group by state and calculate mean for both sectors combined
    state_avg = electricity_data.groupby('state_name')['qty_usage_in_1month'].mean().reset_index()
    # Add household size calculations if available
    if 'hh_size' in electricity_data.columns:
        state_hh_avg = electricity_data.groupby('state_name')['hh_size'].mean().reset_index()
        state_avg = state_avg.merge(state_hh_avg, on='state_name')
        electricity_data['per_capita_usage'] = electricity_data['qty_usage_in_1month'] / electricity_data['hh_size']
        state_per_capita_avg = electricity_data.groupby('state_name')['per_capita_usage'].mean().reset_index()
        state_avg = state_avg.merge(state_per_capita_avg, on='state_name')
    chart_title = "Average Electricity Consumption by State (Rural & Urban)"
else:
    # Filter by selected sector and then group by state
    sector_value = 1 if sector_for_comparison == 'Rural' else 2
    filtered_comparison = electricity_data[electricity_data['sector'] == sector_value]
    state_avg = filtered_comparison.groupby('state_name')['qty_usage_in_1month'].mean().reset_index()
    # Add household size calculations if available
    if 'hh_size' in filtered_comparison.columns:
        state_hh_avg = filtered_comparison.groupby('state_name')['hh_size'].mean().reset_index()
        state_avg = state_avg.merge(state_hh_avg, on='state_name')
        filtered_comparison['per_capita_usage'] = filtered_comparison['qty_usage_in_1month'] / filtered_comparison['hh_size']
        state_per_capita_avg = filtered_comparison.groupby('state_name')['per_capita_usage'].mean().reset_index()
        state_avg = state_avg.merge(state_per_capita_avg, on='state_name')
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

# Add household size comparison graph
if 'hh_size' in electricity_data.columns:
    st.markdown("## Household Size Comparison Across States")
    
    # Create tabs for different household size views
    hh_tab1, hh_tab2 = st.tabs(["Average Household Size", "Per Capita Electricity"])
    
    with hh_tab1:
        # Sort by household size for better visualization
        state_hh_sorted = state_avg.sort_values('hh_size', ascending=False) if 'hh_size' in state_avg.columns else state_avg
        
        fig_hh = px.bar(state_hh_sorted, x='state_name', y='hh_size',
                        title=f"Average Household Size by State ({sector_for_comparison})",
                        labels={
                            'state_name': 'State/UT',
                            'hh_size': 'Average Household Size (people)'
                        },
                        color='hh_size',
                        color_continuous_scale='Blues')

        # Customize layout
        fig_hh.update_layout(
            xaxis_tickangle=90,
            xaxis_title='State/UT',
            yaxis_title='Average Household Size (people)'
        )

        # Add value labels on top of bars
        fig_hh.update_traces(texttemplate='%{y:.1f}', textposition='outside')

        st.plotly_chart(fig_hh)
    
    with hh_tab2:
        if 'per_capita_usage' in state_avg.columns:
            # Sort by per capita usage for better visualization
            state_pc_sorted = state_avg.sort_values('per_capita_usage', ascending=False)
            
            fig_pc = px.bar(state_pc_sorted, x='state_name', y='per_capita_usage',
                            title=f"Per Capita Electricity Consumption by State ({sector_for_comparison})",
                            labels={
                                'state_name': 'State/UT',
                                'per_capita_usage': 'Per Capita Electricity (kWh/person/month)'
                            },
                            color='per_capita_usage',
                            color_continuous_scale='Viridis')

            # Customize layout
            fig_pc.update_layout(
                xaxis_tickangle=90,
                xaxis_title='State/UT',
                yaxis_title='Per Capita Electricity (kWh/person/month)'
            )

            # Add value labels on top of bars
            fig_pc.update_traces(texttemplate='%{y:.2f}', textposition='outside')

            st.plotly_chart(fig_pc)

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
        # Create a new dictionary with only the weights we want to use
        filtered_weights = {
        "Electricity_State": float(w_elec_location),
        "Electricity_MPCE": float(w_elec_mpce),
        "Electricity": float(w_elec_location) + float(w_elec_mpce),  # Add total
        "Public_Transport": w_public,
        "Private_Transport": w_private,
        "Water": w_water,
        "Company": w_company
    }
        
        # Store the weights in session state for backend calculation
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

features = ["Electricity_State", "Electricity_MPCE", "Water", "Public_Transport", "Private_Transport", "Company"]

if test_mode == "CSV Upload":
        st.markdown("### Upload Test Data")   
        # Download template based on features from weights
        if 'weights' in st.session_state:
            required_cols = ["Electricity_State", "Electricity_MPCE", "Water", "Public_Transport", "Private_Transport", "Company", "Company_Name", "Sector_classification", "Environment_Score", "ESG_Rating", "Total_Employees"]

            template_cols = list(dict.fromkeys(required_cols))
            tmpl = pd.DataFrame(columns=template_cols).to_csv(index=False).encode('utf-8')
            st.download_button("Download Test CSV Template",
                             data=tmpl, file_name="test_template.csv", mime="text/csv")
            
            up_test = st.file_uploader("Upload Test Data", type="csv", key="test_uploader")
            if up_test:
                test_df = pd.read_csv(up_test)
                st.dataframe(test_df)
                
                if st.button("Process Test Batch"):
                    if 'weights' not in st.session_state:
                        st.error("Please generate weights first using the 'Generate Weighted Score' button.")
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
    st.markdown("### Your Personal Electricity Analysis")
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

        # Add household size input
        # Add household size input

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
        # grab your full electricity DataFrame out of session_state
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

        # Household Size Comparison Visualization
        if electricity_data is not None and 'hh_size' in electricity_data.columns:
            # Filter data for user's state and sector
            state_sector_hh_data = electricity_data[
                (electricity_data['state_name'] == user_state) & 
                (electricity_data['sector'] == user_sector)
            ]
            
            if not state_sector_hh_data.empty:
                # Calculate household size statistics for user's state/sector
                state_hh_avg = state_sector_hh_data['hh_size'].mean()
                state_hh_median = state_sector_hh_data['hh_size'].median()
                
                # Display household size metrics
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
                
                # Create household size distribution visualization
                fig_hh, ax_hh = plt.subplots(figsize=(10, 6))
                
                # Create histogram of household sizes
                sns.histplot(data=state_sector_hh_data, x='hh_size', bins=range(1, int(state_sector_hh_data['hh_size'].max()) + 2), 
                           kde=True, ax=ax_hh, alpha=0.7)
                
                # Add vertical lines for user's household size, average, and median
                ax_hh.axvline(hh_size, color='red', linestyle='-', linewidth=2, label=f'Your Household ({hh_size} people)')
                ax_hh.axvline(state_hh_avg, color='green', linestyle='--', linewidth=2, label=f'State Average ({state_hh_avg:.1f} people)')
                ax_hh.axvline(state_hh_median, color='orange', linestyle='--', linewidth=2, label=f'State Median ({state_hh_median:.1f} people)')
                
                ax_hh.set_title(f"Household Size Distribution in {user_state} ({user_sector_name})")
                ax_hh.set_xlabel("Household Size (Number of People)")
                ax_hh.set_ylabel("Frequency")
                ax_hh.legend()
                ax_hh.grid(True, alpha=0.3)
                
                st.pyplot(fig_hh)
                
                # Create a comparison with all states
                st.markdown("#### Household Size Comparison Across States")
                
                # Calculate average household size by state
                state_hh_comparison = electricity_data.groupby('state_name')['hh_size'].agg(['mean', 'median']).reset_index()
                state_hh_comparison = state_hh_comparison.sort_values('mean')
                state_hh_comparison['is_user_state'] = state_hh_comparison['state_name'] == user_state
                
                # Create bar chart comparing states
                fig_states, ax_states = plt.subplots(figsize=(14, 8))
                
                bars = ax_states.bar(state_hh_comparison['state_name'], 
                                   state_hh_comparison['mean'],
                                   color=state_hh_comparison['is_user_state'].map({True: 'red', False: 'skyblue'}),
                                   alpha=0.8)
                
                # Add user's household size as horizontal line
                ax_states.axhline(y=hh_size, color='green', linestyle='--', linewidth=2,
                                label=f'Your Household Size ({hh_size} people)')
                
                # Highlight user's state
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
                
                # Create sector comparison (Rural vs Urban)
                if len(electricity_data['sector'].unique()) > 1:
                    st.markdown("#### Rural vs Urban Household Size Comparison")
                    
                    sector_hh_comparison = electricity_data.groupby(['state_name', 'sector'])['hh_size'].mean().reset_index()
                    sector_hh_comparison['sector_name'] = sector_hh_comparison['sector'].map({1: 'Rural', 2: 'Urban'})
                    
                    # Filter for user's state
                    user_state_sectors = sector_hh_comparison[sector_hh_comparison['state_name'] == user_state]
                    
                    if not user_state_sectors.empty:
                        fig_sector, ax_sector = plt.subplots(figsize=(8, 6))
                        
                        bars_sector = ax_sector.bar(user_state_sectors['sector_name'], 
                                                  user_state_sectors['hh_size'],
                                                  color=['lightcoral' if s == user_sector_name else 'lightblue' 
                                                        for s in user_state_sectors['sector_name']],
                                                  alpha=0.8)
                        
                        # Add user's household size as horizontal line
                        ax_sector.axhline(y=hh_size, color='green', linestyle='--', linewidth=2,
                                        label=f'Your Household Size ({hh_size} people)')
                        
                        # Highlight user's sector
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
            
            # Fix: Add safety check for hh_size
            if hh_size and hh_size > 0:
                st.metric("Per Person Electricity Consumption", f"{user_electricity/hh_size:.2f} kWh/person")
            else:
                st.metric("Per Person Electricity Consumption", "N/A (household size not specified)")
            
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
                    
                    # Fix: Safe calculation of per person average
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
            
            # Calculate and display sustainability scores with demographic factors
            if 'feature_stats' in st.session_state and 'Electricity' in st.session_state.feature_stats:
                # FEATURE 1: State and Rural/Urban based calculations
                # Check if we have state-sector specific data
                feature1_score = 0
                feature1_z_score = 0
                
                # Fix: Add safety check for hh_size
                safe_hh_size = hh_size if hh_size and hh_size > 0 else 1
                
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
                        sector_mean = state_sector_data['qty_usage_in_1month'].mean() / safe_hh_size  # Adjust for household size
                        sector_std = state_sector_data['qty_usage_in_1month'].std() / safe_hh_size   # Adjust for household size
                        
                        # Store these stats
                        st.session_state.sector_stats[state_sector_key] = {
                            'mean': sector_mean,
                            'std': sector_std
                        }
                    
                    # Use state-sector specific stats if available, otherwise use state stats
                    if state_sector_key in st.session_state.sector_stats:
                        sector_stats = st.session_state.sector_stats[state_sector_key]
                        feature1_mean = sector_stats['mean'] * safe_hh_size  # Scale back up for comparison
                        feature1_std = sector_stats['std'] * safe_hh_size   # Scale back up for comparison
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
                                mean_usage = range_data['qty_usage_in_1month'].mean() / safe_hh_size  # Adjust for household size
                                std_usage = range_data['qty_usage_in_1month'].std() / safe_hh_size   # Adjust for household size
                                
                                st.session_state.mpce_stats[i] = {
                                    'range_name': mpce_ranges[i],
                                    'mean': mean_usage,
                                    'std': std_usage
                                }
                            else:
                                # If no data for this range, use overall stats
                                st.session_state.mpce_stats[i] = {
                                    'range_name': mpce_ranges[i],
                                    'mean': st.session_state.feature_stats['Electricity']['mean'] / safe_hh_size,
                                    'std': st.session_state.feature_stats['Electricity']['std'] / safe_hh_size
                                }
                
                # Use MPCE range statistics for feature 2
                if user_mpce_range_index in st.session_state.mpce_stats:
                    mpce_stats = st.session_state.mpce_stats[user_mpce_range_index]
                    feature2_mean = mpce_stats['mean'] * safe_hh_size  # Scale back up for comparison
                    feature2_std = mpce_stats['std'] * safe_hh_size   # Scale back up for comparison
                    
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
                            label=f'Your Usage ({user_electricity:.1f} kWh, {user_electricity/safe_hh_size:.1f} kWh/person)')
                    
                    # Highlight user's state's average
                    user_state_avg = state_df[state_df['Is_Your_State']]['Average_Electricity'].values[0] \
                                    if len(state_df[state_df['Is_Your_State']]) > 0 else 0
                    if user_state_avg > 0:
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
                            'Average_Electricity': stats['mean'] * safe_hh_size,  # Scale back up for visualization
                            'Is_Your_Range': idx == user_mpce_range_index
                        })
                    
                    mpce_df = pd.DataFrame(mpce_data)
                    
                    # Create bar chart with user's MPCE range highlighted
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(mpce_df['MPCE_Range'], mpce_df['Average_Electricity'], 
                                color=mpce_df['Is_Your_Range'].map({True: 'red', False: 'blue'}))
                    
                    # Add user's consumption as a horizontal line
                    ax.axhline(y=user_electricity, color='green', linestyle='--', 
                            label=f'Your Usage ({user_electricity:.1f} kWh, {user_electricity/safe_hh_size:.1f} kWh/person)')
                    
                    # Highlight user's MPCE range average
                    user_mpce_avg = mpce_df[mpce_df['Is_Your_Range']]['Average_Electricity'].values[0] \
                                    if len(mpce_df[mpce_df['Is_Your_Range']]) > 0 else 0
                    if user_mpce_avg > 0 and 'user_mpce_range_name' in st.session_state:
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
    st.markdown("### Resource Consumption")
        
    st.markdown("Water (in Litres)")

        # Add dropdown for usage type
    usage_type = st.selectbox("Usage Type", ["Daily Usage", "Monthly Usage"])

        # Add number of people input
    num_people = st.number_input("Number of People", min_value=1, value=1)

        # Initialize new_customer dictionary
    new_customer = {}
        
    water_units = st.number_input("Water", min_value=0.0)
            
    water_min, water_max = 0.0, 1000.0
    if 'feature_constraints' in st.session_state and 'Water' in st.session_state.feature_constraints:
            water_min, water_max = st.session_state.feature_constraints['Water']

        # Calculate monthly water consumption based on usage type
    if usage_type == "Daily Usage":
        monthly_water = water_units * 30
    else:  # Monthly Usage
        monthly_water = water_units

        # Calculate per person monthly consumption
    per_person_monthly = monthly_water / num_people

    new_customer['Water'] = per_person_monthly
        # Average consumption: 130L/day/person = 3900L/month/person
    average_monthly_per_person = 130 * 30

        # Calculate Z-score (assuming standard deviation of 1000L for simplification)
    z_score = (per_person_monthly - average_monthly_per_person) / 1000        
        
        
        
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

import numpy as np

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

# Calculate Z-score statistics from emission factors
def calculate_emission_stats():
    all_values = []
    
    # Two wheelers - use average of min and max
    for category in emission_factors["two_wheeler"]:
        for fuel in emission_factors["two_wheeler"][category]:
            min_val = emission_factors["two_wheeler"][category][fuel]["min"]
            max_val = emission_factors["two_wheeler"][category][fuel]["max"]
            all_values.append((min_val + max_val) / 2)
    
    # Three wheelers - use average of min and max
    for fuel in emission_factors["three_wheeler"]:
        min_val = emission_factors["three_wheeler"][fuel]["min"]
        max_val = emission_factors["three_wheeler"][fuel]["max"]
        all_values.append((min_val + max_val) / 2)
    
    # Four wheelers - use base * uplift
    for car_type in emission_factors["four_wheeler"]:
        for fuel in emission_factors["four_wheeler"][car_type]:
            base = emission_factors["four_wheeler"][car_type][fuel]["base"]
            uplift = emission_factors["four_wheeler"][car_type][fuel]["uplift"]
            all_values.append(base * uplift)
    
    # Public transport - taxi (use base * uplift)
    for car_type in emission_factors["public_transport"]["taxi"]:
        for fuel in emission_factors["public_transport"]["taxi"][car_type]:
            base = emission_factors["public_transport"]["taxi"][car_type][fuel]["base"]
            uplift = emission_factors["public_transport"]["taxi"][car_type][fuel]["uplift"]
            all_values.append(base * uplift)
    
    # Public transport - bus and metro
    for fuel in emission_factors["public_transport"]["bus"]:
        all_values.append(emission_factors["public_transport"]["bus"][fuel])
    all_values.append(emission_factors["public_transport"]["metro"])
    
    return np.mean(all_values), np.std(all_values)

# Calculate mean and standard deviation for Z-score
emission_mean, emission_std = calculate_emission_stats()

def calculate_z_score(emission_factor):
    """Calculate Z-score for given emission factor"""
    if emission_std == 0:
        return 0
    return (emission_factor - emission_mean) / emission_std

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

# Calculate and display Z-score
if emission_factor > 0:
    z_score = calculate_z_score(emission_factor)
    
    # Display Z-score information
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
    # Create new_customer dictionary with collected values
    new_customer = {
        'Water': per_person_monthly if 'per_person_monthly' in locals() else 0,
        'Electricity': user_electricity if 'user_electricity' in locals() else 0,
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
    if 'weights' not in st.session_state:
        st.error("Please generate weights first using the 'Generate Weighted Score' button.")
    else:
        inverse_scoring = True
        
        z_vals = {}
        for feat, val in new_customer.items():
            if feat in features:  
                # Use default statistics if feature_stats not available
                if 'feature_stats' in st.session_state and feat in st.session_state.feature_stats:
                    stats = st.session_state.feature_stats[feat]
                    z = (stats['mean'] - val)/stats['std'] if inverse_scoring else (val - stats['mean'])/stats['std']
                else:
                    # Default normalization without statistics
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
        
        if "user_electricity" in st.session_state:
            electricity_value = st.session_state.user_electricity
            # Simple normalization without relying on feature_stats
            # Assuming electricity values are typically in a reasonable range (0-1000)
            normalized_electricity = electricity_value / 1000  # Normalize to 0-1 range
            z_electricity = (0.5 - normalized_electricity) / 0.5 if inverse_scoring else (normalized_electricity - 0.5) / 0.5
            z_vals["Electricity"] = np.clip(z_electricity, -1, 1)

        weights = st.session_state.weights

# Initialize features_for_z_score with the global `features` list
# This list should contain all features considered in the model
        features_for_z_score = list(z_vals.keys()) # Assuming features_list is stored in session_state

        z_score = sum(z_vals[f] * weights.get(f, 0) for f in features_for_z_score if f in z_vals)
        
       
        if "Environment_Score" in z_vals and "Environment_Score" not in features_for_z_score: # Avoid double counting
            z_score += z_vals["Environment_Score"] * weights.get("Environment_Score", 0)

        sust_score = 500 * (1 - np.tanh(z_score/2.5))

        norm_vals = {}
        for feat, val in new_customer.items():
            if feat in z_vals:   # Only process numeric features for scoring
                # Use default range if feature_constraints not available
                cmin, cmax = (0, 1000)  # Default range
                if cmax == cmin:
                    norm_vals[feat] = 1
                else:
                    if inverse_scoring:
                        norm_vals[feat] = ((cmax - val)/(cmax - cmin))*999 + 1
                    else:
                        norm_vals[feat] = ((val - cmin)/(cmax - cmin))*999 + 1

        if "Environment_Score" in new_customer:
            if "Environment_Score" in features_for_z_score: # check if it's part of main features
                # Use default range instead of feature_constraints
                cmin, cmax = (0, 100)  # Default range for Environment Score
                if cmax == cmin:
                    norm_vals["Environment_Score"] = 1
                else:
                    if inverse_scoring:
                        norm_vals["Environment_Score"] = ((cmax - new_customer["Environment_Score"])/(cmax - cmin))*999 + 1
                    else:
                        norm_vals["Environment_Score"] = ((new_customer["Environment_Score"] - cmin)/(cmax - cmin))*999 + 1

        weighted_score = sum(norm_vals[f] * weights.get(f, 0) for f in norm_vals.keys())

        if "Environment_Score" in norm_vals and "Environment_Score" not in features_for_z_score: # Avoid double counting
            weighted_score += norm_vals["Environment_Score"] * weights.get("Environment_Score", 0)

        sust_rank = 1  # Default rank if no comparison data available
        trad_rank = 1  # Default rank if no comparison data available
        
        st.markdown("---")
        st.markdown("### Customer Score Results")
        
        col1, col2 = st.columns(2)
        with col1:
            if "user_electricity" in st.session_state:
                st.metric("Personal Electricity Usage", f"{st.session_state.user_electricity:.2f} kWh")
            if "Environment_Score" in new_customer:
                st.metric("Company Environment Score", f"{new_customer['Environment_Score']:.2f}")
            
            # Water usage analysis with integrated visualization
            if 'water_units' in locals() and 'num_people' in locals() and 'usage_type' in locals() and num_people > 0:
                # Calculate monthly water consumption based on usage type
                if usage_type == "Daily Usage":
                    monthly_water = water_units * 30
                else:  # Monthly Usage
                    monthly_water = water_units
                
                # Calculate per person monthly consumption
                per_person_monthly = monthly_water / num_people
                
                # Average consumption: 130L/day/person = 3900L/month/person
                average_monthly_per_person = 130 * 30
                
                # Calculate Water Z-score (assuming standard deviation of 1000L for simplification)
                water_usage_z_score = (per_person_monthly - average_monthly_per_person) / 1000                
                if water_usage_z_score > 0:
                    st.warning(f"High water usage detected! Z-score: {water_usage_z_score:.2f} (Above average)")
                else:
                    st.success(f"Good water usage! Z-score: {water_usage_z_score:.2f} (Below or at average)")
                
                # Water usage comparison chart
                fig_water, ax_water = plt.subplots(figsize=(10, 6))
                # Data for comparison
                categories = ['Your Usage', 'Average Usage']
                values = [per_person_monthly, average_monthly_per_person]
                colors = ['#ff6b6b' if per_person_monthly > average_monthly_per_person else '#51cf66', '#74c0fc']
                # Create bar chart
                bars = ax_water.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                # Add value labels on bars
                for bar, value_label in zip(bars, values):
                    height = bar.get_height()
                    ax_water.text(bar.get_x() + bar.get_width()/2., height + 50,
                                f'{value_label:.0f}L', ha='center', va='bottom', fontweight='bold', fontsize=12)
                # Customize the chart
                ax_water.set_ylabel('Water Consumption (Litres/Month/Person)', fontsize=12, fontweight='bold')
                ax_water.set_title('Your Water Usage vs Average Usage', fontsize=14, fontweight='bold', pad=20)
                ax_water.grid(axis='y', alpha=0.3, linestyle='--')
                ax_water.set_ylim(0, max(values) * 1.2 if values else 100) # handle empty values
                # Add percentage difference annotation
                if average_monthly_per_person > 0: # Avoid division by zero
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
            
            st.metric("Z-Score", f"{z_score:.2f}")
            st.metric("Sustainability Score", f"{sust_score:.2f}")
            st.metric("Sustainability Rank", f"{sust_rank}")
            st.metric("Legacy Weighted Score", f"{weighted_score:.2f}")
            st.metric("Legacy Rank", f"{trad_rank}")

            # Get existing sustainability scores from scored_data if available
            if 'scored_data' in st.session_state and 'Sustainability_Score' in st.session_state.scored_data.columns:
                existing_sust = st.session_state.scored_data['Sustainability_Score']
                better_than = (existing_sust < sust_score).mean() * 100
                st.success(f"This customer performs better than **{better_than:.1f}%** of customers in the dataset (based on Z-Score)")
            else:
                st.info("No comparison data available in the dataset.")

            z_description = "above" if z_score > 0 else "below"
            st.info(f"Performance: **{abs(z_score):.2f} SD {z_description} mean**")
        
        with col2:
            # Ensure 'features' here refers to the list of features used in the model/weighting
            # This should align with `features_for_z_score` or a similar comprehensive list.
            # For simplicity, let's assume `st.session_state.features_list` holds this.
            model_features_list = st.session_state.get('features_list', list(z_vals.keys()))

            all_features_weights = {
                feat: abs(weights.get(feat, 0))
                for feat in model_features_list # Iterate over all potential features
                if feat in z_vals # Only include if z_val was calculated
            }
            
            # Ensure Environment_Score and Electricity are in all_features_weights if they have z_vals
            if "Environment_Score" in z_vals and "Environment_Score" not in all_features_weights:
                 all_features_weights['Environment_Score'] = abs(weights.get('Environment_Score',0.15)) # default if not in weights
            elif "Environment_Score" in all_features_weights and all_features_weights['Environment_Score'] == 0 :
                 all_features_weights['Environment_Score'] = 0.15


            if "Electricity" in z_vals and "Electricity" not in all_features_weights:
                all_features_weights['Electricity'] = abs(weights.get('Electricity',0.15)) # default if not in weights
            elif "Electricity" in all_features_weights and all_features_weights['Electricity'] == 0:
                 all_features_weights['Electricity'] = 0.15
            
            all_features_weights = {
                f: w for f, w in all_features_weights.items()
                if w > 0 # Only include features with a positive weight
            }

            if all_features_weights: # Check if dictionary is not empty
                total_weight_sum = sum(all_features_weights.values())
                if total_weight_sum > 0: # Avoid division by zero
                    contribs = {f: w / total_weight_sum for f, w in all_features_weights.items()}
                    explode = [0.1 if f in ['Environment_Score', 'Electricity'] else 0 for f in contribs.keys()]
                    
                    fig_pie_weights, ax_pie_weights = plt.subplots(figsize=(6,6))
                    ax_pie_weights.pie(
                        list(contribs.values()),
                        labels=list(contribs.keys()),
                        autopct='%1.1f%%',
                        startangle=90,
                        explode=explode
                    )
                    ax_pie_weights.set_title('Feature Weightage for Customer')
                    st.pyplot(fig_pie_weights)
                    
                    if "Environment_Score" in contribs:
                        st.info(f"Environment Score contributes {contribs['Environment_Score']*100:.1f}%")
                    if "Electricity" in contribs:
                        st.info(f"Electricity Usage contributes {contribs['Electricity']*100:.1f}%")
                else:
                    st.info("No feature weights to display for pie chart (total weight is zero).")
            else:
                st.info("No features with positive weights to display for pie chart.")

        # Display carbon footprint results if calculation has been done
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
                
                avg_emissions = 200  # Example average emissions
                if avg_emissions > 0: # Avoid division by zero
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

        # Define core_features locally as it's specific to this section
        # Ensure these feature names match those in `new_customer` and `weights`
        core_input_features = ['Water', 'MPCE', 'Public_Transport', 'Private_Transport', 'Electricity', 'Environment_Score']
        input_data = {}
        input_weight = {} # This will store the calculated weightage for the pie chart

        # Initialize input_data and a base for input_weight for core features
        for feat in core_input_features:
            input_data[feat] = new_customer.get(feat, 0) 
            input_weight[feat] = 0.01 # Start with a minimal weight for visibility


        for feat in features_for_z_score: # Iterate over features involved in scoring
            if feat in new_customer and feat in z_vals: # Ensure data and z_val exist
                input_data[feat] = new_customer[feat] # Ensure input_data has the latest
                model_weight = weights.get(feat, 0)
                # Contribution can be seen as |z_val * model_weight|
                # This reflects how much a feature's deviation impacts the score, scaled by its importance
                calculated_weight = abs(z_vals.get(feat, 0) * model_weight)
                if calculated_weight > 0: # Only consider features that contribute
                    input_weight[feat] = calculated_weight
                elif feat in core_input_features: # Keep minimal for core if no contribution
                     input_weight[feat] = max(input_weight.get(feat, 0), 0.01)
                # If not a core feature and no contribution, it can be omitted from pie chart or given minimal
                

        # Special handling for Electricity, MPCE, Environment_Score if their weighting is complex
        # The logic below seems to try and assign specific weights if they weren't captured above.
        # This might override the principled calculation based on z_vals and model_weights.
        # It's better to ensure their weights are derived consistently.

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
            elif new_customer["MPCE"] > 0: # Fallback
                 input_weight["MPCE"] = max(input_weight.get("MPCE",0.01), 0.15)


        if "Environment_Score" in new_customer and new_customer["Environment_Score"] is not None:
            input_data["Environment_Score"] = new_customer["Environment_Score"]
            if "Environment_Score" in z_vals and "Environment_Score" in weights:
                 input_weight["Environment_Score"] = max(input_weight.get("Environment_Score",0.01), abs(z_vals["Environment_Score"] * weights["Environment_Score"]))
            elif 'company_data' in locals() or 'company_data' in globals(): # Fallback logic from original
                if "Environment_Score" in company_data.columns:
                    mean = company_data["Environment_Score"].mean()
                    std = company_data["Environment_Score"].std()
                    if std > 0:
                        env_z_score_temp = (mean - new_customer["Environment_Score"])/std if inverse_scoring else (new_customer["Environment_Score"] - mean)/std
                        env_z_score_temp = np.clip(env_z_score_temp, -1, 1)
                        input_weight["Environment_Score"] = max(input_weight.get("Environment_Score",0.01), abs(env_z_score_temp * (weights.get("Environment_Score", 0.15)))) # Use 0.15 default model weight
                    else:
                        input_weight["Environment_Score"] = max(input_weight.get("Environment_Score",0.01),0.15)
            else:
                input_weight["Environment_Score"] = max(input_weight.get("Environment_Score",0.01),0.15)
        
        # Filter out zero or negligible weights for the pie chart, but ensure core features are shown if they have data
        final_input_weights_for_pie = {}
        for feat in core_input_features: # Prioritize core features
            if feat in input_data: # If core feature has data
                 final_input_weights_for_pie[feat] = max(input_weight.get(feat,0), 0.005) # Ensure miniscule for visibility if weight is 0
        
        for feat, wt in input_weight.items(): # Add other contributing features
            if wt > 0.005 : # Threshold for being included if not core
                 final_input_weights_for_pie[feat] = wt
            elif feat in core_input_features : # Ensure core features retain their (possibly tiny) weight
                 final_input_weights_for_pie[feat] = max(final_input_weights_for_pie.get(feat,0), wt)


        if final_input_weights_for_pie:
            total_pie_weight = sum(final_input_weights_for_pie.values())
            if total_pie_weight > 0:
                
                col1_input_w, col2_input_w = st.columns(2)
                
                with col1_input_w:
                    sorted_inputs_pie = sorted(final_input_weights_for_pie.items(), key=lambda x: x[1], reverse=True)
                    input_labels_pie = [f[0] for f in sorted_inputs_pie]
                    input_sizes_pie = [f[1] for f in sorted_inputs_pie]
                    
                    # Consolidate small slices into "Others" if too many items
                    if len(input_labels_pie) > 7: # For example, if more than 7 slices
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
                        showlegend=True, # Show legend for clarity if many items
                        legend_title_text='Features',
                        margin=dict(t=40, l=0, r=0, b=0),
                        height=400
                    )
                    st.plotly_chart(fig_pie_inputs, use_container_width=True)
                
                with col2_input_w:
                    input_table_data = []
                    # Use sorted_inputs_pie to maintain consistency with the pie chart
                    for feat, weight_val_pie in sorted_inputs_pie:
                        if feat == "Others": # Skip "Others" for the table if it was created
                            continue
                        contrib_pct_pie = (weight_val_pie / total_pie_weight) * 100 if total_pie_weight > 0 else 0.00
                        input_table_data.append({
                            "Input Field": feat,
                            "Value": input_data.get(feat, "N/A"), # Get value from input_data
                            "Influence Score": weight_val_pie, # This is the value used in the pie
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

        # --- INTEGRATED DATA INSIGHTS ---
        st.markdown("---")
        st.markdown("## Data Insights (Current Customer Data)")

        # Use the same data sources as in evaluate customer
        if new_customer and z_vals and 'weights' in st.session_state:
            col1_insights, col2_insights = st.columns(2)
            with col1_insights:
                st.markdown("### Customer Feature Analysis")
                
                # Create a summary of customer data
                customer_summary = {}
                for feat, val in new_customer.items():
                    if feat in z_vals:
                        customer_summary[feat] = {
                            'Value': val,
                            'Z-Score': z_vals[feat],
                            'Weight': st.session_state.weights.get(feat, 0)
                        }
                
                # Add electricity from session state if available
                if "user_electricity" in st.session_state and "Electricity" in z_vals:
                    customer_summary["Electricity"] = {
                        'Value': st.session_state.user_electricity,
                        'Z-Score': z_vals["Electricity"],
                        'Weight': st.session_state.weights.get("Electricity", 0)
                    }
                
                if customer_summary:
                    # Create DataFrame for display
                    summary_df = pd.DataFrame(customer_summary).T
                    st.write("Customer Feature Summary:")
                    st.dataframe(summary_df)
                    
                    # Feature importance chart
                    fig_importance, ax_importance = plt.subplots(figsize=(8, 6))
                    features_list = list(customer_summary.keys())
                    weights_list = [customer_summary[f]['Weight'] for f in features_list]
                    
                    ax_importance.barh(features_list, weights_list, color='skyblue', edgecolor='black')
                    ax_importance.set_xlabel('Feature Weight')
                    ax_importance.set_title('Feature Importance in Customer Evaluation')
                    ax_importance.grid(axis='x', alpha=0.3)
                    st.pyplot(fig_importance)
                else:
                    st.warning("No customer feature data available for analysis.")
                
            with col2_insights:
                st.markdown("### Z-Score Analysis")
                
                if z_vals:
                    # Z-score visualization
                    fig_z, ax_z = plt.subplots(figsize=(8, 6))
                    z_features = list(z_vals.keys())
                    z_values = list(z_vals.values())
                    
                    # Color bars based on z-score values
                    colors = ['red' if z > 0 else 'green' for z in z_values]
                    
                    bars = ax_z.bar(z_features, z_values, color=colors, alpha=0.7, edgecolor='black')
                    ax_z.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                    ax_z.set_ylabel('Z-Score')
                    ax_z.set_title('Customer Z-Scores by Feature')
                    ax_z.grid(axis='y', alpha=0.3)
                    
                    # Rotate x-axis labels for better readability
                    plt.setp(ax_z.get_xticklabels(), rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, z_values):
                        height = bar.get_height()
                        ax_z.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
                                f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                                fontweight='bold', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig_z)
                    
                    # Z-score interpretation
                    st.write("**Z-Score Interpretation:**")
                    st.write("- **Green bars (negative)**: Below average (better for sustainability)")
                    st.write("- **Red bars (positive)**: Above average (higher consumption)")
                    st.write("- Values beyond ±2 indicate significant deviation from average")
                    
                else:
                    st.warning("No Z-score data available for analysis.")
        else:
            st.warning("Customer evaluation data not available. Please evaluate a customer first.")
                          
        
    
    

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
