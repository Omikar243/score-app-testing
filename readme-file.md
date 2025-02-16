# Feature Scoring Application

This streamlit application generates synthetic data for different features and calculates scores based on selected features.

## Features

- Generate synthetic data with configurable ranges
- Select from predefined features (Electricity, Water, Gas, Waste)
- Automatic weight calculation based on selected features
- Leaderboard ranking of records by score
- Score statistics and distribution visualisation
- CSV upload functionality for custom data
- Download results as CSV

## Installation

1. Ensure you have Python 3.12 or higher installed
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Execute the following command in your terminal:

```bash
streamlit run app.py
```

The application will start and automatically open in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501.

## Usage Instructions

1. **Configure Data Generation**:
   - Set the number of records using the slider
   - Select features by checking the corresponding checkboxes
   - Adjust the min/max ranges for each selected feature

2. **Generate Data**:
   - Click the "Generate Data" button to create synthetic data and calculate scores
   - View the leaderboard, statistics, and score distribution
   - Download the results using the "Download CSV" button

3. **Upload Custom Data**:
   - Alternatively, upload your own CSV file
   - Ensure your CSV contains columns matching the selected features
   - View and download the scored results

## Weight Calculation

- The application automatically assigns equal weights to all selected features
- The sum of weights always equals 1.0
- For two features: w1=0.5, w2=0.5
- For three features: w1=0.33, w2=0.33, w3=0.33
- For four features: w1=0.25, w2=0.25, w3=0.25, w4=0.25

## Error Handling

The application validates that:
- At least one feature is selected
- The sum of weights equals 1.0
- Uploaded CSV files contain all required features


