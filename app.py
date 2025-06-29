# filename: app.py
import os
import pandas as pd
from urllib.parse import urlparse, parse_qs

import dash
from dash import Dash, dcc, html, Input, Output, State

# --- Import page layouts and callback registration functions ---
from pages import homepage, planetarysystem, disclaimer

# ==============================================================================
# 1. CONFIGURATION & DATA LOADING
# ==============================================================================
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'solar_system_bucket')

try:
    # Removed print("Attempting to load data from GCS for main app...")
    # Removed print(f"GCS_BUCKET_NAME retrieved: '{GCS_BUCKET_NAME}'")
    base_path = f'gs://{GCS_BUCKET_NAME}'
    # Removed print(f"Constructed GCS base path: '{base_path}'")
    
    # Removed print("Loading three_month_spearman_lagged_correlation.csv...")
    three_month_spearman_lagged_correlations = pd.read_csv(f'{base_path}/three_month_spearman_lagged_correlation.csv', index_col=0)
    # Removed print("Loaded three_month_spearman_lagged_correlation.csv successfully.")

    # Removed print("Loading six_month_spearman_lagged_correlation.csv...")
    six_month_spearman_lagged_correlations = pd.read_csv(f'{base_path}/six_month_spearman_lagged_correlation.csv', index_col=0)
    # Removed print("Loaded six_month_spearman_lagged_correlation.csv successfully.")

    # Removed print("Loading screener_data_df.csv...")
    screener_data_df = pd.read_csv(f'{base_path}/screener_data_df.csv')
    # Removed print("Loaded screener_data_df.csv successfully.")

    # Removed print("Loading gravitational_impact_df.csv...")
    gravitational_impact_df = pd.read_csv(f'{base_path}/gravitational_impact_df.csv')
    # Removed print("Loaded gravitational_impact_df.csv successfully.")
    
    print("Successfully loaded all dataframes for main app.")

except Exception as e:
    print(f"CRITICAL ERROR: Failed to load data from GCS. App may not function. Error: {e}")
    # Create empty dataframes so the app doesn't crash on planettup
    three_month_spearman_lagged_correlations = pd.DataFrame()
    six_month_spearman_lagged_correlations = pd.DataFrame()
    screener_data_df = pd.DataFrame()
    gravitational_impact_df = pd.DataFrame()
    # Removed print("Dataframes set to empty due to loading error.")

# ==============================================================================
# 2. DASH APP INITIALIZATION
# ==============================================================================
app = Dash(
    __name__, 
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&display=swap',
        'data:text/css,body%7Bmargin:0;padding:0%7D'
    ], 
    title='Financial Observatory',
    suppress_callback_exceptions=True
)
server = app.server

# ==============================================================================
# 3. APP LAYOUT & ROUTING
# ==============================================================================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')],
    [State('url', 'search')]
)
def display_page(pathname, search):
    """
    Acts as a router. Based on the URL pathname, it serves the correct page layout.
    """
    if pathname == '/disclaimer':
        return disclaimer.layout
    
    elif pathname == '/planetarysystem':
        # Check if a ticker is present in the URL query string
        if search:
            query_params = parse_qs(urlparse(search).query)
            ticker = query_params.get('ticker', [None])[0]
            if ticker:
                # If a ticker exists, render the solar system layout.
                # The layout function no longer needs the ticker passed to it.
                # A callback within planetarysystem.py will handle setting the value.
                return planetarysystem.create_layout(screener_data_df)
        
        # If no ticker is provided for the /planetarysystem route, redirect to the homepage.
        return dcc.Location(pathname="/", id="redirect-to-home")

    # The default route is the homepage.
    else:
        return homepage.create_layout(screener_data_df)

# ==============================================================================
# 4. REGISTER PAGE-SPECIFIC CALLBACKS
# ==============================================================================
homepage.register_homepage_callbacks(
    app,
    screener_data_df,
    gravitational_impact_df
)
planetarysystem.register_solarsystem_callbacks(
    app, 
    screener_data_df, 
    three_month_spearman_lagged_correlations, 
    six_month_spearman_lagged_correlations, 
    gravitational_impact_df
)

# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
