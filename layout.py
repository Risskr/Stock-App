# filename: layout.py
from dash import dcc, html
import pandas as pd

# --- Zoom levels ---
ZOOM_LEVELS = {'in': 2.0, 'default': 1.0, 'out': 0.5}

def create_layout(screener_data_df):
    """
    Creates the layout for the Dash app.
    Styling is now handled by style.css.
    """
    if not screener_data_df.empty and 'market_capitalization' in screener_data_df.columns:
        sorted_df = screener_data_df.sort_values('market_capitalization', ascending=False)
        ticker_options = [{'label': f"{row['name']} ({row['code']})", 'value': row['code']} 
                          for index, row in sorted_df.iterrows()]
        default_ticker = 'AAPL' if 'AAPL' in screener_data_df['code'].values else screener_data_df['code'].iloc[0]
    else:
        ticker_options = []
        default_ticker = None
    
    LOGO_URL = "https://storage.googleapis.com/financial_observatory_public/assets/Logo_rectangle.PNG"

    floating_controls = html.Div([
        html.Button('-', id='zoom-out-btn', n_clicks=0, className='zoom-button'),
        html.Button('⛶', id='reset-zoom-btn', n_clicks=0, className='zoom-button'),
        html.Button('+', id='zoom-in-btn', n_clicks=0, className='zoom-button'),
    ], className='floating-controls')

    layout = html.Div(className="planetary-background", children=[
        dcc.Store(id='scroll-trigger-store'),
        dcc.Store(id='zoom-level-store', data=ZOOM_LEVELS['default']),
        dcc.Store(id='processed-data-store'), 
        dcc.Store(id='source-data-store'),
        
        dcc.Link(
            href='/',
            className="header-logo-link",
            children=[
                html.Div([
                    html.Img(src=LOGO_URL, className='header-logo-image'),
                    html.H1(["THE FINANCIAL", html.Br(), "OBSERVATORY"], className='header-title')
                ], className='header-logo-container')
            ]
        ),

        html.Div([
            html.P("Search for companies or symbols:", className='dropdown-label'),
            dcc.Dropdown(
                id='ticker-dropdown', 
                options=ticker_options, 
                value=default_ticker, 
                clearable=False, 
                placeholder="Select or type a company name or symbol…",
                style={'height': '40px', 'color': 'black'}
            )
        ], className='dropdown-container'),

        html.Div(className='graph-container', children=[
            dcc.Graph(id='network-graph', className='network-graph'),
            floating_controls
        ]),

        html.Div(id='info-panels-container'),

        html.Footer(
            dcc.Link('Disclaimer', href='/disclaimer', className='footer-link'),
            className='app-footer'
        )
    ])
    
    return layout
