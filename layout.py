from dash import dcc, html
import pandas as pd

# --- Theme Styles ---
THEME = {
    'background': '#0B041A',
    'text': '#EAEAEA',
    'primary': '#FFFFFF',
    'container_bg': 'rgba(30, 15, 60, 0.3)',
    'container_border': 'rgba(255, 255, 255, 0.1)'
}

# --- Zoom levels ---
ZOOM_LEVELS = {'in': 2.0, 'default': 1.5, 'out': 0.5}

def create_layout(screener_data_df):
    """Creates the layout for the Dash app."""
    if not screener_data_df.empty:
        ticker_options = [{'label': row['name'] + f" ({row['code']})", 'value': row['code']} for index, row in screener_data_df.iterrows()]
        default_ticker = 'AAPL' if 'AAPL' in screener_data_df['code'].values else screener_data_df['code'].iloc[0]
    else:
        ticker_options = []
        default_ticker = None

    starry_background_style = {
        'backgroundColor': THEME['background'],
        'backgroundImage': 'radial-gradient(circle, white 0.5px, transparent 1.5px), radial-gradient(circle, white 1px, transparent 2px), radial-gradient(circle, white 0.5px, transparent 1.5px)',
        'backgroundSize': '350px 350px, 250px 250px, 150px 150px',
        'backgroundPosition': '0 0, 40px 60px, 130px 270px',
        'color': THEME['text'],
        'fontFamily': "'Space Grotesk', sans-serif",
        'minHeight': '100vh',
        'padding': '10px 20px'
    }
    
    font_style = {'fontFamily': "'Space Grotesk', sans-serif", 'color': THEME['text']}
    
    LOGO_URL = "https://storage.googleapis.com/financial_observatory_public/assets/Logo_rectangle.PNG"

    # --- MODIFIED: Added box-sizing and a slight top padding to optically center the icons ---
    button_style = { 'background': 'rgba(40, 40, 40, 0.7)', 'border': '1px solid rgba(255, 255, 255, 0.3)', 'color': 'white', 'fontSize': '18px', 'width': '35px', 'height': '35px', 'borderRadius': '50%', 'cursor': 'pointer', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'paddingTop': '2px', 'boxSizing': 'border-box' }
    
    floating_controls = html.Div([
        html.Button('-', id='zoom-out-btn', n_clicks=0, style=button_style),
        html.Button('⛶', id='reset-zoom-btn', n_clicks=0, style=button_style),
        html.Button('+', id='zoom-in-btn', n_clicks=0, style=button_style),
    ], style={ 'position': 'absolute', 'bottom': '20px', 'left': '50%', 'zIndex': 10, 'transform': 'translateX(-50%)', 'display': 'flex', 'flexDirection': 'row', 'gap': '10px' })


    layout = html.Div(style=starry_background_style, children=[
        dcc.Store(id='scroll-trigger-store'),
        dcc.Store(id='zoom-level-store', data=ZOOM_LEVELS['default']),
        dcc.Store(id='processed-data-store'), 
        dcc.Store(id='source-data-store'),
        
        html.Div([
            html.Img(src=LOGO_URL, style={'height': '50px', 'marginRight': '20px'}),
            html.H1(["THE FINANCIAL", html.Br(), "OBSERVATORY"], style={**font_style, 'fontSize': '20px', 'fontWeight': 'bold', 'letterSpacing': '4px', 'margin': '0', 'lineHeight': '1.1'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '0'}),

        html.Div([
            html.P("Search for companies or symbols:", style={'fontSize': 'small', 'color': 'white', 'textAlign': 'center', 'marginBottom': '5px', 'margin': 0}),
            dcc.Dropdown(id='ticker-dropdown', options=ticker_options, value=default_ticker, clearable=False, style={'height': '40px', 'color': 'black'})
        ], style={'width': '90%', 'maxWidth': '500px', 'margin': '15px auto 10px', 'backgroundColor': THEME['container_bg'], 'border': f"1px solid {THEME['container_border']}", 'borderRadius': '12px', 'backdropFilter': 'blur(10px)', 'padding': '10px', 'position': 'relative', 'zIndex': 20}),

        html.Div(style={'position': 'relative', 'height': '50vh', 'width': '98%', 'margin': 'auto'}, children=[
            dcc.Graph(id='network-graph', style={'height': '100%', 'borderRadius': '15px', 'boxShadow': '0 0 25px 5px rgba(255, 255, 255, 0.15)'}),
            floating_controls
        ]),

        html.P(id='prediction-summary-text', style={'textAlign': 'center', 'padding': '15px 0 0 0', 'fontSize': '16px', 'color': THEME['text']}),

        html.Div(id='info-panels-container'),

        html.Footer(
            dcc.Link('Disclaimer', href='/disclaimer', style={
                'color': THEME['text'], 
                'textDecoration': 'underline',
                'opacity': 0.7
            }),
            style={'textAlign': 'center', 'padding': '40px 0 20px 0'}
        )
    ])
    
    return layout
