# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import json
from datetime import datetime

import dash
from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors

# --- App Pages ---
from layout import create_layout as create_dashboard_layout
from pages import disclaimer
# --- Utility/Processing functions ---
from processing import process_and_score_stocks
from visuals import solar_system_visual, create_model_image_svg, get_node_color


# ==============================================================================
# 2. CONFIGURATION & DATA LOADING
# ==============================================================================
# --- Theme Styles ---
THEME = {
    'background': '#0B041A',
    'text': '#EAEAEA',
    'primary': '#FFFFFF',
    'container_bg': 'rgba(30, 15, 60, 0.3)',
    'container_border': 'rgba(255, 255, 255, 0.1)'
}

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'solar_system_bucket')
min_nodes = 5
max_nodes = 20
threshold_percent = 0.9
# Zoom levels
ZOOM_LEVELS = {'in': 2.0, 'default': 1.5, 'out': 0.5}


# --- Load Data from Google Cloud Storage ---
try:
    print("Loading data from GCS...")
    base_path = f'gs://{GCS_BUCKET_NAME}'
    
    three_month_spearman_lagged_correlations = pd.read_csv(f'{base_path}/three_month_spearman_lagged_correlation.csv', index_col=0)
    six_month_spearman_lagged_correlations = pd.read_csv(f'{base_path}/six_month_spearman_lagged_correlation.csv', index_col=0)
    screener_data_df = pd.read_csv(f'{base_path}/screener_data_df.csv')
    gravitational_impact_df = pd.read_csv(f'{base_path}/gravitational_impact_df.csv')
    
    print("Successfully loaded all dataframes.")

except Exception as e:
    print(f"CRITICAL ERROR: Failed to load data from GCS. App may not function. Error: {e}")
    # Create empty dataframes so the app doesn't crash on startup
    three_month_spearman_lagged_correlations = pd.DataFrame()
    six_month_spearman_lagged_correlations = pd.DataFrame()
    screener_data_df = pd.DataFrame()
    gravitational_impact_df = pd.DataFrame()
    
# --- Color Maps ---
# Planet Color Maps
RED_SPECTRUM = {'light': '#FF0000', 'dark': '#8E0000'}
GREEN_SPECTRUM = {'dark': '#1B9D49', 'light': '#A1FF61'}
BLUE_SPECTRUM = {'dark': '#4C216D', 'light': '#9751CB'}

red_cmap = mcolors.LinearSegmentedColormap.from_list('red_cmap', [RED_SPECTRUM['dark'], RED_SPECTRUM['light']])
green_cmap = mcolors.LinearSegmentedColormap.from_list('green_cmap', [GREEN_SPECTRUM['dark'], GREEN_SPECTRUM['light']])
blue_cmap = mcolors.LinearSegmentedColormap.from_list('blue_cmap', [BLUE_SPECTRUM['dark'], BLUE_SPECTRUM['light']])

# Star Color Map
star_cmap = mcolors.LinearSegmentedColormap.from_list('star_cmap', ['#FF0000', '#F7D117', '#159BFF'])


# ==============================================================================
# 3. DASH APP INITIALIZATION
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

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# ==============================================================================
# 4. ROUTER CALLBACK
# ==============================================================================
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/disclaimer':
        return disclaimer.layout
    else:
        return create_dashboard_layout(screener_data_df)

# ==============================================================================
# 5. DASHBOARD-SPECIFIC CALLBACKS
# ==============================================================================

@app.callback(
    [Output('info-panels-container', 'children'),
     Output('processed-data-store', 'data'),
     Output('source-data-store', 'data'),
     Output('scroll-trigger-store', 'data')],
    [Input('ticker-dropdown', 'value')]
)
def update_panels_and_data_on_ticker_change(selected_ticker):
    if not selected_ticker: raise PreventUpdate
    processed_data_df, source_data_df = process_and_score_stocks(six_month_spearman_lagged_correlations, three_month_spearman_lagged_correlations, screener_data_df, selected_ticker, min_nodes, max_nodes, threshold_percent)
    if processed_data_df.empty or source_data_df.empty: 
        return html.Div(f"Data not available for {selected_ticker}", style={'textAlign': 'center', 'padding': '20px'}), None, None, datetime.now()

    def format_market_cap(market_cap): return f"${market_cap / 1e12:.2f}T" if market_cap >= 1e12 else f"${market_cap / 1e9:.2f}B"
    is_weekend = datetime.today().weekday() >= 5
    star_info_screener = screener_data_df[screener_data_df['code'] == selected_ticker].iloc[0]
    star_info_source = source_data_df[source_data_df['ticker'] == selected_ticker].iloc[0]
    grav_impact = star_info_source.get('gravitational_impact', 0)
    net_grav_force = star_info_source.get('net_gravitational_force', 0)
    max_potential_force = star_info_source.get('max_potential_force', 0)
    planets_df = processed_data_df[processed_data_df['source'] == selected_ticker].copy()
    prediction_day_text = "on Monday" if is_weekend else "today"
    daily_change_header = ["Friday's", html.Br(), "Daily Close"] if is_weekend else ["Yesterday's", html.Br(), "Daily Change"]
    
    trends_header_text = "Monday's Top Trends" if is_weekend else "Top Trends"
    source_name = star_info_screener.get('name',selected_ticker)
    
    trend_direction = "an upward" if grav_impact >= 0 else "a downward"
    prediction_summary = f"{source_name} ({selected_ticker}) is showing {trend_direction} trend with a strength of {grav_impact:.0f}% {prediction_day_text}."
    
    star_color = get_node_color(grav_impact, True, star_cmap=star_cmap)
    star_image_src = create_model_image_svg(star_color, 2, 10)
    container_style = {'backgroundColor': THEME['container_bg'], 'border': f"1px solid {THEME['container_border']}", 'padding': '5px', 'borderRadius': '12px', 'backdropFilter': 'blur(10px)', 'width': '100%', 'boxSizing': 'border-box'}
    header_style = {'fontFamily': "'Space Grotesk', sans-serif", 'color': THEME['text'], 'textAlign': 'center', 'fontWeight': 'bold', 'marginTop': 15, 'marginBottom': '15px', 'fontSize': '22px'}
    def create_definition_row(label, text): return html.Div([html.Div(label, style={'fontWeight': 'bold', 'width': '150px', 'flexShrink': 0}), html.Div(text, style={'flexGrow': 1, 'fontSize': '14px'})], style={'display': 'flex', 'marginBottom': '10px'})
    
    divider = html.Hr(style={'border': 'none', 'borderTop': f"1px solid {THEME['container_border']}", 'margin': '20px auto', 'width': '90%'})

    horizontal_mask = 'linear-gradient(to right, black 90%, transparent 100%)'
    star_table_wrapper_style = { 'overflowX': 'auto', 'maskImage': horizontal_mask, 'WebkitMaskImage': horizontal_mask, 'backgroundColor': 'rgba(30, 15, 60, 0.5)', 'padding': '10px', 'borderRadius': '8px' }

    star_info_panel = html.Div([
        html.H3("Star Information", style=header_style),
        html.P(prediction_summary, style={'textAlign': 'center', 'fontWeight': 'bold', 'paddingBottom': '15px', 'fontSize': '16px'}),
        
        html.Div([create_definition_row("Star Size:", "The size of the star is based on the market capitalization."), create_definition_row("Max Grav. Force:", "The theoretical sum of all gravitational forces from the planets acting on the star, assuming all planets have perfect historical correlations."), create_definition_row("Net Grav. Force:", "The current sum of all gravitational forces from the planets acting on the star."), create_definition_row("Trend Strength:", "The Net Gravitational Force shown as a percentage of the Maximum Gravitational Force, indicating directional pressure."), create_definition_row("Star Color:", "The color of the star is based on the Trend Strength.")], style={'marginBottom': '20px'}),
        html.Div([html.Div(style={'height': '15px', 'borderRadius': '5px', 'background': 'linear-gradient(to right, #FF0000, #F7D117, #159BFF)'}), html.Div([html.Span("Decrease", style={'color': 'white', 'fontSize': '12px'}), html.Span("Stable", style={'color': 'white', 'fontSize': '12px'}), html.Span("Increase", style={'color': 'white', 'fontSize': '12px'})], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '5px'})], style={'width': '90%', 'margin': '0 auto 20px auto'}),
        
        divider,
        
        html.Div(html.Table([html.Thead(html.Tr([
            html.Th("Code", style={'textAlign': 'left', 'padding': '8px', 'borderBottom': f"2px solid {THEME['container_border']}"}), 
            html.Th("Name", style={'textAlign': 'left', 'padding': '8px', 'borderBottom': f"2px solid {THEME['container_border']}"}), 
            html.Th("Market Cap", style={'textAlign': 'left', 'padding': '8px', 'borderBottom': f"2px solid {THEME['container_border']}"}), 
            html.Th("Max Grav. Force", style={'textAlign': 'left', 'padding': '8px', 'borderBottom': f"2px solid {THEME['container_border']}"}), 
            html.Th("Net Grav. Force", style={'textAlign': 'left', 'padding': '8px', 'borderBottom': f"2px solid {THEME['container_border']}"}), 
            html.Th("Trend Strength", style={'textAlign': 'left', 'padding': '8px', 'borderBottom': f"2px solid {THEME['container_border']}"})
            ])), html.Tbody([html.Tr([
                html.Td(html.Div([html.Img(src=star_image_src, style={'height':'40px', 'width':'40px', 'marginRight':'10px'}), selected_ticker], style={'display':'flex', 'alignItems':'center'}), style={'padding': '8px'}), 
                html.Td(source_name, style={'padding': '8px'}), 
                html.Td(format_market_cap(star_info_screener.get('market_capitalization', 0)), style={'padding': '8px'}), 
                html.Td(f"{max_potential_force:.2f}", style={'padding': '8px'}), 
                html.Td(f"{net_grav_force:.2f}", style={'padding': '8px'}), 
                html.Td(f"{grav_impact:.2f}%", style={'padding': '8px'})
            ], id={'type': 'star-row', 'index': selected_ticker}, n_clicks=0, style={'cursor': 'pointer'})])], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '14px'}), style=star_table_wrapper_style),
        
        html.Div(html.A("🔍 Live Data & News", href=f"https://www.google.com/search?q=NASDAQ%3A{selected_ticker}", target="_blank", style={'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '10px 20px', 'backgroundColor': 'white', 'color': '#374151', 'borderRadius': '9999px', 'textDecoration': 'none', 'fontWeight': 'bold', 'fontSize': '16px', 'boxShadow': '0 4px 14px 0 rgba(0, 118, 255, 0.39)'}), style={'textAlign': 'center', 'paddingBottom': '15px'})
    ], style=container_style)
    
    top_positive_impacts = gravitational_impact_df.sort_values(by='gravitational_impact', ascending=False).head(10).reset_index(drop=True)
    top_negative_impacts = gravitational_impact_df.sort_values(by='gravitational_impact', ascending=True).head(10).reset_index(drop=True)
    if not top_positive_impacts.empty and not top_negative_impacts.empty:
        combined_impacts = pd.concat([top_positive_impacts.head(5),top_negative_impacts.head(5)])
        def create_prediction_item(row):
            ticker, name = row['ticker'], screener_data_df[screener_data_df['code'] == row['ticker']].iloc[0]['name'] if not screener_data_df[screener_data_df['code'] == row['ticker']].empty else row['ticker']
            return html.Div([html.Span(f"{name} ({ticker})"),html.Span(f"{row['gravitational_impact']:.2f}%",style={'color':'#4ade80' if row['gravitational_impact']>0 else '#f87171','fontWeight':'bold'})],id={'type':'prediction-item','index':ticker},n_clicks=0,style={'display':'flex','justifyContent':'space-between','padding':'8px 0','borderBottom':f'1px solid {THEME["container_border"]}','cursor':'pointer'})
        prediction_items = [create_prediction_item(row) for _, row in combined_impacts.iterrows()]
    else: prediction_items = [html.Div("Top predictions not available.")]
    
    prediction_list_style = {
        'backgroundColor': 'rgba(30, 15, 60, 0.5)',
        'padding': '10px',
        'borderRadius': '8px',
    }
    predictions_panel = html.Div([
        html.H3(trends_header_text,style=header_style),
        html.Div(prediction_items, style=prediction_list_style)
    ],style=container_style)

    planet_headers = ["Code", "Name", f"Correlation with {selected_ticker}", "Market Cap", daily_change_header, "Grav. Force"]
    planet_table_header = [html.Thead(html.Tr([html.Th(col, style={'padding': '12px', 'textAlign': 'left', 'borderBottom': f"2px solid {THEME['container_border']}"}) for col in planet_headers]))]
    planet_table_rows = []
    if not planets_df.empty:
        for _,p_row in planets_df.iterrows():
            s_info = screener_data_df[screener_data_df['code']==p_row['target']].iloc[0]
            planet_color = get_node_color(p_row['Daily Change'], False, red_cmap=red_cmap, green_cmap=green_cmap, blue_cmap=blue_cmap)
            planet_image_src = create_model_image_svg(planet_color, 2, 5)
            ticker_cell = html.Div([html.Img(src=planet_image_src,style={'height':'40px','width':'40px','marginRight':'10px'}),html.Span(p_row['target'])],style={'display':'flex','alignItems':'center'})
            planet_table_rows.append(html.Tr([
                html.Td(ticker_cell, style={'padding':'8px 12px', 'borderBottom': f'1px solid {THEME["container_border"]}'}), 
                html.Td(s_info['name'], style={'padding':'8px 12px', 'borderBottom': f'1px solid {THEME["container_border"]}'}), 
                html.Td(f"{p_row['unified_correlation']:.2%}", style={'padding':'8px 12px', 'borderBottom': f'1px solid {THEME["container_border"]}'}), 
                html.Td(format_market_cap(s_info['market_capitalization']), style={'padding':'8px 12px', 'borderBottom': f'1px solid {THEME["container_border"]}'}), 
                html.Td(f"{p_row['Daily Change']:.2f}%", style={'padding':'8px 12px', 'borderBottom': f'1px solid {THEME["container_border"]}'}), 
                html.Td(f"{p_row['signed_gravitational_force']:.2f}", style={'padding':'8px 12px', 'borderBottom': f'1px solid {THEME["container_border"]}'})
            ], id={'type': 'planet-row', 'index': p_row['target']}, n_clicks=0, style={'cursor': 'pointer'}))
    
    base_horizontal_mask = 'linear-gradient(to right, black 90%, transparent 100%)'
    planet_table_wrapper_style = { 'overflowX': 'auto', 'maskImage': base_horizontal_mask, 'WebkitMaskImage': base_horizontal_mask, 'backgroundColor': 'rgba(30, 15, 60, 0.5)', 'padding': '10px', 'borderRadius': '8px' }
    if len(planets_df) > 5:
        vertical_mask = 'linear-gradient(to bottom, black 90%, transparent 100%)'
        planet_table_wrapper_style.update({ 'maxHeight': '300px', 'overflowY': 'auto', 'maskImage': f'{base_horizontal_mask}, {vertical_mask}', 'WebkitMaskImage': f'{base_horizontal_mask}, {vertical_mask}', 'maskComposite': 'intersect', 'WebkitMaskComposite': 'source-in' })
    
    def create_color_bar(label, start_color, end_color): return html.Div([html.Div(style={'height': '15px', 'borderRadius': '5px', 'background': f'linear-gradient(to right, {start_color}, {end_color})'}), html.Div(label, style={'color': 'white', 'fontSize': '12px', 'textAlign': 'center', 'marginTop': '5px'})], style={'flex': 1})
    stable_bar = html.Div([html.Div(style={'height': '15px', 'borderRadius': '5px', 'background': f"linear-gradient(to right, {BLUE_SPECTRUM['light']}, {BLUE_SPECTRUM['dark']}, {BLUE_SPECTRUM['light']})"}), html.Div('Stable', style={'color': 'white', 'fontSize': '12px', 'textAlign': 'center', 'marginTop': '5px'})], style={'flex': 1})
    planet_table_panel = html.Div([
        html.H3("Planet Information", style=header_style),
        html.Div([create_definition_row("Planet Size:", "The size of the planet is based on the market capitalization."), create_definition_row("Correlation:", "The correlation is calculated by comparing the planet's price to the star's following day price over a given time period."), create_definition_row("Orbital Radius:", "Represents the gravitational force, which is derived from market cap and correlation."), create_definition_row("Planet Color:", "The color of the planet is based on yesterday's change in stock price.")], style={'marginBottom': '20px'}),
        html.Div([create_color_bar('Decrease', RED_SPECTRUM['light'], RED_SPECTRUM['dark']), stable_bar, create_color_bar('Increase', GREEN_SPECTRUM['dark'], GREEN_SPECTRUM['light'])], style={'display': 'flex', 'gap': '10px', 'width': '90%', 'margin': '0 auto 20px auto'}),
        
        divider,

        html.Div(html.Table(planet_table_header + [html.Tbody(planet_table_rows)],style={'width':'100%','borderCollapse':'collapse'}), style=planet_table_wrapper_style)
    ], style=container_style)
    info_panels = html.Div([star_info_panel, planet_table_panel, predictions_panel], style={'display':'flex','flexDirection':'column','gap':'10px','padding':'10px 5px'})
    processed_data_json = processed_data_df.to_json(date_format='iso', orient='split')
    source_data_json = source_data_df.to_json(date_format='iso', orient='split')
    return info_panels, processed_data_json, source_data_json, datetime.now()

@app.callback(
    Output('zoom-level-store', 'data'),
    [Input('zoom-in-btn', 'n_clicks'),
     Input('reset-zoom-btn', 'n_clicks'),
     Input('zoom-out-btn', 'n_clicks')],
    State('zoom-level-store', 'data'),
    prevent_initial_call=True
)
def handle_zoom_buttons(in_clicks, reset_clicks, out_clicks, current_zoom):
    ctx = callback_context
    if not ctx.triggered: raise PreventUpdate
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'zoom-in-btn': new_zoom = max(ZOOM_LEVELS['out'], current_zoom - 0.4)
    elif button_id == 'zoom-out-btn': new_zoom = min(ZOOM_LEVELS['in'], current_zoom + 0.4)
    elif button_id == 'reset-zoom-btn': new_zoom = ZOOM_LEVELS['default']
    else: raise PreventUpdate
    if new_zoom == current_zoom: raise PreventUpdate
    return new_zoom

@app.callback(
    Output('network-graph', 'figure'), 
    [Input('processed-data-store', 'data'),
     Input('source-data-store', 'data'),
     Input('zoom-level-store', 'data')],
    State('ticker-dropdown', 'value'),
    prevent_initial_call=True
)
def update_graph(processed_data_json, source_data_json, zoom_level, selected_ticker):
    if not all([processed_data_json, source_data_json, zoom_level, selected_ticker]): raise PreventUpdate
    processed_data_df = pd.read_json(processed_data_json, orient='split')
    source_data_df = pd.read_json(source_data_json, orient='split')
    if processed_data_df.empty: raise PreventUpdate
    figure = solar_system_visual(selected_ticker, processed_data_df, source_data_df, screener_data_df, zoom_level, star_cmap, red_cmap, green_cmap, blue_cmap)
    return figure

@app.callback(
    Output('ticker-dropdown', 'value'),
    [Input({'type': 'prediction-item', 'index': ALL}, 'n_clicks'),
     Input({'type': 'planet-row', 'index': ALL}, 'n_clicks'),
     Input({'type': 'star-row', 'index': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_dropdown_from_item_click(prediction_clicks, planet_clicks, star_clicks):
    all_clicks = (prediction_clicks or []) + (planet_clicks or []) + (star_clicks or [])
    if not any(all_clicks):
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
    if not triggered_id_str:
        raise PreventUpdate
    
    try:
        triggered_id = json.loads(triggered_id_str)
        new_ticker = triggered_id.get('index')
        if new_ticker:
            return new_ticker
        else:
            raise PreventUpdate
    except (json.JSONDecodeError, KeyError):
        raise PreventUpdate


app.clientside_callback(
    """
    function(trigger) {
        if (trigger) {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('scroll-trigger-store', 'id'), # Dummy output, does not affect anything
    Input('scroll-trigger-store', 'data')
)

# ==============================================================================
# 6. MAIN ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
