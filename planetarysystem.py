# pages/planetarysystem.py 
import os
import json
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import dash
from dash import html, dcc, Input, Output, State, ALL, MATCH, callback_context
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors

from processing import process_and_score_stocks
from visuals import solar_system_visual, create_procedural_sphere, get_node_color, create_model_image_svg
from layout import create_layout as create_dashboard_layout, ZOOM_LEVELS

# ==============================================================================
#  LAYOUT CREATION
# ==============================================================================
def create_layout(screener_data_df):
    """
    This function creates the layout for the solar system page.
    """
    return create_dashboard_layout(screener_data_df)

# ==============================================================================
#  CALLBACK REGISTRATION
# ==============================================================================
def register_solarsystem_callbacks(app, screener_data_df, three_month_corr, six_month_corr, gravitational_impact_df):
    """
    All callbacks for the solar system page are registered within this function.
    """
    
    min_nodes, max_nodes, threshold_percent = 5, 20, 0.9
    search_icon_svg = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxjaXJjbGUgY3g9IjExIiBjeT0iMTEiIHI9IjgiPjwvY2lyY2xlPjxsaW5lIHgxPSIyMSIgeTE9IjIxIiB4Mj0iMTYuNjUiIHkyPSIxNi42NSI+PC9saW5lPjwvc3ZnPg=='

    RED_SPECTRUM = {'light': '#FF0000', 'dark': '#8E0000'}
    GREEN_SPECTRUM = {'dark': '#1B9D49', 'light': '#A1FF61'}
    BLUE_SPECTRUM = {'dark': '#4C216D', 'light': '#9751CB'}
    red_cmap = mcolors.LinearSegmentedColormap.from_list('red_cmap', [RED_SPECTRUM['dark'], RED_SPECTRUM['light']])
    green_cmap = mcolors.LinearSegmentedColormap.from_list('green_cmap', [GREEN_SPECTRUM['dark'], GREEN_SPECTRUM['light']])
    blue_cmap = mcolors.LinearSegmentedColormap.from_list('blue_cmap', [BLUE_SPECTRUM['dark'], BLUE_SPECTRUM['light']])
    planet_cmap = mcolors.LinearSegmentedColormap.from_list('planet_cmap', ['#FF0000', '#F7D117', '#159BFF'])

    @app.callback(
        Output('ticker-dropdown', 'value'),
        Input('url', 'pathname'),
        State('url', 'search'),
    )
    def set_dropdown_from_url(pathname, search):
        if pathname != '/planetarysystem' or not search:
            raise PreventUpdate
        query_params = parse_qs(urlparse(search).query)
        ticker = query_params.get('ticker', [None])[0]
        if ticker:
            return ticker.upper()
        raise PreventUpdate

    @app.callback(
        [Output('info-panels-container', 'children'),
         Output('processed-data-store', 'data'),
         Output('source-data-store', 'data'),
         Output('scroll-trigger-store', 'data')],
        [Input('ticker-dropdown', 'value')]
    )
    def update_panels_and_data_on_ticker_change(selected_ticker):
        if not selected_ticker:
            raise PreventUpdate
        
        processed_data_df, source_data_df = process_and_score_stocks(six_month_corr, three_month_corr, screener_data_df, selected_ticker, min_nodes, max_nodes, threshold_percent)
        
        if processed_data_df.empty or source_data_df.empty: 
            return html.Div(f"Data not available for {selected_ticker}"), None, None, datetime.now()

        def format_market_cap(market_cap): return f"${market_cap / 1e12:.2f}T" if market_cap >= 1e12 else f"${market_cap / 1e9:.2f}B"
        is_weekend = datetime.today().weekday() >= 5
        planet_info_screener = screener_data_df[screener_data_df['code'] == selected_ticker].iloc[0]
        planet_info_source = source_data_df[source_data_df['ticker'] == selected_ticker].iloc[0]
        grav_impact, net_grav_force, max_potential_force = planet_info_source.get('gravitational_impact', 0), planet_info_source.get('net_gravitational_force', 0), planet_info_source.get('max_potential_force', 0)
        moons_df = processed_data_df[processed_data_df['source'] == selected_ticker].copy()
        
        daily_change_header = ["Friday's", html.Br(), "Daily Close"] if is_weekend else ["Yesterday's", html.Br(), "Daily Change"]
        source_name = planet_info_screener.get('name', selected_ticker)
        trend_direction = "an upward" if grav_impact >= 0 else "a downward"
        prediction_summary_text = f"{source_name} ({selected_ticker}) is showing {trend_direction} trend with a strength of {grav_impact:.0f}% {'on Monday' if is_weekend else 'today'}."
        planet_color = get_node_color(grav_impact, True, planet_cmap=planet_cmap)
        planet_image_src = create_model_image_svg(planet_color, 2, 10)

        def create_definition_row(label, text):
            return html.Div([html.Div(label, className='definition-label'), html.Div(text, className='definition-text')], className='definition-row')

        planet_terminology_content = html.Div([
            create_definition_row("Planet Size:", "The size of the planet is based on the market capitalization."),
            create_definition_row("Max Grav. Force:", "The theoretical sum of all gravitational forces from the moons acting on the planet..."),
            create_definition_row("Net Grav. Force:", "The current sum of all gravitational forces from the moons acting on the planet."),
            create_definition_row("Trend Strength:", "The Net Gravitational Force shown as a percentage of the Maximum..."),
            create_definition_row("Planet Color:", "The color of the planet is based on the Trend Strength.")
        ], className='definition-row-container')
        
        moon_terminology_content = html.Div([
            create_definition_row("Moon Size:", "The size of the moon is based on the market capitalization."),
            create_definition_row("Correlation:", "The correlation is calculated by comparing the moon's price..."),
            create_definition_row("Orbital Radius:", "Represents the gravitational force, which is derived from market cap and correlation."),
            create_definition_row("Moon Color:", "The color of the moon is based on yesterday's change in stock price.")
        ], className='definition-row-container')

        planet_info_panel = html.Div([
            html.H3("Planet Information", className='panel-header'),
            html.P(prediction_summary_text, className='panel-prediction-summary'),
            html.Div([html.H4("Terminology"), html.Span('›', className='terminology-chevron')], id={'type': 'terminology-header', 'index': 'planet'}, n_clicks=0, className='terminology-header'),
            html.Div(planet_terminology_content, id={'type': 'terminology-content', 'index': 'planet'}, className='terminology-content'),
            html.Div([html.Div(className='color-legend-bar planet-color-legend'), html.Div([html.Span("Decrease"), html.Span("Stable"), html.Span("Increase")], className='color-legend-labels')], className='color-legend-container'),
            html.Hr(className='panel-divider'),
            # ... table and link here ...
        ], className='info-panel')

        moon_table_panel = html.Div([
             html.H3("Moon Information", className='panel-header'),
             html.Div([html.H4("Terminology"), html.Span('›', className='terminology-chevron')], id={'type': 'terminology-header', 'index': 'moon'}, n_clicks=0, className='terminology-header'),
             html.Div(moon_terminology_content, id={'type': 'terminology-content', 'index': 'moon'}, className='terminology-content'),
             # ... rest of moon panel ...
        ], className='info-panel')
        
        info_panels = html.Div([planet_info_panel, moon_table_panel], className='info-panels-container')
        
        processed_data_json = processed_data_df.to_json(date_format='iso', orient='split')
        source_data_json = source_data_df.to_json(date_format='iso', orient='split')
        
        return info_panels, processed_data_json, source_data_json, datetime.now()

    @app.callback(
        Output({'type': 'terminology-content', 'index': MATCH}, 'className'),
        Output({'type': 'terminology-header', 'index': MATCH}, 'className'),
        Input({'type': 'terminology-header', 'index': MATCH}, 'n_clicks'),
        State({'type': 'terminology-header', 'index': MATCH}, 'className'),
        prevent_initial_call=True
    )
    def toggle_terminology(n_clicks, header_class):
        if n_clicks is None:
            return dash.no_update, dash.no_update
        
        if 'open' in header_class:
            return 'terminology-content', 'terminology-header'
        else:
            return 'terminology-content open', 'terminology-header open'

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
        figure = solar_system_visual(selected_ticker, processed_data_df, source_data_df, screener_data_df, zoom_level, planet_cmap, red_cmap, green_cmap, blue_cmap)
        return figure

    @app.callback(
        Output('ticker-dropdown', 'value', allow_duplicate=True),
        [Input({'type': 'moon-row', 'index': ALL}, 'n_clicks'),
         Input({'type': 'planet-row', 'index': ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def update_dropdown_from_item_click(moon_clicks, planet_clicks):
        all_clicks = (moon_clicks or []) + (planet_clicks or [])
        if not any(all_clicks): raise PreventUpdate
        ctx = callback_context
        if not ctx.triggered: raise PreventUpdate
        triggered_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
        if not triggered_id_str: raise PreventUpdate
        try:
            triggered_id = json.loads(triggered_id_str)
            new_ticker = triggered_id.get('index')
            if new_ticker: return new_ticker
            else: raise PreventUpdate
        except (json.JSONDecodeError, KeyError): raise PreventUpdate

    app.clientside_callback(
        """
        function(trigger) {
            if (trigger) {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('scroll-trigger-store', 'id'),
        Input('scroll-trigger-store', 'data')
    )
