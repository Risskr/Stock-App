# filename: pages/homepage.py

import dash
import pandas as pd
from dash import html, dcc, Input, Output, State, callback_context, clientside_callback
from dash.exceptions import PreventUpdate
import json
from datetime import datetime

def create_layout(screener_data_df):
    """
    Creates the layout for the homepage with a full-screen search overlay
    and a "Top Trends" section. Styling is now handled by style.css.
    """
    
    if not isinstance(screener_data_df, pd.DataFrame) or screener_data_df.empty:
        stock_options = []
    else:
        if 'market_capitalization' in screener_data_df.columns and 'name' in screener_data_df.columns and 'code' in screener_data_df.columns:
            sorted_df = screener_data_df.sort_values('market_capitalization', ascending=False)
            stock_options = [{'label': f"{row['name']} ({row['code']})", 'value': row['code']}
                            for index, row in sorted_df.iterrows()]
        else:
            stock_options = [{'label': f"{row.get('name', 'N/A')} ({row.get('code', 'N/A')})", 'value': row.get('code', '')}
                            for index, row in screener_data_df.iterrows()]

    LOGO_URL = "https://storage.googleapis.com/financial_observatory_public/assets/Logo_rectangle.PNG"
    search_icon_svg = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxjaXJjbGUgY3g9IjExIiBjeT0iMTEiIHI9IjgiPjwvY2lyY2xlPjxsaW5lIHgxPSIyMSIgeTE9IjIxIiB4Mj0iMTYuNjUiIHkyPSIxNi42NSI+PC9saW5lPjwvc3ZnPg=='

    # --- Main page view ---
    main_page_layout = html.Div([
        html.Img(src=LOGO_URL, className="homepage-logo"),
        html.H1("The Financial Observatory", className="homepage-title"),
        html.P("Navigate the gravitational forces of the stock market.", className="homepage-subtitle"),
        
        html.Div([
            html.Img(src=search_icon_svg, className="search-icon"),
            "Search for a company name or symbol…."
        ], id='dummy-search-bar', n_clicks=0, className="dummy-search-bar"),

        html.Div(id='top-trends-container', className="top-trends-container-wrapper"),
        
        html.Div(dcc.Link("Disclaimer", href="/disclaimer", className="homepage-disclaimer-link"), className="homepage-disclaimer-wrapper")
    ], id='main-page-layout', className="homepage-layout")

    # --- Full-screen search overlay ---
    search_overlay_layout = html.Div([
        html.Button("×", id='close-search-btn', n_clicks=0, className="close-search-btn"),
        html.Div([
            dcc.Input(
                id='real-ticker-input',
                type='text',
                placeholder='Search for a stock...',
                className='real-search-input',
                n_submit=0,
            ),
            html.Div(id='stock-list-container', className='stock-list-container')
        ], className="search-overlay-content")
    ], id='search-overlay', className="search-overlay")

    # The complete page layout
    return html.Div([
        dcc.Store(id='stock-options-store', data=stock_options),
        dcc.Store(id='filtered-stock-tickers-store', data=[]),
        html.Div(id='homepage-redirect-output'),
        main_page_layout,
        search_overlay_layout,
    ], className="planetary-background")


def register_homepage_callbacks(app, screener_data_df, gravitational_impact_df):

    @app.callback(
        Output('top-trends-container', 'children'),
        Input('url', 'pathname')
    )
    def update_top_trends(pathname):
        if pathname != '/':
            raise PreventUpdate
        
        if gravitational_impact_df.empty or screener_data_df.empty:
            return html.Div("Trend data is currently unavailable.", style={'color': '#9ca3af', 'textAlign': 'center', 'marginTop': '2rem'})

        is_weekend = datetime.today().weekday() >= 5
        trends_header_text = "Monday's Top Daily Trends" if is_weekend else "Top Daily Trends"
        
        top_positive_impacts = gravitational_impact_df.sort_values(by='gravitational_impact', ascending=False).head(5)
        top_negative_impacts = gravitational_impact_df.sort_values(by='gravitational_impact', ascending=True).head(5)
        combined_impacts = pd.concat([top_positive_impacts, top_negative_impacts]).reset_index(drop=True)

        def create_prediction_item(row):
            ticker = row['ticker']
            screener_info = screener_data_df[screener_data_df['code'] == ticker]
            name = screener_info.iloc[0]['name'] if not screener_info.empty else ticker
            
            return html.A(
                href=f'/planetarysystem?ticker={ticker.upper()}',
                children=[
                    html.Span(f"{name} ({ticker})"),
                    # Dynamic style remains here
                    html.Span(f"{row['gravitational_impact']:.2f}%", style={'color': '#4ade80' if row['gravitational_impact'] > 0 else '#f87171', 'fontWeight': 'bold'})
                ],
                className="prediction-item"
            )

        prediction_items = [create_prediction_item(row) for _, row in combined_impacts.iterrows()]
        
        return html.Div([
            html.H3(trends_header_text, className="trends-header"),
            html.Div(prediction_items)
        ], className="top-trends-container")

    @app.callback(
        Output('stock-list-container', 'children'),
        Output('filtered-stock-tickers-store', 'data'),
        Input('dummy-search-bar', 'n_clicks'),
        Input('real-ticker-input', 'value'),
        State('stock-options-store', 'data'),
        prevent_initial_call=True
    )
    def update_stock_list(n_clicks, search_value, all_stocks):
        ctx = callback_context
        if not ctx.triggered: return dash.no_update, dash.no_update 
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if not all_stocks: return [html.Div("Error: Stock data could not be loaded.", className="stock-list-item-no-match")], []
        search_term = (search_value.lower() if search_value else "")
        filtered_stocks = [stock for stock in all_stocks if search_term in stock['label'].lower()]
        if not filtered_stocks: return [html.Div("No matches found.", className="stock-list-item-no-match")], []
        list_items_html = [html.Div(stock['label'], id={'type': 'stock-list-item', 'index': stock['value']}, className='stock-list-item', n_clicks=0) for stock in filtered_stocks[:100]]
        filtered_ticker_values = [stock['value'] for stock in filtered_stocks[:100]]
        return list_items_html, filtered_ticker_values

    @app.callback(
        Output('homepage-redirect-output', 'children'),
        Input({'type': 'stock-list-item', 'index': dash.dependencies.ALL}, 'n_clicks'),
        Input('real-ticker-input', 'n_submit'),
        State('filtered-stock-tickers-store', 'data'),
        prevent_initial_call=True
    )
    def redirect_from_selection(list_clicks, n_submit, filtered_tickers):
        ctx = callback_context
        if not ctx.triggered: raise PreventUpdate
        triggered = ctx.triggered[0]
        triggered_prop_id = triggered['prop_id']
        triggered_value = triggered['value']

        if triggered_prop_id == 'real-ticker-input.n_submit':
            if filtered_tickers:
                return dcc.Location(href=f'/planetarysystem?ticker={filtered_tickers[0].upper()}', id='redirect-id')
            raise PreventUpdate
        
        if 'stock-list-item' in triggered_prop_id and 'index' in triggered_prop_id:
            if not triggered_value: raise PreventUpdate
            try:
                component_id_str = triggered_prop_id.rsplit('.', 1)[0]
                ticker = json.loads(component_id_str).get('index')
                if ticker: return dcc.Location(href=f'/planetarysystem?ticker={ticker.upper()}', id='redirect-id')
            except (json.JSONDecodeError, KeyError):
                raise PreventUpdate
        
        raise PreventUpdate

    clientside_callback(
        """
        function(dummy_clicks, close_clicks) {
            const ctx = dash_clientside.callback_context;
            if (!ctx.triggered_id) return window.dash_clientside.no_update;
            const overlay = document.getElementById('search-overlay');
            if (!overlay) return window.dash_clientside.no_update;
            if (ctx.triggered_id === 'dummy-search-bar') {
                overlay.classList.add('visible');
                setTimeout(() => {
                    const input = overlay.querySelector('#real-ticker-input');
                    if (input) input.focus();
                }, 50);
            } else if (ctx.triggered_id === 'close-search-btn') {
                overlay.classList.remove('visible');
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('dummy-search-bar', 'data-dummy'),
        Input('dummy-search-bar', 'n_clicks'),
        Input('close-search-btn', 'n_clicks')
    )
