from dash import html, dcc

layout = html.Div([
    html.Div([
        # --- MODIFIED: Link text and color updated ---
        dcc.Link('← Back to Financial Observatory', href='/', style={
            'color': 'black', 
            'textDecoration': 'none', 
            'fontSize': '16px',
            'display': 'inline-block',
            'marginBottom': '30px'
        }),

        html.H1('Disclaimer', style={'textAlign': 'center'}),
        
        # --- MODIFIED: Disclaimer text updated and structured ---
        html.P(
            "This application and all content herein are provided for informational and educational purposes only. "
            "We are not financial advisors, and nothing presented here constitutes financial, investment, legal, "
            "tax, or professional advice."
        ),
        
        html.H4("No Advice:", style={'marginTop': '20px'}),
        html.P(
            "We do not provide personalized recommendations, endorsements, or solicitations to buy, sell, or hold "
            "any securities or investments. All investment decisions are solely your responsibility."
        ),

        html.H4("Investing Risk:", style={'marginTop': '20px'}),
        html.P(
            "Investing in financial markets involves significant risk, including the potential loss of principal. "
            "Past performance is not indicative of future results. You should carefully consider your financial "
            "situation and consult with a qualified financial professional before making any investment decisions."
        ),

        html.H4("Data Accuracy & Timeliness:", style={'marginTop': '20px'}),
        html.P(
            "While we strive for accuracy, the data, information, and tools provided may contain errors, be "
            "incomplete, or be delayed. We do not guarantee the accuracy, completeness, or timeliness of any "
            "information."
        ),

        html.H4("No Liability:", style={'marginTop': '20px'}),
        html.P(
            "We expressly disclaim all liability for any direct, indirect, incidental, consequential, or "
            "special damages arising out of or in any way connected with your access to, use of, or reliance "
            "on this application or its content."
        ),

        html.P(
            "Always conduct your own due diligence and consult with a licensed financial professional before "
            "making any investment decisions. By using this application, you agree to these terms.",
            style={'marginTop': '20px', 'fontStyle': 'italic'}
        )
    ], style={
        'maxWidth': '800px',
        'margin': '40px auto',
        'padding': '20px',
        'lineHeight': '1.6'
    })
])
