import base64
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors

def darken_color(color_hex, factor=0.8):
    """Darkens or lightens a hex color by a given factor."""
    rgb = mcolors.to_rgb(color_hex)
    # Clamp values to ensure they stay within the valid [0, 1] range for RGB
    modified_rgb = [min(max(c * factor, 0), 1) for c in rgb]
    return mcolors.to_hex(modified_rgb)

def lighten_color(color_hex, factor=0.3):
    """Lightens a hex color by mixing it with white."""
    rgb = mcolors.to_rgb(color_hex)
    # Interpolate each component towards white (1.0)
    modified_rgb = [c + (1 - c) * factor for c in rgb]
    return mcolors.to_hex(modified_rgb)

def create_low_poly_sphere(center_x, center_y, center_z, radius, base_color, subdivisions=2, texture_spots=15):
    """
    Generates vertex and face data for a textured low-poly sphere (icosphere).
    """
    # Define the 12 vertices of a regular icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])

    # Define the 20 triangular faces of the icosahedron
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])

    # Subdivide faces to create more polygons
    for _ in range(subdivisions):
        new_faces = []
        mid_points = {}
        for face in faces:
            v_indices = [face[0], face[1], face[2]]
            new_v_indices = []
            for i in range(3):
                v1 = v_indices[i]
                v2 = v_indices[(i + 1) % 3]
                mid_key = tuple(sorted((v1, v2)))
                mid_idx = mid_points.get(mid_key)
                if mid_idx is None:
                    mid_idx = len(vertices)
                    vertices = np.vstack([vertices, (vertices[v1] + vertices[v2]) / 2.0])
                    mid_points[mid_key] = mid_idx
                new_v_indices.append(mid_idx)
            new_faces.append([v_indices[0], new_v_indices[0], new_v_indices[2]])
            new_faces.append([v_indices[1], new_v_indices[1], new_v_indices[0]])
            new_faces.append([v_indices[2], new_v_indices[2], new_v_indices[1]])
            new_faces.append(new_v_indices)
        faces = np.array(new_faces)

    # Normalize vertices to form a sphere, then scale and translate
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    final_vertices = vertices * radius + np.array([center_x, center_y, center_z])

    # Create vertex colors for texture spots
    darker_color_hex = darken_color(base_color, 0.7)
    vertex_colors = [base_color] * len(final_vertices)
    if texture_spots > 0:
        spot_indices = np.random.choice(len(final_vertices), size=texture_spots, replace=False)
        for idx in spot_indices:
            vertex_colors[idx] = darker_color_hex

    return final_vertices, faces, vertex_colors

def create_model_image_svg(base_color, subdivisions, texture_spots_count):
    """
    Creates a base64 encoded SVG data URL that is a 2D representation of the 3D low-poly model.
    This function generates the geometry, projects it, and simulates flat shading to match the plot.
    """
    # 1. Generate the 3D model's vertices and faces for a unit sphere
    # We pass 0 for texture_spots here because we'll draw them separately in the SVG
    vertices, faces, _ = create_low_poly_sphere(0, 0, 0, 1, base_color, subdivisions, 0)

    # 2. Define a light source for shading the facets
    light_source = np.array([-0.5, 0.8, 1.0])
    light_source = light_source / np.linalg.norm(light_source)

    # 3. Process each face for rendering
    face_data = []
    for face in faces:
        # Get the vertices for the current face
        v0, v1, v2 = vertices[face]

        # --- Back-face culling: Don't render faces pointing away from the camera ---
        # The camera is at (0, 0, z), so we check the z-component of the normal
        normal = np.cross(v1 - v0, v2 - v0)
        if np.linalg.norm(normal) == 0: continue
        normal = normal / np.linalg.norm(normal)
        if normal[2] < 0:
            continue # This face is on the back of the sphere, so we skip it

        # --- Shading: Calculate brightness based on angle to the light source ---
        intensity = np.dot(normal, light_source)
        # Map intensity to a brightness factor for the color
        color_factor = 0.65 + intensity * 0.5
        facet_color = darken_color(base_color, color_factor)

        # --- Projection: Convert 3D vertex coordinates to 2D SVG coordinates ---
        # We scale and shift the (x, y) coordinates to fit in a 100x100 SVG
        points_2d_str = " ".join([f"{(v[0] * 48) + 50},{(v[1] * -48) + 50}" for v in [v0, v1, v2]])

        # Store the face's z-depth for sorting, so closer faces draw on top
        avg_z = (v0[2] + v1[2] + v2[2]) / 3
        face_data.append({'z': avg_z, 'points': points_2d_str, 'color': facet_color})

    # 4. Sort faces from back to front
    face_data.sort(key=lambda f: f['z'])

    # 5. Build the SVG polygons from the sorted face data
    svg_polygons = "".join(f'<polygon points="{f["points"]}" fill="{f["color"]}" />' for f in face_data)

    # 6. Add texture spots as random circles
    texture_color = darken_color(base_color, 0.7)
    svg_texture_spots = ""
    np.random.seed(sum(ord(c) for c in base_color)) # Seed for consistency
    for _ in range(texture_spots_count):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, 48)
        spot_size = np.random.uniform(4, 9)
        cx = 50 + radius * np.cos(angle)
        cy = 50 + radius * np.sin(angle)
        svg_texture_spots += f'<circle cx="{cx}" cy="{cy}" r="{spot_size}" fill="{texture_color}" opacity="0.7"/>'

    # 7. Assemble the final SVG string
    svg_string = f'''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <clipPath id="sphereClip">
          <circle cx="50" cy="50" r="48"/>
        </clipPath>
        <filter id="blur-effect">
          <feGaussianBlur in="SourceGraphic" stdDeviation="0.7" />
        </filter>
      </defs>
      <g clip-path="url(#sphereClip)" filter="url(#blur-effect)">
        {svg_polygons}
        {svg_texture_spots}
      </g>
      <circle cx="50" cy="50" r="48" fill="none" stroke="rgba(255, 255, 255, 0.25)" stroke-width="1.5" />
    </svg>
    '''
    encoded_svg = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded_svg}"

def get_node_color(value, is_star, star_cmap=None, red_cmap=None, green_cmap=None, blue_cmap=None):
    """
    Determines the node color based on its value and whether it's a star or planet.
    """
    if is_star:
        # Star color based on grav_impact from #FF0000 -> #F7D117 -> #159BFF
        if value < -80: return '#FF0000'
        elif -80 <= value < 0:
            # Normalize value from -80 -> 0 to 0 -> 0.5 for the colormap
            return mcolors.to_hex(star_cmap(0.5 * (value + 80) / 80))
        elif 0 <= value <= 80:
            # Normalize value from 0 -> 80 to 0.5 -> 1.0 for the colormap
            return mcolors.to_hex(star_cmap(0.5 + 0.5 * (value / 80)))
        else: return '#159BFF'
    else:
        # Planet color based on daily change
        if value > 5: return '#A1FF61'
        elif 1 < value <= 5: return mcolors.to_hex(green_cmap((value - 1) / 4))
        elif 0 < value <= 1: return mcolors.to_hex(blue_cmap(value))
        elif -1 < value <= 0: return mcolors.to_hex(blue_cmap(abs(value))) # Reverses blue map
        elif -5 <= value < -1: return mcolors.to_hex(red_cmap((abs(value) - 1) / 4))
        else: return '#FF0000'

def solar_system_visual(source_ticker, processed_data_df, source_data_df, screener_data_df, zoom=1.5, star_cmap=None, red_cmap=None, green_cmap=None, blue_cmap=None):
    """
    Creates the main 3D solar system visualization with a low-poly aesthetic.
    """
    ticker_connections = processed_data_df[processed_data_df['source'] == source_ticker].copy()
    source_info_row = source_data_df[source_data_df['ticker'] == source_ticker]

    if ticker_connections.empty or source_info_row.empty:
        return go.Figure().update_layout(title=f"Data not available for {source_ticker}", title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')

    source_info = source_info_row.iloc[0]
    fig = go.Figure()

    pos = {source_ticker: (0, 0, 0)}
    actual_target_connections = ticker_connections[ticker_connections['target'] != source_ticker].copy()
    num_connections = len(actual_target_connections)
    radii_for_rings = []
    
    min_visual_radius, max_visual_radius = 3.0, 10.0

    if num_connections > 0:
        original_radii = actual_target_connections['Orbital Radius']
        min_rad, max_rad = original_radii.min(), original_radii.max()
        rad_range = max_rad - min_rad if max_rad > min_rad else 1.0
        visual_range = max_visual_radius - min_visual_radius
        thetas = np.linspace(0, 2 * np.pi, num_connections, endpoint=False)

        for i, (index, row) in enumerate(actual_target_connections.iterrows()):
            scaled_radius = ((row['Orbital Radius'] - min_rad) / rad_range) * visual_range + min_visual_radius
            radii_for_rings.append(scaled_radius)
            theta = thetas[i]
            pos[row['target']] = (scaled_radius * np.cos(theta), scaled_radius * np.sin(theta), 0)

    # --- Add Reticle ---
    reticle_color = 'lightgrey'
    furthest_orbit = max(radii_for_rings) if radii_for_rings else max_visual_radius
    reticle_length = furthest_orbit * 1.1
    tick_length = reticle_length * 0.05

    # Main lines
    fig.add_trace(go.Scatter3d(x=[-reticle_length, reticle_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color=reticle_color, width=1), hoverinfo='none'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-reticle_length, reticle_length], z=[0, 0], mode='lines', line=dict(color=reticle_color, width=1), hoverinfo='none'))
    
    # Tick marks and labels
    scene_annotations = []
    tick_positions = [furthest_orbit * 0.33, furthest_orbit * 0.66, furthest_orbit]
    # Using <br> for line breaks in the labels
    tick_labels = ["Most<br>Correlated", "Correlated", "Least<br>Correlated"]

    for i, pos_val in enumerate(tick_positions):
        # Right tick
        fig.add_trace(go.Scatter3d(x=[pos_val, pos_val], y=[-tick_length, tick_length], z=[0, 0], mode='lines', line=dict(color=reticle_color, width=1), hoverinfo='none'))
        # Left tick
        fig.add_trace(go.Scatter3d(x=[-pos_val, -pos_val], y=[-tick_length, tick_length], z=[0, 0], mode='lines', line=dict(color=reticle_color, width=1), hoverinfo='none'))
        # Top tick
        fig.add_trace(go.Scatter3d(x=[-tick_length, tick_length], y=[pos_val, pos_val], z=[0, 0], mode='lines', line=dict(color=reticle_color, width=1), hoverinfo='none'))
        # Bottom tick
        fig.add_trace(go.Scatter3d(x=[-tick_length, tick_length], y=[-pos_val, -pos_val], z=[0, 0], mode='lines', line=dict(color=reticle_color, width=1), hoverinfo='none'))
        
        # Add labels on the right side, anchored to the top to appear below the line
        scene_annotations.append(
            dict(
                x=pos_val, y=-(tick_length * 2), z=-0.1, # Place slightly behind and below
                text=tick_labels[i], showarrow=False,
                font=dict(color=reticle_color, size=10),
                xanchor="center", yanchor="top"
            )
        )

    # --- Add Orbital Rings ---
    for r in sorted(list(set(radii_for_rings))):
        theta_ring = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter3d(
            x=r * np.cos(theta_ring), y=r * np.sin(theta_ring), z=np.zeros(100),
            mode='lines',
            line=dict(color='#454545', width=2, dash='solid'),
            hoverinfo='none'
        ))

    # --- Loop Through Each Node to Draw Them ---
    for node_name, coords in pos.items():
        center_x, center_y, center_z = coords
        is_source = (node_name == source_ticker)

        screener_info_row = screener_data_df[screener_data_df['code'] == node_name]
        if screener_info_row.empty: continue
        screener_info = screener_info_row.iloc[0]

        market_cap = screener_info.get('market_capitalization', 0)
        market_cap_str = f"${market_cap/1e12:.2f}T" if market_cap > 1e12 else f"${market_cap/1e9:.2f}B"
        min_visual_size, max_visual_size = 0.6, 1.5

        if is_source:
            hover_text = (f"<b>{screener_info.get('name', node_name)} ({node_name})</b><br>"
                          f"Industry: {screener_info.get('industry', 'N/A')}<br>"
                          f"Sector: {screener_info.get('sector', 'N/A')}<br>"
                          f"Avg Volume (1d): {screener_info.get('avgvol_1d', 'N/A')}<br>"
                          f"Market Cap: {market_cap_str}")
            node_color = get_node_color(source_info['gravitational_impact'], True, star_cmap=star_cmap)
            radius = min_visual_size + (source_info.get('source_planet_radius', 0.5) * (max_visual_size - min_visual_size))
            subdivisions = 2
        else:
            processed_info = ticker_connections[ticker_connections['target'] == node_name].iloc[0]
            hover_text = (f"<b>{screener_info.get('name', node_name)} ({node_name})</b><br>"
                          f"Industry: {screener_info.get('industry', 'N/A')}<br>"
                          f"Sector: {screener_info.get('sector', 'N/A')}<br>"
                          f"Avg Volume (1d): {screener_info.get('avgvol_1d', 'N/A')}<br>"
                          f"Daily Change: {processed_info['Daily Change']:.2f}%<br>"
                          f"Market Cap: {market_cap_str}")
            node_color = get_node_color(processed_info['Daily Change'], False, red_cmap=red_cmap, green_cmap=green_cmap, blue_cmap=blue_cmap)
            radius = min_visual_size + (processed_info['Planet Radius'] * (max_visual_size - min_visual_size))
            subdivisions = 2
        
        # --- Add Aura/Atmosphere Effect ---
        aura_fade_color = lighten_color(node_color, 0.8)
        aura_glow_color = lighten_color(node_color, 0.4)

        # Draw the outer, most transparent layer first
        aura_fade_vertices, aura_fade_faces, _ = create_low_poly_sphere(center_x, center_y, center_z, radius * 1.08, node_color, subdivisions, 0)
        fig.add_trace(go.Mesh3d(
            x=aura_fade_vertices[:, 0], y=aura_fade_vertices[:, 1], z=aura_fade_vertices[:, 2],
            i=aura_fade_faces[:, 0], j=aura_fade_faces[:, 1], k=aura_fade_faces[:, 2],
            color=aura_fade_color, opacity=0.1, flatshading=True, hoverinfo='none'
        ))

        # Draw the middle glow layer
        aura_glow_vertices, aura_glow_faces, _ = create_low_poly_sphere(center_x, center_y, center_z, radius * 1.05, node_color, subdivisions, 0)
        fig.add_trace(go.Mesh3d(
            x=aura_glow_vertices[:, 0], y=aura_glow_vertices[:, 1], z=aura_glow_vertices[:, 2],
            i=aura_glow_faces[:, 0], j=aura_glow_faces[:, 1], k=aura_glow_faces[:, 2],
            color=aura_glow_color, opacity=0.2, flatshading=True, hoverinfo='none'
        ))

        # --- Draw the Core Planet ---
        vertices, faces, vertex_colors = create_low_poly_sphere(center_x, center_y, center_z, radius, node_color, subdivisions, texture_spots=15)
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            vertexcolor=vertex_colors,
            opacity=1.0,
            flatshading=True,
            lighting=dict(ambient=0.8, diffuse=0.5, specular=1, roughness=1),
            hoverinfo='text',
            text=hover_text,
            hoverlabel=dict(bgcolor='#0f0524', font=dict(color='#EAEAEA', size=14), bordercolor='rgba(255, 255, 255, 0.3)')
        ))

        # Add labels for nodes
        scene_annotations.append(
            dict(
                x=center_x, y=center_y, z=center_z,
                text=f"<b>{node_name}</b>", showarrow=False,
                font=dict(color='white', size=14),
                bgcolor="rgba(0,0,0,0)", xanchor="center"
            )
        )

    # --- Configure Final Layout ---
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=0, y=-1.6 * zoom, z=0.8 * zoom)), # Changed camera angle
            aspectmode='data',
            annotations=scene_annotations,
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="'Space Grotesk', sans-serif", color='#EAEAEA', size=16)
    )
    return fig
