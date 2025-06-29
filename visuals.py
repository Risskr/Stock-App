# filename: visuals.py
import base64
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from collections import defaultdict

def darken_color(color_hex, factor=0.8):
    """Darkens or lightens a hex color by a given factor."""
    rgb = mcolors.to_rgb(color_hex)
    modified_rgb = [min(max(c * factor, 0), 1) for c in rgb]
    return mcolors.to_hex(modified_rgb)

def lighten_color(color_hex, factor=0.3):
    """Lightens a hex color by mixing it with white."""
    rgb = mcolors.to_rgb(color_hex)
    modified_rgb = [c + (1 - c) * factor for c in rgb]
    return mcolors.to_hex(modified_rgb)

def create_procedural_sphere(center_x, center_y, center_z, radius, base_color, subdivisions=4, noise_scale=1.5, splotch_threshold=0.6):
    """
    Generates vertex and face data for a smooth, high-poly sphere with procedural splotches.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])

    for _ in range(subdivisions):
        new_faces = []
        mid_points = {}
        for face in faces:
            v_indices = [face[0], face[1], face[2]]
            new_v_indices = []
            for i in range(3):
                v1, v2 = v_indices[i], v_indices[(i + 1) % 3]
                mid_key = tuple(sorted((v1, v2)))
                mid_idx = mid_points.get(mid_key)
                if mid_idx is None:
                    mid_idx = len(vertices)
                    vertices = np.vstack([vertices, (vertices[v1] + vertices[v2]) / 2.0])
                    mid_points[mid_key] = mid_idx
                new_v_indices.append(mid_idx)
            new_faces.extend([
                [v_indices[0], new_v_indices[0], new_v_indices[2]],
                [v_indices[1], new_v_indices[1], new_v_indices[0]],
                [v_indices[2], new_v_indices[2], new_v_indices[1]],
                new_v_indices
            ])
        faces = np.array(new_faces)

    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    final_vertices = vertices * radius + np.array([center_x, center_y, center_z])

    def perlin_noise(x, y, z, seed=0):
        np.random.seed(seed)
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.stack([p, p]).flatten()
        
        xi, yi, zi = int(x), int(y), int(z)
        xf, yf, zf = x - xi, y - yi, z - zi
        
        u, v, w = (lambda t: t * t * t * (t * (t * 6 - 15) + 10))(xf), \
                  (lambda t: t * t * t * (t * (t * 6 - 15) + 10))(yf), \
                  (lambda t: t * t * t * (t * (t * 6 - 15) + 10))(zf)
        
        g = lambda i, j, k: p[p[p[i] + j] + k]
        n = [g(xi+dx, yi+dy, zi+dz) for dx in [0,1] for dy in [0,1] for dz in [0,1]]
        
        dot = lambda gx, gy, gz, cx, cy, cz: gx * cx + gy * cy + gz * cz
        grad = np.random.randn(256, 3)
        
        d = [dot(grad[n_i][0], grad[n_i][1], grad[n_i][2], xf-dx, yf-dy, zf-dz) for (dx,dy,dz), n_i in zip(np.ndindex(2,2,2), n)]
        
        lerp = lambda a, b, t: a + t * (b - a)
        return lerp(lerp(lerp(d[0], d[1], w), lerp(d[2], d[3], w), v), lerp(lerp(d[4], d[5], w), lerp(d[6], d[7], w), v), u)
    
    if noise_scale > 0:
        noise_values = [perlin_noise(v[0]*noise_scale, v[1]*noise_scale, v[2]*noise_scale, seed=sum(ord(c) for c in base_color)) for v in vertices]
        darker_color_hex = darken_color(base_color, 0.6)
        vertex_colors = [darker_color_hex if val > splotch_threshold else base_color for val in noise_values]
    else:
        vertex_colors = [base_color] * len(final_vertices)

    return final_vertices, faces, vertex_colors

def create_model_image_svg(base_color, subdivisions, texture_spots_count):
    vertices, faces, _ = create_procedural_sphere(0, 0, 0, 1, base_color, subdivisions=subdivisions, noise_scale=0)
    light_source = np.array([-0.5, 0.8, 1.0])
    light_source /= np.linalg.norm(light_source)
    face_data = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        normal = np.cross(v1 - v0, v2 - v0)
        if np.linalg.norm(normal) == 0: continue
        normal /= np.linalg.norm(normal)
        if normal[2] < 0: continue
        intensity = np.dot(normal, light_source)
        color_factor = 0.65 + intensity * 0.5
        facet_color = darken_color(base_color, color_factor)
        points_2d_str = " ".join([f"{(v[0] * 48) + 50},{(v[1] * -48) + 50}" for v in [v0, v1, v2]])
        avg_z = (v0[2] + v1[2] + v2[2]) / 3
        face_data.append({'z': avg_z, 'points': points_2d_str, 'color': facet_color})
    face_data.sort(key=lambda f: f['z'])
    svg_polygons = "".join(f'<polygon points="{f["points"]}" fill="{f["color"]}" />' for f in face_data)
    svg_string = f'''<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><defs><clipPath id="sphereClip"><circle cx="50" cy="50" r="48"/></clipPath></defs><g clip-path="url(#sphereClip)">{svg_polygons}</g><circle cx="50" cy="50" r="48" fill="none" stroke="rgba(255, 255, 255, 0.25)" stroke-width="1.5" /></svg>'''
    return f"data:image/svg+xml;base64,{base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')}"

def get_node_color(value, is_planet, planet_cmap=None, red_cmap=None, green_cmap=None, blue_cmap=None):
    if is_planet:
        if value < -80: return '#FF0000'
        elif -80 <= value < 0: return mcolors.to_hex(planet_cmap(0.5 * (value + 80) / 80))
        elif 0 <= value <= 80: return mcolors.to_hex(planet_cmap(0.5 + 0.5 * (value / 80)))
        else: return '#159BFF'
    else:
        if value > 5: return '#A1FF61'
        elif 1 < value <= 5: return mcolors.to_hex(green_cmap((value - 1) / 4))
        elif 0 < value <= 1: return mcolors.to_hex(blue_cmap(value))
        elif -1 < value <= 0: return mcolors.to_hex(blue_cmap(abs(value)))
        elif -5 <= value < -1: return mcolors.to_hex(red_cmap((abs(value) - 1) / 4))
        else: return '#FF0000'

def solar_system_visual(source_ticker, processed_data_df, source_data_df, screener_data_df, zoom=1.5, planet_cmap=None, red_cmap=None, green_cmap=None, blue_cmap=None):
    ticker_connections = processed_data_df[processed_data_df['source'] == source_ticker].copy()
    source_info_row = source_data_df[source_data_df['ticker'] == source_ticker]
    if ticker_connections.empty or source_info_row.empty:
        return go.Figure().update_layout(title=f"Data not available for {source_ticker}", title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')

    source_info = source_info_row.iloc[0]
    fig = go.Figure()
    pos = {source_ticker: (0, 0, 0)}
    radii_for_rings, min_visual_radius, max_visual_radius = [], 3.0, 10.0
    actual_target_connections = ticker_connections[ticker_connections['target'] != source_ticker].copy()
    num_connections = len(actual_target_connections)
    if num_connections > 0:
        original_radii = actual_target_connections['Orbital Radius']
        min_rad, max_rad = original_radii.min(), original_radii.max()
        rad_range = max_rad - min_rad if max_rad > min_rad else 1.0
        visual_range = max_visual_radius - min_visual_radius
        thetas = np.linspace(0, 2 * np.pi, num_connections, endpoint=False)
        for i, (index, row) in enumerate(actual_target_connections.iterrows()):
            scaled_radius = ((row['Orbital Radius'] - min_rad) / rad_range) * visual_range + min_visual_radius
            radii_for_rings.append(scaled_radius)
            pos[row['target']] = (scaled_radius * np.cos(thetas[i]), scaled_radius * np.sin(thetas[i]), 0)

    reticle_color = 'rgba(255, 255, 255, 0.4)'
    furthest_orbit = max(radii_for_rings) if radii_for_rings else max_visual_radius
    reticle_length = furthest_orbit * 1.1
    tick_length = reticle_length * 0.05
    fig.add_trace(go.Scatter3d(x=[-reticle_length, reticle_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color=reticle_color, width=0.5), hoverinfo='none'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-reticle_length, reticle_length], z=[0, 0], mode='lines', line=dict(color=reticle_color, width=0.5), hoverinfo='none'))
    
    scene_annotations = []
    tick_positions = [furthest_orbit * 0.33, furthest_orbit * 0.66, furthest_orbit]
    tick_labels = ["Most<br>Correlated", "Correlated", "Least<br>Correlated"]
    for i, pos_val in enumerate(tick_positions):
        for sign in [-1, 1]:
            fig.add_trace(go.Scatter3d(x=[pos_val * sign, pos_val * sign], y=[-tick_length, tick_length], z=[0, 0], mode='lines', line=dict(color=reticle_color, width=0.5), hoverinfo='none'))
            fig.add_trace(go.Scatter3d(x=[-tick_length, tick_length], y=[pos_val * sign, pos_val * sign], z=[0, 0], mode='lines', line=dict(color=reticle_color, width=0.5), hoverinfo='none'))
        scene_annotations.append(dict(x=pos_val, y=-(tick_length * 2), z=-0.1, text=tick_labels[i], showarrow=False, font=dict(color=reticle_color, size=9), xanchor="center", yanchor="top"))

    for r in sorted(list(set(radii_for_rings))):
        theta_ring = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter3d(x=r * np.cos(theta_ring), y=r * np.sin(theta_ring), z=np.zeros(100), mode='lines', line=dict(color='rgba(255, 255, 255, 0.4)', width=1, dash='dot'), hoverinfo='none'))

    for node_name, coords in pos.items():
        is_source = (node_name == source_ticker)
        screener_info_row = screener_data_df[screener_data_df['code'] == node_name]
        if screener_info_row.empty: continue
        screener_info = screener_info_row.iloc[0]
        market_cap = screener_info.get('market_capitalization', 0)
        market_cap_str = f"${market_cap/1e12:.2f}T" if market_cap > 1e12 else f"${market_cap/1e9:.2f}B"
        min_visual_size, max_visual_size = 0.6, 1.5

        if is_source:
            hover_text = f"<b>{screener_info.get('name', node_name)} ({node_name})</b><br>" + "<br>".join([f"{k}: {screener_info.get(k, 'N/A')}" for k in ['industry', 'sector']]) + f"<br>Market Cap: {market_cap_str}"
            node_color = get_node_color(source_info['gravitational_impact'], True, planet_cmap=planet_cmap)
            radius = min_visual_size + (source_info.get('source_moon_radius', 0.5) * (max_visual_size - min_visual_size))
            subdivisions, noise_scale, splotch_threshold = 5, 2.0, 0.68
        else:
            processed_info = ticker_connections[ticker_connections['target'] == node_name].iloc[0]
            hover_text = f"<b>{screener_info.get('name', node_name)} ({node_name})</b><br>" + "<br>".join([f"{k}: {screener_info.get(k, 'N/A')}" for k in ['industry', 'sector']]) + f"<br>Daily Change: {processed_info['Daily Change']:.2f}%<br>Market Cap: {market_cap_str}"
            node_color = get_node_color(processed_info['Daily Change'], False, red_cmap=red_cmap, green_cmap=green_cmap, blue_cmap=blue_cmap)
            radius = min_visual_size + (processed_info['moon Radius'] * (max_visual_size - min_visual_size))
            subdivisions, noise_scale, splotch_threshold = 4, 2.5, 0.7
        
        vertices, faces, vertex_colors = create_procedural_sphere(coords[0], coords[1], coords[2], radius, node_color, subdivisions, noise_scale, splotch_threshold)
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            vertexcolor=vertex_colors,
            lighting=dict(ambient=0.4, diffuse=1.0, specular=2.0, roughness=0.05, fresnel=0.2),
            lightposition=dict(x=2000, y=1000, z=2000),
            hoverinfo='text', text=hover_text,
            hoverlabel=dict(bgcolor='#0f0524', font=dict(color='#EAEAEA', size=14), bordercolor='rgba(255, 255, 255, 0.3)')
        ))
        scene_annotations.append(dict(x=coords[0], y=coords[1], z=coords[2], text=f"<b>{node_name}</b>", showarrow=False, font=dict(color='white', size=14), bgcolor="rgba(0,0,0,0)", xanchor="center"))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=0, y=-1.6 * zoom, z=0.8 * zoom)),
            aspectmode='data', annotations=scene_annotations, bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="'Space Grotesk', sans-serif", color='#EAEAEA', size=16)
    )
    return fig
