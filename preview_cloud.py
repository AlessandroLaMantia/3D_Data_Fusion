"""Esporta il PLY dell'esempio in un HTML 3D autonomo e utilizzabile offline."""
import json
from pathlib import Path
import numpy as np


def main():
    import open3d as o3d
    import plotly.graph_objects as go
    root = Path(__file__).resolve().parent
    demo = root/'outputs'/'demo'
    summary = json.loads((demo/'summary.json').read_text(encoding='utf-8'))
    frame = summary['best_frame']
    path = demo/'best'/f'{frame:010d}_rgb.ply'
    cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(cloud.points)
    rgb = np.round(np.asarray(cloud.colors)*255).astype(np.uint8)
    if len(xyz) == 0 or rgb.shape != xyz.shape:
        raise RuntimeError('PLY vuoto o privo di colori RGB')
    colors = [f'rgb({r},{g},{b})' for r, g, b in rgb]
    fig = go.Figure(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                                mode='markers', marker=dict(size=2, color=colors),
                                hovertemplate='X %{x:.2f} m<br>Y %{y:.2f} m<br>Z %{z:.2f} m<extra></extra>'))
    fig.update_layout(title=f'KITTI frame {frame} · {len(xyz):,} punti RGB reali · trascina per ruotare',
                      template='plotly_dark', margin=dict(l=0, r=0, b=0, t=65),
                      scene=dict(aspectmode='data', xaxis_title='X Velodyne (m)',
                                 yaxis_title='Y Velodyne (m)', zaxis_title='Z Velodyne (m)',
                                 camera=dict(eye=dict(x=-1.5, y=-1.5, z=.8))))
    fig.write_html(str(demo/'nuvola.html'), include_plotlyjs=True, full_html=True)
    gallery = demo/'index.html'
    page = gallery.read_text(encoding='utf-8')
    if 'href="nuvola.html"' not in page:
        page = page.replace('<h1>Camera RGB + LiDAR: demo KITTI reale</h1>',
                            '<h1>Camera RGB + LiDAR: demo KITTI reale</h1><p><a href="nuvola.html">Apri la nuvola RGB 3D interattiva nel browser</a></p>')
        gallery.write_text(page, encoding='utf-8')
    print(f'PLY verificato con Open3D: {len(xyz)} punti RGB. Preview: {demo / "nuvola.html"}')


if __name__ == '__main__':
    main()
