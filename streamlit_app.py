"""Interfaccia locale: selezione RGB, LiDAR associato e detection su richiesta."""
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

from modules.calibration import load_calibration
from modules.projection import load_velodyne, project_points
from modules.detection_fusion import TraditionalDetector, fuse_detections
from modules.visualization import draw_overlay

ROOT = Path(__file__).resolve().parent


@st.cache_data(max_entries=8, show_spinner=False)
def load_frame(sequence_path, frame, signature):
    """Cache limitata; signature invalida i dati quando cambiano i file."""
    sequence = Path(sequence_path)
    calibration = load_calibration(sequence.parent/'calib_cam_to_cam.txt',
                                   sequence.parent/'calib_velo_to_cam.txt')
    image_path = sequence/'image_02'/'data'/f'{frame:010d}.png'
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Unable to read the RGB image')
    points = load_velodyne(sequence/'velodyne_points'/'data'/f'{frame:010d}.bin')
    projection = project_points(points, calibration, image.shape)
    # Il LiDAR non è una fotografia: è rasterizzato sul piano della Camera 2.
    lidar = draw_overlay(np.zeros_like(image), projection, [], language='en')
    return image, points, projection, lidar


def main():
    st.set_page_config(page_title='Camera + LiDAR', page_icon='📷', layout='wide')
    st.title('RGB Camera + LiDAR')
    st.write('Choose an image, view the corresponding LiDAR and start detecting people.')

    sequences = sorted(p for p in (ROOT/'data').glob('*/*_sync')
                       if (p/'image_02'/'data').is_dir())
    if not sequences:
        st.error('No KITTI sequence found in the data folder.')
        st.code('.\\.venv\\Scripts\\python.exe download_demo.py', language='powershell')
        st.stop()
    with st.sidebar:
        st.header('Select an image')
        sequence = next((p for p in sequences if p.name == '2011_09_26_drive_0005_sync'), sequences[0])
        frames = sorted(int(p.stem) for p in (sequence/'image_02'/'data').glob('*.png')
                        if p.stem.isdigit() and (sequence/'velodyne_points'/'data'/f'{p.stem}.bin').is_file())
        if not frames:
            st.error('No RGB/LiDAR pairs are available.')
            st.stop()
        frame = st.selectbox('RGB image', frames, index=frames.index(80) if 80 in frames else 0,
                             format_func=lambda f: f'Frame {f:04d} · {f:010d}.png')

    paths = [sequence/'image_02'/'data'/f'{frame:010d}.png',
             sequence/'velodyne_points'/'data'/f'{frame:010d}.bin',
             sequence.parent/'calib_cam_to_cam.txt', sequence.parent/'calib_velo_to_cam.txt']
    try:
        signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        with st.spinner('Loading RGB and LiDAR…'):
            image, points, projection, lidar = load_frame(str(sequence), frame, signature)
    except (OSError, ValueError, cv2.error) as exc:
        st.error(f'Unable to load the frame: {exc}')
        st.stop()

    selection = (str(sequence), frame, signature, 'en')
    if st.session_state.get('selection') != selection:
        st.session_state.selection = selection
        st.session_state.pop('detection_result', None)

    rgb_column, lidar_column = st.columns(2)
    with rgb_column:
        st.subheader('RGB image')
        st.image(image, channels='BGR', width='stretch', caption=f'Camera 2 · frame {frame:010d}')
    with lidar_column:
        st.subheader('Corresponding LiDAR')
        st.image(lidar, channels='BGR', width='stretch',
                 caption='LiDAR points only, projected into the same camera view.')

    st.subheader('Detected people and fusion')
    if st.button('Find people', type='primary'):
        try:
            with st.spinner('Detecting people and estimating depth…'):
                # Detector creato solo al click: nessuna detection durante la selezione.
                detections = fuse_detections(TraditionalDetector(label='Person').detect(image), projection)
                st.session_state.detection_result = {
                    'overlay': draw_overlay(image, projection, detections, language='en'),
                    'detections': [d.to_dict() for d in detections],
                }
        except (ValueError, cv2.error) as exc:
            st.error(f'Detection failed: {exc}')

    result = st.session_state.get('detection_result')
    if result is None:
        st.info('Click “Find people” to see the result for this image.')
        return
    detections = result['detections']
    a, b, c = st.columns(3)
    a.metric('HOG detections', len(detections))
    b.metric('With estimated depth', sum(d['depth_m'] is not None for d in detections))
    c.metric('LiDAR points in the image', f'{len(projection.depth):,}')
    st.image(result['overlay'], channels='BGR', width='stretch',
             caption='RGB + LiDAR + bounding boxes. Z is the depth in Camera 2, in metres.')
    if not detections:
        st.info('HOG did not detect any people in this frame. People may still be present.')
    else:
        st.dataframe([{'Detection': i+1, 'Class': d['label'],
                       'Depth Z (m)': round(d['depth_m'], 2) if d['depth_m'] is not None else None,
                       'Points in box': d['points_in_box'], 'Points used': d['points_used'],
                       'HOG score': round(d['score'], 3)} for i, d in enumerate(detections)],
                     hide_index=True, width='stretch')
        st.caption('Depth is the median of the points in the box after IQR filtering; the background may affect it.')


if __name__ == '__main__':
    main()
