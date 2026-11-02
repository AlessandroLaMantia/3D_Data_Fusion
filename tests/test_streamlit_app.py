"""Verifica il flusso selezione -> click -> risultato -> cambio frame."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = (importlib.util.find_spec('streamlit') is not None and
             (ROOT/'data/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000080.png').is_file())


@unittest.skipUnless(AVAILABLE, 'Richiede Streamlit e il sottoinsieme KITTI locale')
class StreamlitFlowTests(unittest.TestCase):
    def test_button_and_frame_change(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT/'streamlit_app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.sidebar.selectbox[0].value, 80)
        self.assertNotIn('detection_result', app.session_state)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['detection_result']['detections']), 2)
        app.sidebar.selectbox[0].set_value(5).run()
        self.assertFalse(app.exception)
        self.assertNotIn('detection_result', app.session_state)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['detection_result']['detections'], [])


if __name__ == '__main__':
    unittest.main()
