"""Scarica un sottoinsieme reale KITTI Raw usando richieste HTTP Range.

Solo libreria standard: ZipFile legge l'indice remoto e i membri selezionati.
Non scarica le altre camere o l'intera sequenza. Mantiene gli indici originali.
"""
import argparse
import io
import json
from pathlib import Path, PurePosixPath
import time
import urllib.request
import zipfile

DATE = '2011_09_26'
DRIVE = '2011_09_26_drive_0005_sync'
BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'
SEQUENCE_URL = BASE + '2011_09_26_drive_0005/' + DRIVE + '.zip'
CALIB_URL = BASE + DATE + '_calib.zip'


class RemoteZip(io.RawIOBase):
    """File seekable remoto con read-ahead; rifiuta server senza HTTP Range."""
    def __init__(self, url):
        super().__init__()
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            self.size = int(r.headers['Content-Length'])
        self.position = 0
        self.cache_start = 0
        self.cache = b''
        self.downloaded = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if target < 0:
            raise ValueError('Seek negativo')
        self.position = target
        return target

    def read(self, size=-1):
        size = max(0, self.size-self.position) if size < 0 else min(size, max(0, self.size-self.position))
        if not size:
            return b''
        start = self.position
        if not (self.cache_start <= start and start+size <= self.cache_start+len(self.cache)):
            end = min(self.size-1, start+max(size, 1024*1024)-1)
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=90) as r:
                        if r.status != 206:
                            raise RuntimeError('Server senza supporto Range: download interrotto')
                        data = r.read()
                    if len(data) != end-start+1:
                        raise OSError('Risposta HTTP Range incompleta')
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(1+attempt)
            self.cache_start, self.cache = start, data
            self.downloaded += len(data)
        offset = start-self.cache_start
        self.position += size
        return self.cache[offset:offset+size]


def extract_member(archive, name, root):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Percorso ZIP non sicuro')
    root = root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError('File ZIP fuori dalla directory dati')
    if target.exists():
        if target.stat().st_size == archive.getinfo(name).file_size:
            return
        raise ValueError(f'File locale esistente con dimensione inattesa: {target}')
    data = archive.read(name)  # ZipFile verifica anche il CRC.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--step', type=int, default=5, help='Un frame ogni N; default 5')
    args = parser.parse_args()
    if args.step < 1:
        parser.error('--step deve essere >= 1')
    with urllib.request.urlopen(CALIB_URL, timeout=60) as response:
        calib = response.read()
    with zipfile.ZipFile(io.BytesIO(calib)) as archive:
        for name in [f'{DATE}/calib_cam_to_cam.txt', f'{DATE}/calib_velo_to_cam.txt']:
            extract_member(archive, name, args.output)
    with RemoteZip(SEQUENCE_URL) as remote, zipfile.ZipFile(remote) as archive:
        prefix = f'{DATE}/{DRIVE}'
        names = set(archive.namelist())
        frames = sorted(int(PurePosixPath(n).stem) for n in names
                        if n.startswith(prefix+'/image_02/data/') and n.endswith('.png'))
        selected = frames[::args.step]
        if frames and frames[-1] not in selected:
            selected.append(frames[-1])
        for i, frame in enumerate(selected, 1):
            for folder, suffix in [('image_02', 'png'), ('velodyne_points', 'bin')]:
                name = f'{prefix}/{folder}/data/{frame:010d}.{suffix}'
                if name not in names:
                    raise ValueError(f'Frame sincronizzato mancante: {name}')
                extract_member(archive, name, args.output)
            print(f'[{i}/{len(selected)}] frame {frame:010d} | scaricati {remote.downloaded/1e6:.1f} MB', flush=True)
        manifest = {'dataset': 'KITTI Raw synced+rectified', 'date': DATE, 'sequence': DRIVE,
                    'frames': selected, 'original_frame_count': len(frames),
                    'source': SEQUENCE_URL, 'calibration_source': CALIB_URL,
                    'info': 'Sottoinsieme: gli indici originali non sono consecutivi. Non sono dati sintetici.',
                    'license_info': 'https://www.cvlibs.net/datasets/kitti/'}
    path = args.output/DATE/DRIVE/'demo_manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Dataset pronto: {path.parent}', flush=True)


if __name__ == '__main__':
    main()
