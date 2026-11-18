"""Download only the official, pinned DINOv2 implementation and S/14 weights."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
WEIGHT_HASH = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    target = ROOT/'cache/upstream/dinov2'
    receipt = ROOT/'cache/encoder_source.json'
    if not target.exists():
        url = f'https://codeload.github.com/facebookresearch/dinov2/zip/{REVISION}'
        with urllib.request.urlopen(url) as response:
            raw = response.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            prefix = f'dinov2-{REVISION}/'
            for entry in archive.infolist():
                assert entry.filename.startswith(prefix)
                relative = Path(entry.filename[len(prefix):])
                assert '..' not in relative.parts and not relative.is_absolute()
                if entry.is_dir():
                    continue
                dest = target/relative; dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(archive.read(entry))
        hashes = {p.relative_to(target).as_posix():digest(p) for p in target.rglob('*') if p.is_file()}
        receipt.write_text(json.dumps({'revision':REVISION,'archive_sha256':hashlib.sha256(raw).hexdigest(),'files':hashes},indent=2)+'\n')
    info = json.loads(receipt.read_text())
    assert info['revision']==REVISION
    for name, value in info['files'].items():
        assert digest(target/name)==value, name
    weights = ROOT/'cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
    if not weights.exists():
        weights.parent.mkdir(parents=True,exist_ok=True)
        partial = weights.with_suffix('.partial')
        url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
        with urllib.request.urlopen(url) as response, partial.open('wb') as stream:
            shutil.copyfileobj(response,stream)
        assert digest(partial)==WEIGHT_HASH
        partial.rename(weights)
    assert digest(weights)==WEIGHT_HASH
    print('Pinned DINOv2 source and official S/14 weight identity verified.')


if __name__=='__main__':
    main()
