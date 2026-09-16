"""Verify publication integrity without importing or executing research code."""
from pathlib import Path, PurePosixPath
import argparse, ast, hashlib, io, json, tarfile, zipfile

ROOT=Path(__file__).resolve().parents[1]
digest=lambda b:hashlib.sha256(b).hexdigest()

def verify_archive(raw,kind):
    if kind=='zip':
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            assert archive.testzip() is None
            names=[i.filename for i in archive.infolist() if not i.is_dir()]
            assert len(names)==len(set(names))
            files={n:archive.read(n) for n in names}
    else:
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
            members=[i for i in archive.getmembers() if i.isfile()]
            assert len(members)==len(set(i.name for i in members))
            files={i.name:archive.extractfile(i).read() for i in members}
    for name in files:
        p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts,name
    count=0
    if 'PUBLICATION_MANIFEST.json' in files:
        manifest=json.loads(files['PUBLICATION_MANIFEST.json'])
        assert set(files)=={'PUBLICATION_MANIFEST.json'}|{x['path'] for x in manifest['files']}
        for item in manifest['files']:
            blob=files[item['path']]
            assert len(blob)==item['bytes'] and digest(blob)==item['sha256'],item['path']
        count+=1
    for name,blob in files.items():
        if name.endswith('.zip'):count+=verify_archive(blob,'zip')
        elif name.endswith(('.tar.gz','.tgz')):count+=verify_archive(blob,'tar')
    return count

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path);args=p.parse_args()
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for item in manifest['files']:
        path=ROOT/item['path'];blob=path.read_bytes()
        assert len(blob)==item['bytes'] and digest(blob)==item['sha256'],item['path']
        if path.suffix=='.py':ast.parse(blob,filename=item['path'])
    assets=0; nested=0
    if args.assets:
        for item in json.loads((ROOT/'data/ASSETS.json').read_text())['assets']:
            raw=(args.assets/item['name']).read_bytes()
            assert len(raw)==item['bytes'] and digest(raw)==item['sha256'],item['name']
            nested+=verify_archive(raw,'zip' if item['name'].endswith('.zip') else 'tar')
            assets+=1
    print(json.dumps({'status':'pass','repository_files':len(manifest['files']),
        'release_assets':assets,'publication_manifests':nested,
        'scope':'file hashes, archive structure and Python syntax; no research experiment executed'},indent=2))

if __name__=='__main__':main()
