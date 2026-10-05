"""Build and verify the offline release; no device access or global installs."""
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'jetpack_single'
RELEASE = ROOT / 'outputs' / 'releases'
NAME = 'neurogrip-neon-single-jp512'
FILES = [
    'app.py', 'calibrate.py', 'capture.py', 'check_neon.py', 'depth_runtime.py',
    'ethernet_bus_receiver.py', 'install_service.py', 'mapped_bus.py',
    'network_bus.py', 'protocol.py', 'proximity.py', 'vision_tracking.py',
    'install.sh', 'run.sh', 'calibrate.sh', 'config.json', 'strike_config.json',
    'bus_schema.json', 'requirements-offline.txt', 'README.md', 'BUS.md',
    'WHEEL_AUDIT.json', 'VALIDATION.json',
    'include/neurogrip_bus.h', 'include/neurogrip_shm.h',
    'models/depth.onnx', 'models/manifest.json', 'models/LICENSE', 'models/NOTICE.txt',
]


def digest_file(path):
    with Path(path).open('rb') as handle:
        return digest_stream(handle)


def digest_stream(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024*1024), b''):
        digest.update(block)
    return digest.hexdigest()


def main():
    audit = json.loads((PACKAGE / 'WHEEL_AUDIT.json').read_text(encoding='utf-8'))
    if audit['errors']:
        raise RuntimeError('Wheel metadata audit failed')
    files = list(FILES)
    # Exact audited filenames prevent accidentally bundling desktop QA wheels.
    for wheel in audit['wheels']:
        relative = wheel['file']
        name = Path(relative).name
        if relative != 'wheels/' + name or not name.endswith('.whl'):
            raise RuntimeError('Unexpected wheel path in audit')
        path = PACKAGE / 'wheels' / name
        if digest_file(path) != wheel['sha256']:
            raise RuntimeError('Audited wheel changed: ' + name)
        files.append('wheels/' + name)
    files.sort()
    checksums = {name: digest_file(PACKAGE / name) for name in files}
    (PACKAGE / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (checksums[name], name)
                                             for name in files), encoding='utf-8')
    checksums['SHA256SUMS'] = digest_file(PACKAGE / 'SHA256SUMS')
    files.append('SHA256SUMS')
    RELEASE.mkdir(parents=True, exist_ok=True)
    zip_path, tar_path = RELEASE / (NAME + '.zip'), RELEASE / (NAME + '.tar.gz')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in files:
            archive.write(PACKAGE / name, NAME + '/' + name)
    print('ZIP built; building Linux tar archive.', flush=True)
    with tarfile.open(tar_path, 'w:gz', compresslevel=6) as archive:
        for name in files:
            info = archive.gettarinfo(str(PACKAGE / name), arcname=NAME + '/' + name)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mode = 0o755 if name.endswith('.sh') else 0o644
            with (PACKAGE / name).open('rb') as handle:
                archive.addfile(info, handle)
    with zipfile.ZipFile(zip_path) as archive:
        if set(archive.namelist()) != {NAME + '/' + name for name in files}:
            raise RuntimeError('ZIP file list differs from release allowlist')
        for name in files:
            with archive.open(NAME + '/' + name) as handle:
                if digest_stream(handle) != checksums[name]:
                    raise RuntimeError('ZIP readback failed: ' + name)
    with tarfile.open(tar_path, 'r:gz') as archive:
        members = {member.name: member for member in archive.getmembers()}
        if set(members) != {NAME + '/' + name for name in files}:
            raise RuntimeError('Tar file list differs from release allowlist')
        for name in files:
            member = members[NAME + '/' + name]
            with archive.extractfile(member) as handle:
                if digest_stream(handle) != checksums[name]:
                    raise RuntimeError('Tar readback failed: ' + name)
            if name.endswith('.sh') and member.mode != 0o755:
                raise RuntimeError('Tar script permissions are incorrect')
    artifacts = [{'filename': path.name, 'bytes': path.stat().st_size, 'sha256': digest_file(path)}
                 for path in (tar_path, zip_path)]
    report = {'target': 'JetPack 5.1.2 / L4T R35.4.1 / Python 3.8 / aarch64',
              'version': '2026.10.05', 'file_count': len(files),
              'unpacked_bytes': sum((PACKAGE / name).stat().st_size for name in files),
              'archive_readback': 'Every file verified in both archives', 'artifacts': artifacts,
              'private_capture_images_included': False}
    (RELEASE / 'RELEASE.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (RELEASE / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (item['sha256'],item['filename'])
                                             for item in artifacts), encoding='utf-8')
    print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    main()
