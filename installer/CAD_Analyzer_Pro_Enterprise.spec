# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('../utils', 'utils'), ('../sample_files', 'sample_files'), ('../desktop_app/backend.py', '.')]
binaries = []
hiddenimports = ['PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'matplotlib', 'matplotlib.backends.backend_qt5agg', 'matplotlib.figure', 'numpy', 'shapely', 'shapely.geometry', 'shapely.ops', 'shapely.affinity', 'ezdxf', 'ezdxf.recover', 'concurrent.futures', 'threading', 'hashlib', 'json', 'time', 'os', 'sys', 'pathlib', 'typing', 'dataclasses', 'collections', 'scipy', 'scipy.spatial', 'scipy.spatial.distance', 'sklearn', 'sklearn.cluster', 'PIL', 'PIL.Image']
hiddenimports += collect_submodules('shapely')
hiddenimports += collect_submodules('ezdxf')
hiddenimports += collect_submodules('matplotlib')
tmp_ret = collect_all('PyQt5')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('shapely')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('ezdxf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('PIL')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['..\\desktop_app\\main_production.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [('v', None, 'OPTION')],
    exclude_binaries=True,
    name='CAD_Analyzer_Pro_Enterprise',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CAD_Analyzer_Pro_Enterprise',
)
