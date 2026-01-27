# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['D:\\BG Remover\\BG-Remover\\app.py'],
    pathex=['D:\\BG Remover\\BG-Remover'],
    binaries=[],
    datas=[('D:\\BG Remover\\BG-Remover\\assets', 'assets'), ('D:\\BG Remover\\BG-Remover\\models', 'models')],
    hiddenimports=['onnxruntime', 'rembg', 'PIL', 'cv2', 'numpy'],
    hookspath=['C:\\Users\\ASUS\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pyupdater\\hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='win',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='win',
)
