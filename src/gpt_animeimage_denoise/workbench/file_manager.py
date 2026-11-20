"""Show a saved file in the local desktop's file manager."""
import subprocess
import sys


def _reveal_windows(path):
    import ctypes
    from ctypes import wintypes

    ole32, shell32 = ctypes.windll.ole32, ctypes.windll.shell32
    pointer = ctypes.c_void_p
    ole32.CoInitialize.argtypes = (pointer,)
    ole32.CoInitialize.restype = ctypes.HRESULT
    ole32.CoUninitialize.argtypes = ()
    ole32.CoUninitialize.restype = None
    ole32.CoTaskMemFree.argtypes = (pointer,)
    ole32.CoTaskMemFree.restype = None
    shell32.SHParseDisplayName.argtypes = (
        wintypes.LPCWSTR, pointer, ctypes.POINTER(pointer), wintypes.ULONG, ctypes.POINTER(wintypes.ULONG))
    shell32.SHParseDisplayName.restype = ctypes.HRESULT
    shell32.SHOpenFolderAndSelectItems.argtypes = (pointer, wintypes.UINT, pointer, wintypes.DWORD)
    shell32.SHOpenFolderAndSelectItems.restype = ctypes.HRESULT

    ole32.CoInitialize(None)
    try:
        item = pointer()
        shell32.SHParseDisplayName(str(path), None, ctypes.byref(item), 0, None)
        try:
            # A zero count treats the absolute PIDL as the single item to select.
            shell32.SHOpenFolderAndSelectItems(item, 0, None, 0)
        finally:
            ole32.CoTaskMemFree(item)
    finally:
        ole32.CoUninitialize()


def _run(command):
    subprocess.run(command, check=True, capture_output=True, timeout=5)


def reveal_file(path):
    if sys.platform == 'win32':
        _reveal_windows(path)
    elif sys.platform == 'darwin':
        _run(['/usr/bin/open', '-R', str(path)])
    elif sys.platform.startswith('linux'):
        service = 'org.freedesktop.FileManager1'
        try:
            _run(['gdbus', 'call', '--session', '--dest', service, '--object-path', '/org/freedesktop/FileManager1',
                  '--method', service + '.ShowItems', f"['{path.as_uri()}']", "''"])
        except (OSError, subprocess.SubprocessError):
            _run(['gio', 'open', str(path.parent)])
    else:
        raise OSError('当前系统不支持在文件夹中显示。')
