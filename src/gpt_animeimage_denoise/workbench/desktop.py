"""Start the Windows workbench from a windowless Python interpreter."""
from contextlib import redirect_stderr, redirect_stdout
import ctypes
from pathlib import Path
import traceback

from .server import _open_browser, launch_workbench


def _message(text):
    ctypes.windll.user32.MessageBoxW(None, text, 'GAID 工作台', 0x30)


def _browse(address):
    if not _open_browser(address):
        _message(f'未能自动打开浏览器。请访问 {address}')


def main():
    workspace = Path('outputs/workbench').resolve()
    log = workspace / 'workbench.log'
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8', buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            try:
                return launch_workbench(workspace=workspace, browser=_browse)
            except Exception:
                traceback.print_exc()
                raise
    except Exception as error:
        _message(f'无法启动工作台。\n{error}\n\n日志位置\n{log}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
