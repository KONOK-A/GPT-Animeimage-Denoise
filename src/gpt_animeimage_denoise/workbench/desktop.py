"""Start the Windows workbench from a windowless Python interpreter."""
from contextlib import redirect_stderr, redirect_stdout
import ctypes
from importlib import metadata
from pathlib import Path
import traceback


_PROJECT = 'gpt-animeimage-denoise'
_INSTALL_HELP = (
    '请按 README.md 的「安装」章节准备 PyTorch，然后在项目根目录运行\n'
    r'.\.venv\Scripts\python.exe -m pip install -e .'
)


def _check_dependencies():
    try:
        requirements = metadata.requires(_PROJECT) or ()
    except metadata.PackageNotFoundError as error:
        raise ImportError(f'缺少依赖包 {_PROJECT}') from error

    missing = []
    for requirement in requirements:
        # The project's runtime requirements are unconditional exact pins.
        package = requirement.partition('==')[0]
        try:
            metadata.distribution(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise ImportError(f'缺少依赖包 {", ".join(missing)}')


def _message(text):
    ctypes.windll.user32.MessageBoxW(None, text, 'GAID 工作台', 0x30)


def _browse(address):
    from gpt_animeimage_denoise.workbench.server import _open_browser

    if not _open_browser(address):
        _message(f'未能自动打开浏览器。请访问 {address}')


def main():
    workspace = Path('outputs/workbench').resolve()
    log = workspace / 'workbench.log'
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8', buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            try:
                _check_dependencies()
                from gpt_animeimage_denoise.workbench.server import launch_workbench

                return launch_workbench(workspace=workspace, browser=_browse)
            except Exception:
                traceback.print_exc()
                raise
    except Exception as error:
        message = f'无法启动工作台。\n{error}'
        if isinstance(error, ImportError):
            message += f'\n\n{_INSTALL_HELP}'
        _message(f'{message}\n\n日志位置\n{log}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
