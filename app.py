"""桌面应用启动入口。"""
import multiprocessing


def main():
    from ui.window import main as run
    return run()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
