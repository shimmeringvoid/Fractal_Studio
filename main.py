#!/usr/bin/env python3
"""Fractal Studio — interactive fractal explorer and video renderer."""
import os
import sys


def main():
    os.environ.pop("SESSION_MANAGER", None)  # no session-restore state; avoids Qt SM warnings
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Fractal Studio")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
