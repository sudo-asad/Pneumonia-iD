from PySide6 import QtWidgets
from GUI import MainWindow


def init_app():
    app = QtWidgets.QApplication([])
    gui = MainWindow()
    gui.show()
    app.exec()


if __name__ == "__main__":
    init_app()