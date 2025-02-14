from PyQt6.QtGui import QAction

def mainMenu(parent):
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")

def viewMenu(parent):
    main_menu = parent.menuBar()
    view_menu = main_menu.addMenu("&View")

    histLut = QAction("&HistLUT", parent)
    histLut.setShortcut("Ctrl+L")
    # histLut.triggered.connect(lambda: io._load_image(parent))
    view_menu.addAction(histLut)