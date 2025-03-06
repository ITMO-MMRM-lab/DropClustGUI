import os
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QGraphicsProxyWidget
from gui_components import CustomSlider

def mainMenu(parent):
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")

def viewMenu(parent):
    main_menu = parent.menuBar()
    view_menu = main_menu.addMenu("&View")

    view_al_group =  QActionGroup(view_menu)

    hist_lut = QAction("&HistLUT", parent, checkable=True, checked=True)
    hist_lut.setShortcut("Ctrl+L")
    hist_lut.triggered.connect(lambda: setHistLut(parent))

    custom_slider = QAction("&Frames slider", parent, checkable=True)
    custom_slider.setShortcut("Ctrl+k")
    custom_slider.triggered.connect(lambda: setFramesSlider(parent))

    view_menu.addAction(hist_lut)
    view_menu.addAction(custom_slider)
    
    view_al_group.addAction(hist_lut)
    view_al_group.addAction(custom_slider)


def setHistLut(parent):
    # check last item and remove it
    last_item = parent.image_viewer.getItem(row=1, col=0)
    parent.image_viewer.removeItem(last_item)

    parent.image_viewer.addItem(parent.hist,col=0,row=1)

def setFramesSlider(parent):
    # getting the number of frames from gif/video source
    num_frames = sum([len(files) for _, _, files in os.walk(parent.frames_dir)])

    # check last item and remove it
    last_item = parent.image_viewer.getItem(row=1, col=0)
    parent.image_viewer.removeItem(last_item)

    proxy = QGraphicsProxyWidget()
    proxy.setWidget(CustomSlider(1, num_frames, parent))

    parent.image_viewer.addItem(proxy,col=0,row=1)