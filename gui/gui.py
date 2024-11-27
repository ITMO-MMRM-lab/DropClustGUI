import sys, os
sys.path.insert(1, '/home/mellamoarroz/Documents/drop_clus/')

os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

import cv2
import numpy as np
import natsort
import pyqtgraph as pg
import PIL 
from skimage.io import imshow, show
from PIL import Image, ImageEnhance, ImageQt
from qtpy import QtCore
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QFont, QPalette
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QToolButton, QScrollArea, QCheckBox, QGraphicsOpacityEffect, QGroupBox, QComboBox, QPushButton, QProgressBar, QLineEdit

import gui_components
from gui_components import addCustomSlider, extractFrames, ColorSlider, addFilter, countDroplets
from methods.methods import get_gray_img, pil_to_qpixmap

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setFixedWidth(700)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText('\n\n Drop image or video here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')

    def setPixmap(self, image):
        image = image.scaled(QSize(700, 700))
        super().setPixmap(image)

class AppDemo(QMainWindow):
    def __init__(self, size, dpi, pxr, clipboard):
        super(AppDemo, self).__init__()

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.clipboard = clipboard
        Y = int(925 - (25 * dpi * pxr) / 24)
        self.setGeometry(100, 100, min(1200, size.width()),  min(Y,size.height()))
        self.setWindowTitle("Droplets Cluster GUI")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))

        ### MainWidgetLayout
        TOOLBAR_WIDTH = 7
        SPACING = 3
        WIDTH_0 = 25

        self.loaded = False
        self.progress = QProgressBar(self)
        self.masksOn = True
        self.gamma = 1.0
        self.darkmode = True

        builtin = pg.graphicsItems.GradientEditorItem.Gradients.keys()
        self.default_cmaps = ['grey','cyclic','magma','viridis']
        self.cmaps = self.default_cmaps+list(set(builtin) - set(self.default_cmaps))


        scrollable = 1 
        if scrollable:
            self.main_layout = QGridLayout(self)
            self.scrollArea = QScrollArea(self)
            self.scrollArea.setStyleSheet('QScrollArea {border: none;}') # just for main window
            
            self.scrollArea.setWidgetResizable(True)
            # policy = QtWidgets.QSizePolicy()
            # policy.setRetainSizeWhenHidden(True)
            # self.scrollArea.setSizePolicy(policy)

            self.main_widget = QWidget(self)
            self.main_widget.setLayout(self.main_layout) 
            self.scrollArea.setWidget(self.main_widget)

            self.scrollArea.setMinimumSize(self.main_widget.sizeHint())

            self.setCentralWidget(self.scrollArea)
        else:
            self.main_widget = QWidget(self)
            self.main_layout = QGridLayout()
            self.main_widget.setLayout(self.main_layout)
            self.setCentralWidget(self.main_widget)


        s = int(SPACING)
        self.main_layout.setVerticalSpacing(s)
        self.main_layout.setHorizontalSpacing(s)
        self.main_layout.setContentsMargins(10,10,10,10)

        self.imask = 0

        # cross-hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.vLineOrtho = [pg.InfiniteLine(angle=90, movable=False), pg.InfiniteLine(angle=90, movable=False)]
        self.hLineOrtho = [pg.InfiniteLine(angle=0, movable=False), pg.InfiniteLine(angle=0, movable=False)]

        self.frames_dir = ''
        self.frames_list = []
        self.frames_slider = None
        
        self.current_gs_img = None
        self.current_img = None

        self.filters_boundaries_values = [((0, 255), (0, 255), (0, 255)), 
                                            ((0, 127), (0, 127), (0, 127))]
        self.filters_boundaries_labels = ["No filter", "B/W"]
        ###

        ### ToolsViewer
        b = 0
        c = 0
        # Filter group
        self.filter_box = QGroupBox("Filters")
        self.filter_box_g = QGridLayout()
        self.filter_box.setLayout(self.filter_box_g)
        self.main_layout.addWidget(self.filter_box, b, 0, 1, 1)

        b0 = 0
        self.filter_idx = 0
        self.filter_dropdown = QComboBox()
        self.filter_dropdown.addItems(["No filter", "B/W"])
        self.filter_dropdown.setCurrentIndex(0)
        self.filter_dropdown.currentIndexChanged.connect(self.updateFilterDropDown)
        self.filter_box_g.addWidget(self.filter_dropdown, b0, c, 1, 1)
        b0 += 1

        self.color_red = QLabel('Red')
        self.filter_box_g.addWidget(self.color_red, b0, c, 1, 1)
        b0 += 1
        
        self.color_green = QLabel('Green')
        self.filter_box_g.addWidget(self.color_green, b0, c, 1, 1)
        b0 += 1

        self.color_blue = QLabel('Blue')
        self.filter_box_g.addWidget(self.color_blue, b0, c, 1, 1)
        b0 += 1

        self.add_filter_btt = QPushButton('Add Filter')
        self.add_filter_btt.clicked.connect(lambda: addFilter(self))
        self.filter_box_g.addWidget(self.add_filter_btt, b0, c, 1, 1)
        self.add_filter_btt.setEnabled(True)
        self.add_filter_btt.setToolTip("Press to add a filter to segment a color")

        b0 -= 3
        c += 1
        
        self.sliders = []
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 100, 100]]
        colornames = ["red", "Chartreuse", "DodgerBlue"]
        names = ["red", "green", "blue"]
        for r in range(3):
            self.sliders.append(ColorSlider(self, names[r], colors[r]))
            self.sliders[-1].setMinimum(0)
            self.sliders[-1].setMaximum(255)
            self.sliders[-1].setValue([0, 255])
            self.sliders[-1].setEnabled(False)
            self.sliders[-1].setToolTip(
                "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
            )
            #self.sliders[-1].setTickPosition(QSlider.TicksRight)
            self.filter_box_g.addWidget(self.sliders[-1], b0, c, 1, 1)
            b0 += 1
        
        self.count_drops_btt = QPushButton('Count')
        self.count_drops_btt.clicked.connect(lambda: countDroplets(self))
        self.filter_box_g.addWidget(self.count_drops_btt, b0, c, 1, 1)
        self.count_drops_btt.setEnabled(True)
        self.count_drops_btt.setToolTip("Press count segmented droplets")
        b0 += 1

        b += 1

        # Models group
        self.models_box = QGroupBox("Models")
        self.models_box_g = QGridLayout()
        self.models_box.setLayout(self.models_box_g)
        self.main_layout.addWidget(self.models_box, b, 0, 1, 1)

        b0 = 0
        self.diameter = 30
        label = QLabel("diameter (pixels):")
        label.setToolTip(
            'you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)'
        )
        self.models_box_g.addWidget(label, b0, 0, 1, 4)
        self.Diameter = QLineEdit()
        self.Diameter.setToolTip(
            'you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the "cyto3" model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)'
        )
        self.Diameter.setText(str(self.diameter))
        self.Diameter.returnPressed.connect(lambda: print("CEX"))
        self.Diameter.setFixedWidth(50)
        self.models_box_g.addWidget(self.Diameter, b0, 4, 1, 2)
        b0 += 1


        ### ImageViewer
        # self.image_viewer = ImageLabel()
        # self.main_layout.addWidget(self.image_viewer, 0, c + 1, b, 3 * b)

        self.image_viewer = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(self.image_viewer, 0, c + 1, b, 3 * b)
        self.image_viewer.scene().sigMouseClicked.connect(self.plot_clicked)
        self.image_viewer.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        # self.make_orthoviews()
        # self.main_layout.setColumnStretch(TOOLBAR_WIDTH+1, 1)
        # self.main_layout.setMaximumWidth(100)
        # self.ScaleOn.setChecked(False)  # can only toggle off after make_viewbox is called 

        # hard-coded colormaps entirely replaced with pyqtgraph

        if MATPLOTLIB:
            self.colormap = (plt.get_cmap('gist_ncar')(np.linspace(0.0,.9,1000000)) * 255).astype(np.uint8)
            np.random.seed(42) # make colors stable
            self.colormap = self.colormap[np.random.permutation(1000000)]
        else:
            np.random.seed(42) # make colors stable
            self.colormap = ((np.random.rand(1000000,3)*0.8+0.1)*255).astype(np.uint8)
        

        self.is_stack = True # always loading images of same FOV
        # if called with image, load it
        # if image is not None:
        #     self.filename = image
        #     io._load_image(self, self.filename)

        # # training settings
        # d = datetime.datetime.now()
        # self.training_params = {'model_index': 0,
        #                         'learning_rate': 0.1, 
        #                         'weight_decay': 0.0001, 
        #                         'n_epochs': 100,
        #                         'model_name':'CP' + d.strftime("_%Y%m%d_%H%M%S")
        #                        }
        


        self.setAcceptDrops(True)

        self.image_viewer.show()
        self.show()
        ###

    def keyPressEvent(self, event):
        if event.key()==Qt.Key.Key_Right:
            self.frames_slider.setValue(self.frames_slider.value() + 1)
        elif event.key()==Qt.Key.Key_Left:
            self.frames_slider.setValue(self.frames_slider.value() - 1)
        else:
            QWidget.keyPressEvent(self, event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        # if event.mimeData().hasImage:
        #     event.setDropAction(Qt.DropAction.MoveAction)
        #     file_path = event.mimeData().urls()[0].toLocalFile()
        #     self.setImage(file_path)

        #     event.accept()
        # else:
        #     event.ignore()
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if os.path.splitext(files[0])[-1] == '.npy':
            io._load_seg(self, filename=files[0])
        else:
            gui_components.loadImage(self, filename=files[0], load_seg=False)

    def setImage(self, file_path):
        file_ext = file_path.split('.')[1]
        self.frames_dir, self.frames_list = extractFrames(file_path)
        self.current_img, self.current_gs_img = get_gray_img(os.path.join(self.frames_dir, self.frames_list[0]))
        self.image_viewer.setPixmap(
                            pil_to_qpixmap(
                                Image.fromarray(
                                    self.current_img.astype('uint8')
                                    )))
        
        if len(self.frames_list) > 1:
            addCustomSlider(self, len(self.frames_list))

    def updateFilterDropDown(self):
        curr_idx = self.filter_dropdown.currentIndex()
        if curr_idx == 0:
            for slider in self.sliders:
                slider.setEnabled(False)
            
            self.image_viewer.setPixmap(
                            pil_to_qpixmap(
                                Image.fromarray(
                                    self.current_img.astype('uint8')
                                    )))
        else:
            for idx_slider, slider in enumerate(self.sliders):
                slider.setEnabled(True)
                slider.setValue(self.filters_boundaries_values[curr_idx][idx_slider])

    def plot_clicked(self, event):
        if event.button()==QtCore.Qt.LeftButton and (event.modifiers() != QtCore.Qt.ShiftModifier and
                    event.modifiers() != QtCore.Qt.AltModifier):
            if event.double():
                self.recenter()
            elif self.loaded and not self.in_stroke:
                if self.orthobtn.isChecked():
                    items = self.image_viewer.scene().items(event.scenePos())
                    for x in items:
                        if x==self.p0:
                            pos = self.p0.mapSceneToView(event.scenePos())
                            x = int(pos.x())
                            y = int(pos.y())
                            if y>=0 and y<self.Ly and x>=0 and x<self.Lx:
                                self.yortho = y 
                                self.xortho = x
                                self.update_ortho()

    def mouse_moved(self, pos):
        items = self.image_viewer.scene().items(pos)
        for x in items: #why did this get deleted in CP2?
            if x==self.p0:
                mousePoint = self.p0.mapSceneToView(pos)
                # if self.CHCheckBox.isChecked():
                #     self.vLine.setPos(mousePoint.x())
                #     self.hLine.setPos(mousePoint.y())

    def make_viewbox(self):
        self.p0 = gui_components.ViewBoxNoRightDrag(
            parent=self,
            lockAspect=True,
            # name="plot1",
            # border=[100, 100, 100],
            invertY=True,
            # invertX=True
        )

        self.p0.setCursor(QtCore.Qt.CrossCursor)
        self.brush_size=1
        self.image_viewer.addItem(self.p0, 0, 0, rowspan=1, colspan=1)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self,levels=(0,255))
        self.img.autoDownsample = False

        # self.hist = pg.HistogramLUTItem(image=self.img,orientation='horizontal',gradientPosition='bottom')
        self.hist = gui_components.HistLUT(image=self.img,orientation='horizontal',gradientPosition='bottom')

        self.opacity_effect = QGraphicsOpacityEffect()
        self.hist.setGraphicsEffect(self.opacity_effect)

        # self.set_hist_colors() #called elsewhere. no need
        # print(self.hist.__dict__)
        # self.image_viewer.addItem(self.hist,col=0,row=2)
        self.image_viewer.addItem(self.hist,col=0,row=1)


        self.layer = gui_components.ImageDraw(viewbox=self.p0, parent=self)
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self,levels=(0,255))

        self.Ly,self.Lx = 512,512
        
        self.p0.scene().contextMenuItem = self.p0
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)

        
        # policy = QtWidgets.QSizePolicy()
        # policy.setRetainSizeWhenHidden(True)
        # self.hist.setSizePolicy(policy)

    def make_orthoviews(self):
        self.pOrtho, self.imgOrtho, self.layerOrtho = [], [], []
        for j in range(2):
            self.pOrtho.append(pg.ViewBox(
                                lockAspect=True,
                                name=f'plotOrtho{j}',
                                # border=[100, 100, 100],
                                invertY=True,
                                # invertX=True,
                                enableMouse=False
                            ))
            self.pOrtho[j].setMenuEnabled(False)

            self.imgOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self, levels=(0,255)))
            self.imgOrtho[j].autoDownsample = False

            self.layerOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self))
            self.layerOrtho[j].setLevels([0,255])

            #self.pOrtho[j].scene().contextMenuItem = self.pOrtho[j]
            self.pOrtho[j].addItem(self.imgOrtho[j])
            self.pOrtho[j].addItem(self.layerOrtho[j])
            self.pOrtho[j].addItem(self.vLineOrtho[j], ignoreBounds=False)
            self.pOrtho[j].addItem(self.hLineOrtho[j], ignoreBounds=False)
        
        self.pOrtho[0].linkView(self.pOrtho[0].YAxis, self.p0)
        self.pOrtho[1].linkView(self.pOrtho[1].XAxis, self.p0)

    def recenter(self):
        buffer = 10 # leave some space between histogram and image
        dy = self.Ly+buffer
        dx = self.Lx
        
        # make room for scale disk
        # if self.ScaleOn.isChecked():
        #     dy += self.pr
            
        # set the range for whatever is the smallest dimension
        s = self.p0.screenGeometry()
        if s.width()>s.height():
            self.p0.setXRange(0,dx) #centers in x
            self.p0.setYRange(0,dy)
        else:
            self.p0.setYRange(0,dy) #centers in y
            self.p0.setXRange(0,dx)
            
        # unselect sector buttons
        # self.quadbtns.setExclusive(False)
        # for b in range(9):
        #     self.quadbtns.button(b).setChecked(False)      
        # self.quadbtns.setExclusive(True)

    def reset(self):
        # ---- start sets of points ---- #
        self.selected = 0
        self.X2 = 0
        self.resize = -1
        self.onechan = False
        self.loaded = False
        self.channel = [0,1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.ncells = 0
        self.zdraw = []
        self.removed_cell = []
        self.cellcolors = np.array([255,255,255])[np.newaxis,:]
        # -- set menus to default -- #
        self.color = 0
        # self.RGBDropDown.setCurrentIndex(self.color)
        self.view = 0
        # self.RGBChoose.button(self.view).setChecked(True)
        # self.BrushChoose.setCurrentIndex(1)
        # self.SCheckBox.setChecked(True)
        # self.SCheckBox.setEnabled(False)
        self.restore_masks = 0
        self.states = [None for i in range(len(self.default_cmaps))] 

        # -- zero out image stack -- #
        self.opacity = 128 # how opaque masks should be
        self.outcolor = [200,200,255,200]
        self.NZ, self.Ly, self.Lx = 1,512,512
        self.saturation = [[0,255] for n in range(self.NZ)]
        self.gamma = 1
        # self.slider.setMinimum(0)
        # self.slider.setMaximum(100)
        # self.slider.show()
        self.currentZ = 0
        self.flows = [[],[],[],[],[[]]]
        self.stack = np.zeros((1,self.Ly,self.Lx,3))
        # masks matrix
        self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
        # image matrix with a scale disk
        self.radii = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
        self.cellpix = np.zeros((1,self.Ly,self.Lx), np.uint32)
        self.outpix = np.zeros((1,self.Ly,self.Lx), np.uint32)
        self.ismanual = np.zeros(0, 'bool')
        self.accent = self.palette().brush(QPalette.ColorRole.Highlight).color()
        self.update_plot()
        self.progress.setValue(0)
        # self.orthobtn.setChecked(False)
        self.filename = []
        self.loaded = False
        self.recompute_masks = False

    def enable_buttons(self):
        # if len(self.model_strings) > 0:
        #     # self.ModelButton.setStyleSheet(self.styleUnpressed)
        #     self.ModelButton.setEnabled(True)
        # CP2.0 buttons disabled for now     
        # self.StyleToModel.setStyleSheet(self.styleUnpressed)
        # self.StyleToModel.setEnabled(True)
        # for i in range(len(self.StyleButtons)):
        #     self.StyleButtons[i].setEnabled(True)
        #     self.StyleButtons[i].setStyleSheet(self.styleUnpressed)
       
        # self.SizeButton.setEnabled(True)
        # self.SCheckBox.setEnabled(True)
        # self.SizeButton.setStyleSheet(self.styleUnpressed)
        # self.newmodel.setEnabled(True)
        # self.loadMasks.setEnabled(True)
        # self.saveSet.setEnabled(True)
        # self.savePNG.setEnabled(True)
        # self.saveServer.setEnabled(True)
        # self.saveOutlines.setEnabled(True)
        # self.toggle_mask_ops()
        
        
        # self.threshslider.setEnabled(True)
        # self.probslider.setEnabled(True)

        self.update_plot()
        self.setWindowTitle(self.filename)

    def draw_layer(self):
        if self.masksOn and self.view==0: #disable masks for network outputs
            self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
            self.layerz[...,:3] = self.cellcolors[self.cellpix[self.currentZ],:]
            self.layerz[...,3] = self.opacity * (self.cellpix[self.currentZ]>0).astype(np.uint8)
            if self.selected>0:
                self.layerz[self.cellpix[self.currentZ]==self.selected] = np.array([255,255,255,self.opacity])
            cZ = self.currentZ
            stroke_z = np.array([s[0][0] for s in self.strokes])
            inZ = np.nonzero(stroke_z == cZ)[0]
            if len(inZ) > 0:
                for i in inZ:
                    stroke = np.array(self.strokes[i])
                    self.layerz[stroke[:,1], stroke[:,2]] = np.array([255,0,255,100])
        else:
            self.layerz[...,3] = 0

        # if self.outlinesOn:
        #     self.layerz[self.outpix[self.currentZ]>0] = np.array(self.outcolor).astype(np.uint8)

    def update_layer(self):
        self.draw_layer()
        # if (self.masksOn or self.outlinesOn) and self.view==0:
        self.layer.setImage(self.layerz, autoLevels=False)
            # self.layer.setImage(self.layerz[self.currentZ], autoLevels=False)
            
        # self.update_roi_count()
        self.image_viewer.show()
        self.show()

    # def update_roi_count(self):
    #     self.roi_count.setText(f'{self.ncells} ROIs')

    def compute_scale(self):
        self.diameter = 30.0 # float(self.Diameter.text())
        self.pr = 30 # int(float(self.Diameter.text()))
        self.radii_padding = int(self.pr*1.25)
        self.radii = np.zeros((self.Ly+self.radii_padding,self.Lx,4), np.uint8)
        yy,xx = gui_components.disk([self.Ly+self.radii_padding/2-1, self.pr/2+1],
                            self.pr/2, self.Ly+self.radii_padding, self.Lx)
        # rgb(150,50,150)
        self.radii[yy,xx,0] = 255 # making red to correspond to tooltip
        self.radii[yy,xx,1] = 0
        self.radii[yy,xx,2] = 0
        self.radii[yy,xx,3] = 255
        # self.update_plot()
        self.p0.setYRange(0,self.Ly+self.radii_padding)
        self.p0.setXRange(0,self.Lx)
        self.image_viewer.show()
        self.show()

    def clear_all(self):
        self.prev_selected = 0
        self.selected = 0
        self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
        self.cellpix = np.zeros((self.NZ,self.Ly,self.Lx), np.uint32)
        self.outpix = np.zeros((self.NZ,self.Ly,self.Lx), np.uint32)
        self.cellcolors = np.array([255,255,255])[np.newaxis,:]
        self.ncells = 0
        # self.toggle_removals()
        self.update_layer()

    def new_normalize99(self, Y,lower=0.01,upper=99.99,omni=False):
        """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """
        if omni and OMNI_INSTALLED:
            X = omnipose.utils.normalize99(Y)
        else:
            X = Y.copy()
            x01 = np.percentile(X, 1)
            x99 = np.percentile(X, 99)
            X = (X - x01) / (x99 - x01)
        return X

    def update_plot(self):
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape
        
        # toggle off histogram for flow field 
        if self.view==1:
            self.opacity_effect.setOpacity(0.0)  # Hide the histogram
            # self.hist.gradient.setEnabled(False)
            # self.hist.region.setEnabled(False)
            # self.hist.background = None
            self.hist.show_histogram = False
            # self.hist.fillLevel = None


        else:
            self.opacity_effect.setOpacity(1.0)  # Show the histogram
            # self.hist.gradient.setEnabled(True)
            # self.hist.region.setEnabled(True)
            self.hist.show_histogram = True

        # if self.NZ < 2:
        #     self.scroll.hide()
        # else:
        #     self.scroll.show()
                
            
        if self.view==0:
            # self.hist.restoreState(self.histmap_img)
            image = self.stack[self.currentZ]
            if self.onechan:
                # show single channel
                image = self.stack[self.currentZ,:,:,0]
            
            vals = (0.1, 0.99) # self.slider.value()
            image = self.new_normalize99(image,lower=vals[0],upper=vals[1])**self.gamma

            # if self.invert.isChecked():
            #     image = 1-image
            
            # restore to uint8
            image *= 255

            # if self.color==0:
            #     self.img.setImage(image, autoLevels=False, lut=None)
            # elif self.color>0 and self.color<4:
            #     if not self.onechan:
            #         image = image[:,:,self.color-1]
            #     self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color])
            # elif self.color==4:
            #     if not self.onechan:
            #         image = image.mean(axis=-1)
            #     self.img.setImage(image, autoLevels=False, lut=None)
            # elif self.color==5:
            #     if not self.onechan:
            #         image = image.mean(axis=-1)
            #     self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
            
        else:
            image = np.zeros((self.Ly,self.Lx), np.uint8)
            if len(self.flows)>=self.view-1 and len(self.flows[self.view-1])>0:
                image = self.flows[self.view-1][self.currentZ]
        
                
            # if self.view==2: # distance
            #     # self.img.setImage(image,lut=pg.colormap.get('magma').getLookupTable(), levels=(0,255))
            #     self.img.setImage(image,autoLevels=False)
            # elif self.view==3: #boundary
            #     self.img.setImage(image,sutoLevels=False)
            # else:
            #     self.img.setImage(image, autoLevels=False, lut=None)
            # self.img.setLevels([0.0, 255.0])
            # self.set_hist()
        
        self.img.setImage(image,autoLevels=False)

        # Let users customize color maps and have them persist 
        state = self.states[self.view]
        if state is None: #should adda button to reset state to none and update plot
            self.hist.gradient.loadPreset(self.cmaps[self.view]) # select from predefined list
        else:
            self.hist.restoreState(state) #apply chosen color map
            
        self.set_hist_colors()
       
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0,255.0])
        #self.img.set_ColorMap(self.bwr)
        if self.NZ>1 and self.orthobtn.isChecked():
            self.update_ortho()
        
        # self.slider.setLow(self.saturation[self.currentZ][0])
        # self.slider.setHigh(self.saturation[self.currentZ][1])
        # if self.masksOn or self.outlinesOn:
        #     self.layer.setImage(self.layerz[self.currentZ], autoLevels=False) <<< something to do with it 
        self.image_viewer.show()
        self.show()

    def set_hist_colors(self):
        region = self.hist.region
        # c = self.palette().brush(QPalette.ColorRole.Text).color() # selects white or black from palette
        # selecting from the palette can be handy, but the corresponding colors in light and dark mode do not match up well
        color = '#44444450' if self.darkmode else '#cccccc50'
        # c.setAlpha(20)
        region.setBrush(color) # I hate the blue background
        
        c = self.accent
        c.setAlpha(60)
        region.setHoverBrush(c) # also the blue hover
        c.setAlpha(255) # reset accent alpha 
        
        color = '#777' if self.darkmode else '#aaa'
        pen =  pg.mkPen(color=color,width=1.5)
        ph =  pg.mkPen(self.accent,width=2)
        # region.lines[0].setPen(None)
        # region.lines[0].setHoverPen(color='c',width = 5)
        # region.lines[1].setPen('r')
        
        # self.hist.paint(self.hist.plot)
        # print('sss',self.hist.regions.__dict__)
        
        for line in region.lines:
            # c.setAlpha(100)
            line.setPen(pen)
            # c.setAlpha(200)
            line.setHoverPen(ph)
        
        self.hist.gradient.gradRect.setPen(pen)
        # c.setAlpha(100)
        self.hist.gradient.tickPen = pen
        self.set_tick_hover_color() 
        
        ax = self.hist.axis
        ax.setPen(color=(0,)*4) # transparent 
        # ax.setTicks([0,255])
        # ax.setStyle(stopAxisAtTick=(True,True))

        # self.hist = self.img.getHistogram()
        # self.hist.disableAutoHistogramRange()
        # c = self.palette().brush(QPalette.ColorRole.ToolTipBase).color() # selects white or black from palette
        # print(c.getRgb(),'ccc')
        
        # c.setAlpha(100)
        self.hist.fillHistogram(fill=True, level=1.0, color= '#222' if self.darkmode else '#bbb')
        self.hist.axis.style['showValues'] = 0
        self.hist.axis.style['tickAlpha'] = 0
        self.hist.axis.logMode = 1
        # self.hist.plot.opts['antialias'] = 1
        self.hist.setLevels(min=0, max=255)
        
        # policy = QtWidgets.QSizePolicy()
        # policy.setRetainSizeWhenHidden(True)
        # self.hist.setSizePolicy(policy)
        
        # self.histmap_img = self.hist.saveState()

    def set_tick_hover_color(self):
        for tick in self.hist.gradient.ticks:
            tick.hoverPen = pg.mkPen(self.accent,width=2)

app = QApplication(sys.argv)

screen = app.primaryScreen()
dpi = screen.logicalDotsPerInch()
pxr = screen.devicePixelRatio()
size = screen.availableGeometry()
clipboard = app.clipboard()

demo = AppDemo(size, dpi, pxr, clipboard)
demo.show()
sys.exit(app.exec())


