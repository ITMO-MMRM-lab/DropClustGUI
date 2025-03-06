import sys, os, time
sys.path.insert(1, '../drop_clus/')

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
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QToolButton, QScrollArea, QCheckBox, QGraphicsOpacityEffect, QGroupBox, QComboBox, QPushButton, QProgressBar, QLineEdit, QGraphicsProxyWidget

import gui_components
import models, core, dynamics
import utils
import iio
import iiio
import menus
from transforms import resize_image
# from utils import to_8_bit
from gui_components import ColorSlider, addFilter, countDroplets #, addCustomSlider
from methods import get_gray_img, pil_to_qpixmap, run_clustering

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False


OMNI_INSTALLED = 0
DEFAULT_MODEL = 'cyto2'

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
        self.outlinesOn = False
        self.gamma = 1.0
        self.darkmode = True
        self.load_3D = False
        self.ncolor = False
        self.selected = 0

        builtin = pg.graphicsItems.GradientEditorItem.Gradients.keys()
        self.default_cmaps = ['grey','cyclic','magma','viridis']
        self.cmaps = self.default_cmaps+list(set(builtin) - set(self.default_cmaps))


        self.model_strings = models.MODEL_NAMES.copy()
        self.current_model = "cyto3"

        menus.mainMenu(self)
        menus.viewMenu(self)

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
        self.curr_idx_frame = 1
        self.num_frames = 1
 
        self.current_gs_img = None
        self.current_img = None

        self.filters_boundaries_values = [((0, 255), (0, 255), (0, 255)), 
                                            ((0, 127), (0, 127), (0, 127))]
        self.filters_boundaries_labels = ["No filter", "B/W"]
        ###

        ### ToolsViewer
        curr_row = 0
        curr_col = 0

        ## Filter group (g1)
        self.filter_box = QGroupBox("Filters")
        self.filter_box_g = QGridLayout()
        self.filter_box.setLayout(self.filter_box_g)
        self.main_layout.addWidget(self.filter_box, curr_row, curr_col, 1, 1)

        g1_row = 0
        g1_col = 0
        self.filter_idx = 0
        self.filter_dropdown = QComboBox()
        self.filter_dropdown.addItems(["No filter", "B/W"])
        self.filter_dropdown.setCurrentIndex(0)
        self.filter_dropdown.currentIndexChanged.connect(self.updateFilterDropDown)
        self.filter_box_g.addWidget(self.filter_dropdown, g1_row, g1_col, 1, 1)
        g1_row += 1

        self.color_red = QLabel('Red')
        self.filter_box_g.addWidget(self.color_red, g1_row, g1_col, 1, 1)
        g1_row += 1
        
        self.color_green = QLabel('Green')
        self.filter_box_g.addWidget(self.color_green, g1_row, g1_col, 1, 1)
        g1_row += 1

        self.color_blue = QLabel('Blue')
        self.filter_box_g.addWidget(self.color_blue, g1_row, g1_col, 1, 1)
        g1_row += 1

        self.add_filter_btt = QPushButton('Add Filter')
        self.add_filter_btt.clicked.connect(lambda: addFilter(self))
        self.filter_box_g.addWidget(self.add_filter_btt, g1_row, g1_col, 1, 1)
        self.add_filter_btt.setEnabled(True)
        self.add_filter_btt.setToolTip("Press to add a filter to segment a color")

        g1_row -= 3
        g1_col += 1
        
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
            self.filter_box_g.addWidget(self.sliders[-1], g1_row, g1_col, 1, 1)
            g1_row += 1
        
        self.count_drops_btt = QPushButton('Count')
        self.count_drops_btt.clicked.connect(lambda: countDroplets(self))
        self.filter_box_g.addWidget(self.count_drops_btt, g1_row, g1_col, 1, 1)
        self.count_drops_btt.setEnabled(True)
        self.count_drops_btt.setToolTip("Press count segmented droplets")
        g1_row += 1

        ## K-means group (g3)
        curr_row += 1
        self.cluster_box = QGroupBox("Clustering")
        self.cluster_box_g = QGridLayout()
        self.cluster_box.setLayout(self.cluster_box_g)
        self.main_layout.addWidget(self.cluster_box, curr_row, curr_col, 1, 1)

        g3_row = 0
        g3_col = 0

        ### Clustering
        self.useCentroids = True
        self.CCheckBox = QCheckBox('use centroids')
        self.CCheckBox.setToolTip('Check to use color roots for clustering')
        # self.CCheckBox.setStyleSheet(self.checkstyle)
        # self.CCheckBox.setFont(self.medfont)
        self.CCheckBox.setChecked(self.useCentroids)
        self.CCheckBox.toggled.connect(lambda: self.toggle_clustering(g3_row + 1, g3_col))
        self.cluster_box_g.addWidget(self.CCheckBox, g3_row, g3_col, 1, 1)
        g3_row += 1

        self.seeds_list = []
        self.num_clust_label = QLabel("# Clusters")
        # self.roi_count.setFont(self.boldfont)
        self.num_clust_label.setAlignment(QtCore.Qt.AlignLeft)
        self.cluster_box_g.addWidget(self.num_clust_label, g3_row, g3_col, 1, 1)

        # number of clusters
        self.k_value = 0
        self.K_value = QLineEdit()
        self.K_value.setToolTip(
            'Defines the number of clusters'
        )
        self.K_value.setText(str(0))
        self.K_value.returnPressed.connect(lambda: self.create_seeds(g3_row, g3_col))
        self.K_value.setFixedWidth(50)
        self.cluster_box_g.addWidget(self.K_value, g3_row, g3_col + 1, 1, 1)
        g3_row += 1

        # sub-box for seed buttons
        self.seeds_box = QGroupBox()
        self.seeds_box_g = QGridLayout()
        self.seeds_box.setLayout(self.seeds_box_g)
        self.cluster_box_g.addWidget(self.seeds_box, g3_row, g3_col, 1, 1)
        g3_row += 1

        # run clustering algorithm button
        self.run_cluster_btt = QPushButton('Run clustering')
        self.run_cluster_btt.clicked.connect(lambda: run_clustering(self, self.k_value))
        self.cluster_box_g.addWidget(self.run_cluster_btt, g3_row, g3_col, 1, 1)
        self.run_cluster_btt.setEnabled(True)
        self.run_cluster_btt.setToolTip("Press to cluster the droplets")
        ##

        ## Models group (g2)
        curr_row += 1
        self.models_box = QGroupBox("Models")
        self.models_box_g = QGridLayout()
        self.models_box.setLayout(self.models_box_g)
        self.main_layout.addWidget(self.models_box, curr_row, curr_col, 1, 1)

        g2_row = 0
        g2_col = 0
        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QCheckBox('masks')
        self.MCheckBox.setToolTip('Press X or M to toggle masks')
        # self.MCheckBox.setStyleSheet(self.checkstyle)
        # self.MCheckBox.setFont(self.medfont)
        self.MCheckBox.setChecked(self.masksOn)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.models_box_g.addWidget(self.MCheckBox, g2_row, g2_col, 1, 2)

        self.opacity = 128
        self.Opacity = QLineEdit()
        self.Opacity.setToolTip(
            'Defines the opacity of the masks'
        )
        self.Opacity.setText(str(128))
        self.Opacity.returnPressed.connect(self.update_opacity)
        self.Opacity.setFixedWidth(50)
        self.models_box_g.addWidget(self.Opacity, g2_row, g2_col + 1, 1, 2)

        g2_row += 1
        # turn off outlines
        self.outlinesOn = False # turn off by default
        self.OCheckBox = QCheckBox('outlines')
        self.OCheckBox.setToolTip('Press Z or O to toggle outlines')
        # self.OCheckBox.setStyleSheet(self.checkstyle)
        # self.OCheckBox.setFont(self.medfont)
        self.models_box_g.addWidget(self.OCheckBox, g2_row, g2_col, 1, 2)
        
        self.OCheckBox.setChecked(False)
        self.OCheckBox.toggled.connect(self.toggle_masks)

        g2_row += 1
        # turn off outlines
        self.AllCheckBox = QCheckBox('apply to all frames')
        # self.AllCheckBox.setStyleSheet(self.checkstyle)
        # self.AllCheckBox.setFont(self.medfont)
        self.AllCheckBox.setChecked(False)
        self.models_box_g.addWidget(self.AllCheckBox, g2_row, g2_col, 1, 1)

        g2_row += 1
        self.diameter = 30
        label = QLabel("Cell diameter (pixels):")
        label.setToolTip(
            'you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)'
        )
        self.models_box_g.addWidget(label, g2_row, g2_col, 1, 1)
        self.Diameter = QLineEdit()
        self.Diameter.setToolTip(
            'you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the "cyto3" model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)'
        )
        self.Diameter.setText(str(self.diameter))
        self.Diameter.returnPressed.connect(self.compute_scale)
        self.Diameter.setFixedWidth(50)
        self.models_box_g.addWidget(self.Diameter, g2_row, g2_col + 1, 1, 1)
       
        g2_row += 1
        # choose channel
        self.ChannelChoose = [QComboBox(), QComboBox()]
        self.ChannelChoose[0].addItems(["0: gray", "1: red", "2: green", "3: blue"])
        self.ChannelChoose[1].addItems(["0: none", "1: red", "2: green", "3: blue"])
        cstr = ["chan to segment:", "chan2 (optional): "]
        for i in range(2):
            # self.ChannelChoose[i].setFont(self.medfont)
            label = QLabel(cstr[i])
            # label.setFont(self.medfont)
            if i == 0:
                label.setToolTip(
                    "this is the channel in which the cytoplasm or nuclei exist that you want to segment"
                )
                self.ChannelChoose[i].setToolTip(
                    "this is the channel in which the cytoplasm or nuclei exist that you want to segment"
                )
            else:
                label.setToolTip(
                    "if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option"
                )
                self.ChannelChoose[i].setToolTip(
                    "if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option"
                )
            self.models_box_g.addWidget(label, g2_row + i, g2_col, 1, 1)
            self.models_box_g.addWidget(self.ChannelChoose[i], g2_row + i, g2_col + 1, 1, 1)

        g2_row += 2

        # use GPU
        self.useGPU = QCheckBox("use GPU")
        self.useGPU.setToolTip(
            "if you have specially installed the <i>cuda</i> version of torch, then you can activate this"
        )
        # self.useGPU.setFont(self.medfont)
        self.check_gpu()
        self.models_box_g.addWidget(self.useGPU, g2_row, g2_col, 1, 1)

        # compute segmentation with general models
        self.net_text = ["run cyto3"]
        nett = ["cellpose super-generalist model"]

        #label = QLabel("Run:")
        #label.setFont(self.boldfont)
        #label.setFont(self.medfont)
        #self.segBoxG.addWidget(label, b0, 0, 1, 2)
        self.StyleButtons = []
        for j in range(len(self.net_text)):
            self.StyleButtons.append(
                gui_components.ModelButton(self, self.net_text[j], self.net_text[j]))
            self.models_box_g.addWidget(self.StyleButtons[-1], g2_row, g2_col + 1, 1, 1)
            #self.StyleButtons[-1].setFixedWidth(140)
            self.StyleButtons[-1].setToolTip(nett[j])

        # segmentation models dropdown
        g2_row += 1
        self.ModelChoose = QComboBox()
        if len(self.model_strings) > len(models.MODEL_NAMES):
            current_index = len(models.MODEL_NAMES)
            # self.NetAvg.setCurrentIndex(1)
        else:
            current_index = models.MODEL_NAMES.index(DEFAULT_MODEL)
        self.ModelChoose.addItems(self.model_strings) #added omnipose model names
        # self.ModelChoose.setStyleSheet(self.dropdowns(width=WIDTH_5))
        # self.ModelChoose.setFont(self.smallfont)
        self.ModelChoose.setCurrentIndex(current_index)
        self.models_box_g.addWidget(self.ModelChoose, g2_row, g2_col, 1, 1)

        # progress bar
        self.progress = QProgressBar(self)
        self.models_box_g.addWidget(self.progress, g2_row, g2_col + 1, 1, 1)

        # roi counter
        self.roi_count = QLabel("0 ROIs")
        # self.roi_count.setFont(self.boldfont)
        self.roi_count.setAlignment(QtCore.Qt.AlignCenter)
        self.models_box_g.addWidget(self.roi_count, g2_row, g2_col + 2, 1, 1)


        ### ImageViewer
        # self.image_viewer = ImageLabel()
        # self.main_layout.addWidget(self.image_viewer, 0, c + 1, b, 3 * b)

        curr_col += 1
        self.image_viewer = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(self.image_viewer, 0, curr_col, curr_row + 1, 1)
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
            iio._load_image(self, filename=files[0], load_seg=False)
            # gui_components.loadImage(self, filename=files[0], load_seg=False)

    def setImage(self, file_path):
        file_ext = file_path.split('.')[1]
        self.frames_dir, self.frames_list = iiio.extract_frames(file_path)
        self.current_img, self.current_gs_img = get_gray_img(os.path.join(self.frames_dir, self.frames_list[0]))
        self.image_viewer.setPixmap(
                            pil_to_qpixmap(
                                Image.fromarray(
                                    self.current_img.astype('uint8')
                                    )))
        
        if len(self.frames_list) > 1:
            addCustomSlider(self, len(self.frames_list))

    def create_seeds(self, row, col):
        # num_clust = int(self.num_clust.text())
        self.k_value = int(float(self.K_value.text()))

        # Delete previous widgets
        for seed in self.seeds_list:
            self.seeds_box_g.removeWidget(seed)
        self.seeds_list = []

        # Create new widgets
        if self.useCentroids:
            for idx in range(self.k_value):
                button = QPushButton("Seed color #" + str(idx + 1))
                button.clicked.connect(lambda checked, idx=idx: self.get_drop_color(idx)) #WHYYYYYYYYY
                button.setToolTip("Click and then select the drop color you want to use as a seed for clustering")
                button.setEnabled(True)
                self.seeds_list.append(button)

                self.seeds_box_g.addWidget(button, row + idx + 1, col, 1, 1)

        else:
            for idx in range(self.k_value):
                button = QPushButton("Cluster #" + str(idx + 1))
                # button.clicked.connect(lambda checked, idx=idx: self.get_drop_color(idx)) #WHYYYYYYYYY
                button.setToolTip("Shows the average color of the cluster")
                button.setEnabled(False)
                self.seeds_list.append(button)

                self.seeds_box_g.addWidget(button, row + idx + 1, col, 1, 1)

    def scale_contour(self, cnt, scale):
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)

        return cnt_scaled

    def get_drop_color(self, idx):
        if self.selected > 0:
            # get image
            image = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)

            # get the mask of the selected droplet
            masked = np.copy(self.cellpix[0])
            masked[self.selected != self.cellpix[0]] = 0
            masked[self.selected == self.cellpix[0]] = 255

            masked = masked.astype(np.uint8)

            ### REDUCE MASK
            contours, _ = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = [self.scale_contour(contour, 0.7) for contour in contours]
            
            masked = np.zeros_like(image[:, :, 0])  # This mask is used to get the mean color of the specific bead (contour), for kmeans
            cv2.drawContours(masked, [contours[0]], 0, 255, -1)

            # B_mean, G_mean, R_mean, _ = cv2.mean(image, mask=new_masked)  # Average
            ###

            # get the average
            B_mean, G_mean, R_mean, _ = cv2.mean(image, mask=masked)
            print("B_mean: ", B_mean)
            print("G_mean: ", G_mean)
            print("R_mean: ", R_mean, "\n")
            self.seeds_list[idx].setStyleSheet("background-color: rgb(" + str(int(R_mean)) + ", " + str(int(G_mean)) + ", " + str(int(B_mean)) + ");")

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
                if False: # self.orthobtn.isChecked():
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

    def check_gpu(self, use_torch=True):
        # also decide whether or not to use torch
        self.torch = use_torch
        self.useGPU.setChecked(False)
        self.useGPU.setEnabled(False)    
        if self.torch and core.use_gpu(use_torch=True):
            self.useGPU.setEnabled(True)
            self.useGPU.setChecked(True)
        else:
            self.useGPU.setStyleSheet("color: rgb(80,80,80);")
    
    def get_channels(self):
        channels = [self.ChannelChoose[0].currentIndex(), self.ChannelChoose[1].currentIndex()]
        if self.current_model=='nuclei':
            channels[1] = 0

        # if self.nchan==1:
        #     channels = None
        return channels

    def get_thresholds(self): 
        try:
            return self.threshslider.value(), self.probslider.value()
        except Exception as e:
            print('flow threshold or cellprob threshold not a valid number, setting to defaults')
            self.flow_threshold.setText('0.0')
            self.cellprob_threshold.setText('0.0')
            return 0.0, 0.0

    def toggle_masks(self):
        if self.MCheckBox.isChecked():
            self.masksOn = True
            self.restore_masks = True
        else:
            self.masksOn = False
            self.restore_masks = False
            
        if self.OCheckBox.isChecked():
            self.outlinesOn = True
        else:
            self.outlinesOn = False

        if not self.masksOn and not self.outlinesOn:
            self.p0.removeItem(self.layer)
            self.layer_off = True
        else:
            if self.layer_off:
                self.p0.addItem(self.layer)
            self.draw_layer()
            self.update_layer()
        if self.loaded:
            # self.update_plot()
            self.update_layer()

    def toggle_clustering(self, n_row, n_col):
        if self.CCheckBox.isChecked():
            self.useCentroids = True
            self.create_seeds(n_row, n_col)
        else:
            self.useCentroids = False
            self.create_seeds(n_row, n_col)

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
        # self.opacity = 128 # how opaque masks should be
        self.outcolor = [0, 0, 255, 255]
        self.NZ, self.Ly, self.Lx = 1,512,512
        self.saturation = [[0,255] for n in range(self.NZ)]
        self.gamma = 1
        self.currentZ = 0
        self.flows = [[],[],[],[],[[]]]
        self.masks = []
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
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(True)
            # self.StyleButtons[i].setStyleSheet(self.styleUnpressed)
       
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

    def redraw_masks(self, masks=True, outlines=True, draw=True):
        self.draw_layer()

    def draw_masks(self):
        self.draw_layer()

    def draw_layer(self):
        if self.masksOn and self.view==0: #disable masks for network outputs
            self.layerz = np.zeros((self.Ly,self.Lx,4), np.uint8)
            self.layerz[...,:3] = self.cellcolors[self.cellpix[self.currentZ],:]
            self.layerz[...,3] = self.opacity * (self.cellpix[self.currentZ]>0).astype(np.uint8)
            if self.selected>0:
                self.layerz[self.cellpix[self.currentZ]==self.selected] = np.array([255,255,255, 128])
            cZ = self.currentZ
            stroke_z = np.array([s[0][0] for s in self.strokes])
            inZ = np.nonzero(stroke_z == cZ)[0]
            if len(inZ) > 0:
                for i in inZ:
                    stroke = np.array(self.strokes[i])
                    self.layerz[stroke[:,1], stroke[:,2]] = np.array([255,0,255,100])
        else:
            self.layerz[...,3] = 0

        if self.outlinesOn:
            self.layerz[self.outpix[self.currentZ]>0] = np.array(self.outcolor).astype(np.uint8)

    def update_layer(self):
        self.draw_layer()
        # if (self.masksOn or self.outlinesOn) and self.view==0:
        self.layer.setImage(self.layerz, autoLevels=False)
            # self.layer.setImage(self.layerz[self.currentZ], autoLevels=False)
            
        self.update_roi_count()
        self.image_viewer.show()
        self.show()

    def update_roi_count(self):
        self.roi_count.setText(f'{self.ncells} ROIs')

    def compute_scale(self):
        self.diameter = float(self.Diameter.text())
        self.pr = int(float(self.Diameter.text()))
        self.radii_padding = int(self.pr * 1.25)
        self.radii = np.zeros((self.Ly + self.radii_padding, self.Lx, 4), np.uint8)
        yy,xx = gui_components.disk([self.Ly + self.radii_padding / 2 - 1, self.pr / 2 + 1],
                            self.pr / 2, self.Ly + self.radii_padding, self.Lx)
        # rgb(150,50,150)
        self.radii[yy,xx,0] = 255 # making red to correspond to tooltip
        self.radii[yy,xx,1] = 0
        self.radii[yy,xx,2] = 0
        self.radii[yy,xx,3] = 255
        # self.update_plot()
        self.p0.setYRange(0, self.Ly + self.radii_padding)
        self.p0.setXRange(0, self.Lx)

        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0,255.0])

        self.image_viewer.show()
        self.show()

    def update_opacity(self):
        self.opacity = int(float(self.Opacity.text()))
        self.update_layer()

    def update_cellcolors(self, df_colors):
        # new_cellcolors = np.array([255,255,255])[np.newaxis,:]
        new_colors = []
        # for val in df_colors:
        #     if val == 0:
        #         new_cellcolors = np.append(new_cellcolors, [255, 0, 0], axis=0)
        #     else:
        #         new_cellcolors = np.append(new_cellcolors, [0, 0, 255], axis=0)

        new_colors = [[255, 0, 0] if val == 0 else [0, 0, 255] for val in df_colors]

        new_cellcolors = np.concatenate((np.array([[255,255,255]]), new_colors), axis=0).astype(np.uint8)
        self.cellcolors = new_cellcolors
        self.update_layer()

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

    def select_cell(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z]==idx] = np.array([255,255,255, self.opacity])
            self.update_layer()

    def unselect_cell(self):
        if self.selected > 0:
            idx = self.selected
            if idx < self.ncells+1:
                z = self.currentZ
                self.layerz[self.cellpix[z]==idx] = np.append(self.cellcolors[idx], self.opacity)
                if self.outlinesOn:
                    self.layerz[self.outpix[z]==idx] = np.array(self.outcolor).astype(np.uint8)
                    #[0,0,0,self.opacity])
                self.update_layer()
        self.selected = 0

    def remove_cell(self, idx):
        # remove from manual array
        self.selected = 0
        if self.NZ > 1:
            zextent = ((self.cellpix==idx).sum(axis=(1,2)) > 0).nonzero()[0]
        else:
            zextent = [0]
        for z in zextent:
            cp = self.cellpix[z]==idx
            op = self.outpix[z]==idx
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = 0
            self.outpix[z, op] = 0    
            if z==self.currentZ:
                # remove from mask layer
                self.layerz[cp] = np.array([0,0,0,0])

        # reduce other pixels by -1
        self.cellpix[self.cellpix>idx] -= 1
        self.outpix[self.outpix>idx] -= 1
        
        if self.NZ==1:
            self.removed_cell = [self.ismanual[idx-1], self.cellcolors[idx], np.nonzero(cp), np.nonzero(op)]
            self.redo.setEnabled(True)
            ar, ac = self.removed_cell[2]
            d = datetime.datetime.now()        
            self.track_changes.append([d.strftime("%m/%d/%Y, %H:%M:%S"), 'removed mask', [ar,ac]])
        # remove cell from lists
        self.ismanual = np.delete(self.ismanual, idx-1)
        self.cellcolors = np.delete(self.cellcolors, [idx], axis=0)
        del self.zdraw[idx-1]
        self.ncells -= 1
        print('GUI_INFO: removed cell %d'%(idx-1))
        
        self.update_layer()
        if self.ncells==0:
            self.ClearButton.setEnabled(False)
        if self.NZ==1:
            io._save_sets(self)

    def merge_cells(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected != self.prev_selected:
            for z in range(self.NZ):
                ar0, ac0 = np.nonzero(self.cellpix[z]==self.prev_selected)
                ar1, ac1 = np.nonzero(self.cellpix[z]==self.selected)
                touching = np.logical_and((ar0[:,np.newaxis] - ar1)<3,
                                            (ac0[:,np.newaxis] - ac1)<3).sum()
                ar = np.hstack((ar0, ar1))
                ac = np.hstack((ac0, ac1))
                vr0, vc0 = np.nonzero(self.outpix[z]==self.prev_selected)
                vr1, vc1 = np.nonzero(self.outpix[z]==self.selected)
                self.outpix[z, vr0, vc0] = 0    
                self.outpix[z, vr1, vc1] = 0    
                if touching > 0:
                    mask = np.zeros((np.ptp(ar)+4, np.ptp(ac)+4), np.uint8)
                    mask[ar-ar.min()+2, ac-ac.min()+2] = 1
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    pvc, pvr = contours[-2][0].squeeze().T            
                    vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2
                    
                else:
                    vr = np.hstack((vr0, vr1))
                    vc = np.hstack((vc0, vc1))
                color = self.cellcolors[self.prev_selected]
                self.draw_mask(z, ar, ac, vr, vc, color, idx=self.prev_selected)
            self.remove_cell(self.selected)
            print('GUI_INFO: merged two cells')
            self.update_layer()
            io._save_sets(self)
            self.undo.setEnabled(False)      
            self.redo.setEnabled(False)    

    def undo_remove_cell(self):
        if len(self.removed_cell) > 0:
            z = 0
            ar, ac = self.removed_cell[2]
            vr, vc = self.removed_cell[3]
            color = self.removed_cell[1]
            self.draw_mask(z, ar, ac, vr, vc, color)
            self.toggle_mask_ops()
            self.cellcolors = np.append(self.cellcolors, color[np.newaxis,:], axis=0)
            self.ncells+=1
            self.ismanual = np.append(self.ismanual, self.removed_cell[0])
            self.zdraw.append([])
            print('>>> added back removed cell')
            self.update_layer()
            io._save_sets(self)
            self.removed_cell = []
            self.redo.setEnabled(False)

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

    def initialize_model(self):
        self.get_model_path()


        if self.current_model in models.MODEL_NAMES:

            # make sure 2-channel models are initialized correctly
            if self.current_model in models.C2_MODEL_NAMES:
                self.nchan = 2
                # self.ChanNumber.setText(str(self.nchan))

            # ensure that the boundary/nclasses is set correctly
            # self.boundary.setChecked(self.current_model in models.BD_MODEL_NAMES)
            self.nclasses = 3 # 2 + self.boundary.isChecked()

            # logger.info(f'Initializing model: nchan set to {self.nchan}, nclasses set to {self.nclasses}, dim set to {self.dim}')        

            # if self.SizeModel.isChecked():
            #     self.model = models.Cellpose(gpu=self.useGPU.isChecked(),
            #                                  use_torch=self.torch,
            #                                  model_type=self.current_model,
            #                                  nchan=self.nchan,
            #                                  nclasses=self.nclasses)
            # else:
            self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                                use_torch=self.torch,
                                                model_type=self.current_model,                                             
                                                nchan=self.nchan,
                                                nclasses=self.nclasses)
        else:
            self.nclasses = 3 # 2 + self.boundary.isChecked()
            self.model = models.CellposeModel(gpu=self.useGPU.isChecked(), 
                                              use_torch=True,
                                              pretrained_model=self.current_model_path,                                             
                                              nchan=self.nchan,
                                              nclasses=self.nclasses)

    def compute_model(self):
        self.progress.setValue(10)
        QApplication.processEvents() 

        try:
            tic=time.time()
            self.clear_all()
            self.flows = [[[],[],[]] for i in range(self.num_frames)] if self.AllCheckBox.isChecked() else [[[],[],[]]] # [[],[],[]]
            self.masks = [[] for i in range(self.num_frames)] if self.AllCheckBox.isChecked() else [[]] 
            self.initialize_model()
            # logger.info('using model %s'%self.current_model)
            self.progress.setValue(20)
            QApplication.processEvents() 
            do_3D = False
            if self.NZ > 1:
                do_3D = True
                data = self.stack.copy()
            else:
                data = []
                if self.num_frames == 1: # single image
                    data.append(self.stack[0].copy())
                elif not self.AllCheckBox.isChecked(): # single frame 
                    new_frame = self.frames_dir + "/frame" + str(self.curr_idx_frame - 1) + ".png"
                    image = cv2.imread(new_frame, -1)
                    image = image[np.newaxis,...]

                    data.append(image[0].copy())
                else: # multiple frames
                    for idx in range(self.num_frames):
                        new_frame = self.frames_dir + "/frame" + str(idx) + ".png"
                        image = cv2.imread(new_frame, -1)
                        image = image[np.newaxis,...]
                        
                        data.append(image[0].copy()) # maybe chanchoose here 
            channels = self.get_channels()
            self.diameter = float(self.Diameter.text())
            
            # print('heredebug',self.stack.shape,data.shape, channels)
            
            ### will either have to put in edge cases for worm etc or just generalize model loading to respect what is there 
            try:
                omni_model = 'omni' in self.current_model
                bacterial = 'bact' in self.current_model
                # if omni_model or bacterial:
                #     self.NetAvg.setCurrentIndex(1) #one run net
                if bacterial:
                    self.diameter = 0.
                    self.Diameter.setText('%0.1f'%self.diameter)

                # allow omni to be togged manually or forced by model
                if OMNI_INSTALLED:
                    if omni_model:
                        print('GUI_INFO: turning on Omnipose mask recontruction version for Omnipose models (see menu)')
                        if not self.omni.isChecked():
                            print('WARNING: Omnipose models require Omnipose mask recontruction (toggle back on in menu)')
                        if not self.cluster.isChecked():
                            print(('NOTE: clutering algorithm can help with over-segmentation in thin cells.'
                                   'Default is ON with omnipose models (see menu)'))
                            
                    elif self.omni.isChecked():
                        print('NOTE: using Omnipose mask recontruction with built-in cellpose model (toggle in Omnipose menu)')

                net_avg = False # self.NetAvg.currentIndex()==0 and self.current_model in models.MODEL_NAMES
                resample = True # self.NetAvg.currentIndex()<2
                omni = OMNI_INSTALLED and self.omni.isChecked()
                
                self.threshold, self.cellprob = (0.0, 0.0) # self.get_thresholds()

                # useful printout for easily copying parameters to a notebook etc. 
                s = ('channels={}, mask_threshold={}, '
                     'flow_threshold={}, diameter={}, invert={}, cluster={}, net_avg={}, '
                     'do_3D={}, omni={}'
                    ).format(self.get_channels(),
                             self.cellprob,
                             self.threshold,
                             self.diameter,
                             False, # self.invert.isChecked(),
                             True, # self.cluster.isChecked(),
                             net_avg,
                             do_3D,
                             omni)
                # self.runstring.setPlainText(s)
                self.progress.setValue(30)
                print
                masks, flows = self.model.eval(data, channels=channels,
                                               mask_threshold=self.cellprob,
                                               flow_threshold=self.threshold,
                                               diameter=self.diameter, 
                                               invert=False, # self.invert.isChecked(),
                                               net_avg=net_avg, 
                                               augment=False, 
                                               resample=resample,
                                               do_3D=do_3D, 
                                               progress=self.progress,
                                               verbose=False, # self.verbose.isChecked(),
                                               omni=omni, 
                                               affinity_seg=False, # self.affinity.isChecked(),
                                               cluster = True, # self.cluster.isChecked(),
                                               transparency=True,
                                               channel_axis=-1
                                               )[:2]
                
            except Exception as e:
                print('GUI.py: NET ERROR: %s'%e)
                self.progress.setValue(0)
                return

            self.progress.setValue(75)
            QApplication.processEvents() 

            #if not do_3D:
            #    masks = masks[0][np.newaxis,:,:]
            #    flows = flows[0]
            
            # flows here are [RGB, dP, cellprob, p, bd, tr]
            for flows_idx in range(len(self.flows)):
                self.flows[flows_idx][0] = utils.to_8_bit(flows[flows_idx][0]) #RGB flow for plotting
                self.flows[flows_idx][1] = utils.to_8_bit(flows[flows_idx][2]) #dist/prob for plotting

            if False: #self.boundary.isChecked():
                for flows_idx in range(len(self.flows)):
                    self.flows[flows_idx][2] = utils.to_8_bit(flows[flows_idx][4]) #boundary for plotting
            else:
                for flows_idx in range(len(self.flows)):
                    self.flows[flows_idx][2] = np.zeros_like(self.flows[flows_idx][1])

            if not do_3D:
                for flows_idx in range(len(self.flows)):
                    self.masks[flows_idx] = masks[flows_idx][np.newaxis,...]
                    for i in range(3):
                        self.flows[flows_idx][i] = resize_image(self.flows[flows_idx][i], self.masks[flows_idx].shape[-2], self.masks[flows_idx].shape[-1])
               
                #critical line from what I had commended out below
                for flows_idx in range(len(self.flows)):
                    self.flows[flows_idx] = [self.flows[flows_idx][n][np.newaxis,...] for n in range(len(self.flows[flows_idx]))]
            
            # I think this is a z-component placeholder. Relaceing with boundary output, will
            # put this back later for the 3D update 
            # if not do_3D:
            #     self.flows[2] = np.zeros(masks.shape[1:], dtype=np.uint8)
            #     self.flows = [self.flows[n][np.newaxis,...] for n in range(len(self.flows))]
            # else:
            #     self.flows[2] = (flows[1][0]/10 * 127 + 127).astype(np.uint8)
                

            # this stores the original flow components for recomputing masks
            for flows_idx in range(len(self.flows)):
                if len(flows[flows_idx])>2: 
                    self.flows[flows_idx].append(flows[flows_idx][3].squeeze()) #p put in position -2
                    flws = [flows[flows_idx][1], #self.flows[-1][:self.dim] is dP
                            flows[flows_idx][2][np.newaxis,...]] #self.flows[-1][self.dim] is dist/prob
                    if False: # self.boundary.isChecked():
                        flws.append(flows[flows_idx][4][np.newaxis,...]) #self.flows[-1][self.dim+1] is bd
                    else:
                        flws.append(np.zeros_like(flws[-1]))
                    
                    self.flows[flows_idx].append(np.concatenate(flws))

            # logger.info('%d cells found with model in %0.3f sec'%(len(np.unique(masks)[1:]), time.time()-tic))
            self.progress.setValue(80)
            QApplication.processEvents() 
            z=0
            self.masksOn = True
            # self.MCheckBox.setChecked(True)
            # self.outlinesOn = True #again, this option should persist and not get toggled by another GUI action 
            # self.OCheckBox.setChecked(True)

            iio._masks_to_gui(self, self.masks[self.curr_idx_frame - 1 if len(self.masks) > 1 else 0], outlines=None)
            self.progress.setValue(100)

            # self.toggle_server(off=True)
            # if not do_3D:
            #     self.threshslider.setEnabled(True)
            #     self.probslider.setEnabled(True)
        except Exception as e:
            print('ERROR: %s'%e)

    def get_model_path(self):
        self.current_model = self.ModelChoose.currentText()
        if self.current_model in models.MODEL_NAMES:
            self.current_model_path = models.model_path(self.current_model, 0, self.torch)
        else:
            self.current_model_path = os.fspath(models.MODEL_DIR.joinpath(self.current_model))

app = QApplication(sys.argv)

screen = app.primaryScreen()
dpi = screen.logicalDotsPerInch()
pxr = screen.devicePixelRatio()
size = screen.availableGeometry()
clipboard = app.clipboard()

demo = AppDemo(size, dpi, pxr, clipboard)
demo.show()
sys.exit(app.exec())


