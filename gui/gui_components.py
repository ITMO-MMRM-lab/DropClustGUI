import sys, os, gc
import cv2
import tifffile
import numpy as np
import PIL
from PIL import Image, ImageEnhance, ImageQt
import matplotlib.pyplot as plt
from skimage.io import imshow, show
import pyqtgraph as pg
from pyqtgraph import Point
from qtpy import QtCore
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QToolButton, QFileDialog, QPushButton, QSpacerItem, QSizePolicy, QGraphicsWidget
from superqt import QRangeSlider, QLabeledDoubleRangeSlider, QLabeledRangeSlider
from scipy.ndimage import find_objects
from methods import get_gray_img, pil_to_qpixmap
from iio import _initialize_images, _masks_to_gui

import torch
from torch import nn
from torch.utils import mkldnn as mkldnn_utils

TORCH_ENABLED = True

def imread(filename):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='.tiff':
        img = tifffile.imread(filename)
        return img
    else:
        try:
            img = cv2.imread(filename, -1)#cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2,1,0]]
            return img
        except Exception as e:
            # io_logger.critical('ERROR: could not read file, %s'%e)
            return None

def addFilter(parent):
    parent.filters_boundaries_values.append(
        ((parent.sliders[0].value()), (parent.sliders[1].value()), (parent.sliders[2].value()))
        )
    parent.filters_boundaries_labels.append("Filter#" + str(len(parent.filters_boundaries_values) - 2))

    parent.filter_dropdown.clear()
    parent.filter_dropdown.addItems(parent.filters_boundaries_labels)
    parent.filter_dropdown.setCurrentIndex(1)

def getMask(parent):
    cp_gs_img = parent.current_gs_img.copy()
    curr_boundaries = parent.filters_boundaries_values[parent.filter_dropdown.currentIndex()]

    new_mask = (((parent.current_img[:, :, 0] >= curr_boundaries[0][0]) & (parent.current_img[:, :, 0] <= curr_boundaries[0][1])) &
                ((parent.current_img[:, :, 1] >= curr_boundaries[1][0]) & (parent.current_img[:, :, 1] <= curr_boundaries[1][1])) &
                ((parent.current_img[:, :, 2] >= curr_boundaries[2][0]) & (parent.current_img[:, :, 2] <= curr_boundaries[2][1])))

    return new_mask

def countDroplets(parent):
    tmp_mask = getMask(parent).astype(int)
    print("SHAPE OF MASK: ", tmp_mask.shape)
    print("MAX: ", np.amax(tmp_mask))
    
    tmp_gs = np.full_like(tmp_mask, 1)

    # tmp_gs[tmp_mask] = 0
    # slices = find_objects(tmp_gs.astype(int))

    plt.imshow(tmp_mask, cmap=plt.cm.gray)
    plt.show()

    # print("Amount: ", len(slices))
    # print("Current filter: ", parent.filter_dropdown.currentIndex())

def disk(med, r, Ly, Lx):
    """ returns pixels of disk with radius r and center med """
    yy, xx = np.meshgrid(np.arange(0,Ly,1,int), np.arange(0,Lx,1,int),
                         indexing='ij')
    inds = ((yy-med[0])**2 + (xx-med[1])**2)**0.5 <= r
    y = yy[inds].flatten()
    x = xx[inds].flatten()
    return y,x

class ModelButton(QPushButton):
    def __init__(self, parent, model_name, text):
        super().__init__()
        self.setEnabled(False)
        # self.setStyleSheet(parent.styleInactive)
        self.setText(text)
        # self.setFont(parent.smallfont)
        self.clicked.connect(lambda: self.press(parent))
        self.model_name = model_name
        
    def press(self, parent):
        # for i in range(len(parent.StyleButtons)):
        #     parent.StyleButtons[i].setStyleSheet(parent.styleUnpressed)
        # self.setStyleSheet(parent.stylePressed)
        parent.compute_model()

class ColorSlider(QLabeledDoubleRangeSlider):
    def __init__(self, parent, name, color):
        super().__init__(Qt.Orientation.Horizontal)
        self.setEnabled(True)
        self.setDecimals(0)

        self.setHandleLabelPosition(QLabeledRangeSlider.LabelPosition.NoLabel)
        self.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.LabelIsValue)

        self.valueChanged.connect(lambda: self.changeRGB(parent))
        self.name = name

        self.setStyleSheet(""" QSlider{
                             background-color: transparent;
                             }""")
        self.show()

    def changeRGB(self, parent):
        boundaries = []
        if isinstance(parent.current_img, np.ndarray):
            for idx, color_slider in enumerate(parent.sliders):
                boundary = color_slider.value()
                boundaries.append((int(boundary[0]), int(boundary[1])))

            cp_curr_gs_img = parent.current_gs_img.copy()
            single_mask = (((parent.current_img[:, :, 0] >= boundaries[0][0]) & (parent.current_img[:, :, 0] <= boundaries[0][1])) &
                            ((parent.current_img[:, :, 1] >= boundaries[1][0]) & (parent.current_img[:, :, 1] <= boundaries[1][1])) &
                            ((parent.current_img[:, :, 2] >= boundaries[2][0]) & (parent.current_img[:, :, 2] <= boundaries[2][1])))

            cp_curr_gs_img[single_mask] = parent.current_img[single_mask]
            parent.image_viewer.setPixmap(
                            pil_to_qpixmap(
                                Image.fromarray(
                                    cp_curr_gs_img.astype('uint8')
                                    )))

class ViewBoxNoRightDrag(pg.ViewBox):
    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True, invertY=False, enableMenu=True, name=None, invertX=False):
        pg.ViewBox.__init__(self, None, border, lockAspect, enableMouse,
                            invertY, enableMenu, name, invertX)
        self.parent = parent
        self.axHistoryPointer = -1

    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        The following events are implemented:
        +/= : moves forward in the zooming stack (if it exists)
        - : moves backward in the zooming stack (if it exists)

        """
        ev.accept()
        if ev.text() == '-':
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ['+', '=']:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()
    
    def mouseDragEvent(self, ev, axis=None):
        ## if axis is specified, event will only affect that axis.
        if self.parent is None or (self.parent is not None and not self.parent.in_stroke):
            ev.accept()  ## we accept all buttons

            pos = ev.pos()
            lastPos = ev.lastPos()
            dif = pos - lastPos
            dif = dif * -1

            ## Ignore axes if mouse is disabled
            mouseEnabled = np.array(self.state['mouseEnabled'], dtype=float)
            mask = mouseEnabled.copy()
            if axis is not None:
                mask[1-axis] = 0.0

            ## Scale or translate based on mouse button
            if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MiddleButton):
                if self.state['mouseMode'] == pg.ViewBox.RectMode:
                    if ev.isFinish():  ## This is the final move in the drag; change the view scale now
                        #print "finish"
                        self.rbScaleBox.hide()
                        ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                        ax = self.childGroup.mapRectFromParent(ax)
                        self.showAxRect(ax)
                        self.axHistoryPointer += 1
                        self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                    else:
                        ## update shape of scale box
                        self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                else:
                    tr = dif*mask
                    tr = self.mapToView(tr) - self.mapToView(Point(0,0))
                    x = tr.x() if mask[0] == 1 else None
                    y = tr.y() if mask[1] == 1 else None

                    self._resetTarget()
                    if x is not None or y is not None:
                        self.translateBy(x=x, y=y)
                    self.sigRangeChangedManually.emit(self.state['mouseEnabled'])

class HistLUT(pg.HistogramLUTItem):
    sigLookupTableChanged = pyqtSignal(object)
    sigLevelsChanged = pyqtSignal(object)
    sigLevelChangeFinished = pyqtSignal(object)

    def __init__(self, image=None, fillHistogram=True, levelMode='mono',
                 gradientPosition='right', orientation='vertical'):
        super().__init__(image=image,fillHistogram=fillHistogram,levelMode=levelMode,
                         gradientPosition=gradientPosition,orientation=orientation)
        
        # self.gradient = GradientEditorItem(orientation=self.gradientPosition)
        # self.gradient = GradEditor(orientation=self.gradientPosition) #overwrite with mine
        self.show_histogram = True
        
    def mousePressEvent(self, event):
        if self.show_histogram:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.show_histogram:
            super().mouseMoveEvent(event)

    def paint(self, p, *args):
        # paint the bounding edges of the region item and gradient item with lines
        # connecting them
        if self.levelMode != 'mono' or not self.region.isVisible():
            return

        pen = self.region.lines[0].pen

        mn, mx = self.getLevels()
        vbc = self.vb.viewRect().center()
        gradRect = self.gradient.mapRectToParent(self.gradient.gradRect.rect())
        if self.orientation == 'vertical':
            p1mn = self.vb.mapFromViewToItem(self, Point(vbc.x(), mn)) + Point(0, 5)
            p1mx = self.vb.mapFromViewToItem(self, Point(vbc.x(), mx)) - Point(0, 5)
            if self.gradientPosition == 'right':
                p2mn = gradRect.bottomLeft()
                p2mx = gradRect.topLeft()
            else:
                p2mn = gradRect.bottomRight()
                p2mx = gradRect.topRight()
        else:
            p1mn = self.vb.mapFromViewToItem(self, Point(mn, vbc.y())) - Point(5, 0)
            p1mx = self.vb.mapFromViewToItem(self, Point(mx, vbc.y())) + Point(5, 0)
            if self.gradientPosition == 'bottom':
                p2mn = gradRect.topLeft()
                p2mx = gradRect.topRight()
            else:
                p2mn = gradRect.bottomLeft()
                p2mx = gradRect.bottomRight()

        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for pen in [pen]: #get rid of dirst entry, shadow of some sort
            p.setPen(pen)

            # lines from the linear region item bounds to the gradient item bounds
            p.drawLine(p1mn, p2mn)
            p.drawLine(p1mx, p2mx)

            # lines bounding the edges of the gradient item
            if self.orientation == 'vertical':
                p.drawLine(gradRect.topLeft(), gradRect.topRight())
                p.drawLine(gradRect.bottomLeft(), gradRect.bottomRight())
            else:
                p.drawLine(gradRect.topLeft(), gradRect.bottomLeft())
                p.drawLine(gradRect.topRight(), gradRect.bottomRight())

class CustomSlider(QWidget):
    def __init__(self, minimum, maximum, parent):
        super(CustomSlider, self).__init__()   #.__init__(parent=parent)

        sliderLayout = QHBoxLayout()

        ### Left arrow
        leftArrowButton = QToolButton()
        leftArrowButton.setArrowType(Qt.ArrowType.LeftArrow)

        leftArrowButton.clicked.connect(lambda: self.switchFrame(self, parent, 'L'))

        sliderLayout.addWidget(leftArrowButton)
        ###

        ### Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        # slider.setGeometry(10, 480, 400, 20)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(lambda: self.updateFramesSlider(self, parent))
        self.slider.setToolTip(str(parent.curr_idx_frame))
        self.slider.setValue(parent.curr_idx_frame)

        # if parent is not None:
        #     print("WACHA AQUI OE")
        #     parent.frames_slider = slider
        sliderLayout.addWidget(self.slider)
        ###

        ### Right arrow
        rightArrowButton = QToolButton()
        rightArrowButton.setArrowType(Qt.ArrowType.RightArrow)

        rightArrowButton.clicked.connect(lambda: self.switchFrame(self, parent, 'R'))

        sliderLayout.addWidget(rightArrowButton)
        ###
        self.setLayout(sliderLayout)

    def switchFrame(e, self, parent, button_side):
        parent.curr_idx_frame = self.slider.value()
        self.slider.setToolTip(str(self.slider.value()))

        if button_side == 'L':
            self.slider.setValue(self.slider.value() - 1)
        else:
            self.slider.setValue(self.slider.value() + 1)

    def updateFramesSlider(e, self, parent):
        parent.curr_idx_frame = e.sender().value()
        self.slider.setToolTip(str(e.sender().value()))

        idx = e.sender().value() - 1

        img = cv2.imread(parent.frames_dir + "/frame"+ str(idx) + ".png", -1)   #cv2.LOAD_IMAGE_ANYDEPTH)
        if img.ndim > 2:
            img = img[..., [2,1,0]]

        _initialize_images(parent, img, resize=parent.resize, X2=0)
        if len(parent.masks) > 1 and parent.masksOn:
            _masks_to_gui(parent, parent.masks[parent.curr_idx_frame - 1], outlines=None)
        parent.update_plot()

class ImageDraw(pg.ImageItem):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    GraphicsObject displaying an image. Optimized for rapid update (ie video display).
    This item displays either a 2D numpy array (height, width) or
    a 3D array (height, width, RGBa). This array is optionally scaled (see
    :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
    with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
    before being displayed.
    ImageItem is frequently used in conjunction with
    :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
    :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
    for controlling the levels and lookup table used to display the image.
    """

    sigImageChanged = pyqtSignal()

    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        super(ImageDraw, self).__init__()
        #self.image=None
        #self.viewbox=viewbox
        self.levels = np.array([0,255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = 'row-major'
        self.removable = False

        self.parent = parent
        #kernel[1,1] = 1
        self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.parent.current_stroke = []
        self.parent.in_stroke = False

    def mouseClickEvent(self, ev):	
        if self.parent.masksOn or self.parent.outlinesOn:	
            if  self.parent.loaded and (ev.button() == QtCore.Qt.RightButton or 	
                    ev.modifiers() & QtCore.Qt.ShiftModifier and not ev.double()):	
                if not self.parent.in_stroke:	
                    ev.accept()	
                    self.create_start(ev.pos())	
                    self.parent.stroke_appended = False	
                    self.parent.in_stroke = True	
                    self.drawAt(ev.pos(), ev)	
                else:	
                    ev.accept()	
                    self.end_stroke()	
                    self.parent.in_stroke = False	
            elif not self.parent.in_stroke:	
                y,x = int(ev.pos().y()), int(ev.pos().x())	
                if y>=0 and y<self.parent.Ly and x>=0 and x<self.parent.Lx:	
                    if ev.button() == QtCore.Qt.LeftButton and not ev.double():	
                        idx = self.parent.cellpix[self.parent.currentZ][y,x]	
                        if idx > 0:	
                            if ev.modifiers() & QtCore.Qt.ControlModifier:	
                                # delete mask selected	
                                self.parent.remove_cell(idx)	
                            elif ev.modifiers() & QtCore.Qt.AltModifier:	
                                self.parent.merge_cells(idx)	
                            elif self.parent.masksOn:	
                                self.parent.unselect_cell()	
                                self.parent.select_cell(idx)	
                        elif self.parent.masksOn:	
                            self.parent.unselect_cell()

    def mouseDragEvent(self, ev):
        ev.ignore()
        return

    def hoverEvent(self, ev):
        #QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        if self.parent.in_stroke:
            if self.parent.in_stroke:
                # continue stroke if not at start
                self.drawAt(ev.pos())
                if self.is_at_start(ev.pos()):
                    self.end_stroke()
                    # self.parent.in_stroke = False
                    
        else:
            ev.acceptClicks(QtCore.Qt.RightButton)
            #ev.acceptClicks(QtCore.Qt.LeftButton)

    def create_start(self, pos):
        self.scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], pxMode=False,
                                        pen=pg.mkPen(color=(255,0,0), width=self.parent.brush_size),
                                        size=max(3*2, self.parent.brush_size*1.8*2), brush=None)
        self.parent.p0.addItem(self.scatter)

    def is_at_start(self, pos):
        thresh_out = max(6, self.parent.brush_size*3)
        thresh_in = max(3, self.parent.brush_size*1.8)
        # first check if you ever left the start
        if len(self.parent.current_stroke) > 3:
            stroke = np.array(self.parent.current_stroke)
            dist = (((stroke[1:,1:] - stroke[:1,1:][np.newaxis,:,:])**2).sum(axis=-1))**0.5
            dist = dist.flatten()
            #print(dist)
            has_left = (dist > thresh_out).nonzero()[0]
            if len(has_left) > 0:
                first_left = np.sort(has_left)[0]
                has_returned = (dist[max(4,first_left+1):] < thresh_in).sum()
                if has_returned > 0:
                    return True
                else:
                    return False
            else:
                return False

    def end_stroke(self):
        self.parent.p0.removeItem(self.scatter)
        if not self.parent.stroke_appended:
            self.parent.strokes.append(self.parent.current_stroke)
            self.parent.stroke_appended = True
            self.parent.current_stroke = np.array(self.parent.current_stroke)
            ioutline = self.parent.current_stroke[:,3]==1
            self.parent.current_point_set.extend(list(self.parent.current_stroke[ioutline]))
            self.parent.current_stroke = []
            if self.parent.autosave:
                self.parent.add_set()
        if len(self.parent.current_point_set) > 0 and self.parent.autosave:
            self.parent.add_set()
        self.parent.in_stroke = False
    

    def tabletEvent(self, ev):
        pass
        #print(ev.device())
        #print(ev.pointerType())
        #print(ev.pressure())

    def drawAt(self, pos, ev=None):
        mask = self.strokemask
        set = self.parent.current_point_set
        stroke = self.parent.current_stroke
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]
        kcent = kc.copy()
        if tx[0]<=0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx    = sx
            kcent[0] = 0
        if ty[0]<=0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty    = sy
            kcent[1] = 0
        if tx[1] >= self.parent.Ly-1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.Ly - kc[0] - 1
            tx[1] = self.parent.Ly
            kcent[0] = tx[1]-tx[0]-1
        if ty[1] >= self.parent.Lx-1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.Lx - kc[1] - 1
            ty[1] = self.parent.Lx
            kcent[1] = ty[1]-ty[0]-1


        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        self.image[ts] = mask[ss]

        for ky,y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx,x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx==kcent[0], ky==kcent[1])
                stroke.append([self.parent.currentZ, x, y, iscent])
        self.updateImage()

    def setDrawKernel(self, kernel_size=3):
        bs = kernel_size
        kernel = np.ones((bs,bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [int(np.floor(kernel.shape[0]/2)),
                                 int(np.floor(kernel.shape[1]/2))]
        onmask = 255 * kernel[:,:,np.newaxis]
        offmask = np.zeros((bs,bs,1))
        opamask = 100 * kernel[:,:,np.newaxis]
        self.redmask = np.concatenate((onmask,offmask,offmask,onmask), axis=-1)
        self.strokemask = np.concatenate((onmask,offmask,onmask,opamask), axis=-1)