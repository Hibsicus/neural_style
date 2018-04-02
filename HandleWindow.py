# -*- coding: utf-8 -*-
from pyqtmainwindow import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import style_tool
import sys
import cv2
import numpy as np
import os
import threading
import time

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        
        self.setupBtnClick()
        self.setupCombo()
        self.setupCheck()
        self.setupValueEdit()
        self.setupLineText()
        
    def setupLineText(self):
        self.outname_line.textChanged.connect(self.HandleOutputName)
        
    def setupValueEdit(self):
        self.iteration_spin.valueChanged.connect(self.HandleIteration)
        self.learingrate_doubleSpin.valueChanged.connect(self.HandleLearningRate)
        self.maxSize_spin.valueChanged.connect(self.HandleMaxSize)
        
    def setupBtnClick(self):
        self.btn_inputImage.clicked.connect(self.ImportImageAndShowPath)
        self.btn_styles_convert.clicked.connect(self.ConvertToStyle)
        self.btn_inputStyle.clicked.connect(self.ImportStyleImage)
    
    def setupCombo(self):
        self.Device_Combo.currentIndexChanged.connect(self.HandleDevice)
        self.InitImageType_Combo.currentIndexChanged.connect(self.HandleImageType)
        self.ColorConvertType_Combo.currentIndexChanged.connect(self.HandleColorConvert)
        self.content_loss_func_combo.currentIndexChanged.connect(self.HandleLossFunction)
        self.pool_args_combo.currentIndexChanged.connect(self.HandlePoolArgs)
        self.optimizer_combo.currentIndexChanged.connect(self.HandleOpt)
        
    def setupCheck(self):
        self.stylemask_check.stateChanged.connect(self.HandleStyleMask)
        self.originalColor_check.stateChanged.connect(self.HandleOriginalColor)
    
    def HandleOutputName(self, text):
        style_tool.img_name = text
    
    def HandleMaxSize(self):
        style_tool.max_size = self.maxSize_spin.value()
    
    def HandleLearningRate(self):
        style_tool.learning_rate = self.learingrate_doubleSpin.value()
    
    def HandleIteration(self):
        style_tool.max_iterations = self.iteration_spin.value()
    
    def HandleStyleMask(self, state):
        if state == Qt.Checked:
            style_tool.style_mask = True
        else:
            style_tool.style_mask = False
    
    def HandleOriginalColor(self, state):
        if state == Qt.Checked:
            style_tool.original_colors = True
        else:
            style_tool.original_colors = False
        
    def HandleOpt(self, i):
        style_tool.optimizer = self.optimizer_combo.currentText()
        
    def HandlePoolArgs(self, i):
        style_tool.pool_args = self.pool_args_combo.currentText()
        
    def HandleLossFunction(self, i):
        style_tool.content_loss_function = i + 1
        
    
    def HandleColorConvert(self, i):
        style_tool.color_convert_type = self.ColorConvertType_Combo.currentText()
        
    def HandleImageType(self, i):
        style_tool.init_img_type = self.InitImageType_Combo.currentText()
        
    def HandleDevice(self, i):
        if self.Device_Combo.currentIndex == 0:
            style_tool.device = '/cpu:0'
        else:
            style_tool.device = '/gpu:0'
    
    def ImportImageAndShowPath(self):
        fname = QFileDialog.getOpenFileName(self, 'ImportImage', 'c:\\', 'Image files(*.png *.jpg *.jpeg)')
        self.edit_input.setText(str(fname[0]))
    
        img = cv2.imread(str(fname[0]), cv2.IMREAD_COLOR)
        style_tool.check_image(img, str(fname[0]))
        
        self.lab_originalImg.setStyleSheet("")
        self.lab_originalImg.setPixmap(self.Conv_CV2QPixmap(img, self.lab_originalImg.width(), self.lab_originalImg.height()))
        
        style_tool.content_img_dir = os.path.split(str(fname[0]))[0]
        style_tool.content_img_name = os.path.split(str(fname[0]))[1]
    
    def ImportStyleImage(self):
        fname = QFileDialog.getOpenFileNames(self, 'ImportImage', 'c:\\', 'Image files(*.png *.jpg *.jpeg)')
        
        if len(fname[0]) > 0:
            self.line_styleImgWeights.setEnabled(True)
        else:
            self.line_styleImgWeights.setEnabled(False)
        
        if len(fname[0]) > 1:
            self.stylemask_check.setEnabled(True)
        else:
            self.stylemask_check.setEnabled(False)
        
        self.edit_styleInput.setText(str(fname[0]))
        
        style_tool.style_imgs_name = []
        
        self.lab_styleImg_multi.setPixmap(self.ShowMultiPixmap(fname[0]))
        
        path_name = os.path.split(str(fname[0][0]))[0]
        style_tool.style_imgs_dir = path_name
        
        for i in fname[0]:
            style_tool.style_imgs_name.append(os.path.split(str(i))[1])
            
            
    def ConvertToStyle(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Save Directory", "C:\\")
        style_tool.img_output_dir = str(dir_path)
        style_tool.img_name = 'styles'
        
        if style_tool.checkPath():
            self.lab_style_img.setStyleSheet("")
            style_tool.handleParameter()
            threading.Thread(target=self.isNeuralStyleSuccess).start()
            style_tool.render_single_image()
    
    def Conv_CV2QPixmap(self, img, width, height):
        h, w, d = img.shape
        bytesPerLine = 3 * w
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        qImg = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        qPix = QPixmap.fromImage(qImg)
        
        return qPix.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
    def ShowMultiPixmap(self, paths):
        pix = QPixmap(512, 128)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.begin(self)
        
        total = len(paths)
        print('lengths of %i' % (total))
        if total <= 4:
            for i in range(len(paths)):
                print(i)
                img = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)
                painter.drawPixmap((128 * i), 0, 128, 128, self.Conv_CV2QPixmap(img , 128,128 ))
        else:
            for i in range(len(paths)):
                img = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)
                painter.drawPixmap(i * (512 / (len(paths))), 0, 512 / (len(paths)), 128, self.Conv_CV2QPixmap(img ,512 / (len(paths)),128 ))
        
        painter.end()
        
        return pix
    
    
    def isNeuralStyleSuccess(self):
        while style_tool.sucess_style_path == '':
            time.sleep(1)
            continue
        print('Success!!')
        fname = style_tool.sucess_style_path
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        self.lab_style_img.setPixmap(self.Conv_CV2QPixmap(img, self.lab_style_img.width(), self.lab_style_img.height()))
        
            
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
    
    