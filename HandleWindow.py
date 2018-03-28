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

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setupBtnClick()
    
    def setupBtnClick(self):
        self.btn_inputImage.clicked.connect(self.ImportImageAndShowPath)
        self.btn_styles_convert.clicked.connect(self.ConvertToStyle)
        self.btn_inputStyle.clicked.connect(self.ImportStyleImage)
    
    def ImportImageAndShowPath(self):
        fname = QFileDialog.getOpenFileName(self, 'ImportImage', 'c:\\', 'Image files(*.png *.jpg *.jpeg)')
        self.edit_input.setText(str(fname[0]))
    
        img = cv2.imread(str(fname[0]), cv2.IMREAD_COLOR)
        style_tool.check_image(img, str(fname[0]))
        
        self.lab_originalImg.setStyleSheet("")
        self.lab_originalImg.setPixmap(self.Conv_CV2QPixmap(img))
        
        style_tool.content_img_dir = os.path.split(str(fname[0]))[0]
        style_tool.content_img_name = os.path.split(str(fname[0]))[1]
    
    def ImportStyleImage(self):
        fname = QFileDialog.getOpenFileNames(self, 'ImportImage', 'c:\\', 'Image files(*.png *.jpg *.jpeg)')
        self.edit_styleInput.setText(str(fname[0]))
        
        style_tool.style_imgs_name = []
        
        path_name = os.path.split(str(fname[0][0]))[0]
        style_tool.style_imgs_dir = path_name
        
        for i in fname[0]:
            style_tool.style_imgs_name.append(os.path.split(str(i))[1])
            
                
            
    def ConvertToStyle(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Save Directory", "C:\\")
        style_tool.img_output_dir = str(dir_path)
        style_tool.img_name = 'styles'
        
        if style_tool.checkPath():
            style_tool.handleParameter()
            style_tool.render_single_image()
    
    def Conv_CV2QPixmap(self, img):
        h, w, d = img.shape
        bytesPerLine = 3 * w
        qImg = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        qPix = QPixmap.fromImage(qImg)
        
        lw = self.lab_originalImg.width()
        lh = self.lab_originalImg.height()
        return qPix.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
    def SplitPathAndName(self, path):
        pass
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
    
    