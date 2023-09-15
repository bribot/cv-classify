import typing
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPalette, QColor, QImage, QPixmap
from PyQt5.QtWidgets import QWidget,QLabel
import cv2
import numpy as np


class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)
        


