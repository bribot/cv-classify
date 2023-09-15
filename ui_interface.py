# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interfaceyaLjwB.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1262, 882)
        MainWindow.setMaximumSize(QSize(16777215, 16777215))
        MainWindow.setStyleSheet(u"*{\n"
"	border: none;\n"
"	background-color: transparent;\n"
"	background: none;\n"
"	padding: 0;\n"
"	margin: 0;\n"
"	color: #fff;\n"
"}\n"
"\n"
"#centralwidget {\n"
"	background-color: #1f232a;\n"
"}\n"
"\n"
"#leftMenuSubcontainer{\n"
"	background-color: #16191d;\n"
"}\n"
"\n"
"QPushButton {\n"
"	text-align: left;\n"
"	Font-size: 10\n"
"	padding 2px 5px;\n"
"	background-color: #161910;\n"
"	color: #fff;\n"
"	border: 1px solid #fff;\n"
"}\n"
"QPushButton:hover,\n"
"QPushButton:focus {\n"
"  background-color: rgb(100, 113, 145)\n"
"}")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.leftMenuContainer = QFrame(self.centralwidget)
        self.leftMenuContainer.setObjectName(u"leftMenuContainer")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.leftMenuContainer.sizePolicy().hasHeightForWidth())
        self.leftMenuContainer.setSizePolicy(sizePolicy1)
        self.leftMenuContainer.setMinimumSize(QSize(0, 0))
        self.leftMenuContainer.setFrameShape(QFrame.StyledPanel)
        self.leftMenuContainer.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.leftMenuContainer)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.leftMenuSubcontainer = QFrame(self.leftMenuContainer)
        self.leftMenuSubcontainer.setObjectName(u"leftMenuSubcontainer")
        self.leftMenuSubcontainer.setMinimumSize(QSize(150, 0))
        self.leftMenuSubcontainer.setFrameShape(QFrame.StyledPanel)
        self.leftMenuSubcontainer.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.leftMenuSubcontainer)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.frame_4 = QFrame(self.leftMenuSubcontainer)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.frame_4)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_2 = QLabel(self.frame_4)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setPixmap(QPixmap(u":/Logo/Logo_small.png"))
        self.label_2.setScaledContents(False)
        self.label_2.setAlignment(Qt.AlignCenter)
        self.label_2.setWordWrap(False)

        self.verticalLayout_2.addWidget(self.label_2)

        self.label = QLabel(self.frame_4)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setScaledContents(False)
        self.label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label, 0, Qt.AlignTop)


        self.verticalLayout.addWidget(self.frame_4, 0, Qt.AlignTop)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.verticalLayout.addItem(self.verticalSpacer_2)

        self.frame_5 = QFrame(self.leftMenuSubcontainer)
        self.frame_5.setObjectName(u"frame_5")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy2)
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame_5)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.btnVideo = QPushButton(self.frame_5)
        self.btnVideo.setObjectName(u"btnVideo")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.btnVideo.sizePolicy().hasHeightForWidth())
        self.btnVideo.setSizePolicy(sizePolicy3)
        icon = QIcon()
        icon.addFile(u":/icons/Icons/video.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btnVideo.setIcon(icon)
        self.btnVideo.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.btnVideo)

        self.btnCaptureOK = QPushButton(self.frame_5)
        self.btnCaptureOK.setObjectName(u"btnCaptureOK")
        sizePolicy3.setHeightForWidth(self.btnCaptureOK.sizePolicy().hasHeightForWidth())
        self.btnCaptureOK.setSizePolicy(sizePolicy3)
        icon1 = QIcon()
        icon1.addFile(u":/icons/Icons/aperture.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btnCaptureOK.setIcon(icon1)
        self.btnCaptureOK.setIconSize(QSize(24, 24))
        self.btnCaptureOK.setFlat(False)

        self.verticalLayout_3.addWidget(self.btnCaptureOK)

        self.btnCaptureNOK = QPushButton(self.frame_5)
        self.btnCaptureNOK.setObjectName(u"btnCaptureNOK")
        sizePolicy3.setHeightForWidth(self.btnCaptureNOK.sizePolicy().hasHeightForWidth())
        self.btnCaptureNOK.setSizePolicy(sizePolicy3)
        self.btnCaptureNOK.setIcon(icon1)
        self.btnCaptureNOK.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.btnCaptureNOK)

        self.btnTrain = QPushButton(self.frame_5)
        self.btnTrain.setObjectName(u"btnTrain")
        sizePolicy3.setHeightForWidth(self.btnTrain.sizePolicy().hasHeightForWidth())
        self.btnTrain.setSizePolicy(sizePolicy3)
        icon2 = QIcon()
        icon2.addFile(u":/icons/Icons/radio.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btnTrain.setIcon(icon2)
        self.btnTrain.setIconSize(QSize(24, 24))
        self.btnTrain.setFlat(False)

        self.verticalLayout_3.addWidget(self.btnTrain)

        self.btnSave = QPushButton(self.frame_5)
        self.btnSave.setObjectName(u"btnSave")
        sizePolicy3.setHeightForWidth(self.btnSave.sizePolicy().hasHeightForWidth())
        self.btnSave.setSizePolicy(sizePolicy3)
        icon3 = QIcon()
        icon3.addFile(u":/icons/Icons/cpu.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btnSave.setIcon(icon3)
        self.btnSave.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.btnSave)


        self.verticalLayout.addWidget(self.frame_5)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.frame_6 = QFrame(self.leftMenuSubcontainer)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_6)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.btnConfig = QPushButton(self.frame_6)
        self.btnConfig.setObjectName(u"btnConfig")
        icon4 = QIcon()
        icon4.addFile(u":/icons/Icons/tool.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btnConfig.setIcon(icon4)
        self.btnConfig.setIconSize(QSize(24, 24))

        self.verticalLayout_4.addWidget(self.btnConfig)

        self.btnExit = QPushButton(self.frame_6)
        self.btnExit.setObjectName(u"btnExit")
        icon5 = QIcon()
        icon5.addFile(u":/icons/Icons/x-square.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btnExit.setIcon(icon5)
        self.btnExit.setIconSize(QSize(24, 24))

        self.verticalLayout_4.addWidget(self.btnExit)


        self.verticalLayout.addWidget(self.frame_6, 0, Qt.AlignBottom)


        self.horizontalLayout_2.addWidget(self.leftMenuSubcontainer, 0, Qt.AlignLeft)


        self.horizontalLayout.addWidget(self.leftMenuContainer, 0, Qt.AlignLeft)

        self.mainBodyContainer = QFrame(self.centralwidget)
        self.mainBodyContainer.setObjectName(u"mainBodyContainer")
        sizePolicy4 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.mainBodyContainer.sizePolicy().hasHeightForWidth())
        self.mainBodyContainer.setSizePolicy(sizePolicy4)
        self.mainBodyContainer.setMinimumSize(QSize(0, 0))
        self.mainBodyContainer.setStyleSheet(u"")
        self.mainBodyContainer.setFrameShape(QFrame.StyledPanel)
        self.mainBodyContainer.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.mainBodyContainer)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.frame = QFrame(self.mainBodyContainer)
        self.frame.setObjectName(u"frame")
        sizePolicy5 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy5)
        self.frame.setMaximumSize(QSize(16777215, 16777215))
        self.frame.setStyleSheet(u"")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.frame)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.label_3 = QLabel(self.frame)
        self.label_3.setObjectName(u"label_3")
        font1 = QFont()
        font1.setFamily(u"OCR A Extended")
        font1.setPointSize(26)
        self.label_3.setFont(font1)

        self.verticalLayout_8.addWidget(self.label_3)


        self.verticalLayout_5.addWidget(self.frame, 0, Qt.AlignTop)

        self.cam0Container = QFrame(self.mainBodyContainer)
        self.cam0Container.setObjectName(u"cam0Container")
        sizePolicy2.setHeightForWidth(self.cam0Container.sizePolicy().hasHeightForWidth())
        self.cam0Container.setSizePolicy(sizePolicy2)
        self.cam0Container.setFrameShape(QFrame.StyledPanel)
        self.cam0Container.setFrameShadow(QFrame.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.cam0Container)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.img0 = QLabel(self.cam0Container)
        self.img0.setObjectName(u"img0")
        self.img0.setMaximumSize(QSize(16777215, 300))

        self.verticalLayout_7.addWidget(self.img0)


        self.verticalLayout_5.addWidget(self.cam0Container)


        self.horizontalLayout.addWidget(self.mainBodyContainer)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_2.setText("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"Huginn", None))
        self.btnVideo.setText(QCoreApplication.translate("MainWindow", u"Cam Stream", None))
        self.btnCaptureOK.setText(QCoreApplication.translate("MainWindow", u"Capturar OK", None))
        self.btnCaptureNOK.setText(QCoreApplication.translate("MainWindow", u"Capturar NG", None))
        self.btnTrain.setText(QCoreApplication.translate("MainWindow", u"Train", None))
        self.btnSave.setText(QCoreApplication.translate("MainWindow", u"Evaluar", None))
        self.btnConfig.setText(QCoreApplication.translate("MainWindow", u"Configuracion", None))
        self.btnExit.setText(QCoreApplication.translate("MainWindow", u"Salir", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"OK", None))
        self.img0.setText("")
    # retranslateUi

