from PySide6 import QtWidgets, QtGui, QtCore
from os import path, getcwd
import sys
from collections import OrderedDict
from CNN_Infer import Predict
from Utils import get_model_dir
import traceback

BTN_HEIGHT = 50
BTN_ICON_SIZE = QtCore.QSize(40, 40)
BTN_FONT_SIZE = 12

RESOURCE_DIR = get_model_dir()

class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()

        #Make sure the model file is accessible
        if not path.exists(path.join(RESOURCE_DIR, "model_state.pth")):

            QtWidgets.QMessageBox.critical(self, "Model File Error",
                                       "The required pretrained model file was not found in the resources folder!\nPlease make sure your "
                                       "system drive has enough space, some antivirus isn't blocking this program to create temporary files or restart the program.")

            sys.exit(1)

        self.lbl_header = QtWidgets.QLabel(self)
        default_info_text = ("This program is capable of classifying Lung X-Ray scans of adults as either having"
                             " Pneumonia or being Healthy with very high sensitivity and specificity.")

        # header_font = QtGui.QFont()
        # header_font.setPointSize(20)
        # header_font.setBold(True)
        self.lbl_header.setText("Pneumonia iD - v1.2")
        self.lbl_header.setStyleSheet("QLabel {color : #a0c4ff; font-size: 20px}")

        self.lbl_images = QtWidgets.QLabel(self)
        self.lbl_images.setText("Select one or more X-Ray scan images to add to the classify queue:")

        self.main_text = QtWidgets.QTextEdit(self)
        self.main_text.setMinimumWidth(750)
        self.main_text.setMinimumHeight(400)
        self.main_text.setText("<span style='color:gray;'>(No files added yet)</span>")
        self.btn_add = QtWidgets.QPushButton(self)
        self.btn_add.setText("Add Images")
        self.btn_add.clicked.connect(self.add_images)
        self.main_layout = QtWidgets.QGridLayout(self)
        self.main_layout.addWidget(self.lbl_header)
       # self.main_layout.setHorizontalSpacing(15)
        #self.main_layout.setVerticalSpacing(25)
        self.btn_classify = QtWidgets.QPushButton(self)
        self.btn_classify.setText("CLASSIFY")
        self.btn_classify.clicked.connect(self.classify)
        # self.btn_export = QtWidgets.QPushButton(self)
        # self.btn_export.setText("Export to CSV")
        # self.btn_export.clicked.connect(self.export)
        self.main_layout.addWidget(self.lbl_header, 0, 0, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.lbl_images, 2, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.main_text, 3, 0, 1, 2)
        self.main_layout.addWidget(self.btn_add, 2, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        self.main_layout.addWidget(self.btn_classify, 5, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        #self.main_layout.addWidget(self.btn_export, 5, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        self.main_text.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.main_text.setReadOnly(True)

        self.main_text_dict = OrderedDict()
        #Set the buttons disabled until needed by the user
        self.btn_classify.setDisabled(True)
        #elf.btn_export.setDisabled(True)

        self.setWindowTitle("Pneumonia iD")


    def signal_receive_status(self, status):
        self.update_main_text_dict(status[0], status[1])

    def update_main_text(self):
        new_text = ""
        for k, v in self.main_text_dict.items():
            if v is None:
                status = "<span style='color:#3498db'>Pending...</span>"
            elif v > 0.5:
                status = "<span style='color:#e74c3c'>Pneumonia</span> -- Confidence: {}".format(v)

            else:
                status = "<span style='color:#50c878'>Healthy</span> -- Confidence: {}".format(v)

            new_text += "[" + k + "] -- Status: {}<br>".format(status)

        self.main_text.setText(new_text)


    def update_main_text_dict(self, k, state):
        self.main_text_dict[k] = state
        self.update_main_text()

    def add_images(self):
        selected_files = QtWidgets.QFileDialog.getOpenFileNames(self,"Select one or more Image files", getcwd(),
                                                                    "Images (*.png *.webp *.jpg *.jpeg)")


        if len(selected_files) < 1:
            return

        for f in selected_files[0]:
            self.btn_classify.setDisabled(False)
            if f != "":
                self.update_main_text_dict(path.normpath(f), None)


    def export(self):
        pass


    def classify_error(self):
        QtWidgets.QMessageBox.critical(self, "Fatal Error",
                                       "There was an unexpected error when classifying your images\nPlease check the image formats and try again, or check the log file")

        self.btn_classify.setDisabled(True)


    def classify_done(self, status):
        if status == "done":
            self.btn_classify.setDisabled(False)
            self.btn_classify.setText("Classify")
            #self.btn_export.setDisabled(False)

    def classify(self):

        self.btn_classify.setDisabled(True)
        self.btn_classify.setText("Processing...")

        self.classifier = Predict([k for k, v in self.main_text_dict.items() if v is None])
        self.classifier.signal_status.connect(self.signal_receive_status)
        self.classifier.signal_done.connect(self.classify_done)
        self.classifier.start()
