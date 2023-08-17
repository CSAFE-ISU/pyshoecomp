# -*- coding: utf-8 -*-
"""
    Minimal PyQt5 GUI for entering options.
    Select source TIFF files, output target folder, and
    toggle advanced options if needed.
"""

import PyQt5.QtWidgets as qtgui
import PyQt5.QtCore as qtcore
import threading
import time

from runner import runner

class PercentageWorker(qtcore.QObject):
    # https://stackoverflow.com/questions/66265219
    started = qtcore.pyqtSignal()
    finished = qtcore.pyqtSignal()
    percentageChanged = qtcore.pyqtSignal(int)
    txtChanged = qtcore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._percentage = 0
        self._debug_text = "starting"

    @property
    def debug_text(self):
        return self._debug_text

    @debug_text.setter
    def debug_text(self, t):
        self._debug_text = t
        self.txtChanged.emit(self.debug_text)

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, value):
        if self._percentage == value:
            return
        self._percentage = value
        self.percentageChanged.emit(self.percentage)

    def start(self):
        self.started.emit()

    def finish(self):
        self.finished.emit()


class SelWindow(qtgui.QMainWindow):
    """
    PyQt Window Subclass with members for following options:

    * Browse for Test Impression TIFF, allow corner extraction, enter DPI if required
    * Browse for Crime Scene TIFF, allow corner extraction, enter DPI if required
    """

    def __init__(self):
        super(qtgui.QMainWindow, self).__init__()
        self.gripSize = 16
        self.grips = []
        for i in range(4):
            grip = qtgui.QSizeGrip(self)
            grip.resize(self.gripSize, self.gripSize)
            self.grips.append(grip)

        self.done = False
        # self.setFixedSize(1020, 620)

        self.central = qtgui.QWidget()

        self.go_button = qtgui.QPushButton("Go!", parent=self)
        self.go_button.clicked.connect(self.listener)
        self.progress = qtgui.QProgressBar()
        self.dbg = qtgui.QLabel("")

        self.file1 = qtgui.QLineEdit()
        self.file1.setDisabled(True)
        self.browse1 = qtgui.QPushButton("Select File")
        self.browse1.clicked.connect(self.listener)

        self.file2 = qtgui.QLineEdit()
        self.file2.setDisabled(True)
        self.browse2 = qtgui.QPushButton("Select File")
        self.browse2.clicked.connect(self.listener)

        self.targ_dir = qtgui.QLineEdit()
        self.targ_dir.setDisabled(True)
        self.browse3 = qtgui.QPushButton("Browse")
        self.browse3.clicked.connect(self.listener)

        self.clique_heur = qtgui.QCheckBox("Use Heuristic for Alignment (faster)")
        self.clique_heur.setChecked(False)
        self.clique_eps = qtgui.QLineEdit("0.5")
        self.clique_alpha = qtgui.QLineEdit("5.0")
        self.score_options = qtgui.QComboBox()
        self.score_options.addItems(["clique_fraction", "ImageNCC"])

        self.layout = qtgui.QGridLayout()

        self.layout.addWidget(qtgui.QLabel("<b> Test Impression: </b> "), 1, 0)
        self.layout.addWidget(self.file1, 1, 1)
        self.layout.addWidget(self.browse1, 1, 2)

        self.layout.addWidget(qtgui.QLabel("<b> Crime Scene: </b> "), 4, 0)
        self.layout.addWidget(self.file2, 4, 1)
        self.layout.addWidget(self.browse2, 4, 2)

        self.layout.addWidget(qtgui.QLabel("<b> Target Folder: </b>"), 8, 0)
        self.layout.addWidget(self.targ_dir, 8, 1)
        self.layout.addWidget(self.browse3, 8, 2)

        self.layout.addWidget(qtgui.QLabel("<b> Advanced Options: </b>"), 9, 0)
        self.layout.addWidget(self.clique_heur, 9, 1)
        self.layout.addWidget(
            qtgui.QLabel("Flexibility for Alignment <i>(lower -> more strict)</i>"),
            10,
            1,
        )
        self.layout.addWidget(self.clique_eps, 10, 2)
        self.layout.addWidget(
            qtgui.QLabel("Sparsity for Alignment <i>(higher -> more strict)</i>"),
            11,
            1,
        )
        self.layout.addWidget(self.clique_alpha, 11, 2)
        self.layout.addWidget(qtgui.QLabel("Similarity Score: "), 12, 1)
        self.layout.addWidget(self.score_options, 12, 2)
        self.layout.addWidget(self.go_button, 13, 0, 1, 4)
        self.layout.addWidget(qtgui.QLabel("Progress: "), 14, 0)
        self.layout.addWidget(self.progress, 14, 1)
        self.layout.addWidget(self.dbg, 14, 2, 1, 2)

        self.central.setLayout(self.layout)
        self.setCentralWidget(self.central)

    def resizeEvent(self, event):
        # https://stackoverflow.com/questions/62807295
        qtgui.QMainWindow.resizeEvent(self, event)
        rect = self.rect()
        self.grips[1].move(rect.right() - self.gripSize, 0)
        self.grips[2].move(rect.right() - self.gripSize, rect.bottom() - self.gripSize)
        self.grips[3].move(0, rect.bottom() - self.gripSize)

    def listener(self):
        sender = self.sender()
        if sender == self.browse1:
            self.set_file(1)
        elif sender == self.browse2:
            self.set_file(2)
        elif sender == self.browse3:
            self.set_file(3)
        elif sender == self.go_button:
            self.done_button()

    def set_file(self, file_no):
        """To open the appropriate file/directory selection

        :param file_no: to indicate file/folder that is being selected

        """
        if file_no == 1:
            fl = self.file1
            fl_text = qtgui.QFileDialog.getOpenFileName(
                parent=self, caption="Select Shoeprint TIFF file", filter="*.tiff"
            )
        elif file_no == 2:
            fl = self.file2
            fl_text = qtgui.QFileDialog.getOpenFileName(
                parent=self, caption="Select Shoeprint TIFF file", filter="*.tiff"
            )
        elif file_no == 3:
            fl = self.targ_dir
            fl_text = [
                qtgui.QFileDialog.getExistingDirectory(
                    caption="Select Target Directory for Report"
                )
            ]

        if fl_text[0] != "":
            fl.setText(fl_text[0])

    def done_button(self):
        self.done = True
        worker = PercentageWorker()
        worker.percentageChanged.connect(self.progress.setValue)
        worker.txtChanged.connect(self.dbg.setText)
        threading.Thread(
            target=runner,
            kwargs=dict(window=self, worker=worker),
            daemon=True,
        ).start()
