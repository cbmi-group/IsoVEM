from PyQt5.QtWidgets import QApplication, QFileDialog, \
    QMessageBox, QHBoxLayout
from PyQt5 import QtWidgets
import MainWindow
import json
import pyqtgraph as pg
from utils import *
import os
import threading
import argparse
from train import train_func
from test import test_func
from skimage import io
import h5py

class Stats(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = MainWindow.Ui_isoVEM()
        self.ui.setupUi(self)
        self.ui.retranslateUi(self)
        self.resize(800, 600)

        # Training
        self.train_configName = self.ui.train_edit_config_name.text()
        self.ui.train_edit_config_name.textChanged.connect(self.train_configName_change)
        self.ui.pBut_add_train_data_path.clicked.connect(self.openTrainInputDataPath)
        self.ui.edit_train_data_path.textChanged.connect(self.train_edit_data_path_change)
        self.ui.pBut_add_train_output_path.clicked.connect(self.openTrainOutputDataPath)
        self.ui.train_edit_output_path.textChanged.connect(self.train_edit_output_path_change)
        self.ui.train_edit_data_split.textChanged.connect(self.train_val_split_change)
        self.train_val_split = float(self.ui.train_edit_data_split.text())
        self.train_upscale_factor = self.ui.train_sb_upscale.value()
        self.ui.train_sb_upscale.valueChanged.connect(self.train_upscale_change)
        self.train_max_epochs = self.ui.train_sb_maxEpochs.value()
        self.ui.train_sb_maxEpochs.valueChanged.connect(self.train_maxEpochs_change)
        self.train_batch_size = self.ui.train_sb_batchSize.value()
        self.ui.train_sb_batchSize.valueChanged.connect(self.train_batchSize_change)
        self.train_ckpt_interval = self.ui.train_sb_ckpt_interval.value()
        self.ui.train_sb_ckpt_interval.valueChanged.connect(self.train_ckpt_interval_change)
        self.ui.train_edit_gpuIds.textChanged.connect(self.train_gpuIds_change)
        self.train_gpuIds = self.ui.train_edit_gpuIds.text()
        self.ui.train_edit_lr.textChanged.connect(self.train_lr_change)
        self.train_lr = float(self.ui.train_edit_lr.text())
        self.ui.train_ckb_inpaint.stateChanged.connect(self.train_inpaint_stateChanged)
        self.train_inpaint = True if self.ui.train_ckb_inpaint.isChecked() else False
        self.ui.train_ckb_resume.stateChanged.connect(self.train_resume_stateChanged)
        self.ui.pBut_add_ckpt_path.clicked.connect(self.train_load_weightPath)
        self.train_resume_weightPath=self.ui.pBut_add_ckpt_path.text()
        if not self.ui.train_ckb_resume.isChecked():
            self.train_resume = False
            self.ui.train_edit_ckpt_path.setText("")
            self.ui.label_32.setVisible(False)
            self.ui.train_edit_ckpt_path.setVisible(False)
            self.ui.pBut_add_ckpt_path.setVisible(False)
        self.ui.train_pBut_saveConfigs.clicked.connect(self.train_saveConfigs)
        self.ui.train_pBut_loadConfigs.clicked.connect(self.train_loadConfigs)
        self.ui.train_pBut_ok.clicked.connect(self.train_ok)
        self.ui.train_pBut_stop.clicked.connect(self.train_stop)

        # Testing
        self.test_configName = self.ui.test_edit_config_name.text()
        self.ui.test_edit_config_name.textChanged.connect(self.test_configName_change)
        self.ui.pBut_add_test_data_path.clicked.connect(self.openTestInputDataPath)
        self.ui.test_edit_data_path.textChanged.connect(self.test_edit_data_path_change)
        self.ui.pBut_add_test_output_path.clicked.connect(self.openTestOutputDataPath)
        self.ui.test_edit_output_path.textChanged.connect(self.test_edit_output_path_change)
        self.ui.pBut_add_test_ckpt_path.clicked.connect(self.test_load_weightPath)
        self.ui.edit_test_ckpt_path.textChanged.connect(self.test_ckpt_path_change)
        self.ui.pBut_add_train_config_path.clicked.connect(self.test_loadTrainConfigs)
        self.ui.test_edit_train_config_path.textChanged.connect(self.test_TrainConfig_change)
        self.test_upscale_factor = self.ui.test_sb_upscale.value()
        self.ui.test_sb_upscale.valueChanged.connect(self.test_upscale_change)
        self.ui.test_edit_gpuIds.textChanged.connect(self.test_gpuIds_change)
        self.test_gpuIds = self.ui.test_edit_gpuIds.text()
        self.ui.test_ckb_inpaint.stateChanged.connect(self.test_inpaint_stateChanged)
        if self.ui.test_ckb_inpaint.isChecked():
            self.test_inpaint = True
            self.ui.test_edit_inpaint_ids.setVisible(True)
            self.ui.label_35.setVisible(True)
        else:
            self.test_inpaint = False
            self.ui.test_edit_inpaint_ids.setVisible(False)
            self.ui.label_35.setVisible(False)
        self.ui.test_edit_inpaint_ids.textChanged.connect(self.test_inpaint_ids_change)
        self.test_inpaint_ids = self.ui.test_edit_inpaint_ids.text()
        self.ui.test_pBut_saveConfigs.clicked.connect(self.test_saveConfigs)
        self.ui.test_pBut_loadConfigs.clicked.connect(self.test_loadConfigs)
        self.ui.test_pBut_ok.clicked.connect(self.test_ok)
        self.ui.test_pBut_stop.clicked.connect(self.test_stop)

        # Visualization
        self.sectionView("")
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.win, stretch=3)
        self.hbox.addWidget(self.ui.gBox_res_load, stretch=2)
        self.ui.tab_res_show.setLayout(self.hbox)
        self.sub1.scene().sigMouseClicked.connect(self.mouseClicked)
        pg.SignalProxy(self.sub1.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClicked)
        self.ui.pBut_show_tomo.clicked.connect(self.showImg)
        self.ui.pBut_show_label.clicked.connect(self.showUncertImg)
        self.uncert_data = []
        self.ui.cbox_uncert_show.stateChanged.connect(self.isShowUncert)
        self.flag_show_uncert = False
        self.ui.hSlider_uncert_alpha.setVisible(False)
        self.ui.edit_uncert_alpha.setVisible(False)
        self.ui.label_48.setVisible(False)
        self.ui.hSlider_uncert_alpha.valueChanged.connect(self.uncertAlphaChange)
        self.ui.hSlider_uncert_alpha.setValue(20)
        self.uncert_alpha = self.ui.hSlider_uncert_alpha.value()
        self.old_alpha = self.uncert_alpha
        self.ui.edit_uncert_alpha.setText(f"{float(self.uncert_alpha) / 100.:0.2f}")
        self.color = (255, 0, 0)

    """
    Training funcs
    """
    def train_show_info(self, info):
        self.ui.train_txtB.insertPlainText(f"{info}\n")
        self.ui.train_txtB.ensureCursorVisible()
    def train_configName_change(self):
        self.train_configName = self.ui.train_edit_config_name.text()
        self.train_show_info(f"Training config name: {self.train_configName}")

    def openTrainInputDataPath(self):
        self.train_input_data_path, _ = QFileDialog.getOpenFileName(self, 'Select the train data path')
        if self.train_input_data_path != "":
            self.ui.edit_train_data_path.setText(self.train_input_data_path)
            self.train_show_info(f"Training - input_data_path: {self.train_input_data_path}")

    def train_edit_data_path_change(self):
        self.train_input_data_path = self.ui.edit_train_data_path.text()
        self.train_show_info(f"Training - input_data_path: {self.train_input_data_path}")

    def openTrainOutputDataPath(self):
        self.train_output_data_path = QFileDialog.getExistingDirectory(self, 'Select the train data path')
        if self.train_output_data_path != "":
            self.ui.train_edit_output_path.setText(self.train_output_data_path)
            self.train_show_info(f"Training - output_data_path: {self.train_output_data_path}")

    def train_edit_output_path_change(self):
        self.train_output_data_path = self.ui.train_edit_output_path.text()
        self.train_show_info(f"Training - output_data_path: {self.train_output_data_path}")

    def train_val_split_change(self):
        self.train_val_split = float(self.ui.train_edit_data_split.text())
        self.train_show_info(f"Training - val_split: {self.train_val_split}")

    def train_upscale_change(self):
        self.train_upscale_factor = self.ui.train_sb_upscale.value()
        self.train_show_info(f"Training - upscale_factor: {self.train_upscale_factor}")

    def train_maxEpochs_change(self):
        self.train_max_epochs = self.ui.train_sb_maxEpochs.value()
        self.train_show_info(f"Training - max epochs: {self.train_max_epochs}")

    def train_batchSize_change(self):
        self.train_batch_size = self.ui.train_sb_batchSize.value()
        self.train_show_info(f"Training - batch size: {self.train_batch_size}")

    def train_ckpt_interval_change(self):
        self.train_ckpt_interval = self.ui.train_sb_ckpt_interval.value()
        self.train_show_info(f"Training - ckpt interval: {self.train_ckpt_interval}")

    def train_gpuIds_change(self):
        self.train_gpuIds = self.ui.train_edit_gpuIds.text()
        self.train_show_info(f"Training - gpu ids: {self.train_gpuIds}")

    def train_lr_change(self):
        self.train_lr = float(self.ui.train_edit_lr.text())
        self.train_show_info(f"Training - learning rate: {self.train_lr}")
    
    def train_inpaint_stateChanged(self):
        if self.ui.train_ckb_inpaint.isChecked():
            self.train_inpaint = True
            self.train_show_info(f"Training - is_inpaint: {self.train_inpaint}")
        else:
            self.train_inpaint = False
            self.train_show_info(f"Training - is_inpaint: {self.train_inpaint}")

    def train_load_weightPath(self):
        self.train_resume_weightPath, _ = QFileDialog.getOpenFileName(self, 'Select the training weight')
        if self.train_resume_weightPath != "":
            self.ui.train_edit_ckpt_path.setText(self.train_resume_weightPath)
            self.train_show_info(f"Load training weight: {self.train_resume_weightPath}")

    def train_ckpt_path_change(self):
        self.train_resume_weightPath = self.ui.train_edit_ckpt_path.text()
        self.train_show_info(f"Load training weight: {self.train_resume_weightPath}")

    def train_resume_stateChanged(self):
        if self.ui.train_ckb_resume.isChecked():
            self.train_resume = True
            self.ui.label_32.setVisible(True)
            self.ui.train_edit_ckpt_path.setVisible(True)
            self.ui.pBut_add_ckpt_path.setVisible(True)
            self.train_show_info(f"Training - is_resume: {self.train_resume}")
        else:
            self.train_resume = False
            self.train_resume_weightPath = ""
            self.ui.train_edit_ckpt_path.setText("")
            self.ui.label_32.setVisible(False)
            self.ui.train_edit_ckpt_path.setVisible(False)
            self.ui.pBut_add_ckpt_path.setVisible(False)
            self.train_show_info(f"Training - is_resume: {self.train_resume}")

    def train_saveConfigs(self):
        if self.ui.train_edit_config_name.text() == "" \
                or self.ui.edit_train_data_path.text() == "" \
                or self.ui.train_edit_output_path.text() == "" \
                or self.ui.train_edit_data_split.text() == "" \
                or self.ui.train_sb_upscale.text() == "" \
                or self.ui.train_sb_maxEpochs.text() == "" \
                or self.ui.train_sb_batchSize.text() == "" \
                or self.ui.train_sb_ckpt_interval.text() == "" \
                or self.ui.train_edit_gpuIds.text() == "" \
                or self.ui.train_edit_lr.text() == "" \
                or (self.ui.train_edit_ckpt_path.text() == "" and self.ui.train_ckb_resume.isChecked()):
            QMessageBox.critical(self, 'Error', 'Incomplete information')
        else:
            train_output_dir=self.train_output_data_path if isinstance(self.train_output_data_path, str) else self.train_output_data_path[0]
            self.train_configs = dict(
                train_config_name=(self.train_configName if isinstance(self.train_configName, str) else self.train_configName[0]),
                train_data_pth=self.train_input_data_path if isinstance(self.train_input_data_path, str) else self.train_input_data_path[0],
                train_output_dir=self.train_output_data_path if isinstance(self.train_output_data_path, str) else self.train_output_data_path[0],
                train_data_split=self.train_val_split if isinstance(self.train_val_split, float) else self.train_val_split[0],
                train_upscale=self.train_upscale_factor if isinstance(self.train_upscale_factor, int) else self.train_upscale_factor[0],
                train_inpaint=True if self.ui.train_ckb_inpaint.isChecked() else False,
                train_epoch=self.train_max_epochs if isinstance(self.train_max_epochs, int) else self.train_max_epochs[0],
                train_bs=self.train_batch_size if isinstance(self.train_batch_size, int) else self.train_batch_size[0],
                train_lr=self.train_lr if isinstance(self.train_lr, float) else self.train_lr[0],
                train_ckpt_interval=self.train_ckpt_interval if isinstance(self.train_ckpt_interval, int) else self.train_ckpt_interval[0],
                train_is_resume=True if self.ui.train_ckb_resume.isChecked() else False,
                train_resume_ckpt_path=self.train_resume_weightPath if isinstance(self.train_resume_weightPath, str) else self.train_resume_weightPath[0],
                train_gpu_ids=self.train_gpuIds if isinstance(self.train_gpuIds, str) else self.train_gpuIds[0],
            )
            config_save_path = './configs'
            # config_save_path = os.path.join(train_output_dir, 'configs')
            os.makedirs(config_save_path, exist_ok=True)
            self.train_config_save_path = f"{config_save_path}/{self.train_configName}.json"
            with open(self.train_config_save_path, 'w') as f:
                f.write(json.dumps(self.train_configs, indent=4))
                # f.write("train_configs=")
                # json.dump(self.train_configs, f, separators=(',\n'+' '*len('train_configs={'), ': '))
            self.train_show_info(f"save train configs to '{config_save_path}/{self.train_configName}.json'")

    def train_loadConfigs(self):
        self.train_config_file, _ = QFileDialog.getOpenFileName(self, 'Select the training configs')

        if self.train_config_file != "":
            self.ui.train_edit_loadConfigs.setText(self.train_config_file)
            self.train_show_info(f"Load training configs: {self.train_config_file}")

        with open(self.train_config_file, 'r') as f:
            self.train_configs = json.loads(''.join(f.readlines()))
            # self.train_configs = json.loads(''.join(f.readlines()).lstrip('train_configs='))

        self.train_configName = self.train_configs['train_config_name']
        self.train_input_data_path = self.train_configs['train_data_pth']
        self.train_output_data_path = self.train_configs['train_output_dir']
        self.train_val_split = self.train_configs['train_data_split']
        self.train_upscale_factor = self.train_configs['train_upscale']
        self.train_max_epochs = self.train_configs['train_epoch']
        self.train_inpaint = True if self.train_configs['train_inpaint'] is True else False
        self.train_resume = True if self.train_configs['train_is_resume'] is True else False
        self.train_batch_size = self.train_configs['train_bs']
        self.train_ckpt_interval = self.train_configs['train_ckpt_interval']
        self.train_gpuIds = self.train_configs['train_gpu_ids']
        self.train_lr = self.train_configs['train_lr']
        self.train_resume_weightPath = self.train_configs['train_resume_ckpt_path']


        self.ui.train_edit_config_name.setText(self.train_configName)
        self.ui.edit_train_data_path.setText(self.train_input_data_path)
        self.ui.train_edit_output_path.setText(self.train_output_data_path)
        self.ui.train_edit_data_split.setText(str(self.train_val_split))
        self.ui.train_sb_upscale.setValue(self.train_upscale_factor)
        self.ui.train_sb_maxEpochs.setValue(self.train_max_epochs)
        self.ui.train_sb_batchSize.setValue(self.train_batch_size)
        self.ui.train_sb_ckpt_interval.setValue(self.train_ckpt_interval)
        self.ui.train_edit_gpuIds.setText(self.train_gpuIds)
        self.ui.train_edit_lr.setText(str(self.train_lr))
        if self.train_inpaint:
            self.ui.train_ckb_inpaint.setChecked(True)
        else:
            self.ui.train_ckb_inpaint.setChecked(False)

        if self.train_resume:
            self.ui.train_ckb_resume.setChecked(True)
            self.ui.train_edit_ckpt_path.setText(self.train_resume_weightPath)
        else:
            self.ui.train_edit_ckpt_path.setText("")
            self.train_resume_weightPath = ""
            self.ui.train_ckb_resume.setChecked(False)

        self.train_show_info('*' * 100)
        for i in self.train_configs.keys():
            self.train_show_info(f'{i}: {self.train_configs[i]}')
        self.train_show_info('*' * 100)

    def train_loadConfigs_v1(self):
        self.train_config_file = self.train_config_save_path

        if self.train_config_file != "":
            self.ui.train_edit_loadConfigs.setText(self.train_config_file)
            self.train_show_info(f"Load training configs: {self.train_config_file}")

        with open(self.train_config_file, 'r') as f:
            self.train_configs = json.loads(''.join(f.readlines()))
            # self.train_configs = json.loads(''.join(f.readlines()).lstrip('train_configs='))

        self.train_configName = self.train_configs['train_config_name']
        self.train_input_data_path = self.train_configs['train_data_pth']
        self.train_output_data_path = self.train_configs['train_output_dir']
        self.train_val_split = self.train_configs['train_data_split']
        self.train_upscale_factor = self.train_configs['train_upscale']
        self.train_max_epochs = self.train_configs['train_epoch']
        self.train_inpaint = True if self.train_configs['train_inpaint'] is True else False
        self.train_resume = True if self.train_configs['train_is_resume'] is True else False
        self.train_batch_size = self.train_configs['train_bs']
        self.train_ckpt_interval = self.train_configs['train_ckpt_interval']
        self.train_gpuIds = self.train_configs['train_gpu_ids']
        self.train_lr = self.train_configs['train_lr']
        self.train_resume_weightPath = self.train_configs['train_resume_ckpt_path']


        self.ui.train_edit_config_name.setText(self.train_configName)
        self.ui.edit_train_data_path.setText(self.train_input_data_path)
        self.ui.train_edit_output_path.setText(self.train_output_data_path)
        self.ui.train_edit_data_split.setText(str(self.train_val_split))
        self.ui.train_sb_upscale.setValue(self.train_upscale_factor)
        self.ui.train_sb_maxEpochs.setValue(self.train_max_epochs)
        self.ui.train_sb_batchSize.setValue(self.train_batch_size)
        self.ui.train_sb_ckpt_interval.setValue(self.train_ckpt_interval)
        self.ui.train_edit_gpuIds.setText(self.train_gpuIds)
        self.ui.train_edit_lr.setText(str(self.train_lr))
        if self.train_inpaint:
            self.ui.train_ckb_inpaint.setChecked(True)
        else:
            self.ui.train_ckb_inpaint.setChecked(False)

        if self.train_resume:
            self.ui.train_ckb_resume.setChecked(True)
            self.ui.train_edit_ckpt_path.setText(self.train_resume_weightPath)
        else:
            self.ui.train_edit_ckpt_path.setText("")
            self.train_resume_weightPath = ""
            self.ui.train_ckb_resume.setChecked(False)

        self.train_show_info('*' * 100)
        for i in self.train_configs.keys():
            self.train_show_info(f'{i}: {self.train_configs[i]}')
        self.train_show_info('*' * 100)

    def train_stop(self):
        try:
            # self.train_thread.n = 0
            # self.train_thread.join()
            # os.system(f"kill -9 {self.train_thread.pid_num}")
            stop_thread(self.train_thread)
            # self.train_t.pause()
            # os.system(f"kill {self.train_pid}")
            # self.train_thread.terminate()
        except:
            pass
        self.train_show_info('*' * 100)
        self.train_show_info('Training Stopped')
        self.train_show_info('*' * 100)

    def train_ok(self):
        self.train_saveConfigs()
        self.train_loadConfigs_v1()
        if self.ui.train_edit_config_name.text() == "" \
                or self.ui.edit_train_data_path.text() == "" \
                or self.ui.train_edit_output_path.text() == "" \
                or self.ui.train_edit_data_split.text() == "" \
                or self.ui.train_sb_upscale.text() == "" \
                or self.ui.train_sb_maxEpochs.text() == "" \
                or self.ui.train_sb_batchSize.text() == "" \
                or self.ui.train_sb_ckpt_interval.text() == "" \
                or self.ui.train_edit_gpuIds.text() == "" \
                or self.ui.train_edit_lr.text() == "" \
                or (self.ui.train_edit_ckpt_path.text() == "" and self.ui.train_ckb_resume.isChecked()):
            QMessageBox.critical(self, 'Error', 'Incomplete information')
        else:
            self.train_show_info('*' * 100)
            self.train_show_info('Final training configuration parameters')
            self.train_show_info('*' * 100)
            self.train_show_info(f"Training - config name: {self.train_configName}")
            self.train_show_info(f"Training - input data file: {self.train_input_data_path}")
            self.train_show_info(f"Training - output data path: {self.train_output_data_path}")
            self.train_show_info(f"Training - train_val_split: {self.train_val_split}")
            self.train_show_info(f"Training - upscale_factor: {self.train_upscale_factor}")
            self.train_show_info(f"Training - max epochs: {self.train_max_epochs}")
            self.train_show_info(f"Training - batch size: {self.train_batch_size}")
            self.train_show_info(f"Training - train_ckpt_interval: {self.train_ckpt_interval}")
            self.train_show_info(f"Training - gpu ids: {self.train_gpuIds}")
            self.train_show_info(f"Training - learning rate: {self.train_lr}")
            self.train_show_info(f"Training - is_inpaint: {self.train_inpaint}")
            self.train_show_info(f"Training - is_resume: {self.train_resume}")
            if self.train_resume:
                self.train_show_info(f"Training - resume_weightPath: {self.train_resume_weightPath}")
            self.train_show_info('*' * 100)

            """
            threading.Thread
            """
            parser = argparse.ArgumentParser(description='Parameters for IsoVEM Training')
            parser.add_argument('--train_config_path', help='path of train config file', type=str,
                                default=f"configs/{self.train_configName}.json")

            with open(parser.parse_args().train_config_path, 'r', encoding='UTF-8') as f:
                train_config = json.load(f)
            add_dict_to_argparser(parser, train_config)
            args = parser.parse_args()

            self.train_emit = EmittingStr()
            self.train_emit.textWritten.connect(self.train_show_info)
            self.train_thread = threading.Thread(target=train_func, args=(args, self.train_emit))
            # self.train_thread = myThread(1, train, args, self.train_emit)
            self.train_thread.start()

    """
    Predicting
    """
    def test_show_info(self, info):
        self.ui.test_txtB.insertPlainText(f"{info}\n")
        self.ui.test_txtB.ensureCursorVisible()

    def test_configName_change(self):
        self.test_configName = self.ui.test_edit_config_name.text()
        self.test_show_info(f"Predicting config name: {self.test_configName}")

    def openTestInputDataPath(self):
        self.test_input_data_path, _ = QFileDialog.getOpenFileName(self, 'Select the predicted data path')
        if self.test_input_data_path != "":
            self.ui.test_edit_data_path.setText(self.test_input_data_path)
            self.test_show_info(f"Predicting - input_data_path: {self.test_input_data_path}")

    def test_edit_data_path_change(self):
        self.test_input_data_path = self.ui.test_edit_data_path.text()
        self.test_show_info(f"Predicting - input_data_path: {self.test_input_data_path}")

    def openTestOutputDataPath(self):
        self.test_output_data_path = QFileDialog.getExistingDirectory(self, 'Select the predicted data path')
        if self.test_output_data_path != "":
            self.ui.test_edit_output_path.setText(self.test_output_data_path)
            self.test_show_info(f"Predicting - output_data_path: {self.test_output_data_path}")

    def test_edit_output_path_change(self):
        self.test_output_data_path = self.ui.test_edit_output_path.text()
        self.test_show_info(f"Predicting - output_data_path: {self.test_output_data_path}")

    def test_load_weightPath(self):
        self.test_weightPath, _ = QFileDialog.getOpenFileName(self, 'Select the pretrained weight')
        if self.test_weightPath != "":
            self.ui.edit_test_ckpt_path.setText(self.test_weightPath)
            self.test_show_info(f"Load training weight: {self.test_weightPath}")

    def test_ckpt_path_change(self):
        self.test_weightPath = self.ui.edit_test_ckpt_path.text()
        self.test_show_info(f"Load training weight: {self.test_weightPath}")

    def test_loadTrainConfigs(self):
        self.test_Trainconfig_path, _ = QFileDialog.getOpenFileName(self, 'Select the training configs')

        if self.test_Trainconfig_path != "":
            self.ui.test_edit_train_config_path.setText(self.test_Trainconfig_path)
            self.test_show_info(f"Load training configs: {self.test_Trainconfig_path}")

    def test_TrainConfig_change(self):
        self.test_Trainconfig_path = self.ui.test_edit_train_config_path.text()
        self.test_show_info(f"Load training configs: {self.test_Trainconfig_path}")

    def test_upscale_change(self):
        self.test_upscale_factor = self.ui.test_sb_upscale.value()
        self.test_show_info(f"Predicting - upscale_factor: {self.test_upscale_factor}")

    def test_gpuIds_change(self):
        self.test_gpuIds = self.ui.test_edit_gpuIds.text()
        self.test_show_info(f"Predicting - gpu ids: {self.test_gpuIds}")

    def test_inpaint_stateChanged(self):
        if self.ui.test_ckb_inpaint.isChecked():
            self.test_inpaint = True
            self.ui.test_edit_inpaint_ids.setVisible(True)
            self.ui.label_35.setVisible(True)
            self.test_show_info(f"Predicting - is_inpaint: {self.test_inpaint}")
        else:
            self.test_inpaint = False
            self.ui.test_edit_inpaint_ids.setVisible(False)
            self.ui.label_35.setVisible(False)
            self.test_show_info(f"Predicting - is_inpaint: {self.test_inpaint}")

    def test_inpaint_ids_change(self):
        self.test_inpaint_ids = self.ui.test_edit_inpaint_ids.text()
        self.test_show_info(f"Predicting - inpaint_ids: {self.test_inpaint_ids}")

    def test_saveConfigs(self):
        if self.ui.test_edit_config_name.text() == "" \
                or self.ui.test_edit_data_path.text() == "" \
                or self.ui.test_edit_output_path.text() == "" \
                or self.ui.edit_test_ckpt_path.text() == "" \
                or self.ui.test_edit_train_config_path.text() == "" \
                or self.ui.test_sb_upscale.text() == "" \
                or self.ui.test_edit_gpuIds.text() == "" \
                or (self.ui.test_edit_inpaint_ids.text() == "" and self.ui.test_ckb_inpaint.isChecked()):
            QMessageBox.critical(self, 'Error', 'Incomplete information')
        else:
            test_output_dir=self.test_output_data_path if isinstance(self.test_output_data_path, str) else self.test_output_data_path[0]
            self.test_configs = dict(
                test_config_name=(self.test_configName if isinstance(self.test_configName, str) else self.test_configName[0]),
                test_data_pth=self.test_input_data_path if isinstance(self.test_input_data_path, str) else self.test_input_data_path[0],
                test_output_dir=self.test_output_data_path if isinstance(self.test_output_data_path, str) else self.test_output_data_path[0],
                test_ckpt_path=self.test_weightPath if isinstance(self.test_weightPath, str) else self.test_weightPath[0],
                train_config_path=self.test_Trainconfig_path if isinstance(self.test_Trainconfig_path, str) else self.test_Trainconfig_path[0],
                test_upscale=self.test_upscale_factor if isinstance(self.test_upscale_factor, int) else self.test_upscale_factor[0],
                test_gpu_ids=self.test_gpuIds if isinstance(self.test_gpuIds, str) else self.test_gpuIds[0],
                test_inpaint=True if self.ui.test_ckb_inpaint.isChecked() else False,
                test_inpaint_index=self.test_inpaint_ids if isinstance(self.test_inpaint_ids, str) else self.test_inpaint_ids[0],
            )
            config_save_path = './configs'
            os.makedirs(config_save_path, exist_ok=True)
            self.test_config_save_path = f"{config_save_path}/{self.test_configName}.json"
            with open(self.test_config_save_path, 'w') as f:
                f.write(json.dumps(self.test_configs, indent=4))
                # f.write("test_configs=")
                # json.dump(self.test_configs, f, separators=(',\n'+' '*len('test_configs={'), ': '))
            self.test_show_info(f"save predicting configs to '{self.test_config_save_path}.json'")

    def test_loadConfigs(self):
        self.test_config_file, _ = QFileDialog.getOpenFileName(self, 'Select the predicting configs')

        if self.test_config_file != "":
            self.ui.test_edit_loadConfigs.setText(self.test_config_file)
            self.test_show_info(f"Load predicting configs: {self.test_config_file}")

        with open(self.test_config_file, 'r') as f:
            self.test_configs = json.loads(''.join(f.readlines()))
            # self.test_configs = json.loads(''.join(f.readlines()).lstrip('test_configs='))

        self.test_configName = self.test_configs['test_config_name']
        self.test_input_data_path = self.test_configs['test_data_pth']
        self.test_output_data_path = self.test_configs['test_output_dir']
        self.test_weightPath = self.test_configs['test_ckpt_path']
        self.test_Trainconfig_path = self.test_configs['train_config_path']
        self.test_upscale_factor = self.test_configs['test_upscale']
        self.test_gpuIds = self.test_configs['test_gpu_ids']
        self.test_inpaint = True if self.test_configs['test_inpaint'] is True else False
        self.test_inpaint_ids = self.test_configs['test_inpaint_index']

        self.ui.test_edit_config_name.setText(self.test_configName)
        self.ui.test_edit_data_path.setText(self.test_input_data_path)
        self.ui.test_edit_output_path.setText(self.test_output_data_path)
        self.ui.edit_test_ckpt_path.setText(self.test_weightPath)
        self.ui.test_edit_train_config_path.setText(self.test_Trainconfig_path)
        self.ui.test_sb_upscale.setValue(self.test_upscale_factor)
        self.ui.test_edit_gpuIds.setText(self.test_gpuIds)

        if self.test_inpaint:
            self.ui.test_ckb_inpaint.setChecked(True)
            self.ui.test_edit_inpaint_ids.setText(self.test_inpaint_ids)
        else:
            self.ui.test_ckb_inpaint.setChecked(False)
            self.test_inpaint_ids=""
            self.ui.test_edit_inpaint_ids.setText("")

        self.test_show_info('*' * 100)
        for i in self.test_configs.keys():
            self.test_show_info(f'{i}: {self.test_configs[i]}')
        self.test_show_info('*' * 100)

    def test_loadConfigs_v1(self):
        self.test_config_file = self.test_config_save_path

        if self.test_config_file != "":
            self.ui.test_edit_loadConfigs.setText(self.test_config_file)
            self.test_show_info(f"Load predicting configs: {self.test_config_file}")

        with open(self.test_config_file, 'r') as f:
            self.test_configs = json.loads(''.join(f.readlines()))
            # self.test_configs = json.loads(''.join(f.readlines()).lstrip('test_configs='))

        self.test_configName = self.test_configs['test_config_name']
        self.test_input_data_path = self.test_configs['test_data_pth']
        self.test_output_data_path = self.test_configs['test_output_dir']
        self.test_weightPath = self.test_configs['test_ckpt_path']
        self.test_Trainconfig_path = self.test_configs['train_config_path']
        self.test_upscale_factor = self.test_configs['test_upscale']
        self.test_gpuIds = self.test_configs['test_gpu_ids']
        self.test_inpaint = True if self.test_configs['test_inpaint'] is True else False
        self.test_inpaint_ids = self.test_configs['test_inpaint_index']

        self.ui.test_edit_config_name.setText(self.test_configName)
        self.ui.test_edit_data_path.setText(self.test_input_data_path)
        self.ui.test_edit_output_path.setText(self.test_output_data_path)
        self.ui.edit_test_ckpt_path.setText(self.test_weightPath)
        self.ui.test_edit_train_config_path.setText(self.test_Trainconfig_path)
        self.ui.test_sb_upscale.setValue(self.test_upscale_factor)
        self.ui.test_edit_gpuIds.setText(self.test_gpuIds)

        if self.test_inpaint:
            self.ui.test_ckb_inpaint.setChecked(True)
            self.ui.test_edit_inpaint_ids.setText(self.test_inpaint_ids)
        else:
            self.ui.test_ckb_inpaint.setChecked(False)
            self.test_inpaint_ids=""
            self.ui.test_edit_inpaint_ids.setText("")

        self.test_show_info('*' * 100)
        for i in self.test_configs.keys():
            self.test_show_info(f'{i}: {self.test_configs[i]}')
        self.test_show_info('*' * 100)

    def test_stop(self):
        try:
            stop_thread(self.test_thread)
        except:
            pass

        self.test_show_info('*' * 100)
        self.test_show_info('Testing stopped!')
        self.test_show_info('*' * 100)

    def test_ok(self):
        self.test_saveConfigs()
        self.test_loadConfigs_v1()
        if self.ui.test_edit_config_name.text() == "" \
                or self.ui.test_edit_data_path.text() == "" \
                or self.ui.test_edit_output_path.text() == "" \
                or self.ui.edit_test_ckpt_path.text() == "" \
                or self.ui.test_edit_train_config_path.text() == "" \
                or self.ui.test_sb_upscale.text() == "" \
                or self.ui.test_edit_gpuIds.text() == "" \
                or (self.ui.test_edit_inpaint_ids.text() == "" and self.ui.test_ckb_inpaint.isChecked()):
            QMessageBox.critical(self, 'Error', 'Incomplete information')
            return 0
        else:
            self.test_show_info('*' * 100)
            self.train_show_info('Final inference configuration parameters')
            self.train_show_info('*' * 100)
            self.train_show_info(f"Testing - config Name: {self.test_configName}")
            self.train_show_info(f"Testing - input data path: {self.test_input_data_path}")
            self.train_show_info(f"Testing - output data path: {self.test_output_data_path}")
            self.train_show_info(f"Testing - train config path: {self.test_Trainconfig_path}")
            self.train_show_info(f"Testing - upscale factor: {self.test_upscale_factor}")
            self.train_show_info(f"Testing - gpu ids: {self.test_gpuIds}")
            self.train_show_info(f"Testing - is_inpaint: {self.test_inpaint}")
            if self.test_inpaint:
                self.train_show_info(f"Testing - inpaint ids: {self.test_inpaint_ids}")
            self.train_show_info('*' * 100)

        parser = argparse.ArgumentParser(description='Parameters for IsoVEM Training')
        parser.add_argument('--test_config_path', help='path of test config file', type=str,
                            default="configs/demo_test.json")

        with open(parser.parse_args().test_config_path, 'r', encoding='UTF-8') as f:
            test_config = json.load(f)
        add_dict_to_argparser(parser, test_config)
        args = parser.parse_args()
        """
        threading.Thread
        """
        self.test_emit = EmittingStr()
        self.test_emit.textWritten.connect(self.test_show_info)
        self.test_thread = threading.Thread(target=test_func, args=(args, self.test_emit))
        self.test_thread.start()

    """
    Visualization
    """
    def visual_show_info(self, info):
        self.ui.visual_txtB.insertPlainText(f"{info}\n")
        self.ui.visual_txtB.ensureCursorVisible()

    def sectionView(self, tomo_path):
        if tomo_path == "":
            self.tomo_data = np.random.randn(200, 400, 400)
        else:
            if self.show_tomoFile.split('.')[-1] == 'tif':
                self.tomo_data = io.imread(self.show_tomoFile)
            elif self.show_tomoFile.split('.')[-1] == 'h5':
                self.tomo_data = np.array(h5py.File(self.show_tomoFile, 'r')['raw'])
            else:
                raise ValueError(f'Not support the image format of {self.show_tomoFile}')
            #self.tomo_data = stretch(self.tomo_data)
        z_max, y_max, x_max = self.tomo_data.shape

        self.data_xy = np.transpose(self.tomo_data[int(z_max // 2), :, :])
        self.data_zy = self.tomo_data[:, :, x_max // 2]
        self.data_xz = np.transpose(self.tomo_data[:, y_max // 2, :])
        # print(z_max, y_max, x_max)

        self.win = pg.GraphicsLayoutWidget()
        self.win.show()  ## show widget alone in its own window
        self.win.setWindowTitle('pyqtgraph example: ImageItem')
        self.win.resize(x_max + z_max, y_max + z_max)
        # win.ci.setBorder((5, 5, 10))

        self.win.nextRow()
        self.sub1 = self.win.addLayout(border=(100, 10, 10))
        self.sub1.setContentsMargins(0, 0, 0, 0)
        # self.p_xy = self.sub1.addViewBox(row=0, col=0, rowspan=y_max, colspan=x_max)
        self.p_xy = self.sub1.addViewBox()
        self.p_xy.disableAutoRange()
        self.p_xy.setAspectLocked(True)  ## lock the aspect ratio so pixels are always square
        self.p_xy.setXRange(0, x_max)
        self.p_xy.setYRange(0, y_max)
        # self.p_xy.setLimits(xMin=-50, xMax=x_max+50, yMin=-50, yMax=y_max+50)
        self.img_xy = pg.ImageItem(border='b')
        self.p_xy.addItem(self.img_xy)
        self.img_xy.setImage(self.data_xy)

        # self.p_zy = self.sub1.addViewBox(row=0, col=x_max, rowspan=1, colspan=1)
        self.p_zy = self.sub1.addViewBox()
        self.p_zy.disableAutoRange()
        self.p_zy.setAspectLocked(True)  ## lock the aspect ratio so pixels are always square
        self.p_zy.setXRange(0, z_max)
        self.p_zy.setYRange(0, y_max)
        self.img_zy = pg.ImageItem(border='b')
        self.p_zy.addItem(self.img_zy)
        self.img_zy.setImage(self.data_zy)
        self.p_zy.linkView(self.p_xy.YAxis, self.p_xy)

        self.sub1.nextRow()
        # self.p_xz = self.sub1.addViewBox(row=y_max, col=0, rowspan=z_max, colspan=z_max)
        self.p_xz = self.sub1.addViewBox()
        self.p_xz.disableAutoRange()
        self.p_xz.setAspectLocked(True)  ## lock the aspect ratio so pixels are always square
        self.p_xz.setXRange(0, x_max)
        self.p_xz.setYRange(0, z_max)
        self.img_xz = pg.ImageItem(border='b')
        self.p_xz.addItem(self.img_xz)
        self.img_xz.setImage(self.data_xz)
        self.p_xz.linkView(self.p_xy.XAxis, self.p_xy)
        # print(xz_h, xz_w)

        self.text = pg.LabelItem(justify='center')
        self.sub1.addItem(self.text)
        self.text.setText(f"({z_max // 2}, {y_max // 2}, {x_max // 2})")
        # self.setSliderXYZ(z_max // 2, y_max // 2, x_max // 2, self.tomo_data.shape)

        # cross hair
        self.vLine_xy = pg.InfiniteLine(angle=90, movable=False)
        self.hLine_xy = pg.InfiniteLine(angle=0, movable=False)
        self.p_xy.addItem(self.vLine_xy, ignoreBounds=True)
        self.p_xy.addItem(self.hLine_xy, ignoreBounds=True)

        self.vLine_zy = pg.InfiniteLine(angle=90, movable=False)
        self.hLine_zy = pg.InfiniteLine(angle=0, movable=False)
        self.p_zy.addItem(self.vLine_zy, ignoreBounds=True)
        self.p_zy.addItem(self.hLine_zy, ignoreBounds=True)

        self.vLine_xz = pg.InfiniteLine(angle=90, movable=False)
        self.hLine_xz = pg.InfiniteLine(angle=0, movable=False)
        self.p_xz.addItem(self.vLine_xz, ignoreBounds=True)
        self.p_xz.addItem(self.hLine_xz, ignoreBounds=True)

        self.x, self.y, self.z = 0, 0, 0

    def mouseClicked(self, evt):
        print(evt)
        pos = evt.scenePos()
        if self.p_xy.sceneBoundingRect().contains(pos):
            mousePoint = self.p_xy.mapSceneToView(pos)
            self.x, self.y = mousePoint.x(), mousePoint.y()
        elif self.p_zy.sceneBoundingRect().contains(pos):
            mousePoint = self.p_zy.mapSceneToView(pos)
            self.z, self.y = mousePoint.x(), mousePoint.y()
        elif self.p_xz.sceneBoundingRect().contains(pos):
            mousePoint = self.p_xz.mapSceneToView(pos)
            self.x, self.z = mousePoint.x(), mousePoint.y()

        self.MC_updata()

    def MC_updata(self):
        self.vLine_xy.setPos(self.x)
        self.hLine_xy.setPos(self.y)
        self.vLine_zy.setPos(self.z)
        self.hLine_zy.setPos(self.y)
        self.vLine_xz.setPos(self.x)
        self.hLine_xz.setPos(self.z)

        if self.flag_show_uncert:
            self.text.setText(f"({self.x:.0f}, {self.y:.0f}, {self.z:.0f}), "
                              f"{self.tomo_data[int(self.z), int(self.y), int(self.x)]:.2f},"
                              f"{self.uncert_data[int(self.z), int(self.y), int(self.x)]:.2f}")
        else:
            self.text.setText(f"({self.x:.0f}, {self.y:.0f}, {self.z:.0f}), "
                          f"{self.tomo_data[int(self.z), int(self.y), int(self.x)]:.2f}")
        # self.setSliderXYZ(self.z, self.y, self.x, self.tomo_data.shape)
        self.data_xy = np.transpose(self.tomo_data[int(self.z), :, :])
        self.data_zy = self.tomo_data[:, :, int(self.x)]
        self.data_xz = np.transpose(self.tomo_data[:, int(self.y), :])

        # print(np.min(self.data_xy), np.max(self.data_xy)
        if self.uncert_data != []:
            if self.flag_show_uncert:
                self.label_xy = np.transpose(self.uncert_data[int(self.z), :, :])
                self.label_zy = self.uncert_data[:, :, int(self.x)]
                self.label_xz = np.transpose(self.uncert_data[:, int(self.y), :])

                self.img_xy.setImage(
                    add_transparency(self.data_xy, self.label_xy, float(self.uncert_alpha) / 100., self.color, 0.5))
                self.img_zy.setImage(
                    add_transparency(self.data_zy, self.label_zy, float(self.uncert_alpha) / 100., self.color, 0.5))
                self.img_xz.setImage(
                    add_transparency(self.data_xz, self.label_xz, float(self.uncert_alpha) / 100., self.color, 0.5))
            else:
                self.img_xy.setImage(self.data_xy)
                self.img_zy.setImage(self.data_zy)
                self.img_xz.setImage(self.data_xz)
        else:
            self.img_xy.setImage(self.data_xy)
            self.img_zy.setImage(self.data_zy)
            self.img_xz.setImage(self.data_xz)

    def showImg(self):
        self.show_tomoFile, _ = QFileDialog.getOpenFileName(self,
                                                            "Select the image file")

        if self.show_tomoFile != "":
            self.ui.edit_show_tomo.setPlainText(self.show_tomoFile)
            self.visual_show_info(f"Showing tomo file: {self.show_tomoFile}")

            if self.show_tomoFile.split('.')[-1] == 'tif':
                self.tomo_data = io.imread(self.show_tomoFile)
            elif self.show_tomoFile.split('.')[-1] == 'h5':
                self.tomo_data = np.array(h5py.File(self.show_tomoFile, 'r')['raw'])
            else:
                raise ValueError(f'Not support the image format of {self.show_tomoFile}')
            # self.tomo_data = stretch(self.tomo_data)
            self.tomo_shape = self.tomo_data.shape
            self.tomo_orig = self.tomo_data
            self.visual_show_info(f"Image shape: {self.tomo_shape}")

            self.MC_updata()

    def showUncertImg(self):
        self.show_uncertFile, _ = QFileDialog.getOpenFileName(self, "Select the uncert image")

        if self.show_uncertFile != "":
            self.ui.edit_show_label.setPlainText(self.show_uncertFile)
            self.visual_show_info(f"Showing uncert image: {self.show_uncertFile}")

            if self.show_uncertFile.split('.')[-1] == 'tif':
                self.uncert_data = io.imread(self.show_uncertFile)
            elif self.show_uncertFile.split('.')[-1] == 'h5':
                self.uncert_data = np.array(h5py.File(self.show_uncertFile, 'r')['raw'])
            else:
                raise ValueError(f'Not support the uncert image format of {self.show_uncertFile}')
            # self.tomo_data = stretch(self.tomo_data)
            self.uncert_shape = self.uncert_data.shape
            self.uncert_orig = self.uncert_data

            self.MC_updata()

    def isShowUncert(self):
        if self.ui.cbox_uncert_show.isChecked():
            self.flag_show_uncert = True
            self.ui.hSlider_uncert_alpha.setVisible(True)
            self.ui.edit_uncert_alpha.setVisible(True)
            self.ui.label_48.setVisible(True)
            self.uncert_alpha = self.old_alpha
            self.ui.hSlider_uncert_alpha.setValue(self.uncert_alpha)
            self.MC_updata()
        else:
            self.flag_show_uncert = False
            self.ui.hSlider_uncert_alpha.setVisible(False)
            self.ui.edit_uncert_alpha.setVisible(False)
            self.ui.label_48.setVisible(False)
            self.old_alpha = self.uncert_alpha
            self.uncert_alpha = 0
            self.MC_updata()

    def uncertAlphaChange(self):
        self.uncert_alpha = self.ui.hSlider_uncert_alpha.value()
        self.ui.edit_uncert_alpha.setText(f"{float(self.uncert_alpha) / 100.:0.2f}")
        self.MC_updata()

app = QApplication([])
stats = Stats()
stats.show()
app.exec_()