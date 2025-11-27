import sys

from PySide6.QtGui import QPalette, Qt

import synth
from PySide6.QtWidgets import *


class SingleOscillator(QFrame):
    def __init__(self, main_synth: synth.Synth, item: int, osc_id: int):
        super().__init__()
        self.item = item
        self.main_synth = main_synth
        self.osc_id = osc_id

        self.h_layout = QHBoxLayout()

        self.amplitude = QDoubleSpinBox()
        self.wave = QComboBox()
        self.frequency = QDoubleSpinBox()
        self.absolute = QPushButton()

        self.amplitude.setDecimals(3)
        self.wave.addItems(["No", "W1", "W2", "W3", "W4"])
        self.frequency.setDecimals(3)
        self.absolute.setFixedWidth(30)

        self.amplitude.valueChanged.connect(lambda x: self.update_amplitude(x))
        self.wave.currentIndexChanged.connect(lambda x: self.update_wavetable(x))
        self.frequency.valueChanged.connect(lambda x: self.update_frequency(x))
        self.absolute.pressed.connect(lambda: self.update_absolute())

        self.h_layout.addWidget(self.amplitude)
        self.h_layout.addWidget(QLabel("×"))
        self.h_layout.addWidget(self.wave)
        self.h_layout.addWidget(QLabel("("))
        self.h_layout.addWidget(self.frequency)
        self.h_layout.addWidget(self.absolute)
        self.h_layout.addWidget(QLabel(")"))

        self.setLayout(self.h_layout)

    def update_amplitude(self, amplitude):
        tmp = list(self.main_synth.volume[self.item])
        tmp[self.osc_id] = amplitude
        self.main_synth.update_volume(self.item, tuple(tmp))

    def update_wavetable(self, wavetable):
        tmp = list(self.main_synth.outputs[self.item])
        tmp[self.osc_id] = wavetable - 1
        self.main_synth.update_output_wavetable(self.item, tuple(tmp))
        self.update_items()

    def update_frequency(self, frequency):
        tmp = list(self.main_synth.frequency[self.item])
        tmp[self.osc_id] = frequency
        self.main_synth.update_frequency(self.item, tuple(tmp))

    def update_absolute(self):
        tmp = list(self.main_synth.absolute[self.item])
        tmp[self.osc_id] = not tmp[self.osc_id]
        self.main_synth.update_frequency_type(self.item, tuple(tmp))
        self.update_items()

    def update_items(self):
        self.amplitude.setValue(self.main_synth.volume[self.item][self.osc_id])
        self.wave.setCurrentIndex(self.main_synth.outputs[self.item][self.osc_id] + 1)
        self.frequency.setValue(self.main_synth.frequency[self.item][self.osc_id])
        self.absolute.setText("Hz" if self.main_synth.absolute[self.item][self.osc_id] else "f")


class WaveModulationLayout(QHBoxLayout):
    def __init__(self, main_synth: synth.Synth, item: int):
        super().__init__()
        self.item = item
        self.main_synth = main_synth

        self.modulation = QComboBox()
        self.modulation.addItems(["Mix", "AM", "Ring", "FM", "PM"])
        self.modulation.setFixedWidth(50)
        self.modulation.currentIndexChanged.connect(lambda x: self.update_modulation(x))

        self.wave_1 = SingleOscillator(main_synth, item, 0)
        self.wave_2 = SingleOscillator(main_synth, item, 1)

        self.addWidget(self.wave_1)
        self.addWidget(self.modulation)
        self.addWidget(self.wave_2)
        self.update_items()

    def update_modulation(self, modulation):
        self.main_synth.modulations[self.item] = modulation
        self.update_items()

    def update_items(self):
        self.wave_1.update_items()
        self.modulation.setCurrentIndex(self.main_synth.modulations[self.item])
        self.wave_2.update_items()


class EnvelopeFilterLayout(QHBoxLayout):
    def __init__(self, main_synth: synth.Synth, item: int):
        super().__init__()
        self.main_synth = main_synth
        self.item = item

        self.envelope_box = QFrame()
        self.filter_box = QFrame()
        self.envelope_layout = QHBoxLayout()
        self.filter_layout = QHBoxLayout()

        self.envelope_params = [QDoubleSpinBox() for _ in range(4)]
        self.envelope_params[0].valueChanged.connect(lambda x: self.update_envelope_item(0, x))
        self.envelope_params[1].valueChanged.connect(lambda x: self.update_envelope_item(1, x))
        self.envelope_params[2].valueChanged.connect(lambda x: self.update_envelope_item(2, x))
        self.envelope_params[3].valueChanged.connect(lambda x: self.update_envelope_item(3, x))

        self.filter_params = [QDoubleSpinBox() for _ in range(2)]

        filter_labels = ["Low", "High"]
        self.filter_enabled = [QPushButton(filter_labels[i]) for i in range(2)]
        for i in range(2):
            self.filter_params[i].setDecimals(0)
            self.filter_params[i].setMaximum(20000.0)
            self.filter_params[i].setMinimum(1.0)
            self.filter_enabled[i].setCheckable(True)
        self.filter_enabled[0].clicked.connect(lambda x: self.update_filter_item(0, checked=x))
        self.filter_params[0].valueChanged.connect(lambda x: self.update_filter_item(0, value=x))
        self.filter_enabled[1].clicked.connect(lambda x: self.update_filter_item(1, checked=x))
        self.filter_params[1].valueChanged.connect(lambda x: self.update_filter_item(1, value=x))

        self.update_values()

        self.envelope_layout.addWidget(QLabel("Env:"))
        envelope_labels = 'ADSR'
        for i in range(4):
            label = QLabel(envelope_labels[i])
            self.envelope_layout.addWidget(label)
            self.envelope_layout.addWidget(self.envelope_params[i])
        self.filter_layout.addWidget(QLabel("Filt:"))
        for i in range(2):
            self.filter_layout.addWidget(self.filter_enabled[i])
            self.filter_layout.addWidget(self.filter_params[i])

        self.envelope_box.setLayout(self.envelope_layout)
        self.filter_box.setLayout(self.filter_layout)

        self.addWidget(self.envelope_box)
        self.addWidget(self.filter_box)

    def update_envelope_item(self, item, new):
        tmp = list(self.main_synth.envelope[self.item])
        tmp[item] = new
        self.main_synth.update_envelope(self.item, tuple(tmp))

    def update_filter_item(self, item, checked=None, value=None):
        if checked is None:
            checked = self.filter_enabled[item].isChecked()
        update = False
        if value is None:
            value = self.filter_params[item].value()
            update = True
        new = value if checked else None
        tmp = list(self.main_synth.filter_parameters[self.item])
        tmp[item] = new
        self.main_synth.update_filters(self.item, tuple(tmp))
        if update:
            self.update_values()

    def update_values(self):
        tmp = list(self.main_synth.envelope[self.item])
        for i in range(4):
            self.envelope_params[i].setValue(tmp[i])
        tmp = list(self.main_synth.filter_parameters[self.item])
        for i in range(2):
            if tmp[i] is None:
                self.filter_enabled[i].setChecked(False)
            else:
                self.filter_enabled[i].setChecked(True)
                self.filter_params[i].setValue(tmp[i])


class OutputLayout(QVBoxLayout):
    def __init__(self, main_synth: synth.Synth):
        super().__init__()
        self.main_synth = main_synth
        for i in range(4):
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(
                "QFrame { background-color: #888888; }"
            )
            self.addWidget(sep)
            self.addLayout(WaveModulationLayout(self.main_synth, i))
            self.addLayout(EnvelopeFilterLayout(self.main_synth, i))


class SynthUI(QWidget):
    def __init__(self, main_synth: synth.Synth):
        super().__init__()
        self.main_synth = main_synth
        self.setWindowTitle("4W4O Python Edition")
        self.setLayout(OutputLayout(self.main_synth))

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, "#101010")  # Black background
        palette.setColor(QPalette.WindowText, "#FFFFFF")  # White text
        self.setPalette(palette)

        self.setStyleSheet(
            "QWidget { background-color: #000000; color: #FFFFFF; }"
            "QDoubleSpinBox { color: #FFFFFF; background-color: #1A1A1A; border: 1px solid #333333; }"
            "QComboBox { color: #FFFFFF; background-color: #1A1A1A; border: 1px solid #333333; }"
            "QPushButton { color: #FFFFFF; background-color: #222222; border: 1px solid #444444; }"
            "QPushButton:checked { color: #FFFFFF; background-color: #777777; border: 1px solid #444444; }"
            "QFrame { color: #FFFFFF; background-color: #111111; border: 1px solid #444444; }"
            "QLabel { background-color: #111111; color: #FFFFFF; border: 0px; }"
        )

        self.show()
        self.setFixedSize(self.size())


if __name__ == '__main__':
    import numpy

    app = QApplication(sys.argv)
    s = synth.Synth()

    s.update_wavetable(0, numpy.arange(128) / 64 - 1)
    for i in range(4):
        s.update_output_wavetable(i, (0, -1))
        s.update_volume(i, (0.2, 0))
        s.update_envelope(i, (0.2, 0.0, 1.0, 0.2))
        s.update_filters(i, (5000, None))
        s.update_frequency(i, (1.006 - i * 0.003, 1.0))
        s.update_frequency_type(i, (False, False))
        s.update_modulation(i, 0)

    window = SynthUI(s)
    app.exec()
