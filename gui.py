import sys

from PySide6.QtGui import QPalette

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


class SingleOutputLayout(QHBoxLayout):
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



class OutputLayout(QVBoxLayout):
    def __init__(self, main_synth: synth.Synth):
        super().__init__()
        self.main_synth = main_synth
        for i in range(4):
            self.addLayout(SingleOutputLayout(self.main_synth, i))


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
