import math
import numpy
from functools import partial
from collections import deque
from PySide6.QtCore import *
from PySide6.QtGui import *
import soundfile
from synth import synth
from PySide6.QtWidgets import *


class SynthEditor:
    def __init__(self, synthesizer: synth.Synth, maximum_undo_depth: int = 50):
        self.synth = synthesizer
        self.undo_stack = deque(maxlen=maximum_undo_depth)
        self.redo_stack = []

    def record_state(self):
        self.undo_stack.append(self.synth.output_state())
        self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) > 0:
            self.redo_stack.append(self.synth.output_state())
            self.synth.read_from_state(self.undo_stack.pop())

    def redo(self):
        if len(self.redo_stack) > 0:
            self.undo_stack.append(self.synth.output_state())
            self.synth.read_from_state(self.redo_stack.pop())


class SingleOscillator(QFrame):
    def __init__(self, main_synth: SynthEditor, item: int, osc_id: int):
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
        self.frequency.setMaximum(20000)
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
        tmp = list(self.main_synth.synth.volume[self.item])
        tmp[self.osc_id] = amplitude
        self.main_synth.record_state()
        self.main_synth.synth.update_volume(self.item, tuple(tmp))

    def update_wavetable(self, wavetable):
        tmp = list(self.main_synth.synth.outputs[self.item])
        tmp[self.osc_id] = wavetable - 1
        self.main_synth.record_state()
        self.main_synth.synth.update_output_wavetable(self.item, tuple(tmp))
        self.update_items()

    def update_frequency(self, frequency):
        tmp = list(self.main_synth.synth.frequency[self.item])
        tmp[self.osc_id] = frequency
        self.main_synth.record_state()
        self.main_synth.synth.update_frequency(self.item, tuple(tmp))

    def update_absolute(self):
        tmp = list(self.main_synth.synth.absolute[self.item])
        tmp[self.osc_id] = not tmp[self.osc_id]
        self.main_synth.record_state()
        self.main_synth.synth.update_frequency_type(self.item, tuple(tmp))
        self.update_items()

    def update_items(self):
        self.amplitude.blockSignals(True)
        self.wave.blockSignals(True)
        self.frequency.blockSignals(True)
        self.absolute.blockSignals(True)
        self.amplitude.setValue(self.main_synth.synth.volume[self.item][self.osc_id])
        self.wave.setCurrentIndex(self.main_synth.synth.outputs[self.item][self.osc_id] + 1)
        self.frequency.setValue(self.main_synth.synth.frequency[self.item][self.osc_id])
        self.absolute.setText("Hz" if self.main_synth.synth.absolute[self.item][self.osc_id] else "f")
        self.amplitude.blockSignals(False)
        self.wave.blockSignals(False)
        self.frequency.blockSignals(False)
        self.absolute.blockSignals(False)


class WaveModulationLayout(QHBoxLayout):
    def __init__(self, main_synth: SynthEditor, item: int):
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
        self.main_synth.record_state()
        self.main_synth.synth.modulations[self.item] = modulation
        self.update_items()

    def update_items(self):
        self.modulation.blockSignals(True)
        self.wave_1.update_items()
        self.modulation.setCurrentIndex(self.main_synth.synth.modulations[self.item])
        self.wave_2.update_items()
        self.modulation.blockSignals(False)


class EnvelopeFilterLayout(QHBoxLayout):
    def __init__(self, main_synth: SynthEditor, item: int):
        super().__init__()
        self.main_synth = main_synth
        self.item = item

        self.envelope_box = QFrame()
        self.filter_box = QFrame()
        self.envelope_layout = QHBoxLayout()
        self.filter_layout = QHBoxLayout()

        self.envelope_params = [QDoubleSpinBox() for _ in range(4)]
        for i in range(4):
            self.envelope_params[i].valueChanged.connect(partial(self.update_envelope_item, i))

        self.filter_params = [QDoubleSpinBox() for _ in range(2)]

        filter_labels = ["Low", "High"]
        self.filter_enabled = [QPushButton(filter_labels[i]) for i in range(2)]
        for i in range(2):
            self.filter_params[i].setDecimals(0)
            self.filter_params[i].setMaximum(20000.0)
            self.filter_params[i].setMinimum(1.0)
            self.filter_enabled[i].setCheckable(True)
            self.filter_enabled[i].clicked.connect(partial(self.update_filter_item, i, value=None))
            self.filter_params[i].valueChanged.connect(partial(self.update_filter_item, i, None))

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
        tmp = list(self.main_synth.synth.envelope[self.item])
        tmp[item] = new
        self.main_synth.record_state()
        self.main_synth.synth.update_envelope(self.item, tuple(tmp))

    def update_filter_item(self, item, checked=None, value=None):
        if checked is None:
            checked = self.filter_enabled[item].isChecked()
        update = False
        if value is None:
            value = self.filter_params[item].value()
            update = True
        new = value if checked else None
        tmp = list(self.main_synth.synth.filter_parameters[self.item])
        tmp[item] = new
        self.main_synth.record_state()
        self.main_synth.synth.update_filters(self.item, tuple(tmp))
        if update:
            self.update_values()

    def update_values(self):
        for i in range(4):
            self.envelope_params[i].blockSignals(True)
        for i in range(2):
            self.filter_params[i].blockSignals(True)
        tmp = list(self.main_synth.synth.envelope[self.item])
        for i in range(4):
            self.envelope_params[i].setValue(tmp[i])
        tmp = list(self.main_synth.synth.filter_parameters[self.item])
        for i in range(2):
            if tmp[i] is None:
                self.filter_enabled[i].setChecked(False)
            else:
                self.filter_enabled[i].setChecked(True)
                self.filter_params[i].setValue(tmp[i])
        for i in range(4):
            self.envelope_params[i].blockSignals(False)
        for i in range(2):
            self.filter_params[i].blockSignals(False)


def clamp(a, b, value):
    return max(a, min(b, value))


class WaveDisplay(QFrame):
    def __init__(self, main_synth: SynthEditor, item: int):
        super().__init__()
        self.main_synth = main_synth
        self.item = item
        self.samples = self.main_synth.synth.wavetables[item].tolist()
        self.prev_x = None
        self.prev_y = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        points = [QPoint(round(i * self.width() / 128),
                         round(self.height() / 2 - self.samples[i] * self.height() * 0.45))
                  for i in range(128)]
        painter.setPen(QPen(Qt.white, 1.5))
        painter.drawPolyline(points)
        painter.end()

    def mousePressEvent(self, event, /):
        x = clamp(0, 127, round(event.position().x() / self.width() * 128))
        y = clamp(-1.0, 1.0, (self.height() / 2 - event.position().y()) / (self.height() * 0.45))
        self.samples[x] = y
        self.prev_x = x
        self.prev_y = y
        self.update()

    def mouseMoveEvent(self, event, /):
        x = clamp(0, 127, round(event.position().x() / self.width() * 128))
        y = clamp(-1.0, 1.0, (self.height() / 2 - event.position().y()) / (self.height() * 0.45))
        if self.prev_x is not None:
            movement = 0
            if self.prev_x < x:
                movement = 1
            elif self.prev_x > x:
                movement = -1
            if movement != 0:
                px = self.prev_x
                py = self.prev_y
                step = (y - py)/abs(x - px)
                while px != x:
                    px += movement
                    py += step
                    self.samples[px] = py
        self.samples[x] = y
        self.prev_x = x
        self.prev_y = y
        self.update()

    def mouseReleaseEvent(self, event, /):
        self.main_synth.record_state()
        self.main_synth.synth.update_wavetable(self.item, numpy.array(self.samples))
        self.update_waveform()
        self.prev_x = None
        self.prev_y = None
        self.update()

    def update_waveform(self):
        self.samples = self.main_synth.synth.wavetables[self.item].tolist()


class SingleWavetable(QWidget):
    def __init__(self, main_synth: SynthEditor, item: int):
        super().__init__()
        self.main_synth = main_synth
        self.item = item
        self.waveform = WaveDisplay(self.main_synth, self.item)
        self.button_layout_1 = QHBoxLayout()
        self.setStyleSheet("QLabel { background-color: #000000; }")
        self.button_layout_1.addWidget(QLabel(f"W{item + 1}"))
        waveform_types = ['ZRO', 'SIN', 'SQR', 'TRI', 'SAW']
        self.buttons = [QPushButton(waveform_types[i]) for i in range(5)]
        for i in range(5):
            self.buttons[i].clicked.connect(partial(self.change_waveform, i))
            self.button_layout_1.addWidget(self.buttons[i])

        self.sample_button = QPushButton('Sample')
        self.sample_button.clicked.connect(lambda x: self.get_samples())
        self.spectrum_button = QPushButton('Spectrum')
        self.spectrum_button.clicked.connect(lambda x: self.get_spectrum())
        self.file_button = QPushButton('File')
        self.file_button.clicked.connect(lambda x: self.get_file())

        self.button_layout_2 = QHBoxLayout()
        self.button_layout_2.addWidget(self.sample_button)
        self.button_layout_2.addWidget(self.spectrum_button)
        self.button_layout_2.addWidget(self.file_button)

        self.waveform.setFixedHeight(80)

        layout = QVBoxLayout()
        layout.addLayout(self.button_layout_1)
        layout.addLayout(self.button_layout_2)
        layout.addWidget(self.waveform)

        self.setLayout(layout)

    def change_waveform(self, wave_type):
        if wave_type == 0:
            wave = numpy.zeros((128,), dtype=numpy.float32)
        elif wave_type == 1:
            wave = numpy.sin(numpy.arange(128, dtype=numpy.float32) / 64 * numpy.pi)[:128]
        elif wave_type == 2:
            wave = 1 - numpy.floor(numpy.arange(128, dtype=numpy.float32) / 64)[:128] * 2
        elif wave_type == 3:
            wave = numpy.array([0, 1, 0, -1])
        elif wave_type == 4:
            wave = numpy.arange(128, dtype=numpy.float32) / 64 - 1
        else:
            raise ValueError("wave_type must be an integer between 0 and 4.")
        self.main_synth.record_state()
        self.main_synth.synth.update_wavetable(self.item, wave)
        self.waveform.update_waveform()

    def get_samples(self):
        text, ok = QInputDialog.getMultiLineText(self, f"W{self.item+1} Samples", "Enter numbers separated by commas:")
        if not ok:
            return
        try:
            wave = numpy.array(list(map(float, text.split(","))), dtype=numpy.float32)
            self.main_synth.record_state()
            self.main_synth.synth.update_wavetable(self.item, numpy.clip(wave, -1, 1))
            self.waveform.update_waveform()
        except ValueError:
            QMessageBox.critical(self, "Error", "Error processing input.")

    def get_spectrum(self):
        text, ok = QInputDialog.getMultiLineText(self, f"W{self.item+1} Spectrum", "Enter numbers separated by commas:")
        if not ok:
            return
        try:
            spectrum = list(map(float, text.split(",")))
            wave = numpy.zeros((128,), dtype=numpy.float32)
            arange = numpy.arange(128, dtype=numpy.float32) / 64 * numpy.pi
            for i in range(len(spectrum)):
                wave += numpy.sin(arange * (i+1)) * spectrum[i]
            self.main_synth.record_state()
            self.main_synth.synth.update_wavetable(self.item, numpy.clip(wave, -1, 1))
            self.waveform.update_waveform()
        except ValueError:
            QMessageBox.critical(self, "Error", "Error processing input.")

    def get_file(self):
        file, ok = QFileDialog.getOpenFileUrl(self, f"Import W{self.item+1} from file", filter="WAV files (*.wav)")
        if not ok:
            return
        try:
            path = file.toLocalFile()
            data, sr = soundfile.read(path, always_2d=True)
            data = numpy.asarray(data, dtype=numpy.float32)
            wave = numpy.mean(data, axis=1)
            factor = max(numpy.max(wave), -numpy.min(wave))
            if factor != 0:
                wave /= factor
            self.main_synth.record_state()
            self.main_synth.synth.update_wavetable(self.item, numpy.clip(wave, -1, 1))
            self.waveform.update_waveform()
        except Exception as e:
            print(repr(e))
            QMessageBox.critical(self, "Error", f"Error processing input:\n{e}")


class WavetableLayout(QVBoxLayout):
    def __init__(self, main_synth: SynthEditor):
        super().__init__()
        self.main_synth = main_synth
        self.wavetable_elements = [SingleWavetable(self.main_synth, i) for i in range(4)]
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        for i in range(2):
            tmp = QHBoxLayout()
            tmp.setContentsMargins(0, 0, 0, 0)
            tmp.setSpacing(0)
            tmp.addWidget(self.wavetable_elements[i * 2])
            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)
            sep.setStyleSheet(
                "QFrame { background-color: #888888; }"
            )
            tmp.addWidget(sep)
            tmp.addWidget(self.wavetable_elements[i * 2 + 1])
            self.addLayout(tmp)
            if i == 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setStyleSheet(
                    "QFrame { background-color: #888888; }"
                )
                self.addWidget(sep)

    def update_values(self):
        for i in range(4):
            self.wavetable_elements[i].waveform.blockSignals(True)
        for i in range(4):
            self.wavetable_elements[i].waveform.update_waveform()
        for i in range(4):
            self.wavetable_elements[i].waveform.blockSignals(False)


class OutputLayout(QVBoxLayout):
    def __init__(self, main_synth: SynthEditor):
        super().__init__()
        self.main_synth = main_synth
        self.wave_mod_layouts = [WaveModulationLayout(self.main_synth, i) for i in range(4)]
        self.env_filt_layouts = [EnvelopeFilterLayout(self.main_synth, i) for i in range(4)]
        for i in range(4):
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(
                "QFrame { background-color: #888888; }"
            )
            self.addWidget(sep)
            self.addLayout(self.wave_mod_layouts[i])
            self.addLayout(self.env_filt_layouts[i])

    def update_values(self):
        for i in range(4):
            self.wave_mod_layouts[i].update_items()
            self.env_filt_layouts[i].update_values()


class Key(QPushButton):
    def __init__(self, main_synth: SynthEditor, note: int):
        super().__init__()
        self.note = note
        self.setFixedHeight(50)
        self.main_synth = main_synth
        if note % 12 in [1, 3, 6, 8, 10]:
            self.setStyleSheet(
                "QPushButton { background-color: #222222; border: 1px solid #444444; }"
                "QPushButton:hover { background-color: #444444; }"
            )
        else:
            self.setStyleSheet(
                "QPushButton { background-color: #AAAAAA; border: 1px solid #444444; }"
                "QPushButton:hover { background-color: #CCCCCC; }"
            )
        self.pressed.connect(self.key_down)
        self.released.connect(self.key_up)

    def key_down(self):
        self.main_synth.synth.start_note(self.note)

    def key_up(self):
        self.main_synth.synth.stop_note(self.note)


class Player(QFrame):
    def __init__(self, main_synth: SynthEditor):
        super().__init__()
        self.main_synth = main_synth

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for i in range(48, 84):
            layout.addWidget(Key(self.main_synth, i))
        self.setLayout(layout)


class KeyPressFilter(QObject):
    def __init__(self, parent, main_synth: SynthEditor):
        super().__init__(parent=parent)
        self.key_map = {
            90: 48, 83: 49, 88: 50, 68: 51, 67: 52, 86: 53, 71: 54, 66: 55, 72: 56, 78: 57, 74: 58, 77: 59,
            44: 60, 76: 61, 46: 62, 59: 63, 47: 64,
            81: 60, 50: 61, 87: 62, 51: 63, 69: 64, 82: 65, 53: 66, 84: 67, 54: 68, 89: 69, 55: 70, 85: 71,
            73: 72, 57: 73, 79: 74, 48: 75, 80: 76, 91: 77, 61: 78, 93: 79
        }
        self.note_to_freq = lambda x: 440.0 * 2.0**((x-69)/12)
        self.main_synth = main_synth

    def eventFilter(self, widget, event: QEvent):
        if event.type() == QEvent.KeyPress and not event.isAutoRepeat():
            if event.modifiers():
                return False
            kei = event.key()
            if kei in self.key_map:
                self.main_synth.synth.start_note(self.key_map[kei])
        elif event.type() == QEvent.KeyRelease and not event.isAutoRepeat():
            if event.modifiers():
                return False
            kei = event.key()
            if kei in self.key_map:
                self.main_synth.synth.stop_note(self.key_map[kei])
        return False


class MenuBar(QHBoxLayout):
    def __init__(self, main_synth: SynthEditor, update_method):
        super().__init__()
        self.main_synth = main_synth
        self.update_method = update_method

        self.new_button = QPushButton("New")
        self.open_button = QPushButton("Open")
        self.save_button = QPushButton("Save")
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")

        self.save_button.clicked.connect(self.save)
        self.open_button.clicked.connect(self.open)
        self.new_button.clicked.connect(self.new)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

        self.addWidget(self.new_button)
        self.addWidget(self.open_button)
        self.addWidget(self.save_button)
        self.addWidget(self.undo_button)
        self.addWidget(self.redo_button)

        self.default = {
            'wavetables': [[math.sin(i / 64 * math.pi) for i in range(128)],
                           [0.0 for _ in range(128)], [0.0 for _ in range(128)], [0.0 for _ in range(128)]],
            'outputs': [[0, -1], [-1, -1], [-1, -1], [-1, -1]], 'volume': [[0.2, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
            'envelope': [[0.0, 0.0, 1.0, 0.0] for _ in range(4)], 'filter_parameters': [[None, None] for _ in range(4)],
            'frequency': [[1.0, 1.0] for _ in range(4)], 'absolute': [[False, False] for _ in range(4)],
            'modulations': [0 for _ in range(4)]
        }

    def save(self):
        file, ok = QFileDialog.getSaveFileUrl(self.save_button, f"Save Configuration", filter="JSON files (*.json)")
        if not ok:
            return
        try:
            path = file.toLocalFile()
            self.main_synth.synth.output_file(path)
            self.update_method()
        except Exception as e:
            print(repr(e))
            QMessageBox.critical(self.save_button, "Error", f"Error processing file:\n{e}")

    def open(self):
        file, ok = QFileDialog.getOpenFileUrl(self.open_button, f"Load Configuration", filter="JSON files (*.json)")
        if not ok:
            return
        try:
            path = file.toLocalFile()
            self.main_synth.record_state()
            self.main_synth.synth.read_from_file(path)
            self.update_method()
        except Exception as e:
            print(repr(e))
            QMessageBox.critical(self.open_button, "Error", f"Error processing file:\n{e}")

    def new(self):
        self.main_synth.synth.read_from_state(self.default)
        self.update_method()

    def undo(self):
        self.main_synth.undo()
        self.update_method()

    def redo(self):
        self.main_synth.redo()
        self.update_method()


class SynthUI(QWidget):
    def __init__(self, main_synth: SynthEditor):
        super().__init__()
        self.main_synth = main_synth
        self.setWindowTitle("4W4O Python Edition")

        layout = QVBoxLayout()
        self.menu_bar = MenuBar(self.main_synth, self.update_everything)
        layout.addLayout(self.menu_bar)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("QFrame { background-color: #888888; }")
        layout.addWidget(sep)
        self.wavetable_layout = WavetableLayout(self.main_synth)
        layout.addLayout(self.wavetable_layout)
        self.output_layout = OutputLayout(self.main_synth)
        layout.addLayout(self.output_layout)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("QFrame { background-color: #888888; }")
        layout.addWidget(sep)
        self.player = Player(self.main_synth)
        layout.addWidget(self.player)
        self.setLayout(layout)

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
            "QPushButton:hover,QComboBox:hover,QDoubleSpinBox:hover { background-color: #444444; }"
            "QPushButton:checked { border: 2px solid #EEEEEE; }"
            "QFrame { color: #FFFFFF; background-color: #111111; border: 1px solid #444444; }"
            "QLabel { background-color: #111111; color: #FFFFFF; border: 0px; }"
        )

        self.setFixedSize(650, 800)
        self.eventFilter = KeyPressFilter(self, self.main_synth)
        self.installEventFilter(self.eventFilter)
        self.menu_bar.new()

    def update_everything(self):
        self.blockSignals(True)
        self.wavetable_layout.update_values()
        self.output_layout.update_values()
        self.blockSignals(False)

