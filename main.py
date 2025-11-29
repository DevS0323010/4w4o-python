from PySide6.QtWidgets import QApplication
import synth
import gui

if __name__ == '__main__':

    app = QApplication()
    s = synth.Synth()
    e = gui.SynthEditor(s)

    window = gui.SynthUI(e)

    window.update_everything()
    window.show()

    app.exec()