if __name__ == '__main__':
    import time
    import synth
    import numpy
    s = synth.Synth()
    s.update_wavetable(0, numpy.arange(128) / 64 - 1)

    s.start_frequency(220)
    time.sleep(3)

    s.stop_all()
    time.sleep(0.5)
    s.close()