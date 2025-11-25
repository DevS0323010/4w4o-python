if __name__ == '__main__':
    import time
    import synth
    import numpy
    s = synth.Synth()
    s.update_wavetable(0, numpy.arange(128) / 64 - 1)

    s.stop_stream()
    s.start_frequency(440)

    print(time.time())
    for i in range(1000):
        s.generate_samples(1024)
    print(time.time())

    s.stop_frequency(440)
    s.start_stream()

    s.start_frequency(440)
    time.sleep(3)
    s.stop_frequency(440)
    time.sleep(1)

    s.stop_all()
    time.sleep(0.5)
    s.close()