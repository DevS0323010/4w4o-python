if __name__ == '__main__':
    import time
    import synth
    import numpy
    s = synth.Synth()
    t = synth.Synth()

    s.update_wavetable(0, numpy.arange(128) / 64 - 1)
    t.update_wavetable(0, numpy.array([0, 1, 0, -1]))

    for i in range(4):
        s.update_output_wavetable(i, (0, -1))
        s.update_volume(i, (0.2, 0))
        s.update_envelope(i, (0.2, 0.0, 1.0, 0.2))
        s.update_filters(i, (5000, None))
        s.update_frequency(i, (1.006-i*0.003, 1.0))
        s.update_frequency_type(i, (False, False))
        s.update_modulation(i, 0)

    t.update_output_wavetable(0, (0, -1))
    t.update_volume(0, (0.25, 0))

    s.stop_stream()
    s.start_frequency(440)

    print(time.time())
    for i in range(1000):
        s.generate_samples(1024)
    print(time.time())

    s.stop_frequency(440)

    s.generate_samples(44100)
    notes = [261.63, 293.66, 329.63, 349.23, 392, 440, 493.88, 523.25]

    s.start_stream()

    for i in notes:
        s.start_frequency(i / 2, 0.3)
        t.start_frequency(i * 1.5, 0.3)
        time.sleep(1)
        s.stop_frequency(i / 2)
        t.stop_frequency(i * 1.5)

    time.sleep(0.5)
    s.stop_all()
    time.sleep(0.5)
    s.close()
