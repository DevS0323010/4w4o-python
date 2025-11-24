import numpy
import scipy
import pyaudio
import threading


class Synth:
    def __init__(self, sample_rate=44100, buffer_size=1024):
        """
        A simple sine wave synthesizer that can play multiple frequencies simultaneously.

        Args:
            sample_rate: Audio sample rate in Hz (default 44100)
            buffer_size: Size of audio buffer (default 1024)
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.active_frequencies = set()
        self.playing_frequencies = {}
        self.lock = threading.Lock()
        self.p = pyaudio.PyAudio()
        self.wavetables = [numpy.zeros((128,), dtype=numpy.float32) for _ in range(4)]
        self.outputs = [0, -1, -1, -1]
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.buffer_size,
            stream_callback=self._audio_callback
        )

    def update_wavetable(self, item, data: numpy.ndarray):
        self.wavetables[item] = data

    def _generate_output(self, item, freq, t, frame_count):
        if self.outputs[item] == -1:
            return numpy.zeros(frame_count, dtype=numpy.float32)
        f = t * freq * 128
        i, p = numpy.modf(f)
        p = numpy.int8(p) & 127
        np = (p + 1) & 127
        return self.wavetables[self.outputs[item]][p]*(1-i) + self.wavetables[self.outputs[item]][np]*i

    def _generate_frequency(self, freq, data, frame_count):
        phase = data[0]
        t = (numpy.arange(frame_count) + phase) / self.sample_rate
        audio_data = numpy.zeros(frame_count, dtype=numpy.float32)
        for i in range(4):
            audio_data += self._generate_output(i, freq, t, frame_count)
        self.playing_frequencies[freq][0] = phase + frame_count
        return audio_data

    def generate_samples(self, frame_count):
        if not self.active_frequencies:
            audio_data = numpy.zeros(frame_count, dtype=numpy.float32)
        else:
            audio_data = numpy.zeros(frame_count, dtype=numpy.float32)
            for freq, data in self.playing_frequencies.items():
                audio_data += self._generate_frequency(freq, data, frame_count)
        return audio_data

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function that generates audio samples.
        """
        with self.lock:
            audio_data = self.generate_samples(frame_count)

        return audio_data.tobytes(), pyaudio.paContinue

    def start_frequency(self, frequency):
        """
        Start playing at the specified frequency.

        Args:
            frequency: Frequency in Hz (can be any float value, including microtonal)
        """
        with self.lock:
            if frequency not in self.active_frequencies:
                self.active_frequencies.add(frequency)
                self.playing_frequencies[frequency] = [0, 0]
                print(f"Started frequency: {frequency} Hz")
            else:
                print(f"Frequency {frequency} Hz is already playing")

    def stop_frequency(self, frequency):
        """
        Stop playing at the specified frequency.

        Args:
            frequency: Frequency in Hz to stop
        """
        with self.lock:
            if frequency in self.active_frequencies:
                self.active_frequencies.remove(frequency)
                self.playing_frequencies[frequency][1] = self.playing_frequencies[frequency][0]
                print(f"Stopped frequency: {frequency} Hz")
            else:
                print(f"Frequency {frequency} Hz is not currently playing")

    def get_active_frequencies(self):
        """
        Get a list of currently playing frequencies.

        Returns:
            List of active frequencies
        """
        with self.lock:
            return list(self.active_frequencies.keys())

    def stop_all(self):
        """
        Stop all playing frequencies.
        """
        with self.lock:
            self.active_frequencies.clear()
            print("Stopped all frequencies")

    def close(self):
        """
        Close the synthesizer and clean up resources.
        """
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("Synthesizer closed")