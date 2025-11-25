import numpy
from scipy import signal
import pyaudio
import threading

MIX = 0
AMPLITUDE = 1
RING = 2
FREQUENCY = 3
PHASE = 4


class Synth:
    def __init__(self, sample_rate=44100, buffer_size=1024):
        """
        A simple synthesizer that can play multiple frequencies simultaneously.

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
        self.outputs: list[tuple[int, int]] = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]
        self.volume: list[tuple[int | float, int | float]] = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.envelope: list[tuple[int | float, int | float, int | float, int | float]] \
            = [(0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0)]
        self.filter_parameters: list[tuple[int | float | None, int | float | None]] \
            = [(None, None), (None, None), (None, None), (None, None)]
        self.filters = [[(None, None), (None, None)],
                        [(None, None), (None, None)],
                        [(None, None), (None, None)],
                        [(None, None), (None, None)]]
        self.frequency: list[tuple[int | float, int | float]] = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
        self.absolute: list[tuple[bool, bool]] = [(False, False), (False, False), (False, False), (False, False)]
        self.modulations: list[int] = [0, 0, 0, 0]
        self._setup_filters()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.buffer_size,
            stream_callback=self._audio_callback
        )

    def _setup_filters(self):
        for i in range(4):
            if self.filter_parameters[i][0] is not None:
                self.filters[i][0] = signal.butter(2, self.filter_parameters[i][0], fs=self.sample_rate,
                                                   btype='lowpass', analog=False)
            if self.filter_parameters[i][1] is not None:
                self.filters[i][1] = signal.butter(2, self.filter_parameters[i][1], fs=self.sample_rate,
                                                   btype='highpass', analog=False)

    def update_wavetable(self, item: int, data: numpy.ndarray):
        with self.lock:
            self.wavetables[item] = data

    def update_filters(self, item: int, data: tuple[int | float | None, int | float | None]):
        with self.lock:
            self.filter_parameters[item] = data
            self._setup_filters()

    def update_envelope(self, item: int, data: tuple[int | float, int | float, int | float, int | float]):
        with self.lock:
            self.envelope[item] = data

    def update_modulation(self, item: int, data: int):
        with self.lock:
            self.modulations[item] = data

    def update_frequency(self, item: int, data: tuple[int | float, int | float]):
        with self.lock:
            self.frequency[item] = data

    def update_frequency_type(self, item: int, data: tuple[bool, bool]):
        with self.lock:
            self.absolute[item] = data

    def update_output_wavetable(self, item: int, data: tuple[int, int]):
        with self.lock:
            self.outputs[item] = data

    def update_volume(self, item: int, data: tuple[int | float, int | float]):
        with self.lock:
            self.volume[item] = data

    def _get_wavetable(self, table_num, t):
        f = t * 128
        i, p = numpy.modf(f)
        p = numpy.int8(p) & 127
        np = (p + 1) & 127
        return self.wavetables[table_num][p] * (1 - i) + self.wavetables[table_num][np] * i

    def _get_envelope(self, item, t, end_time=None):
        e = self.envelope[item]
        offset = 1/e[1] if e[1] > 0 else 2147483647
        start = 1/e[0] if e[0] > 0 else 2147483647
        result = numpy.where(t < e[0], t * start, numpy.maximum(1 - (1 - e[2]) * (t - e[0]) * offset, e[2]))
        if end_time is not None:
            end_time /= self.sample_rate
            end_status = end_time * start if end_time < e[0] else max(1 - (1 - e[2]) * (end_time - e[0]) * offset, e[2])
            if e[3] > 0:
                result = numpy.where(t > end_time, end_status * (1 - (t - end_time) / e[3]), result)
                result = numpy.maximum(result, 0)
            else:
                result = numpy.where(t > end_time, 0, result)
        return result

    def _generate_output(self, item, freq, t, frame_count):
        freq1 = self.frequency[item][0] if self.absolute[item][0] else freq * self.frequency[item][0]
        freq2 = self.frequency[item][1] if self.absolute[item][1] else freq * self.frequency[item][1]
        if self.outputs[item][0] == -1:
            return numpy.zeros(frame_count, dtype=numpy.float32)
        if self.outputs[item][1] == -1:
            total = self._get_wavetable(self.outputs[item][0], t * freq1) * self.volume[item][0]
        else:
            w2 = self._get_wavetable(self.outputs[item][1], t * freq2) * self.volume[item][1]
            if self.modulations[item] == MIX:
                w1 = self._get_wavetable(self.outputs[item][0], t * freq1) * self.volume[item][0]
                total = (w1 + w2) / 2
            elif self.modulations[item] == AMPLITUDE:
                w1 = self._get_wavetable(self.outputs[item][0], t * freq1) * self.volume[item][0]
                total = w1 * (1 + w2)
            elif self.modulations[item] == RING:
                w1 = self._get_wavetable(self.outputs[item][0], t * freq1) * self.volume[item][0]
                total = w1 * w2
            elif self.modulations[item] == PHASE:
                phase = t + w2 / freq1
                w1 = self._get_wavetable(self.outputs[item][0], phase * freq1) * self.volume[item][0]
                total = w1
            elif self.modulations[item] == FREQUENCY:
                instantaneous_freq = freq1 + freq1 * w2
                dt = 1.0 / self.sample_rate
                phase_increments = instantaneous_freq * dt
                accumulated_phase = numpy.cumsum(phase_increments) + self.playing_frequencies[freq][2]
                self.playing_frequencies[freq][2] = accumulated_phase[-1]
                w1 = self._get_wavetable(self.outputs[item][0], accumulated_phase) * self.volume[item][0]
                total = w1
            else:
                total = numpy.zeros(frame_count, dtype=numpy.float32)
        total = total * self._get_envelope(item, t, self.playing_frequencies[freq][1])
        if self.filter_parameters[item][0] is not None:
            total = signal.filtfilt(*self.filters[item][0], total)
        if self.filter_parameters[item][1] is not None:
            total = signal.filtfilt(*self.filters[item][1], total)
        return total

    def _generate_frequency(self, freq, data, frame_count):
        phase = data[0]
        t = (numpy.arange(frame_count) + phase) / self.sample_rate
        audio_data = numpy.zeros(frame_count, dtype=numpy.float32)
        for i in range(4):
            audio_data += self._generate_output(i, freq, t, frame_count)
        self.playing_frequencies[freq][0] = phase + frame_count
        return audio_data * self.playing_frequencies[freq][3]

    def generate_samples(self, frame_count):
        if not self.playing_frequencies:
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

    def start_frequency(self, frequency, volume=1.0):
        """
        Start playing at the specified frequency.

        Args:
            frequency: Frequency in Hz (can be any float value, including microtonal)
        """
        with self.lock:
            if frequency not in self.active_frequencies:
                self.active_frequencies.add(frequency)
                self.playing_frequencies[frequency] = [0, None, 0, volume]
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
            return list(self.active_frequencies)

    def stop_all(self):
        """
        Stop all playing frequencies.
        """
        with self.lock:
            self.active_frequencies.clear()
            self.playing_frequencies.clear()
            print("Stopped all frequencies")

    def stop_stream(self):
        self.stream.stop_stream()

    def start_stream(self):
        self.stream.start_stream()

    def close(self):
        """
        Close the synthesizer and clean up resources.
        """
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("Synthesizer closed")
