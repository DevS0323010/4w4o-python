import numpy
from scipy import signal
import pyaudio
import threading
import json

MIX = 0
AMPLITUDE = 1
RING = 2
FREQUENCY = 3
PHASE = 4


class Synth:
    waveforms: list[numpy.ndarray]
    outputs: list[tuple[int, int]]
    volume: list[tuple[int | float, int | float]]
    envelope: list[tuple[int | float, int | float, int | float, int | float]]
    filter_parameters: list[tuple[int | float | None, int | float | None]]
    frequency: list[tuple[int | float, int | float]]
    absolute: list[tuple[bool, bool]]
    modulations: list[int]

    def __init__(self, sample_rate=44100, buffer_size=1024, from_file=None):
        """
        A simple synthesizer that can play multiple frequencies simultaneously.

        :param sample_rate: Audio sample rate in Hz (default 44100)
        :param buffer_size: Size of audio buffer (default 1024)
        """
        self.filters = [[(None, None), (None, None)],
                        [(None, None), (None, None)],
                        [(None, None), (None, None)],
                        [(None, None), (None, None)]]
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.active_frequencies = set()
        self.playing_frequencies = {}
        self.lock = threading.Lock()
        self.p = pyaudio.PyAudio()
        self.filter_states = {}
        if from_file is None:
            self.wavetables = [numpy.zeros((128,), dtype=numpy.float32) for _ in range(4)]
            self.outputs = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]
            self.volume = [(0, 0), (0, 0), (0, 0), (0, 0)]
            self.envelope = [(0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0)]
            self.filter_parameters = [(None, None), (None, None), (None, None), (None, None)]
            self.frequency = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
            self.absolute = [(False, False), (False, False), (False, False), (False, False)]
            self.modulations = [0, 0, 0, 0]
        else:
            self.read_from_file(from_file)

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
                self.filters[i][0] = signal.butter(8, self.filter_parameters[i][0], fs=self.sample_rate,
                                                   btype='lowpass', analog=False)
            if self.filter_parameters[i][1] is not None:
                self.filters[i][1] = signal.butter(8, self.filter_parameters[i][1], fs=self.sample_rate,
                                                   btype='highpass', analog=False)

    def update_wavetable(self, item: int, data: numpy.ndarray):
        """
        Updates one of the 4 wavetables.

        :param item: The specified item ID (integer from 0 to 3)
        :param data: The new wavetable as an 1D numpy array
        :return: Nothing
        """
        with self.lock:
            if data.ndim != 1:
                raise ValueError("Array must be 1-dimensional.")
            new_data = numpy.interp(
                numpy.linspace(0, len(data), 129),
                numpy.arange(len(data) + 1),
                numpy.append(data, [data[0]])
            )[:128]
            self.wavetables[item] = new_data

    def update_filters(self, item: int, data: tuple[int | float | None, int | float | None]):
        """
        Updates the filters of a specific item of outputs.

        :param item: The specified item ID (integer from 0 to 3)
        :param data: The cutoff frequency of the filters (low pass, high pass), None if no filter is used.
        :return: Nothing
        """
        with self.lock:
            self.filter_parameters[item] = data
            self._setup_filters()

    def update_envelope(self, item: int, data: tuple[int | float, int | float, int | float, int | float]):
        """
        Updates the envelope of a specific item of outputs.

        :param item: The specified item ID (integer from 0 to 3)
        :param data: The new envelope (attack, decay, sustain, release)
        :return: Nothing
        """
        with self.lock:
            self.envelope[item] = data

    def update_modulation(self, item: int, data: int):
        """
        Updates the modulation method of a specific item of outputs.

        :param item: The specified item ID (integer from 0 to 3)
        :param data: An integer (0: mix, 1: amplitude, 2: ring, 3: frequency, 4: phase)
        :return: Nothing
        """
        with self.lock:
            self.modulations[item] = data

    def update_frequency(self, item: int, data: tuple[int | float, int | float]):
        """
        Updates the frequency of a specific item of outputs.

        :param item: The specified item ID (integer from 0 to 3)
        :param data: The new frequencies of 2 oscillators (hertz if absolute, times the base frequency if not)
        :return: Nothing
        """
        with self.lock:
            self.frequency[item] = data

    def update_frequency_type(self, item: int, data: tuple[bool, bool]):
        """
        Updates the frequency type (absolute or not) of a specific item of outputs.

        :param item: The specified item ID (integer from 0 to 3)
        :param data: If the frequency is absolute or not for the 2 oscillators
        :return: Nothing
        """
        with self.lock:
            self.absolute[item] = data

    def update_output_wavetable(self, item: int, data: tuple[int, int]):
        """
        Updates the wavetable id of a specific item of outputs.

        :param item: The specified item ID (integer from 0 to 3)
        :param data: The wavetable ID (integer from 0 to 3) of the 2 oscillators
        :return: Nothing
        """
        with self.lock:
            self.outputs[item] = data

    def update_volume(self, item: int, data: tuple[int | float, int | float]):
        """
        Updates the volume of a specific item of outputs.

        :param item: The specified item ID (integer from 0 to 3)
        :param data: The amplitude of the 2 oscillators
        :return: Nothing
        """
        with self.lock:
            self.volume[item] = data

    def output_state(self):
        """
        Returns the current state of the synth as a JSON-serializable dictionary.

        :return: The synth state data as a dictionary
        """
        data = {
            "wavetables": [table.tolist() for table in self.wavetables],
            "outputs": [list(x) for x in self.outputs],
            "volume": [list(x) for x in self.volume],
            "envelope": [list(x) for x in self.envelope],
            "filter_parameters": [list(x) for x in self.filter_parameters],
            "frequency": [list(x) for x in self.frequency],
            "absolute": [list(x) for x in self.absolute],
            "modulations": self.modulations
        }
        return data

    def read_from_state(self, data):
        """
        Restores the synth state from a dictionary of state data.

        :param data: Dictionary containing synth state data (as returned by output_state)
        :return: Nothing
        """
        self.wavetables = [numpy.array(table) for table in data["wavetables"]]
        self.outputs = [tuple(x) for x in data["outputs"]]
        self.volume = [tuple(x) for x in data["volume"]]
        self.envelope = [tuple(x) for x in data["envelope"]]
        self.filter_parameters = [tuple(x) for x in data["filter_parameters"]]
        self.frequency = [tuple(x) for x in data["frequency"]]
        self.absolute = [tuple(x) for x in data["absolute"]]
        self.modulations = data["modulations"]
        self._setup_filters()

    def output_file(self, filename: str):
        """
        Saves the current synth state to a JSON file.

        :param filename: Path to the output JSON file
        :return: Nothing
        """
        data = self.output_state()
        with open(filename, "w") as f:
            json.dump(data, f)

    def read_from_file(self, filename: str):
        """
        Loads and restores the synth state from a JSON file.

        :param filename: Path to the JSON file containing the state
        :return: Nothing
        """
        with open(filename) as f:
            data = json.load(f)
            self.read_from_state(data)

    def _get_wavetable(self, table_num, t):
        f = t * 128
        i, p = numpy.modf(f)
        p = numpy.int8(p) & 127
        np = (p + 1) & 127
        return self.wavetables[table_num][p] * (1 - i) + self.wavetables[table_num][np] * i

    def _get_envelope(self, item, t, end_time=None):
        e = self.envelope[item]
        offset = 1 / e[1] if e[1] > 0 else 2147483647
        start = 1 / e[0] if e[0] > 0 else 2147483647
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
        if freq not in self.filter_states:
            self.filter_states[freq] = {0:[None, None], 1:[None, None], 2:[None, None], 3:[None, None]}
        if self.filter_parameters[item][0] is not None:
            if freq not in self.filter_states or self.filter_states[freq][item][0] is None:
                self.filter_states[freq][item][0] = signal.lfilter_zi(*self.filters[item][0]) * total[0]
            total, self.filter_states[freq][item][0] = signal.lfilter(
                *self.filters[item][0], total, zi=self.filter_states[freq][item][0]
            )

        if self.filter_parameters[item][1] is not None:
            if freq not in self.filter_states or self.filter_states[freq][item][1] is None:
                self.filter_states[freq][item][1] = signal.lfilter_zi(*self.filters[item][1]) * total[0]
            total, self.filter_states[freq][item][1] = signal.lfilter(
                *self.filters[item][1], total, zi=self.filter_states[freq][item][1]
            )
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
        """
        Generate samples of the playing frequencies.

        :param frame_count: The amount of samples to generate
        :return: The generated samples as a numpy array
        """
        removed_frequencies = []
        if not self.playing_frequencies:
            audio_data = numpy.zeros(frame_count, dtype=numpy.float32)
        else:
            audio_data = numpy.zeros(frame_count, dtype=numpy.float32)
            for freq, data in self.playing_frequencies.items():
                audio_data += self._generate_frequency(freq, data, frame_count)
                if data[1] is None:
                    continue
                end_sample = data[0] - data[1]
                max_release = 0
                for i in range(4):
                    max_release = max(max_release, self.envelope[i][3])
                if end_sample > max_release * self.sample_rate:
                    removed_frequencies.append(freq)
            for freq in removed_frequencies:
                del self.playing_frequencies[freq]
                if freq in self.filter_states:
                    del self.filter_states[freq]
        audio_data = numpy.clip(audio_data, -1.0, 1.0)
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

        :param frequency: Frequency in Hz (can be any float value, including microtonal)
        :param volume: Volume of the note (default 1)
        """
        with self.lock:
            if frequency not in self.active_frequencies:
                self.active_frequencies.add(frequency)
                self.playing_frequencies[frequency] = [0, None, 0, volume]

    def stop_frequency(self, frequency):
        """
        Stop playing at the specified frequency.

        :param frequency: Frequency in Hz to stop
        :return: Nothing
        """
        with self.lock:
            if frequency in self.active_frequencies:
                self.active_frequencies.remove(frequency)
                self.playing_frequencies[frequency][1] = self.playing_frequencies[frequency][0]

    def get_active_frequencies(self):
        """
        Get a list of currently playing frequencies.

        :return: List of active frequencies
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
