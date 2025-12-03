import numpy
from scipy import signal
import sounddevice
import threading
import json
from numba import jit
from numba.types import float32, int32

MIX = 0
AMPLITUDE = 1
RING = 2
FREQUENCY = 3
PHASE = 4

@jit((float32[:], float32[:]), nopython=True, fastmath=True, cache=True)
def _get_wavetable_numba(wavetable, t):
    """Optimized wavetable lookup with linear interpolation."""
    f = t * 128.0
    p = numpy.floor(f).astype(numpy.int32) & 127
    i = f - numpy.floor(f)
    np_idx = (p + 1) & 127
    return (wavetable[p] * (1.0 - i) + wavetable[np_idx] * i).astype(numpy.float32)

@jit((float32[:], float32, float32, float32, float32, float32, float32),
     nopython=True, fastmath=True, cache=True)
def _get_envelope_numba(t, attack, decay, sustain, release, end_time, sample_rate):
    """Optimized envelope calculation."""
    result = numpy.zeros_like(t)
    attack_rate = 1.0 / attack if attack > 0 else 1e9
    decay_rate = (1.0 - sustain) / decay if decay > 0 else 0.0
    
    for i in range(len(t)):
        ti = t[i]
        if ti < attack:
            result[i] = ti * attack_rate
        else:
            result[i] = max(1.0 - (ti - attack) * decay_rate, sustain)
        
        if end_time >= 0:
            end_time_sec = end_time / sample_rate
            if ti > end_time_sec:
                if end_time_sec < attack:
                    end_status = end_time_sec * attack_rate
                else:
                    end_status = max(1.0 - (end_time_sec - attack) * decay_rate, sustain)
                
                if release > 0:
                    result[i] = max(end_status * (1.0 - (ti - end_time_sec) / release), 0.0)
                else:
                    result[i] = 0.0
    
    return result

@jit((float32[:], float32[:], int32, float32, float32),
     nopython=True, fastmath=True, cache=True)
def _mix_oscillators(w1, w2, modulation_type, vol1, vol2):
    """Optimized oscillator mixing."""
    if modulation_type == 0:  # MIX
        return (w1 * vol1 + w2 * vol2) * float32(0.5)
    elif modulation_type == 1:  # AMPLITUDE
        return w1 * vol1 * (float32(1.0) + w2 * vol2)
    elif modulation_type == 2:  # RING
        return w1 * w2 * vol1 * vol2
    else:
        return numpy.zeros_like(w1)

@jit((float32[:], float32, float32[:], float32, float32, float32, float32),
     nopython=True, fastmath=True, cache=True)
def _mix_oscillators_frequency(w1_table, freq1, w2, vol1, vol2, dt, pf):
    instantaneous_freq = freq1 + freq1 * w2 * vol2
    phase_increments = instantaneous_freq * dt
    accumulated_phase = numpy.cumsum(phase_increments) + pf
    w1 = _get_wavetable_numba(w1_table, accumulated_phase)
    return w1 * vol1, accumulated_phase[-1]

@jit((float32[:], float32[:], float32[:], float32, float32, float32),
     nopython=True, fastmath=True, cache=True)
def _mix_oscillators_phase(w1_table, t, w2, freq1, vol1, vol2):
    phase = t + w2 * vol2 / freq1
    w1 = _get_wavetable_numba(w1_table, phase * freq1)
    return w1 * vol1

class Synth:
    waveforms: list[numpy.ndarray]
    outputs: list[tuple[int, int]]
    volume: list[tuple[int | float, int | float]]
    envelope: list[tuple[int | float, int | float, int | float, int | float]]
    filter_parameters: list[tuple[int | float | None, int | float | None]]
    frequency: list[tuple[int | float, int | float]]
    absolute: list[tuple[bool, bool]]
    modulations: list[int]

    def __init__(self, sample_rate=44100, buffer_size=2048, from_file=None, latency: str = 'low'):
        """
        A simple synthesizer that can play multiple frequencies simultaneously.

        :param sample_rate: Audio sample rate in Hz (default 44100)
        :param buffer_size: Size of audio buffer (default 1024)
        :param from_file: The file of the preset to load from (optional)
        :param latency: The latency of the output for ``sounddevice`` (default 'low')
        """
        self.filters = [[(None, None), (None, None)],
                        [(None, None), (None, None)],
                        [(None, None), (None, None)],
                        [(None, None), (None, None)]]
        self.sample_rate = sample_rate
        self.sample_rate_reciprocal = 1.0 / self.sample_rate
        self.buffer_size = buffer_size
        self.active_frequencies = set()
        self.playing_frequencies = {}
        self.lock = threading.Lock()
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
        sounddevice.default.latency = latency
        self.stream = sounddevice.OutputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype=numpy.float32,
            blocksize=self.buffer_size,
            callback=self._audio_callback
        )
        self.stream.start()

    def _setup_filters(self):
        for i in range(4):
            if self.filter_parameters[i][0] is not None:
                self.filters[i][0] = signal.butter(8, self.filter_parameters[i][0], fs=self.sample_rate,
                                                   btype='lowpass', analog=False)
            if self.filter_parameters[i][1] is not None:
                self.filters[i][1] = signal.butter(8, self.filter_parameters[i][1], fs=self.sample_rate,
                                                   btype='highpass', analog=False)
        
        self.filter_states = [[None, None] for _ in range(4)]

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
            self.wavetables[item] = new_data.astype(numpy.float32)

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
        with self.lock:
            self.wavetables = [numpy.array(table, dtype=numpy.float32) for table in data["wavetables"]]
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

    def _generate_output(self, item, freq, t, frame_count):
        freq1 = self.frequency[item][0] if self.absolute[item][0] else freq * self.frequency[item][0]
        freq2 = self.frequency[item][1] if self.absolute[item][1] else freq * self.frequency[item][1]
        
        if self.outputs[item][0] == -1:
            return numpy.zeros(frame_count, dtype=numpy.float32)
        
        w1 = _get_wavetable_numba(self.wavetables[self.outputs[item][0]], t * freq1)
        
        if self.outputs[item][1] == -1:
            total = w1 * self.volume[item][0]
        else:
            w2 = _get_wavetable_numba(self.wavetables[self.outputs[item][1]], t * freq2)
            
            if self.modulations[item] == PHASE:
                total = _mix_oscillators_phase(self.wavetables[self.outputs[item][0]], t, w2, freq1, 
                                               self.volume[item][0], self.volume[item][1])
            elif self.modulations[item] == FREQUENCY:
                total, self.playing_frequencies[freq][2][item] = \
                    _mix_oscillators_frequency(self.wavetables[self.outputs[item][0]], freq1,
                                              w2, self.volume[item][0], self.volume[item][1],
                                              self.sample_rate_reciprocal, self.playing_frequencies[freq][2][item])
            else:
                total = _mix_oscillators(w1, w2, self.modulations[item], 
                                        self.volume[item][0], self.volume[item][1])
        
        envelope = _get_envelope_numba(t, self.envelope[item][0], self.envelope[item][1],
                                       self.envelope[item][2], self.envelope[item][3],
                                       self.playing_frequencies[freq][1] if self.playing_frequencies[freq][1] is not None else -1,
                                       self.sample_rate)
        total = total * envelope
        return total

    def _generate_channel(self, channel, frame_count):
        total = numpy.zeros(frame_count, dtype=numpy.float32)
        for freq, data in self.playing_frequencies.items():
            phase = data[0]
            t = (numpy.arange(frame_count, dtype=numpy.float32) + phase) * self.sample_rate_reciprocal
            total += self._generate_output(channel, freq, t, frame_count) * self.playing_frequencies[freq][3]
        
        if self.filter_parameters[channel][0] is not None and self.filters[channel][0] is not None:
            if self.filter_states[channel][0] is None:
                self.filter_states[channel][0] = signal.lfilter_zi(*self.filters[channel][0]) * total[0]
            total, self.filter_states[channel][0] = signal.lfilter(
                *self.filters[channel][0], total, zi=self.filter_states[channel][0]
            )

        if self.filter_parameters[channel][1] is not None and self.filters[channel][1] is not None:
            if self.filter_states[channel][1] is None:
                self.filter_states[channel][1] = signal.lfilter_zi(*self.filters[channel][1]) * total[0]
            total, self.filter_states[channel][1] = signal.lfilter(
                *self.filters[channel][1], total, zi=self.filter_states[channel][1]
            )

        return total

    def generate_samples(self, frame_count):
        """
        Generate samples of the playing frequencies.

        :param frame_count: The amount of samples to generate
        :return: The generated samples as a numpy array
        """
        if not self.playing_frequencies:
            return numpy.zeros(frame_count, dtype=numpy.float32)
        
        audio_data = numpy.zeros(frame_count, dtype=numpy.float32)
        removed_frequencies = []
        
        max_release = max(e[3] for e in self.envelope) * self.sample_rate

        for item in range(4):
            audio_data += self._generate_channel(item, frame_count)

        for freq, data in self.playing_frequencies.items():
            data[0] += frame_count

        numpy.clip(audio_data, -1.0, 1.0, out=audio_data)
        return audio_data

    def _audio_callback(self, outdata: numpy.ndarray, frames, time, status):
        with self.lock:
            audio_data = self.generate_samples(outdata.size)

        outdata[:, 0] = audio_data

    def start_frequency(self, frequency, volume=1.0):
        """
        Start playing at the specified frequency.

        :param frequency: Frequency in Hz (can be any float value, including microtonal)
        :param volume: Volume of the note (default 1)
        :return: Nothing
        """
        with self.lock:
            if frequency not in self.active_frequencies:
                self.active_frequencies.add(frequency)
                self.playing_frequencies[frequency] = [0, None, [0, 0, 0, 0], volume]

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
    
    def start_note(self, note, volume=1.0):
        """
        Start playing at the specified note (using MIDI notes).

        :param note: MIDI note number (0-127)
        :param volume: Volume of the note (default 1)
        :return: Nothing
        """
        self.start_frequency(440 * 2 ** ((note - 69) / 12), volume=volume)

    def stop_note(self, note):
        """
        Stop playing at the specified note (using MIDI notes).

        :param note: MIDI note number (0-127)
        :return: Nothing
        """
        self.stop_frequency(440 * 2 ** ((note - 69) / 12))

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
        self.stream.stop()

    def start_stream(self):
        self.stream.start()

    def close(self):
        """
        Close the synthesizer and clean up resources.
        """
        self.stream.stop()
        self.stream.close()
