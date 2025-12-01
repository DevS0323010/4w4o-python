# 4W4O: Simple Python Synth with 4 Wavetables & 4 Outputs

A real-time, customizable synthesizer for music creation in Python.

---

## Key features

* **4 Wavetables**: Customizable waveforms (with 128 samples)
* **4 Outputs**: Simultaneous audio channels with the following features:
  * **Wavetable Customization**: Adjustable amplitude and frequency (both absolute and relative to root)
  * **Wavetable Modulation**: Supports mix, amplitude, ring, frequency, and phase modulation
  * **ADSR Envelope**: Modifiable envelope for every individual output
  * **Adjustable Filters**: Either/both high-pass and low-pass filters with frequency control
* **GUI Preset Editor**: GUI editor that makes preset making easier (optional)
* **Two Installation Paths**:
  * `minimal`: Just the synth logic
  * `full`: Synth + GUI editor

---

## Installation

### Minimal Install (Core Synth Only)

To install the synth in your project:

```bash
# Clone repo
git clone https://github.com/DevS0323010/4w4o-python.git
# Add synth/ folder to your project
cp -r synth path/to/your/project
# Install minimal dependencies only
pip install -r requirements-minimal.txt
```

### Full Install (Synth + Editor GUI)

```bash
# Clone repo
git clone https://github.com/DevS0323010/4w4o-python.git
# Install full dependencies
pip install -r requirements.txt
```

---

## How to Use

### Launch Synth Editor GUI

1. Install using the **Full Install** guide
2. Run the GUI Editor using the following command:

    ```bash
    python main.py 
    ```

### Control the Synth Using Code

1. Install using either method
2. Load a preset:

    ```python
    from synth.synth import Synth
    s = Synth()
    s.read_from_file("presets/my-presets.json") # Replace with your preset
    ```

3. Play a note:

   ```python
   import time
   s.start_frequency(440.0)
   time.sleep(1)
   s.stop_frequency(440.0)
   ```
   > **Important**: MIDI note support is not implemented yet. Use absolute frequencies (Hz) directly.
4. Close the synthesizer:

    ```python
    s.close()
    ```

---

## Preset and Synthesizer Structure
>
> It is **not** recommended to edit presets by hand.
> Use the GUI editor for advanced editing.
>
> However, the format is the same when using the GUI editor.

Your preset follow this format:

| Field                 | Type                      | Example Value                     | Notes                                                      |
|-----------------------|---------------------------|-----------------------------------|------------------------------------------------------------|
| `"wavetables"`        | `list[list[float]]`       | `[[0.0, 0.049, 0.098, ...], ...]` | 4 waveforms, 128 samples each (`-1.0` to `1.0`)            |
| `"outputs"`           | `list[list[int]]`         | `[[0, -1], [-1, -1], ...]`       | `0-3` = wavetable ID; `-1` = no wavetable                  |
| `"volume"`            | `list[list[float]]`       | `[[0.2, 0.0], [0.0, 0.0], ...]`   | Amplitude (≥ `0.0`) for active wavetables                  |
| `"frequency"`         | `list[list[float]]`       | `[[1.0, 1.0], [2.0, 1.0], ...]`   | Frequency for active wavetables                            |
| `"absolute"`          | `list[list[bool]]`        | `[[True, False], ...]`            | `True` = absolute Hz; `False` = relative to root           |
| `"modulations"`       | `list[int]`               | `[0, 1, 2, 4]`                    | `0`=mix, `1`=amplitude, `2`=ring, `3`=frequency, `4`=phase |
| `"envelope"`          | `list[list[float]]`       | `[[0.01, 0.1, 0.5, 0.2], ...]`    | ADSR parameters (attack, decay, sustain, release)          |
| `"filter_parameters"` | `list[list[float\|null]]` | `[[1000.0, null], ...]`           | `[low-cut, high-cut]` (null = disabled)                    |

### Minimal Working Example

```json
{
  "wavetables": [
    [0.0, 0.049, 0.098, ...], [0.0, 0.0, 0.0, ...], 
    [0.0, 0.0, 0.0, ...], [0.0, 0.0, 0.0, ...]
  ],
  "outputs": [[0, -1], [-1, -1], [-1, -1], [-1, -1]], 
  "volume": [[0.2, 0], [0.0, 0], [0.0, 0], [0.0, 0]], 
  "frequency": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], 
  "absolute": [[false, false], [false, false], [false, false], [false, false]], 
  "modulations": [0, 0, 0, 0],
  "envelope": [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]], 
  "filter_parameters": [[null, null], [null, null], [null, null], [null, null]]
}
```

---

## License

This project is licensed under the MIT License — see `LICENSE` for details.
