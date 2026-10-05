import io
import os
import base64
import struct
import wave

import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display

from flask import Flask, render_template, request, send_file, url_for

app = Flask(__name__)


SAMPLES_DIR = os.path.join(app.root_path, "static", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)


def list_sample_files():
    return sorted(
        f for f in os.listdir(SAMPLES_DIR)
        if f.lower().endswith(".wav") and os.path.isfile(os.path.join(SAMPLES_DIR, f))
    )




def get_signal_hz(hz, sample_rate, length_ts_sec):
    ts1sec = list(np.linspace(0, np.pi * 2 * hz, sample_rate))
    ts = ts1sec * length_ts_sec
    return np.array(ts[: sample_rate * length_ts_sec])


def build_composite_signal(freqs1, freqs2, sample_rate, tone_len, silence_len):
    tone1 = np.zeros(sample_rate * tone_len)
    for f in freqs1:
        tone1 = tone1 + get_signal_hz(f, sample_rate, tone_len)

    ts_silence = np.zeros(sample_rate * silence_len)

    tone2 = np.zeros(sample_rate * tone_len)
    for f in freqs2:
        tone2 = tone2 + get_signal_hz(f, sample_rate, tone_len)

    ts = np.concatenate([tone1, ts_silence, tone2])
    return ts


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_waveform_plot(ts, sample_rate):
    total_sec = len(ts) / sample_rate
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.linspace(0, total_sec, len(ts)), ts, linewidth=0.6)
    ax.set_xlabel("Time (second)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def make_spectrum_plot(ts, sample_rate):
    n = len(ts)
    yf = fft(ts)[: n // 2]
    xf = fftfreq(n, 1 / sample_rate)[: n // 2]
    amp = (2 / n) * np.abs(yf)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xf, amp)
    ax.set_title("Spectrum of the Signal (FFT)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, sample_rate / 2)
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def make_spectrogram_plot(ts, sample_rate):
    f, t, Sxx = signal.spectrogram(np.array(ts), sample_rate)
    fig, ax = plt.subplots(figsize=(9, 4))
  
    mesh = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    fig.colorbar(mesh, ax=ax, label="Intensity [dB]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")
    ax.set_title("Spectrogram (STFT)")
    return fig_to_base64(fig)


def make_librosa_stft_plot(ts, sample_rate, n_fft=2048, hop_length=512):
    audio_data = np.array(ts, dtype=np.float32)
    stft_audio = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    s_magnitude = np.abs(stft_audio)
    s_db = librosa.amplitude_to_db(s_magnitude, ref=np.max)

    fig, ax = plt.subplots(figsize=(9, 4))
    img = librosa.display.specshow(
        s_db, sr=sample_rate, hop_length=hop_length,
        x_axis="time", y_axis="hz", cmap="magma", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("STFT Magnitude Spectrogram (librosa)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    return fig_to_base64(fig)


def make_wav_spectrogram_plots(sample_rate, samples):
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    mesh1 = ax1.pcolormesh(times, frequencies, spectrogram, shading="auto")
    fig1.colorbar(mesh1, ax=ax1, label="Intensity")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [sec]")
    ax1.set_title("Spectrogram (linear intensity)")
    linear_plot = fig_to_base64(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    mesh2 = ax2.pcolormesh(
        times, frequencies, 10 * np.log10(spectrogram + 1e-12), shading="gouraud"
    )
    fig2.colorbar(mesh2, ax=ax2, label="Intensity [dB]")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_xlabel("Time [sec]")
    ax2.set_title("Spectrogram (dB)")
    db_plot = fig_to_base64(fig2)

    return linear_plot, db_plot


def signal_to_wav_bytes(ts, sample_rate):
    ts = np.asarray(ts, dtype=np.float64)
    peak = np.max(np.abs(ts)) or 1.0
    normalized = ts / peak
    pcm = (normalized * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf



LAST_SIGNAL = {"ts": None, "sample_rate": None}


def parse_freq_list(raw):
    vals = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            vals.append(float(chunk))
    return vals or [440.0]


DEFAULTS = {
    "freqs1": "697,1209",
    "freqs2": "697,1336",
    "sample_rate": 4000,
    "tone_len": 3,
    "silence_len": 1,
}


@app.route("/", methods=["GET", "POST"])
def index():
    params = DEFAULTS.copy()
    error = None
    plots = None

    if request.method == "POST":
        params["freqs1"] = request.form.get("freqs1", DEFAULTS["freqs1"])
        params["freqs2"] = request.form.get("freqs2", DEFAULTS["freqs2"])
        params["sample_rate"] = int(request.form.get("sample_rate", DEFAULTS["sample_rate"]))
        params["tone_len"] = int(request.form.get("tone_len", DEFAULTS["tone_len"]))
        params["silence_len"] = int(request.form.get("silence_len", DEFAULTS["silence_len"]))

        try:
            freqs1 = parse_freq_list(params["freqs1"])
            freqs2 = parse_freq_list(params["freqs2"])
            sample_rate = params["sample_rate"]
            tone_len = params["tone_len"]
            silence_len = params["silence_len"]

            if sample_rate <= 0 or tone_len < 0 or silence_len < 0:
                raise ValueError("Sample rate must be positive; durations must be >= 0.")
            if max(freqs1 + freqs2) >= sample_rate / 2:
                raise ValueError(
                    "A frequency exceeds the Nyquist limit (sample_rate / 2) "
                    "for this sample rate — increase sample rate or lower the frequency."
                )

            ts = build_composite_signal(freqs1, freqs2, sample_rate, tone_len, silence_len)

            LAST_SIGNAL["ts"] = ts
            LAST_SIGNAL["sample_rate"] = sample_rate

            plots = {
                "waveform": make_waveform_plot(ts, sample_rate),
                "spectrum": make_spectrum_plot(ts, sample_rate),
                "spectrogram": make_spectrogram_plot(ts, sample_rate),
                "librosa_spectrogram": make_librosa_stft_plot(ts, sample_rate),
                "duration": round(len(ts) / sample_rate, 3),
                "n_points": len(ts),
            }
        except Exception as e:
            error = str(e)

    return render_template("index.html", params=params, plots=plots, error=error)



LAST_UPLOAD = {"bytes": None, "filename": None}


@app.route("/upload", methods=["GET", "POST"])
def upload():
    upload_error = None
    upload_plots = None
    filename = None
    playback_url = None
    samples_available = list_sample_files()

    if request.method == "POST":
        action = request.form.get("action", "upload")
        try:
            if action == "sample":
                chosen = request.form.get("sample_choice", "")
                if chosen not in samples_available:
                    raise ValueError("Please choose a valid sample file.")
                filename = chosen
                sample_path = os.path.join(SAMPLES_DIR, chosen)
                sample_rate, samples = wavfile.read(sample_path)
                playback_url = url_for("static", filename=f"samples/{chosen}")
            else:
                f = request.files.get("wavfile")
                if f is None or f.filename == "":
                    raise ValueError("Please choose a .wav file to upload.")
                filename = f.filename
                raw_bytes = f.read()
                sample_rate, samples = wavfile.read(io.BytesIO(raw_bytes))

                
                LAST_UPLOAD["bytes"] = raw_bytes
                LAST_UPLOAD["filename"] = filename
                playback_url = url_for("uploaded_audio") + f"?_={len(raw_bytes)}"

            samples = samples.astype(np.float64)
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1)

            linear_plot, db_plot = make_wav_spectrogram_plots(sample_rate, samples)

            upload_plots = {
                "linear": linear_plot,
                "db": db_plot,
                "sample_rate": sample_rate,
                "n_samples": int(samples.shape[0]),
                "duration": round(samples.shape[0] / sample_rate, 3),
            }
        except Exception as e:
            upload_error = f"Could not process that file: {e}"

    return render_template(
        "upload.html",
        error=upload_error,
        plots=upload_plots,
        filename=filename,
        samples_available=samples_available,
        playback_url=playback_url,
    )


@app.route("/uploaded_audio.wav")
def uploaded_audio():
    if LAST_UPLOAD["bytes"] is None:
        return "No uploaded audio yet — upload a file on the /upload page first.", 404
    buf = io.BytesIO(LAST_UPLOAD["bytes"])
    return send_file(
        buf,
        mimetype="audio/wav",
        as_attachment=False,
        download_name=LAST_UPLOAD["filename"] or "uploaded.wav",
    )


@app.route("/audio.wav")
def audio_wav():
    if LAST_SIGNAL["ts"] is None:
        return "No signal generated yet — submit the form first.", 404
    buf = signal_to_wav_bytes(LAST_SIGNAL["ts"], LAST_SIGNAL["sample_rate"])
    return send_file(buf, mimetype="audio/wav", as_attachment=False, download_name="signal.wav")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
