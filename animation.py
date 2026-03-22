"""Animation script."""

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile as sf
from matplotlib.animation import FuncAnimation

from realtimeplp import BeatAnalyzer, RealTimeBeatTracker

TESTAUDIO = "assets/audio/DrumBeat.wav"

# Setup audio streaming and beat tracking
SR = librosa.get_samplerate(TESTAUDIO)
HOP = 512

audio_stream = librosa.stream(
    path=TESTAUDIO, block_length=HOP, frame_length=1, hop_length=1, fill_value=0
)

tracker = RealTimeBeatTracker.from_args(
    N=2 * HOP,
    H=HOP,
    samplerate=SR,
    N_time=6,
    Theta=np.arange(30, 300, 1),
    lookahead=0,
)
analyzer = BeatAnalyzer(tracker)

# Process every frame of audio in the audio stream
for frame in audio_stream:
    analyzer.process(audio_frame=frame)


# Calculate normalization for buffer (already done in realtimeplp.py)
N = tracker.plp.N
H = 1
L = 2 * N
w_type = "hann"
w = scipy.signal.get_window(w_type, N)
# Note: plp_buffers are already normalized in realtimeplp.py

# Setup plot
fig = plt.figure(dpi=300, figsize=(6, 3))  # Increased figure size
ax = plt.axes(xlim=(-3, 3), ylim=(-1.1, 1.1))
ax.set_title(r"PLP Buffer")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("PLP")
# ax.axhline(0, c="C7", ls=":", zorder=1)
middle_line = ax.axvline(0, c="k", ls="--", lw=1, zorder=1)
(line,) = ax.plot([], [], lw=2, c="C0", label=r"$\Gamma_{n_0}$")
(dots,) = ax.plot([], [], "ro")
# Add text for frame number in fixed position
frame_text = ax.text(
    0.82,
    0.98,
    r"$n_0 = 0$",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    horizontalalignment="left",
)

# Adjust layout to ensure labels are visible
plt.tight_layout()

# Track beat detection for sustained highlight
beat_highlight_frames = 0
highlight_duration = 3  # Number of frames to keep highlight


def init():
    """Init for animation"""
    line.set_data([], [])
    dots.set_data([], [])
    frame_text.set_text(r"$n_0 = 0$")
    return line, dots, frame_text


def animate(i):
    """Animation function"""
    global beat_highlight_frames

    # Get the already-normalized PLP buffer for frame i
    buffer = analyzer.plp_buffers[i]

    # Create time axis centered at 0
    n = (np.arange(len(buffer)) / tracker.tempogram.framerate) - 3

    # Pick peaks from the current buffer
    from realtimeplp import Peaks

    peaks = Peaks.pick(buffer, np.arange(tracker.plp.N))

    # Update plot
    line.set_data(n, buffer)
    dots.set_data(n[peaks.i], buffer[peaks.i])

    # Update frame number text
    frame_text.set_text(rf"$n_0 = {i}$")

    # Check if beat is detected at current frame
    beat_detected_at_frame = analyzer.beat_detection_frames[i]
    if beat_detected_at_frame:
        beat_highlight_frames = highlight_duration

    # Highlight middle line when beat is detected or within highlight duration
    if beat_highlight_frames > 0:
        middle_line.set_color("r")
        middle_line.set_linestyle("-")
        middle_line.set_linewidth(3)
        beat_highlight_frames -= 1
    else:
        middle_line.set_color("k")
        middle_line.set_linestyle("--")
        middle_line.set_linewidth(1)

    return line, dots, frame_text


interval = (1 / tracker.activation.samplerate) * 1000 * HOP

anim = FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(analyzer.plp_buffers),
    interval=interval,
    blit=True,
)

# Ensure layout is adjusted before saving
plt.tight_layout()

# anim.save('sine_wave.gif', writer='imagemagick')
anim.save("plp_buffer.mp4", bitrate=2000, dpi=200, writer="ffmpeg")

# Generate audio with detected beat clicks
# Reconstruct audio from frames
audio = np.concatenate(analyzer.audio_frames)

# Get frame indices where beats are detected (beat_detection_frames contains booleans)
beat_frame_indices = np.where(np.array(analyzer.beat_detection_frames))[0]
# Convert frame indices to sample indices
beat_sample_indices = beat_frame_indices * HOP
t_peaks = beat_sample_indices / SR
clicks = librosa.clicks(times=t_peaks, sr=SR, click_freq=1000, length=len(audio))

# Mix audio with proper amplitude control to prevent clipping
audio_gain = 0.7  # Reduce original audio to leave headroom
click_gain = 0.3  # Control click volume
mixed_audio = audio_gain * audio + click_gain * clicks

# Ensure no clipping occurs
mixed_audio = np.clip(mixed_audio, -1.0, 1.0)

sf.write("clicks.wav", mixed_audio, SR)

# Combine audio and video
os.system(
    "ffmpeg -y -i plp_buffer.mp4 -i clicks.wav \
    -map 0:v -map 1:a -c:v copy -c:a aac -b:a 320k -ar 44100 -shortest output.mp4"
)
os.system("mv output.mp4 plp_buffer.mp4")
# os.system('rm clicks.wav')

# Add Audio to Video
# https://stackoverflow.com/q/11779490
# ffmpeg -i plp_buffer.mp4 -i audios/FMP_C6_F19_Brahms_Ormandy_sec35-53.wav
#        -map 0:v -map 1:a -c:v copy -shortest output.mp4
