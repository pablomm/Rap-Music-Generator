r"""

Original code writed by carykh in https://github.com/carykh/rapLyrics.
Extract of the original MIT License:

Copyright (c) 2019 carykh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import samplerate
from gtts import gTTS
from scipy.io import wavfile
import numpy as np
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from pydub import AudioSegment
import pydub
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm as tqdm_gui
import os


filename_lyrics = "examples/redes_neuronales.txt"
input_mp3 = "tracks/manifiesto.mp3"
output_filename = "rap_redes_neuronales2"
song_length = 130
snap_time = 60/95
waiting_time = 11 # 11 para base.mp3 , 14 para manifiesto
backing_track_volume = 0.2
speed_up = 2/3
have_echo = True
end = -1


# Global Params
SAMPLE_RATE = 44100
LOW_FACTOR = 1.35  # original 1.42
LANGUAGE = "es"
AUX_FOLDER = "song_aux/"
PLACEHOLDER_MP3_AUX = AUX_FOLDER + "placeholder.mp3"
PLACEHOLDER_WAV_AUX = AUX_FOLDER + "placeholder.wav"
LOW_PLACEHOLDER_WAV_AUX = AUX_FOLDER + "lowPlaceholder.wav"
BACKTRACKING_AUX = AUX_FOLDER + "backingTrack.wav"
STRECH = AUX_FOLDER + "stretchholder.wav"


def getFirstLoudPart(d):
    THRESHOLD = 0.2
    for i in range(len(d)):
        if(abs(d[i]) > THRESHOLD):
            return i


def getLastLoudPart(d):
    THRESHOLD = 0.07
    LAST_SYLLABLE_LENGTH = SAMPLE_RATE*0.21
    for i in range(len(d)-1, 0, -1):
        if(abs(d[i]) > THRESHOLD):
            return i-LAST_SYLLABLE_LENGTH


def getStretchedData(low, sf):
    s = PLACEHOLDER_WAV_AUX
    playSpeed = 1/sf
    if low:
        s = LOW_PLACEHOLDER_WAV_AUX
        playSpeed *= LOW_FACTOR
    with WavReader(s) as reader:
        with WavWriter(STRECH, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=playSpeed)
            tsm.run(reader, writer)
    _, s = wavfile.read(STRECH)
    d = np.zeros(s.shape)
    if low:
        d += s
    else:
        d += s*0.81
    return d


def doFileStuff(line, isSlow):
    myobj = gTTS(text=line, lang=LANGUAGE, slow=isSlow)
    myobj.save(PLACEHOLDER_MP3_AUX)

    y, sr = librosa.load(PLACEHOLDER_MP3_AUX)
    data = librosa.resample(y, sr, SAMPLE_RATE)
    librosa.output.write_wav(PLACEHOLDER_WAV_AUX, data, SAMPLE_RATE)
    d, sr = sf.read(PLACEHOLDER_WAV_AUX)
    sf.write(PLACEHOLDER_WAV_AUX, d, sr)

    y, sr = librosa.load(PLACEHOLDER_MP3_AUX)
    lowData = librosa.resample(y, sr, SAMPLE_RATE*LOW_FACTOR)
    librosa.output.write_wav(LOW_PLACEHOLDER_WAV_AUX, lowData, SAMPLE_RATE)
    d, sr = sf.read(LOW_PLACEHOLDER_WAV_AUX)
    sf.write(LOW_PLACEHOLDER_WAV_AUX, d, sr)

    return data


def generate_song(filename_lyrics, input_mp3, output_filename, song_length,
                  snap_time, waiting_time, backing_track_volume, speed_up,
                  have_echo=False, start=0, end=-1, plot=False,
                  shouldChangeSinger=True, apply_mask=False, low_register=True):
    if output_filename.endswith(".mp3") or output_filename.endswith(".wav"):
        output_filename = output_filename[:-4]

    print("Generating: ", output_filename + ".wav")
    print("Lyrics file:", filename_lyrics)
    print("Backing track:", input_mp3)

    waiting_time *= SAMPLE_RATE
    snap_time *= SAMPLE_RATE

    master_length = SAMPLE_RATE*song_length
    sound = AudioSegment.from_mp3(input_mp3)
    sound.export(BACKTRACKING_AUX, format="wav")
    _, backingTrack = wavfile.read(BACKTRACKING_AUX)

    masterTrack = np.zeros((master_length))
    if apply_mask:  # very optional. Definitely don't need it : it just loudens
        mask = 1+1.4*np.clip((np.arange(0, master_length) -
                              86.5*SAMPLE_RATE)/SAMPLE_RATE, 0, 1)
        masterTrack += backingTrack[0:master_length,
                                    0]*backing_track_volume * mask
    else:
        masterTrack += backingTrack[0:master_length, 0]*backing_track_volume

    beatOn = 0

    linesFile = open(filename_lyrics)
    lines = [line for line in linesFile.read().split("\n")
             if line.strip() != ''][start:end]

    lastEchoData = np.zeros((1))
    lastBeatOn = 0
    bar = tqdm_gui(len(lines))

    for i, line in enumerate(lines):
        bar.update(i)

        if len(line) == 0:
            continue

        # Skip comment lines
        if "[" in line[0]:
            if shouldChangeSinger:
                low_register = (not low_register)
            continue

        data = doFileStuff(line, False)

        firstLoudPart = getFirstLoudPart(data)
        lastLoudPart = max(getLastLoudPart(data), firstLoudPart+snap_time)
        loudLength = lastLoudPart-firstLoudPart
        snappedLength = round(speed_up*loudLength/snap_time)*snap_time
        if snappedLength <= 0:
            snappedLength = snap_time
        beatsUsed = int(round(snappedLength/snap_time))
        scalingFactor = snappedLength/loudLength

        # print(scalingFactor)
        # uh-oh, this a stretch: quality isn't as good. Get the slow version.
        if scalingFactor >= 0.9:
            data = doFileStuff(line, True)

            firstLoudPart = getFirstLoudPart(data)
            beatsUsed = int(round(snappedLength/snap_time))
            lastLoudPart = max(getLastLoudPart(data), firstLoudPart+snap_time)
            loudLength = lastLoudPart-firstLoudPart
            scalingFactor = snappedLength/loudLength
            #print("new: "+str(scalingFactor))

        stretchedData = getStretchedData(low_register, scalingFactor)

        nextBeatOn = beatOn+beatsUsed+1
        jumpGap = 0
        if ((beatOn//16) != (nextBeatOn//16) and
            (beatOn+beatsUsed) % 16 != 0 and nextBeatOn % 16 != 0):
            jumpGap = (nextBeatOn//16)*16-beatOn
            beatOn = (nextBeatOn//16)*16
            nextBeatOn = beatOn+beatsUsed+1
        #print("uhh?  "+str(beatOn))

        if jumpGap >= 1 and have_echo:
            echoEdge = min(jumpGap, 1)
            echoStart = int((lastBeatOn+echoEdge)*snap_time -
                            firstLoudPart*scalingFactor+waiting_time)
            echoEnd = echoStart+len(lastEchoData)

            fadeStart = end-snap_time*2
            fadeMask = np.clip(
                (np.arange(echoStart, echoEnd)-fadeStart)/snap_time*3-2, 0, 1)
            masterTrack[echoStart:echoEnd] += lastEchoData*fadeMask*0.8

        lastEchoData = getStretchedData(not low_register, scalingFactor)

        start = int(beatOn*snap_time-firstLoudPart*scalingFactor+waiting_time)
        end = start+len(stretchedData)
        masterTrack[start:end] += stretchedData*0.8

        lastBeatOn = beatOn
        beatOn = nextBeatOn

        max_line = 70
        line_print = "Line " + str(i+1) + ": " + line
        if len(line_print) > max_line:
            line_print = line_print[:max_line-3] + "..."
        else:
            line_print = line_print + (max_line - len(line_print)) * " "

        print("   ", line_print, end="\r")

    extreme = max(np.amax(masterTrack), -np.amin(masterTrack))
    masterTrack = masterTrack/extreme

    wavfile.write(output_filename + ".wav", SAMPLE_RATE, masterTrack)
    # bar.close()

    return masterTrack


def plot_gif(wav, nframes=100, npoints=20000, start=.5, step=100, save=None):
    plt.style.use('seaborn')
    fig, ax = plt.subplots()

    t = np.linspace(0, 1, npoints)
    l = len(wav)
    start = int(start*l)
    plt.xticks([])

    line, = plt.plot(t, wav[start:start+npoints])

    def update(i):
        line.set_ydata(wav[min(start+step*i, l-npoints):
                           min(start+step*i+npoints, l)])

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(
        fig, update, frames=np.arange(0, nframes), interval=100)

    if save is not None:
        anim.save(save, dpi=80, writer='imagemagick')

    # plt.show()

    return anim


def setup():
    os.makedirs(AUX_FOLDER, exist_ok=True)


if __name__ == "__main__":
    setup()
    track = generate_song(filename_lyrics, input_mp3, output_filename,
                          song_length, snap_time, waiting_time,
                          backing_track_volume, speed_up, have_echo, end=end)

    #plot_gif(track, save="track.gif")
