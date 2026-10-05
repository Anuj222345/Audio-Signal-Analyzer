Drop .wav files here and they'll automatically appear as pickable
options on the app's /upload page (dropdown + "Analyze sample" button),
with no re-uploading needed each time.

To add the exact files used in the "Beginning with Audio Sampling and
Spectrogram" Kaggle notebook:

1. Go to the notebook's Input tab:
   https://www.kaggle.com/code/ashiqnazir/beginning-with-audio-sampling-and-spectrogram/input

2. Click the download icon (⬇) next to any of these files:
     - BAK.wav
     - Beethoven_Diabelli_Variation_No._13.wav
     - Circle_of_fifths_chord_progression.wav
     - Divisive_rhythm_in_4-4.wav
     - Mozart_from_Piano_Sonata_K310_first_movement.wav

3. Move the downloaded .wav file(s) into this folder
   (static/samples/), then restart the app (or just reload the
   /upload page — Flask picks up new files automatically, no restart
   needed for the dropdown list itself).

The Mozart file is also on Wikimedia Commons (public domain / CC
licensed recording) if you'd rather grab it from there instead of
Kaggle:
   https://commons.wikimedia.org/wiki/File:Mozart,_from_Piano_Sonata_K310,_first_movement.wav
(open that page and use its own "Download" link to get the raw .wav)

Any other .wav file works too — this folder isn't limited to these
five, it's just a convenient "sample library" for whatever you want to
inspect without re-uploading each time.
