from flask import Flask, render_template, request, send_file
import numpy as np
import soundfile as sf
import librosa
import io

app = Flask(__name__)

# Function for multiband dynamics processing
def multiband_dynamics(audio_data, num_bands=30, threshold=80, ratio=60):
    # Split the audio into frequency bands
    stft = librosa.stft(audio_data)
    bands = np.array_split(stft, num_bands, axis=0)

    # Apply dynamics processing to each band
    processed_bands = []
    for band in bands:
        # Compute energy in each frequency band
        band_energy = np.mean(np.abs(band), axis=0)
        
        # Compute gain based on energy and threshold
        gain = np.ones_like(band_energy)
        gain[band_energy > threshold] = 10 + (ratio - 10) * ((band_energy[band_energy > threshold] - threshold) / threshold)
        
        # Transpose gain to match the shape of the band
        gain = gain.T

        # Apply gain to the band
        processed_band = band * gain
        processed_bands.append(processed_band)

    # Combine the processed bands back into a single audio signal
    processed_audio = librosa.istft(np.concatenate(processed_bands, axis=0))
    return processed_audio

@app.route('/')
def index():
    return render_template('master.html')

@app.route('/process', methods=['POST'])
def process_audio():
    # Get the uploaded audio file
    audio_file = request.files['audio']

    # Read the audio file
    audio_data, sr = librosa.load(audio_file, sr=None)

    # Perform multiband dynamics processing
    processed_audio = multiband_dynamics(audio_data)

    # Return the processed audio as a downloadable file
    processed_wav = io.BytesIO()
    sf.write(processed_wav, processed_audio, sr, format='wav')
    processed_wav.seek(0)

    return send_file(
        processed_wav,
        mimetype='audio/wav',
        as_attachment=True,
        download_name='processed_audio.wav'
    )

if __name__ == '__main__':
    app.run(debug=True)
