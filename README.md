# Speech Enhancement and Noise Suppression

A deep learning-based speech enhancement system built using **PyTorch and Convolutional Neural Networks (CNNs)** to suppress background noise and improve speech quality.

## Overview

This project enhances noisy speech recordings using a CNN-based model that predicts a mask from the noisy speech spectrogram. The predicted mask is applied to the noisy magnitude spectrogram to reduce background noise.

## Technologies Used

- Python
- PyTorch
- Librosa
- NumPy
- SciPy
- SoundFile
- Fourier Transform and Short-Time Fourier Transform (STFT)

## How It Works

1. Load a noisy audio recording.
2. Resample the audio to 16 kHz.
3. Convert the audio into a frequency-domain representation using STFT.
4. Use a trained CNN to predict a noise-suppression mask.
5. Apply the mask to the noisy magnitude spectrogram.
6. Reconstruct the enhanced audio using the inverse STFT.
7. Save the enhanced recording as a WAV file.

## Project Files

- `speech_enhancement.py` — Script for enhancing noisy audio recordings.
- `speech_Enhancement_training.ipynb` — Notebook containing the model training workflow.
- `speech_enhancement_model.pth` — Trained PyTorch model weights.

## How to Run

Install the required libraries:

```bash
pip install torch librosa numpy scipy soundfile
```

Run the enhancement script:

```bash
python speech_enhancement.py
```

Enter the path to a noisy audio file when prompted. Choose whether to save the enhanced recording and provide an output filename.

## Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Framework:** PyTorch
- **Sample rate:** 16,000 Hz
- **FFT size:** 512
- **Hop length:** 128 samples
- **Enhancement method:** Magnitude spectrogram masking

## Future Improvements

- Improve speech quality while preserving intelligibility.
- Evaluate performance using objective speech-quality metrics.
- Test the model on diverse real-world noise conditions.

## Author

**Manikan1a**

GitHub: [@Manikan1a](https://github.com/Manikan1a)
