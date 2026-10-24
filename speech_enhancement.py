import torch
import torch.nn as nn
import librosa
import soundfile as sf
import numpy as np

class SpeechEnhancementCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

device=torch.device("cpu")

model=SpeechEnhancementCNN().to(device)

model.load_state_dict(
    torch.load(
        "speech_enhancement_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

#loading the noisy audio

audio_path=input("Entere the path of noisy audio file:")
noisy_audio,sr =librosa.load(audio_path,sr=16000,mono=True)

#converting the noisy audioo
noisy_stft=librosa.stft(noisy_audio,n_fft=512,hop_length=128)
noisy_mag=np.abs(noisy_stft)

#spectrogram to tensor
input_tensor=torch.tensor(
    noisy_mag,dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(device)


#enhancing
with torch.no_grad():
    predicted_mask=model(input_tensor)

predicted_mask=predicted_mask.squeeze().cpu().numpy()

enhanced_mag=noisy_mag*predicted_mask

enhanced_stft=enhanced_mag*np.exp(1j*np.angle(noisy_stft))

#converting the enhanced spectrogram back to audio wave form
enhanced_audio=librosa.istft(enhanced_stft,
                             hop_length=128,
                             length=len(noisy_audio)
                             )

save_choice = input("Save the enhanced audio? (Y/N): ").strip().lower()

if save_choice == "y":
    output_path = input("Enter output filename (e.g., enhanced.wav): ").strip()
    sf.write(output_path, enhanced_audio, 16000)
    print(f"Enhanced audio saved to: {output_path}")
else:
    print("Audio not saved.")