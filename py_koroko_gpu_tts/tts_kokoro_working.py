
### tts_bjorn

# .\.venv\scripts\activate.bat

### phoneme / fonem
# [björn](/bjʊɳ/)
# [Kokoro](/kˈOkəɹO/) 

from kokoro import KPipeline
# from IPython.display import display, Audio
import soundfile as sf
import torch
import time

pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice, reference above.

# This text is for demonstration purposes only, unseen during training
# Björn is pronunced as [björn](/bjʊɳ/) in Swedish.
# [aaaaaaaa](/a:a:a:a:a:a:a:a:/), I got stuck in a hole.
# [aaaaooow](/aaaaooowww/), I got stuck in a hole.
# [hahahahaha](/h'Oh'Oh'Oh'Oh'O/), I got stuck in a hole.

text = '''

hähä, I got stuck in a hole.

'''

# 4️⃣ Generate, display, and save audio files in a loop.
generator = pipeline(
    text,
    voice='af_heart', # <= change voice here
    speed=1,
    split_pattern=r'\n+'
)

# Alternatively, load voice tensor directly:
# voice_tensor = torch.load('path/to/voice.pt', weights_only=True)
# generator = pipeline(
#     text, voice=voice_tensor,
#     speed=1, split_pattern=r'\n+'
# )

ts = time.strftime("%Y%m%d%H%M%S")

for i, (gs, ps, audio) in enumerate(generator):
    print(f"index: {i}")  # i => index
    print(f"graphemes/text: {gs}") # gs => graphemes/text
    print(f"phonemes: {ps}") # ps => phonemes
    #display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'tts_koroko_{ts}-{i}.mp3', audio, 24000) # save each audio file

    