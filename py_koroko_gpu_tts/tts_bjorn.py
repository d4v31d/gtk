
### tts_bjorn

# .\.venv\scripts\activate.bat

# [björn](/bjʊɳ/)
# [Kokoro](/kˈOkəɹO/) 

from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import time

pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice, reference above.

# This text is for demonstration purposes only, unseen during training
text = '''
My son Björn is a great kid. He loves to play outside and explore nature. His favorite animal is the bear, which he often pretends to be while running around the yard. Björn is curious about everything and always asks questions about the world around him. He enjoys reading books about animals and adventures, and he dreams of becoming a scientist one day.
Björn's name means "bear" in Swedish, which is fitting because he has a strong and adventurous spirit. He is also very kind-hearted and loves to help others. Whether it's picking up litter in the park or helping his friends with their homework, Björn always tries to make the world a better place.
Björn is pronunced as [björn](/bjʊɳ/) in Swedish, which captures the essence of his name. He is a unique and wonderful child who brings joy to everyone around him.
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
    print(i)  # i => index
    print(gs) # gs => graphemes/text
    print(ps) # ps => phonemes
    #display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'tts_koroko_{ts}-{i}.mp3', audio, 24000) # save each audio file

    