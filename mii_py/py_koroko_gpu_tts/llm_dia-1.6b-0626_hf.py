
### llm_dia-1.6b-0626_hf

## its not working good yet, but it is working!!

from transformers import AutoProcessor, DiaForConditionalGeneration


torch_device = "cuda"
model_checkpoint = "nari-labs/Dia-1.6B-0626"

# (laughs), 
# (clears throat), 
# (sighs), 
# (gasps), 
# (coughs), 
# (singing), 
# (sings), 
# (mumbles), 
# (beep), 
# (groans), 
# (sniffs), 
# (claps), 
# (screams), 
# (inhales), 
# (exhales), 
# (applause), 
# (burps), 
# (humming), 
# (sneezes), 
# (chuckle), 
# (whistles)

text = [
    "[S1] did you see that stroke, its soo close to be a hole in one."
    "[S2] Wow. Amazing. (applause)  Now its my turn to try."
    "[S1] That was good a bit to long though."
    "[S2] I will chip it in for sure."
]

# text = [
#     "[S1] (screams) did you see that stroke, its soo close to be a hole in one."
#     "[S2] Wow. Amazing. (applause)  Now its my turn to try. (chuckle)"
#     "[S1] That was good a bit to long though. (whistles)"
#     "[S2] I will chip it in for sure. (laughs)"
# ]

# text = [
#     "[S1] Dia is an open weights text to dialogue model."
#     "[S2] You get full control over scripts and voices."
#     "[S1] Wow. Amazing. (laughs)"
#     "[S2] Try it now on Git hub or Hugging Face."
# ]

processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(
    text=text
    , padding=True
    , return_tensors="pt"
).to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(
    **inputs, max_new_tokens=3072, guidance_scale=3.0, temperature=1.8, top_p=0.90, top_k=45
)

outputs = processor.batch_decode(outputs)
processor.save_audio(outputs, "example.mp3")
