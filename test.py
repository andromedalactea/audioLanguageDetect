import whisper


model = whisper.load_model("large", device="cpu")
_ = model.half()
_ = model.cuda()

# exception without following code
# reason : model.py -> line 31 -> super().forward(x.float()).type(x.dtype)
for m in model.modules():
    if isinstance(m, whisper.model.LayerNorm):
        m.float()