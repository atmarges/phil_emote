from phil_emote.models import PhilEmoteModel

dataset = [
    "Nakakaasar nman! ahhh... ang dami dami kong gagawin... kulang time...",
    "Asar... Badtrip!",
    "May bagong bagyo ang namataan sa kanlurang bahagi ng bansa.",
    "Thank you po Lord Jesus at gumana din sya.",
    "Congrats po sa inyong lahat!",
]

model = PhilEmoteModel()
pred = model.predict_dataset(dataset, output_type='emotion')
print(pred)

#model = phil_emote.load_model()
#output = phil_emote.predict_dataset(dataset, model)
# print(output)
