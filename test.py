from phil_emote.models import PhilEmoteModel

dataset = [
    "Nakakaasar nman! ahhh... ang dami dami kong gagawin... kulang time...",
    "Asar... Badtrip!",
    "May bagong bagyo ang namataan sa kanlurang bahagi ng bansa.",
    "Thank you po Lord Jesus at gumana din sya.",
    "Congrats po sa inyong lahat!",
    "Ano, sasagot ka pa? Ha?"
]

model = PhilEmoteModel(json_file=json_file, weight_file=weight_file)
pred = model.predict_dataset(dataset, output_type='emotion')
print(pred)
