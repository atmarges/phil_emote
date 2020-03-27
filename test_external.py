# Use this template script to load external files
# Be sure to modify the model_dir to the location of the new models

from phil_emote.models import PhilEmoteModel

dataset = [
    "Nakakaasar nman! ahhh... ang dami dami kong gagawin... kulang time...",
    "Asar... Badtrip!",
    "May bagong bagyo ang namataan sa kanlurang bahagi ng bansa.",
    "Thank you po Lord Jesus at gumana din sya.",
    "Congrats po sa inyong lahat!",
    "Ano, sasagot ka pa? Ha?",
    "New strain of corona virus was reported to be spreading worldwide."
]

[print(i) for i in dataset]

import os
model_dir = 'D:/location/of/downloaded/files/'
json_file = os.path.join(model_dir, 'emoji_emotion_classification_cnn.json')
weight_file = os.path.join(model_dir, 'emoji_emotion_classification_cnn.h5')

model = PhilEmoteModel(json_file=json_file, weight_file=weight_file)
pred = model.predict_dataset(dataset, output_type='emotion')
print(pred)
