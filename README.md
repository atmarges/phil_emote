# phil_emote
A ready-to-use sentiment analysis and emotion classification model for Philippine tweets.


Using tweets collected from the Philippine, an LSTM model was trained for sentiment analysis
and emotion classification. This model was trained into 11 classes specifically:

|Emoji	|	Emotion		|	Sentiment	|
|-------|---------------|---------------|
|	📝	|	neutral		|	neutral		|
|	😄	|	happy		|	positive	|
|	😌	|	relief		|	positive	|
|	😑	|	unammused	|	negative	|
|	😘	|	love		|	positive	|
|	😜	|	playful		|	positive	|
|	😞	|	sad			|	negative	|
|	😡	|	angry		|	negative	|
|	😱	|	shocked		|	negative	|
|	😷	|	sick		|	negative	|
|	🤔	|	pondering	|	neutral		|

Labels used were the 11 emojis which were selected by clustering a set of facial expression emojis
into ten clusters. Then, tweets containing news were added to the dataset to represent the neutral class.


## Quickstart
Create an instance of `PhilEmoteModel`. Then, pass a list of tweets to `predict_dataset` to
predict its respective emotion. Alternatively, the model can be used to predict sentiment by 
setting `output_type = 'sentiment'`. Also, setting `output_type = 'emoji'` will output the 
emoji labels used.

```python
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

```