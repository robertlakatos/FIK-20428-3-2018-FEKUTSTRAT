import etl
import json
import numpy
import pandas
import pickle
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Read source
soruce = "sources/source_idydptc_0_10000.xlsx"
data = pandas.read_excel(soruce)

# Create corpus
etl.create_corpus(output_file_name="sources/corpus.txt",
                  data=data["obsval"])

# Create word2vec
embedding_dim = 16
etl.create_word2vec(input_file_name="sources/corpus.txt",
                    output_dir="sources",
                    embedding_dim=embedding_dim,
                    window=10,
                    iter=10,
                    min_count=1)

# Create training data
templates = json.load(open("config/config_pattern_templates.json",
                           "r",
                           encoding="utf-8"))
etl.create_train_to_pattern(output_file_name="sources/corpus_train.txt",
                            data=data,
                            templates=templates)

# Load trainig data
x_train = etl.load_train_to_pattern("sources/corpus_train.txt")

# Tokenize training data
my_filters = '"#$&()*+/:;<=>?@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(num_words=10000,
                      filters=my_filters)
tokenizer.fit_on_texts(x_train)
print("TOKENIZER FITTED ON TEXTS")
x_train = tokenizer.texts_to_sequences(x_train)
print("TOKENIZED TEXT TO SEQUENCES")
samples = [item["sample"] for item in templates]
tokenized_samples = tokenizer.texts_to_sequences(samples)
print("TOKENIZED ENVIROMENT")

maxlen = max(len(x.split(" ")) for x in samples)
tokenizer_json = tokenizer.to_json()
tokenizer_json = {
    "max_lenght": maxlen,
    "vocab_size": len(tokenizer.word_index) + 1,
    "tokens": tokenizer_json
}

with open('sources/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
print("SAVED TOKENS")

# Padding data
x_train = pad_sequences(x_train,
                        padding='post',
                        maxlen=maxlen)

tokenized_samples = pad_sequences(tokenized_samples,
                                  padding='post',
                                  maxlen=maxlen)
print("PADDED ENVIROMENT")

# Shuffeling and splitting trainig data
y_train = numpy.copy(x_train).astype(numpy.int32)

x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=1000)
print("SHUFFLED AND SPLITED ENVIROMENT")

# Loading embedded vectors
embedding_matrix = etl.load_embedding_space(input_file_name="sources/word2vec_pattern.vec",
                                            embedding_dim=embedding_dim,
                                            tokenizer=tokenizer)

# Creating model
model = models.Sequential()

model.add(layers.Embedding(len(tokenizer.index_word)+1,
                           output_dim=embedding_dim,
                           input_length=maxlen,
                           weights=[embedding_matrix],
                           trainable=True))
model.add(layers.LSTM(embedding_dim))
model.add(layers.Dense(len(samples), activation="sigmoid"))

model.compile(loss="mse",
              optimizer='adam',
              metrics=['mae'])

model.summary()

# Converting data
weights = model.layers[0].get_weights()[0]
y_train_sim = etl.create_similarity_matrix(data=y_train,
                                           weights=weights,
                                           samples=tokenized_samples)

y_test_sim = etl.create_similarity_matrix(data=y_test,
                                          weights=weights,
                                          samples=tokenized_samples)

# Training model
checkpoint = ModelCheckpoint('models/model_{epoch:03d}-{mae:03f}-{val_mae:03f}_pattern_recgoniser.h5',
                             verbose=1,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='auto')

history = model.fit(x_train,
                    y_train_sim,
                    epochs=5,
                    batch_size=100,
                    validation_data=(x_test, y_test_sim),
                    callbacks=[checkpoint],
                    verbose=1)

print("TARAINIG COMPLETED")
