from functools import reduce
from gensim.models import Word2Vec
from tf.keras.callbacks import TensorBoard
from tf.keras.layers import Input, LSTM, Dense, Dropout
from tf.keras.models import Model, Sequential
from tf.keras.preprocessing.sequence import pad_sequences
from logging import FileHandler, Formatter, StreamHandler
from os import path

import json
import logging
import numpy as np
import os


def bind_word(sentence, w_model):
    for sentence_index in range(0, len(sentences)):

        sentences[sentence_index] = list(filter(
            lambda word: word in w_model.wv.vocab,
            sentences[sentence_index]
        ))

        for word_index in range(0, len(sentences[sentence_index])):
            original_text = sentences[sentence_index][word_index]

            sentences[sentence_index][word_index] = w_model[original_text]

    return sentences


def chunkify(sentences, values, chunk_size, batch_size):
    zipped = zip(sentences, values)
    sentences_sorted = sorted(sentences, key=lambda x: len(x[0]))
    last_len = 0
    chunked = [
        {
            'x_list': [],
            'y_list': [],
            'max_size': 0
        }
    ]

    def process_last_chunk():
        last_len = len(sentence)
        max_size = len(chunked[-1]['x_list'][-1])

        chunked[-1]['x_list'] = np.array(pad_sequence(chunked[-1]['x_list'], maxlen=max_size))
        chunked[-1]['y_list'] = np.array(pad_sequence(chunked[-1]['y_list'], maxlen=max_size))
        chunked[-1]['max_size'] = max_size

    for (sentence, value) in sentences_sorted:
        if (last_len + chunk_size < len(sentence)) or (len(chunked[-1]) > batch_size):
            process_last_chunk()

            chunked.append({
                'x_list': [],
                'y_list': [],
                'max_size': 0
            })

        chunked[-1]['x_list'].append(sentence)
        chunked[-1]['y_list'].append(value)

    process_last_chunk()

    return chunked


def process_data(args, logger, dataset_label):
    x_set = []
    y_set = []

    try:
        with open('./fit/dataset/%s' % dataset_label, 'r') as f:
            dataset = json.loads(f.read())

            for data in dataset:
                x_set.append(bind_word(data[0]))
                y_set.append(data[1])

            f.close()

    except IOError:
        logger.error("[Fit] Error while reading dataset %s!" % dataset_label)

    return x_set, y_set


def run(args):
    if path.exists("./fit/logs/deep-news.log"):
        os.remove("./fit/logs/deep-news.log")

    file_formatter = Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    file_handler = FileHandler('./fit/logs/deep-news.log')
    file_handler.setFormatter(file_formatter)

    # stream_handler = ChalkHandler()
    stream_handler = StreamHandler()

    logger = logging.getLogger("DeepNews")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    epoch = args.epoch
    seq_chunk = args.seq_chunk
    batch_size = args.batch_size
    word2vec_size = args.word2vec_size
    verbosity = args.verbosity
    tensorboard = args.tensorboard

    # Creating news model
    logger.info("[Fit] Generating model...")

    model = Sequential([
        Bidirectional(
            LSTM(
                10, activation='relu', name='lstm1',
                dropout=0.2, return_sequences=True,
                input_shape=(None, word2vec_size)
            )
        ),

        LSTM(20, activation='relu', name='lstm2', dropout=0.2, return_sequences=True),
        LSTM(1, activation='relu', name='lstm2', dropout=0.2, return_sequences=True)
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Reading parsed news

    logger.info("[Fit] Reading parsed dataset...")

    [x_train, y_train] = process_data(args, logger, "train")
    [x_test, y_test] = process_data(args, logger, "test")
    logger.info("[Fit] Done reading %d train set & %d test sets!" % (len(x_train), len(x_test)))

    # Creating Word embedding model
    if not path.exists("./fit/models/word2vec.txt"):
        logger.info("[Fit] Creating word2vec model...")

        w_model = Word2Vec(x_train, min_count=1, size=word2vec_size, iter=10, sg=0)
        w_model.save("./fit/models/word2vec.txt")

    else:
        logger.info("[Fit] Reading from saved word2vec model...")
        w_model = Word2Vec.load("./fit/models/word2vec.txt")

    # Preprocess input, outputs
    logger.info("[Fit] Preprocessing train dataset...")
    train_chunks = chunkify(bind_word(x_train, w_model), y_train, seq_chunk, batch_size)

    logger.info("[Fit] Preprocessing test dataset...")
    test_chunks = chunkify(bind_word(x_test, w_model), y_test, seq_chunk, batch_size)

    # Fit the model
    logger.info("[Fit] Fitting the model...")

    callbacks = []

    if tensorboard:
        callbacks.append(TensorBoard(log_dir=tensorboard))

    model.fit_generator(
        generator=[(chunk['x_list'], chunk['y_list']) for chunk in train_chunks],
        validation_data=[(chunk['x_list'], chunk['y_list']) for chunk in test_chunks],
        epochs=epoch, verbose=verbosity, callbacks=callbacks
    )

    model.save("./fit/models/filter.h5")


def check_and_create_dir(dir_name):
    if not path.isdir(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":
    check_and_create_dir("./fit/")
    check_and_create_dir("./fit/dataset/")
    check_and_create_dir("./fit/dataset/test/")
    check_and_create_dir("./fit/dataset/train/")
    check_and_create_dir("./fit/models/")
    check_and_create_dir("./fit/logs/")
    check_and_create_dir("./fit/logs/model/")

    default_configuration = {
        'epoch': 20,
        'seq_chunk': 10,
        'batch_size': 128,
        'word2vec_size': 50,
        'verbosity': 1,
        'tensorboard': './fit/logs/model/'
    }

    if not path.exists("./fit/config.json"):
        with open('./fit/config.json', 'w') as f:
            json.dump(default_configuration, f, indent=4)

    with open('./fit/config.json', 'r') as f:
        configuration = json.load(f)

    configuration.update(default_configuration)
    run(configuration)
