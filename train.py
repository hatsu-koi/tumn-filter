from functools import reduce
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Bidirectional, LSTM, Reshape
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from logging import FileHandler, Formatter, StreamHandler
from os import path

import datetime
import json
import logging
import numpy as np
import os


def bind_word(sentences, w_model):
    for sentence_index in range(0, len(sentences)):

        sentences[sentence_index] = list(map(
            lambda word: w_model[word] if word in w_model.wv.vocab else np.zeros(50),
            sentences[sentence_index]
        ))

    return sentences


def split_train_set(train_set):
    train_len = len(train_set)
    test_amount = int(train_len / 10)

    return train_set[test_amount:], train_set[:test_amount]


def len_chunk(sentences_sorted, chunk_size, batch_size):
    last_len = len(sentences_sorted[0][0])
    chunk = 0
    yield_amount = 1

    for (sentence, value) in sentences_sorted:
        if (last_len + chunk_size < len(sentence)) or (chunk > batch_size):
            yield_amount += 1
            chunk = 0
            last_len = len(sentence)

        chunk += 1

    return yield_amount


def chunkify(sentences_sorted, chunk_size, batch_size):
    last_len = len(sentences_sorted[0][0])
    chunked = [[], []]

    def process_last_chunk(sentence):
        nonlocal last_len, chunked

        last_len = len(sentence)
        max_size = len(chunked[0][-1])

        chunked[0] = np.array(pad_sequences(chunked[0], maxlen=max_size))
        chunked[1] = np.array(pad_sequences(chunked[1], maxlen=max_size))

    for (sentence, value) in sentences_sorted:
        if (last_len + chunk_size < len(sentence)) or (len(chunked[0]) > batch_size):
            process_last_chunk(sentence)
            yield chunked

            chunked = [[], []]

        chunked[0].append(sentence)
        chunked[1].append(value)

    process_last_chunk([])
    yield chunked


def process_data(args, logger, dataset_name, dataset_label):
    x_set = []
    y_set = []

    try:
        with open('./fit/dataset/%s/%s' % (dataset_name, dataset_label), 'r', encoding='utf-8') as f:
            dataset = json.loads(f.read())

            for data in dataset:
                x_set.append(data['content'])
                y_set.append(data['filter'])

            f.close()

    except IOError:
        logger.error("[Fit] Error while reading dataset %s!" % dataset_label)

    return [x_set, y_set]


def run(args):
    if path.exists("./fit/logs/tumn.log"):
        os.remove("./fit/logs/tumn.log")

    file_formatter = Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    file_handler = FileHandler('./fit/logs/tumn.log')
    file_handler.setFormatter(file_formatter)

    # stream_handler = ChalkHandler()
    stream_handler = StreamHandler()

    logger = logging.getLogger("Tumn")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    dataset_name = args['dataset_name']
    epoch = args['epoch']
    seq_chunk = args['seq_chunk']
    batch_size = args['batch_size']
    word2vec_size = args['word2vec_size']
    verbosity = args['verbosity']
    tensorboard = args['tensorboard']

    # Creating tumn model
    logger.info("[Fit] Generating model...")

    model = Sequential([
        Bidirectional(
            LSTM(10, activation='relu', dropout=0.1, return_sequences=True),

            input_shape=(None, word2vec_size)
        ),
        LSTM(20, activation='relu', dropout=0.15, return_sequences=True),
        LSTM(1, activation='relu', dropout=0.1, return_sequences=True),
        Reshape((-1, ))
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Reading parsed comments
    logger.info("[Fit] Reading parsed dataset...")

    train_set = process_data(args, logger, dataset_name, "train")
    train_zipped = []
    test_set = [[], []]
    test_zipped = []
    test_from_train = True

    if path.exists("./fit/dataset/%s/%s" % (dataset_name, 'test')):
        test_set = process_data(args, logger, dataset_name, "test")
        test_zipped = []
        test_from_train = False

    else:
        logger.info("[Fit] No validation set found. It will split train set into train set and validation set.")

    logger.info("[Fit] Done reading %d train set & %d test sets!" % (len(train_set[0]), len(test_set[0])))

    # Creating Word embedding model
    if not path.exists("./fit/models/word2vec.txt"):
        logger.info("[Fit] Creating word2vec model...")

        w_model = Word2Vec(train_set[0], min_count=1, size=word2vec_size, iter=10, sg=0)
        w_model.save("./fit/models/word2vec.txt")

    else:
        logger.info("[Fit] Reading from saved word2vec model...")
        w_model = Word2Vec.load("./fit/models/word2vec.txt")

    train_set[0] = bind_word(train_set[0], w_model)
    train_zipped = list(zip(*train_set))

    # Zipping Models
    if test_from_train:
        train_zipped, test_zipped = split_train_set(train_zipped)

    else:
        test_set[0] = bind_word(test_set[0], w_model)
        test_zipped = list(zip(*test_set))

    train_zipped = sorted(train_zipped, key=lambda zip: len(zip[0]))
    test_zipped = sorted(test_zipped, key=lambda zip: len(zip[0]))

    # Preprocess input, outputs
    logger.info("[Fit] Preprocessing train dataset...")
    train_len = len_chunk(train_zipped, seq_chunk, batch_size)
    train_generator = chunkify(train_zipped, seq_chunk, batch_size)

    logger.info("[Fit] Preprocessing test dataset...")
    test_len = len_chunk(test_zipped, seq_chunk, batch_size)
    test_generator = chunkify(test_zipped, seq_chunk, batch_size)

    # Fit the model
    logger.info("[Fit] Fitting the model...")

    model_path = \
        "./fit/models/%s (date%s" % (dataset_name, datetime.datetime.now().strftime("%Y%m%d")) + \
        ", epoch {epoch:02d}, loss ${val_loss:.2f}).hdf5"

    callbacks = [
        ModelCheckpoint(save_best_only=True, filepath=model_path)
    ]

    if tensorboard:
        callbacks.append(TensorBoard(log_dir=tensorboard))

    model.fit_generator(
        generator=train_generator, validation_data=test_generator,
        steps_per_epoch=train_len, validation_steps=test_len,
        epochs=epoch, verbose=verbosity, callbacks=callbacks
    )


def check_and_create_dir(dir_name):
    if not path.isdir(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":
    check_and_create_dir("./fit/")
    check_and_create_dir("./fit/dataset/")
    check_and_create_dir("./fit/models/")
    check_and_create_dir("./fit/logs/")
    check_and_create_dir("./fit/logs/model/")

    default_configuration = {
        'epoch': 20,
        'seq_chunk': 10,
        'batch_size': 128,
        'word2vec_size': 50,
        'verbosity': 1,
        'tensorboard': './fit/logs/model/',
        'dataset_name': 'swearwords'
    }

    if not path.exists("./fit/config.json"):
        with open('./fit/config.json', 'w', encoding='utf-8') as f:
            json.dump(default_configuration, f, indent=4)

    with open('./fit/config.json', 'r', encoding='utf-8') as f:
        configuration = json.load(f)

    configuration.update(default_configuration)

    dataset_basepath = "./fit/dataset/%s/" % configuration['dataset_name']
    check_and_create_dir(dataset_basepath)

    if not path.exists(dataset_basepath + "train"):
        print("Dataset not given!")
        exit()

    run(configuration)
