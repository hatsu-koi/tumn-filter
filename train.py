from functools import reduce
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Bidirectional, LSTM
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from logging import FileHandler, Formatter, StreamHandler
from os import path

import datetime
import json
import logging
import numpy as np
import os


def bind_word(sentence, w_model):
    for sentence_index in range(0, len(sentences)):

        sentences[sentence_index] = list(map(
            lambda word: w_model[word] if word in w_model.wv.vocab else w_model['.'],
            sentences[sentence_index]
        ))

    return sentences


def split_train_set(train_set):
    train_len = len(train_set)
    test_amount = floor(train_len / 10)

    return train_set[test_amount:], train_set[:test_amount]

def chunkify(zipped, chunk_size, batch_size):
    sentences_sorted = sorted(zipped, key=lambda x: len(x[0]))
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


def process_data(args, logger, dataset_name, dataset_label):
    x_set = []
    y_set = []

    try:
        with open('./fit/dataset/%s/%s' % (dataset_name, dataset_label), 'r') as f:
            dataset = json.loads(f.read())

            for data in dataset:
                x_set.append(data['content'])
                y_set.append(data['filter'])

            f.close()

    except IOError:
        logger.error("[Fit] Error while reading dataset %s!" % dataset_label)

    return x_set, y_set


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

    # Reading parsed comments
    logger.info("[Fit] Reading parsed dataset...")

    train_set = process_data(args, logger, dataset_name, "train")
    train_zipped = []
    test_set = []
    test_zipped = []
    test_from_train = True

    if path.exists("./fit/dataset/%s/%s" % (dataset_name, 'test')):
        test_set = process_data(args, logger, dataset_name, "test")
        test_zipped = zip(*test_set)
        test_from_train = False

    logger.info("[Fit] Done reading %d train set & %d test sets!" % (len(train_set), len(test_set)))

    # Creating Word embedding model
    if not path.exists("./fit/models/word2vec.txt"):
        logger.info("[Fit] Creating word2vec model...")

        w_model = Word2Vec(train_set[0], min_count=1, size=word2vec_size, iter=10, sg=0)
        w_model.save("./fit/models/word2vec.txt")

    else:
        logger.info("[Fit] Reading from saved word2vec model...")
        w_model = Word2Vec.load("./fit/models/word2vec.txt")

    # Zipping Models
    train_zipped = zip(*train_set)

    if test_from_train:
        train_zipped, test_zipped = split_train_set(train_set)

    # Preprocess input, outputs
    logger.info("[Fit] Preprocessing train dataset...")
    train_chunks = chunkify(train_zipped, seq_chunk, batch_size)

    logger.info("[Fit] Preprocessing test dataset...")
    test_chunks = chunkify(test_zipped, seq_chunk, batch_size)

    # Fit the model
    logger.info("[Fit] Fitting the model...")

    model_path = \
        "./fit/models/%s (date%s" % (dataset_name, dt.strftime("%Y%m%d")) + \
        ", epoch {epoch:02d}, loss ${val_loss:.2f}).hdf5"

    callbacks = [
        ModelCheckpoint(save_best_only=True, filepath=model_path)
    ]

    if tensorboard:
        callbacks.append(TensorBoard(log_dir=tensorboard))

    model.fit_generator(
        generator=[(chunk['x_list'], chunk['y_list']) for chunk in train_chunks],
        validation_data=[(chunk['x_list'], chunk['y_list']) for chunk in test_chunks],
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
        with open('./fit/config.json', 'w') as f:
            json.dump(default_configuration, f, indent=4)

    with open('./fit/config.json', 'r') as f:
        configuration = json.load(f)

    configuration.update(default_configuration)

    dataset_basepath = "./fit/dataset/%s/" % configuration['dataset_name']
    check_and_create_dir(dataset_basepath)

    if not path.exists(dataset_basepath + "train"):
        print("Dataset not given!")
        exit()

    run(configuration)
