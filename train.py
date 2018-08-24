from functools import reduce
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Bidirectional, LSTM, Reshape
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from logging import FileHandler, Formatter, StreamHandler
from os import path
from main import bind_word, TumnSequence

import argparse
import datetime
import json
import logging
import numpy as np
import os
import random


def split_train_set(train_set):
    random.shuffle(train_set)
    train_len = len(train_set)
    test_amount = int(train_len / 10)

    return train_set[test_amount:], train_set[:test_amount]


def process_data(args, logger, dataset_name, dataset_label):
    x_set = []
    y_set = []

    try:
        with open('./fit/dataset/%s/%s' % (dataset_name, dataset_label), 'r', encoding='utf-8') as f:
            dataset = json.loads(f.read())

            for data in dataset:
                x_set.append(data['content'])
                y_set.append([1.0 if filter else 0.0 for filter in data['filter']])

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
    model = None

    if args['load']:
        model = load_model(args['load'])
        logger.info("[Fit] Loaded model from %s." % args['load'])

    else:
        model = Sequential([
            Bidirectional(
                LSTM(5, activation='relu', dropout=0.05, return_sequences=True),

                input_shape=(None, word2vec_size)
            ),
            LSTM(10, activation='relu', dropout=0.1, return_sequences=True),
            LSTM(1, activation='sigmoid', dropout=0.05, return_sequences=True),
            Reshape((-1, ))
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

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
    train_generator = TumnSequence(train_zipped, seq_chunk, batch_size)

    logger.info("[Fit] Preprocessing test dataset...")
    test_generator = TumnSequence(test_zipped, seq_chunk, batch_size)

    logger.info("[Fit] Done generating %d train set & %d test sets!" % (len(train_zipped), len(test_zipped)))

    # Fit the model
    logger.info("[Fit] Fitting the model...")

    model_path = \
        "./fit/models/%s (Date %s" % (dataset_name, datetime.datetime.now().strftime("%m-%d %Hh %Mm ")) + \
        ", Epoch {epoch:02d}, Acc {val_categorical_accuracy:.3f}, Loss {val_loss:.3f}).hdf5"

    callbacks = [
        ModelCheckpoint(filepath=model_path)
    ]

    if tensorboard:
        callbacks.append(TensorBoard(log_dir=tensorboard))

    model.fit_generator(
        generator=train_generator, validation_data=test_generator,
        epochs=epoch, verbose=verbosity, callbacks=callbacks,
        shuffle=True
    )


def check_and_create_dir(dir_name):
    if not path.isdir(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--load', dest='load', metavar='[hd5]', help='Resume from hd5 file.')

    args = parser.parse_args()

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
        default_configuration.update(configuration)

        configuration = default_configuration

    dataset_basepath = "./fit/dataset/%s/" % configuration['dataset_name']
    check_and_create_dir(dataset_basepath)

    if not path.exists(dataset_basepath + "train"):
        print("Dataset not given!")
        exit()

    if args.load:
        configuration['load'] = args.load

    else:
        configuration['load'] = None

    run(configuration)
