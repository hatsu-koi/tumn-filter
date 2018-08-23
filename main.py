from gensim.models import Word2Vec
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from konlpy.tag import Twitter
import numpy as np
import re

model_prefix = 'default'
wv_model = None
threshold = 0.8
configuration = {}

models = {}
filters = {}
twitter = Twitter()


class TumnSequence(Sequence):
    def __init__(self, sentences_sorted, chunk_size, batch_size):
        self.dataset = []

        last_len = len(sentences_sorted[0][0])
        chunked = [[], []]

        def process_last_chunk(sentence):
            nonlocal last_len, chunked

            last_len = len(sentence)
            max_size = len(chunked[0][-1])

            chunked[0] = np.array(pad_sequences(chunked[0], maxlen=max_size))
            chunked[1] = np.array(pad_sequences(chunked[1], maxlen=max_size))

        for (sentence, value) in sentences_sorted:
            if (last_len + chunk_size < len(sentence)) or (len(chunked[0]) >= batch_size):
                process_last_chunk(sentence)
                self.dataset.append(chunked)

                chunked = [[], []]

            chunked[0].append(sentence)
            chunked[1].append(value)

        process_last_chunk([])
        self.dataset.append(chunked)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def bind_word(sentences, w_model):
    for sentence_index in range(0, len(sentences)):

        sentences[sentence_index] = list(map(
            lambda word: w_model[word] if word in w_model.wv.vocab else np.zeros(50),
            sentences[sentence_index]
        ))

    return sentences


def load():
    with open('./fit/config.json', 'r', encoding='utf-8') as f:
        configuration = json.load(f)

    wv_model = Word2Vec.load("./fit/models/word2vec.txt")

    for model_name in ['swearwords', 'hatespeech', 'mature']:
        models[model_name] = load_model('./fit/models/%s.hdf5' % model_name)

        def filter_model(orig_paragraph):
            paragraph_mapped, orig_zipped = bind_paragraphs(orig_paragraph)

            zipped = split_sentences(orig_zipped)
            zipped = sorted(zipped, key=lambda x: len(x[1]))
            id_maps, sentences, positions = zip(*zipped) # Unzipping

            sentences = bind_word(sentences, wv_model)
            sentences_generator = TumnSequence(sentences, configuration['seq_chunk'], configuration['batch_size'])

            return_output = []

            for state_chunk in sentences_generator:
                input_chunk, start_index = state_chunk

                output = model[model_name].predict(input_chunk)

                for i, sentence in enumerate(output):
                    sentence_index = start_index + i
                    position_map = positions[sentence_index]
                    output_map = []

                    for word_index, words in enumerate(sentence):
                        if word > threshold:
                            output_map.append(position_map[word_index])

                    if len(output_map) > 0:
                        return_output.append([id_maps[sentence_index], output_map])


            return remap_to_paragraph(paragraph_mapped, return_output)

        filters["%s.%s" % (model_prefix, model_name)] = filter_model


def split_sentences(zipped_sentences):
    return [[
            zipped[0],
            *split_and_get_position(zipped[1])
        ] for zipped in zipped_sentences
    ]


def split_and_get_position(sentence):
    results = twitter.pos(sentences[sentence_index])
    words = []
    positions = []
    start = 0
    for result in results:
        if result.start() == 0:
            continue

        positions.append([start, start + len(result[0]) - 1])
        words.append("%s/%s" % result);

        start += len(result[0])

    return words, positions


def bind_paragraphs(paragraphs):
    sentence_list = []
    id_list = []

    for paragraph in paragraphs:
        sentences = []
        sentence_id = []
        total_len = 0

        for sentence in paragraph:
            sentences.append(sentence[1])
            sentence_id.append([total_len, total_len + len(sentence[1]) - 1, sentence[0]])
            total_len += len(sentence[1])

        sentence_list.append(sentences)
        id_list.append(sentence_id)

    return sentence_list, id_list


def remap_to_paragraph(output_values):
    return_values = []

    def find_sentence(paragraph_map, i):
        for key, sentence_map in enumerate(paragraph_map):
            if sentence_map[0] <= i <= sentence_map[1]:
                return key

    for output_value in output_values:
        paragraph_map, output_ranges = output_value
        start_id_key = find_sentence(paragraph_map, output_range[0])
        end_id_key = find_sentence(paragraph_map, output_range[1])

        i = start_id_key

        while i <= end_id_key:
            start = 0
            end = paragraph_map[i][1] - paragraph_map[i][0]

            if i == start_id_key:
                start = output_range[0]

            if i == end_id_key:
                end = output_range[1]

            return_values.append([paragraph_map[i][2], [start, end]])
            i += 1

    return return_values
