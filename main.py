from gensim.models import Word2Vec
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from konlpy.tag import Okt
from os import path
import json
import numpy as np
import re
import tensorflow as tf
import time

model_prefix = 'default'
wv_model = None
# Oops, already trained model with word2vec, but fasttext is needed in kNN
ft_model = None
threshold = 0.8
min_length = 25
configuration = {}
hangul_regex = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')

models = {}
filters = {}
twitter = Okt()

keywords = {
    'swearwords': {},
    'mature': {},
    'hatespeech': {}
}

class TumnSequence(Sequence):
    def __init__(self, sentences_sorted, chunk_size, batch_size, value_disable_pad=False):
        self.dataset = []

        last_len = len(sentences_sorted[0][0])
        chunked = [[], []]

        def process_last_chunk(sentence):
            nonlocal last_len, chunked

            last_len = len(sentence)
            max_size = len(chunked[0][-1])

            chunked[0] = np.array(pad_sequences(chunked[0], maxlen=max_size))

            if not value_disable_pad:
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
    filter_path = path.dirname(path.abspath(__file__))

    with open(path.join(filter_path, 'fit/config.json'), 'r', encoding='utf-8') as f:
        configuration = json.load(f)

    with open(path.join(filter_path, 'fit/models/words.json'), 'r', encoding='utf-8') as f:
        keywords = json.load(f)

    wv_model = Word2Vec.load(path.join(filter_path, "fit/models/word2vec.txt"))

    def prepare_sharedres(orig_paragraph):
        sharedres = {}
        start_time = time.time()

        # paragraph_mapped
        # [[[0, 3, id], [4, 7, id2], [8, 10, id3]], [0, 3, id4], [0, 2, id5]]

        # paragraph_text
        # [['text', 'text', '...'], ['text'], '...']

        # paragraph_zipped
        """
            [
                [['text', 'text', '...'], [[0, 3, id], [4, 7, id2], [8, 10, id3]]],
                [['text'], [[0, 3, id4]]],
                [['...'], [[0, 2, id5]]]
            ]
        """

        paragraph_text, paragraph_mapped = bind_paragraphs(orig_paragraph)
        if len(paragraph_text) <= 0:
            return None

        paragraph_zipped = zip(paragraph_text, paragraph_mapped)

        # sentence_zipped
        # Zip of (paragraph_mapped, sentences, tag_positions)
        """
            [
                [[[0, 3, id], [4, 7, id2], [8, 10, id3]], ['te/Noun', 'xttext/Verb', '.../Punctuation'], [[0, 1], [2, 7], [8, 10]]],
                [[[0, 3, id4]], ['text/Noun'], [[0, 3]]],
                [[[0, 2, id5]], ['.../Punctuation'], [[0, 2]]]
            ]
        """
        sentence_zipped = split_sentences(paragraph_zipped)
        sorted_sentence_zipped = sorted(sentence_zipped, key=lambda x: len(x[1]))

        # id_maps
        # [ [[0, 3, id], [4, 7, id2], [8, 10, id3]], [[0, 3, id4]], [[0, 2, id5]] ]

        # sentences
        # [ ["te/Noun", "xttext/Verb", ".../Punctuation"], ["text/Noun"], [".../Punctuation"] ]

        # positions
        # [ [[0, 1], [2, 7], [8, 10]], [[0, 3]], [[0, 2]] ]

        # Unzip sorted into three parts
        id_maps, sentences, positions = zip(*sorted_sentence_zipped)

        sharedres['sentences'] = list(sentences)
        sharedres['id_maps'] = list(id_maps)
        sharedres['positions'] = list(positions)
        sharedres['paragraph_mapped'] = paragraph_mapped

        # Replace tags with word2vec vector
        sentences = bind_word(list(sentences), wv_model)
        sentences = list(zip(sentences, range(len(sentences))))

        # Network Input
        sentences_generator = TumnSequence(sentences, configuration['seq_chunk'], configuration['batch_size'], value_disable_pad=True)
        sharedres['generator'] = sentences_generator

        print("Preprocessed %d sentences in %d seconds." % (len(sentences), time.time() - start_time))
        return sharedres

    filters['__prepare_sharedres'] = prepare_sharedres
    graph = tf.get_default_graph()

    def filter_model(mname, orig_paragraph, sharedres):
        if sharedres is None:
            return []

        start_time = time.time()
        sentences_generator = sharedres['generator']
        positions = sharedres['positions']
        id_maps = sharedres['id_maps']
        paragraph_mapped = sharedres['paragraph_mapped']

        return_output = []

        for sentences_generator_index in range(len(sentences_generator)):
            state_chunk = sentences_generator[sentences_generator_index]
            input_chunk, sentence_indexes = state_chunk

            with graph.as_default():
                output = models[mname].predict(input_chunk)

            for i, sentence in enumerate(output):
                sentence_index = sentence_indexes[i]
                position_map = positions[sentence_index]

                # output_map
                # Array of ranges, which will be filtered
                output_map = []

                for word_index, words_predict in enumerate(sentence):
                    if words_predict > threshold and word_index < len(position_map):
                        output_map.append(position_map[word_index])

                if len(output_map) > 0:
                    return_output.append([id_maps[sentence_index], output_map])

            for sentence_index, s in enumerate(sharedres['sentences']):
                output_map = []
                for j, word in enumerate(s):
                    if word in keywords[mname]:
                        output_map.append(positions[sentence_index][j])

                if len(output_map) > 0:
                    return_output.append([id_maps[sentence_index], output_map])


        final_output = remap_to_paragraph(return_output)
        print("Processed %s in %d seconds." % (mname, time.time() - start_time))
        return final_output

    for model_name in ['swearwords', 'hatespeech', 'mature']:
        models[model_name] = load_model(path.join(filter_path, 'fit/models/%s.hdf5' % model_name))
        models[model_name]._make_predict_function()

        # Closure T_T
        def filter_closure(mname):
            return lambda x, y: filter_model(mname, x, y)

        filters["%s.%s" % (model_prefix, model_name)] = filter_closure(model_name)


# Zip paragraph_mapped, sentences, tag_positions
def split_sentences(zipped_sentences):
    return [[
            zipped[1],
            *split_and_get_position(zipped[0])
        ] for zipped in zipped_sentences]


# Split text with Twitter POS Tagging
# Returns Splitted tags, Index mapping data for Tag <-> Word
def split_and_get_position(sentences):
    results = twitter.pos(''.join(sentences), norm=True, stem=True)
    words = []
    positions = []
    for result in results:
        positions.append([int(result[2]), int(result[2]) + int(result[3]) - 1])
        words.append("%s/%s" % (result[0], result[1]))

    return words, positions


# Merge ParagraphFragments into Sentences
# Returns Sentences, Index mapping data for Sentence <-> ParagraphFragments
def bind_paragraphs(paragraphs):
    sentence_list = []
    id_list = []

    for paragraph in paragraphs:
        sentences = []
        sentence_id = []
        total_len = 0
        contains_hangul = False

        for sentence in paragraph:
            if not contains_hangul and hangul_regex.search(sentence[1]):
                contains_hangul = True

            sentences.append(sentence[1])
            sentence_id.append([total_len, total_len + len(sentence[1]) - 1, sentence[0]])
            total_len += len(sentence[1])

        if total_len > min_length and contains_hangul:
            sentence_list.append(sentences)
            id_list.append(sentence_id)

    return sentence_list, id_list


# Split Sentences into ParagraphFragments
# Returns ParagraphFragments
def remap_to_paragraph(output_values):
    return_values = []

    def find_sentence(paragraph_map, i):
        for key, sentence_map in enumerate(paragraph_map):
            if sentence_map[0] <= i <= sentence_map[1]:
                return key

    for output_value in output_values:
        paragraph_map, output_ranges = output_value
        # paragraph_map
        # [[0, 3, id], [4, 7, id2], [8, 10, id3]]

        # ouptut_ranges
        # [[6, 8], [9, 9]]

        for output_range in output_ranges:
            start_id_key = find_sentence(paragraph_map, output_range[0])
            end_id_key = find_sentence(paragraph_map, output_range[1])

            if start_id_key is None or end_id_key is None:
                print(paragraph_map, output_range)
                continue

            i = start_id_key

            while i <= end_id_key:
                start = 0
                end = paragraph_map[i][1] - paragraph_map[i][0]

                if i == start_id_key:
                    start = output_range[0] - paragraph_map[i][0]

                if i == end_id_key:
                    end = output_range[1] - paragraph_map[i][0]

                return_values.append([paragraph_map[i][2], [start, end]])

                i += 1

    return return_values


if __name__ == "__main__":
    load()

    paragraph = [[
      ["id0", "씨발"],
      ["id1", " 개"],
      ["id2", "새끼"],
      ["id3", "ㅇㅁㅇ"],
      ["id4", "텀 개발 너무 힘들어요 흑흑 인정? 어 인정 으아아 나도 이거 좀 끝내고 싶다고 이 필터들아"]
    ]]
    sharedres = filters['__prepare_sharedres'](paragraph)

    print(filters['default.swearwords'](paragraph, sharedres))
