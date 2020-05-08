import sys
import numpy
import click
import gensim
import logging
from keras_preprocessing import text
from sklearn.metrics.pairwise import cosine_similarity


def max_token(tokens, mark):
    max_value = 0

    for item in tokens:
        if item[mark] > max_value:
            max_value = item[mark]

    return max_value


def min_token(tokens, mark):
    min_value = sys.maxsize

    for item in tokens:
        if item[mark] < min_value:
            min_value = item[mark]

    return min_value


def create_corpus(output_file_name, data):
    len_data = len(data)
    output_corpus = open(output_file_name, "w", encoding="utf-8")
    my_filters = '"#$&()*+/:;<=>?@[\\]^_`{|}~\t\n'

    with click.progressbar(length=len_data, label="CREATE CORPUS: ", fill_char=click.style('=', fg='white')) as bar:
        for i in range(0, len_data):
            tmp = str(data[i]).lower()
            tmp = text.text_to_word_sequence(text=tmp,
                                             filters=my_filters)
            tmp = " ".join(map(str, tmp))
            output_corpus.write(tmp + "\n")

            bar.update(1)

    output_corpus.close()


def create_train_to_pattern(output_file_name, data, templates):
    len_data = len(data)
    output_corpus = open(output_file_name, "w", encoding="utf-8")
    min_prev_tokens_count = min_token(templates, "prev_tokens_count")
    min_next_tokent_count = min_token(templates, "next_tokens_count")
    max_prev_tokens_count = max_token(templates, "prev_tokens_count")
    max_next_tokent_count = max_token(templates, "next_tokens_count")
    position = max_prev_tokens_count + 1
    len_mask = max_prev_tokens_count + max_next_tokent_count

    lenv = len(data["obsval"])
    with click.progressbar(length=lenv, label="CREATE TRAINING TO PATTERN (STANDARD EXAMPELS): ", fill_char=click.style('=', fg='white')) as bar:
        for i in range(0, lenv):
            try:
                tmp_split = data["obsval"][i].lower().split(" ")
                sample = ["<unk>"] * len_mask

                len_tmp = len(tmp_split)
                for index in range(0, len_tmp):

                    tmp_min_next_tokent_count = index + 1 + min_next_tokent_count
                    tmp_min_prev_tokens_count = index - min_prev_tokens_count
                    if tmp_min_prev_tokens_count >= 0 and tmp_min_next_tokent_count < len_tmp and len(tmp_split[index]) > 0:

                        tmp_max_next_tokent_count = index + 1 + max_next_tokent_count
                        if(tmp_max_next_tokent_count > len_tmp):
                            tmp_max_next_tokent_count = len_tmp

                        tmp_max_prev_tokens_count = index - max_prev_tokens_count
                        if(tmp_max_prev_tokens_count < 0):
                            tmp_max_prev_tokens_count = 0

                        prev_env = tmp_split[tmp_max_prev_tokens_count:index]
                        next_env = tmp_split[(
                            index+1):tmp_max_next_tokent_count]

                        start_position = position - len(prev_env) - 1
                        for item in prev_env:
                            sample[start_position] = item
                            start_position = start_position + 1

                        start_position = position - 1
                        for item in next_env:
                            sample[start_position] = item
                            start_position = start_position + 1

                        output_corpus.write(
                            tmp_split[index] + "\t" + " ".join(sample) + "\n")

                bar.update(1)
            except:
                bar.update(1)

    for template in templates:
        sample = ["<unk>"] * len_mask
        start_position = position - template["prev_tokens_count"] - 1
        tmp_sample = template["sample"].lower().split(" ")
        len_env = len(tmp_sample) + start_position

        for i in range(start_position, len_env):
            sample[i] = tmp_sample[i-start_position]

        with click.progressbar(length=template["sample_count"], label="CREATE TRAINING TO PATTERN (POSITIVE EXAMPELS): ", fill_char=click.style('=', fg='white')) as bar:
            for j in range(0, template["sample_count"]):
                output_corpus.write("SAMPLE\t" + " ".join(sample) + "\n")
                bar.update(1)


def create_word2vec(input_file_name, output_dir, embedding_dim, window, iter, min_count):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    model = gensim.models.Word2Vec(corpus_file=input_file_name,
                                   window=window,
                                   size=embedding_dim,
                                   iter=iter,
                                   min_count=min_count)

    model.save(output_dir + "/word2vec.model")
    model.wv.save_word2vec_format(output_dir + "/word2vec_pattern.vec")


def create_embedding_matrix(input_file_name, word_index, vocab_size, embedding_dim, act_range=10000000):
    # Adding again 1 because of reserved 0 index
    vocab_size = vocab_size + 1
    embedding_matrix = numpy.zeros((vocab_size, embedding_dim))
    embedding_matrix[0].fill(-1.0)

    f = open(input_file_name, encoding='utf8', errors='ignore')
    max_range = int(f.readline().split(" ")[0])
    if(act_range > max_range):
        act_range = max_range

    index = 0
    with click.progressbar(length=act_range, label="LOAD EMBEDDING MATRIX: ", fill_char=click.style('=', fg='white')) as bar:
        while index < act_range:
            word, *vector = f.readline().split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = numpy.array(vector,
                                                    dtype=numpy.float32)[:embedding_dim]

            index = index + 1
            bar.update(1)

    f.close()

    return embedding_matrix


def create_similarity_matrix(data, weights, samples):
    
    lenv = len(data)
    lens = len(samples[0])
    lenss = len(samples)

    weights_sim = cosine_similarity(weights)
    weights_sim = 1 + weights_sim
    weights_sim = weights_sim / 2
    weights_sim = weights_sim / lens
    similarities = []
    with click.progressbar(length=lenv, label="CREATING SIMILARITY MATRIX: ", fill_char=click.style('=', fg='white')) as bar:
        for i in range(0, lenv):
            calc_buffer = []
            for n in range(0, lenss):
                calc_buffer.append(0.0)

            for j in range(0, lens):
                for n in range(0, lenss):
                    # K = data[i][j]
                    # T = samples[n][j]
                    # F = weights_sim[K][F]
                    calc_buffer[n] = calc_buffer[n] + \
                        weights_sim[data[i][j]][samples[n][j]]

            similarities.append(calc_buffer)

            if i % 1000 == 0:
                bar.update(1000)

    similarities = numpy.asarray(similarities)

    return similarities


def load_embedding_space(input_file_name, embedding_dim, tokenizer):
    # Load embedding space
    vocab_size = len(tokenizer.word_index)
    embedding_matrix = create_embedding_matrix(input_file_name,
                                               tokenizer.word_index,
                                               len(tokenizer.word_index),
                                               embedding_dim,
                                               500000)

    nonzero_elements = numpy.count_nonzero(numpy.count_nonzero(embedding_matrix,
                                                               axis=1))
    print("\nvocabulary is covered by the pretrained model: ".upper() +
          str(round(nonzero_elements / vocab_size * 100, 2)) + "%")
    print("LOADING EMBEDDING SPACE: COMPLETE")

    return embedding_matrix


def load_train_to_pattern(input_file_name):
    results = []
    input_file = open(input_file_name, "r", encoding="utf-8")

    with click.progressbar(input_file, label="LOAD PATTERNS TO TRAINING: ", fill_char=click.style('=', fg='white')) as input_file_bar:
        for row in input_file_bar:
            if(len(row) == 0):
                continue

            tmp_row = row.split("\t")
            results.append(tmp_row[1])

    return results