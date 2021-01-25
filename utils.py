import nltk


def remove_punctuation(orig_data):
    data_copy = orig_data.copy()
    for index in orig_data.index:
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        line = orig_data[index].strip().lower()
        new_words = tokenizer.tokenize(line)
        data_copy[index] = ' '.join(new_words)
    return data_copy
