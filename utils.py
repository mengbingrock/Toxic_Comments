import re
import string


def remove_punctuation(orig_data):
    data_copy = orig_data.copy()
    for index in orig_data.index:
        line = orig_data[index].strip().lower().replace('\n', '')
        words = re.split(r'\W+', line)
        filter_table = str.maketrans('', '', string.punctuation)
        data_copy[index] = [w.translate(filter_table) for w in words if len(w.translate(filter_table))]
    return data_copy


def show_parameters(logger, config, phase):
    title = phase + " config parameters"
    logger.info(title.center(40, '-'))
    for para in config:
        logger.info("---{} = {}".format(para, config[para]))
    return
