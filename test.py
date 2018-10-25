from jieba import cut
from tensorflow.contrib import learn
import numpy as np

DOCUMENTS = [
    '这是一条测试1',
    '这是一条测试2',
    '这是一条测试3',
    '这是其他测试',
]


def chinese_tokenizer(docs):
    for doc in docs:
        yield list(cut(doc))


vocab = learn.preprocessing.VocabularyProcessor(10, 0)
x = list(vocab.fit_transform(DOCUMENTS))
print(np.array(x))

