# 隐含狄利克雷分布（LDA)
import pandas as pd
import jieba
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

data_str = open('tibet.json').read()


df = pd.read_json(data_str, orient='records')
df.head()

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

df["content_cutted"] = df.content.apply(chinese_word_cut)

print(df.content_cutted.head())


n_features = 1000
tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df=0.5,
                                min_df=10)
tf = tf_vectorizer.fit_transform(df.content_cutted)
n_topics = 5
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


n_top_words = 20
pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
pyLDAvis.enable_notebook()

pyLDAvis.show(data)
