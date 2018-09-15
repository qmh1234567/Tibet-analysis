from sklearn.feature_extraction.text import CountVectorizer

# 初始化
vectorizer=CountVectorizer()
corpus=[
    'This', 'is', 'the', 'first','doucment.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X=vectorizer.fit_transform(corpus)
print(X.toarray())
print('\n',vectorizer.get_feature_names())
