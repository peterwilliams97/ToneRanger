import re
import gensim
from gensim.models import Doc2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import utils


# The purpose of below functions are:
# 1. Tokenization
# 2. Remove URLs
# 3. Remove email address
# 4. Remove tags
# 5. Remove puntuations
# 6. Remove stop words
# 7. Apply Stemming:


#Data cleaning
def email_cleaning(text):
    email = text.lower()
    # clean and tokenize document string
    email_content = email.split()
    word_list = []
    for i in email_content:
        if (('http' not in i) and ('@' not in i) and ('<.*?>' not in i) and i.isalnum()
            and (not i in stop_words)):
            word_list += [i]

    return word_list

#Data Pre-processing
def preprocessing(text):
    # remove numbers
    number_tokens = [re.sub(r'[\d]', ' ', i) for i in text]
    number_tokens = ' '.join(number_tokens).split()
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
    # remove empty
    length_tokens = [i for i in stemmed_tokens if len(i) > 1]
    return length_tokens


# **Create a list of tagged emails. **

LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
all_content = []
texts = []
j = 0
k = 0
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

sentences_path = 'all.paragraphs.json'
raw_sentences = utils.load_json(sentences_path)
for em in raw_sentences:
    # Data cleaning
    clean_content = email_cleaning(em)

    # Pre-processing
    processed_email = preprocessing(clean_content)

    # add tokens to list
    if processed_email:
        all_content.append(LabeledSentence1(processed_email, [j]))
        j += 1

    k += 1

print("Number of emails processed: ", k)
print("Number of non-empty emails vectors: ", j)

# **Printout the sample processed email**
print(all_content[278])

# **Create a model using Doc2Vec and train it**
d2v_model = Doc2Vec(all_content, size=2000, window=10, min_count=500, workers=3, dm=1,
                    alpha=0.025, min_alpha=0.001)

print("corpus_count: ", d2v_model.corpus_count)

d2v_model.train(all_content, total_examples=d2v_model.corpus_count, epochs=10,
    start_alpha=0.002, end_alpha=-0.016)


# **Print the emails similar to email with tagged id as 1 **

# shows the similar docs with id = 99
print('=' * 100)
print('Most similar')
for i in range(10):
    print(i)
    print(d2v_model.docvecs.most_similar(1))

for n_clusters in range(5, 11):
    print('n_clusters=%d' % n_clusters)
    # **Apply K-means clustering on the model**
    kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100)
    X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
    labels = kmeans_model.labels_.tolist()

    kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)
    pca = PCA(n_components=2).fit(d2v_model.docvecs.doctag_syn0)
    datapoint = pca.transform(d2v_model.docvecs.doctag_syn0)

    # **Plot the clustering result**

    plt.figure
    label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", 'r', 'g', 'b']
    color = [label1[i % len(label1)] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)
    plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
    plt.show()
