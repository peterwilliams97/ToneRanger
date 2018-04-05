# coding: utf-8
"""
    Performing Model Selection Using Topic Coherence

    This notebook will perform topic modeling on the 20 Newsgroups corpus using LDA. We will perform
    model selection (over the number of topics) using topic coherence as our evaluation metric. This
    will showcase some of the features of the topic coherence pipeline implemented in `gensim`. In
    particular, we will see several features of the `CoherenceModel`.
"""
import os
import re
import logging
from collections import OrderedDict

from gensim.corpora import TextCorpus, MmCorpus
from gensim import utils, models

import multiprocessing
import time
from sklearn import datasets


do_1 = True
do_2 = False
do_3 = False

our_models_dir = os.path.join('home', 'models')  # use whatever directory you prefer

home = os.path.expanduser('~/')
models_dir = os.path.join(home, 'data', 'models')
vectors_path = os.path.join(models_dir, 'GoogleNews-vectors-negative300.bin')
vectors2_path = os.path.join(models_dir, 'fasttext', 'wiki.en.bin')

assert os.path.exists(vectors_path), vectors_path
assert os.path.exists(vectors2_path), vectors2_path

n_cpus = multiprocessing.cpu_count()
n_workers = max(1, n_cpus - 1)

# logging.basicConfig(level=logging.ERROR)  # disable warning logging
logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

logger = logging.getLogger()
logger.info('Hello')


# ## Loading the Dataset
#
# The 20 Newsgroups dataset consists of 20 different newsgroup (forum discussion) groups, with many
# posts per group. The original data is available here: http://qwone.com/~jason/20Newsgroups/.
# However, `sklearn` also provides a wrapper around this data, so we'll use that for simplicity. It
# takes care of downloading the text and loading them into memory.
#
# The documents are in the newsgroup format, which includes some headers, quoting of previous
# messages in the thread, and possibly PGP signature blocks. The code below builds on the
# `TextCorpus` preprocessing to handle the newsgroup-specific text parsing. By default, `TextCorpus`
# preprocessing performs asciifolding and lowercases all text, then tokenizes by pulling out
# contiguous sequences of alphabetic characters, then discards stopwords and tokens less than length
# 3.


class NewsgroupCorpus(TextCorpus):
    """Parse 20 Newsgroups dataset."""

    def __init__(self, *args, **kwargs):
        super(NewsgroupCorpus, self).__init__(
            datasets.fetch_20newsgroups(subset='all'), *args, **kwargs)

    def getstream(self):
        for doc in self.input.data:
            yield doc  # already unicode

    def preprocess_text(self, text):
        body = extract_body(text)
        return super(NewsgroupCorpus, self).preprocess_text(body)


def extract_body(text):
    return strip_newsgroup_header(
        strip_newsgroup_footer(
            strip_newsgroup_quoting(text)))


def strip_newsgroup_header(text):
    """Given text in "news" format, strip the headers, by removing everything
        before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')
    return after


_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


def strip_newsgroup_quoting(text):
    """Given text in "news" format, strip lines beginning with the quote
        characters > or |, plus lines that often introduce a quoted section
        (for example, because they contain the string 'writes:'.)
    """
    good_lines = [line for line in text.split('\n') if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)


_PGP_SIG_BEGIN = "-----BEGIN PGP SIGNATURE-----"


def strip_newsgroup_footer(text):
    """Given text in "news" format, attempt to remove a signature block."""
    try:
        return text[:text.index(_PGP_SIG_BEGIN)]
    except ValueError:
        return text


# ### Loading the Dataset
#
# Now that we have defined the necessary code for preprocessing the dataset, let's load it up and
# serialize it into Matrix Market format. We'll do this because we want to train LDA on it with
# several different parameter settings, and this will allow us to avoid repeating the preprocessing.
if do_1:
    corpus = NewsgroupCorpus()
    corpus.dictionary.filter_extremes(no_below=5, no_above=0.8)
    dictionary = corpus.dictionary
    print(len(corpus))
    print(dictionary)
    print('~' * 100)

    mm_path = '20_newsgroups.mm'
    MmCorpus.serialize(mm_path, corpus, id2word=dictionary)
    mm_corpus = MmCorpus(mm_path)  # load back in to use for LDA training")

# ## Training the Models
#
# Our goal is to determine which number of topics produces the most coherent topics for the 20
# Newsgroups corpus. The corpus contains 18,846 documents. If we used 100 topics and the documents
# were evenly distributed among topics, we'd have clusters of ~188 documents. This seems like a
# reasonable upper bound. In this case, the corpus actually has categories, which we show below.
# There are 20 of these (hence the name of the dataset), so we'll use 20 as our lower bound for the
# number of topics.
#
# One could argue that we already know the model should have 20 topics. I'll argue there may be
# additional categorizations within each newsgroup and we might hope to capture those by using more
# topics. We'll step by increments of 10 from 20 to 100.

if do_2:
    print('\n'.join(corpus.input.target_names))

    os.makedirs(our_models_dir, exist_ok=True)
    trained_models = OrderedDict()
    for num_topics in range(20, 51, 10):
        print("Training LDA(k=%d)" % num_topics, end=' ')
        t0 = time.perf_counter()
        lda = models.LdaMulticore(mm_corpus, id2word=dictionary, num_topics=num_topics, workers=n_workers,
                passes=10, iterations=100, random_state=42, eval_every=None,
                alpha='asymmetric',  # shown to be better than symmetric in most cases
                decay=0.5, offset=64  # best params from Hoffman pape
            )
        trained_models[num_topics] = lda
        print('duration=%.1f sec' % (time.perf_counter() - t0))

        model_path = os.path.join(our_models_dir, 'lda-newsgroups-k%d.lda' % num_topics)
        lda.save(model_path, separately=False)
        trained_models[num_topics] = models.LdaMulticore.load(model_path)

# Some useful utility functions in case you want to save your models.


def save_models(named_models):
    for num_topics, model in named_models.items():
        model_path = os.path.join(our_models_dir, 'lda-newsgroups-k%d.lda' % num_topics)
        model.save(model_path, separately=False)


def load_models():
    trained_models = OrderedDict()
    for num_topics in range(20, 51, 10):
        model_path = os.path.join(our_models_dir, 'lda-newsgroups-k%d.lda' % num_topics)
        print("Loading LDA(k=%d) from %s" % (num_topics, model_path))
        trained_models[num_topics] = models.LdaMulticore.load(model_path)

    return trained_models


# save_models(trained_models)
trained_models = load_models()


# ## Evaluation Using Coherence
#
# Now we get to the heart of this notebook. In this section, we'll evaluate each of our LDA models
# using topic coherence. Coherence is a measure of how interpretable the topics are to humans. It is
# based on the representation of topics as the top-N most probable words for a particular topic. More
# specifically, given the topic-term matrix for LDA, we sort each topic from highest to lowest term
# weights and then select the first N terms.
#
# Coherence essentially measures how similar these words are to each other. There are various
# methods for doing this, most of which have been explored in the paper ["Exploring the Space of
# Topic Coherence Measures"](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf). The
# authors performed a comparative analysis of various methods, correlating them to human judgements.
# The method named "c_v" coherence was found to be the most highly correlated. This and several of
# the other methods have been implemented in `gensim.models.CoherenceModel`. We will use this to
# perform our evaluations.
#
# The "c_v" coherence method makes an expensive pass over the corpus, accumulating term occurrence
# and co-occurrence counts. It only accumulates counts for the terms in the lists of top-N terms for
# each topic. In order to ensure we only need to make one pass, we'll construct a "super topic" from
# the top-N lists of each of the models. This will consist of a single topic with all the relevant
# terms from all the models. We choose 20 as N.


# Now estimate the probabilities for the CoherenceModel.
# This performs a single pass over the reference corpus, accumulating
# the necessary statistics for all of the models at once.
cm = models.CoherenceModel.for_models(
    trained_models.values(), dictionary, texts=corpus.get_texts(), coherence='c_v')


coherence_estimates = cm.compare_models(trained_models.values())
coherences = dict(zip(trained_models.keys(), coherence_estimates))


def print_coherence_rankings(coherences):
    avg_coherence = [(num_topics, avg_coherence)
                     for num_topics, (_, avg_coherence) in coherences.items()]
    ranked = sorted(avg_coherence, key=lambda tup: tup[1], reverse=True)
    print("Ranked by average '%s' coherence:\n" % cm.coherence)
    for item in ranked:
        print("num_topics=%d:\t%.4f" % item)
    print("\nBest: %d" % ranked[0][0])


print_coherence_rankings(coherences)


# ### Results so Far
#
# So far in this notebook, we have used `gensim`'s `CoherenceModel` to perform model selection over
# the number of topics for LDA. We found that for the 20 Newsgroups corpus, 30 topics is best,
# followed by 20. We showcased the ability of the coherence pipeline to evaluate individual
# aggregated model coherence (Note that the individual topic coherence is also computed by
# `cm.compare_models`).
# We also demonstrated how to avoid repeated passes over the corpus, estimating the term similarity
# probabilities for all relevant terms just once. Topic coherence is a powerful alternative to
# evaluation using perplexity on a held-out document set. It is appropriate to use whenever the
# objective of the topic modeling is to present the topics as top-N lists for human consumption.
#
# Note that coherence calculations are generally much more accurate when a larger reference corpus
# is used to estimate the probabilities. In this case, we used the same corpus as for our modeling,
# which is relatively small at only 20,000 documents. A better reference corpus is the full
# Wikipedia corpus. The motivated explorer of this notebook is encouraged to download that corpus
# (see [Experiments on the English Wikipedia](https://radimrehurek.com/gensim/wiki.html)) and use it
# for probability estimation.
#
# Next we'll look at another method of coherence evaluation using distributed word embeddings.

# ### Evaluating Coherence with Word2Vec
#
# The fact that "c_v" coherence uses distributional semantics to evaluate word similarity motivates
# the use of Word2Vec for coherence evaluation. This idea is explored further in an appendix at the
# end of the notebook. The `CoherenceModel` implemented in `gensim` also supports this, so let's
# look at a few examples.


class TextsIterable(object):
    """Wrap a TextCorpus in something that yields texts from its __iter__.
       It's necessary to use this because the Word2Vec model is built by scanning
       over the texts several times. Passing in corpus.get_texts() would result in
       an empty iterable on passes after the first.
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        return self.corpus.get_texts()


if do_3:
    print('@@1')
    cm = models.CoherenceModel.for_models(trained_models.values(), dictionary,
        texts=TextsIterable(corpus), coherence='c_w2v')

    print('@@2')
    coherence_estimates = cm.compare_models(trained_models.values())
    coherences = dict(zip(trained_models.keys(), coherence_estimates))
    print_coherence_rankings(coherences)
    print('@@3')


# #### Using pre-trained word vectors for coherence evaluation.
#
# Whoa! These results are completely different from those of the "c_v" method, and "c_w2v" is
# saying the models we thought were best are actually some of the worst! So what happened here?
#
# The same note must be made for Word2Vec ("c_w2v") that we made for "c_v": results are more
# accurate when a larger reference corpus is used. Except for "c_w2v", this is actually _way,
# way_ more important. Distributional word embedding techniques such as Word2Vec are fitting a
# probability distribution with a large number of parameters, and doing that takes a lot of data.
#
# Luckily, there are a variety of pre-trained word vectors [freely available for download]
# (http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/).
# Below we demonstrate using word vectors trained on ~100 billion words from Google News,
# [available at this link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).
# Note that this file is 1.5G, so downloading it can take quite some time. It is also quite slow to
# load and ends up occupying about 3.35G in memory (this load time is included in the timing below).
# There is no need to use such a large set of word vectors for this evaluation; this one is just
# readily available.

print('@@4')
keyed_vectors = models.KeyedVectors.load_word2vec_format(vectors_path, binary=True)
# still need to estimate_probabilities, but corpus is not scanned
cm = models.CoherenceModel.for_models(
    trained_models.values(), dictionary, texts=corpus.get_texts(),
    coherence='c_w2v', keyed_vectors=keyed_vectors)

print('@@5')
coherence_estimates = cm.compare_models(trained_models.values())
coherences = dict(zip(trained_models.keys(), coherence_estimates))
print_coherence_rankings(coherences)


# #### Watch out for Out-of-Vocabulary (OOV) terms!
#
# At first glance, it might seem like the "c_w2v" coherence with the GoogleNews vectors is a great
# improvement on the "c_v" method; it certainly improves on training on the newsgroups corpus.
# However, if you run the above code with logging enabled, you'll notice a TON of warning messages
# stating something like "3 terms for topic 10 not in word2vec model." This is a real gotcha to
# watch out for! In this case, we might suspect there is significant mismatch because all the
# coherence measures are so similar (within about 0.02 of each other).
#
# When using pre-trained word vectors, there is likely to be some vocabulary mismatch. So unless the
# corpus you're modeling on was included in the training data for the vectors, you need to watch out
# for this. In the results above, it is easy to diagnose because all of the models have very similar
# coherence rankings. You can use the function below to dig in and see exactly how bad the issue is.

def report_on_oov_terms(cm, topic_models):
    """OOV = out-of-vocabulary"""
    topics_as_topn_terms = [
        models.CoherenceModel.top_topics_as_word_lists(model, dictionary)
        for model in topic_models
    ]

    oov_words = cm._accumulator.not_in_vocab(topics_as_topn_terms)
    print('number of oov words: %d' % len(oov_words))

    for num_topics, words in zip(trained_models.keys(), topics_as_topn_terms):
        oov_words = cm._accumulator.not_in_vocab(words)
        print('number of oov words for num_topics=%d: %d' % (num_topics, len(oov_words)))


report_on_oov_terms(cm, trained_models.values())


# *Yikes!* That's a lot of terms that are being ignored when calculating the coherence metrics. So
# these results are not really reliable. Let's use a different set of pre-trained word vectors. I
# trained these on a recent Wikipedia dump using skip-gram negative sampling (SGNS) with a context
# window of 5 and 300 dimensions.

# print('@@ loading fasttest vectors')
# keyed_vectors = models.fasttext.FastText.load_fasttext_format(vectors2_path)


# # still need to estimate_probabilities, but corpus is not scanned
# cm = models.CoherenceModel.for_models(
#     trained_models.values(), dictionary, texts=corpus.get_texts(),
#     coherence='c_w2v', keyed_vectors=keyed_vectors)


# coherence_estimates = cm.compare_models(trained_models.values())
# coherences = dict(zip(trained_models.keys(), coherence_estimates))
# print_coherence_rankings(coherences)

# report_on_oov_terms(cm, trained_models.values())


# #### Looks like we've now restored order
#
# The results with the Wikipedia-trained word vectors are much better because we have less terms
# OOV. The "c_w2v" evalution is now agreeing with "c_v" on the best two models, and the rest of the
# ordering is generally similar. Note that the "c_w2v" values should not be compared directly to
# those produced by the "c_v" method. Only the ranking of models is comparable.

# ## Appendix: Why Word2Vec for Coherence?
#
# The "c_v" coherence method drags a sliding window across all documents in the corpus to accumulate
# co-occurrence statistics. Similarity is calculated using normalized pointwise mutual information
# (PMI) values estimated from these statistics. More specifically, each word is represented by a
# vector of its NPMI with every other word in its top-N topic list. These vectors are then used to
# compute (cosine) similarity between words. The restriction to the other words in the top-N list
# was found to produce better results than using the entire vocabulary and other methods of reducing
# the vocabulary (see section 3.2.2 of http://www.aclweb.org/anthology/W13-0102).
#
# The fact that a reduced space is superior for these metrics indicates there is noise getting in
# the way. The "c_v" method can be seen as constructing an NPMI matrix between words. The vector of
# NPMI values for a particular word can then be looked up by indexing the row or column corresponding
# to that word's `Dictionary` ID. The reduction to the "topic word space" can then be achieved by
# using a mask to select out the top-N topic words. If we are constructing an NPMI matrix between
# words, then discarding some elements to reduce noise, why not factorize the matrix instead?
# Dimensionality reduction techniques such as SVD do a great job of reducing noise along with
# dimensionality, while also providing a compressed representation to work with.
#
# [Recent work](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization)
# has shown that Word2Vec (trained with Skip-Gram Negative Sampling (SGNS)) is actually implicitly
# factorizing a PMI matrix shifted by a positive constant. [A subsequent paper]
# (http://dl.acm.org/citation.cfm?id=2914720) compared Word2Vec to a few different PMI-based metrics
# and showed that it found coherence values that correlated more strongly with human judgements.
