import os
import sqlite3
from tqdm import tqdm
from scipy.sparse import coo_matrix

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prs_project.settings")
import django
from datetime import datetime

from prs_project import settings

import logging
import numpy as np

import pyLDAvis
import pyLDAvis.gensim

import operator
import math

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from gensim import corpora, models, similarities

django.setup()

from recommender.models import MovieDescriptions, LdaSimilarity


def dot_product(v1, v2):
    dp = sum(map(operator.mul, v1, v2))
    return dp


def vector_cos(v1, v2):
    prod = dot_product(v1, v2)
    sqrt1 = math.sqrt(dot_product(v1, v1))
    sqrt2 = math.sqrt(dot_product(v2, v2))
    return prod / (sqrt1 * sqrt2)


def cosine_similarity(ldas):
    size = ldas.shape[0]
    similarity_matrix = np.zeros((size, size))

    for i in range(ldas.shape[0]):

        for j in range(ldas.shape[0]):
            similarity_matrix[i, j] = vector_cos(ldas[i,], ldas[j, ])

    return similarity_matrix


def load_data():
    docs = list(MovieDescriptions.objects.all())
    data = ["{},{}".format(d.title, d.description) for d in docs] # 使用title和description两类文本信息

    if len(data) == 0:
        print("No descriptions were found, run populate_sample_of_descriptions")
    return data, docs


class LdaModel(object):

    def __init__(self, min_sim=0.1):
        self.dirname, self.filename = os.path.split(os.path.abspath(__file__))
        self.min_sim = min_sim
        self.db = settings.DATABASES['default']['ENGINE']

    def train(self, data=None, docs=None):
        """
        LDA模型训练过程：
        data:list.其中每个元素代表一个文档
        """

        if data is None:
            data, docs = load_data()

        NUM_TOPICS = 10  # 主题模型topic的个数
        self.lda_path = self.dirname + '/models/lda/' # 模型结果统一存到models目录下
        if not os.path.exists(self.lda_path):
            os.makedirs(self.lda_path)

        self.build_lda_model(data, docs, NUM_TOPICS)

    @staticmethod
    def tokenize(texts):
        """
         RegexpTokenizer() 参数是将要匹配的字符串的正则表达式，返回值是所有匹配到的字符串组成的列表
         tokenizer = RegexpTokenizer("\w+")
         print(tokenizer.tokenize("Don't hesitate to ask questions!"))
           ==> ['Don', 't', 'hesitate', 'to', 'ask', 'questions']
        """
        tokenizer = RegexpTokenizer(r'\w+') # 按照标点进行切分

        return [tokenizer.tokenize(d) for d in texts]

    def build_lda_model(self, data:list, docs, n_topics=5):
        """
        LDA模型生成：1、文本分词。2、使用gensim的LdaModel模块训练模型
        """
        texts = []
        tokenizer = RegexpTokenizer(r'\w+')

        for d in tqdm(data):  # 在for循环中，进行文本预处理：转为小写、分词、去停用词。
            raw = d.lower()
            tokens = tokenizer.tokenize(raw)
            stopped_tokens = self.remove_stopwords(tokens)
            stemmed_tokens = stopped_tokens
            texts.append(stemmed_tokens)

        dictionary = corpora.Dictionary(texts)  # 单词到其索引的映射字典
        corpus = [dictionary.doc2bow(text) for text in texts]  # 文本转换为索引向量。

        lda_model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                                             num_topics=n_topics)

        # 生成相似度矩阵。sims = index[query] 使用的时候输入向量化之后的文本，返回query和其它所有训练文本的相似度。
        index = similarities.MatrixSimilarity(corpus)

        self.save_lda_model(lda_model, corpus, dictionary, index)
        self.save_similarities(index, docs)

        return dictionary, texts, lda_model

    def save_lda_model(self, lda_model, corpus, dictionary, index):

        index.save(self.lda_path + 'index.lda')
        pyLDAvis.save_json(pyLDAvis.gensim.prepare(lda_model, corpus, dictionary), self.lda_path + '../../../static/js/lda.json')
        print(lda_model.print_topics())
        lda_model.save(self.lda_path + 'model.lda')

        dictionary.save(self.lda_path + 'dict.lda')
        corpora.MmCorpus.serialize(self.lda_path + 'corpus.mm', corpus)

    @staticmethod
    def remove_stopwords(tokenized_data):

        en_stop = get_stop_words('en')

        stopped_tokens = [token for token in tokenized_data if token not in en_stop]
        return stopped_tokens

    def save_similarities(self, index, docs, created=datetime.now()):

        self.save_similarities_with_django(index, docs, created)

    def save_similarities_with_django(self, index, docs, created=datetime.now()):
        start_time = datetime.now()
        result = LdaSimilarity.objects.all().count()
        if result > 1:
            LdaSimilarity.objects.all().delete()
        print(f'truncating table in {datetime.now() - start_time} seconds')

        no_saved = 0
        start_time = datetime.now()
        # A sparse matrix in COOrdinate format
        #  row  = np.array([0, 3, 1, 0])
        #  col  = np.array([0, 3, 1, 2])
        #  data = np.array([4, 5, 7, 9])
        #  coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
        #   >>>array([[4, 0, 9, 0],
        #             [0, 7, 0, 0],
        #             [0, 0, 0, 0],
        #             [0, 0, 0, 5]])
        coo = coo_matrix(index)
        csr = coo.tocsr() # 将此矩阵转换为压缩稀疏行格式重复的条目将被汇总在一起。通过csr[x,y]来获取对应位置的两个文档之间的相似度。

        print(f'instantiation of coo_matrix in {datetime.now() - start_time} seconds')
        print(f'{coo.count_nonzero()} similarities to save')
        xs, ys = coo.nonzero()  # 返回非零值的(x,y)索引
        for x, y in zip(xs, ys):

            if x == y:
                continue

            sim = float(csr[x, y])
            x_id = str(docs[x].movie_id)
            y_id = str(docs[y].movie_id)
            if sim < self.min_sim:
                continue
            print(x_id,y_id,sim)
            one_record = LdaSimilarity(created=created, source=x_id, target=y_id, similarity=sim)
            one_record.save()
            no_saved += 1

        print('{} Similarity items saved, done in {} seconds'.format(no_saved, datetime.now() - start_time))

if __name__ == '__main__':
    print("Calculating lda model...")
    # LDA（Latent Dirichlet Allocation）模型

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data, docs = load_data()
    lda = LdaModel()
    print(lda.dirname,lda.filename)
    lda.train(data, docs)
