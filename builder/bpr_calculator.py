import os
import logging

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prs_project.settings")
import django
django.setup()

import pickle
from tqdm import tqdm
from datetime import datetime
from math import exp

import random
import pandas as pd
import numpy as np
from collections import defaultdict

from analytics.models import Rating

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger('BPR calculator')


class BayesianPersonalizationRanking(object):

    def __init__(self, save_path):

        self.save_path = save_path
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None
        self.movie_ids = None
        self.ratings = None
        self.user_movies = None
        self.error = 0

        self.learning_rate = 0.05
        self.bias_regularization = 0.002
        self.user_regularization = 0.005
        self.positive_item_regularization = 0.003
        self.negative_item_regularization = 0.0003

    def initialize_factors(self, train_data, k=25):
        """
        train_data:df，user_id,movie_id,rating
        k:隐向量维度
        """
        self.ratings = train_data[['user_id', 'movie_id', 'rating']].to_numpy() # df转换为array
        self.k = k
        self.user_ids = pd.unique(train_data['user_id'])
        self.movie_ids = pd.unique(train_data['movie_id'])

        self.u_inx = {r: i for i, r in enumerate(self.user_ids)} # user_id:索引id
        self.i_inx = {r: i for i, r in enumerate(self.movie_ids)} # movie_id:索引id

        self.user_factors = np.random.random_sample((len(self.user_ids), k)) # 用户数*k
        self.item_factors = np.random.random_sample((len(self.movie_ids), k)) # 电影数*k
        self.user_movies = train_data.groupby('user_id')['movie_id'].apply(lambda x: x.tolist()).to_dict()
        self.item_bias = defaultdict(lambda: 0)
        self.create_loss_samples()

    def build(self, ratings, params):

        if params:
            k = params['k']
            num_iterations = params['num_iterations']

            self.train(ratings, k, num_iterations)

    def train(self, train_data, k=25, num_iterations=4):

        self.initialize_factors(train_data, k)
        for iteration in tqdm(range(num_iterations)):
            self.error = self.loss() # 所有训练样本的损失值求和

            logger.debug('iteration {} loss {}'.format(iteration, self.error))

            for usr, pos, neg in self.draw(self.ratings.shape[0]): # 重新抽样计算参数
                self.step(usr, pos, neg)

            self.save(iteration, iteration == num_iterations - 1)

    def step(self, u, i, j):
        """
        BPR模型里参数的更新过程：
        https://www.iteye.com/blog/liuzhiqiangruc-2073526
        """
        lr = self.learning_rate
        ur = self.user_regularization
        br = self.bias_regularization
        pir = self.positive_item_regularization
        nir = self.negative_item_regularization

        ib = self.item_bias[i] # 0
        jb = self.item_bias[j] # 0

        # user_u * (item_i - item_j)
        u_dot_i = np.dot(self.user_factors[u, :],
                         self.item_factors[i, :] - self.item_factors[j, :])

        x = ib - jb + u_dot_i

        z = 1.0/(1.0 + exp(x))

        ib_update = z - br * ib
        self.item_bias[i] += lr * ib_update

        jb_update = - z - br * jb
        self.item_bias[j] += lr * jb_update

        update_u = ((self.item_factors[i,:] - self.item_factors[j,:]) * z
                    - ur * self.user_factors[u,:])
        self.user_factors[u,:] += lr * update_u

        update_i = self.user_factors[u,:] * z - pir * self.item_factors[i,:]
        self.item_factors[i,:] += lr * update_i

        update_j = -self.user_factors[u,:] * z - nir * self.item_factors[j,:]
        self.item_factors[j,:] += lr * update_j

    def loss(self):
        """
        计算损失值
         r_uij = np.sum(user_u * (item_i - item_j), axis = 1)
        sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))  ==>同乘 np.exp(r_uij) ==>ranking_loss
        """
        br = self.bias_regularization
        ur = self.user_regularization
        pir = self.positive_item_regularization
        nir = self.negative_item_regularization

        ranking_loss = 0
        for u, i, j in self.loss_samples:
            x = self.predict(u, i) - self.predict(u, j)
            ranking_loss += 1.0 / (1.0 + exp(x))

        c = 0
        # 正则项 u^2 + i^2 + j^2 + bias_i^2 + bias_j^2
        for u, i, j in self.loss_samples:

            c += ur * np.dot(self.user_factors[u], self.user_factors[u])
            c += pir * np.dot(self.item_factors[i], self.item_factors[i])
            c += nir * np.dot(self.item_factors[j], self.item_factors[j])
            c += br * self.item_bias[i] ** 2
            c += br * self.item_bias[j] ** 2

        return ranking_loss + 0.5 * c

    def predict(self, user, item):
        """
        decompose the estimator, compute the difference between
        the score of the positive items and negative items; a
        naive implementation might look like the following:
        r_ui = np.diag(user_u.dot(item_i.T))
        r_uj = np.diag(user_u.dot(item_j.T))
        r_uij = r_ui - r_uj
        """
        i_fac = self.item_factors[item]  # 电影数*k => 长度为k的一维向量
        u_fac = self.user_factors[user]  #  人数*k => 长度为k的一维向量
        pq = i_fac.dot(u_fac)  #

        return pq + self.item_bias[item]

    def create_loss_samples(self):
        """
        生成BPR需要的输入样本(user_id,i,j)
        """
        num_loss_samples = int(100 * len(self.user_ids) ** 0.5)  # 负样本数量，理论上是远大于正样本个数
        logger.debug("[BEGIN]building {} loss samples".format(num_loss_samples))

        self.loss_samples = [t for t in self.draw(num_loss_samples)]
        logger.debug("[END]building {} loss samples".format(num_loss_samples))

    def draw(self, nums):
        """
        采样负样本，返回值为 user_id索引，正样本索引，负样本索引。
        nums:个数
        """
        for _ in range(nums):
            u = random.choice(self.user_ids) # 随机选择一个用户
            user_items = self.user_movies[u] #  user_id:[movie_id1,movie_id2,...] => 视频列表
            pos = random.choice(user_items) # 评分过的为正样本
            neg = pos # 一种获取负样本的trick。while循环结束的样本，必是负样本。采样nums个
            while neg in user_items:
                neg = random.choice(self.movie_ids)

            yield self.u_inx[u], self.i_inx[pos], self.i_inx[neg]

    def save(self, factor, finished):

        save_path = self.save_path + '/model/'
        if not finished:
            save_path += str(factor) + '/'

        ensure_dir(save_path)

        logger.info("saving factors in {}".format(save_path))
        item_bias = {iid: self.item_bias[self.i_inx[iid]] for iid in self.i_inx.keys()}

        uf = pd.DataFrame(self.user_factors,
                          index=self.user_ids)
        it_f = pd.DataFrame(self.item_factors,
                            index=self.movie_ids)

        with open(save_path + 'user_factors.json', 'w') as outfile:
            outfile.write(uf.to_json())
        with open(save_path + 'item_factors.json', 'w') as outfile:
            outfile.write(it_f.to_json())
        with open(save_path + 'item_bias.data', 'wb') as ub_file:
            pickle.dump(item_bias, ub_file)

def load_all_ratings(min_ratings=1):
    """
    加载用户评分数据，构造评分矩阵
    """
    columns = ['user_id', 'movie_id', 'rating', 'type', 'rating_timestamp']

    ratings_data = Rating.objects.all().values(*columns)

    ratings = pd.DataFrame.from_records(ratings_data, columns=columns)
    item_count = ratings[['movie_id', 'rating']].groupby('movie_id').count() # 一共有多少部电影被打过分


    item_count = item_count.reset_index()
    item_ids = item_count[item_count['rating'] > min_ratings]['movie_id']
    ratings = ratings[ratings['movie_id'].isin(item_ids)]
    ratings['rating'] = ratings['rating'].astype(float)
    return ratings


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':

    number_of_factors = 10
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    logger = logging.getLogger('BPR calculator')

    train_data = load_all_ratings(1)
    bpr = BayesianPersonalizationRanking(save_path='./models/bpr/{}/'.format(datetime.now().strftime("%m-%d-%H")))
    bpr.train(train_data, 10, 20)





