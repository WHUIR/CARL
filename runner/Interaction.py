# encoding: utf-8
import tensorflow as tf
import numpy as np
from auxiliaryTools.ExtractData import Dataset
from auxiliaryTools.GetTest import get_test_list
from time import time
import math, os



def get_train_instance(train):
    user_input, item_input, rates = [], [], []

    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        rates.append(train[u,i])
    return user_input, item_input, rates

def get_train_instance_batch_change(count, batch_size, user_input, item_input, ratings, user_reviews, item_reviews):
    users_batch, items_batch, user_input_batch, item_input_batch, labels_batch = [], [], [], [], []

    for idx in xrange(batch_size):
        index = (count*batch_size + idx) % len(user_input)
        users_batch.append(user_input[index])
        items_batch.append(item_input[index])
        user_input_batch.append(user_reviews.get(user_input[index]))
        item_input_batch.append(item_reviews.get(item_input[index]))
        labels_batch.append([ratings[index]])

    return users_batch, items_batch, user_input_batch, item_input_batch, labels_batch

def train_model():
    users = tf.placeholder(tf.int32, shape=[None])
    items = tf.placeholder(tf.int32, shape=[None])
    ratings = tf.placeholder(tf.float32, shape=[None, 1])

    user_entity_embedding = tf.Variable(tf.random_normal([num_users, latent_dim], mean=0, stddev=0.02), name="user_entity_embeddings")
    item_entity_embedding = tf.Variable(tf.random_normal([num_items, latent_dim], mean=0, stddev=0.02), name="item_entity_embeddings")

    user_entity_embeds = tf.nn.embedding_lookup(user_entity_embedding, users)
    item_entity_embeds = tf.nn.embedding_lookup(item_entity_embedding, items)

    entity_embeds_sum = tf.concat([tf.multiply(user_entity_embeds, item_entity_embeds), user_entity_embeds, item_entity_embeds],1)

    #FM layer
    w_entity_0 = tf.Variable(tf.zeros(1), name="entity_w_0")
    w_entity_1 = tf.Variable(tf.truncated_normal([1, latent_dim*3], stddev=0.3), name="entity_w_1")
    v_entity = tf.Variable(tf.truncated_normal([latent_dim*3, v_dim], stddev=0.3), name="entity_v")

    J_e_1 = w_entity_0 + tf.matmul(entity_embeds_sum, w_entity_1, transpose_b=True)

    entity_embeds_sum_1 = tf.expand_dims(entity_embeds_sum, -1)
    entity_embeds_sum_2 = tf.expand_dims(entity_embeds_sum, 1)
    J_e_2 = tf.reduce_sum(
        tf.reduce_sum(tf.multiply(tf.matmul(entity_embeds_sum_1, entity_embeds_sum_2), tf.matmul(v_entity, v_entity, transpose_b=True)),
                      2), 1, keep_dims=True)
    J_e_3 = tf.trace(tf.multiply(tf.matmul(entity_embeds_sum_1, entity_embeds_sum_2), tf.matmul(v_entity, v_entity, transpose_b=True)))
    J_e_total = (J_e_1 + 0.5 * (J_e_2 - J_e_3))

    predict_rating = J_e_total
    loss = tf.reduce_mean(tf.squared_difference(predict_rating, ratings))
    loss += lambda_1 * (tf.nn.l2_loss(user_entity_embedding) + tf.nn.l2_loss(item_entity_embedding) + tf.nn.l2_loss(v_entity))
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in xrange(epochs):
            t = time()
            loss_total = 0.0
            count = 0.0
            for i in xrange(int(math.ceil(len(user_input) / float(batch_size)))):
                user_batch, item_batch, user_input_batch, item_input_batch, rates_batch = get_train_instance_batch_change(i, batch_size,user_input,
                                                                                                  item_input, rateings,
                                                                                                  user_reviews,item_reviews)
                _, loss_val = sess.run([train_step, loss],
                                       feed_dict={users: user_batch, items: item_batch,
                                                  ratings: rates_batch})
                loss_total += loss_val
                count += 1.0
            t1 = time()
            mses, maes = [], []
            for i in xrange(len(user_input_test)):
                mses, maes = eval_model(users, items, predict_rating, user_tests[i], item_tests[i], rating_input_test[i], sess, mses, maes)
            mse = np.array(mses).mean()
            mae = np.array(maes).mean()
            t2 = time()
            print "epoch%d train time: %.3fs test time: %.3f  loss = %.3f  mse = %.3f  mae = %.3f"%(e, (t1 - t), (t2 - t1), loss_total/count, mse, mae)


def eval_model(users, items, predict_rating, user_test, item_test, rate_tests, sess, rmses, maes):

    predicts = sess.run(predict_rating, feed_dict={users: user_test, items: item_test})
    row, col = predicts.shape
    for r in xrange(row):
        rmses.append(pow((predicts[r, 0] - rate_tests[r][0]), 2))
        maes.append(abs(predicts[r, 0] - rate_tests[r][0]))
    return rmses, maes

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    word_latent_dim = 300
    latent_dim = 30
    max_doc_length = 300
    windows = 3
    v_dim = 50
    learning_rate = 0.001
    lambda_1 = 0.05
    batch_size = 100
    epochs = 180

    # loading data
    firTime = time()
    dataSet = Dataset(max_doc_length, "/the parent directory of the training files/",
                      "WordDict.out")
    word_dict, user_reviews, item_reviews, train, testRatings = dataSet.word_id_dict, dataSet.userReview_dict, dataSet.itemReview_dict, dataSet.trainMtrx, dataSet.testRatings
    secTime = time()

    num_users, num_items = train.shape
    print "load data: %.3fs" % (secTime - firTime)
    print num_users, num_items


    # get train instances
    user_input, item_input, rateings = get_train_instance(train)
    # get test instances
    user_tests, item_tests, user_input_test, item_input_test, rating_input_test = get_test_list(200, testRatings, user_reviews, item_reviews)

    train_model()
