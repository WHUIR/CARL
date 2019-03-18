# encoding: utf-8
import tensorflow as tf
import numpy as np
from auxiliaryTools.ExtractData import Dataset
from auxiliaryTools.GetTest import get_test_list
from time import time
import math, os


def ini_word_embed(num_words, latent_dim):
    word_embeds = np.random.rand(num_words, latent_dim)
    return word_embeds

def word2vec_word_embed(num_words, latent_dim, path, word_id_dict):
    word2vect_embed_mtrx = np.zeros((num_words, latent_dim))
    with open(path, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            row_id = word_id_dict.get(arr[0])
            vect = arr[1].strip().split(" ")
            for i in xrange(len(vect)):
                word2vect_embed_mtrx[row_id, i] = float(vect[i])
            line = f.readline()

    return word2vect_embed_mtrx

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


def  cnn_model_average(filters, user_reviews_representation_expnd, item_reviews_representation_expnd, W_u, W_i, W_u_1, W_i_1, rand_matrix):
    #convolution layer
    convU = tf.nn.conv2d(user_reviews_representation_expnd, W_u, strides=[1, 1, word_latent_dim, 1], padding='SAME')
    convI = tf.nn.conv2d(item_reviews_representation_expnd, W_i, strides=[1, 1, word_latent_dim, 1], padding='SAME')

    hU = tf.nn.relu(tf.squeeze(convU, 2))
    hI = tf.nn.relu(tf.squeeze(convI, 2))

    # attentive layer
    sec_dim = int(hU.get_shape()[1])
    tmphU = tf.reshape(hU, [-1, filters])
    hU_mul_rand = tf.reshape(tf.matmul(tmphU, rand_matrix), [-1, sec_dim, filters])
    f = tf.matmul(hU_mul_rand, hI, transpose_b=True)
    f = tf.expand_dims(f, -1)
    att1 = tf.tanh(f)

    pool_user = tf.reduce_mean(att1, 2)
    pool_item = tf.reduce_mean(att1, 1)

    user_flat = tf.squeeze(pool_user, -1)
    item_flat = tf.squeeze(pool_item, -1)

    weight_user = tf.nn.softmax(user_flat)
    weight_item = tf.nn.softmax(item_flat)

    weight_user_exp = tf.expand_dims(weight_user, -1)
    weight_item_exp = tf.expand_dims(weight_item, -1)

    hU = tf.expand_dims(hU * weight_user_exp, -1)
    hI = tf.expand_dims(hI * weight_item_exp, -1)

    #abstracting layer
    hU_1 = tf.nn.relu(tf.nn.conv2d(hU, W_u_1, strides=[1, 1, 1, 1], padding='VALID'))
    hI_1 = tf.nn.relu(tf.nn.conv2d(hI, W_i_1, strides=[1, 1, 1, 1], padding='VALID'))

    sec_dim = hU_1.get_shape()[1]

    oU = tf.nn.avg_pool(hU_1, ksize=[1, sec_dim, 1, 1], strides=[1, 1, 1, 1],
        padding='VALID')
    oI = tf.nn.avg_pool(hI_1, ksize=[1, sec_dim, 1, 1], strides=[1, 1, 1, 1],
        padding='VALID')

    att_user = tf.squeeze(oU)
    att_item = tf.squeeze(oI)
    #print "attention", att_user.get_shape(), att_item.get_shape()

    return att_user, att_item

def train_model():
    users = tf.placeholder(tf.int32, shape=[None])
    items = tf.placeholder(tf.int32, shape=[None])
    users_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    items_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    ratings = tf.placeholder(tf.float32, shape=[None, 1])
    dropout_rate = tf.placeholder(tf.float32)

    user_entity_embedding = tf.Variable(tf.random_normal([num_users, latent_dim], mean=0, stddev=0.02), name="user_entity_embeddings")
    item_entity_embedding = tf.Variable(tf.random_normal([num_items, latent_dim], mean=0, stddev=0.02), name="item_entity_embeddings")
    user_bias = tf.Variable(tf.random_normal([num_users, 1], mean=0, stddev=0.02), name="user_bias")
    item_bias = tf.Variable(tf.random_normal([num_items, 1], mean=0, stddev=0.02), name="item_bias")

    user_bs = tf.nn.embedding_lookup(user_bias, users)
    item_bs = tf.nn.embedding_lookup(item_bias, items)

    user_entity_embeds = tf.nn.embedding_lookup(user_entity_embedding, users)
    item_entity_embeds = tf.nn.embedding_lookup(item_entity_embedding, items)

    text_embedding = tf.Variable(word_embedding_mtrx, dtype=tf.float32, name="review_text_embeds")
    padding_embedding = tf.Variable(np.zeros([1, word_latent_dim]), dtype=tf.float32)

    text_mask = tf.constant([1.0] * text_embedding.get_shape()[0] + [0.0])

    word_embeddings = tf.concat([text_embedding, padding_embedding], 0)
    word_embeddings = word_embeddings * tf.expand_dims(text_mask, -1)

    user_reviews_representation = tf.nn.embedding_lookup(word_embeddings, users_inputs)
    user_reviews_representation_expnd = tf.expand_dims(user_reviews_representation, -1)
    item_reviews_representation = tf.nn.embedding_lookup(word_embeddings, items_inputs)
    item_reviews_representation_expnd = tf.expand_dims(item_reviews_representation, -1)

    # CNN layers
    W_u = tf.Variable(
        tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3), name="review_W_u")
    W_i = tf.Variable(
        tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3), name="review_W_i")
    W_u_1 = tf.Variable(
        tf.truncated_normal([window_size, num_filters, 1, num_filters], stddev=0.3), name="review_W_u_1")
    W_i_1 = tf.Variable(
        tf.truncated_normal([window_size, num_filters, 1, num_filters], stddev=0.3), name="review_W_i_1")
    # b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]))
    rand_matrix = tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.3), name="review_rand_matrix")

    user_embeds, item_embeds = cnn_model_average(num_filters, user_reviews_representation_expnd, item_reviews_representation_expnd, W_u, W_i, W_u_1, W_i_1, rand_matrix)
    W_mlp = tf.Variable(tf.random_normal([num_filters, latent_dim], mean=0, stddev=0.02), name="review_W_mlp")
    W_mlp = tf.nn.dropout(W_mlp, dropout_rate)
    b_mlp = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="review_b_mlp")

    #shared MLP layer
    user_embeds = tf.nn.relu(tf.matmul(user_embeds, W_mlp) + b_mlp)
    item_embeds = tf.nn.relu(tf.matmul(item_embeds, W_mlp) + b_mlp)


    embeds_sum = tf.concat([tf.multiply(user_embeds, item_embeds), user_embeds, item_embeds], 1, name="concat_embed")
    entity_embeds_sum = tf.concat([tf.multiply(user_entity_embeds, item_entity_embeds), user_entity_embeds, item_entity_embeds],1)

    #FM layer
    w_0 = tf.Variable(tf.zeros(1), name="review_w_0")
    w_1 = tf.Variable(tf.truncated_normal([1, latent_dim * 3], stddev=0.3), name="review_w_1")
    v = tf.Variable(tf.truncated_normal([latent_dim * 3, latent_dim * 3], stddev=0.3), name="review_v")

    w_entity_0 = tf.Variable(tf.zeros(1), name="entity_w_0")
    w_entity_1 = tf.Variable(tf.truncated_normal([1, latent_dim*3], stddev=0.3), name="entity_w_1")
    v_entity = tf.Variable(tf.truncated_normal([latent_dim*3, v_dim], stddev=0.3), name="entity_v")

    J_e_1 = w_entity_0 + tf.matmul(entity_embeds_sum, w_entity_1, transpose_b=True)
    J_1 = w_0 + tf.matmul(embeds_sum, w_1, transpose_b=True)

    entity_embeds_sum_1 = tf.expand_dims(entity_embeds_sum, -1)
    entity_embeds_sum_2 = tf.expand_dims(entity_embeds_sum, 1)
    embeds_sum_1 = tf.expand_dims(embeds_sum, -1)
    embeds_sum_2 = tf.expand_dims(embeds_sum, 1)

    J_e_2 = tf.reduce_sum(
        tf.reduce_sum(tf.multiply(tf.matmul(entity_embeds_sum_1, entity_embeds_sum_2), tf.matmul(v_entity, v_entity, transpose_b=True)),
                      2), 1, keep_dims=True)
    J_2 = tf.reduce_sum(
        tf.reduce_sum(tf.multiply(tf.matmul(embeds_sum_1, embeds_sum_2), tf.matmul(v, v, transpose_b=True)),
                      2), 1, keep_dims=True)
    J_total = (J_1 + 0.5 * (J_2))
    J_e_total = (J_e_1 + 0.5 * (J_e_2))
    numerator = J_total + J_e_total
    predict_rating = tf.div(J_total, numerator) * J_total + tf.div(J_e_total,numerator) * J_e_total + user_bs + item_bs
    loss = tf.reduce_mean(tf.squared_difference(predict_rating, ratings))
    loss += lambda_1 * (tf.nn.l2_loss(W_i_1) + tf.nn.l2_loss(W_u_1) + tf.nn.l2_loss(user_entity_embedding) + tf.nn.l2_loss(item_entity_embedding) + tf.nn.l2_loss(W_u) + tf.nn.l2_loss(W_i) + tf.nn.l2_loss(v) + tf.nn.l2_loss(v_entity) + tf.nn.l2_loss(rand_matrix) + tf.nn.l2_loss(user_bs) + tf.nn.l2_loss(item_bs))
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()

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
                _, loss_val, words = sess.run([train_step, loss, word_embeddings],
                                       feed_dict={users: user_batch, items: item_batch, users_inputs: user_input_batch, items_inputs: item_input_batch,
                                                  ratings: rates_batch, dropout_rate:drop_out})
                loss_total += loss_val
                count += 1.0
            t1 = time()
            val_mses, val_maes = [], []
            for i in xrange(len(user_input_val)):
                eval_model(users, items, users_inputs, items_inputs, dropout_rate, predict_rating, sess, user_vals[i],
                           item_vals[i], user_input_val[i], item_input_val[i], rating_input_val[i], val_mses, val_maes)
            val_mse = np.array(val_mses).mean()
            t2 = time()
            mses, maes = [], []
            for i in xrange(len(user_input_test)):
                eval_model(users, items, users_inputs, items_inputs, dropout_rate, predict_rating, sess, user_tests[i], item_tests[i], user_input_test[i], item_input_test[i], rating_input_test[i], mses, maes)
            mse = np.array(mses).mean()
            mae = np.array(maes).mean()
            t3 = time()
            print "epoch%d train time: %.3fs test time: %.3f  loss = %.3f val_mse = %.3f mse = %.3f mae = %.3f"%(e, (t1 - t), (t3 - t2), loss_total/count, val_mse, mse, mae)


def eval_model(users, items, users_inputs, items_inputs, dropout_rate, predict_rating, sess, user_batch, item_batch, user_input_batch, item_input_batch, rate_tests, rmses, maes):

    predicts = sess.run(predict_rating, feed_dict={users: user_batch, items: item_batch, users_inputs: user_input_batch, items_inputs: item_input_batch, dropout_rate:1.0})
    row, col = predicts.shape
    for r in xrange(row):
        rmses.append(pow((predicts[r, 0] - rate_tests[r][0]), 2))
        maes.append(abs((predicts[r, 0] - rate_tests[r][0])))
    return rmses, maes

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    word_latent_dim = 300
    latent_dim = 15
    max_doc_length = 300
    num_filters = 50
    window_size = 3
    v_dim = 50
    learning_rate = 0.001
    lambda_1 = 0.05#normally we set from [0.05, 0.01, 0.005, 0.001]
    drop_out = 0.8
    batch_size = 200
    epochs = 180

    # loading data
    firTime = time()
    dataSet = Dataset(max_doc_length, "/the parent directory of the training files/",
                      "WordDict.out")
    word_dict, user_reviews, item_reviews, train, valRatings, testRatings = dataSet.word_id_dict, dataSet.userReview_dict, dataSet.itemReview_dict, dataSet.trainMtrx, dataSet.valRatings, dataSet.testRatings
    secTime = time()

    num_users, num_items = train.shape
    print "load data: %.3fs" % (secTime - firTime)
    print num_users, num_items
    print  latent_dim

    #load word embeddings
    word_embedding_mtrx = ini_word_embed(len(word_dict), word_latent_dim)
    # word_embedding_mtrx = word2vec_word_embed(len(word_dict), word_latent_dim,
    #                                           "Directory of pretrained WordEmbedding.out",
    #                                           word_dict)

    print "shape", word_embedding_mtrx.shape

    # get train instances
    user_input, item_input, rateings = get_train_instance(train)
    print len(user_input), len(item_input), len(rateings)

    # get test/val instances
    user_vals, item_vals, user_input_val, item_input_val, rating_input_val = get_test_list(200, valRatings, user_reviews, item_reviews)
    user_tests, item_tests, user_input_test, item_input_test, rating_input_test = get_test_list(200, testRatings, user_reviews, item_reviews)

    #train & eval model
    train_model()
