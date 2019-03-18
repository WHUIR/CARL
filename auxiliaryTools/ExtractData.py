import numpy as np
import scipy.sparse as sp

class Dataset(object):
    "'extract dataset from file'"

    def __init__(self, max_length, path, word_id_path):
        self.word_id_dict = self.load_word_dict(path + word_id_path)
        print "wordId_dict finished"
        self.userReview_dict = self.load_reviews(max_length, len(self.word_id_dict), path + "UserReviews.out")
        self.itemReview_dict = self.load_reviews(max_length, len(self.word_id_dict), path + "ItemReviews.out")
        print "load reviews finished"
        self.num_users, self.num_items = len(self.userReview_dict), len(self.itemReview_dict)
        self.trainMtrx = self.load_ratingFile_as_mtrx(path + "TrainInteraction.out")
        self.valRatings = self.load_ratingFile_as_list(path + "ValInteraction.out")
        self.testRatings = self.load_ratingFile_as_list(path + "TestInteraction.out")

    def load_word_dict(self, path):
        wordId_dict = {}

        with open(path, "r") as f:
            line = f.readline().replace("\n", "")
            while line != None and line != "":
                arr = line.split("\t")
                wordId_dict[arr[0]] = int(arr[1])
                line = f.readline().replace("\n", "")

        return wordId_dict

    def load_reviews(self, max_doc_length, padding_word_id, path):
        entity_review_dict = {}

        with open(path, "r") as f:
            line = f.readline().replace("\n", "")
            while line != None and line != "":
                review = []
                arr = line.split("\t")
                entity = int(arr[0])
                word_list = arr[1].split(" ")

                for i in xrange(len(word_list)):
                    if (word_list[i] == "" or word_list[i] == None or (not self.word_id_dict.has_key(word_list[i]))):
                        continue
                    review.append(self.word_id_dict.get(word_list[i]))
                    if (len(review) >= max_doc_length):
                        break
                if (len(review) < max_doc_length):
                    review = self.padding_word(max_doc_length, padding_word_id, review)
                entity_review_dict[entity] = review
                line = f.readline().replace("\n", "")
        return entity_review_dict

    def padding_word(self, max_size, max_word_idx, review):
        review.extend([max_word_idx]*(max_size - len(review)))
        return review

    def load_ratingFile_as_mtrx(self, file_path):
        mtrx = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        with open(file_path, "r") as f:
            line = f.readline()
            line = line.strip()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mtrx[user, item] = rating
                line = f.readline()

        return mtrx

    def load_ratingFile_as_list(self, file_path):
        rateList = []

        with open(file_path, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                rate = float(arr[2])
                rateList.append([user, item, rate])
                line = f.readline()

        return rateList