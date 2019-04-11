# CARL
The implementation of “A Context-Aware User-Item Representation Learning for Item Recommendation”, Libing Wu, Cong Quan, Chenliang Li, Qian Wang, Bolong Zheng, Xiangyang Luo, https://dl.acm.org/citation.cfm?id=3298988


## Requirements
Tensorflow 1.2

Python 2.7

Numpy

Scipy

## Data Preparation
To run CARL, 6 files are required: 

### Training Rating records: 
file_name=TrainInteraction.out

each training sample is a sequence as:

UserId\tItemId\tRating\tDate

Example: 0\t3\t5.0\t1393545600

### Validate Rating records: 
file_name=ValInteraction.out

The format is the same as the training data format. 

### Testing Rating records: 
file_name=TestInteraction.out

The format is the same as the training data format.

### Word2Id diction: 
file_name=WordDict.out 

Each line follows the format as:

Word\tWord_Id

Example: love\t0

### User Review Document: 
file_name=UserReviews.out

each line is the format as:

UserId\tWord1 Word2 Word3 …

Example:0\tI love to eat hamburger …

### Item Review Document: 
file_name=ItemReviews.out

The format is the same as the user review doc format.

## Note that: 
All files need to be located in the same directory. 

Besides, the code also supports to leverage the pretrained word embedding via uncomment the loading function “word2vec_word_embed” in the main file . 

Carl.py denotes the model named CARL; Review.py denotes the review-based component while Interaction.py denotes the interaction-based component.

## Configurations
word_latent_dim: the dimension size of word embedding;

latent_dim: the latent dimension of the representation learned from the review documents (entity);

max_len: the maximum doc length;

num_filters: the number of filters of CNN network;

window_size: the length of the sliding window of CNN;

learning_rate: learning rate;

lambda_1: the weight of the regularization part;

drop_out: the keep probability of the drop out strategy;

batch_size: batch size;

epochs: number of training epoch;


