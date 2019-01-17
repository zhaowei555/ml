#coding=utf-8

import tensorflow as tf
import os
from dataset import dataset_input_fn, test_input_fn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sex = tf.contrib.layers.sparse_column_with_keys(
    column_name = "sex",
    keys = ["男频","女频",'出版', '短篇', ' ']
)

isHotBook = tf.contrib.layers.sparse_column_with_keys(
    column_name = "isHotBook",
    keys = ["Yes", "No"]
)

isHotAuthor = tf.contrib.layers.sparse_column_with_keys(
    column_name = 'isHotAuthor',
    keys = ['Yes', 'No']
)

serialize = tf.contrib.layers.sparse_column_with_keys(
    column_name = 'serialize',
    keys = ['1', '2']
)

userId = tf.contrib.layers.sparse_column_with_hash_bucket("userId",hash_bucket_size=1800000)
bookId = tf.contrib.layers.sparse_column_with_hash_bucket("bookId",hash_bucket_size=100000)
hotBook = tf.contrib.layers.sparse_column_with_hash_bucket("hotBook",hash_bucket_size=5500)
hotAuthor = tf.contrib.layers.sparse_column_with_hash_bucket("hotAuthor",hash_bucket_size=800)
author = tf.contrib.layers.sparse_column_with_hash_bucket("author",hash_bucket_size=100000)
cpName = tf.contrib.layers.sparse_column_with_hash_bucket("cpName",hash_bucket_size=50)
type1 = tf.contrib.layers.sparse_column_with_hash_bucket("type1",hash_bucket_size=10)
type2 = tf.contrib.layers.sparse_column_with_hash_bucket("type2",hash_bucket_size=30)
loveType = tf.contrib.layers.sparse_column_with_hash_bucket("loveType",hash_bucket_size=30)

# continuous base columns
hotBookProb = tf.contrib.layers.real_valued_column("hotBookProb")
hotAuthorProb = tf.contrib.layers.real_valued_column("hotAuthorProb")
chap = tf.contrib.layers.real_valued_column("chap")
wordCount = tf.contrib.layers.real_valued_column("wordCount")

# step2: define wide feature columns
wide_columns = [
    userId, bookId, hotBook, hotAuthor, cpName, type2, loveType, author, hotBookProb, hotAuthorProb, isHotBook, isHotAuthor, sex, type1,
    tf.contrib.layers.crossed_column([loveType, type2, isHotAuthor],hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([loveType, type2, isHotBook],hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([sex, loveType, type2],hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([sex, type1],hash_bucket_size=int(1e2)),
    tf.contrib.layers.crossed_column([loveType, type2],hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([sex, type2],hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([author, hotAuthor],hash_bucket_size=int(1e6))
]

# step3: define deep feature columns
deep_columns = [
    tf.contrib.layers.embedding_column(userId,dimension=8),
    tf.contrib.layers.embedding_column(bookId,dimension=8),
    tf.contrib.layers.embedding_column(sex,dimension=8),
    tf.contrib.layers.embedding_column(hotBook,dimension=8),
    tf.contrib.layers.embedding_column(hotAuthor,dimension=8),
    tf.contrib.layers.embedding_column(author,dimension=8),
    tf.contrib.layers.embedding_column(cpName,dimension=8),
    tf.contrib.layers.embedding_column(type2,dimension=8),
    tf.contrib.layers.embedding_column(loveType,dimension=8),
    tf.contrib.layers.embedding_column(isHotBook,dimension=8),
    tf.contrib.layers.embedding_column(isHotAuthor,dimension=8),
    hotBookProb, hotAuthorProb
]


model_dir = './Model/test/' # create a temp path
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir = model_dir,
    linear_feature_columns = wide_columns,
    dnn_feature_columns = deep_columns,
    dnn_hidden_units = [256, 128]
)

# step5: process input_data
import pandas as pd
import urllib

COLUMNS = [
    'userId', 'bookId', 'label', 'sex', 'loveType', 'hotBookProb', 'hotBook', 'hotAuthorProb', 'hotAuthor', \
			'bookName', 'author', 'cpName', 'serialize', 'type1' , 'type2', 'chap', 'wordCount', 'isHotBook' , 'isHotAuthor'
]

LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = [
    'userId', 'bookId', 'sex', 'loveType', 'hotBook', 'hotAuthor', 'author', 'cpName', 'serialize', 'type1', 'type2', 'isHotBook', 'isHotAuthor'
]
CONTINUOUS_COLUMNS = [
    'hotBookProb', 'hotAuthorProb', 'chap', 'wordCount' 
]

#train_file = './Data/train.reading'
test_file = './Data/test.reading'

column_types = {'userId':'category', 'bookId':'category', 'sex':'category', 'loveType':'category', 'hotAuthor':'category', 'hotBook':'category',\
        'author':'category', 'cpName':'category', 'serialize':'category', 'type1':'category', 'type2':'category', 'isHotBook':'category', \
        'isHotAuthor':'category', 'hotBookProb':'float32', 'hotAuthorProb':'float32', 'chap':'float32', 'wordCount':'float32'}

#df_train = pd.read_csv(train_file, header=None, names=COLUMNS, dtype=column_types)
#df_test = pd.read_csv(test_file, header=None, names=COLUMNS, dtype=column_types)

#print(df_train.dtypes)

#print(df_train[0:5])

#df_train[LABEL_COLUMN] = df_train['label'].astype(int)
#df_test[LABEL_COLUMN] = df_test['label'].astype(int)

def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONTINUOUS_COLUMNS}
    
    categorical_cols = {
        k: tf.SparseTensor(
            indices = [[i,0] for i in range(df[k].size)],
            values = df[k].values,
            dense_shape = [df[k].size,1]
        )
        for k in CATEGORICAL_COLUMNS
    }

    feature_cols = continuous_cols
    feature_cols.update(categorical_cols)

    label = tf.constant(df[LABEL_COLUMN].values)

    return feature_cols,label

def train_input_fn():
    return input_fn(df_train)
def eval_input_fn():
    #fea, la = input_fn(df_test)
    #return fea
    return input_fn(df_test)
'''
feature, label = dataset_input_fn()

ccc = 0
with tf.Session() as sess:
    while True:
        x, y = sess.run(feature), sess.run(label)
        print(ccc)
        ccc += 1
        m.fit(x=x, y=y, steps = 1)

'''
ep = 0
train_path = './Data/train/'
epoch = 5
for i in range(epoch):
    for fi in os.listdir(train_path):
        train_file = train_path + fi
        df_train = pd.read_csv(train_file, header=None, names=COLUMNS, dtype=column_types)
        print(df_train[0:2])
        df_train[LABEL_COLUMN] = df_train['label'].astype(int)
        m.fit(input_fn = train_input_fn, steps=10)

        #m.fit(input_fn = train_input_fn,steps=200)

df_test = pd.read_csv(test_file, header=None, names=COLUMNS, dtype=column_types)
df_test[LABEL_COLUMN] = df_test['label'].astype(int)

'''
prob = m.predict_proba(input_fn=eval_input_fn)
gg = 0
for i in prob:
    gg += 1
    if gg>50:
        break
    print(i)
'''


results = m.evaluate(input_fn=test_input_fn,steps=1)
for key in sorted(results):
    print ("%s: %s" % (key,results[key]))
