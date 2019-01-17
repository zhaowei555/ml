#coding:utf-8

import numpy as np
import os
#import theano
#import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from MySumLayer import MySumLayer
from keras.layers import *
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
from myInit import initEmbedding
import sys
import argparse
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    sex_input = Input(shape=(1,), dtype='int32', name = 'sex_input')
    loveType_input = Input(shape=(1,), dtype='int32', name = 'loveType_input')
    type1_input = Input(shape=(1,), dtype='int32', name = 'type1_input')
    type2_input = Input(shape=(1,), dtype='int32', name = 'type2_input')
    uif_input = Input(shape=(64,), dtype='float32', name = 'uif_input')
    bif_input = Input(shape=(64,), dtype='float32', name = 'bif_input')
    ishot_input = Input(shape=(1,), dtype='float32', name = 'ishot_input')
    hotprob_input = Input(shape=(1,), dtype='float32', name = 'hotprob_input')

    #first order'
    #ishot_1d = Reshape([1])(Embedding(2,1)(ishot_input))
    sim_numeric = Concatenate()([uif_input, bif_input])
    dense_numeric_1 = Dense(1)(sim_numeric)
    #hot_numeric = Concatenate()([ishot_1d, hotprob_input])
    #dense_numeric_2 = Dense(1)(hot_numeric)
    hot_love = Subtract()([ishot_input, hotprob_input])
    dense_numeric_2 = Multiply()([hot_love, hot_love])
    emb_userId_1d = Reshape([1])(Embedding(num_users, 1)(user_input))
    emb_itemId_1d = Reshape([1])(Embedding(num_items, 1)(item_input))
    emb_sex_1d = Reshape([1])(Embedding(5, 1)(sex_input))
    emb_loveType_1d = Reshape([1])(Embedding(25, 1)(loveType_input))
    emb_type1_1d = Reshape([1])(Embedding(5, 1)(type1_input))
    emb_type2_1d = Reshape([1])(Embedding(25, 1)(type2_input))
    
    #y_first = Concatenate()([dense_numeric_1, dense_numeric_2, emb_userId_1d, emb_itemId_1d, emb_sex_1d, emb_loveType_1d, \
    #            emb_type1_1d, emb_type2_1d])
    y_first = Concatenate()([emb_userId_1d, emb_itemId_1d])
    y_first_order = Dense(1, activation='sigmoid')(y_first)
    #second order
    latent_dim = 32
    userW = initEmbedding('./vec/user32.vec', latent_dim)
    itemW = initEmbedding('./vec/item32.vec', latent_dim)

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, input_length = 1)(user_input)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, input_length = 1)(item_input)
    #MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
    #        weights = [userW], input_length = 1, trainable = True)(user_input)
    #MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
    #        weights = [itemW], input_length = 1, trainable = True)(item_input)
    MLP_Embedding_sex = Embedding(5, latent_dim, input_length=1)(sex_input)
    MLP_Embedding_loveType = Embedding(input_dim = 25, output_dim = latent_dim, name = 'loveType_embedding',
                                 embeddings_initializer = 'uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)(loveType_input)
    MLP_Embedding_type1 = Embedding(input_dim = 5, output_dim = latent_dim, name = 'type1_embedding',
                                 embeddings_initializer = 'uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)(type1_input)
    MLP_Embedding_type2 = Embedding(input_dim = 25, output_dim = latent_dim, name = 'type2_embedding',
                                 embeddings_initializer = 'uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)(type2_input)
    MLP_Embedding_ishot = Embedding(input_dim = 2, output_dim = latent_dim, name = 'ishot_embedding', 
                                 embeddings_initializer = 'uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)(ishot_input)

    emb_numeric_1 = RepeatVector(1)(Dense(32)(sim_numeric))
    emb_numeric_2 = RepeatVector(1)(Dense(32)(dense_numeric_2))
    
    #emb = Concatenate(axis=1)([emb_numeric_1, emb_numeric_2, MLP_Embedding_User, MLP_Embedding_Item, MLP_Embedding_sex,\
    #                            MLP_Embedding_loveType, MLP_Embedding_type1, MLP_Embedding_type2])
    
    emb = Concatenate(axis=1)([MLP_Embedding_User, MLP_Embedding_Item])
    sum_fea_emb = MySumLayer(axis=1)(emb)
    sum_fea_emb_square = Multiply()([sum_fea_emb, sum_fea_emb])
    
    square_fea_emb = Multiply()([emb, emb])
    square_sum_fea_emb = MySumLayer(axis=1)(square_fea_emb)

    sub = Subtract()([sum_fea_emb_square, square_sum_fea_emb])
    sub = Lambda(lambda x:x*0.5)(sub)

    y_second_order = Dense(1, activation='sigmoid')(MySumLayer(axis=1)(sub))

	#deep parts
    y_deep = Flatten()(emb)
    for idx in range(1, num_layer):
        y_deep = Dropout(0.5)(Dense(layers[idx], activation='relu')(y_deep))
        
    #deepFM
    y = Concatenate(axis=1)([y_first_order, y_second_order, y_deep])
    #y = Concatenate(axis=1)([y_first_order, y_second_order])
    y = Dense(1, activation='sigmoid')(y)
    
    model = Model(input=[user_input, item_input, sex_input, loveType_input, type1_input, type2_input, uif_input, bif_input, ishot_input, hotprob_input], 
                  output=[y])
    
    return model

def get_train_instances(train, num_negatives, batch_size):
    while 1:
        cnt = 0
        fct = 67
        book_info = {}
        user_info = {}
        with open('side_info/result/u.vec', 'r') as f:
            for line in f:
                temp = line.split('\t')
                user_info[int(temp[0])] = temp[1:]
        with open('side_info/result/i.vec', 'r') as f:
            for line in f:
                temp = line.split('\t')
                book_info[int(temp[0])] = temp[1:]
        sex, loveType, type1, type2, uif, bif, ishot, hotprob = [],[],[],[],[],[],[],[]
        user_input, item_input, labels = [],[],[]
        num_users = train.shape[0]
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            if u in user_info:
                sex.append(int(user_info[u][0]))
                loveType.append(int(user_info[u][1]))
                hotprob.append(float(user_info[u][2]))
                if len(user_info[u]) >= 67:
                    uif.append(user_info[u][3:67])
                else:
                    uif.append([0.0 for kkk in range(64)])
            else:
                sex.append(-1)
                loveType.append(-1)
                hotprob.append(1.0)
                uif.append([0.0 for kkk in range(67)])
            if i in book_info:
                type1.append(int(book_info[i][0]))
                type2.append(int(book_info[i][1]))
                ishot.append(int(book_info[i][2]))
                if len(book_info[i]) >= 67:
                    bif.append(book_info[i][3:67])
                else:
                    bif.append([0.0 for kkk in range(64)])
            else:
                type1.append(-1)
                type2.append(-1)
                ishot.append(0)
                bif.append([0.0 for kkk in range(64)])
            labels.append(1)
            # negative instances
            for t in range(num_negatives):
                j = np.random.randint(num_items)
                #while train.has_key((u, j)):
                while (u,j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)

                if u in user_info:
                    sex.append(int(user_info[u][0]))
                    loveType.append(int(user_info[u][1]))
                    hotprob.append(float(user_info[u][2]))
                    if len(user_info[u]) >= 67:
                        uif.append(user_info[u][3:67])
                    else:
                        uif.append([0.0 for kkk in range(64)])
                else:
                    sex.append(-1)
                    loveType.append(-1)
                    hotprob.append(1.0)
                    uif.append([0.0 for kkk in range(64)])
                if j in book_info:
                    type1.append(int(book_info[j][0]))
                    type2.append(int(book_info[j][1]))
                    ishot.append(int(book_info[j][2]))
                    if len(book_info[j]) >= 67:
                        bif.append(book_info[j][3:67])
                    else:
                        bif.append([0.0 for kkk in range(64)])
                else:
                    type1.append(-1)
                    type2.append(-1)
                    ishot.append(0)
                    bif.append([0.0 for kkk in range(64)])
                labels.append(0)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                #print(uif)
                yield ([np.array(user_input), np.array(item_input), np.array(sex), np.array(loveType), \
                        np.array(type1), np.array(type2), np.array(uif), np.array(bif), np.array(ishot), np.array(hotprob)], np.array(labels))
                #print(np.array(uif).shape, np.array(bif).shape)
                user_input, item_input, sex, loveType, type1, type2, uif, bif, ishot, hotprob, labels = [],[],[],[],[],[],[],[],[],[],[]
    #return user_input, item_input, sex, loveType, type1, type2, uif, bif, labels

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 100
    evaluation_threads = 1 #mp.cpu_count()
    print("MLP arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    num_users = 1793875
    num_items = 128843
    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
    
    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
    
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        #user_input, item_input, sex, loveType, type1, type2, uif, bif, labels = get_train_instances(train, num_negatives, 256)
        hist = model.fit_generator(get_train_instances(train, num_negatives, batch_size=256), steps_per_epoch=99854, nb_epoch=1, \
                        verbose=0, shuffle=True)
        # Training       
        '''
        hist = model.fit([np.array(user_input), np.array(item_input), np.array(sex), np.array(loveType), \
                         np.array(type1), np.array(type2), np.array(uif), np.array(bif)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        '''
        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" %(model_out_file))
