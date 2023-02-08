import sys
sys.path.append("../src")
import debias
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models import KeyedVectors
import numpy as np
from sklearn.svm import LinearSVC, SVC
import tqdm
import pickle
from collections import defaultdict, Counter
from typing import List, Dict
import time

print(f"\n\n\n\n\n\nStart of Fair-professional BERT run-time measurement")

def load_dataset(path):
    
    with open(path, "rb") as f:
        
        data = pickle.load(f)
    return data

def load_dictionary(path):
    
    with open(path, "r", encoding = "utf-8") as f:
        
        lines = f.readlines()
        
    k2v, v2k = {}, {}
    for line in lines:
        
        k,v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k
    
    return k2v, v2k
    
def count_profs_and_gender(data: List[dict]):
    
    counter = defaultdict(Counter)
    for entry in data:
        gender, prof = entry["g"], entry["p"]
        counter[prof][gender] += 1
        
    return counter

train = load_dataset("../data/biasbios/train.pickle")
dev = load_dataset("../data/biasbios/dev.pickle")
test = load_dataset("../data/biasbios/test.pickle")
counter = count_profs_and_gender(train+dev+test)
p2i, i2p = load_dictionary("../data/biasbios/profession2index.txt")
g2i, i2g = load_dictionary("../data/biasbios/gender2index.txt")

path = "../data/bert_encode_biasbios/"
x_train = np.load(path + "train_cls.npy")
x_dev = np.load(path + "dev_cls.npy")
x_test = np.load(path + "test_cls.npy")

assert len(train) == len(x_train)
assert len(dev) == len(x_dev)
assert len(test) == len(x_test)

y_train = np.array([p2i[entry["p"]] for entry in train])
y_dev = np.array([p2i[entry["p"]] for entry in dev])
y_test = np.array([p2i[entry["p"]] for entry in test])

y_dev_gender = np.array([g2i[d["g"]] for d in dev])
y_train_gender = np.array([g2i[d["g"]] for d in train])
y_test_gender = np.array([g2i[d["g"]] for d in test])
y_train_gender_2d = np.asarray([y_train_gender, - y_train_gender + 1]).T

y_dev_gender.shape, y_train_gender.shape, y_test_gender.shape

A = np.dot(x_train.T, y_train_gender_2d) / x_train.shape[0]
t = time.time()
u, s, vh = np.linalg.svd(A, full_matrices=True)
elapsed = time.time() - t
print(f"sal took {elapsed} seconds when X is in the shape of {x_train.shape}")

MLP = False

def get_projection_matrix(num_clfs, X_train, Y_train_gender, X_dev, Y_dev_gender, Y_train_task, Y_dev_task, dim):

    is_autoregressive = True
    min_acc = 0.
    #noise = False
    dim = 768
    n = num_clfs
    #random_subset = 1.0
    start = time.time()
    TYPE= "svm"
    
    
    if MLP:
        x_train_gender = np.matmul(x_train, clf.coefs_[0]) + clf.intercepts_[0]
        x_dev_gender = np.matmul(x_dev, clf.coefs_[0]) + clf.intercepts_[0]
    else:
        x_train_gender = x_train.copy()
        x_dev_gender = x_dev.copy()
        
    
    if TYPE == "sgd":
        gender_clf = SGDClassifier
        params = {'loss': 'hinge', 'penalty': 'l2', 'fit_intercept': False, 'class_weight': None, 'n_jobs': 32}
    else:
        gender_clf = LinearSVC
        params = {'penalty': 'l2', 'C': 0.01, 'fit_intercept': True, 'class_weight': None, "dual": False}
        
    P,rowspace_projections, Ws = debias.get_debiasing_projection(gender_clf, params, n, dim, is_autoregressive, min_acc,
                                              X_train, Y_train_gender, X_dev, Y_dev_gender,
                                       Y_train_main=Y_train_task, Y_dev_main=Y_dev_task, by_class = True)
    print("time: {}".format(time.time() - start))
    return P,rowspace_projections, Ws

num_clfs = 300
y_dev_gender = np.array([g2i[d["g"]] for d in dev])
y_train_gender = np.array([g2i[d["g"]] for d in train])
idx = np.random.rand(x_train.shape[0]) < 1.
t = time.time()
P,rowspace_projections, Ws = get_projection_matrix(num_clfs, x_train[idx], y_train_gender[idx], x_dev, y_dev_gender, y_train, y_dev, 300)
elapsed = time.time() - t
print(f"INLP took {elapsed} seconds when X is in the shape of {x_train.shape}")


print(f"\n\n\n\n\n\nStart of Fair-professional FastText run-time measurement")

def load_word_vectors(fname):
    
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    vecs = model.vectors
    words = list(model.vocab.keys())
    return model, vecs, words


def get_embeddings_based_dataset(data: List[dict], word2vec_model, p2i, filter_stopwords = False):
    
    X, Y = [], []
    unk, total = 0., 0.
    unknown = []
    vocab_counter = Counter()
    
    for entry in tqdm.tqdm_notebook(data, total = len(data)):
        
        y = p2i[entry["p"]]
        words = entry["hard_text_tokenized"].split(" ")
        if filter_stopwords:
            words = [w for w in words if w.lower() not in STOPWORDS]
            
        vocab_counter.update(words) 
        bagofwords = np.sum([word2vec_model[w] if w in word2vec_model else word2vec_model["unk"] for w in words], axis = 0)
        #print(bagofwords.shape)
        X.append(bagofwords)
        Y.append(y)
        total += len(words)
        
        unknown_entry = [w for w in words if w not in word2vec_model]
        unknown.extend(unknown_entry)
        unk += len(unknown_entry)
    
    X = np.array(X)
    Y = np.array(Y)
    print("% unknown: {}".format(unk/total))
    return X,Y,unknown,vocab_counter

train = load_dataset("../data/biasbios/train.pickle")
dev = load_dataset("../data/biasbios/dev.pickle")
test = load_dataset("../data/biasbios/test.pickle")

p2i, i2p = load_dictionary("../data/biasbios/profession2index.txt")
g2i, i2g = load_dictionary("../data/biasbios/gender2index.txt")

word2vec, vecs, words = load_word_vectors("../data/embeddings/crawl-300d-2M.vec")
x_train, y_train, unknown_train, vocab_counter_train = get_embeddings_based_dataset(train, word2vec, p2i)
x_dev, y_dev, unknown_dev, vocab_counter_dev =  get_embeddings_based_dataset(dev, word2vec, p2i)
x_test, y_test, unknown_test, vocab_counter_test =  get_embeddings_based_dataset(test, word2vec, p2i)

y_dev_gender = np.array([g2i[d["g"]] for d in dev])
y_test_gender = np.array([g2i[d["g"]] for d in test])
y_train_gender = np.array([g2i[d["g"]] for d in train])
y_train_gender_2d = np.asarray([y_train_gender, - y_train_gender + 1]).T
y_dev_gender.shape, y_train_gender.shape, y_test_gender.shape
t = time.time()
A = np.dot(x_train.T, y_train_gender_2d) / x_train.shape[0]
u, s, vh = np.linalg.svd(A, full_matrices=True)
elapsed = time.time() - t

print(f"sal took {elapsed} seconds when X is in the shape of {x_train.shape}")

def get_projection_matrix(num_clfs, X_train, Y_train, X_dev, Y_dev, Y_train_task, Y_dev_task, dim, all_data_prob, by_class = False):

    is_autoregressive = True
    min_acc = 0.
    dim = 300
    n = num_clfs
    random_subset = 1
    start = time.time()
    TYPE= "svm"
    penalty = "l2"
    MLP = False
    
    if MLP:
        x_train_gender = np.matmul(X_train, clf.coefs_[0]) + clf.intercepts_[0]
        x_dev_gender = np.matmul(X_dev, clf.coefs_[0]) + clf.intercepts_[0]
    else:
        x_train_gender = X_train.copy()
        x_dev_gender = X_dev.copy()
        
    
    if TYPE == "sgd":
        print("using sgd")
        gender_clf = SGDClassifier
        params = {'alpha': 0.01, 'penalty': penalty, 'loss': 'hinge', 'fit_intercept': True, 'class_weight': "balanced", 'n_jobs': 16}
    elif TYPE == "svm":
        gender_clf = LinearSVC
        params = {'fit_intercept': True, 'C': 0.3, 'class_weight': None, "dual": False}
    elif TYPE == "perceptron":
        gender_clf = Perceptron
        params = {'max_iter': 1000, 'fit_intercept': True, 'class_weight': None}
    elif TYPE == "logistic":
        gender_clf = LogisticRegression
        params = {}
        
    result = debias.get_debiasing_projection(gender_clf, params, n, dim, is_autoregressive, min_acc,
                                              x_train_gender, Y_train, x_dev_gender, Y_dev,
                                       Y_train_main=Y_train_task, Y_dev_main=Y_dev_task, 
                                        by_class = by_class)
    print("time: {}".format(time.time() - start))
    return result

# was c=0.15, num_clfs=130
num_clfs = 150
Y_dev_gender = np.array([g2i[d["g"]] for d in dev])
Y_test_gender = np.array([g2i[d["g"]] for d in test])
Y_train_gender = np.array([g2i[d["g"]] for d in train])
t = time.time()
P, rowspace_projs, Ws = get_projection_matrix(num_clfs, x_train, y_train_gender, x_dev, y_dev_gender, y_train, y_dev, 300, 0.0, by_class= True)
elapsed = time.time() - t
print(f"INLP took {elapsed} seconds when X is in the shape of {x_train.shape}")


print(f"\n\n\n\n\n\nStart of Fair-sentiment run-time measurement")

ratio = 0.5

saved_dataset = np.load(f"../data/saved_models/fair_emoji_sent_race/{ratio}/all.npz")

x_train = saved_dataset['x_train']
y_m_train = saved_dataset['y_m_train']
y_p_train = saved_dataset['y_p_train']
y_p_train_2d = np.asarray([y_p_train, - y_p_train + 1]).T

x_dev = saved_dataset['x_dev']
y_p_dev = saved_dataset['y_p_dev']
y_m_dev = saved_dataset['y_m_dev']

t = time.time()
A = np.dot(x_train.T, y_p_train_2d) / x_train.shape[0]
u, s, vh = np.linalg.svd(A, full_matrices=True)
elapsed = time.time() - t

print(f"SAL took {elapsed} seconds when X is in the shape of {x_train.shape}")


print(f"\n\n\n\n\n\nStart of Word Embedding run-time measurement")

saved_dataset = np.load("../data/saved_models/general/all.npz")
X_dev = saved_dataset['x_dev']
X_train = saved_dataset['x_train']
X_test = saved_dataset['x_test']

Y_dev = saved_dataset['y_p_dev']
Y_train = saved_dataset['y_p_train']
Y_test = saved_dataset['y_p_test']

# Y_dev_label = Y_dev
# Y_train_label = Y_train
# Y_test_label = Y_test

Y_dev_2d = np.asarray([Y_dev, -Y_dev + 1]).T
Y_train_2d = np.asarray([Y_train, -Y_train + 1]).T
Y_test_2d = np.asarray([Y_test, -Y_test + 1]).T

Y_train.shape

t = time.time()
A = np.dot(X_train.T, Y_train_2d) / X_train.shape[0]
u, s, vh = np.linalg.svd(A, full_matrices=True)
elapsed = time.time() - t
print(f"sal took {elapsed} seconds when X is in the shape of {X_train.shape}")

gender_clf = LinearSVC

params_svc = {'fit_intercept': False, 'class_weight': None, "dual": False, 'random_state': 0}
params = params_svc
n = 35
min_acc = 0
is_autoregressive = True
dropout_rate = 0

t = time.time()
P, rowspace_projs, Ws = debias.get_debiasing_projection(gender_clf, params, n, 300, is_autoregressive, min_acc,
                                    X_train, Y_train, X_dev, Y_dev,
                                       Y_train_main=None, Y_dev_main=None, 
                                        by_class = False, dropout_rate = dropout_rate)
elapsed = time.time() - t
print(f"INLP took {elapsed} seconds when X is in the shape of {X_train.shape}")