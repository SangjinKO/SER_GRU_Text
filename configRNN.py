N_CATEGORY = 4

#     Training
CAL_ACCURACY_FROM = 0
MAX_EARLY_STOP_COUNT = 7
EPOCH_PER_VALID_FREQ = 0.3

#     DATA (Training / Validation / Test )
DATA_TRAIN_LABEL = 'train_label.npy'
DATA_TRAIN_TRANS = 'train_nlp_trans.npy'

DATA_DEV_LABEL = 'dev_label.npy'
DATA_DEV_TRANS = 'dev_nlp_trans.npy'

DATA_TEST_LABEL = 'test_label.npy'
DATA_TEST_TRANS = 'test_nlp_trans.npy'

DIC = 'dic.pkl'
GLOVE = 'W_embedding.npy'

#     NLP
N_SEQ_MAX_NLP = 128
DIM_WORD_EMBEDDING = 100
EMBEDDING_TRAIN = True
IS_LOGGING = False