# SER_GRU_Text
 
Speech Emotion Recognition System using GRU(Gated recurrent units, RNN)

Dependencies: tensorflow, numpy, pickle, random
* Use pre-processed data sets (conversation transcript)

Corpos: IEMOCAP (English, audios from 10 people (5 m / 5 f), 10 emotions)

Train (Features): use pre-trained embedding (GloVe)

Result: Best Testing Accuracy: 64.9%

Parameter seeting: 
batch_size = 128
encoder_size = 128
num_layer = 1
hidden_dim = 200
learning rate = 0.001
num_train_steps = 10000
drop out = 0.3

*MEMO: Files (pre-trained data sets) should be locased in 'Data' folder

*Reference: Yoon, S., Byun, S., & Jung, K. (2018, December). Multimodal speech emotion recognition using audio and text. In 2018 IEEE Spoken Language Technology Workshop (SLT) (pp. 112-118). IEEE.
