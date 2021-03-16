import csv
import nltk
import numpy as np
from numpy.random import seed
from gensim.models import Word2Vec
# from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

def get_lstm_model():
    json_file = open('./lstm.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('./lstm.h5') 
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    return model

dataset = []
wordVecDataset = []

training_file = open('./training.csv', encoding = 'utf-8', errors = 'ignore')
csv_reader = csv.reader(training_file, delimiter = ',')

line_count = 0
for row in csv_reader:
    if line_count != 0:
        tokens = nltk.word_tokenize(row[1])
        dataset.append([tokens, row[2]])
        wordVecDataset.append(tokens)
    line_count += 1

model = Word2Vec(wordVecDataset, min_count = 1, size = 50, workers = 3, window = 3, sg = 1)
model.save('word2vec.model')
model = Word2Vec.load('./word2vec.model')

x_train, y_train, x_test, y_test = [], [], [], []

for tweet in dataset:
    tweet_tokens = tweet[0]
    embeddings = []
    for token in tweet_tokens:
        embeddings.append([round(abs(sum(model[token])) * 10, 4)])
    padding = [[0]] * (128 - len(embeddings))
    embeddings = embeddings.copy() + padding
    x_train.append(embeddings)
    label = int(tweet[1])
    if label == 0:
        y_train.append([0, 1])
    else:
        y_train.append([1, 0])

dataset = []
testing_file = open('./testing.csv', encoding = 'utf-8', errors = 'ignore')
csv_reader = csv.reader(testing_file, delimiter = ',')

line_count = 0
for row in csv_reader:
    if line_count != 0:
        tokens = nltk.word_tokenize(row[1])
        dataset.append([tokens, row[2]])
    line_count += 1

for tweet in dataset:
    tweet_tokens = tweet[0]
    embeddings = []
    for token in tweet_tokens:
        embeddings.append([round(abs(sum(model[token])) * 10, 4)])
    padding = [[0]] * (128 - len(embeddings))
    embeddings = embeddings.copy() + padding
    x_test.append(embeddings)
    label = int(tweet[1])
    if label == 0:
        y_test.append([0, 1])
    else:
        y_test.append([1, 0])

x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape(len(x_train),128,3)
x_test = x_test.reshape(len(x_test),128,3)


model = Sequential()
model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(128,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)


# #lstm
# model = Sequential()
# model.add(Bidirectional(LSTM(128, input_shape = (128, 3))))
# model.add(Dense(2, activation = 'softmax'))
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
# model.fit(x_train, y_train, batch_size = 200, epochs = 60, shuffle = True)
# pred_output = model.predict(x_test, batch_size = 200)

# model_json = model.to_json()
# with open('./lstm.json', 'w') as json_file:
#     json_file.write(model_json)
# model.save_weights('./lstm.h5')

# model = get_lstm_model()

# matrix = [[0, 0], [0, 0]]
# actual_yes, actual_no, predicted_yes = 0, 0, 0

# for i in range(len(y_test)):
#     if y_test[i][0] == 1:
#         actual_yes += 1
#     elif y_test[i][0] == 0:
#         actual_no += 1
#     if pred_output[i][0] > 0.5:
#         predicted_yes += 1
#     x, y = 0, 0
#     if y_test[i][0] > 0.5:
#         x = 1
#     else:
#         x = 0
#     if pred_output[i][0] > 0.5:
#         y = 1
#     else:
#         y = 0
#     matrix[x][y] += 1

# TP = matrix[1][1]
# TN = matrix[0][0]
# FP = matrix[0][1]
# FN = matrix[1][0]

# total = len(y_test)
# accuracy = (TP + TN) / total
# misclassfication = (FP + FN) / total
# recall = TP / actual_yes
# specificity = TN / actual_no
# precision = TP / predicted_yes
# f_score = 2 * ((recall * precision) / (recall + precision))

# print("Confusion Matrix:", matrix)
# print("Accuracy: ", accuracy)
# print("Misclassfication Rate: ", misclassfication)
# print("True Positive Rate (Recall): ", recall)
# print("True Negative Rate (Specificity): ", specificity)
# print("Precision: ", precision)
# print("F Score: ", f_score)
