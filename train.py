# coding: utf-8
import argparse
import numpy as np

from mnist import MNIST
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from SVM import SVMHelper as SVM
from helpers import TimeHelper, DataHelper

np.random.seed(100)

#################################################

step_1 = TimeHelper('1: Import data')

parser = argparse.ArgumentParser()
parser.add_argument('--x_train_dir')
parser.add_argument('--y_train_dir')
parser.add_argument('--model_output_dir')

args = parser.parse_args()
file_x = args.x_train_dir
file_y = args.y_train_dir
file_model = args.model_output_dir

source = MNIST(gz=True, return_type='numpy')
source.train_img_fname = file_x
source.train_lbl_fname = file_y
X, y = source.load_training()
print(X.shape)

step_1.finish()
#################################################


step_2 = TimeHelper('2: Preprocessing')

X = DataHelper.normalize(X)
X = DataHelper.bias(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

digits = np.unique(y_train)
indexes = {}
for d in digits:
    indexes[d] = np.where(y_train == d)[0]
    print('\t"{}": {} %'.format(d, round(100 * len(indexes[d]) / len(y_train)) ))

# save empty features indexes for skipping in learning
empty_features = np.where(np.max(X, 0) == 0)[0]

initial_weights = np.ndarray(shape=(10, X.shape[1]))

step_2.finish()
#################################################

step_3 = TimeHelper('2: Learn model')

learning_rate = 0.1
C_range = [0.0001, 0.001, 0.1, 1.0, 10.0, 20.0, 50.0, 100.0, 1000.0]

best_f1 = 0
best_weights = None

for C_coef in C_range:
    print('Coefficient: {}:'.format(C_coef))

    # learn every digit-model
    for d in digits:
        print('Class "{}":'.format(d))
        initial_weights[d] = SVM.init_weights(X_train, -0.5, 0.5)
        initial_weights[d][empty_features] = 0

        loss_func = SVM.regul_loss_in_point(C_coef)

        y_mod = np.ones(len(y_train)) * -1
        y_mod[indexes[d]] = 1
        optimal_weights = SVM.gradient_descent(
            loss_func, X_train, y_mod, initial_weights[d], learning_rate, empty_features
        )
        initial_weights[d] = optimal_weights

    predict_y = np.argmax(np.matmul(X_val, initial_weights.transpose()), 1)
    f1 = f1_score(predict_y, y_val, average='macro')
    print("Coefficient: %f, f1_avg: %f" % (C_coef, f1))

    # Choose best weights
    if f1 > best_f1:
        best_f1 = f1
        best_weights = initial_weights

step_3.finish()
#################################################

step_4 = TimeHelper('2: Predict')
predict_y = np.argmax(np.matmul(X_test, best_weights.transpose()), 1)
step_4.finish()

print(classification_report(predict_y, y_test))

#################################################

# save model
best_weights.dump(file_model)
