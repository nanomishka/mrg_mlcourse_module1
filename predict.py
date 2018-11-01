# coding: utf-8
import argparse
import numpy as np

from mnist import MNIST
from sklearn.metrics import classification_report

from hw1.helpers import TimeHelper, DataHelper

np.random.seed(420)

#################################################

step_1 = TimeHelper('1: Import data')

parser = argparse.ArgumentParser()
parser.add_argument('--x_test_dir')
parser.add_argument('--y_test_dir')
parser.add_argument('--model_input_dir')

args = parser.parse_args()
file_x = args.x_test_dir
file_y = args.y_test_dir
file_model = args.model_input_dir

source = MNIST(gz=True, return_type='numpy')
source.test_img_fname = file_x
source.test_lbl_fname = file_y
X, y = source.load_testing()

step_1.finish()
#################################################

step_2 = TimeHelper('2: Preprocessing')

X = DataHelper.normalize(X)
X = DataHelper.bias(X)

step_2.finish()
#################################################

step_3 = TimeHelper('3: Import model')
initial_weights = np.load(file_model)
step_3.finish()
#################################################

step_4 = TimeHelper('4: Predict')
predict_y = np.argmax(np.matmul(X, initial_weights.transpose()), 1)
step_4.finish()

print(classification_report(predict_y, y))
