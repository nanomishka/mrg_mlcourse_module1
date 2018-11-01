# mrg_mlcourse_module1

```
# обучение
python3 train.py --x_train_dir=data/train-images --y_train_dir=data/train-labels --model_output_dir=model_params

# тестирование
python3 predict.py --x_test_dir=data/test-images --y_test_dir=data/test-labels --model_input_dir=model_params
```

_Результаты для модели "model_params"_
```
             precision    recall  f1-score   support

          0       0.96      0.91      0.94      1033
          1       0.96      0.92      0.94      1190
          2       0.82      0.89      0.85       950
          3       0.84      0.86      0.85       980
          4       0.89      0.85      0.87      1027
          5       0.78      0.82      0.80       847
          6       0.90      0.91      0.91       950
          7       0.88      0.88      0.88      1029
          8       0.82      0.75      0.78      1064
          9       0.78      0.85      0.82       930

avg / total       0.87      0.87      0.87     10000
```
