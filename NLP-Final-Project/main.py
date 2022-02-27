from utile import *
from evaluation import *

data_path = os.path.join('..', 'Data', 'train_full.csv')
valid_test_path = os.path.join('..', 'Data', 'test_full.csv')

model = "RF"

if model == "NN":
    x, x_test, exp_range_train, y, y_test, exp_range_test, exp2id = process_data(data_path)
    model = train(x, y, exp2id)
    evaluate(model, x_test, y_test, exp_range_test)    
    
elif model == "LR":
    traindf, valid_testdf = process_data2(data_path, valid_test_path)
    model, abv_dict, vect = train_lr(traindf)
    evaluate_lr(model, valid_testdf, abv_dict, vect)

elif model == "RF":
    traindf, valid_testdf = process_data2(data_path, valid_test_path)
    model, vect = train_rf(traindf)    
    evaluate_rf(model, valid_testdf, vect)












