import csv
import text_processor
import recorder
import modelizer
import timeit
import copy
import predictor
import random_forest

def read_train_data(read_fn, tk_case, gram_num, rec_text_fn, for_non):
    total_text_word_cases = []
    word_cases_count_dict = {}

    with open(read_fn, mode = 'r') as file:
        csvFile = csv.reader(file)

        for row in csvFile:
            if len(row) == 0:
                continue
            
            text_word_cases = []
            lined = row[0].splitlines()
            for line in lined:

                text_word_cases = text_processor.setting_tokenizer(line, tk_case)

                text_processor.normalize_set(text_word_cases)
                text_processor.n_gram(text_word_cases, gram_num)
                modelizer.count_text_word_cases(text_word_cases, word_cases_count_dict)

                if for_non == False:
                    modelizer.count_text_word_cases(text_word_cases, word_cases_count_dict)
                    total_text_word_cases.append(text_word_cases)
                
                total_text_word_cases.append(text_word_cases)
    
    recorder.record_text_word_cases(total_text_word_cases, rec_text_fn)
    
    return [total_text_word_cases, word_cases_count_dict]


# main
# initialize setting
start = timeit.default_timer()

# main settings
tk_case = True
default_sort_order = True
high_sort_order = True
low_sort_order = False
run_case = True

max_depth = 100
num_trees = 100
num_split_candidates = 2

tree_lim = 200

# model settings
gram_num = 200
high_freq = 6000
low_freq = 10
alpha_num = 1

# case settings
train_neg_fn = '../data/train.negative.csv'
train_non_fn = '../data/train.non-negative.csv'
# train_neg_fn = '../data/mytrain.negative.csv'
# train_non_fn = '../data/mytrain.non-negative.csv'
rec_train_neg_fn = '../record/train.negative.texts.txt'
rec_train_non_fn = '../record/train.non-negative.texts.txt'

tmp = read_train_data(train_neg_fn, tk_case, gram_num, rec_train_neg_fn, True)
train_neg_texts = tmp[0]
main_neg_cases_count_dict = modelizer.sort_word_cases(tmp[1], default_sort_order)

tmp = read_train_data(train_non_fn, tk_case, gram_num, rec_train_non_fn, False)
# tmp = read_train_data(train_non_fn, tk_case, gram_num, rec_train_non_fn, True)
train_non_texts = tmp[0]
main_non_cases_count_dict = modelizer.sort_word_cases(tmp[1], default_sort_order)

main_model = modelizer.mk_model(alpha_num, train_neg_texts, train_non_texts, main_neg_cases_count_dict, main_non_cases_count_dict)

if run_case == True:
    # start main train
    tmp_neg_count_dict = modelizer.make_tmp_data(main_neg_cases_count_dict, default_sort_order)
    tmp_non_count_dict = modelizer.make_tmp_data(main_non_cases_count_dict, default_sort_order)
    tmp_model = copy.deepcopy(main_model)

    name = '../analysis/main/main.csv'
    # recorder.init_test_analysis(name)
    recorder.init_forest()
    
    modelizer.finalize_model(tmp_model, high_freq, low_freq, tmp_neg_count_dict, tmp_non_count_dict, run_case, default_sort_order)
    recorder.direct_test(name, str(high_freq)+'/'+str(low_freq))

    # new RF ================================================================================================================
    train_samples = []

    modelizer.mk_samples(train_samples, tmp_model, train_neg_texts, True)
    modelizer.mk_samples(train_samples, tmp_model, train_non_texts, False)

    print("> recording train samples")
    recorder.record_samples(train_samples, 'train.samples-model')

    print("> building random forest samples")
    num_split_candidates = len(list(train_samples[0][0].keys()))
    # if num_split_candidates == 0:
    #     num_split_candidates = 1
    for i in range(1):
        trees = random_forest.build_random_forest(train_samples, max_depth, num_trees, num_split_candidates)
        predictor.test_random_forest(trees, tmp_model, gram_num, tk_case, max_depth, num_trees, num_split_candidates)
    
    # end new RF ============================================================================================================

    # exec(open("predictor.py").read())
    print()
else:
    # start main train
    tmp_neg_count_dict = modelizer.make_tmp_data(main_neg_cases_count_dict, default_sort_order)
    tmp_non_count_dict = modelizer.make_tmp_data(main_non_cases_count_dict, default_sort_order)
    tmp_model = copy.deepcopy(main_model)

    name = '../analysis/case1/max_depth.csv'
    recorder.init_test_analysis(name)
    
    modelizer.finalize_model(tmp_model, high_freq, low_freq, tmp_neg_count_dict, tmp_non_count_dict, True, default_sort_order)
    recorder.direct_test(name, str(high_freq)+'/'+str(low_freq))

    # new RF ================================================================================================================
    # clf = RandomForestClassifier(random_state=0)

    train_samples = []

    modelizer.mk_samples(train_samples, tmp_model, train_neg_texts, True)
    modelizer.mk_samples(train_samples, tmp_model, train_non_texts, False)

    # print("> scaling sample features")
    # train_samples = StandardScaler().fit(train_samples).transform(train_samples)
    print("> recording train samples")
    recorder.record_samples(train_samples, 'train.samples-model')

    max_depth = 1
    while(max_depth < tree_lim):
        recorder.init_forest()

        print("> building random forest samples")
        num_split_candidates = len(list(train_samples[0][0].keys())) // 2
        trees = random_forest.build_random_forest(train_samples, max_depth, num_trees, num_split_candidates)
        # clf.fit(train_samples, train_samples_classes)

        predictor.test_random_forest(trees, tmp_model, gram_num, tk_case, max_depth, num_trees)
        print()

        max_depth += 20
    
    # end new RF ============================================================================================================

    # exec(open("predictor.py").read())
    # print()

stop = timeit.default_timer()
print('Time: ', stop - start)
