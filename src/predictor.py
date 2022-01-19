import csv
import modelizer
import text_processor
import recorder
import random_forest

def predict_samples(trees, test_samples):
    print("> predicting test samples...")

    results = []
    for i in range(len(test_samples)):
        result = random_forest.forest_classify(trees, test_samples[i][0])
        results.append(result)

    print("> done predicting test samples...")

    return results

def read_test_data(read_fn, tk_case, gram_num, rec_text_fn):
    total_text_word_cases = []

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
                
                total_text_word_cases.append(text_word_cases)
    
    recorder.record_text_word_cases(total_text_word_cases, rec_text_fn)

    return total_text_word_cases

def get_model(model_fn):
    p_model = {}

    with open(model_fn, mode = 'r') as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            if len(lines) == 0:
                continue

            p_model[lines[0]] = [float(lines[1]), float(lines[2])]
    
    return p_model

# new RF ================================================================================================================
def test_random_forest(trees, tmp_model, gram_num, tk_case, max_depth, num_trees):
    # case settings
    test_neg_fn = '../data/test.negative.csv'
    test_non_fn = '../data/test.non-negative.csv'
    rec_test_neg_fn = '../record/test.negative.texts.txt'
    rec_test_non_fn = '../record/test.non-negative.texts.txt'

    test_neg_text_cases = read_test_data(test_neg_fn, tk_case, gram_num, rec_test_neg_fn)
    test_non_text_cases = read_test_data(test_non_fn, tk_case, gram_num, rec_test_non_fn)

    test_samples = []
    
    modelizer.mk_samples(test_samples, tmp_model, test_neg_text_cases, True)
    modelizer.mk_samples(test_samples, tmp_model, test_non_text_cases, False)

    # test_samples = StandardScaler().fit(test_samples).transform(test_samples)
    recorder.record_samples(test_samples, 'test.samples-model')

    # results = clf.predict(test_samples)
    results = predict_samples(trees, test_samples)
    # print(results)
    calc_statistics(test_samples, results, max_depth, num_trees)

def calc_statistics(test_samples, results, max_depth, num_trees):
    print("> calculate random forest test results...")

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(results)):
        if test_samples[i][1] == True and results[i] == True:
            tp += 1
        elif test_samples[i][1] == False and results[i] == True:
            fp += 1
        elif test_samples[i][1] == False and results[i] == False:
            tn += 1
        elif test_samples[i][1] == True and results[i] == False:
            fn += 1
    
    # print(tp)
    # print(fp)
    # print(tn)
    # print(fn)

    acc = (tp + tn) / (tp + fn + tn + fp)
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    print("> done calculating random forest test results...")
    
    result = [tp, tn, fp, fn, acc, prec, rec, max_depth, num_trees]
    modelizer.print_result_info(result)

# end new RF ============================================================================================================