import csv

def record_text_word_cases(text_word_cases, rec_text_fn):
    with open(rec_text_fn, 'w') as file:
        for cases_text in text_word_cases:
            file.write(str(cases_text))
            file.write('\n')

def record_case_count_dict(count_dict, cound_dict_fn):
    with open(cound_dict_fn, 'w') as file:
        writer = csv.writer(file)

        for case in count_dict:
            row = [case, count_dict[case]]
            writer.writerow(row)

def record_model(model):
    with open('../model/predictor-model.csv', 'w') as file:
        writer = csv.writer(file)

        for case in model:
            row = [case, model[case][0], model[case][1]]
            writer.writerow(row)

def init_test_analysis(analysis_name):
    directories = analysis_name.split('/')
    splits = directories[3].split('.')
    set_name = splits[0]

    with open(analysis_name, 'w') as file:
        writer = csv.writer(file)

        header = [set_name, 'tp', 'fn', 'fp', 'tn', 'accuracy', 'precision', 'recall']
        writer.writerow(header)
        
def direct_test(analysis_name, val):
    data = [analysis_name, val]
    with open('../analysis/direction.csv', 'w') as file:
        writer = csv.writer(file)

        writer.writerow(data)

def record_samples(samples, fn):
    with open('../model/'+fn+'.txt', 'w') as file:
        for i in range(len(samples)):
            line = str(i+1) + ". " + compacted_sample(samples[i])
            file.write(line)

def compacted_sample(sample):
    n = 5
    keys = list(sample[0].keys())

    line = '({'
    for i in range(5):
        line += '\'' + str(keys[i]) + '\': ' + str(sample[0][keys[i]])
        if i != n-1:
            line += ', '
        else:
            line += ', ...'
    line += '}, ' + str(sample[1]) + ')\n'

    return line

def init_forest():
    with open('../record/random_forest.txt', 'w') as file:
        line = 'Random Forest Information\n'
        file.write(line)

def write_forest(tree):
    with open('../record/random_forest.txt', 'a') as file:
        line = str(tree) + '\n'
        file.write(line)