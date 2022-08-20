import csv
import pprint as pp


def nearest_xth(number: float, xth: int):
    return round(number * xth) / xth


with open("iris.csv", mode='r') as csv_file:
    full_data = list(csv.DictReader(csv_file))
    # hardcoded preprocessing
    for value in full_data:
        for ky in ["sepallength", "sepalwidth", "petallength", "petalwidth"]:
            value[ky] = "nearest 1/4: " + str(nearest_xth(float(value[ky]), 4))


def organize_by_classification(data: list):
    res = dict()
    for val in data:
        key = val['class']
        if key not in res:
            res[key] = list()
        res[key].append(val)
    return res


def probability_of_class(classification: dict):
    res = dict()
    for key in classification:
        res[key] = len(classification[key])
    den = sum(res.values())
    for key in res:
        res[key] = res[key] / den
    return res


# Probability of attribute value given class dictionary:
# dict structure: class --> attribute value --> probability
def attribute_given_class(data: list, clazz: str):
    headers = list(data[0].keys())[:-1]
    res = dict()
    for header in headers:  # for each column
        each = dict((row[header], 0) for row in full_data if row['class'] == clazz)  # store all attribute values for it
        for row in full_data:
            if not row['class'] == clazz:
                continue
            each[row[header]] = each[row[header]] + 1  # store frequency
        den = sum(each.values())  # convert to frequencey
        for key in each:
            each[key] = each[key] / den
        res[header] = each
    return res


def probability_table(data: list, prob_of_classes: dict):
    res = dict()
    for key in prob_of_classes:
        res[key] = attribute_given_class(data, key)
    return res


# Given the output of attribute_given_class, predict P(X|C)
def probability_x_given_class(datapoint, chance_dict: dict, clazz: str):
    chance = 1
    headers = list([key for key in datapoint.keys() if key != "class"])
    for key in headers:
        diction = chance_dict[clazz][key]
        if datapoint[key] in diction:
            chance *= diction[datapoint[key]]
        else:
            chance *= 0
    return chance


def most_probable_class(datapoint, class_probabilites: dict, chance_dict: dict):
    res = dict()
    for clazz in class_probabilites:
        res[clazz] = class_probabilites[clazz] * probability_x_given_class(datapoint, chance_dict, clazz)
    return max(res, key=res.get)


class_prob = probability_of_class(organize_by_classification(full_data))  # P(C)
attr_from_class = probability_table(full_data, class_prob)  # Part of P(X|C)

n_ = 0
d_ = 0
for val in full_data:
    acc = val['class']
    prob = most_probable_class(val, class_prob, attr_from_class)
    print(acc, prob)
    d_ += 1
    if acc == prob:
        n_ += 1
print((n_ / d_))

actual_predicted = dict()
classes = set([value['class'] for value in full_data])

for clazz in classes:
    actual_predicted[clazz] = dict()
    for clazz_inner in classes:
        actual_predicted[clazz][clazz_inner] = 0

for value in full_data:
    actual = value['class']
    # [actual][predicted]
    actual_predicted[actual][most_probable_class(value, class_prob, attr_from_class)] += 1
print("Confusion Matrix:", actual_predicted)
