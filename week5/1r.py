import csv

def nearest_fourth(number: float):
    return round(number * 4) / 4


with open("iris.csv", mode='r') as csv_file:
    full_data = list(csv.DictReader(csv_file))
    # hardcoded preprocessing
    for value in full_data:
        for key in ["sepallength", "sepalwidth", "pedalling", "petalwidth"]:
            value[key] = "nearest 1/4: " + str(nearest_fourth(float(value[key])))

headers = list(full_data[0].keys())[:-1]
clazz = list(full_data[0].keys())[-1]
header_accuracy = dict()

storage = dict()
for header in headers:
    each = dict((row[header], {'freq': 0, 'values': dict()}) for row in full_data)
    for row in full_data:
        working_dict = each[row[header]]
        working_dict["freq"] = working_dict["freq"] + 1
        if row[clazz] not in working_dict['values']:
            working_dict['values'][row[clazz]] = 0
        working_dict['values'][row[clazz]] = working_dict['values'][row[clazz]] + 1

        working_dict["clazz"] = max(working_dict['values'], key=working_dict['values'].get)
        working_dict["wrongly_classified"] = working_dict['freq'] - working_dict['values'][working_dict["clazz"]]

    n = sum([each[value]['wrongly_classified'] for value in each])
    d = sum([each[value]['freq'] for value in each])
    value = {'error_rate': (n / d), 'data': each}
    storage[header] = value

most_acc_attr = min(storage, key=lambda k: storage[k]['error_rate'])

print("Attribute:", most_acc_attr)
for key in storage[most_acc_attr]['data']:
    print("   ", key, "==>", storage[most_acc_attr]['data'][key]['clazz'])

actual_predicted = dict()
classes = set([value['class'] for value in full_data])

for clazz in classes:
    actual_predicted[clazz] = dict()
    for clazz_inner in classes:
        actual_predicted[clazz][clazz_inner] = 0

for value in full_data:
    actual = value['class']

    # [actual][predicted]
    actual_predicted[actual][storage[most_acc_attr]['data'][value[most_acc_attr]]['clazz']] += 1

print("Confusion Matrix:", actual_predicted)
