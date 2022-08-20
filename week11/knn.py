import csv
import random
import matplotlib.pyplot as plt

with open("iris.csv", mode='r') as csv_file:
    full_data = list(csv.DictReader(csv_file))
random.shuffle(full_data)

training, testing = full_data[0:120], full_data[120:150]


def distanceSquared(p1, p2):
    return sum([(float(p2[attr]) - float(p1[attr])) ** 2 for attr in p1 if attr != 'class'])


k_list = list([k for k in range(2, 20)])
accuracy = list()

for k in range(2, 20):

    n, d = 0, 0

    for point in training:
        neighbors = sorted([(distanceSquared(point, val), val['class']) for val in testing])[0:k]
        neighbor_classes = [x[1] for x in neighbors]
        frequency = dict((val, neighbor_classes.count(val)) for val in set(neighbor_classes))
        predicted_class = max(frequency, key=frequency.get)
        if predicted_class == point['class']:
            n += 1
        d += 1
    accuracy.append(n / d)

plt.plot(k_list, accuracy)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()
