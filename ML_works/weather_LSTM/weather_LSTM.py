
import numpy as np
import os
import matplot.pyplot as plt

data_dir = '/root/MySource/jena'  # TODO

fname = os.path.join(data_dir, 'climate.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header)-1))


# the first column of the data is Date
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# the new first column is temp(erature)
temp = float_data[:, 1]
plt.plot(range(1440), temp[:1440])
plt.show()

# standardize the data
# take the first 200000 time steps as training data
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


# def a generator that returns a tuple (samples, targets)
# samples for a batch of input
# targets for a list of temperatures
def generator(
    data,           # standardized data
    lookback,       # how many previous time steps should consider
                    # in the next procces
    delay,          # how many time steps between now and target
    min_index,      # define to choose which steps
    max_index,
    shuffle=False,  # whether to shuffle the data
    batch_size=128,
    step=6          # the interval step between two samples
):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(
                min_index+lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, mim(i+batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),          # num of [input for the next procces]
                            lookback // step,   # num of previous data
                            data.shape[-1]))    # num of features
        # num of [output for the next procces]
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            # indice mean which steps should be drawn for the next procces
            indices = range(rows[j]-lookback, rows[j], step)
            # mat[(range)] means to choose the rows in the range
            samples[j] = data[indices]
            targets[j] = data[rows[j]+delay][1]
        yield samples, targets
