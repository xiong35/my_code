from weather_LSTM import test_gen,
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('/root/MySource/GRU.h5')

x_test, y_test = next(test_gen)

predictions = model.predict(x_test)

print(len(predictions))
print(len(y_test))
