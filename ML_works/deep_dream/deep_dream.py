import numpy as np
from keras import backend as K
from keras.applications import inception_v3

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)


model.summary()
input()

# disable training
K.set_learning_phase(0)

layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
}

layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)

for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    # prevent the shadow at edges
    loss += coeff*K.sum(K, square(activation[:, 2:-2, 2:-2, :]))/scaling


dream = model.input

grads = K.gradients(loss, dream)[0]

grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grads_value = outs[1]
    return loss_value, grads_value


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grads_value = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...loss value at ', i, ': ', loss_value)
        x += step*grads_value
    return x


step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 20

# if loss is too large, stop training
max_loss = 10.

base_image_dir = ''  # TODO

img = preprocess_image(base_image_dir)

# zoom the image while layers goes lower
original_shape = img.shape[1:3]
succesive_shape = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim/(octave_scale**i))
                   for dim in original_shape])
    succesive_shape.append(shape)

# reverse the list
succesive_shape = succesive_shape[::-1]

# TODO
