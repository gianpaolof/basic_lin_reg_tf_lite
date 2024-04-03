My first example of a basic linear regression for Android with tflite.

The app has simply an edit text where I insert an integer, a button to be pressed to start tflite inference, a text view to display the result.

![immagine](https://github.com/gianpaolof/basic_lin_reg_tf_lite/assets/6586650/2f86da96-c828-4347-b9c4-974c3bcb3dcf)

The model:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ys = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])


model = keras.Sequential([
    layers.Dense(1, input_shape=[1])  # Single input layer
])


model.compile(optimizer='sgd', loss='mean_squared_error')


model.fit(xs, ys, epochs=500)


new_x_values = np.array([30, 31, 32, 33])
predictions = model.predict(new_x_values)
print("Predictions for x =", new_x_values, ":", predictions.flatten())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```


