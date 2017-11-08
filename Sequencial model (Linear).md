Sequential model: linear stack of layers.

말그대로 레이어를 일자로 쭉 나열한 형태를 말하는데, CNN의 대표적인 모델이다.

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

input shape는 28\*28=784 이며, CNN의 대표적 함수들인 Relu와 Softmax를 포함시킨다.

즉 Input -\> relu -\> softmax 의 모델이다. 이는 CNN레이어가 1개인 아주 간단한 모델이다.

위를 그냥 아주 간단하게 add() 함수로 처리할수도 있다.

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

모델에는 input shape를 명확히 정의해 주어야 한다. 

그 이유는, Sequential 모델의 첫 번째 레이어는 반드시 이를 알아야만 동작하도록 설계되었기 때문이다.

위의 input\_dim은 input dimension을 말한다. 

한줄형태의 input일 경우 다음과 같이 input\_length를 사용하면 된다.

dense는 출력형태이다.

```python
model = Sequential()
model.add(Dense(32, input_length=784))
model.add(Activation('relu'))
```

batch size를 fix하고 싶다면 다음과 같이 명령해주면 된다.

(RNN 모델에서 유용하다)

```python
model.add(batch_size = 32)
model.add(input_shape = (6, 8))
```

위 코드의 기본 batch size는 32 이고, 6\*8 형태이다.

즉 default batch shape는 (32, ,6, 8) 이다.

아래 단계부터는 Compilation을 표현했다.

이 단계는 compile() 함수를 사용하는데, 총 세 가지의 변수를 받는다.

1. Optimizer. 예로 rmsprop 이나 adagrad 가 있다. 이는 Keras의 [optimizer 페이지](https://keras.io/optimizers/)를 자세히 보는편이 좋다.
2. Loss function. 이 값을 최소화 하기 위해 모델을 작성하는데, 이는 categorical\_crossentropy나 mse 함수가 있다. 이는 [losses function 페이지](https://keras.io/losses/)를 보는것이 좋다.
3. List of metrics. 개인이 만든 함수 등으로 problem의 classification을 정의하는 방법이다. (ex. metrics=[‘accuracy’] )

```python
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

```python
# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```