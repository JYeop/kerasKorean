Keras 설치는 pip로 간단히 설치할 수 있다.

sudo pip install keras 

git을 이용하는법은 다음과 같다.

git clone https://github.com/fchollet/keras.git
cd keras
sudo python setup.py install

Keras는 RNN, CNN을 지원하는 Tensorflow 써드파티 라이브러리다.

하나하나 빌드하면 굉장히 오래 걸리는 CNN의 모델을 다음과 같이 간단하게 작성해볼 수 있다.

```python
from keras.models import Sequential

model = Sequential()
```

Sequential 모델은 단순히 Convolution layers를 linear 형태로 단순나열하는 모델을 말한다.

레이어를 쌓는 작업은 add() 함수로 가능하다.

```python
from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
```

위처럼 자신이 원하는 NN 모델을 만들고 나면, compile() 함수로 컴파일하면 사실상 끝난다.

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

위에 나온 optimizer는 SGD, RMSprop 등의 모델이 있는데, 이는 추후에 다루도록 한다.

아래는 optimizer 설정을 좀 더 상세히한 예시 코드이다.

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

이후 다음과 같이 loss를 계산할 수 있다.

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

혹은 prediction 함수를 데이터로부터 바로 실행시킬 수 있다.

```python
classes = model.predict(x_test, batch_size=128)
```



