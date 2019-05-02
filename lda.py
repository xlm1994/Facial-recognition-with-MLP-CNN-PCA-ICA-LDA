from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Dropout
from keras.callbacks import EarlyStopping
import keras

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from numpy import mean, std, argmax

batch_size = 256
tol = 0.001

lfw_dataset = fetch_lfw_people(min_faces_per_person=100)

_, h, w = lfw_dataset.images.shape
X = lfw_dataset.data
y_original = lfw_dataset.target
y = keras.utils.to_categorical(y_original)
target_names = lfw_dataset.target_names
num_classes = target_names.shape[0]

X_train_og, X_test_og, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_components = 100
lda = LinearDiscriminantAnalysis(n_components=n_components).fit(X_train_og,argmax(y_train, axis=1))

X_train = lda.transform(X_train_og)
X_test = lda.transform(X_test_og)

def train_data():
    model = Sequential()
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=5000,
            validation_split=0.2,
            callbacks=[es],
            verbose=0)

    score = model.evaluate(X_test, y_test, verbose=0)

    return score[1]

accs = []
for i in range(20):
    acc = train_data()
    accs += [acc]

print("Average accuracy: ", mean(accs))
print("Standard deviation of accuracy: ", std(accs))