import tensorflow as tf
from keras_tqdm import TQDMCallback
from tensorflow.keras import backend as K 
from .utils import which_GPU

K.set_floatx("float32")

import silence_tensorflow.auto

from nvsmi import get_available_gpus

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


epochs=100

def nn(X_train, y_train):
    model10 = tf.keras.Sequential([
        tf.keras.layers.Input(X_train.shape[1]),
        tf.keras.layers.Dense(50, activation="relu", activity_regularizer=tf.keras.regularizers.l1_l2(3e-6, 6.5e-5)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(50, activation="relu", activity_regularizer=tf.keras.regularizers.l1_l2(3e-6, 6.5e-5)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(40, activation="relu", activity_regularizer=tf.keras.regularizers.l1_l2(3e-6, 6.5e-5)),
        tf.keras.layers.Dense(30, activation="relu", activity_regularizer=tf.keras.regularizers.l1_l2(3e-6, 6.5e-5)),
        tf.keras.layers.Dense(20, activation="relu", activity_regularizer=tf.keras.regularizers.l1_l2(3e-6, 6.5e-5)),
        tf.keras.layers.Dense(2, activation="sigmoid")
        ])

    model10.compile(optimizer=tf.keras.optimizers.SGD(1e-3),
                    loss="binary_crossentropy",
                    metrics=[tf.keras.metrics.AUC()])


    hist=model10.fit(
        X_train,
        y_train,
        epochs=epochs,
        verbose = 0,
        callbacks=[
            #TQDMCallback(),
            tf.keras.callbacks.EarlyStopping()
        ],
        validation_split=0.25
    )
    
    return hist, model10


def ae_lr(X_train, y_train, latent_dim, reconstruction_weight=0.5):

    n, d = X_train.shape

    classification_weight=1-reconstruction_weight

    input_ = tf.keras.Input(shape=(d,), )
    encoder = tf.keras.layers.Dense(d/2, 
                                    activation="linear",
                                    activity_regularizer=tf.keras.regularizers.l1_l2(3e-6, 6.5e-5))(input_)
    encoder = tf.keras.layers.Dense(latent_dim, activation="linear", name="latent_representation")(encoder)


    classifier = tf.keras.layers.Dense(1, activation="sigmoid", name="classification")(encoder)

    decoder = tf.keras.layers.Dense(d/2, activation="linear")(encoder)
    decoder = tf.keras.layers.Dense(d, activation="linear", name="reconstruction")(decoder)

    model = tf.keras.models.Model(input_, [classifier, decoder])

    #X_train, X_test, y_train, y_test = train_test_split(dset["X"], dset["y"], test_size=0.2)


    model.compile(optimizer='adam', 
                loss=[
                    "binary_crossentropy",
                    tf.keras.losses.MeanSquaredError()
                ],
                loss_weights=[
                    1-reconstruction_weight,  #classification weight
                    reconstruction_weight   #reconstruction weight
                ],
                metrics={
                    "classification": [tf.keras.metrics.AUC()]
                })

    hist = model.fit(X_train, 
                    [
                        y_train,
                        X_train
                    ], 
                    epochs=epochs, 
                    validation_split=0.25, 
                    callbacks=
                    [
                        tf.keras.callbacks.ReduceLROnPlateau("val_classification_loss"),
                        #TQDMCallback(),
                        tf.keras.callbacks.EarlyStopping(
                            "val_classification_loss", 
                            patience=5, 
                            restore_best_weights=True)
                    ],
                    verbose=0)
    return hist.history, model


def lr(X_train, y_train):
    n, d = X_train.shape

    clf = tf.keras.models.Sequential([
        tf.keras.Input(shape=(d,), ),
        tf.keras.layers.Dense(2, activation="sigmoid")
    ])

    clf.compile(
        loss="binary_crossentropy",
        metrics=tf.keras.metrics.AUC()
        )

    hist = clf.fit(
        X_train,
        y_train, 
        epochs=epochs,
        validation_split=0.25,
        callbacks=[
            #TQDMCallback(),
            tf.keras.callbacks.EarlyStopping(patience=3)
        ],
        verbose=0
    )

    return hist.history, clf

class TiedDense(tf.keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)
    
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias",
                                      shape=[self.dense.input_shape[-1]],
                                      initializer="zeros")
        super().build(batch_input_shape)
    
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

    def get_config(self):
        pass


def autoencoder(sizes, activation="linear", input_shape=10000):
    depth = len(sizes)
    input_layer = tf.keras.Input((input_shape,))
    encoder_layers = []
    decoder_layers = []
    
    for size in sizes:
        enc = tf.keras.layers.Dense(size, activation=activation)
        encoder_layers.append(enc)
    
    for size in reversed(sizes[:-1]):
        dec = tf.keras.layers.Dense(size, activation=activation)
        decoder_layers.append(dec)
        
    out = [input_layer] + encoder_layers + decoder_layers + [tf.keras.layers.Dense(input_shape, activation=activation)]
    return out

def compiled_autoencoder(X_train, sizes, activation="linear", input_shape=10000):
    depth = len(sizes)

    encoder_layers = []
    decoder_layers = []

    input_layer = tf.keras.Input((input_shape,))
    
    for size in sizes:
        enc = tf.keras.layers.Dense(size, activation=activation)
        encoder_layers.append(enc)
    
    for size in reversed(sizes[:-1]):
        dec = tf.keras.layers.Dense(size, activation=activation)
        decoder_layers.append(dec)
        
    out = [input_layer] + encoder_layers + decoder_layers + [tf.keras.layers.Dense(input_shape, activation=activation)]

    model = tf.keras.models.Sequential(out)

    model.compile(optimizer='adam', 
                loss = tf.keras.losses.MeanSquaredError()
    )

    hist = model.fit(
        X_train/2,
        X_train/2, 
        epochs=epochs, 
        validation_split=0.25, 
        callbacks=
                [
                    #tf.keras.callbacks.ReduceLROnPlateau("val_clf_loss"),
                    #TQDMCallback()
                ],
        verbose=0)

    return hist, model

def supervised_autoencoder(X_train, y_train, sizes, activation="linear", reconstruction_weight=0.5,l1=0, dropout=True):
    gpu_to_use = which_GPU()
    print(f"USING GPU: {gpu_to_use}")
    input_shape = X_train.shape[1]

    with tf.device(f"/GPU:{gpu_to_use}"):
        input_layer = tf.keras.Input((input_shape,), name="input")
        if len(sizes) == 1:
            enc = tf.keras.layers.Dense(sizes[0], activation=activation, name="latent_dim", activity_regularizer=tf.keras.regularizers.l1(1e-5))(input_layer)
        else:
            enc = tf.keras.layers.Dense(sizes[0], activation=activation, name="encoder_1", activity_regularizer=tf.keras.regularizers.l1(1e-5))(input_layer)

        if dropout:
            enc = tf.keras.layers.Dropout(0.4)(enc)

        for i, size in enumerate(sizes[1:]):
            name = "latent_dim" if size == sizes[-1] else f"encoder_{i+2}"
            enc = tf.keras.layers.Dense(size, activation=activation, name=name, activity_regularizer=tf.keras.regularizers.l1(1e-5))(enc)
            
        #clf = tf.keras.layers.Dense(50, activation="relu", activity_regularizer=tf.keras.regularizers.l1_l2(3e-6, 6.5e-5))(enc)

        #for i in [50, 40, 30, 20]:
        #    clf = tf.keras.layers.Dense(i, activation="relu", activity_regularizer=tf.keras.regularizers.l1_l2(3e-6, 6.5e-5))(clf)
        
        clf = tf.keras.layers.Dense(2, activation="sigmoid", name="clf")(enc)
    
        dec = enc
        for i, size in enumerate(reversed(sizes[:-1])):
            dec = tf.keras.layers.Dense(size, activation=activation, name=f"decoder_{i+1}", activity_regularizer=tf.keras.regularizers.l1(1e-5))(dec)

        #dec = tf.keras.layers.Dropout(0.4)(dec)    
        dec = tf.keras.layers.Dense(input_shape, activation=activation, name="decoder")(dec)
        
        model = tf.keras.models.Model(input_layer, [clf, dec])
        
        model.compile(optimizer='adam', 
                    loss={
                        "clf": "binary_crossentropy",
                        "decoder": tf.keras.losses.MeanSquaredError()
                    },
                    loss_weights={
                        "clf": 1-reconstruction_weight,
                        "decoder": reconstruction_weight
                    },
                    metrics={
                        "clf": [tf.keras.metrics.AUC(name="")]
                    })

        hist = model.fit(X_train/2, 
                        {
                            "clf": y_train,
                            "decoder": X_train/2
                        }, 
                        epochs=epochs, 
                        validation_split=0.25, 
                        callbacks=
                        [
                            #tf.keras.callbacks.ReduceLROnPlateau("val_clf_loss"),
                            #TQDMCallback()
                        ],
                        verbose=0)
        
    return hist, model