import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

@keras.saving.register_keras_serializable()
class PinnModel(keras.Model):
    def __init__(self, true_phy_params, train_params,
                 n_hid_layers, n_neurons, act="tanh", **kwargs):
        super().__init__(**kwargs)
        self.true_params  = true_phy_params
        self.params = self.true_params
        self.train_params = train_params
        self.n_hid_layers = n_hid_layers
        self.n_neurons = n_neurons
        self.act = act

        #Architecture
        self.hidden_layers = [
            keras.layers.Dense(n_neurons, activation=act,
                                  name=f"dense_{i}")
            for i in range(n_hid_layers)
        ]
        self.out_layer = tf.keras.layers.Dense(1, activation="linear")

        #physics setup
        for key, value in self.params.items():
            if key in self.train_params: #joint training
                self.params[key] = self.add_weight(name=key, shape=(), initializer="zeros",
                            trainable=True, dtype=tf.float32)
            else: #convert to tensorflow
                self.params[key] = tf.constant(value, dtype=tf.float32, name=key)

    def call(self, inputs, training=False):
        x = inputs
        for lyr in self.hidden_layers:
            x = lyr(x)
        return self.out_layer(x)

    def get_config(self):
        return { #add to super config
            "true_phy_params": self.true_params,
            "train_params"   : self.train_params,
            "n_hid_layers"   : self.n_hid_layers,
            "n_neurons"      : self.n_neurons,
            "act"            : self.act,
            **super().get_config()
        }

class CombinedLoss(keras.Loss):
    def __init__(self, model: keras.Model, X_c: np.ndarray,
                 scaler_X: MinMaxScaler, scaler_Y: StandardScaler, alpha: float):
        super().__init__()

        self.model = model
        self.alpha_d = alpha  #dictates data/physics ratio
        self.alpha_p = 1-alpha
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        self.t_c, self.u, self.v, self.x = (tf.constant(X_c[:, i]) for i in range(X_c.shape[1]))
        self.X_c_norm = tf.constant(scaler_X.transform(X_c), dtype=tf.float32)

    def data_loss(self, y_true, y_pred):
        sqr_diff = tf.square(y_true - y_pred)
        return tf.reduce_mean(sqr_diff)

    def physics_loss(self):
        g = self.model.params['g']
        c0 = self.model.params['c0']
        c1 = self.model.params['c1']
        Cd = self.model.params['Cd']
        m = self.model.params['m']
        p = self.model.params['p']
        Af = self.model.params['Af']
        road_grade = self.model.params["road_grade"]

        a_norm = self.model(self.X_c_norm) #use PUNN to predict on collocation points
        a_norm = tf.reshape(a_norm, shape=([-1])) #flatten for shape matching
        a_phy = a_norm * self.scaler_Y.scale_ + self.scaler_Y.mean_ #denormalize

        #resistive forces
        Fr = (c0 + c1 * self.v) * (m*g*tf.math.cos(road_grade))
        Fa = (p * Cd * Af * self.v**2) / 2.0
        Fg = m * g * tf.math.sin(road_grade)

        #physics residuals
        flow = (self.u - Fr - Fa - Fg)/m
        res = a_phy - flow  # Newton's law

        return tf.reduce_mean(tf.square(res))

    def call(self, y_true, y_pred):
        data_loss = self.data_loss(y_true, y_pred)
        pde_loss = self.physics_loss()

        return(
            self.alpha_d * data_loss +
            self.alpha_p * pde_loss
        )