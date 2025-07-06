import numpy as np 
import keras
import tensorflow as tf
from keras.layers import Dense, Normalization, Rescaling
from keras import Loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import Callback

@keras.saving.register_keras_serializable()
class PinnModel(keras.Model):
    def __init__(self, true_phy_params, train_params,
                 n_hid_layers, n_neurons, 
                 scalerX_params:dict, scalerY_params:dict, act="tanh", **kwargs):
        super().__init__(**kwargs)
        self.true_params  = true_phy_params
        self.params = self.true_params
        self.train_params = train_params
        self.n_hid_layers = n_hid_layers
        self.n_neurons = n_neurons
        self.act = act
        self.scalerX_params = scalerX_params
        self.scalerY_params = scalerY_params

        #Architecture
        Xscale = np.array(self.scalerX_params['scale']).reshape(1, -1)
        Xmin = np.array(self.scalerX_params['min']).reshape(1,-1)
        self.input_rescaling = Rescaling(scale=Xscale, 
                                         offset=Xmin,
                                         name="input_rescaling_layer")
        self.hidden_layers = [
            Dense(n_neurons, activation=act,
                                  name=f"dense_{i}")
            for i in range(n_hid_layers)
        ]
        self.out_layer = Dense(1, activation="linear")
        #normX = (x-mean)/std -> x = normX*std+mean
        self.output_denorm = Rescaling(offset=self.scalerY_params['mean'],
                                       scale=self.scalerY_params['scale'],
                                        name='output_denorm_layer')
        #physics setup
        for key, value in self.params.items():
            if key in self.train_params.keys(): #if joint training
              init = keras.initializers.Constant(self.train_params[key])
              self.params[key] = self.add_weight(name=key, shape=(), initializer=init,
                                                   trainable=True, dtype=tf.float32)
            else: #for given true values
              self.params[key] = tf.constant(value, dtype=tf.float32, name=key)

    def call(self, inputs, training=False):
        x = inputs
        
        x = self.input_rescaling(x)        
        for lyr in self.hidden_layers:
            x = lyr(x)        
        x = self.out_layer(x)        
        x = self.output_denorm(x)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "true_phy_params": self.true_params,
            "train_params"   : self.train_params,
            "n_hid_layers"   : self.n_hid_layers,
            "n_neurons"      : self.n_neurons,
            "act"            : self.act,
            "scalerX_params" : {"min": self.scalerX_params['min'].tolist(),
                                "scale": self.scalerX_params['scale'].tolist()},
            "scalerY_params" : {"mean": self.scalerY_params['mean'].tolist(),
                                "scale": self.scalerY_params['scale'].tolist()},
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CombinedLoss(Loss):
    def __init__(self, model: keras.Model, X_c: np.ndarray, alpha: float):
        '''
        Assumes u feature in second column of X_c provided and v in third.
        Model should handle scaling internally.
        '''
        super().__init__()

        self.model = model
        self.alpha_d = alpha  #dictates data/physics ratio
        self.alpha_p = 1-alpha

        # Raw collocation points - model will handle scaling
        self.u_phy = tf.constant(X_c[:, 1], dtype=tf.float32)
        self.v_phy = tf.constant(X_c[:, 2], dtype=tf.float32)
        self.X_c = tf.constant(X_c, dtype=tf.float32)

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

        # Use raw collocation points - model handles scaling internally
        a_phy = self.model(self.X_c) #predict on collocation points
        a_phy = tf.reshape(a_phy, shape=([-1])) #flatten for shape matching

        #resistive forces
        Fr = (c0 + c1 * self.v_phy) * (m*g*tf.math.cos(road_grade))
        Fa = (p * Cd * Af * tf.square(self.v_phy)) / 2.0
        Fg = m *g*tf.math.sin(road_grade)

        #physics residuals
        flow = (self.u_phy - Fr - Fa - Fg)/m
        res = a_phy - flow  # Newton's law

        return tf.reduce_mean(tf.square(res))

    def call(self, y_true, y_pred):
        data_loss = self.data_loss(y_true, y_pred)
        pde_loss = self.physics_loss()

        return(
            self.alpha_d * data_loss +
            self.alpha_p * pde_loss
        )

class CustomWeightLog(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if len(self.model.train_params) > 0: #if joint training
            for key, value in self.model.params.items():
                logs[key] = value
    def on_train_end(self, logs=None):
        print(f"Custom trainable weights converged to:")
        for key in self.model.train_params: #in trainable params list
            value = self.model.params[key]
            true_value = self.model.true_params[key]
            percentage = abs((value-true_value)/true_value)*100
            print(f"{key} = {value.numpy():.5e}, true value = {true_value} ({percentage:.3f}% difference)")    