import tensorflow as tf
import numpy as np
import pickle
import json
import random

from pathlib import Path

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
tf.keras.backend.set_floatx('float64')

    
class WaveLayer(Layer):
    """Custom wave layer for continuous rho approximation
    """
    
    def __init__(self):
        super(WaveLayer, self).__init__(name='wave_layer')
        
    def build(self, input_shape):
        self.w_1 = self.add_weight(shape=(1,))
        self.w_2 = self.add_weight(shape=(1,))
        self.w_3 = self.add_weight(shape=(1,))

    def call(self, X):
        t, x, rho = X[:, 0:1], X[:, 1:2], X[:, 2:3]   
        # rho scaling (based on non-dimensionalization)
        rho_1 = tf.math.sqrt(rho)
        rho_2 = rho
        z = (self.w_1*rho_1)*x + (self.w_2*rho_2)*t + self.w_3
        return z
    

class NeuralNet(Model):
    """NeuralNet class that is used to approximate the solution function.
       Configuration is based on Fisher's equation.
    """
    
    # From config (set as class attributes)
    args = ['version', 'seed', 'rho', 'model',
            'N_hidden', 'N_neurons', 'activation', 
            'N_epochs', 'learning_rate', 'decay_rate',
            'N_ICBC', 'N_col', 'lambda_F']
    
    # default path where log files are saved to
    log_path = Path('logs')
    
    
    def __init__(
            self, 
            config: dict, 
            verbose: bool = True,
        ) -> None:
        """Initialize NeuralNet instance with arguments provided by config.
           Sets computational domain, builds architecture and sets logger.

        Args:
            config (dict): Configuration dictionary.
            verbose (bool): Whether to print network architecture
        """  
        
        # call parent constructor & build NN
        super().__init__(name='Neural_Network')  
        # load class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])         
        tf.random.set_seed(self.seed)  
        np.random.seed(self.seed)
        random.seed(self.seed)

        # set computational domain (scaler)
        self._set_domain()
        
        # build network architecture
        self._build_layers(verbose) 
        
        # create callback instance
        self._set_logger(config)        
   

    def _set_domain(
            self
        ) -> None:
        """Sets the computational domain and system parameters.
           Domain information is used for feature scaling (only used for time and rho domain).
        """
        t_min, t_max = 0, 0.004
        self.mu = 10

        # Continuous rho approximation
        if isinstance(self.rho, list):
            (r_min, r_max) = self.rho
            # Feature scaling for rho is performed within the (custom) wave layer
            if 'wave' in self.model:
                self.X_min = tf.constant([t_min, 0, 0], dtype=tf.float64)
                self.X_max = tf.constant([t_max, 1, 1], dtype=tf.float64)  
            # else use standard feature scaling         
            else:
                self.X_min = tf.constant([t_min, 0, r_min], dtype=tf.float64)
                self.X_max = tf.constant([t_max, 1, r_max], dtype=tf.float64)  

        # Discrete rho approximation
        else:
            self.X_min = tf.constant([t_min, 0], dtype=tf.float64)
            self.X_max = tf.constant([t_max, 1], dtype=tf.float64)    

     
    def _build_layers(
            self, 
            verbose: bool = True,
        ) -> None:
        """Build neural network architecture.
           Based on whether discrete or continuous rho approximation,
           an wave layer is used, different architectures are built.

        Args:
            verbose (bool): Whether to print network architecture
        """
        # Nested model to overwrite call function
        self.u_net = Sequential(name=self.model)

        # Continous rho approximation 
        if isinstance(self.rho, list):
            # Input layer (t,x,rho)
            self.u_net.add(InputLayer(shape=(3,)))
            # Wave layer (custom layer)
            if 'wave' in self.model:
                self.u_net.add(WaveLayer())

        # Discrete rho approximation
        else:
            # Input layer (t,x)
            self.u_net.add(InputLayer(shape=(2,)))
            # Wave layer (linear layer)
            if 'wave' in self.model:
                self.u_net.add(Dense(units=1, 
                                     activation='linear',
                                     name='wave_layer'))

        # sine activation (not build-in by default)
        if self.activation == 'sin':
            self.activation = tf.math.sin
        
        # Hidden layers
        for i in range(self.N_hidden):
            self.u_net.add(Dense(units=self.N_neurons, 
                                 activation=self.activation,
                                 name=f'hidden_{i}'))

        # Sigmoid activation for u to be inbetween 0 and 1
        self.u_net.add(Dense(1, activation='sigmoid', name='output'))

        if verbose:
            self.u_net.summary() 
            print("*** Layers build ***")                 
 

    def call(
            self, 
            X: tf.Tensor,
        ) -> tf.Tensor:
        """Forward pass through network.

        Args:
            X (tf.tensor): Input data points

        Returns:
            u_pred (tf.tensor): Corresponding network prediction
        """
        # For discrete rho approximation ...
        if not isinstance(self.rho, list):
            # drop rho dimension
            X = X[:, 0:2]

        X = self.scale(X)
        u_pred = self.u_net(X)
        return u_pred


    def scale(
            self, 
            X: tf.Tensor,
        ) -> tf.Tensor:
        """Feature scaling process.

        Args:
            X (tf.tensor): Original input data points

        Returns:
            X_scaled (tf.tensor): Scaled input data points
        """
        return 2 * (X - self.X_min) / (self.X_max - self.X_min)


    def train(
            self, 
            data
        ) -> None:
        """Main function that triggers model training. Calls _train_step() function
        for a number of epochs specifid in the config. Also, prints and writes logs.

        Args:
            data: DataLoader instance that provides the training data
        """

        # Learning rate decay
        learning_rate = ExponentialDecay(initial_learning_rate=self.learning_rate,
                                         decay_steps=1000,
                                         decay_rate=self.decay_rate)
                                               
        # Adam optimizer with default settings for momentum estimates
        self.optimizer = Adam(learning_rate) 

        # Test set data
        X_test, u_test = data.sample_domain(N=1024) 

        print("Training started...")
        for epoch in range(self.N_epochs):

            # Sample new data points at each epoch
            X_ICBC, u_ICBC = data.sample_ICBC(self.N_ICBC)
            X_col, _ = data.sample_domain(self.N_col)   
            # Auxiliary data for continuous rho approximation
            if isinstance(self.rho, list):
                X_aux, u_aux = data.sample_domain(N=1024, rho=self.rho)
            # or data-driven ANN models for discrete rho approximation
            elif 'ANN' in self.model:
                X_aux, u_aux = data.sample_domain(N=1024) 
            # for PINN models leave empty
            else:
                X_aux = u_aux = None
            
            train_logs = self._train_step(X_ICBC, u_ICBC, X_col, X_aux, u_aux) 
            test_logs = self._test_step(X_test, u_test)
            
            # provide logs to callback
            logs = {**train_logs, **test_logs}
            self._write_logs(logs, epoch)
     
        # Final model evaluation
        if isinstance(self.rho, list):
            X_eval, u_eval = data.sample_domain(N=10000)
        else:
            X_eval, u_eval, _, _ = data.mesh(N=100)
        eval_logs = self._evaluate(X_eval, u_eval)
        self._write_logs(eval_logs, self.N_epochs)

        # Saving logs and network weights
        self._save_logs()
        self._save_weights()
        print("Training finished!")
        return
        

    @tf.function
    def _train_step(
            self, 
            X_ICBC: tf.Tensor, 
            u_ICBC: tf.Tensor, 
            X_col: tf.Tensor, 
            X_aux: tf.Tensor, 
            u_aux: tf.Tensor,
        ) -> dict:
        """Train step function that performs a single update step.

        Args:
            X_ICBC (tf.Tensor): Training data for initial and boundary conditions
            u_ICBC (tf.Tensor): Respective data labels
            X_col (tf.Tensor): Collocation data points
            X_aux (tf.Tensor): Auxiliary training data for ANN models and continuous-rho
            u_aux (tf.Tensor): Respective data lavels

        Returns:
            logs (dict): training logs
        """
        
        with tf.GradientTape() as tape:     
            # Initial and boundary condition loss   
            loss_ICBC = self.loss_u(X_ICBC, u_ICBC)
            # Total loss function
            loss_train = loss_ICBC
            # Physics loss only for PINN models
            loss_F = self.loss_F(X_col)
            if 'PINN' in self.model:
                loss_train += loss_F
            # Auxiliary data loss only for ANN models and continuous rho
            if X_aux is not None:
                loss_aux = self.loss_u(X_aux, u_aux)
                loss_train += loss_aux
            
        # retrieve gradients
        grads = tape.gradient(loss_train, self.weights)            
        # perform single GD step
        self.optimizer.apply_gradients(zip(grads, self.weights))            
        return {'loss_train': loss_train, 'loss_F': loss_F, 'loss_ICBC': loss_ICBC}

    
    @tf.function
    def _test_step(
            self, 
            X_test: tf.Tensor, 
            u_test: tf.Tensor
        ) -> dict:   
        """Test step function to assess model performance during training

        Args:
            X_test (tf.Tensor): Test set data
            u_test (tf.Tensor): Respective labels

        Returns:
            logs (dict): test logs
        """ 
        loss_test = self.loss_u(X_test, u_test) 
        return {'loss_test': loss_test}
       
   
    @tf.function  
    def loss_u(self, 
            X: tf.Tensor, 
            u_true: tf.Tensor,
        ) -> tf.Variable:
        """Data loss function - Mean squared error.

        Args:
            X (tf.Tensor): Input coordinates
            u_true (tf.Tensor): True data label

        Returns:
            loss_u (tf.Variable): single loss value
        """    
        u_pred = self(X)
        return tf.reduce_mean(tf.square(u_pred - u_true))   
    
    
    @tf.function 
    def loss_F(
            self, 
            X: tf.Tensor,
        ) -> tf.Variable:
        """Physics loss function. Uses _F_residuals() to obtain 
        physics loss residuals.

        Args:
            X (tf.Tensor): Collocation points

        Returns:
            loss_F (tf.Variable): single loss value
        """    
        _, _, _, _, F_res_weighted = self._F_residuals(X)
        
        loss_F = tf.reduce_mean(tf.square(F_res_weighted)) 
        return loss_F
    
    
    def _F_residuals(
            self, 
            X: tf.Tensor,
        ) -> tuple[tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor]:
        """Evalutes physics loss residuals (and its components for post-processing)

        Args:
            X (tf.Tensor): Collocation points

        Returns:
            u_t (tf.Tensor): First-order time derivatives
            u_xx (tf.Tensor): Second-order space derivative
            source_term (tf.Tensor): Reaction source term
            F_res (tf.Tensor): unweighted physics loss residuals
            F_res_weighted (tf.Tensor): weighted physics loss residuals
        """    
        # Gradient tape to obtain derivatives
        with tf.GradientTape() as t:
            t.watch(X)
            with tf.GradientTape() as tt:
                tt.watch(X)
                u = self(X)
            u_d = tt.batch_jacobian(u, X)        
        u_dd = t.batch_jacobian(u_d, X)  
        # Time and space derivatives
        u_t = u_d[:, :, 0]
        u_xx = u_dd[:, :, 1, 1]
        # Reaction term
        rho = X[:, 2:3]
        source_term = rho * (u - u**2)
        # Physics loss residuals
        F_res = u_t - self.mu * u_xx - source_term
        # Weighted residuals
        F_res_weighted = F_res / (self.lambda_F* tf.abs(source_term) + 1)
        return u_t, u_xx, source_term, F_res, F_res_weighted
                  

    def _evaluate(
            self,
            X_eval: tf.Tensor,
            u_eval: tf.Tensor,
        ) -> dict:
        """Final evaluation step determining the L2-Error

        Args:
            X_eval (tf.Tensor): Evaluation set data
            u_eval (tf.Tensor): Respective labels

        Returns:
            logs (dict): evaluation logs
        """
        # Network prediction
        u_pred = self(X_eval)
        # L2 error
        L2_error = tf.math.sqrt(tf.reduce_mean((u_pred-u_eval)**2))
        return {'L2_error': L2_error}

   
    def _set_logger(
            self, 
            config: dict,
        ) -> None:
        """Prepares storing and writing logs.

        Args:
            config (dict): Configuration dictionary
        """      
        # create model path to save logs
        self.model_path = self.log_path.joinpath(self.version)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # create log from config file
        self.log = config.copy()
      
        self.freq_log = config['freq_log']
        self.freq_print = config['freq_print']
        self.keys_print = config['keys_print']       
        
        # determines digits for 'fancy' log printing
        self.digits = int(np.log10(self.N_epochs)+1)      
  
        
    def _write_logs(
            self, 
            logs: dict, 
            epoch: int,
        ) -> None:  
        """Writes and prints logs.

        Args:
            logs (dict): Training and test logs
            epoch (int): Current epoch in optimization
        """      
        # store training logs
        if (epoch % self.freq_log) == 0:
            # exceptions errors are used to catch the different data formats provided
            for key, item in logs.items():
                # append if list already exists
                try:
                    self.log[key].append(item.numpy().astype(np.float64))
                # create list otherwise
                except KeyError:
                    try:
                        self.log[key] = [item.numpy().astype(np.float64)]
                    # if list is given 
                    except AttributeError:
                        self.log[key] = item          
        # print training logs
        if (epoch % self.freq_print) == 0:
            
            s = f"{epoch:{self.digits}}/{self.N_epochs}"
            for key in self.keys_print:
                try:
                    s += f" | {key}: {logs[key]:2.2e}"
                except:
                    pass
            print(s) 
            
            
    def _save_logs(
            self
        ) -> None:
        """Saves recorded logs to log file.
        """        
        log_file = self.model_path.joinpath(f'log_{self.seed}.json')       
        with open(log_file, "w") as f:
            json.dump(self.log, f, indent=2)
        print("*** logs saved ***")

    
    def _save_weights(
            self, 
            flag='', 
            verbose=True
        ) -> None:
        """Saves network weights.
        """                
        weights_file = self.model_path.joinpath(f'weights{flag}_{self.seed}.pkl')
        with open(weights_file, 'wb') as pickle_file:
            pickle.dump(self.u_net.get_weights(), pickle_file)  
        if verbose:                  
            print("*** model weights saved ***")
  

    def load_weights(
            self, 
            weights_file, 
            verbose=True
        ) -> None:
        """Loads network weights.
        """   
        with open(weights_file, 'rb') as pickle_file:
            weights = pickle.load(pickle_file)
        self.u_net.set_weights(weights)
        if verbose:
            print("*** model weights loaded ***")
        
        