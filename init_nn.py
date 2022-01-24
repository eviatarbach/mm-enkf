from utils import NNPredictor

#################
# General Setup #
#################
m = 40  # size of the state space
p = 20  # Number of obs at each time step (50%)
std_m = 0.1  # standard deviation of model noise
std_o = 1.  # standard devation of observational noise

ncycle = 40  # Number of cycles
nepochs_init = 40 # Number of epochs for initializing the weights
nepochs = 20 # Number of epochs during training in a cycle
Texpe = 2000 # Length of the experiment in model time unit

######################

##########################
# Machine learning setup #
##########################
# Parameters of the neural network (architecture + training):
param_nn = {'archi': ((24, 5, 'relu', 0.0), (37, 5, 'relu', 0.0)),  # CNN layer setup
	'bilin': True,  # activate bilinear layer
	'batchnorm': True,  # activate batchnorm normalization
	'reg': ('ridge', 1e-4),  # L2 regularization for the last layer
	'weighted': True,  # Using a variance matrix for the loss function
	'finetuning': False,  # Deactivate a finetuning of the last layer after optimization
	'npred': 1,  # Number of forecast time step in the loss function
	'Nepochs': nepochs,  # Number of epochs
	'batch_size': 256  # Batchsize during the training
}
nn = NNPredictor(m, **param_nn)

# Load the neural net with  weights
nn._smodel.load_weights('weights_nn.h5')