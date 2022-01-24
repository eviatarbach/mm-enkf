from keras import backend as K
import numpy as np
from keras.layers import Input, Lambda, BatchNormalization, Conv1D, Dropout, Add, Multiply, Concatenate
from keras.constraints import maxnorm
from keras.models import Model
from keras import regularizers
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# Size
m = 40

####################
# Neural net utils #
####################

# to make a periodic padding of a tensor
def keras_padding ( v ):
	if isinstance(v, int):
		v = (v, v)
	vleft, vright = v

	def padlayer ( x ):
		leftborder = x[..., -vleft:, :]
		rigthborder = x[..., :vright, :]
		return K.concatenate([leftborder, x, rigthborder], axis=-2)

	return padlayer

# Add an artificial feature (to handle the weights in the cost function)
def dummy_feature( x ):
	return K.concatenate([x,x],axis=-1)

class NNPredictor:
	def __init__ (self,m,archi,
	Ntrain=-1,npred=1,nin=1,
	Nepochs=10,bilin=False,batchnorm=True,
	weighted=True, reg=None,finetuning=True,
	batch_size=128,optimizer='Adagrad',patience=100):
		"""
		Main class to handle neural nets
		:param m: size of the model
		:param archi: architecture in form of a dictionnary of tuples (size, filter size, activation, dropout rate)
		:param Ntrain: Number of data taken as training (the rest is taken as test)
		:param npred: Nummber of forecast steps in the loss function
		:param nin: Number of time steps as input
		:param Nepochs: Number of epochs during traning
		:param bilin: Activate bilinera layer for the first layer
		:param batchnorm: Activate a batchnorm layer in input
		:param weighted: Use the inverse of diagonal covariance in the loss function (identity otherwise)
		:param reg: Regulariation of the last layer
		:param finetuning: Fintune the last layer using a linear regression after optimization
		:param batch_size: Batch size during the training
		:param optimizer: Optimizer used for training
		:param patience: Number of epochs to retain the best test score (has an effect only if Ntrain < size of data)
		"""
		assert nin==1 or npred==1, 'Time seq both in and out not implemented'
		self._m = m
		self._archi = archi
		self._Ntrain = Ntrain
		if np.isnan(npred):
			npred = 1
		self._npred = int(npred)
		self._nin = nin
		self._Nepochs = Nepochs
		self._bilin = bilin
		self._batchnorm = batchnorm
		self._batchnorm = batchnorm
		self._weighted = weighted
		self._batchsize = batch_size
		self._optimizer = optimizer
		self._verbfit = 1
		self._patience = patience
		if reg is None:
			self._reg = 'ridge',0
		else:
			self._reg = reg
		self._finetuning = finetuning
		self._smodel = self.buildmodels()

	def buildmodels(self):
		"""
		buid the neuronel model
		:return: return a tuple containing:
		the short model,
		the model with dummy output (to handle covariance),
		the recurrent model to handle Nf > 1
		All the three models share the same weights
		"""
		border = int(np.sum(np.array([kern//2 for fil,kern,activ,dropout in self._archi])))
		xin = Input(shape=(self._m,self._nin))
		x3 = None
		padlayer = keras_padding(border)
		x = Lambda(padlayer)(xin)
		if self._batchnorm:
			x = BatchNormalization()(x)
		bilintodo = self._bilin
		for nfil,nkern,activ,drop in self._archi:
			if bilintodo: #bilinear layer (only once)
				if drop > 0:  # Add the maxnormvalue
					x1 = Conv1D(nfil, nkern, activation=activ, kernel_constraint=maxnorm(3.))(x)
					x1 = Dropout(rate=drop)(x1)
					x2 = Conv1D(nfil, nkern, activation=activ, kernel_constraint=maxnorm(3.))(x)
					x2 = Dropout(rate=drop)(x2)
				else:
					x1 = Conv1D(nfil, nkern, activation=activ)(x)
					x2 = Conv1D(nfil, nkern, activation=activ)(x)
				x3 = Multiply()([x1,x2])

			if drop>0: #Add the maxnormvalue
				x = Conv1D(nfil,nkern,activation=activ,kernel_constraint=maxnorm(3.))(x)
				x =  Dropout(rate=drop)(x)
			else:
				x = Conv1D(nfil,nkern,activation=activ)(x)

			if bilintodo:
				x = Concatenate()([x, x3])
				bilintodo = False
		if self._reg[1]>0:
			if self._reg[0] == 'ridge':
				dy = Conv1D(1,1,activation='linear',kernel_regularizer=regularizers.l2(self._reg[1]))(x)
			else:
				raise NotImplementedError(self._reg[0],'regularization no implemented')
		else:
			dy = Conv1D(1,1,activation='linear')(x)
		soutput = Add()([xin,dy])

		smodel = Model(xin,soutput)
		return smodel