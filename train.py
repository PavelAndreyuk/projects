from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import load_model
import numpy as np

X_train=np.load('train.npy')
frames=X_train.shape[2]

frames=frames-frames%10

X_train=X_train[:,:,:frames]
X_train=X_train.reshape(-1,227,227,10)
X_train=np.expand_dims(X_train,axis=4)
Y_train=X_train.copy()

epochs=2
batch_size=1

if __name__=="__main__":

	model=load_model()

	callback_save = ModelCheckpoint("AnomalyDetector.h5", monitor='loss', save_best_only=True)

	callback_early_stopping = EarlyStopping(monitor='loss', patience=3)

	print('Model has been loaded')

	model.fit(X_train,Y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  callbacks = [callback_save,callback_early_stopping]
			  )