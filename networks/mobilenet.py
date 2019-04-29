from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D
from keras.models import Model, Sequential

name = 'mobilenet'

input_shape = (128, 128, 3) # (128, 128, 1) for grayscale image. You can change the 128 to any input size you want
batch_size = 32
color_mode = 'rgb'  # 'rgb' or 'grayscale'

def build_model():
	base_model = MobileNet(input_shape=input_shape, include_top=False)
	x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
	x = Dropout(1e-3, name='dropout')(x)
	#TODO: replace the 1 with the number of classes your want to train
	x = Dense(1, activation='softmax', name='predictions')(x) 
	return Model(inputs=[base_model.input], outputs=[x])

