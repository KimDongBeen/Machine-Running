# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from util import load_Fashion_MNIST, draw_result, visualization


class ModelMgr():
    def __init__(self, target_class=10, data_number = 100):
        self.target_class = target_class

        print('\n load dataset ...')
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = load_Fashion_MNIST(100)

    def train(self):
        print('\n train start ...')

        model = self.get_my_model() 
        #model = self.get_sample_model() 

        hp = self.get_hyperparameter() 
        model.summary() 
        print('*****************')
        print('\tbatch size :', hp['batch_size'])
        print('\tepochs :', hp['epochs'])
        print('\toptimizer :', hp['optimizer'])
        print('\tlearning rate :', hp['learning_rate'])
        print('*****************')

    
        model.compile(optimizer=hp['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

        if hp['epochs'] > 100: 
            hp['epochs'] = 100

        history = model.fit(self.train_images, self.train_labels,
                            batch_size=hp['batch_size'], epochs=hp['epochs'], verbose=2,
                            validation_split=0.4)

        self.model = model
        self.history = history
        history.history['hypers'] = hp

    def get_hyperparameter(self):
     
        hyper = dict()
        hyper['batch_size'] = 370 
        hyper['epochs'] = 50 
        hyper['learning_rate'] = 0.01 
        hyper['optimizer'] = keras.optimizers.Adam(learning_rate=hyper['learning_rate'])  ## default: SGD
        return hyper

    def get_sample_model(self):
        ## 샘플 모델 입니다.
        ## 모델 구현시 참고하세요.
        model = tf.keras.models.Sequential()
        model.add(keras.layers.Dense(128, input_shape=(784,), name='dense_layer', activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(128, name='dense_layer_2', activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(self.target_class, name='dense_layer_3', activation='softmax'))
        return model

    def get_my_model(self):
     
        model = tf.keras.models.Sequential()
        model.add(keras.layers.Dense(370, input_shape=(784,), name='dense_layer', activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(350, name='dense_layer_2', activation='relu'))
        model.add(keras.layers.Dropout(0.6))
        model.add(keras.layers.Dense(self.target_class, name='dense_layer_3', activation='softmax'))
     
        return model

    def test(self, model=None):
        print('\ntest model')
        if model is None:
            model = self.model
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        train_loss, train_acc = self.model.evaluate(self.train_images, self.train_labels)
        print('test dataset result : ', test_acc)
        print('train dataset result : ', train_acc)

    def visualization_result(self, file_path='saves/result_visualization.png'):
      
        predictions = self.model.predict(self.test_images)
        visualization(predictions, self.test_images, self.test_labels, file_path=file_path)

    def draw_history_graph(self, file_path='saves/result.png'):
       
        print('\nvisualize results : \"{}\"'.format(file_path))
        draw_result(self.history.history, file_path=file_path)

    def load_model(self, model_path='saves/model.h5'):
        print('\nload model : \"{}\"'.format(model_path))
        self.model = tf.keras.models.load_model(model_path)

    def save_model(self, model_path='saves/model.h5'):
        print('\nsave model : \"{}\"'.format(model_path))
        self.model.save(model_path)


if __name__ == '__main__':
    trained_model = None
    #trained_model = 'saves/model.h5' 

    modelMgr = ModelMgr(data_number=100)
    if trained_model is None:
        modelMgr.train() 
        modelMgr.save_model() 
        modelMgr.test() 
        modelMgr.visualization_result() 
        modelMgr.draw_history_graph() 
    else:
        modelMgr.load_model(trained_model)
        modelMgr.test()
