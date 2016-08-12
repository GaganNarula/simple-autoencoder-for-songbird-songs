def vec_abs(X_sample):
    import numpy as np
    X_abs = np.power(X_sample, 2)
    X_abs = np.sum(X_abs, axis=1)
    X_abs = np.power(X_abs, 0.5)
    return X_abs

def show_plot(Y):
    import matplotlib.pyplot as plt
    plt.plot(Y)
    plt.show()

# vectorizing a function
def vec_np(a):
    import numpy as np
    return np.vectorize(a)

def data_split(x, y, a, b, c=0):
    # a denotes the percent of total number of samples needed
    # b denotes required percent of test data relative to train data
    # c denotes required percent of val data relative to train data
    from sklearn.cross_validation import train_test_split
    x, junk1, y, junk2 = train_test_split(x, y, test_size=1 - a, random_state=16)
    x, x_test, y, y_test = train_test_split(x, y, test_size=b / (1 + b + c), random_state=37)
    if c == 0:
        x_train = x
        y_train = y
        x_val = None
        y_val = None
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=c / (1 + c), random_state=42)
    return (x_train, y_train, x_test, y_test, x_val, y_val)

def to_utils(n, x, y=None, z=None):
    # n denotes the number of classes
    # x, y and z are the label arrays
    from keras.utils import np_utils
    if y is None:
        return (np_utils.to_categorical(x, n))
    elif z is None:
        return (np_utils.to_categorical(x, n), np_utils.to_categorical(y, n))
    else:
        return (np_utils.to_categorical(x, n), np_utils.to_categorical(y, n), np_utils.to_categorical(z, n))

# builds a basic autoencoder given the list of number of hidden units
def form_ae(n_hidden, dropout=0, activation='relu', loss='mean_squared_error', optimizer='rmsprop'):
    from keras.models import Sequential
    from keras.layers import containers
    from keras.layers.core import Dense, AutoEncoder, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    encoder_layers = []
    decoder_layers = []
    for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:]):
        encoder_layers.append([Dense(n_out, input_dim=n_in, activation=activation)])
        if dropout > 0:
            encoder_layers.append([Dropout(dropout)])
        decoder_layers.append([Dense(n_in, input_dim=n_out, activation=activation)])
        if dropout > 0 & n_out != n_hidden[len(n_hidden) - 1]:
            decoder_layers.append([Dropout(dropout)])
    decoder_layers.reverse()
    encoder = containers.Sequential(encoder_layers)
    decoder = containers.Sequential(decoder_layers)
    ae = Sequential()
    ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    ae.compile(loss=loss, optimizer=optimizer)
    ae.summary()
    return ae

from keras import callbacks

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# trains encoders/decoders on a given training data given the list of number of hidden units
def ae_pretrain(x_train, n_hidden, greedy=True, batch_size=16, nb_epoch=3, dropout=0, activation='relu', loss='mean_squared_error', optimizer='rmsprop'):
    from keras.models import Sequential
    from keras.layers import containers
    from keras.layers.core import Dense, AutoEncoder, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    trained_encoders = []
    trained_decoders = []
    x_tmp = x_train
    x_predicted = []
    x_prediction_scores = []
    loss_history = []
    if greedy:
        for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:]):
            print('Training: Input {} -> Output {}'.format(n_in, n_out))
            ae = Sequential()
            encoder = containers.Sequential([Dense(n_out, input_dim=n_in, activation=activation)])
            if dropout > 0:
                encoder.add([Dropout(dropout)])
            decoder = containers.Sequential([Dense(n_in, input_dim=n_out, activation=activation)])
            ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
            ae.compile(loss=loss, optimizer=optimizer)
            history = LossHistory()
            tmp = ae.fit(x_tmp, x_tmp, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[history])
            decoder = ae.layers[0].decoder
            if dropout > 0 & n_out != n_hidden[len(n_hidden) - 1]:
                decoder.add(Dropout(dropout))
            trained_encoders.append(ae.layers[0].encoder)
            trained_decoders.append(decoder)
            x_tmp = ae.predict(x_tmp)
            x_predicted.append(x_tmp)
            x_prediction_score.append(tmp)
            loss_history.append(history.losses)
        return (trained_encoders, trained_decoders, loss_history, x_predicted, x_prediction_score)
    else:
        encoder_layers = []
        decoder_layers = []
        for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:]):
            encoder_layers.append([Dense(n_out, input_dim=n_in, activation=activation)])
            if dropout > 0:
                encoder_layers.append([Dropout(dropout)])
            decoder_layers.append([Dense(n_in, input_dim=n_out, activation=activation)])
            if dropout > 0 & n_out != n_hidden[len(n_hidden) - 1]:
                decoder_layers.append([Dropout(dropout)])
            decoder.reverse()
        encoder = containers.Sequential(encoder_layers)
        decoder = containers.Sequential(decoder_layers)
        ae = Sequential()
        ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
        ae.compile(loss=loss, optimizer=optimizer)
        tmp = ae.fit(x_tmp, x_tmp, batch_size=batch_size, nb_epoch=nb_epoch)
        return ([encoder], [decoder], [ae.predict(x_tmp)], [tmp])

# the function is written with above function in mind
def encoder_trained_mlp(encoder, loss='mean_squared_error', optimizer='rmsprop'):
    from keras.models import Sequential
    from keras.layers import containers
    from keras.layers.core import Dense, AutoEncoder, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    model = Sequential()
    for encoder_layer in encoder:
        model.add(encoder_layer)
        model.layers[-1].set_weights(encoder_layer.get_weights())
    model.compile(loss=loss, optimizer=optimizer)
    return model

# builds an mlp given the list of number of hidden layer units
def form_mlp(n_hidden, dropout=0, activation='tanh', loss='mean_squared_error', optimizer='rmsprop'):
    from keras.models import Sequential
    from keras.layers import containers
    from keras.layers.core import Dense, AutoEncoder, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    mlp = Sequential()
    for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:]):
        mlp.add(Dense(n_out, input_dim=n_in, activation=activation))
        mlp.add(Dropout(dropout))
    mlp.compile(loss=loss, optimizer=optimizer)
    return mlp

# the function is written with ae_pretrain function in mind
def form_encoder_decoder(encoder, decoder, loss='mean_squared_error', optimizer='rmsprop'):
    from keras.models import Sequential
    from keras.layers import containers
    from keras.layers.core import Dense, AutoEncoder, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    model = Sequential()
    decoder = decoder[::-1]
    for encoder_layer in encoder:
        model.add(encoder_layer)
        model.layers[-1].set_weights(encoder_layer.get_weights())
    for decoder_layer in decoder:
        model.add(decoder_layer)
        model.layers[-1].set_weights(decoder_layer.get_weights())
    model.compile(loss=loss, optimizer=optimizer)
    return model

# adds a softmax layer at the end of a model
def add_softmax(model, n_hidden, nb_classes):
    from keras.models import Sequential
    from keras.layers import containers
    from keras.layers.core import Dense, AutoEncoder, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    model.add(Dense(nb_classes, input_dim=n_hidden[len(n_hidden) - 1], activation='softmax'))
    return model

# trains an mlp model
def mlp_train(x_train, y_train, x_test, y_test, model, batch_size=128, nb_epoch=30, loss='categorical_crossentropy', optimizer='adam', show_accuracy=True, validation_data=None):
    from keras.models import Sequential
    from keras.layers import containers
    from keras.layers.core import Dense, AutoEncoder, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    if validation_data is None:
        validation_data = (x_test, y_test)
    model.compile(loss=loss, optimizer=optimizer)
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=show_accuracy, validation_data=validation_data)
    score = model.evaluate(x_test, y_test, show_accuracy=show_accuracy, verbose=0)
    return (score, model)

# pca classifier
def pca_classifier(n_components, x_train, x_test=None, x_val=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    if x_test is None:
        return (x_train_pca, pca)
    elif x_val is None:
        x_test_pca = pca.transform(x_test)
        return (x_train_pca, x_test_pca, pca)
    else:
        x_test_pca = pca.transform(x_test)
        x_val_pca = pca.transform(x_val)
        return (x_train_pca, x_test_pca, x_val_pca, pca)

# knn classifier
def knn_classifier(n_neighbors, x_train, y_train, x_test, y_test, x_val=None, y_val=None):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(x_train, y_train)
    train_score = neigh.score(x_train, y_train)
    test_score = neigh.score(x_test, y_test)
    if x_val == None:
        return (train_score, test_score)
    else:
        val_score = neigh.score(x_val, y_val)
        return (train_score, test_score, val_score)

# K-means classifier
def k_means_classifier(n_clusters, x_train, init='k-means++', x_val=None):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, init=init)
    kmeans.fit(x_train)
    return (kmeans, kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_)

# label comparision between k-means and human labels
def label_compare(human, kmeans, nb_classes):
    import numpy as np
    comparision_matrix = []
    for label in np.arange(nb_classes):
        tmp = []
        for label1 in np.arange(nb_classes):
            tmp.append(np.size(human[np.logical_and(human == label, kmeans == label1)]))
        comparision_matrix.append(tmp)
    return comparision_matrix

# accuracy of kmeans
def kmeans_accuracy(compare, nb_classes):
    import numpy as np
    flat = np.asarray(compare)
    flat = flat.flatten()
    flat = np.sort(flat)
    flat = flat[::-1]
    tmp = np.float(0)
    for i in np.arange(nb_classes):
        tmp = tmp + np.float(flat[i])
    tmp = tmp / np.sum(flat)
    return (tmp)

# extract data from single label
def label_separation(x, y):
    import numpy as np
    x_sep = []
    for i in np.unique(y):
        x_sep.append(x[y == i])
    x_sep = np.asarray(x_sep)
    return x_sep

# find mean of all values in list of arrays
def mean_of_clusters(x):
    import numpy as np
    mean = []
    for cluster in x:
        mean.append(np.mean(cluster, axis=0))
    mean = np.asarray(mean)
    return mean

# tsne clustering
def tsne_clustering(x, n_components=2, random_state=0):
    import numpy as np
    from sklearn.manifold import TSNE
    x = np.asarray(x)
    model = TSNE(n_components=n_components, random_state=random_state)
    np.set_printoptions(suppress=True)
    return model.fit_transform(x)

# recon error
def recon_error(x, y):
    import numpy as np
    return np.sum(np.sum(np.multiply(x - y, x - y), axis=1) / np.shape(x)[0]) / np.shape(x)[1]

# autoencoder pre training based on functional api
def ae_pretrain_fapi(x_train, n_hidden, greedy=True, batch_size=16, nb_epoch=3, dropout=0, activation='relu', loss='mean_squared_error', optimizer='adam'):
    from keras.models import Model
    from keras.layers import Dense, Input, Dropout
    from keras.layers.core import Dense, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    trained_encoders = []
    trained_decoders = []
    x_tmp = x_train
    x_predictions = []
    x_prediction_scores = []
    loss_history = []
    if greedy:
        for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:]):
            print('Training: Input {} -> Output {}'.format(n_in, n_out))
            input = Input(shape=(n_in,))
            encoder = Dense(n_out,activation=activation)
            if dropout > 0:
                dl = Dropout(dropout)
            decoder = Dense(n_in,activation=activation)
            enc_dec = Model(input=input, output=decoder(encoder(input)))
            enc_dec.compile(loss=loss, optimizer=optimizer)
            history = LossHistory()
            tmp = enc_dec.fit(x_tmp, x_tmp, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[history])
            if dropout > 0 & n_out != n_hidden[len(n_hidden) - 1]:
                decoder.add(Dropout(dropout))
            trained_decoders.append(decoder)
            trained_encoders.append(encoder)
            enc = Model(input=input,output=encoder(input))
            enc.compile(loss=loss,optimizer=optimizer)
            x_tmp = enc.predict(x_tmp)
            x_predictions.append(x_tmp)
            x_prediction_scores.append(tmp)
            loss_history.append(history)
        return (trained_encoders,trained_decoders,loss_history,x_prediction_scores,x_predictions)
    else:
        print('Training the whole autoencoder')
        input = Input(shape=(n_hidden[0],))
        encoder_layers = []
        for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:]):
            encoder = Dense(n_out,activation=activation)
            encoder_layers.append(encoder)
        decoder_layers = []
        for i in range(0,len(n_hidden)-1):
   			decoder = Dense(n_hidden[len(n_hidden)-2-i],activation=activation)
   			decoder_layers.append(decoder)
        state = input
        for enc in encoder_layers:
            state = enc(state)
        state_enc = state
        for dec in decoder_layers:
            state = dec(state)
        enc_dec = Model(input=input,output=state)
        enc_dec.compile(loss=loss, optimizer=optimizer)
        history = LossHistory()
        tmp = enc_dec.fit(x_tmp, x_tmp, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[history])
        enc = Model(input=input,output=state_enc)
        enc.compile(loss=loss,optimizer=optimizer)
        return (encoder_layers, decoder_layers, history, tmp, enc.predict(x_tmp))

# reconstructing encoder mlp from ae_pretrain_fapi output
def enc_mlp_fapi(encoder, nb_hidden, loss='mean_squared_error', optimizer='adam'):
    from keras.models import Model
    from keras.layers import Dense, Input
    from keras.layers.core import Dense, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    input = Input(shape=(nb_hidden[0],))
    state = input
    for enc in encoder:
        state = enc(state)
    enc_mlp = Model(input=input, output = state)
    enc_mlp.compile(loss=loss, optimizer=optimizer)
    return enc_mlp

def enc_dec_mlp_fapi(encoder, decoder, nb_hidden, greedy = True, loss = 'mean_squared_error', optimizer = 'adam'):
    from keras.models import Model
    from keras.layers import Dense, Input
    from keras.layers.core import Dense, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    input = Input(shape=(nb_hidden[0],))
    state = input
    for enc in encoder:
        state = enc(state)
    for i in range(0,len(decoder)):
    	if greedy == True:
            state = decoder[len(decoder)-1-i](state)
        else:
        	state = decoder[i](state)
    enc_dec_mlp = Model(input=input, output = state)
    enc_dec_mlp.compile(loss=loss, optimizer = optimizer)
    return enc_dec_mlp










