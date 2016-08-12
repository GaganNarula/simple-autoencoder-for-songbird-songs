def extract_data(pathh,nfiles="all",Nsamp="all",permuut=1,flattn=1,rsize=1,W=40,H=60):
    #all file names in the provided path
    a = os.listdir(pathh)
    
    #choose a random subset of these
    if nfiles!="all":
        sub = np.randint(0,nfiles)
        a = a[sub] 
               
    x = []
    y = []
    for f in a:
        print('\n loading file \n')
        print(f)
        d = loadmat(f)      
        data = d['DATA']
        meta = d['META']       
        del d
        
        if permuut:
            p = np.random.permutation(len(data))
            data = data[p]
            meta = meta[p]
        
        #numb of samples in this birds data from this day
        print('numb available samples is \n')
        
        print(len(data))
        
        if (Nsamp=="all" or Nsamp>len(data)):
            N = len(data)
        else:
            N = Nsamp
            
        for i in np.arange(N-1):
            d = data[i][0]	
            label = meta[i][0]
            label = label['clustID'].astype('int')
            label = label.flatten()
                
            if rsize:
                #resize data
                d = zoom(d,np.array([float(H)/float(d.shape[0]),float(W)/float(d.shape[1])]),order=2)
                if (d.shape[0]<H or d.shape[1]<W):
                    continue
                    
            if flattn:
                d = d.flatten()
            
            #add data and label to x and y
            x.append(d)
            y.append(label)
            
            
        del data,meta
    return x,y




#function for making a simple autoencoder
def gn_build_ae(n_hidden,dropout=0,activh='relu',activo='tanh',wreg=[0,0],loss='mean_squared_error',optimizer='rmsprop'):
    from keras.models import Sequential    
    from keras.layers.core import Dense, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    from keras.regularizers import l1l2
    
    ae = Sequential()
    
    #add encoder layers
    for i in np.arange(len(n_hidden)-1):
    
        if (wreg[0]>0 or wreg[1]>0):
            if i==0:
                #for first layer input_shape argument is needed
                ae.add(Dense(output_dim=n_hidden[i+1], batch_input_shape=(None,n_hidden[i]),
                                                                          W_regularizer=l1l2(wreg[0],wreg[1])))
            else:
                ae.add(Dense(output_dim=n_hidden[i+1], input_dim=n_hidden[i],W_regularizer=l1l2(wreg[0],wreg[1])))
                
        else:
            if i==0:
                #for first layer input_shape argument is needed
                ae.add(Dense(output_dim=n_hidden[i+1], batch_input_shape=(None,n_hidden[i])))
            else:
                ae.add(Dense(output_dim=n_hidden[i+1], input_dim=n_hidden[i]))
        
        ae.add(Activation(activh))
        if(dropout > 0):
            ae.add(Dropout(dropout))
        
    
    #decoder layers
    n_hidden.reverse()
    
    for i in np.arange(len(n_hidden)-2):
        
        if (wreg[0]>0 or wreg[1]>0):
            ae.add(Dense(output_dim=n_hidden[i+1], input_dim=n_hidden[i],W_regularizer=l1l2(wreg[0],wreg[1])))
        else:                
            ae.add(Dense(output_dim=n_hidden[i+1], input_dim=n_hidden[i]))
        
        ae.add(Activation(activh))
        if(dropout > 0):
            ae.add(Dropout(dropout))
            
    #for the output layer
    if (wreg[0]>0 or wreg[1]>0):
        ae.add(Dense(output_dim=n_hidden[-1],input_dim = n_hidden[-2],W_regularizer=l1l2(wreg[0],wreg[1])))
    else:
        ae.add(Dense(output_dim=n_hidden[-1], input_dim=n_hidden[-2]))
    
    ae.add(Activation(activo))
    if(dropout > 0):
        ae.add(Dropout(dropout))
        
    #compile and print summary
    ae.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
    ae.summary()
    return ae






#function for making a simple autoencoder
def gn_build_ae_softmaxbottleneck(n_hidden,dropout=0,activh='relu',activo='tanh',wreg=[0,0],
                                  loss='mean_squared_error',optimizer='rmsprop'):
    
    from keras.models import Sequential    
    from keras.layers.core import Dense, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    from keras.regularizers import l1l2
    
    ae = Sequential()
    
    #add encoder layers
    for i in np.arange(len(n_hidden)-2):
    
        if (wreg[0]>0 or wreg[1]>0):
            if i==0:
                #for first layer input_shape argument is needed
                ae.add(Dense(output_dim=n_hidden[i+1], batch_input_shape=(None,n_hidden[i]),
                                                                          W_regularizer=l1l2(wreg[0],wreg[1])))
            else:
                ae.add(Dense(output_dim=n_hidden[i+1], input_dim=n_hidden[i],W_regularizer=l1l2(wreg[0],wreg[1])))
                
        else:
            if i==0:
                #for first layer input_shape argument is needed
                ae.add(Dense(output_dim=n_hidden[i+1], batch_input_shape=(None,n_hidden[i])))
            else:
                ae.add(Dense(output_dim=n_hidden[i+1], input_dim=n_hidden[i]))
        
        ae.add(Activation(activh))
        if(dropout > 0):
            ae.add(Dropout(dropout))
    
    
    #for the bottle neck layer
    if (wreg[0]>0 or wreg[1]>0):
        ae.add(Dense(output_dim=n_hidden[-1],input_dim = n_hidden[-2],W_regularizer=l1l2(wreg[0],wreg[1])))
    else:
        ae.add(Dense(output_dim=n_hidden[-1], input_dim=n_hidden[-2]))
    
    ae.add(Activation('softmax'))
    if(dropout > 0):
        ae.add(Dropout(dropout))
    
    
    #decoder layers
    n_hidden.reverse()
    
    for i in np.arange(len(n_hidden)-2):
        
        if (wreg[0]>0 or wreg[1]>0):
            ae.add(Dense(output_dim=n_hidden[i+1], input_dim=n_hidden[i],W_regularizer=l1l2(wreg[0],wreg[1])))
        else:                
            ae.add(Dense(output_dim=n_hidden[i+1], input_dim=n_hidden[i]))
        
        ae.add(Activation(activh))
        if(dropout > 0):
            ae.add(Dropout(dropout))
            
    #for the output layer
    if (wreg[0]>0 or wreg[1]>0):
        ae.add(Dense(output_dim=n_hidden[-1],input_dim = n_hidden[-2],W_regularizer=l1l2(wreg[0],wreg[1])))
    else:
        ae.add(Dense(output_dim=n_hidden[-1], input_dim=n_hidden[-2]))
    
    ae.add(Activation(activo))
    if(dropout > 0):
        ae.add(Dropout(dropout))
        
    #compile and print summary
    ae.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
    ae.summary()
    return ae