# residual block
def ResidualBlock(filters,kernel_size,strides,pool_size,inputs):
    if activation == 'selu':
        new_input = MaxPool1D(pool_size=pool_size, padding = 'same', strides = strides)(inputs)
        new_inp_2 = Activation(activation='selu')(inputs)
        new_inp_2 = Dropout(dropout_rate)(new_inp_2)
        new_inp_2 = Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',kernel_initializer = kernel_initializer, kernel_regularizer=regularizers.l2(l_2))(new_inp_2)
        new_inp_2 = Activation(activation='selu')(new_inp_2)
        new_inp_2 = Dropout(dropout_rate)(new_inp_2)
        new_inp_2 = Conv1D(filters = filters, kernel_size = kernel_size, strides = 1, padding = 'same',kernel_initializer = kernel_initializer, kernel_regularizer=regularizers.l2(l_2))(new_inp_2)
        new_inp_2 = Add()([new_inp_2, new_input])
        return new_inp_2
    else:
        new_input = MaxPool1D(pool_size=pool_size, padding = 'same', strides = strides)(inputs)
        new_inp_2 = BatchNormalization()(inputs)
        new_inp_2 = Activation(activation=activation)(new_inp_2)
        new_inp_2 = Dropout(dropout_rate)(new_inp_2)
        new_inp_2 = Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',kernel_initializer = kernel_initializer, kernel_regularizer=regularizers.l2(l_2))(new_inp_2)
        new_inp_2 = BatchNormalization()(new_inp_2)
        new_inp_2 = Activation(activation=activation)(new_inp_2)
        new_inp_2 = Dropout(dropout_rate)(new_inp_2)
        new_inp_2 = Conv1D(filters = filters, kernel_size = kernel_size, strides = 1, padding = 'same',kernel_initializer = kernel_initializer, kernel_regularizer=regularizers.l2(l_2))(new_inp_2)
        new_inp_2 = Add()([new_inp_2, new_input])
        return new_inp_2
    
    
