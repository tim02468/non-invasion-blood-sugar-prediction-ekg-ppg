import numpy as np
import random
# train generator
def train_generator(train_file,batch_size,multi_CNN,use_meta,X5,X10,X25,X50,meta,train_lab):
    if multi_CNN:
        train_list = list(np.copy(train_file))
        while 1:
            X5_ = []
            X10_ = []
            X25_ = []
            X50_ = []
            M_ = []
            Y_ = []
            for _ in range(batch_size):
                # randomly select a record
                if len(train_list) == 0:
                    train_list = list(np.copy(train_file))
                    random.shuffle(train_list)
                train_f = train_list.pop()
                x = X5[train_f] 
                x10 = X10[train_f]
                x25 = X25[train_f]
                x50 = X50[train_f]
                x2 = meta[train_f]
                # random select a subrecord from 0~4
                j = random.randint(0, 4)
                x1 = x[j]
                x101 = x10[j]
                x251 = x25[j]
                x501 = x50[j]
                # append on list
                X5_.append(x1)
                X10_.append(x101)
                X25_.append(x251)
                X50_.append(x501)
                M_.append(x2)
                Y_.append(np.array(train_lab[train_f])[0])

            # stack together batch size numbers of training data    
            X5_ = np.asarray(X5_)
            X10_ = np.asarray(X10_)
            X25_ = np.asarray(X25_)
            X50_ = np.asarray(X50_)
            M_ = np.asarray(M_)
            if use_meta:
                COM_ = [X5_,X10_,X25_,X50_,M_]
            else:
                COM_ = [X5_,X10_,X25_,X50_]
            Y_ = np.asarray(Y_)
            yield COM_, Y_
    else:    
        while 1:
            X_ = []
            M_ = []
            Y_ = []
            for _ in range(batch_size):
                if len(train_list) == 0:
                    train_list = list(np.copy(train_file))
                    random.shuffle(train_list)
                train_f = train_list.pop()       # random select a sample
                signal_id = random.randint(0, 4) # random select a signal
                x = X[train_f][signal_id]
                X_.append(x)
                M_.append(meta[train_f])
                Y_.append(train_lab[train_f])
            X_ = np.asarray(X_)
            M_ = np.asarray(M_)
            Y_ = np.asarray(Y_)
            if use_meta:
                yield [X_, M_], Y_
            else:
                yield [X_], Y_
                
def train_generator_single(train_file, batch_size, X, meta, train_lab, use_meta):
    train_list = list(np.copy(train_file))
    while 1:
        X_ = []
        M_ = []
        Y_ = []
        for _ in range(batch_size):
            if len(train_list) == 0:
                train_list = list(np.copy(train_file))
                random.shuffle(train_list)
            train_f = train_list.pop()       # random select a sample
            signal_id = random.randint(0, 4) # random select a signal
            x = X[train_f][signal_id]
            X_.append(x)
            M_.append(meta[train_f])
            Y_.append(train_lab[train_f][0])
        X_ = np.asarray(X_)
        M_ = np.asarray(M_)
        Y_ = np.asarray(Y_)
        if use_meta:
            yield [X_, M_], Y_
        else:
            yield [X_], Y_
                
# validation set generation
def validation_generation(test_file,test_lab,multi_CNN,use_meta,X5,X10,X25,X50,meta):
    if multi_CNN:
        X5_ = []
        X10_ = []
        X25_ = []
        X50_ = []
        M_ = []
        Y_ = []
        for i in range(len(test_file)):
            x = X5[test_file[i]] 
            x10 = X10[test_file[i]]
            x25 = X25[test_file[i]]
            x50 = X50[test_file[i]]
            x2 = meta[test_file[i]]
            for j in range(5):
                X5_.append(x[j])
                X10_.append(x10[j])
                X25_.append(x25[j])
                X50_.append(x50[j])
                M_.append(x2)
                # Y_.append(test_lab[test_file[i]][0])
                Y_.append(np.array(test_lab[test_file[i]])[0])
        X5_ = np.asarray(X5_)
        X10_ = np.asarray(X10_)
        X25_ = np.asarray(X25_)
        X50_ = np.asarray(X50_)
        M_ = np.asarray(M_)
    #     COM_ = [X5_,X10_,X25_,X50_,M_]
        Y_ = np.asarray(Y_)
        print("# records = ", len(Y_))
        if use_meta:
            return X5_, X10_,X25_,X50_,M_, Y_ 
        else:
            return X5_, X10_,X25_,X50_, Y_
    else:
        X_ = []
        M_ = []
        Y_ = []
        for i in range(len(test_file)):
            id_ = test_file[i]
            for j in range(5):
                signal_id = j # random select a signal
                x = X[id_][signal_id]
                X_.append(x)
                M_.append(meta[id_])
                Y_.append(test_lab[id_])
        X_ = np.asarray(X_)
        M_ = np.asarray(M_)
        Y_ = np.asarray(Y_)
        print("# records = ", len(Y_))
        if use_meta:
            return X_, M_, Y_
        else:
            return X_, Y_


def validation_generation_single(test_file,test_lab,X,meta,use_meta):
    X_ = []
    M_ = []
    Y_ = []
    for i in range(len(test_file)):
        id_ = test_file[i]
        for j in range(5):
            signal_id = j # random select a signal
            x = X[id_][signal_id]
            X_.append(x)
            M_.append(meta[id_])
            Y_.append(test_lab[id_][0])
    X_ = np.asarray(X_)
    M_ = np.asarray(M_)
    Y_ = np.asarray(Y_)
    print("# records = ", len(Y_))
    if use_meta:
        return X_, M_, Y_
    else:
        return X_, Y_

# one train set
def one_train_set(multi_CNN,use_meta,X5,X10,X25,X50,meta,train_lab,train_file,size = 1610):
    if multi_CNN:
        X5_ = []
        X10_ = []
        X25_ = []
        X50_ = []
        M_ = []
        Y_ = []
        count = 0
        while count < size:
            count = count + 1
            # select a random sample
            i = random.randint(0,len(train_file)-1)
            x = X5[train_file[i]] 
            x10 = X10[train_file[i]]
            x25 = X25[train_file[i]]
            x50 = X50[train_file[i]]
            x2 = meta[train_file[i]]
            j = random.randint(0, 9)
            X5_.append(x[j])
            X10_.append(x10[j])
            X25_.append(x25[j])
            X50_.append(x50[j])
            M_.append(x2)
            Y_.append(train_lab[train_file[i]][0])
        X5_ = np.asarray(X5_)
        X10_ = np.asarray(X10_)
        X25_ = np.asarray(X25_)
        X50_ = np.asarray(X50_)
    #     COM_ = [X5_,X10_,X25_,X50_,M_]
        M_ = np.asarray(M_)
        Y_ = np.asarray(Y_) 
        if use_meta:
            return X5_, X10_,X25_,X50_,M_, Y_
        else:
            return X5_, X10_,X25_,X50_, Y_
    else:
        X_ = []
        M_ = []
        Y_ = []
        count = 0
        while count < size:
            i = random.randint(0,len(train_file)-1)
            id_ = train_file[i]
            signal_id = random.randint(0,4) # random select a signal
            x = X[id_][signal_id]
            X_.append(x)
            M_.append(meta[id_])
            Y_.append(train_lab[id_])
            count += 1
        X_ = np.asarray(X_)
        M_ = np.asarray(M_)
        Y_ = np.asarray(Y_)
        if use_meta:
            return X_, M_, Y_
        else:
            return X_,Y_
        
def one_train_set_single(X,meta,train_lab,train_file,use_meta,size = 2000):
    X_ = []
    M_ = []
    Y_ = []
    count = 0
    while count < size:
        i = random.randint(0,len(train_file)-1)
        id_ = train_file[i]
        signal_id = random.randint(0,4) # random select a signal
        x = X[id_][signal_id]
        X_.append(x)
        M_.append(meta[id_])
        Y_.append(train_lab[id_][0])
        count += 1
    X_ = np.asarray(X_)
    M_ = np.asarray(M_)
    Y_ = np.asarray(Y_)
    if use_meta:
        return X_, M_, Y_
    else:
        return X_,Y_

        
        
        
        
        
        
        
        
        














