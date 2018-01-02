import keras.backend as K
'''
    File containing the necessary weights for getting weight vector given k,b,g,s,t from head
    M: (batch,N,M)
    k: (batch,M)
    b: (batch,)
    g: (batch,)
    s: (batch,num_shift)
'''
DEBUG = True
def _content_addressing(M,k,b):
    # cosine similarity
    nM = K.l2_normalize(M,axis = 1)     # (batch,N,M)
    nk = K.l2_normalize(k,axis = -1)    # (batch,M)
    nkx = K.expand_dims(nk,axis = -1)   # (batch,M,1)
    sim = K.batch_dot(nM,nkx)           # (batch,N,1)

    # multiply by b
    bx = K.expand_dims(b,axis = -1)     # (batch,1)

    mul = K.batch_dot(sim,bx)           # (batch,N)

    return K.softmax(mul)               # (batch,N)


# eq. 7 of the paper
def _interpolate(wc,w,g):
    wcx = K.expand_dims(wc,axis = -1)
    gx = K.expand_dims(g,axis = -1)
    wx = K.expand_dims(w,axis = -1)

    prod = K.batch_dot(wcx,gx)

    prod = prod + K.batch_dot(wx, 1 - gx)

    return prod

# eq. 8 of the paper
def _shift(wi,s):
    # TODO: what now?!
    pass

# eq. 9 of the paper
def _sharpen(ws,g):
    pow = K.pow(ws,g)
    return pow / (K.sum(pow) + 1e-12) # try not to divide by 0

# functions for getting weights from head
def _get_weight(M,w,k,b,g,s,t):
    wc = _content_addressing(M,k,b)
    wi = _interpolate(wc,w,g)
    ws = _shift(wi,s)
    w_fin  = _sharpen(ws,t)
    return w_fin

if __name__ == '__main__':
    # unit testing playground
    batch = 100
    n = 10
    m = 20
    num_shift = 3

    M = K.ones((batch,n,m))
    k = K.ones((batch,m))
    b = K.ones((batch,))
    g = K.ones((batch,))
    w = K.ones((batch,n))

    ca = _content_addressing(M,k,b)
    wg = _interpolate(ca,w,g)
    print(wg)
