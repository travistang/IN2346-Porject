import keras.backend as K
'''
    File containing the necessary weights for getting weight vector given k,b,g,s,t from head
    M: (batch,N,M)
    k: (batch,M)
    b: (batch,)
    g: (batch,)
    s: (batch,num_shift)
'''
DEBUG = False
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
    wcx = K.expand_dims(wc,axis = -1)       # (batch,N,1)
    gx = K.expand_dims(g,axis = -1)         # (batch,1)
    wx = K.expand_dims(w,axis = -1)         # (batch,N,1)

    prod = K.batch_dot(wcx,gx)              # (batch,N)

    prod = prod + K.batch_dot(wx, 1 - gx)   # (batch,N)

    return prod

# eq. 8 of the paper
def _shift(wi,s,n,num_shift):
    wix = K.expand_dims(wi,axis = -1)       # (batch,N,1)
    wix = K.expand_dims(wix,axis = -1)      # (batch,N,1,1)
    if DEBUG:
        print('wix',wix)
    res = [K.zeros((wix.shape[0],1)) for i in range(n)]   # N * (batch,1)
    st = [K.expand_dims(s[:,i],-1) for i in range(num_shift)] # num_shift * (batch,1)
    if DEBUG:
        print(res,st)
    # TODO: can this be any faster?
    # (faithfully) following the eq. 8 in section 3.3.2
    for i in range(n):
        for j in range(n):
            w = wix[:,j,:,:]                                        # (batch,1,1)
            si = st[(i - j) % num_shift]                            # (batch,1)
            res[i] += K.batch_dot(w,si)  # (bat)
    if DEBUG:
        print('res',res)
    res = K.stack(res,axis = 1)                    # (batch,N,1)
    res = res[:,:,0]                               # (batch,N)
    if DEBUG:
        print('res',res)
    return res

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
    s = K.ones((batch,num_shift)) * 0.5

    ca = _content_addressing(M,k,b)
    wg = _interpolate(ca,w,g)
    ws = _shift(wg,s,n,num_shift)

    with K.get_session().as_default():
        print(ws.eval())
