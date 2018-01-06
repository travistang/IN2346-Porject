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
    if DEBUG:
        assert K.ndim(M) == 3
        assert K.ndim(k) == 2
        assert K.ndim(b) == 1

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
    if DEBUG:
        assert K.ndim(wc) == 2 == K.ndim(w)
        assert K.ndim(g) == 1
    wcx = K.expand_dims(wc,axis = -1)       # (batch,N,1)
    gx = K.expand_dims(g,axis = -1)         # (batch,1)
    wx = K.expand_dims(w,axis = -1)         # (batch,N,1)
    if DEBUG:
        print('wcx:',wcx)
        print('gx:',gx)
        print('wx:',wx)
    prod = K.batch_dot(wcx,gx)              # (batch,N)
    if DEBUG:
        print('prod',prod)
    prod = prod + K.batch_dot(wx, 1 - gx)   # (batch,N)

    return prod

# eq. 8 of the paper
'''
    wi: weight from interpolation (eq. 7), shape (?,N)
    s:  shifting kernel, shape (?,num_shift)
    n:  N (n_slots)
    num_shift: length of shifting kernel
'''
def _shift(wi,s):
    if DEBUG:
        assert type(n) == type(num_shift) == int
        assert K.ndim(wi) == 2 == K.ndim(s)
    num_shift = s.shape[-1].value

    bound = num_shift // 2
    head,tail = wi[:,:bound],wi[:,-bound:]
    wi = K.concatenate([tail,wi,head])
    wix = K.expand_dims(wi,1)              # (batch,1,N + num_shift)

    st = K.permute_dimensions(s,(1,0))     # (num_shift,batch)
    sx = K.expand_dims(st,1)               # (num_shift,1,batch)
    wix = K.permute_dimensions(wix,(0,2,1)) # (batch,N + num_shift,1)
    res = K.conv1d(wix,sx,data_format = 'channels_last')                 # (batch,N,batch)
    return K.stack([res[_,:,_] for _ in range(res.shape[0].value)])

# eq. 9 of the paper
def _sharpen(ws,g):
    if DEBUG:
        assert K.ndim(ws) == 2
        assert K.ndim(g) == 1

    gx = K.expand_dims(g,-1)                           # (batch,1)
    gx = K.repeat_elements(gx,ws.shape[-1],axis = 1)   # (batch,N)
    pow = K.pow(ws,gx)
    return pow / (K.sum(pow,axis = -1,keepdims = True) + 1e-12) # try not to divide by 0

# functions for getting weights from head
def _get_weight(M,w,k,b,g,s,t):
    wc = _content_addressing(M,k,b)
    wi = _interpolate(wc,w,g)
    ws = _shift(wi,s)
    w_fin  = _sharpen(ws,t)
    return w_fin

if __name__ == '__main__':
    import numpy as np
    # unit testing playground
    # batch = 100
    # n = 10
    # m = 20
    # num_shift = 3

    # M = K.ones((batch,n,m))
    # k = K.ones((batch,m))
    # b = K.ones((batch,))
    # g = K.ones((batch,))
    # w = K.ones((batch,n))
    # s = K.ones((batch,num_shift)) * 0.5
    # t = K.zeros((batch,))
    # ca = _content_addressing(M,k,b)
    # wg = _interpolate(ca,w,g)
    # ws = _shift(wg,s)
    # w_fin = _sharpen(ws,t)
    # with K.get_session().as_default():
    #     print(w_fin.eval())
    a = K.stack([K.variable(np.arange(_,10 + _,1)) for _ in range(5)])
    s = K.stack([K.variable([1. + _,0,0]) for _ in range(5)])
    with K.get_session().as_default():
        print(_shift(a,s).eval())
