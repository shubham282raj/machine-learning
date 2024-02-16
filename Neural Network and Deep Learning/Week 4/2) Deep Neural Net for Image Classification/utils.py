# %%
#loading required libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt

# %%
#some magic I don't understand
# %matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2

#except for this line
np.random.seed(1)

# %% [markdown]
# # Initializing Parameters

# %%
def initializeParams(nx, nh, ny):
    """
    Returns w1, b1, w2, b2 in a dictionary "params"
    """
    np.random.seed(1)

    w1 = np.random.rand(nh, nx) * 0.01
    b1 = np.zeros((nh, 1))
    w2 = np.random.rand(ny, nh) * 0.01
    b2 = np.zeros((ny, 1))

    params = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }

    return params

# %%
def initializeParamsDeep(layerDims):
    assert (layerDims.shape == (len(layerDims),))

    np.random.seed(1)
    params = {}

    for i in range(1, len(layerDims)):
        w = np.random.rand(layerDims[i], layerDims[i-1])
        b = np.zeros((layerDims[i], 1))
        params[f"w{i}"] = w
        params[f"b{i}"] = b
    
        assert(params[f"w{i}"].shape == (layerDims[i], layerDims[i-1]))
        assert(params[f"b{i}"].shape == (layerDims[i], 1))
    
    return params

# %% [markdown]
# # Mathematical Funtions

# %%
def sigmoid(z):
    a = 1./(1 + np.exp(-z))
    cache = z
    return a, cache

# %%
def sigmoidBackward(da, cache):
    z = cache
    
    s = 1./(1+np.exp(-z))
    dz = da * s * (1-s)
    
    assert (dz.shape == z.shape)
    
    return dz

# %%
def relu(z):
    a = np.maximum(0,z)
    assert(a.shape == z.shape)
    cache = z
    return a, cache

# %%
def reluBackward(da, cache):
    z = cache
    dz = np.array(da, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dz[z <= 0] = 0
    
    assert (dz.shape == z.shape)
    
    return dz

# %% [markdown]
# # Forward Propagation

# %%
#linear forward
def linearForward(a, w, b):
    z = w.dot(a) + b
    cache = (a, w, b)

    assert(z.shape == (w.shape[0], a.shape[1]))

    return z, cache

# %%
def linearActivationForward(aPrev, w, b, activation):
    z, linearCache = linearForward(aPrev, w, b)
    if activation == "sigmoid":
        a, activationCache = sigmoid(z)
    elif activation == "relu":
        a, activationCache = relu(z)
    
    assert (a.shape == (w.shape[0], aPrev.shape[1]))
    cache = (linearCache, activationCache)

    return a, cache


# %%
def LModelForward(x, params):
    """
        return AL; last post activation value
        cache: list of cache containing every cache of linearActivationForward(); L-1 caches from 0 to L-2
    """

    caches = []
    a = x
    L = len(params) // 2    #number of layers in the deep neural net
    for i in range(1,L):
        aPrev = a
        a, cache = linearActivationForward(aPrev, params[f"w{i}"], params[f"b{i}"], "relu")
        caches.append(cache)

    al , cache = linearActivationForward(a, params[f"w{L}"], params[f"b{L}"], "sigmoid")
    cache.append(cache)

    assert (al.shape == (1, x.shape[1]))

    return al, caches


# %% [markdown]
# # Compute Loss Function

# %%
def computeCost(al, y):
    m = y.shape[1]
    cost = -1./m * ( y.dot(np.log(al).T) + (1-y).dot(np.log(1-al).T) )
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

# %% [markdown]
# # BackWard Propagation
# ### Linear Backward

# %%
def linearBackward(dz, cache):
    # cache = ( aPrev, w, b ) coming from the forward propagation
    aPrev, w, b = cache
    m = aPrev.shape[1]

    dw = 1./m * dz.dot(aPrev.T)
    db = 1./m * np.sum(dz, axis = 1, keepdims= True)
    daPrev = w.T.dot(dz)

    assert (daPrev.shape == aPrev.shape)
    assert (dw.shape == w.shape)
    assert (db.shape == b.shape)

    return daPrev, dw, db

# %%
def linearActivationBackward(da, cache, activation):
    linearCache, activationCache = cache
    
    if activation == "sigmoid":
        dz = sigmoidBackward(da, activationCache)
    elif activation == "relu":
        dz = reluBackward(da, activationCache)
    
    daPrev , dw, db = linearBackward(dz, linearCache)

    return daPrev, dw, db

# %%
def LModelBackward(al, y, cache):
    grads = {}
    l = len(cache)
    m = al.shape[1]
    y = y.reshape(al.shape)

    dal = -( np.divide(y, al) - np.divide(1-y, 1-al) )

    currentCache = cache[l-1]
    grads[f"da{l-1}"], grads[f"dw{l}"], grads[f"db{l}"] = linearActivationBackward(dal, currentCache, "sigmoid")

    for i in reversed(range(l-1)):
        currentCache = cache[i]
        grads[f"da{i}"], grads[f"dw{i+1}"], grads[f"db{i+1}"] = linearActivationBackward(grads[f"da{i+1}"], currentCache, "relu")

    return grads

# %%
def updateParams(params, grads, learningRate):
    l = len(params) // 2
    for i in range(l):
        params[f"w{i+1}"] -= learningRate * grads[f"dw{i+1}"]
        params[f"b{i+1}"] -= learningRate * grads[f"db{i+1}"]

    return params


