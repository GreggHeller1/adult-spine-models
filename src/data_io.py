# +
import os
import scipy.io as sio
import h5py


def readfile(path):
    print(f'Reading file at {path}')
    with open(path, 'r') as f:
        lines = f.read()
        print(lines)


def loadmat(path):
    try:
        f = _loadmat(path)
    except NotImplementedError as E:
        f = h5py.File(path,'r')

    if 'soma_cell' in f.keys():
        return f['soma_cell']
    elif 'dend_cell' in f.keys():
        pass
        #return f['dend_cell']

    return f


def _check_keys( dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def _loadmat(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)