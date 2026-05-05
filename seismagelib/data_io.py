import yaml
import os
import sys
import numpy as np
from seismagelib.data_structures import NDArrayStructIntf
# import _seismagelib_external

import importlib.util


def load_model(path: str) -> dict:
    modelspec = load_yaml(path)
    dirpath = os.path.dirname(path)

    params = modelspec.get('params', {})
    shape = modelspec['shape']
    
    for k, v in params.items():

        modelspec['params'][k] = np.fromfile(os.path.join(dirpath, v), dtype=np.float32).reshape(shape)
        if 'params_ranges' in modelspec:
            ranges = modelspec['params_ranges'][k]
            modelspec['params'][k] = np.clip(modelspec['params'][k], ranges[0], ranges[1])

    return modelspec


def save_model(modeldict: dict, name: str, dirpath: str):
    os.makedirs(dirpath, exist_ok=True)

    params: dict = modeldict['params']
    for pname, p in params.items():
        pfname = f'{name}-{pname}.bin'
        modeldict['params'][pname] = f'./{pfname}'
        ppath = os.path.join(dirpath, pfname)
        p.tofile(ppath)

    with open(os.path.join(dirpath, f'{name}.yml'), 'w+') as fout:
        yaml.dump(modeldict, fout)


def load_seis_data(path):
    dmeta = load_yaml(path)
    ddir = os.path.dirname(path)
    ns = dmeta['ns']
    nt = dmeta['nt']
    d = np.fromfile(os.path.join(ddir, dmeta['data']), dtype=np.float32)
    nrec = d.shape[0]//(ns * nt)
    dmeta['data'] = d.reshape(
        (ns, nrec, nt)
    )

    if 'geometry' in dmeta:
        gpath = os.path.join(ddir, dmeta['geometry'])
        dmeta['geometry'] = load_geometry(gpath)

    return dmeta


def load_geometry(path):
    geospec = load_yaml(path)
    ns = geospec['ns']
    nr = geospec['nr']

    src = geospec['src']
    for k, v in src.items():
        if type(v) in [int, float]:
            geospec['src'][k] = v * np.ones(ns, dtype=np.float32)
        elif type(v) == list:
            geospec['src'][k] = np.linspace(v[0], v[1], num = ns, dtype=np.float32)


    rec = geospec['rec']
    for k, v in rec.items():
        if type(v) in [int, float]:
            geospec['rec'][k] = v * np.ones(nr, dtype=np.float32)
        elif type(v) == list:
            geospec['rec'][k] = np.linspace(v[0], v[1], num = nr, dtype=np.float32)


    if geospec.get('source', 'wav://Ricker').startswith('wav://'):
        geospec['source'] = geospec.get('source', 'wav://Ricker')[len('wav://'):]
    else:
        srcpath = os.path.join(os.path.dirname(path), geospec['source'])
        geospec['source'] = np.fromfile(srcpath, dtype=np.float32)
    
    return geospec


def load_yaml(path):
    with open(path, 'r') as fin:
        data = yaml.load(fin, yaml.Loader)
    return data


def make_ndarraystruct(m):
    param_list = []
    size = 0
    fields = {}
    psize = 1
    
    for s in m['shape']:
        psize *= s
    
    for pname, pdata in m['params'].items():
        param_list.append(pdata.reshape(-1))
        fields[pname] = {
            'offset': size,
            'size': len(pdata.reshape(-1)),
            'shape': pdata.shape
        }
        size += psize
    
    return NDArrayStructIntf(np.array(param_list).reshape(-1), fields)


_external_module_id = 0
def import_module_file(path):
    global _external_module_id
    module_name = f'_external_module{_external_module_id}'
    _external_module_id += 1

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module