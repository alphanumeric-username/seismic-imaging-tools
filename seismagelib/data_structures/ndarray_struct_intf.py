import numpy as np


class NDArrayStructIntf():
    def __init__(self, array: np.ndarray, fields: dict):
        self.array = array
        self.fields = fields
    
    def _get_field_data(self, field_name: str):
        if field_name in self.fields:
            offset = self.fields[field_name].get('offset', None)
            size   = self.fields[field_name].get('size', None)
            shape  = self.fields[field_name].get('shape', -1)
            
            if offset is None:
                raise ValueError(f'No offset value was found for field `{field_name}`.')
            elif size is None:
                raise ValueError(f'No size value was found for field `{field_name}`.')
            
            return offset, size, shape
        raise ValueError(f'Attempted to access non-existent field `{field_name}` in a structured array.')

    def __getitem__(self, field_name: str):
        offset, size, shape = self._get_field_data(field_name)
        v = self.array[offset:offset + size].reshape(shape)
        if size == 1:
            return v[0]
        return v
    
    def __setitem__(self, field_name: str, value):
        offset, size, _ = self._get_field_data(field_name)
        current = self.__getitem__(field_name)
        if hasattr(value, 'shape'):
            if value.shape == current.shape:
                self.array[offset:offset + size] = value.reshape(-1)
        else:
            self.array[offset] = value

    def to_dict(self):
        d = {}
        for fname, _ in self.fields.items():
            d[fname] = self[fname]
        return d

