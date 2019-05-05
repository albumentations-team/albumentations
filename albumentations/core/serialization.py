import json

try:
    import yaml
    yaml_available = True
except ImportError:
    yaml_available = False


from albumentations import __version__


__all__ = ['dump', 'load']


SERIALIZABLE_REGISTRY = {}


class SerializableMeta(type):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(SerializableMeta, cls).__new__(cls, clsname, bases, attrs)
        SERIALIZABLE_REGISTRY[newclass.get_name()] = newclass
        return newclass


def dump(obj, data_format='json'):
    dumped_obj = {
        '__version__': __version__,
        'state': obj.get_state(),
    }
    if data_format == 'json':
        return json.dumps(dumped_obj)
    elif data_format == 'yaml':
        if not yaml_available:
            raise RuntimeError(
                "Can't dump the object to YAML because PyYAML library is not installed. "
                "You can install the library by running 'pip install pyyaml'"
            )
        return yaml.dumps(dumped_obj)
    elif data_format == 'dict':
        return dumped_obj
    else:
        raise ValueError(
            "Unknown data_format {}. Supported formats are: 'json', 'yaml' and 'dict'".format(data_format)
        )


def initialize_transforms(data):
    name = data['__name__']
    data = {k: v for k, v in data.items() if k != '__name__'}
    cls = SERIALIZABLE_REGISTRY[name]
    if 'transforms' in data:
        data['transforms'] = [initialize_transforms(t) for t in data['transforms']]
    return cls(**data)


def load(obj, data_format='json'):
    if data_format == 'json':
        loaded_obj = json.loads(obj)
    elif data_format == 'yaml':
        if not yaml_available:
            raise RuntimeError(
                "Can't dump the object to YAML because PyYAML library is not installed. "
                "You can install the library by running 'pip install pyyaml'"
            )
        loaded_obj = yaml.loads(obj)
    elif data_format == 'dict':
        loaded_obj = obj
    else:
        raise ValueError(
            "Unknown data_format {}. Supported formats are: 'json', 'yaml' and 'dict'".format(data_format)
        )
    return initialize_transforms(loaded_obj['state'])
