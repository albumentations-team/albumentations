import json
import warnings

try:
    import yaml
    yaml_available = True
except ImportError:
    yaml_available = False


from albumentations import __version__


__all__ = ['to_dict', 'from_dict', 'save', 'load']


SERIALIZABLE_REGISTRY = {}


class SerializableMeta(type):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(SerializableMeta, cls).__new__(cls, clsname, bases, attrs)
        SERIALIZABLE_REGISTRY[newclass.get_class_fullname()] = newclass
        return newclass


def initialize_transforms(data):
    name = data['__class_fullname__']
    data = {k: v for k, v in data.items() if k != '__class_fullname__'}
    cls = SERIALIZABLE_REGISTRY[name]
    if 'transforms' in data:
        data['transforms'] = [initialize_transforms(t) for t in data['transforms']]
    return cls(**data)


def to_dict(transforms, on_not_implemented_error='raise'):
    if on_not_implemented_error not in {'raise', 'warn'}:
        raise ValueError(
            "Unknown on_not_implemented_error value: {}. Supported values are: 'raise' and 'warn'".format(
                on_not_implemented_error
            )
        )
    try:
        dumped_transforms = transforms.dump()
    except NotImplementedError as e:
        if on_not_implemented_error == 'raise':
            raise e
        else:
            dumped_transforms = {}
            warnings.warn(
                "Got NotImplementedError while trying to serialize {obj}. Object arguments are not preserved. "
                "Implement either '{cls_name}.get_transform_init_args_names' or '{cls_name}.get_transform_init_args' "
                "method to make the transforms serializable".format(
                    obj=transforms,
                    cls_name=transforms.__class__.__name__,
                )
            )
    return {
        '__version__': __version__,
        'transforms': dumped_transforms,
    }


def from_dict(transforms_dict):
    transforms = transforms_dict['transforms']
    name = transforms['__class_fullname__']
    data = {k: v for k, v in transforms.items() if k != '__class_fullname__'}
    cls = SERIALIZABLE_REGISTRY[name]
    if 'transforms' in data:
        data['transforms'] = [initialize_transforms(t) for t in data['transforms']]
    return cls(**data)


def check_data_format(data_format):
    if data_format not in {'json', 'yaml'}:
        raise ValueError(
            "Unknown data_format {}. Supported formats are: 'json' and 'yaml'".format(data_format)
        )


def save(transforms, filepath, data_format='json', on_not_implemented_error='raise'):
    check_data_format(data_format)
    transforms_dict = to_dict(transforms, on_not_implemented_error=on_not_implemented_error)
    dump_fn = json.dump if data_format == 'json' else yaml.dump
    with open(filepath, 'w') as f:
        dump_fn(transforms_dict, f)


def load(filepath, data_format='json'):
    check_data_format(data_format)
    load_fn = json.load if data_format == 'json' else yaml.load
    with open(filepath) as f:
        transforms_dict = load_fn(f)
    return from_dict(transforms_dict)
