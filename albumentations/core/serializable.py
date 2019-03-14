import inspect
import json


class Serializable(object):
    def __init__(self):
        frame = inspect.currentframe()
        while frame.f_back.f_locals.get('self', object).__class__ == frame.f_locals['self'].__class__:
            frame = frame.f_back
        args_count = frame.f_code.co_argcount
        varnames = frame.f_code.co_varnames[1:args_count]
        params = frame.f_locals
        self.class_name = str(params['self'].__class__.__name__)
        self.init_params = {k: v for k, v in params.items() if k in varnames}

    def get_init_params(self):
        raise NotImplementedError

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.get_init_params(), f, indent=4)
