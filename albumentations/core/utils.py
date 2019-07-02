from __future__ import absolute_import

from ..core.six import string_types


def format_args(args_dict):
    formatted_args = []
    for k, v in args_dict.items():
        if isinstance(v, string_types):
            v = "'{}'".format(v)
        formatted_args.append('{}={}'.format(k, v))
    return ', '.join(formatted_args)
