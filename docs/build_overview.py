#!/usr/bin/env python
# -*- coding: utf-8 -*-


import inspect
import importlib
import os

here = os.path.dirname(os.path.realpath(__file__))

def build_autosummaries(module_name, class_overview = True, functions = True, classes = True):
    """

    Parameters
    ----------
    module_name
        Name of the module.
    class_overview
        If true, builds a table with every class.
    functions
        If true, builds a table with every function.
    classes
        If true, builds a table for every single class.

    Returns
        string
            A string with .rst.
    -------

    """

    out = []
    module = importlib.import_module(module_name)
    print(module)

    if class_overview:
        classes_in_module = inspect.getmembers(module, inspect.isclass)

        out.append('List of all public classes')
        out.append('------------------------------------------')
        out.append('\n.. autosummary::\n')

        for name, cls in classes_in_module:
            out.append('    ~' + cls.__module__ + '.' + name)
        out.append('')

    if functions:
        functions_in_module = inspect.getmembers(module, inspect.isfunction)

        out.append('List of all public functions')
        out.append('------------------------------------------')
        out.append('\n.. autosummary::\n')
        for name, function in functions_in_module:
            out.append('    ~' + function.__module__ + '.' + name)
        out.append('')

    if classes:
        classes_in_module = inspect.getmembers(module, inspect.isclass)

        out.append('All public classes (with methods)')
        out.append('------------------------------------------')
        for cls_name, cls in sorted(classes_in_module, reverse = True):

            # Find inheritance information
            bases = [basecls for basecls in cls.__bases__ if basecls != object]
            inherits = ', '.join([':class:`~{}`'.format(
                basecls.__module__ + '.' + basecls.__name__) for basecls in
                bases])

            header_clsname = cls.__module__ + '.' + cls_name
            header = 'Methods for class :class:`~{}` '.format(header_clsname)
            # Add inheritance if it exists
            if len(inherits) > 0:
                header += '(inherits from: {} )'.format(inherits)
            out.append(header)
            out.append('~'*(25 +len(header)))
            out.append('\n.. autosummary::\n')
            members = inspect.getmembers(cls)

            out.append('    ~' + header_clsname)
            for member_name, member in members:

                # If it's a function (normal method) or @classmethod
                if (inspect.isfunction(member) or
                        (inspect.ismethod(member) and member.__self__ is cls)):
                    print(member_name, member, type(member))

                    # If inherited, continue
                    if member_name not in list(cls.__dict__.keys()):
                        continue

                    if member_name[0] == '_' and member_name[1] != '_':
                        # Private method, continue
                        continue

                    out.append('    ~' + cls.__module__ + '.' + cls_name + '.' + member_name)

            out.append('  ')
        out.append('')

    return '\n'.join(out)


if True:# __name__ == '__main__':
    out = build_autosummaries('abelian')
    print(out)

    filename = 'autodoc_overview.rst'
    with open(os.path.join(here, filename), 'w', encoding = 'utf-8') as file:
        file.write(out)