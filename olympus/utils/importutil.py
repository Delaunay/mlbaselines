import inspect
import importlib


def find_main_module(function):
    file = inspect.getmodule(function).__file__
    folders = file.split('/')[:-1]

    def _find_top_module():
        for idx, folder in enumerate(folders):
            try:
                importlib.import_module(folder)
                return folder
            except:
                pass

    top_module = _find_top_module()

    if top_module is None:
        return None

    start = file.rfind(top_module)

    import_path = file[start:].split('.')[0].replace('/', '.')
    return import_path


def get_import_path(function):
    module = function
    modules = []

    while not modules or module.__name__ != modules[0]:
        modules.insert(0, module.__name__)
        module = inspect.getmodule(module)

    if modules[0] == "__main__":
        # being main does not mean we are not in a module
        import_path = find_main_module(function)

        if import_path is None:
            raise RuntimeError("Cannot register functions defined in __main__")

        return import_path

    return ".".join(modules[:-1])
