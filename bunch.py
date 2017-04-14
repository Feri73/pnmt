class Bunch(dict):
    _setattr__ = dict.__setitem__

    def __getattr__(self, item):
        if dict.__contains__(self, item):
            return dict.__getitem__(self, item)
        else:
            return None