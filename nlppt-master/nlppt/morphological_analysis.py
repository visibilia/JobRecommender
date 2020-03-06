from .unitex import Unitex
unitex = Unitex()


def lemma(token, pos='N'):
    return unitex.lemma(token, pos)


def morf(token, pos=None):
    return list(set(unitex.morf(token, pos)))
