class LoadingException(Exception):
    pass


class BaseSubjectLoader(object):

    def __init__(self):
        pass

    def get_nr_of_sets(self):
        raise NotImplementedError("This method is not implemented in base class.")

    def get_trial_by_set_nr(self, trial_nr: int):
        raise NotImplementedError("This method is not implemented in base class.")
