class LoadingException(Exception):
    pass


class BaseSubjectLoader(object):

    def __init__(self, subject_name: str):
        self._subject_name = subject_name

    def get_nr_of_sets(self):
        raise NotImplementedError("This method is not implemented in base class.")

    def get_trial_by_set_nr(self, trial_nr: int):
        raise NotImplementedError("This method is not implemented in base class.")
