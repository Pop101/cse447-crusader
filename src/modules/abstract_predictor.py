# Model supplied by class

class AbstractPredictor:
    """
    Abstract predictor interface we must extend for this assignment
    """

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        raise NotImplementedError('This model cannot be trained')

    def run_pred(self, data):
        raise NotImplementedError('This model cannot be used for prediction')

    def save(self, work_dir):
        raise NotImplementedError('This model cannot be saved')

    @classmethod
    def load(cls, work_dir):
        raise NotImplementedError('This model cannot be loaded')