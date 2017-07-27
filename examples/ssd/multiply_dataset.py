import chainer


class MultiplyDataset(chainer.dataset.DatasetMixin):

    def __init__(self, datasets, multipliers):
        self._datasets = datasets
        self._multipliers = multipliers

        self.lengths = []
        for i in range(len(datasets)):
            self.lengths.append(len(datasets[i]) * multipliers[i])

    def __len__(self):
        return sum(self.lengths)

    def get_example(self, i):
        if i < 0:
            raise IndexError
        for length, mult, dataset in zip(
                self.lengths, self._multipliers,  self._datasets):
            if i < length:
                return dataset[i / mult]
            i -= length
        raise IndexError


if __name__ == '__main__':
    from original_detection_dataset import OriginalDetectionDataset
    import yaml

    label_names_fn = '/home/leus/playground/image-labelling-tool/examples/ssd/train/at_home_nagoya_label_names.yml'
    with open(label_names_fn, 'r') as f:
        label_names = tuple(yaml.load(f))

    dataset = OriginalDetectionDataset(
        '/home/leus/data/at_home/', label_names)

    dataset = MultiplyDataset([dataset], [2])
