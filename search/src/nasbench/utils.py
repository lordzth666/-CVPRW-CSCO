from src.nasbench.dataset import Dataset as NASDataset
import pickle

def concat_dataset(list_sets):
	assert len(list_sets) >= 1
	with open(list_sets[0], 'rb') as fp:
		dataset = pickle.load(fp)
	assert isinstance(dataset, NASDataset), "Datasets %s must be a NASDataset Instance." %list_sets[0]
	for i in range(1, len(list_sets)):
		with open(list_sets[i], 'rb') as fp:
			dataset_ = pickle.load(fp)
		assert isinstance(dataset_, NASDataset), "Datasets %s must be a NASDataset Instance." % list_sets[i]
		print("Dataset %s has %d items!" %(list_sets[i], len(dataset_.accs)))
		dataset.concat(dataset_)
	return dataset