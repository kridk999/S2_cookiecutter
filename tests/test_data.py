from torch.utils.data import Dataset
import torch
from mnist_cookie.data import corrupt_mnist


def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000,  "Dataset did not have the correct number of samples" 
    assert len(test) == 5000, "Dataset did not have the correct number of samples"
    for data in train:
        assert data[0].shape == (1, 28, 28)
    for data in test:
        assert data[0].shape == (1, 28, 28)
    
    train_targets = train.tensors[1].unique()
    test_targets = test.tensors[1].unique()
    
    assert (train_targets == torch.arange(0,10)).all() 
    assert (test_targets == torch.arange(0,10)).all()
        
