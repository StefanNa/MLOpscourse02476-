import pytest
import sys
import os
import torch

sys.path.append(os.getcwd()+'/src/models')

from src.models.train_model import TrainOREvaluate




def test_error_on_wrong_shape():
    with pytest.raises(ImportError, match='model dict file is not present'):
        TrainOREvaluate()