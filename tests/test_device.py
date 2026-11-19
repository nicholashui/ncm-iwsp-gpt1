import pytest
import torch

from ncm_iwsp_gpt1.device import resolve_device


def test_resolve_device_auto_and_cpu():
    auto = resolve_device("auto")
    if torch.cuda.is_available():
        assert auto.type == "cuda"
    else:
        assert auto.type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device(None).type == auto.type


def test_resolve_device_rejects_unknown():
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_cuda_flag_requires_a_gpu():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
