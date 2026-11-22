from pathlib import Path

import ncm_iwsp_gpt1.writers.stub as stub

from ncm_iwsp_gpt1.audit import audit_source, audit_writer_module, code_only
from ncm_iwsp_gpt1.writers.stub import MODEL_DESCRIPTION


def test_comments_and_strings_are_not_code():
    src = '''
# loss.backward()
x = "torch.optim.AdamW"
def write_weights(tokenizer, generator, iteration=0):
    return
'''
    assert ".backward(" not in code_only(src)
    assert "AdamW" not in code_only(src)
    assert audit_source(src) == []


def test_stub_writer_passes():
    problems = audit_writer_module(stub, Path("ncm_iwsp_gpt1/writers/stub.py"))
    assert problems == []
    assert len(MODEL_DESCRIPTION) >= 80


def test_rejections(tmp_path):
    cases = {
        "backward.py": "def write_weights(tokenizer, generator, iteration=0):\n    loss.backward()\n",
        "adam.py": "import torch\ndef write_weights(tokenizer, generator, iteration=0):\n    torch.optim.Adam(tokenizer.parameters())\n",
        "adamw.py": "import torch\ndef write_weights(tokenizer, generator, iteration=0):\n    torch.optim.AdamW(tokenizer.parameters())\n",
        "sklearn_fit.py": "def write_weights(tokenizer, generator, iteration=0):\n    model.fit(x, y)\n",
        "download.py": "def write_weights(tokenizer, generator, iteration=0):\n    net.from_pretrained('gpt2')\n",
        "torch_load.py": "def write_weights(tokenizer, generator, iteration=0):\n    torch.load('weights.pt')\n",
        "hidden.py": "def write_weights(tokenizer, generator, iteration=0):\n    x = cache['hidden']\n",
        "hidden_log.py": "def write_weights(tokenizer, generator, iteration=0):\n    x = 'hidden_metrics'\n",
        "empty_desc.py": (
            "MODEL_SHORTHAND = 'x'\nMODEL_DESCRIPTION = 'too short'\n"
            "def write_weights(tokenizer, generator, iteration=0):\n    return\n"
        ),
        "takes_data.py": (
            "MODEL_SHORTHAND = 'bad-data'\n"
            "MODEL_DESCRIPTION = '" + ("This writer takes a data argument and that is not allowed. " * 2) + "'\n"
            "def write_weights(tokenizer, generator, iteration=0, hidden=None):\n    return\n"
        ),
    }
    for name, src in cases.items():
        path = tmp_path / name
        path.write_text(src, encoding="utf-8")
        if name in {"takes_data.py", "empty_desc.py"}:
            import importlib.util

            spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            problems = audit_writer_module(module, path)
        else:
            problems = audit_source(src)
        assert problems, name


_FINETUNE = Path("ncm_iwsp_gpt1/finetune.py")


def test_package_sources_do_not_train():
    root = Path("ncm_iwsp_gpt1")
    for path in root.rglob("*.py"):
        if path.as_posix() == _FINETUNE.as_posix():
            continue
        code = code_only(path.read_text(encoding="utf-8"))
        for banned in (".backward(", "torch.optim", "sklearn", "AdamW", "train_tokenizer", "train_generator"):
            assert banned not in code, f"{path} contains {banned}"


def test_finetune_is_the_only_optimizer_module():
    hits = []
    for path in Path("ncm_iwsp_gpt1").rglob("*.py"):
        code = code_only(path.read_text(encoding="utf-8"))
        if any(bit in code for bit in (".backward(", "torch.optim", "AdamW")):
            hits.append(path.as_posix())
    assert hits == ["ncm_iwsp_gpt1/finetune.py"]
    for rel in (
        "ncm_iwsp_gpt1/loop.py",
        "ncm_iwsp_gpt1/evaluate.py",
        "ncm_iwsp_gpt1/audit.py",
        "ncm_iwsp_gpt1/writers/stub.py",
        "ncm_iwsp_gpt1/convert.py",
    ):
        assert "ncm_iwsp_gpt1.finetune" not in Path(rel).read_text(encoding="utf-8")
