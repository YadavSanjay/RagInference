"""The public-safety check rejects real shapes and values, and passes placeholders.

Fixtures are assembled at runtime so this file does not trip the check itself.

    conda run -n py312 python -m pytest test/test_check_public_safe.py -q
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_public_safe.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_public_safe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load()


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("cd " + "D:" + "\\Tools\\AI\\repo", "machine path"),
        ("see /" + "home/alice/repo", "home directory"),
        ("key=" + "AKIA" + "ABCDEFGHIJKLMNOP", "access key"),
        ("-----BEGIN " + "RSA PRIVATE KEY-----", "private key"),
        ("mail alice" + "@" + "gmail.com", "personal email"),
        ("sg-" + "0123456789abcdef0", "AWS resource id"),
        ("arn:aws:iam::" + "1234" + "56789012:role/x", "account id"),
    ],
)
def test_real_shapes_are_rejected(line, kind):
    assert kind in check.scan_line(line, [])


@pytest.mark.parametrize(
    "line",
    [
        "arn:aws:iam::<ACCOUNT_ID>:role/<raginfer-task-role>",
        "image: 111122223333.dkr.ecr.us-east-1.amazonaws.com/raginfer:slim",
        "MODEL_S3_URI=s3://raginfer-models",
        "user@users.noreply.github.com",
    ],
)
def test_placeholders_pass(line):
    assert check.scan_line(line, []) == []


def test_denylisted_value_is_case_insensitive():
    assert check.scan_line("Profile: SECRET-Name", ["secret-name"]) == ["denylisted value"]


def test_load_literals_reads_deploy_env_and_denylist_but_not_region(tmp_path):
    aws = tmp_path / "infra" / "aws"
    aws.mkdir(parents=True)
    (aws / "deploy.env").write_text("ACCT=1234\nREGION=us-east-1\nAWS_PROFILE=Prof\n", "utf-8")
    (tmp_path / ".public-denylist").write_text("Name\n\n", "utf-8")
    assert check.load_literals(tmp_path) == ["1234", "prof", "name"]


def test_literals_are_not_applied_to_the_license(tmp_path):
    path = tmp_path / "LICENSE"
    path.write_text("Copyright (c) Name\n", "utf-8")
    assert check.scan_file(path, "LICENSE", ["name"]) == []
    assert check.scan_file(path, "NOTICE", ["name"]) == ["NOTICE:1: denylisted value"]


def test_findings_never_carry_the_value(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("Profile: secret-name\n", "utf-8")
    assert "secret-name" not in " ".join(check.scan_file(path, "notes.md", ["secret-name"]))
