# coding: utf-8
import json
import os
import pickle
import subprocess
import sys
from typing import NamedTuple, Dict

import azure.cognitiveservices.vision.computervision
import boto3
import pytest
import requests
from _pytest.python_api import ApproxBase, ApproxMapping, ApproxSequenceLike
from google.cloud import vision

from tests import MOCK_DIR


class ApproxBaseReprMixin(ApproxBase):
    def __repr__(self) -> str:
        def recur_repr_helper(obj):
            if isinstance(obj, dict):
                return dict((k, recur_repr_helper(v)) for k, v in obj.items())
            elif isinstance(obj, tuple):
                return tuple(recur_repr_helper(o) for o in obj)
            elif isinstance(obj, list):
                return list(recur_repr_helper(o) for o in obj)
            else:
                return self._approx_scalar(obj)

        return "approx({!r})".format(recur_repr_helper(self.expected))


class ApproxNestedSequenceLike(ApproxBaseReprMixin, ApproxSequenceLike):
    def _yield_comparisons(self, actual):
        for k in range(len(self.expected)):
            if isinstance(self.expected[k], dict):
                mapping = ApproxNestedMapping(self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
                for el in mapping._yield_comparisons(actual[k]):
                    yield el
            elif isinstance(self.expected[k], (tuple, list)):
                mapping = ApproxNestedSequenceLike(self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
                for el in mapping._yield_comparisons(actual[k]):
                    yield el
            else:
                yield actual[k], self.expected[k]

    def _check_type(self):
        pass


class ApproxNestedMapping(ApproxBaseReprMixin, ApproxMapping):
    def _yield_comparisons(self, actual):
        for k in self.expected.keys():
            if isinstance(self.expected[k], dict):
                mapping = ApproxNestedMapping(self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
                for el in mapping._yield_comparisons(actual[k]):
                    yield el
            elif isinstance(self.expected[k], (tuple, list)):
                mapping = ApproxNestedSequenceLike(self.expected[k], rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
                for el in mapping._yield_comparisons(actual[k]):
                    yield el
            else:
                yield actual[k], self.expected[k]

    def _check_type(self):
        pass


def nested_approx(expected, rel=None, abs=None, nan_ok=False):
    if isinstance(expected, dict):
        return ApproxNestedMapping(expected, rel, abs, nan_ok)
    if isinstance(expected, (tuple, list)):
        return ApproxNestedSequenceLike(expected, rel, abs, nan_ok)
    return pytest.approx(expected, rel, abs, nan_ok)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture
def mock_tesseract(monkeypatch):
    def mock_check_output(*args, **kwargs):
        if "tesseract --list-langs" in args:
            return "Langs\neng".encode("utf-8")
        else:
            with open(os.path.join(MOCK_DIR, "tesseract_hocr.html"), "r") as f:
                return f.read().encode("utf-8")

    def mock_run(*args, **kwargs):
        class MResp:
            @property
            def returncode(self):
                return 0
        return MResp()

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    monkeypatch.setattr(subprocess, "run", mock_run)


@pytest.fixture
def mock_vision(monkeypatch):
    class MockPost:
        def json(self, *args, **kwargs):
            with open(os.path.join(MOCK_DIR, "vision.json"), "r") as f:
                return json.load(f)

    def mock_post(*args, **kwargs):
        return MockPost()

    # Mock post to API
    monkeypatch.setattr(requests, "post", mock_post)

    def mock_init(*args, **kwargs):
        pass

    def mock_annotate(*args, **kwargs):
        with open(os.path.join(MOCK_DIR, "vision.pkl"), "rb") as f:
            resp = pickle.load(f)

        return resp

    # Mock Vision API annotate
    monkeypatch.setattr(vision.ImageAnnotatorClient, "__init__", mock_init)
    monkeypatch.setattr(vision.ImageAnnotatorClient, "batch_annotate_images", mock_annotate)


@pytest.fixture
def mock_textract(monkeypatch):
    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def detect_document_text(*args, **kwargs):
            with open(os.path.join(MOCK_DIR, "textract.json"), "r") as f:
                resp = json.load(f)

            return resp

    # Mock boto3 client
    monkeypatch.setattr(boto3, "client", MockClient)


@pytest.fixture
def mock_azure(monkeypatch):
    class MockRead(NamedTuple):
        headers: Dict

    def mock_read_in_stream(*args, **kwargs):
        return MockRead(headers={"Operation-Location": "zz/zz"})

    def mock_get_read_result(*args, **kwargs):
        with open(os.path.join(MOCK_DIR, "azure.pkl"), "rb") as f:
            resp = pickle.load(f)
        return resp

    # Mock azure client
    monkeypatch.setattr(azure.cognitiveservices.vision.computervision.ComputerVisionClient,
                        "read_in_stream",
                        mock_read_in_stream)
    monkeypatch.setattr(azure.cognitiveservices.vision.computervision.ComputerVisionClient,
                        "get_read_result",
                        mock_get_read_result)


@pytest.fixture
def mock_surya(monkeypatch):
    def mock_run_ocr(*args, **kwargs):
        with open(os.path.join(MOCK_DIR, "surya.pkl"), "rb") as f:
            resp = pickle.load(f)
        return resp

    if sys.version_info >= (3, 10):
        import surya.recognition
        # Mock surya
        monkeypatch.setattr(surya.recognition.RecognitionPredictor,
                            "__call__",
                            mock_run_ocr)

