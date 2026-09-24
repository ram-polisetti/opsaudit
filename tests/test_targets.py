"""Tests for the Phase 1 target adapters.

No test in this file touches the network. HTTP-backed adapters are
exercised through a fake ``httpx`` module injected into ``sys.modules``;
``transformers`` is genuinely absent, so the lazy-import error path is
real.
"""

import json
import sys
import types

import pytest

from opsaudit.targets import (
    HuggingFaceTarget,
    OllamaTarget,
    OpenAICompatTarget,
    RagTarget,
    TabularTarget,
    Target,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
class _BareTarget(Target):
    def describe(self):
        return {"target_type": "bare", "name": self.name}


class _FakeEstimator:
    def __init__(self, preds, proba=None):
        self._preds = preds
        self._proba = proba

    def predict(self, X):
        return self._preds

    def predict_proba(self, X):
        if self._proba is None:
            raise AttributeError("no proba")
        return self._proba


def _install_fake_httpx(monkeypatch, handler):
    """Inject a fake httpx module whose Client POSTs to ``handler``.

    ``handler(url, headers, payload)`` returns the JSON body.
    """
    calls = []

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    class FakeClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):
            calls.append({"url": url, "headers": headers, "json": json})
            return FakeResponse(handler(url, headers or {}, json or {}))

    module = types.ModuleType("httpx")
    module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "httpx", module)
    return calls


# ----------------------------------------------------------------------
# Target base class
# ----------------------------------------------------------------------
class TestTargetBase:
    def test_default_methods_raise_not_implemented(self):
        t = _BareTarget()
        with pytest.raises(NotImplementedError):
            t.generate(["hi"])
        with pytest.raises(NotImplementedError):
            t.predict([[1, 2]])

    def test_describe_is_abstract(self):
        with pytest.raises(TypeError):
            Target()  # type: ignore[abstract]

    def test_supports_flags(self):
        assert _BareTarget().supports_generate is False
        assert _BareTarget().supports_predict is False
        assert TabularTarget(_FakeEstimator([0])).supports_predict is True
        assert TabularTarget(_FakeEstimator([0])).supports_generate is False
        assert RagTarget(lambda q: q).supports_generate is True

    def test_generate_empty_list_short_circuits_without_network(self):
        # Even HTTP adapters must not touch the network for [].
        t = OllamaTarget(model="m", base_url="http://127.0.0.1:9")
        assert t.generate([]) == []


# ----------------------------------------------------------------------
# TabularTarget
# ----------------------------------------------------------------------
class TestTabularTarget:
    def test_predict_passthrough(self):
        t = TabularTarget(_FakeEstimator([0, 1, 1]), name="clf")
        assert t.predict([[1], [2], [3]]) == [0, 1, 1]

    def test_predict_numpy_like(self):
        class Numpyish:
            def predict(self, X):
                import numpy as np

                return np.array([1, 0])

        assert TabularTarget(Numpyish()).predict([[0], [1]]) == [1, 0]

    def test_predict_returns_python_natives(self):
        # Regression: numpy scalars (np.int64) failed _is_number in the
        # agentic campaign's scorer, so a real sklearn-backed tabular
        # target silently produced strength 0.0 on every round.
        class Numpyish:
            def predict(self, X):
                import numpy as np

                return np.array([1, 0])

        preds = TabularTarget(Numpyish()).predict([[0], [1]])
        assert all(type(p) is int for p in preds)

    def test_predict_proba_when_supported(self):
        t = TabularTarget(_FakeEstimator([1], proba=[[0.2, 0.8]]))
        assert t.predict_proba([[9]]) == [[0.2, 0.8]]

    def test_predict_proba_none_when_unsupported(self):
        class NoProba:
            def predict(self, X):
                return [0]

        assert TabularTarget(NoProba()).predict_proba([[1]]) is None

    def test_rejects_non_estimator(self):
        with pytest.raises(ValueError, match="predict"):
            TabularTarget(object())
        with pytest.raises(ValueError, match="predict"):
            TabularTarget("not-an-estimator")

    def test_rejects_none_X(self):
        with pytest.raises(ValueError, match="must not be None"):
            TabularTarget(_FakeEstimator([0])).predict(None)

    def test_wraps_estimator_failure(self):
        class Boom:
            def predict(self, X):
                raise RuntimeError("kaput")

        with pytest.raises(RuntimeError, match="estimator.predict failed"):
            TabularTarget(Boom()).predict([[1]])

    def test_rejects_non_iterable_predictions(self):
        class Weird:
            def predict(self, X):
                return 42

        with pytest.raises(ValueError, match="iterable"):
            TabularTarget(Weird()).predict([[1]])

    def test_describe_json_safe_no_network(self):
        d = TabularTarget(_FakeEstimator([0]), name="t").describe()
        assert d["target_type"] == "tabular"
        json.dumps(d)  # must be JSON-serializable


# ----------------------------------------------------------------------
# RagTarget
# ----------------------------------------------------------------------
class TestRagTarget:
    def test_generate_passthrough(self):
        t = RagTarget(lambda q: f"answer: {q}", name="r")
        assert t.generate(["q1", "q2"]) == ["answer: q1", "answer: q2"]

    def test_empty_prompts(self):
        assert RagTarget(lambda q: q).generate([]) == []

    def test_none_response_becomes_empty_string(self):
        assert RagTarget(lambda q: None).generate(["q"]) == [""]

    def test_rejects_non_callable(self):
        with pytest.raises(ValueError, match="callable"):
            RagTarget(42)

    def test_wraps_pipeline_exception(self):
        def bad(q):
            raise KeyError("missing doc")

        with pytest.raises(RuntimeError, match="rag_fn raised"):
            RagTarget(bad).generate(["q"])

    def test_rejects_non_string_response(self):
        with pytest.raises(ValueError, match="must return a string"):
            RagTarget(lambda q: ["not", "a", "string"]).generate(["q"])

    def test_describe_json_safe_no_network(self):
        def my_pipeline(q):
            return q

        d = RagTarget(my_pipeline, name="r").describe()
        assert d["target_type"] == "rag"
        assert d["pipeline"] == "my_pipeline"
        json.dumps(d)


# ----------------------------------------------------------------------
# HuggingFaceTarget (transformers genuinely absent here)
# ----------------------------------------------------------------------
class TestHuggingFaceTarget:
    def test_rejects_empty_model_id(self):
        with pytest.raises(ValueError, match="model_id"):
            HuggingFaceTarget("")
        with pytest.raises(ValueError, match="model_id"):
            HuggingFaceTarget("   ")

    def test_describe_without_network_or_dependency(self):
        d = HuggingFaceTarget("some-org/some-model").describe()
        assert d["target_type"] == "huggingface"
        assert d["model_id"] == "some-org/some-model"
        json.dumps(d)

    def test_generate_raises_helpful_import_error(self):
        t = HuggingFaceTarget("some-org/some-model")
        with pytest.raises(ImportError, match="transformers"):
            t.generate(["hello"])

    def test_empty_prompts_never_loads_model(self):
        t = HuggingFaceTarget("some-org/some-model")
        assert t.generate([]) == []


# ----------------------------------------------------------------------
# OpenAICompatTarget
# ----------------------------------------------------------------------
class TestOpenAICompatTarget:
    def test_rejects_empty_config(self):
        with pytest.raises(ValueError, match="base_url"):
            OpenAICompatTarget("", "k", "m")
        with pytest.raises(ValueError, match="api_key"):
            OpenAICompatTarget("https://x", "", "m")
        with pytest.raises(ValueError, match="model"):
            OpenAICompatTarget("https://x", "k", "")

    def test_generate_via_fake_http(self, monkeypatch):
        def handler(url, headers, payload):
            assert url == "https://llm.example.com/v1/chat/completions"
            assert headers["Authorization"] == "Bearer sk-test"
            assert payload["model"] == "test-model"
            assert payload["messages"] == [{"role": "user", "content": "hi"}]
            return {"choices": [{"message": {"content": "hello back"}}]}

        calls = _install_fake_httpx(monkeypatch, handler)
        t = OpenAICompatTarget(
            base_url="https://llm.example.com/v1/",
            api_key="sk-test",
            model="test-model",
        )
        assert t.generate(["hi"]) == ["hello back"]
        assert len(calls) == 1

    def test_generate_empty_list(self, monkeypatch):
        calls = _install_fake_httpx(
            monkeypatch, lambda u, h, p: {"choices": []}
        )
        t = OpenAICompatTarget("https://x", "k", "m")
        assert t.generate([]) == []
        assert calls == []

    def test_malformed_response_raises(self, monkeypatch):
        _install_fake_httpx(monkeypatch, lambda u, h, p: {"nope": True})
        t = OpenAICompatTarget("https://x", "k", "m")
        with pytest.raises(ValueError, match="unexpected response shape"):
            t.generate(["hi"])

    def test_missing_httpx_raises_helpful_error(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "httpx", raising=False)
        monkeypatch.setattr(
            "builtins.__import__",
            _raise_on_httpx(__import__),
        )
        t = OpenAICompatTarget("https://x", "k", "m")
        with pytest.raises(ImportError, match="httpx"):
            t.generate(["hi"])

    def test_describe_omits_api_key(self):
        d = OpenAICompatTarget("https://x", "sk-secret", "m").describe()
        assert "sk-secret" not in json.dumps(d)
        assert "api_key" not in d
        assert d["target_type"] == "openai_compat"


def _raise_on_httpx(real_import):
    def fake(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    return fake


# ----------------------------------------------------------------------
# OllamaTarget
# ----------------------------------------------------------------------
class TestOllamaTarget:
    def test_env_config(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "env-model")
        monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.example.com")
        monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
        t = OllamaTarget()
        assert t.model == "env-model"
        assert t.base_url == "https://ollama.example.com"
        assert t.api_key == "env-key"

    def test_explicit_args_beat_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "env-model")
        t = OllamaTarget(
            model="arg-model", base_url="http://localhost:11434"
        )
        assert t.model == "arg-model"

    def test_missing_model_raises(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        with pytest.raises(ValueError, match="OLLAMA_MODEL"):
            OllamaTarget()

    def test_default_base_url_is_local(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.setenv("OLLAMA_MODEL", "m")
        t = OllamaTarget()
        assert t.base_url == "http://localhost:11434"

    def test_generate_via_fake_http(self, monkeypatch):
        def handler(url, headers, payload):
            assert url == "http://localhost:11434/api/chat"
            assert payload["model"] == "m"
            assert payload["stream"] is False
            assert payload["messages"] == [{"role": "user", "content": "ping"}]
            return {"message": {"content": "pong"}}

        _install_fake_httpx(monkeypatch, handler)
        t = OllamaTarget(model="m")
        assert t.generate(["ping"]) == ["pong"]

    def test_auth_header_sent_when_key_set(self, monkeypatch):
        seen = {}

        def handler(url, headers, payload):
            seen.update(headers)
            return {"message": {"content": "ok"}}

        _install_fake_httpx(monkeypatch, handler)
        OllamaTarget(model="m", api_key="k").generate(["q"])
        assert seen["Authorization"] == "Bearer k"

    def test_malformed_response_raises(self, monkeypatch):
        _install_fake_httpx(monkeypatch, lambda u, h, p: {"bogus": 1})
        with pytest.raises(ValueError, match="unexpected response shape"):
            OllamaTarget(model="m").generate(["q"])

    def test_describe_omits_api_key(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "m")
        d = OllamaTarget(api_key="super-secret").describe()
        assert "super-secret" not in json.dumps(d)
        assert d["target_type"] == "ollama"

    def test_describe_needs_no_network(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "m")
        json.dumps(OllamaTarget().describe())


class TestSecureOllamaTarget:
    """SecureOllamaTarget: Ollama Cloud via the stored custom.ollama
    connector (authd surrogates). No raw key anywhere."""

    def _install_fake_surrogates(self, monkeypatch, seen):
        import opsaudit.targets.ollama as ollama_mod

        def fake_add_surrogate(req, cred, entry_name=None, allowed_hosts=None):
            seen["cred"] = cred
            seen["hosts"] = allowed_hosts
            # Surrogate marker only — never a raw key.
            req.add_header("X-Hatch-Surrogate", "hsurr:fake")

        def fake_read_json(resp):
            return {"message": {"content": "secure-ok"}}

        monkeypatch.setattr(
            ollama_mod, "_surrogate_helpers",
            lambda: (fake_add_surrogate, fake_read_json),
        )

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["auth_header"] = req.get_header("Authorization")
            seen["surrogate_header"] = req.get_header("X-hatch-surrogate")
            return FakeResp()

        monkeypatch.setattr(
            "urllib.request.urlopen", fake_urlopen
        )

    def test_generate_uses_surrogate_not_key(self, monkeypatch):
        from opsaudit.targets.ollama import SecureOllamaTarget

        seen = {}
        self._install_fake_surrogates(monkeypatch, seen)
        # No OLLAMA_API_KEY in the environment at all.
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        out = SecureOllamaTarget(model="glm-5.1").generate(["hello"])
        assert out == ["secure-ok"]
        assert seen["cred"] == "custom.ollama"
        assert seen["hosts"] == ["ollama.com"]
        assert seen["url"] == "https://ollama.com/api/chat"
        assert seen["auth_header"] is None  # no Bearer key header
        assert seen["surrogate_header"] == "hsurr:fake"

    def test_needs_no_api_key_env(self, monkeypatch):
        from opsaudit.targets.ollama import SecureOllamaTarget

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        t = SecureOllamaTarget(model="glm-5.1")
        assert t.api_key == ""
        assert t.base_url == "https://ollama.com"

    def test_describe_leaks_no_credential(self, monkeypatch):
        from opsaudit.targets.ollama import SecureOllamaTarget

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        d = SecureOllamaTarget(model="glm-5.1").describe()
        blob = json.dumps(d)
        assert "api_key" not in blob.lower().replace("api key", "")
        assert "hsurr" not in blob
        assert "Bearer" not in blob
        assert d["target_type"] == "ollama-secure"

    def test_missing_surrogate_helper_raises_clearly(self, monkeypatch):
        import opsaudit.targets.ollama as ollama_mod
        from opsaudit.targets.ollama import SecureOllamaTarget

        def boom():
            raise ImportError("no surrogate helpers here")

        monkeypatch.setattr(ollama_mod, "_surrogate_helpers", boom)
        with pytest.raises(ImportError):
            SecureOllamaTarget(model="glm-5.1").generate(["hi"])

    def test_model_required(self, monkeypatch):
        from opsaudit.targets.ollama import SecureOllamaTarget

        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        with pytest.raises(ValueError, match="model"):
            SecureOllamaTarget()
