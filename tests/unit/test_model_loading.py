"""Disk-cache loading avoids Hub checks without preventing first downloads."""

from unittest.mock import Mock, call
import pytest
from weldscope.core.model_loading import load_cached_model


def test_cached_model_uses_only_local_files():
    factory = Mock()
    assert load_cached_model(factory, "model") is factory.return_value
    factory.assert_called_once_with("model", local_files_only=True)


def test_missing_cache_downloads_normally():
    model = object()
    factory = Mock(side_effect=[OSError("Not cached"), model])
    assert load_cached_model(factory, "model") is model
    assert factory.call_args_list == [call("model", local_files_only=True), call("model")]


def test_invalid_model_configuration_is_not_retried():
    factory = Mock(side_effect=ValueError("Invalid configuration"))
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_cached_model(factory, "model")
    factory.assert_called_once()
