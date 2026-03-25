# Copyright (c) 2021, yolort team. All rights reserved.
import warnings

import pytest
from yolort.utils.dependency import check_version, deprecated, is_module_available, requires_module


class TestCheckVersion:
    def test_current_greater_than_minimum(self):
        assert check_version(current="1.2.0", minimum="1.0.0") is True

    def test_current_equals_minimum(self):
        assert check_version(current="1.0.0", minimum="1.0.0") is True

    def test_current_less_than_minimum(self):
        assert check_version(current="0.9.0", minimum="1.0.0") is False

    def test_pinned_exact_match(self):
        assert check_version(current="1.0.0", minimum="1.0.0", pinned=True) is True

    def test_pinned_not_matching(self):
        assert check_version(current="1.1.0", minimum="1.0.0", pinned=True) is False

    def test_hard_assertion_passes(self):
        # Should not raise
        check_version(current="2.0.0", minimum="1.0.0", hard=True)

    def test_hard_assertion_fails(self):
        with pytest.raises(AssertionError):
            check_version(current="0.5.0", minimum="1.0.0", hard=True)

    def test_verbose_no_warning_when_met(self, caplog):
        check_version(current="1.0.0", minimum="1.0.0", verbose=True)
        assert "required by yolort" not in caplog.text

    def test_verbose_warning_when_not_met(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            check_version(current="0.5.0", minimum="1.0.0", verbose=True)
        assert "required by yolort" in caplog.text


class TestIsModuleAvailable:
    def test_available_module(self):
        assert is_module_available("os") is True

    def test_unavailable_module(self):
        assert is_module_available("nonexistent_module_xyz") is False

    def test_multiple_available_modules(self):
        assert is_module_available("os", "sys") is True

    def test_one_unavailable_in_multiple(self):
        assert is_module_available("os", "nonexistent_module_xyz") is False


class TestRequiresModule:
    def test_decorator_with_available_module(self):
        @requires_module("os")
        def my_func():
            return 42

        assert my_func() == 42

    def test_decorator_with_unavailable_module(self):
        @requires_module("nonexistent_module_xyz")
        def my_func():
            return 42

        with pytest.raises(RuntimeError, match="requires module"):
            my_func()

    def test_decorator_preserves_function_name(self):
        @requires_module("nonexistent_module_xyz")
        def my_special_func():
            return 42

        assert my_special_func.__name__ == "my_special_func"

    def test_decorator_with_multiple_missing_modules(self):
        @requires_module("nonexistent1", "nonexistent2")
        def my_func():
            return 42

        with pytest.raises(RuntimeError, match="requires module"):
            my_func()


class TestDeprecated:
    def test_deprecated_issues_warning(self):
        @deprecated("Use new_func instead.")
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            assert result == 42
            assert len(w) == 1
            assert "has been deprecated" in str(w[0].message)
            assert "Use new_func instead." in str(w[0].message)

    def test_deprecated_with_version(self):
        @deprecated("Use new_func instead.", version="v2.0")
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_func()
            assert "v2.0" in str(w[0].message)

    def test_deprecated_without_version(self):
        @deprecated("Use new_func instead.")
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_func()
            assert "future" in str(w[0].message)

    def test_deprecated_preserves_function_name(self):
        @deprecated("Use new_func instead.")
        def my_old_func():
            return 42

        assert my_old_func.__name__ == "my_old_func"
