try:
    from unittest import mock
    from unittest.mock import Mock, MagicMock, call
except ImportError:
    from mock import mock
    from mock import Mock, MagicMock, call
