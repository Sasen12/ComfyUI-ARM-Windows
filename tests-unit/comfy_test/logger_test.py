import io
from collections import deque

import app.logger as logger


class TestLogInterceptor:
    def test_write_errors_do_not_crash_the_app(self, monkeypatch):
        monkeypatch.setattr(logger, "logs", deque(maxlen=10))

        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        interceptor = logger.LogInterceptor(stream)
        stream.close()

        payload = "\rprogress 1"
        assert interceptor.write(payload) == len(payload)
        assert logger.logs[-1]["m"] == payload

        interceptor.flush()

    def test_carriage_return_updates_work_with_empty_log_buffer(self, monkeypatch):
        monkeypatch.setattr(logger, "logs", deque(maxlen=10))

        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        interceptor = logger.LogInterceptor(stream)

        payload = "\rprogress 2"
        assert interceptor.write(payload) == len(payload)
        assert logger.logs[-1]["m"] == payload
