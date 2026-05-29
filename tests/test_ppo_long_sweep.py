from io import BytesIO

from scripts.run_ppo_long_sweep import _CarriageReturnLogFilter


def _filtered_log(*chunks: bytes) -> bytes:
    output = BytesIO()
    log_filter = _CarriageReturnLogFilter(output)
    for chunk in chunks:
        log_filter.write(chunk)
    log_filter.close()
    return output.getvalue()


def test_progress_log_filter_collapses_carriage_return_redraws():
    output = _filtered_log(
        b"started\r\n",
        b"\rPPO training: 1%",
        b"\rPPO training: 2%",
        b"\rPPO training: 100%\r\n",
        b"finished\r\n",
    )

    assert output == b"started\nPPO training: 100%\nfinished\n"


def test_progress_log_filter_handles_split_crlf():
    output = _filtered_log(b"summary line\r", b"\n")

    assert output == b"summary line\n"


def test_progress_log_filter_flushes_final_unterminated_line():
    output = _filtered_log(b"\rEvaluate random: 100%")

    assert output == b"Evaluate random: 100%\n"
