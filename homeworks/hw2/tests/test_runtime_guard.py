import json
from unittest.mock import Mock

from homeworks.hw2.runtime_guard import finalize_run


def test_release_even_when_log_close_and_drive_both_fail(tmp_path):
    log=Mock();log.close.side_effect=OSError(107,'Transport endpoint is not connected')
    local=tmp_path/'execution.log';local.write_text('original failure evidence')
    prepared=tmp_path/'missing-drive'
    release=Mock();status={'returncode':1}
    finalize_run(log,status,prepared,local,release)
    release.assert_called_once_with()
    assert 'log_close_error' in status and 'artifact_error' in status
    assert local.read_text()=='original failure evidence'


def test_persist_local_log_and_status_before_release(tmp_path):
    prepared=tmp_path/'drive';prepared.mkdir()
    local=tmp_path/'execution.log';log=local.open('w');log.write('checkpoint complete')
    def release():
        assert (prepared/'execution.log').read_text()=='checkpoint complete'
        assert json.loads((prepared/'execution.complete.json').read_text())['returncode']==0
    finalize_run(log,{'returncode':0},prepared,local,release)
    assert log.closed
