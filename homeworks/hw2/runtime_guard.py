"""Release Colab compute even when Drive or log finalization fails."""
import json
import time


def finalize_run(log, status, prepared, local_log, release):
    try:
        try:
            log.close()
        except Exception as error:
            status['log_close_error']=repr(error)
        status['finished_utc']=time.time()
        try:
            (prepared/'execution.log').write_bytes(local_log.read_bytes())
            (prepared/'execution.complete.json').write_text(json.dumps(status,indent=2))
        except Exception as error:
            status['artifact_error']=repr(error)
            print('RUN_ARTIFACT_ERROR',json.dumps(status),flush=True)
            try:
                print('LOCAL_LOG_TAIL',local_log.read_text()[-4000:],flush=True)
            except Exception:
                pass
    finally:
        release()
