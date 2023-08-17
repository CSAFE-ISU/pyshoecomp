__all__ = ("runner",)

import time


def runner(window, worker):
    worker.start()

    worker.debug_text = "loading images"
    worker.percentage = 5
    time.sleep(0.5)

    worker.debug_text = "extracting interest points"
    worker.percentage = 25
    time.sleep(0.5)

    worker.debug_text = "aligning impressions"
    worker.percentage = 45
    time.sleep(0.5)

    worker.debug_text = "calculating similarity"
    worker.percentage = 75
    time.sleep(0.5)

    worker.debug_text = "creating report"
    worker.percentage = 95
    time.sleep(0.5)

    while worker.percentage < 100:
        worker.percentage += 1
        time.sleep(0.15)

    window.success = True
    worker.finish()
