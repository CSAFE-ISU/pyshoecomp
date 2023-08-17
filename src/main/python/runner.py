__all__ = ("runner",)

import time

from imdesc import ImageDesc
from extractor import EXTRACTOR_MAP
from corresponder import CORRESPONDER_MAP
from aligner import (
    ALIGNER_MAP,
    get_QK_correspondence,
    get_alignment_function,
)
from scorer import SCORINGMETHOD_MAP


def _runner(
    worker, res, k_path, q_path, etor_name, scorer_name, aligner_name, epsilon, alpha
):
    # ouch
    try:
        worker.debug_text = "loading images"
        q = ImageDesc.from_file(
            q_path,
            is_k=False,
            is_match=True,
        )
        k = ImageDesc.from_file(
            k_path,
            is_k=True,
            is_match=True,
        )
        worker.percentage = 5
    except Exception as e:
        res["message"] = e
        return False

    try:
        worker.debug_text = "extracting interest points"
        extractor = EXTRACTOR_MAP[etor_name]()
        q.points = extractor(q.img)
        k.points = extractor(k.img)
        worker.percentage = 25
        time.sleep(0.5)
    except Exception as e:
        res["message"] = e
        return False

    try:
        worker.debug_text = "aligning impressions"
        cder = CORRESPONDER_MAP["clique2"](
            epsilon=float(epsilon), epsilon2=5, alpha=float(alpha)
        )
        corr = cder(q, k)
        mapping = get_alignment_function(q, k, corr, method_name=aligner_name)
        map_func = mapping(q, k, corr)
        q.aligned_img = mapping.align_Q_to_K(q, k, corr, map_func=map_func)
        worker.percentage = 75
        time.sleep(0.5)
    except Exception as e:
        res["message"] = e
        return False

    try:
        worker.debug_text = "calculating similarity"
        scor = SCORINGMETHOD_MAP[scorer_name](
            Q=q, K=k, corr=corr, map_func=map_func, epsilon=5
        )
        point = scor()
        worker.percentage = 85
        time.sleep(0.5)
    except Exception as e:
        res["message"] = e
        return False

    try:
        worker.debug_text = "creating report"
        worker.percentage = 95
        time.sleep(0.5)
    except Exception as e:
        res["message"] = e
        return False

    while worker.percentage < 100:
        worker.percentage += 1
        time.sleep(0.05)

    res["q"] = q
    res["k"] = k
    res["cder"] = cder
    res["corr"] = corr
    details = {
        "extractor": etor_name,
        "q_pts": len(q.points),
        "k_pts": len(k.points),
        "corresponder": "clique2",
        "alignment": aligner_name,
        "metric": scorer_name,
        "score": point,
        "eps1": epsilon,
        "alpha": alpha,
    }
    res.update(details)

    return True


def runner(window, worker):
    worker.start()
    res = dict(message="")

    k_path = window.file1.text()
    q_path = window.file2.text()
    etor_name = window.point_options.currentText()
    aligner_name = window.align_options.currentText()
    scorer_name = window.score_options.currentText()
    epsilon = window.clique_eps.text()
    alpha = window.clique_alpha.text()

    window.success = _runner(
        worker=worker,
        res=res,
        k_path=k_path,
        q_path=q_path,
        etor_name=etor_name,
        aligner_name=aligner_name,
        scorer_name=scorer_name,
        epsilon=epsilon,
        alpha=alpha,
    )
    window.sinfo = res
    worker.finish()
