import pandas as pd

from selmroute.data.splits import id_split
from selmroute.inference import align_features
from selmroute.paper.ablation import build_semantic_ablation_features
from selmroute.routing.catboost_router import CatBoostQualityRouter


def _semantic_frame(prefix: str = "jev") -> pd.DataFrame:
    row: dict[str, object] = {"sample_id": "a"}
    binary = ["math_reasoning","code_reasoning","formal_logic","factual_recall","social_affective","tool_interaction","external_knowledge","current_information"]
    scores = ["domain_specialization","reasoning_depth","constraint_density","context_integration","decomposition_need","ambiguity","exactness","answer_openness"]
    for name in binary:
        row[f"{prefix}_{name}__noul"] = 0.5
        row[f"{prefix}_{name}__entropy"] = 1.0
    for name in scores:
        for i, p in enumerate([0.1,0.2,0.3,0.4]):
            row[f"{prefix}_{name}__p__{i}"] = p
        row[f"{prefix}_{name}__entropy"] = 0.9
        row[f"{prefix}_{name}__expected"] = 2.0
        row[f"{prefix}_{name}__score"] = 2.0
        row[f"{prefix}_{name}__confidence"] = 0.4
        row[f"{prefix}_{name}__spread"] = 1.0
    return pd.DataFrame([row])


def test_probability_mass_is_40_features():
    pm = build_semantic_ablation_features(_semantic_frame(), "probability_mass")
    assert len(pm.columns) - 1 == 40


def test_grouped_split_keeps_duplicate_query_together():
    rows=[]
    for i in range(10):
        rows.append({"sample_id":str(i),"dataset":"d","query":"same" if i < 2 else f"q{i}"})
    s=pd.DataFrame(rows)
    tr,te=id_split(s,0.7,3407,group_duplicates=True)
    assert ({"0","1"} <= set(tr)) or ({"0","1"} <= set(te))


def test_catboost_save_load_and_alignment(tmp_path):
    X=pd.DataFrame({"sample_id":["a","b","c","d"],"f1":[0.,1.,0.,1.],"f2":[1.,0.,1.,0.]})
    Y=pd.DataFrame({"m1":[0.,1.,0.,1.],"m2":[1.,0.,1.,0.]})
    r=CatBoostQualityRouter(iterations=5,depth=2,seed=1).fit(X,Y)
    r.save(tmp_path)
    loaded=CatBoostQualityRouter.load(tmp_path)
    aligned=align_features(X.iloc[[0]],loaded)
    pred=loaded.predict(aligned)
    assert pred.shape == (1,2)
    assert loaded.model_names == ["m1","m2"]
