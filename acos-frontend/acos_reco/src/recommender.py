from __future__ import annotations
from typing import Dict, List
from .textnorm import normalize_feature


class Recommender:
    def __init__(self, agg_rows: List[Dict]):
        self.agg = [
            {
                **r,
                "category": str(r["category"]),
                "feature": str(r["feature"]),
            }
            for r in agg_rows
        ]

    def recommend(
        self,
        selections: Dict[str, str],
        mode: str = "AND",
        cat_weights: Dict[str, float] | None = None,
        top_k: int = 20,
    ) -> List[Dict]:
        if not selections:
            return []
        # 정규화된 비교 키 준비
        sel_norm = {c: normalize_feature(f or "") for c, f in selections.items() if f}
        if not sel_norm:
            return []

        # candidate rows satisfying any selection
        cand: List[Dict] = []
        for r in self.agg:
            c = r.get("category")
            f = r.get("feature", "")
            if not c:
                continue
            if c in sel_norm:
                if normalize_feature(f) == sel_norm[c]:
                    cand.append(r)
        if not cand:
            return []

        # scoring per row
        def row_score(r: Dict) -> float:
            base = float(r.get("score", 0.0))
            freq = int(r.get("pos", 0)) + int(r.get("neu", 0)) + int(r.get("neg", 0))
            s = 0.6 * base + 0.4 * (freq ** 0.5)
            if cat_weights and r["category"] in cat_weights:
                s *= float(cat_weights[r["category"]])
            return s

        per_pid: Dict[str, Dict] = {}
        for r in cand:
            pid = r["product_id"]
            sc = row_score(r)
            if pid not in per_pid:
                per_pid[pid] = {
                    "product_id": pid,
                    "final_score": 0.0,
                    "categories": set(),
                    "best_row_score": -1e9,
                    "best_cat": None,
                    "best_feat": None,
                }
            per_pid[pid]["final_score"] += sc
            per_pid[pid]["categories"].add(r["category"])
            # 대표 노출용 행(카드 뱃지) 선정
            if sc > per_pid[pid]["best_row_score"]:
                per_pid[pid]["best_row_score"] = sc
                per_pid[pid]["best_cat"] = r["category"]
                per_pid[pid]["best_feat"] = r.get("feature", "")

        # enforce AND if needed
        need = set(sel_norm.keys())
        results: List[Dict] = []
        for pid, info in per_pid.items():
            if mode.upper() == "AND" and not need.issubset(info["categories"]):
                continue
            best_cat = info.get("best_cat")
            best_feat = info.get("best_feat") or ""
            results.append({
                "product_id": pid,
                "final_score": info["final_score"],
                "cats_covered": len(info["categories"]),
                "category": best_cat,
                "feature": best_feat,
                "feature_disp": normalize_feature(best_feat),
            })
        results.sort(key=lambda x: (x["final_score"], x["cats_covered"]), reverse=True)
        return results[:top_k]
