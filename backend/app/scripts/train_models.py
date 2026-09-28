"""Entrena y compara los 4 modelos y despliega el mejor.

Uso:  python -m app.scripts.train_models [--target pct_fair_poor_health] [--year 2023]
"""

import argparse

from app.core.database import SessionLocal
from app.services import ml_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=ml_service.DEFAULT_TARGET, choices=ml_service.TARGETS)
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        m = ml_service.train(db, target=args.target, year=args.year)
    finally:
        db.close()

    print(f"Objetivo: {m['target']['name']} ({m['year']}) · n={m['n_samples']} "
          f"(train {m['n_train']} / test {m['n_test']})")
    print(f"{'Modelo':<22}{'F1 CV':>14}{'F1 test':>10}{'Acc test':>10}{'AUC test':>10}")
    for r in m["models"]:
        mark = "  ← desplegado" if r["selected"] else ""
        print(f"{r['name']:<22}{r['cv_f1_mean']:>8.4f}±{r['cv_f1_std']:.3f}"
              f"{r['test_f1_macro']:>10.4f}{r['test_accuracy']:>10.4f}{r['test_roc_auc']:>10.4f}{mark}")
    print(f"Guardado en {ml_service.model_path(args.target)} ({m['total_seconds']} s)")


if __name__ == "__main__":
    main()
