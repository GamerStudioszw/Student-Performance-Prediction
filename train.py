"""Explicitly prepare new immutable model/evaluation runs; preserve legacy evidence."""
from model import ROOT, PUBLIC_SUBJECTS, load_data, train_and_evaluate
from model_service import SOURCE, file_hash, finalize_bundle, save_bundle


def main():
    for subject in PUBLIC_SUBJECTS:
        path = ROOT / "data" / f"student-{subject}.csv"
        bundle = finalize_bundle(*train_and_evaluate(load_data(path)), subject=subject,
                                 fingerprint=file_hash(path), source=SOURCE)
        directory = save_bundle(bundle)
        metrics = bundle[1]
        print(f"{metrics['subject_label']}: {metrics['selected_model']} ({metrics['feature_set']}); "
              f"MAE={metrics['mae']:.6f}; coverage={metrics['interval']['test_coverage']:.3%}; "
              f"width={metrics['interval']['average_width']:.6f}; saved to {directory}", flush=True)


if __name__ == "__main__":
    main()
