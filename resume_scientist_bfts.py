"""Resume an AI Scientist run after its shared tree search has completed."""

import argparse
import json
import os
import os.path as osp
import shutil

from ai_scientist.llm import create_client
from ai_scientist.perform_coevaluation import (
    create_coevaluation_branch,
    run_coevaluation_pipeline,
    snapshot_baseline_artifacts,
    write_comparison_manifest,
)
from ai_scientist.perform_icbinb_writeup import (
    gather_citations,
    perform_writeup as perform_icbinb_writeup,
)
from ai_scientist.perform_llm_review import load_paper, perform_review
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.perform_writeup import perform_writeup as perform_normal_writeup
from ai_scientist.utils.token_tracker import token_tracker
from launch_scientist_bfts import find_pdf_path_for_review


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Continue writing, review, and co-evaluation from completed experiments"
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--writeup-type", choices=["normal", "icbinb"], default="icbinb"
    )
    parser.add_argument("--writeup-retries", type=int, default=3)
    parser.add_argument("--model_agg_plots", default="o3-mini-2025-01-31")
    parser.add_argument("--model_writeup", default="o1-preview-2024-09-12")
    parser.add_argument("--model_citation", default="gpt-4o-2024-11-20")
    parser.add_argument("--num_cite_rounds", type=int, default=20)
    parser.add_argument("--model_writeup_small", default="gpt-4o-2024-05-13")
    parser.add_argument("--model_review", default="gpt-4o-2024-11-20")
    parser.add_argument("--model_coeval", default=None)
    parser.add_argument("--coeval-experiment-steps", type=int, default=3)
    parser.add_argument("--co-evaluation", action="store_true")
    args = parser.parse_args()
    # run_coevaluation_pipeline checks these launcher-compatible fields.
    args.skip_writeup = False
    args.skip_review = False
    return args


def _validate_completed_experiment(base_folder):
    required = [
        "idea.json",
        "logs/0-run/baseline_summary.json",
        "logs/0-run/research_summary.json",
        "logs/0-run/ablation_summary.json",
    ]
    missing = [path for path in required if not osp.isfile(osp.join(base_folder, path))]
    if missing:
        raise FileNotFoundError(
            "Cannot resume before tree search is complete; missing: " + ", ".join(missing)
        )


def _save_resume_tokens(base_folder):
    with open(osp.join(base_folder, "token_tracker_resume.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(
        osp.join(base_folder, "token_tracker_resume_interactions.json"), "w"
    ) as f:
        json.dump(token_tracker.get_interactions(), f)


def _prepare_plots(base_folder, model):
    aggregator_path = osp.join(base_folder, "auto_plot_aggregator.py")
    if osp.isfile(aggregator_path):
        print(f"Reusing existing plot aggregation: {aggregator_path}")
        return

    result_src = osp.join(base_folder, "logs", "0-run", "experiment_results")
    result_dst = osp.join(base_folder, "experiment_results")
    if osp.isdir(result_src):
        shutil.copytree(result_src, result_dst, dirs_exist_ok=True)
    aggregate_plots(base_folder=base_folder, model=model)
    if osp.isdir(result_dst):
        shutil.rmtree(result_dst)


def _write_baseline(base_folder, args):
    citations_text = None
    if args.writeup_type == "icbinb":
        citations_text = gather_citations(
            base_folder,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )

    success = False
    for attempt in range(args.writeup_retries):
        print(f"Writeup attempt {attempt + 1} of {args.writeup_retries}")
        if args.writeup_type == "normal":
            success = perform_normal_writeup(
                base_folder=base_folder,
                small_model=args.model_writeup_small,
                big_model=args.model_writeup,
                page_limit=8,
            )
        else:
            success = perform_icbinb_writeup(
                base_folder=base_folder,
                small_model=args.model_writeup_small,
                big_model=args.model_writeup,
                page_limit=4,
                citations_text=citations_text,
            )
        if success:
            break
    if not success:
        raise RuntimeError("Baseline writeup did not complete successfully.")

    pdf_path = find_pdf_path_for_review(base_folder)
    if not pdf_path or not osp.isfile(pdf_path):
        raise RuntimeError("Baseline writeup reported success but produced no PDF.")
    paper_content = load_paper(pdf_path)
    client, client_model = create_client(args.model_review)
    review = perform_review(paper_content, client_model, client)
    image_review = perform_imgs_cap_ref_review(client, client_model, pdf_path)
    with open(osp.join(base_folder, "review_text.txt"), "w") as f:
        json.dump(review, f, indent=4)
    with open(osp.join(base_folder, "review_img_cap_ref.json"), "w") as f:
        json.dump(image_review, f, indent=4)
    print("Paper review completed.")


def main():
    args = parse_arguments()
    os.environ["AI_SCIENTIST_ROOT"] = osp.dirname(osp.abspath(__file__))
    base_folder = osp.normpath(args.experiment_dir)
    _validate_completed_experiment(base_folder)

    coeval_folder = None
    if args.co_evaluation:
        coeval_folder = base_folder + "_coeval"
        if osp.isdir(coeval_folder):
            print(f"Reusing existing co-evaluation branch: {coeval_folder}")
        else:
            coeval_folder = create_coevaluation_branch(base_folder)

    _prepare_plots(base_folder, args.model_agg_plots)
    _write_baseline(base_folder, args)
    baseline_manifest = snapshot_baseline_artifacts(base_folder)
    _save_resume_tokens(base_folder)

    if coeval_folder is not None:
        if not baseline_manifest.get("final_pdf"):
            raise RuntimeError("The resumed baseline branch did not produce a PDF.")
        coeval_manifest = run_coevaluation_pipeline(coeval_folder, args)
        comparison_path = write_comparison_manifest(
            base_folder, baseline_manifest, coeval_manifest
        )
        _save_resume_tokens(coeval_folder)
        print(f"Co-evaluation comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
