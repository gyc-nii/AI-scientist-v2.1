import hashlib
import json
import os
import os.path as osp
import shutil
from pathlib import Path

import yaml

from ai_scientist.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)
from ai_scientist.perform_icbinb_writeup import (
    compile_latex,
    gather_citations,
    perform_writeup as perform_icbinb_writeup,
)
from ai_scientist.perform_llm_review import load_paper, perform_review
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.perform_writeup import perform_writeup as perform_normal_writeup
from ai_scientist.treesearch.bfts_utils import (
    edit_bfts_config_file,
    idea_to_markdown,
)
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)

SUMMARY_NAMES = (
    "draft_summary.json",
    "baseline_summary.json",
    "research_summary.json",
    "ablation_summary.json",
)

EXPERIMENT_REVIEW_SYSTEM = """You are an independent senior ML experimental reviewer.
You are not the experiment agent and have no access to its conversation history. Assess the
provided topic, experiment summaries, stage reports, and artifact inventory. Keep the topic and
central hypothesis fixed. Recommend only experiments that correct a concrete weakness, test a
claim, or materially improve scientific completeness. Do not propose cosmetic topic changes.
Return valid JSON in a ```json block and no other machine-readable object."""

EXPERIMENT_REVIEW_PROMPT = """Review the completed tree-search evidence before paper writing.
The context contains the fixed research topic, summaries of the selected experiments and
ablations, stage-level findings, prior co-evaluation evidence if any, and an artifact inventory.

Return this exact JSON shape:
```json
{{
  "assessment": "specific assessment of the current evidence",
  "experiment_strengths": ["..."],
  "claim_risks": ["..."],
  "required_experiments": [
    {{
      "name": "short experiment name",
      "rationale": "which weakness or claim it addresses",
      "design": "actionable design using the same topic and compatible data",
      "success_criteria": "what result would resolve the concern",
      "priority": "high|medium|low"
    }}
  ],
  "optional_experiments": ["..."],
  "ready_for_writeup": false
}}
```

Use an empty required_experiments list only when the evidence is already sufficient.

EXPERIMENT CONTEXT:
{context}
"""

PDF_REVIEW_SYSTEM = """You are an independent senior reviewer of an ML paper. You did not
participate in the experiments or writing. Keep the paper's topic and hypothesis fixed. Separate
feedback that requires new empirical evidence from feedback that can be resolved by rewriting.
Never ask the writer to invent results. Return valid JSON in a ```json block."""

PDF_REVIEW_PROMPT = """Review this completed PDF for one actionable revision round.

Return this exact JSON shape:
```json
{{
  "assessment": "specific overall assessment",
  "required_experiments": [
    {{
      "name": "short experiment name",
      "rationale": "paper claim or weakness it addresses",
      "design": "actionable experiment design",
      "success_criteria": "evidence needed to resolve the concern",
      "priority": "high|medium|low"
    }}
  ],
  "writing_feedback": [
    {{
      "section": "paper section",
      "issue": "specific writing or presentation issue",
      "revision": "concrete correction using only available evidence"
    }}
  ],
  "claim_corrections": ["unsupported or overstated claim to correct"],
  "estimated_overall_score_if_unchanged": 1
}}
```

Use required_experiments only for issues that genuinely need new evidence. Put clarity,
organization, qualification, citation, and presentation fixes in writing_feedback.

PAPER:
{paper}
"""


def _read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path, data):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _canonical_hash(data):
    payload = json.dumps(
        data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _initial_experiment_hash(base_folder):
    summaries = {}
    summary_dir = osp.join(base_folder, "logs", "0-run")
    for name in SUMMARY_NAMES:
        summaries[name] = _read_json(osp.join(summary_dir, name), {})
    return _canonical_hash(summaries)


def _normalize_review(review, review_type):
    review = review if isinstance(review, dict) else {}
    review.setdefault("assessment", "Reviewer returned no assessment.")
    review.setdefault("required_experiments", [])
    if not isinstance(review["required_experiments"], list):
        review["required_experiments"] = []
    if review_type == "experiment":
        review.setdefault("experiment_strengths", [])
        review.setdefault("claim_risks", [])
        review.setdefault("optional_experiments", [])
        review.setdefault("ready_for_writeup", not review["required_experiments"])
    else:
        review.setdefault("writing_feedback", [])
        review.setdefault("claim_corrections", [])
        review.setdefault("estimated_overall_score_if_unchanged", None)
    return review


def _call_json_reviewer(system_message, prompt, model, review_type, retries=3):
    last_response = ""
    for attempt in range(retries):
        client, client_model = create_client(model)
        request = prompt
        if attempt:
            request += (
                "\n\nYour previous response could not be parsed. Return only the requested "
                "JSON inside one ```json block."
            )
        last_response, _ = get_response_from_llm(
            prompt=request,
            client=client,
            model=client_model,
            system_message=system_message,
            print_debug=False,
            temperature=0.2,
        )
        parsed = extract_json_between_markers(last_response)
        if isinstance(parsed, dict):
            return _normalize_review(parsed, review_type)
    raise ValueError(
        f"Could not parse co-evaluation reviewer response: {last_response}"
    )


def _artifact_inventory(base_folder, limit=500):
    result_dir = Path(base_folder) / "experiment_results"
    if not result_dir.exists():
        result_dir = Path(base_folder) / "logs" / "0-run" / "experiment_results"
    inventory = []
    if result_dir.exists():
        for path in sorted(p for p in result_dir.rglob("*") if p.is_file()):
            inventory.append(
                {
                    "path": str(path.relative_to(base_folder)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                }
            )
            if len(inventory) >= limit:
                inventory.append({"truncated": True})
                break
    return inventory


def _collect_experiment_context(base_folder):
    summary_dir = Path(base_folder) / "logs" / "0-run"
    summaries = {name: _read_json(summary_dir / name, {}) for name in SUMMARY_NAMES}
    stage_progress = []
    for path in sorted(summary_dir.glob("stage_*/notes/stage_progress.json")):
        stage_progress.append(_read_json(path, {}))

    prior_rounds = []
    for path in sorted(
        (Path(base_folder) / "coevaluation").glob("round_*/summary_bundle.json")
    ):
        prior_rounds.append(_read_json(path, {}))

    return {
        "fixed_topic": _read_json(osp.join(base_folder, "idea.json"), {}),
        "initial_tree_search_summaries": summaries,
        "stage_progress": stage_progress,
        "prior_coevaluation_rounds": prior_rounds,
        "artifact_inventory": _artifact_inventory(base_folder),
    }


def _review_experiments(base_folder, model):
    context = json.dumps(
        _collect_experiment_context(base_folder), indent=2, ensure_ascii=False
    )
    review = _call_json_reviewer(
        EXPERIMENT_REVIEW_SYSTEM,
        EXPERIMENT_REVIEW_PROMPT.format(context=context),
        model,
        "experiment",
    )
    path = osp.join(base_folder, "coevaluation", "pre_writeup_review.json")
    _write_json(path, review)
    return review


def _review_pdf(pdf_path, model):
    paper = load_paper(pdf_path)
    review = _call_json_reviewer(
        PDF_REVIEW_SYSTEM,
        PDF_REVIEW_PROMPT.format(paper=paper),
        model,
        "pdf",
    )
    return review


def _as_text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def _research_code_from_summary(summary_path):
    summary = _read_json(summary_path, {})
    if not isinstance(summary, dict):
        return None
    best = summary.get("best node", {})
    return best.get("code") if isinstance(best, dict) else None


def _find_seed_code(base_folder):
    round_paths = sorted(
        (Path(base_folder) / "coevaluation").glob(
            "round_*/logs/0-run/research_summary.json"
        ),
        reverse=True,
    )
    candidates = round_paths + [
        Path(base_folder) / "logs" / "0-run" / "research_summary.json"
    ]
    for path in candidates:
        code = _research_code_from_summary(path)
        if code:
            return code

    solutions = sorted(
        (Path(base_folder) / "logs" / "0-run").glob("stage_3_*/best_solution_*.py"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if solutions:
        return solutions[0].read_text(encoding="utf-8")
    return None


def _rewrite_result_paths(value, round_number):
    if isinstance(value, dict):
        return {
            key: _rewrite_result_paths(item, round_number)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_result_paths(item, round_number) for item in value]
    if isinstance(value, str):
        marker = "experiment_results/"
        if marker in value:
            suffix = value.split(marker, 1)[1]
            return f"experiment_results/coevaluation_round_{round_number}/{suffix}"
    return value


def _append_evidence(base_folder, heading, payload):
    # Both writeup implementations prefer research_idea.md when it exists.
    # Put reviewer feedback in that same source so it cannot be hidden by the
    # fallback order in load_idea_text().
    idea_md = osp.join(base_folder, "research_idea.md")
    if not osp.exists(idea_md):
        idea_md = osp.join(base_folder, "idea.md")
    with open(idea_md, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {heading}\n\n")
        f.write(
            "The topic and hypothesis remain unchanged. The following is reviewer feedback "
            "or additional empirical evidence that must be addressed without inventing results.\n\n"
        )
        f.write("```json\n")
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n```\n")


def _configure_followup(config_path, steps):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    steps = max(1, int(steps))
    config["agent"]["steps"] = steps
    for stage_number in range(1, 5):
        config["agent"]["stages"][f"stage{stage_number}_max_iters"] = steps
    config["agent"]["num_workers"] = min(
        config["agent"].get("num_workers", steps), steps
    )
    config["agent"]["multi_seed_eval"]["num_seeds"] = min(
        config["agent"]["multi_seed_eval"].get("num_seeds", steps), steps
    )
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _run_followup_experiments(base_folder, review, round_number, experiment_steps):
    required = review.get("required_experiments", [])
    if not required:
        return None

    seed_code = _find_seed_code(base_folder)
    if not seed_code:
        raise FileNotFoundError(
            "Could not find the research-stage best solution for co-evaluation follow-up."
        )

    round_dir = osp.join(base_folder, "coevaluation", f"round_{round_number}")
    if osp.exists(round_dir):
        raise FileExistsError(f"Co-evaluation round already exists: {round_dir}")
    os.makedirs(round_dir)

    idea = _read_json(osp.join(base_folder, "idea.json"), {})
    idea["Experiments"] = (
        _as_text(idea.get("Experiments", ""))
        + "\n\nIndependent co-evaluation required experiments:\n"
        + _as_text(required)
    )
    idea["Risk Factors and Limitations"] = (
        _as_text(idea.get("Risk Factors and Limitations", ""))
        + "\n\nCo-evaluation claim risks to resolve:\n"
        + _as_text(review.get("claim_risks", review.get("claim_corrections", [])))
    )
    idea["Code"] = seed_code

    idea_json = osp.join(round_dir, "idea.json")
    idea_md = osp.join(round_dir, "idea.md")
    _write_json(idea_json, idea)
    idea_to_markdown(idea, idea_md, None)

    config_path = edit_bfts_config_file(
        osp.join(base_folder, "bfts_config.yaml"), round_dir, idea_json
    )
    _configure_followup(config_path, experiment_steps)
    perform_experiments_bfts(config_path)

    run_dir = osp.join(round_dir, "logs", "0-run")
    summaries = {
        name: _rewrite_result_paths(
            _read_json(osp.join(run_dir, name), {}), round_number
        )
        for name in SUMMARY_NAMES
    }
    bundle = {
        "round": round_number,
        "review_request": review,
        "followup_summaries": summaries,
    }
    _write_json(osp.join(round_dir, "summary_bundle.json"), bundle)

    result_src = osp.join(run_dir, "experiment_results")
    result_dst = osp.join(
        base_folder,
        "experiment_results",
        f"coevaluation_round_{round_number}",
    )
    if osp.isdir(result_src):
        shutil.copytree(result_src, result_dst)

    _append_evidence(
        base_folder, f"Co-Evaluation Experiment Round {round_number}", bundle
    )
    return bundle


def _find_pdf(base_folder):
    pdfs = [
        path
        for path in Path(base_folder).glob("*.pdf")
        if path.name not in {"final_baseline.pdf", "final_coevaluation.pdf"}
    ]
    final_reflections = [p for p in pdfs if "final" in p.name.lower()]
    if final_reflections:
        return str(max(final_reflections, key=lambda p: p.stat().st_mtime))
    reflections = [p for p in pdfs if "reflection" in p.name.lower()]
    if reflections:
        return str(max(reflections, key=lambda p: p.stat().st_mtime))
    if pdfs:
        return str(max(pdfs, key=lambda p: p.stat().st_mtime))
    return None


def _write_paper(base_folder, args, citations_text=None):
    for _ in range(args.writeup_retries):
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
            return _find_pdf(base_folder)
    return None


def _compile_existing_paper(base_folder):
    latex_folder = osp.join(base_folder, "latex")
    if not osp.isfile(osp.join(latex_folder, "template.tex")):
        raise FileNotFoundError("Cannot resume compilation: latex/template.tex is missing.")
    pdf_path = osp.join(
        base_folder,
        f"{osp.basename(base_folder)}_reflection_final_page_limit.pdf",
    )
    print(f"Compiling existing co-evaluation LaTeX: {latex_folder}")
    if not compile_latex(latex_folder, pdf_path, timeout=120):
        raise RuntimeError("Existing co-evaluation LaTeX did not compile cleanly.")
    return pdf_path


def _load_cached_citations(base_folder):
    path = osp.join(base_folder, "cached_citations.bib")
    try:
        with open(path, "r", encoding="utf-8") as f:
            citations = f.read()
    except FileNotFoundError:
        return None
    return citations or None


def _score_pdf(base_folder, pdf_path, model):
    paper_content = load_paper(pdf_path)
    client, client_model = create_client(model)
    review = perform_review(paper_content, client_model, client)
    image_review = perform_imgs_cap_ref_review(client, client_model, pdf_path)
    _write_json(osp.join(base_folder, "review_text.txt"), review)
    _write_json(osp.join(base_folder, "review_img_cap_ref.json"), image_review)
    return review


def _review_score(review):
    return review.get("Overall") if isinstance(review, dict) else None


def create_coevaluation_branch(baseline_folder):
    coeval_folder = baseline_folder + "_coeval"
    if osp.exists(coeval_folder):
        raise FileExistsError(f"Co-evaluation branch already exists: {coeval_folder}")
    shutil.copytree(baseline_folder, coeval_folder)

    idea = _read_json(osp.join(coeval_folder, "idea.json"), {})
    metadata = {
        "branch": "coevaluation",
        "source_branch": osp.abspath(baseline_folder),
        "topic_fingerprint": _canonical_hash(idea),
        "initial_experiment_fingerprint": _initial_experiment_hash(coeval_folder),
    }
    _write_json(osp.join(coeval_folder, "coevaluation", "branch_origin.json"), metadata)
    return coeval_folder


def snapshot_baseline_artifacts(base_folder):
    pdf_path = _find_pdf(base_folder)
    stable_pdf = None
    if pdf_path:
        stable_pdf = osp.join(base_folder, "final_baseline.pdf")
        shutil.copy2(pdf_path, stable_pdf)
    review = _read_json(osp.join(base_folder, "review_text.txt"), {})
    idea = _read_json(osp.join(base_folder, "idea.json"), {})
    manifest = {
        "branch": "baseline",
        "topic_fingerprint": _canonical_hash(idea),
        "initial_experiment_fingerprint": _initial_experiment_hash(base_folder),
        "final_pdf": osp.abspath(stable_pdf) if stable_pdf else None,
        "final_review": osp.abspath(osp.join(base_folder, "review_text.txt")),
        "overall_score": _review_score(review),
    }
    _write_json(osp.join(base_folder, "branch_manifest.json"), manifest)
    return manifest


def run_coevaluation_pipeline(base_folder, args):
    if args.skip_writeup or args.skip_review:
        raise ValueError(
            "--co-evaluation requires both writing and review; do not combine it with "
            "--skip_writeup or --skip_review."
        )

    result_src = osp.join(base_folder, "logs", "0-run", "experiment_results")
    result_dst = osp.join(base_folder, "experiment_results")
    if osp.isdir(result_src) and not osp.exists(result_dst):
        shutil.copytree(result_src, result_dst)

    reviewer_model = args.model_coeval or args.model_review
    resume_from_compile = getattr(args, "resume_from_compile", False)
    if resume_from_compile:
        review_path = osp.join(base_folder, "coevaluation", "pre_writeup_review.json")
        experiment_review = _read_json(review_path)
        if not isinstance(experiment_review, dict):
            raise FileNotFoundError(
                "Cannot resume co-evaluation compilation: pre-writeup review is missing."
            )
        round_one_summary = osp.join(
            base_folder, "coevaluation", "round_1", "summary_bundle.json"
        )
        if experiment_review.get("required_experiments") and not osp.isfile(
            round_one_summary
        ):
            raise RuntimeError(
                "Cannot skip to compilation before required round-1 experiments finish."
            )
        citations_text = _load_cached_citations(base_folder)
        first_pdf = _compile_existing_paper(base_folder)
    else:
        experiment_review = _review_experiments(base_folder, reviewer_model)
        _append_evidence(
            base_folder, "Independent Pre-Writeup Experiment Review", experiment_review
        )
        _run_followup_experiments(
            base_folder,
            experiment_review,
            round_number=1,
            experiment_steps=args.coeval_experiment_steps,
        )

        aggregate_plots(base_folder=base_folder, model=args.model_agg_plots)
        citations_text = None
        if args.writeup_type == "icbinb":
            citations_text = gather_citations(
                base_folder,
                num_cite_rounds=args.num_cite_rounds,
                small_model=args.model_citation,
            )
        first_pdf = _write_paper(base_folder, args, citations_text)
    if not first_pdf:
        raise RuntimeError("Co-evaluation could not produce the pre-review PDF.")

    before_review_pdf = osp.join(base_folder, "coevaluation", "pdf_before_revision.pdf")
    shutil.copy2(first_pdf, before_review_pdf)
    pdf_review = _review_pdf(first_pdf, reviewer_model)
    _write_json(
        osp.join(base_folder, "coevaluation", "pdf_revision_review.json"),
        pdf_review,
    )
    _append_evidence(base_folder, "Independent PDF Revision Review", pdf_review)

    if pdf_review.get("required_experiments"):
        _run_followup_experiments(
            base_folder,
            pdf_review,
            round_number=2,
            experiment_steps=args.coeval_experiment_steps,
        )
        aggregate_plots(base_folder=base_folder, model=args.model_agg_plots)

    final_pdf = _write_paper(base_folder, args, citations_text)
    if not final_pdf:
        raise RuntimeError("Co-evaluation could not produce the final revised PDF.")

    stable_pdf = osp.join(base_folder, "final_coevaluation.pdf")
    shutil.copy2(final_pdf, stable_pdf)
    final_review = _score_pdf(base_folder, stable_pdf, args.model_review)

    origin = _read_json(osp.join(base_folder, "coevaluation", "branch_origin.json"), {})
    manifest = {
        "branch": "coevaluation",
        "topic_fingerprint": origin.get("topic_fingerprint"),
        "initial_experiment_fingerprint": origin.get("initial_experiment_fingerprint"),
        "pre_revision_pdf": osp.abspath(before_review_pdf),
        "final_pdf": osp.abspath(stable_pdf),
        "final_review": osp.abspath(osp.join(base_folder, "review_text.txt")),
        "overall_score": _review_score(final_review),
        "pre_writeup_experiment_review": osp.abspath(
            osp.join(base_folder, "coevaluation", "pre_writeup_review.json")
        ),
        "pdf_revision_review": osp.abspath(
            osp.join(base_folder, "coevaluation", "pdf_revision_review.json")
        ),
    }
    _write_json(osp.join(base_folder, "branch_manifest.json"), manifest)
    return manifest


def write_comparison_manifest(base_folder, baseline_manifest, coeval_manifest):
    comparison = {
        "same_topic": baseline_manifest.get("topic_fingerprint")
        == coeval_manifest.get("topic_fingerprint"),
        "same_initial_experiments": baseline_manifest.get(
            "initial_experiment_fingerprint"
        )
        == coeval_manifest.get("initial_experiment_fingerprint"),
        "baseline": baseline_manifest,
        "coevaluation": coeval_manifest,
    }
    path = osp.join(base_folder, "coevaluation_comparison.json")
    _write_json(path, comparison)
    return path
