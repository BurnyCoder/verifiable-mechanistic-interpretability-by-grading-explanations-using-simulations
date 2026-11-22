"""Report generator: marked blocks are replaced in place, unmarked text is untouched, and tables render from results/.

Global context: `vmi analyse` must regenerate every number in the documents from results/ (no hand-typed numbers).
"""

import json

from vmi.analysis import report, tables


def test_inject_all_replaces_only_marked_blocks(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("intro\n<!-- BEGIN a -->\nold\n<!-- END a -->\nmiddle\n<!-- BEGIN b -->\nx\n<!-- END b -->\n", encoding="utf-8")
    filled = report.inject_all(doc, {"a": "| new |", "c": "unused"})
    text = doc.read_text(encoding="utf-8")
    assert filled == ["a"]
    assert "| new |" in text and "old" not in text
    assert "<!-- BEGIN b -->\nx\n<!-- END b -->" in text and text.startswith("intro\n")


def test_inject_all_is_idempotent(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("<!-- BEGIN a -->\n<!-- END a -->\n", encoding="utf-8")
    report.inject_all(doc, {"a": "row"})
    once = doc.read_text(encoding="utf-8")
    report.inject_all(doc, {"a": "row"})
    assert doc.read_text(encoding="utf-8") == once


def test_tables_render_from_committed_results():
    for text in (tables.scoreboard(1), tables.baselines_table(), tables.controls_table(), tables.cost_table(), tables.runs_table(1)):
        assert isinstance(text, str) and text  # never raises on default-temperature (Claude) runs
    assert tables._t(None) == "default" and tables._t(0.6) == "0.6"


def test_h5_h6_marked_not_evaluable_when_local_ablation_incomplete():
    from vmi.analysis import hypotheses
    H = hypotheses.evaluate(1)["hypotheses"]
    local = [r for r in hypotheses.load_runs(1) if r["model"] in hypotheses.LOCAL_MODELS]
    complete = all(len([r for r in local if r["model"] == m]) >= 4 for m in hypotheses.LOCAL_MODELS)
    if local and not complete:  # deviation 5: stopped early, so no test statistic, only a description
        for k in ("H5", "H6"):
            assert H[k]["verdict"].startswith("not evaluable") and "above chance" in H[k]["label"]


def test_predictions_cover_every_hypothesis_and_match_rule():
    from vmi.analysis import predictions
    assert predictions._match({"a": True, "b": True}) == "yes"
    assert predictions._match({"a": False, "b": False}).startswith("no (a; b")
    assert predictions._match({"a": True, "b": False}) == "partly (b)"
    rows = predictions.prediction_rows(1)
    if rows:  # one row per preregistered hypothesis that has been evaluated, each with a prediction and an observation
        assert {r["id"] for r in rows} <= set(predictions.PREDICTED) and all(r["observed"] and r["matches"] for r in rows)
        assert "| H1 |" in tables.predictions_table(1)


def test_ablations_table_lists_the_unablated_condition_first():
    text = tables.ablations_table(1)
    lines = text.splitlines()
    assert lines[0].startswith("| Change |") and len(lines) >= 2
    if len(lines) > 2:  # the main condition comes first so every variant reads against it
        assert lines[2].startswith("| none (main condition) | E3 |")


def test_highlights_show_both_what_works_and_what_does_not():
    text = tables.highlights_table(1)
    if "| ✅ |" in text or "| ❌ |" in text:  # results exist: both kinds of example must be shown, with numbers
        assert "| ✅ |" in text and "| ❌ |" in text and "HS " in text


def test_agent_examples_quote_whole_sentences_and_parse_both_log_formats():
    from vmi.analysis import agent_examples as ae
    assert ae.first_sentence("It passes. Then more.") == "It passes."
    long = "word " * 80
    assert ae.first_sentence(long, 50).endswith(" …") and len(ae.first_sentence(long, 50)) <= 52
    claude = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "input": {"command": "python3 selftest.py"}}]}},
              {"type": "user", "message": {"content": [{"type": "tool_result", "content": "accuracy on all 12769 inputs: 1.0"}]}},
              {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done. It works."}]}},
              {"type": "result", "total_cost_usd": 0.01}]
    calls, results, final, cost, _, says = ae._claude_events(claude)
    assert calls == ["python3 selftest.py"] and "accuracy" in results[0] and final == "Done. It works." and cost == 0.01
    assert says == [(1, "Done. It works.")]  # narration keeps its position among the tool calls
    harness = [{"type": "tool_use", "part": {"state": {"input": {"command": "ls"}, "output": "x"}}},
               {"type": "error", "part": {"error": {"type": "output-limit"}}},
               {"type": "text", "part": {"text": "[harness] exit status: TimeExceeded"}},
               {"type": "text", "part": {"text": "[harness] prompt: do the task"}},
               {"type": "text", "part": {"text": "THOUGHT: check the files.\n\n```bash\nls\n```"}}]
    calls, _, _, _, errors, says = ae._harness_events(harness)
    assert calls == ["ls"] and errors == ["output-limit", "exit status: TimeExceeded"]  # the prompt is not an error
    assert says == [(1, "check the files.")]  # the model's words without its command block


def test_verdict_rules_follow_preregistration_sections_5_and_6():
    from vmi.analysis import stats as st
    # §6: 3 vs 3 one-sided Mann-Whitney cannot go below 1/20, so it cannot decide at alpha = 0.05 even with full separation
    assert st.min_p_mann_whitney(3, 3) == 1 / 20
    assert st.verdict(0.05, True, st.min_p_mann_whitney(3, 3)) == "cannot tell at this N"
    assert st.verdict(0.001, True, st.min_p_mann_whitney(3, 9)) == "supported"  # 3 vs 9: smallest p = 1/220
    assert st.verdict(0.3, True, st.min_p_mann_whitney(3, 9)) == "not supported"
    from vmi.analysis import hypotheses
    H = hypotheses.evaluate(2)["hypotheses"]
    if "H1" in H:  # §5: HS and KL_out are tested separately, both reported
        assert set(H["H1"]["verdicts"]) == {"HS", "KL_out"}
    if "H2" in H and H["H2"]["N"] == 3:
        assert set(H["H2"]["verdicts"].values()) == {"cannot tell at this N"}


def test_glossary_explains_labels_once_per_unit_and_leaves_code_and_links_alone():
    from vmi.analysis.glossary import gloss
    assert gloss("HS on T0 is 0.9; HS again") == ("HS (hidden-state similarity, 1 = same internals) on T0 (main target "
                                                   "network) is 0.9; HS again")
    assert gloss("0.05 (E1, algorithm only)") == "0.05 (E1, algorithm only)"  # meaning already given
    assert gloss("hidden-state similarity (HS)") == "hidden-state similarity (HS)"  # meaning given just before
    assert gloss("E3>W-freq and E3/E4") == "E3 (+ per-frequency details) > W-freq (wrong frequencies) and E3 and E4 (+ per-neuron table)"
    assert gloss("`KF = E3` and [T0](https://x/T0__E3)") == "`KF = E3` and [T0](https://x/T0__E3)"
    assert gloss("| E3 | 1 |\n| E3 | 2 |") == "| E3 (+ per-frequency details) | 1 |\n| E3 (+ per-frequency details) | 2 |"  # per row
    assert gloss("T2 (0.274)") == "T2 (target from another seed) (0.274)"  # a number in parentheses is not a meaning
    s = "E0–E4 and ## x\n## Heading HS\n```\nHS\n```"
    assert gloss(s) == s.replace("E0–E4 and ## x", "E0–E4 and ## x")  # ranges, headings and code fences unchanged
    t = "Mixed: E2 then T-new, KL_out 0.1"
    assert gloss(gloss(t)) == gloss(t)  # idempotent


def test_agent_example_explanations_name_the_cause():
    from vmi.analysis import agent_examples as ae
    assert ae._freqs("KF = [2, 3, 23, 37, 39]  # key frequencies") == [2, 3, 23, 37, 39]
    assert ae._freqs("heads = (0, 1)") is None  # too short to be a frequency list
    assert ae.sentence_with("Plan first. I'll pick 14 and 35. Done.", ["14"]) == "I'll pick 14 and 35."
    base = {"accuracy": 1.0, "hs": 0.05, "accs": [1.0], "harness_errors": [], "final": "Done.", "reaction": "",
            "said": [], "compiler_hs": 0.05, "plan": "No frequencies given. I'll pick 14, 35, 41, 42, 52."}
    memo, memo_short = ae.why({**base, "freq_lines": ["KEY_FREQS = [14, 35, 41, 42, 52]"]}, "T0", "E1")
    assert "from memory" in memo_short and "wrong internals" in memo_short
    assert memo.startswith("Wrong mechanism, right answers") and "published paper's" in memo and "I'll pick" in memo
    ok, _ = ae.why({**base, "hs": 0.94, "compiler_hs": 0.93, "freq_lines": ["KF = [2, 3, 23, 37, 39]"]}, "T0", "E3")
    assert ok.startswith("Right:") and "T0's own" in ok
    dead, _ = ae.why({**base, "accuracy": 0.0088, "accs": [0.0088], "harness_errors": ["output-limit", "exit status: TimeExceeded"],
                   "freq_lines": []}, "T0", "E3")
    assert dead.startswith("Wrong: it never built a working network") and "output limit" in dead


def test_diagnosed_failure_explains_the_bug_with_stored_numbers(tmp_path, monkeypatch):
    from vmi.analysis import agent_examples as ae
    from vmi.analysis import diagnose
    rid = "fake_run"
    (tmp_path / f"{rid}.json").write_text(json.dumps({
        "run_id": rid, "cause": "a layout bug", "short": "a layout bug", "blamed": ["ReLU"], "kept": ["faithful"],
        "owner": "The slip is the agent's own",
        "replicate_check": {"replicates": [{"run_id": rid, "accuracy": 0.56, "identical": True},
                                           {"run_id": "r1", "accuracy": 0.9994, "identical": True},
                                           {"run_id": "r2", "accuracy": 0.9986, "identical": True}],
                            "identical_runs": 3, "failures": 1, "failure_rate_ci95": [0.0084, 0.9057],
                            "served": ["claude-haiku-5-5"], "claude_code": "2.1.330"},
        "variants": [{"name": "as submitted", "accuracy": 0.56, "hs": 0.92, "kl_out": 2.0},
                     {"name": "head 3 moved to free dimensions", "accuracy": 1.0, "hs": 0.92, "kl_out": 0.0}]}))
    monkeypatch.setattr(diagnose, "OUT", tmp_path)
    x = {"run_id": rid, "tool_calls": 22, "minutes": 6.4, "accuracy": 0.56, "hs": 0.92, "accs": [0.68, 0.78, 0.56], "harness_errors": [], "final": "",
         "reaction": "accuracy is only 0.68.", "plan": "", "compiler_hs": 0.93, "freq_lines": ["F = [2, 3, 23, 37, 39]"],
         "said": ["ReLU neurons fit badly.", "I'll keep the faithful setting."]}
    long, short = ae.why(x, "T0", "E3")
    assert "a layout bug" in long and "1.0000" in long and "ReLU neurons fit badly." in long and "faithful" in long
    assert "The slip is the agent's own." in long and "22 tool calls and 6.4 minutes" in long
    assert "1 of 3 identical runs" in long and "1%–91%" in long and "0.9994 and 0.9986" in long
    assert short.endswith("(bug in 1 of 3 identical runs)")
    assert "fixing it gives accuracy 1.0000" in short


def test_key_results_report_the_target_swap_when_present():
    import json
    fq = tables.RES / "frequencies_round1.json"
    rows = json.loads(fq.read_text())["rows"] if fq.exists() else []
    text = tables.key_results(1)
    if any("hs_vs_paper" in r for r in rows):  # memorisation evidence: same weights, graded against the published model
        assert "target swap" in text and "against the published checkpoint" in text
