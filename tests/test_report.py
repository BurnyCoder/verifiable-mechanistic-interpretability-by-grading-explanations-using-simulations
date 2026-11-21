"""Report generator: marked blocks are replaced in place, unmarked text is untouched, and tables render from results/.

Global context: `vmi analyse` must regenerate every number in the documents from results/ (no hand-typed numbers).
"""

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
    calls, results, final, cost, _ = ae._claude_events(claude)
    assert calls == ["python3 selftest.py"] and "accuracy" in results[0] and final == "Done. It works." and cost == 0.01
    harness = [{"type": "tool_use", "part": {"state": {"input": {"command": "ls"}, "output": "x"}}},
               {"type": "error", "part": {"error": {"type": "output-limit"}}},
               {"type": "text", "part": {"text": "[harness] exit status: TimeExceeded"}},
               {"type": "text", "part": {"text": "[harness] prompt: do the task"}}]
    calls, _, _, _, errors = ae._harness_events(harness)
    assert calls == ["ls"] and errors == ["output-limit", "exit status: TimeExceeded"]  # the prompt is not an error


def test_key_results_report_the_target_swap_when_present():
    import json
    fq = tables.RES / "frequencies_round1.json"
    rows = json.loads(fq.read_text())["rows"] if fq.exists() else []
    text = tables.key_results(1)
    if any("hs_vs_paper" in r for r in rows):  # memorisation evidence: same weights, graded against the published model
        assert "target swap" in text and "against the published checkpoint" in text
