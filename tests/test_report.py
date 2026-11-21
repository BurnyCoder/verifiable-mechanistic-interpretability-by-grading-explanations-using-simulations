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


def test_key_results_report_the_target_swap_when_present():
    import json
    fq = tables.RES / "frequencies_round1.json"
    rows = json.loads(fq.read_text())["rows"] if fq.exists() else []
    text = tables.key_results(1)
    if any("hs_vs_paper" in r for r in rows):  # memorisation evidence: same weights, graded against the published model
        assert "target swap" in text and "against the published checkpoint" in text
