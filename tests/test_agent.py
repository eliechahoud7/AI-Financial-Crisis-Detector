"""Tests for the LangGraph workflow (docs/decisions.md, D17).

Graph tests use fake tools, so they need no data, index, or model. Integration
tests run the real read-only tools on the project's frozen outputs (skipped if
absent), confirm that no file changes, and run the whole workflow on altered
copies of those outputs to check that it stops instead of estimating.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from datetime import date
from pathlib import Path

import yaml

from fincrisis.agents import graph as g
from fincrisis.agents import runtime, tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
PATHS = SETTINGS["paths"]

MARKET = {"as_of": "2026-10-23", "snapshot_age_days": 1, "warnings": [], "test_prevalence": 0.21,
          "test_period": {"first_date": "2015-01-02", "last_date": "2026-09-25"}, "horizon_sessions": 20,
          "label_definition": "a 5% drawdown within 20 sessions",
          "primary": {"name": "rule_x", "kind": "rule", "score": 0.08, "threshold": 0.09, "elevated": False,
                      "test_average_precision": 0.36, "test_roc_auc": 0.68, "test_precision_at_threshold": 0.33,
                      "test_recall_at_threshold": 0.68, "historical_percentile": 0.46, "history_rows": 100},
          "secondary": None}
DISTRESS = {"test_rows": 10, "test_label_1": 2, "label_caveat": "Labels unverified.",
            "families": {"xgboost": {"config": "x", "average_precision": 0.5, "average_precision_95ci": [0.4, 0.6],
                                     "roc_auc": 0.9}}}
ANSWERED = {"status": "answered", "answer": "The target is defined in D2 [1].", "warnings": [],
            "citations": [{"marker": 1, "source": "docs/decisions.md", "section": "D2"}]}
INSUFFICIENT = {"status": "insufficient_evidence", "answer": "Not enough evidence.", "warnings": [], "citations": []}
MARKET_Q = "What is the current market stress level?"
DISTRESS_Q = "How well do the bankruptcy models perform?"


class Recorder:
    """Fake tools that record their calls; any of them can be made to fail."""

    def __init__(self, rag=ANSWERED, market=MARKET, market_error=None, distress_error=None, rag_error=None):
        self.calls, self.rag, self.market = [], rag, market
        self.market_error, self.distress_error, self.rag_error = market_error, distress_error, rag_error

    def tools(self):
        def assess():
            self.calls.append("market")
            if self.market_error:
                raise self.market_error
            return self.market

        def distress():
            self.calls.append("distress")
            if self.distress_error:
                raise self.distress_error
            return DISTRESS

        def rag_answer(question):
            self.calls.append(("rag", question))
            if self.rag_error:
                raise self.rag_error
            return self.rag

        def context_answer(branch):
            self.calls.append(("context", branch))
            if self.rag_error:
                raise self.rag_error
            return self.rag
        return g.AgentTools(assess, distress, rag_answer, context_answer)


def run(question, recorder):
    return g.build_graph(recorder.tools()).invoke({"question": question})


class TestRouting(unittest.TestCase):
    def test_routes(self):
        cases = {
            MARKET_Q: "market",
            "Is the S&P 500 signal elevated today?": "market",
            "Should I sell my stocks now?": "advice",
            "Is it a good time to buy index funds?": "advice",
            "Do you recommend investing in bank shares?": "advice",
            "Should I sell now that the market stress signal is elevated?": "advice",  # advice wins over market
            DISTRESS_Q: "distress",
            "What are the company distress model results?": "distress",
            "How is the market-stress target defined?": "documents",
            "Which VIX value is used for date t?": "documents",
            "Do we know what the bankruptcy labels mean?": "documents",
            "Who won the 2018 FIFA World Cup?": "documents",
        }
        for question, expected in cases.items():
            self.assertEqual(g.route(question), expected, question)


class TestGraphBranches(unittest.TestCase):
    def test_secondary_fitted_model_gets_no_threshold_verdict(self):
        """Regression (D35): the secondary model's frozen threshold is not transferable, so no signal is claimed."""
        secondary = {"name": "xgboost", "kind": "fit", "score": 0.144, "threshold": 0.065, "elevated": True}
        text = g.market_text({**MARKET, "secondary": secondary})
        [line] = [l for l in text.splitlines() if "`xgboost`" in l]
        self.assertIn("score 0.144", line)
        self.assertNotRegex(line.lower(), r"threshold \d|\belevated\b")
        self.assertIn("no decision threshold", line)
        primary = next(l for l in text.splitlines() if "Primary model" in l)
        self.assertIn("vs frozen threshold 0.0900 -> not elevated", primary)  # the rule's threshold stays valid

    def test_market_branch_scores_then_grounds_with_documents(self):
        r = Recorder()
        state = run(MARKET_Q, r)
        self.assertEqual(state["trace"], ["route_request", "assess_market", "retrieve_context", "compose_answer"])
        self.assertEqual(r.calls, ["market", ("context", "market")])
        self.assertIn("as of 2026-10-23", state["answer"])
        self.assertIn("not elevated", state["answer"])
        self.assertIn("From the project documents:", state["answer"])
        self.assertIn(g.DISCLAIMER, state["answer"])
        self.assertEqual(state["citations"], ANSWERED["citations"])

    def test_market_branch_keeps_signal_when_documents_have_no_evidence(self):
        state = run(MARKET_Q, Recorder(rag=INSUFFICIENT))
        self.assertEqual(state["trace"][-1], "compose_answer")
        self.assertIn("not elevated", state["answer"])
        self.assertIn("No supporting passage", state["answer"])
        self.assertEqual(state["citations"], [])

    def test_market_branch_keeps_signal_when_retrieval_fails(self):
        state = run(MARKET_Q, Recorder(rag_error=RuntimeError("index is stale")))
        self.assertEqual(state["trace"][-1], "compose_answer")
        self.assertIn("not elevated", state["answer"])
        self.assertIn("Document context is unavailable", state["answer"])
        self.assertIn("document retrieval unavailable: index is stale", state["warnings"])
        self.assertNotIn("error", state)

    def test_market_error_branch_stops_without_retrieval(self):
        r = Recorder(market_error=tools.ToolError("selection.json differs"))
        state = run(MARKET_Q, r)
        self.assertEqual(state["trace"], ["route_request", "assess_market", "report_error"])
        self.assertIn("selection.json differs", state["answer"])
        self.assertIn("Nothing was estimated", state["answer"])
        self.assertNotIn("elevated", state["answer"])
        self.assertEqual((r.calls, state["citations"]), (["market"], []))

    def test_stale_snapshot_warning_is_propagated(self):
        state = run("Is the market signal elevated now?", Recorder(market={**MARKET, "warnings": ["snapshot is 30 days old"]}))
        self.assertIn("snapshot is 30 days old", state["warnings"])

    def test_distress_branch(self):
        r = Recorder()
        state = run(DISTRESS_Q, r)
        self.assertEqual(state["trace"], ["route_request", "summarize_distress", "retrieve_context", "compose_answer"])
        self.assertEqual(r.calls, ["distress", ("context", "distress")])
        self.assertIn("average precision 0.500 [0.400, 0.600]", state["answer"])
        self.assertIn("Labels unverified.", state["answer"])

    def test_distress_failure_is_reported_not_estimated(self):
        state = run(DISTRESS_Q, Recorder(distress_error=tools.ToolError("xgboost.joblib does not match")))
        self.assertEqual(state["trace"][-1], "compose_answer")
        self.assertIn("Company-distress results are unavailable: xgboost.joblib does not match", state["answer"])
        self.assertNotIn("average precision", state["answer"])
        self.assertTrue(any("distress results unavailable" in w for w in state["warnings"]))

    def test_document_branch_answers_or_abstains(self):
        r = Recorder()
        state = run("How is the market-stress target defined?", r)
        self.assertEqual(state["trace"], ["route_request", "retrieve_context", "compose_answer"])
        self.assertEqual(r.calls, [("rag", "How is the market-stress target defined?")])
        self.assertTrue(state["answer"].startswith(ANSWERED["answer"]))
        state = run("Who won the 2018 FIFA World Cup?", Recorder(rag=INSUFFICIENT))
        self.assertEqual(state["trace"], ["route_request", "retrieve_context", "abstain"])
        self.assertEqual((state["answer"], state["citations"]), (INSUFFICIENT["answer"], []))

    def test_document_retrieval_failure_reports_error(self):
        state = run("How is the market-stress target defined?", Recorder(rag_error=RuntimeError("index missing")))
        self.assertEqual(state["trace"], ["route_request", "retrieve_context", "report_error"])
        self.assertIn("document retrieval unavailable: index missing", state["answer"])
        self.assertEqual(state["citations"], [])

    def test_advice_is_refused_without_calling_any_tool(self):
        r = Recorder()
        state = run("Should I buy S&P 500 futures today?", r)
        self.assertEqual(state["trace"], ["route_request", "refuse_advice"])
        self.assertEqual(state["answer"], g.REFUSAL)
        self.assertEqual(r.calls, [])


class TestContextAnswer(unittest.TestCase):
    ENTRIES = [{"question": "q1", "source": "a.md", "section": "S1"}, {"question": "q2", "source": "b.md", "section": "S2"},
               {"question": "q3", "source": "c.md", "section": "S3"}]

    def test_one_sentence_per_entry_with_renumbered_citations(self):
        seen = []

        def answer_fn(question, scope):
            seen.append((question, scope))
            if question == "q2":
                return INSUFFICIENT
            return {"status": "answered", "answer": f"Sentence for {question}. [1]", "warnings": [],
                    "citations": [{"marker": 1, "source": scope[0]["source"], "section": scope[0]["section"]}]}
        result = tools.context_answer(self.ENTRIES, answer_fn)
        self.assertEqual(seen[1], ("q2", [{"source": "b.md", "section": "S2"}]))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "Sentence for q1. [1] Sentence for q3. [2]")
        self.assertEqual([(c["marker"], c["source"]) for c in result["citations"]], [(1, "a.md"), (2, "c.md")])

    def test_no_entry_answered_is_insufficient_evidence(self):
        result = tools.context_answer(self.ENTRIES, lambda q, s: {**INSUFFICIENT, "warnings": ["w"]})
        self.assertEqual((result["status"], result["citations"]), ("insufficient_evidence", []))
        self.assertEqual(result["warnings"], ["w", "w", "w"])


def tree_hashes(*dirs) -> dict:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


MARKET_DIRS = [Path(PATHS["raw_data"]) / "market", Path(PATHS["data_reports"]) / "market_stress" / "models",
               Path(PATHS["models"]) / "market_stress", Path(PATHS["processed_data"]) / "market_stress"]
DISTRESS_DIRS = [Path(PATHS["models"]) / "taiwan_bankruptcy", Path(PATHS["data_reports"]) / "taiwan_bankruptcy" / "baselines"]
FROZEN = [PROJECT_ROOT / d for d in MARKET_DIRS + DISTRESS_DIRS]


def real_tools(root: Path) -> g.AgentTools:
    """The real market and distress tools on `root`; documents are faked (no index needed)."""
    return g.AgentTools(assess_market=lambda: tools.assess_market(root, PATHS, 7, today=date(2026, 10, 24)),
                        distress_summary=lambda: tools.distress_summary(root, PATHS),
                        rag_answer=lambda q: ANSWERED, context_answer=lambda b: ANSWERED)


@unittest.skipUnless(all(d.exists() for d in FROZEN), "frozen market/distress outputs not present")
class TestRealTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for d in MARKET_DIRS + DISTRESS_DIRS:
            shutil.copytree(PROJECT_ROOT / d, self.root / d)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tools_use_frozen_outputs_read_only(self):
        before = tree_hashes(*FROZEN)
        selection = json.loads((FROZEN[1] / "selection.json").read_text(encoding="utf-8"))
        m = tools.assess_market(PROJECT_ROOT, PATHS, 7, today=date(2026, 10, 24))
        self.assertEqual(m["primary"]["name"], selection["primary_model"]["name"])
        self.assertEqual(m["primary"]["threshold"], selection["thresholds_max_f1"][m["primary"]["name"]])
        self.assertEqual(m["primary"]["elevated"], m["primary"]["score"] >= m["primary"]["threshold"])
        self.assertTrue(0 <= m["primary"]["historical_percentile"] <= 1)
        stale = tools.assess_market(PROJECT_ROOT, PATHS, 7, today=date(2027, 1, 1))
        self.assertTrue(stale["warnings"])
        d = tools.distress_summary(PROJECT_ROOT, PATHS)
        self.assertIn("unverified", d["label_caveat"])
        self.assertEqual(set(d["families"]), {"dummy", "logistic_regression", "xgboost"})
        state = g.build_graph(real_tools(PROJECT_ROOT)).invoke({"question": MARKET_Q})
        self.assertEqual(state["trace"][-1], "compose_answer")
        self.assertEqual(tree_hashes(*FROZEN), before)

    def test_altered_market_model_stops_workflow_before_loading(self):
        (self.root / MARKET_DIRS[2] / "xgboost.joblib").write_bytes(b"not the recorded model")
        with unittest.mock.patch.object(tools.joblib, "load") as load:
            state = g.build_graph(real_tools(self.root)).invoke({"question": MARKET_Q})
            load.assert_not_called()
        self.assertEqual(state["trace"], ["route_request", "assess_market", "report_error"])
        self.assertIn("refusing to load it", state["answer"])

    def test_altered_selection_or_snapshot_stops_workflow(self):
        selection = self.root / MARKET_DIRS[1] / "selection.json"
        original = selection.read_bytes()
        selection.write_bytes(original.replace(b'"primary_model"', b'"primary_model" ', 1))
        state = g.build_graph(real_tools(self.root)).invoke({"question": MARKET_Q})
        self.assertEqual(state["trace"][-1], "report_error")
        self.assertIn("selection.json differs", state["answer"])
        selection.write_bytes(original)
        raw = next((self.root / MARKET_DIRS[0]).glob("*.csv"))
        raw.write_bytes(raw.read_bytes() + b"\n")
        state = g.build_graph(real_tools(self.root)).invoke({"question": MARKET_Q})
        self.assertEqual(state["trace"][-1], "report_error")
        self.assertIn("does not match its recorded checksum", state["answer"])

    def test_altered_distress_model_is_reported_unavailable(self):
        meta = json.loads((self.root / DISTRESS_DIRS[0] / "model_metadata.json").read_text(encoding="utf-8"))
        first = next(iter(meta["models"].values()))["file"]
        (self.root / DISTRESS_DIRS[0] / first).write_bytes(b"tampered")
        state = g.build_graph(real_tools(self.root)).invoke({"question": DISTRESS_Q})
        self.assertIn("Company-distress results are unavailable", state["answer"])
        self.assertNotIn("ROC-AUC", state["answer"])


class TestScript(unittest.TestCase):
    def test_cli_refuses_advice_without_loading_anything(self):
        out = subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "run_crisis_agent.py"),
                              "Should I buy stocks today?", "--json"], capture_output=True, text=True, check=True)
        state = json.loads(out.stdout)
        self.assertEqual((state["route"], state["answer"]), ("advice", g.REFUSAL))


INDEX_DIR = PROJECT_ROOT / SETTINGS["rag"]["index_dir"]


@unittest.skipUnless(INDEX_DIR.exists() and all(d.exists() for d in FROZEN), "RAG index or frozen outputs not present")
class TestRealContext(unittest.TestCase):
    def test_context_cites_only_configured_sections(self):
        agent_tools, _ = runtime.make_tools(SETTINGS, use_llm=False)
        for branch, entries in SETTINGS["agent"]["context"].items():
            result = agent_tools.context_answer(branch)
            self.assertEqual(result["status"], "answered", branch)
            self.assertEqual(len(result["citations"]), len(entries), branch)  # every entry found evidence
            for c in result["citations"]:
                self.assertTrue(any(c["source"] == e["source"] and e["section"] in c["section"] for e in entries), c)
            markers = [int(m) for m in g.re.findall(r"\[(\d+)\]", result["answer"])]
            self.assertEqual(markers, list(range(1, len(entries) + 1)))
        market = agent_tools.context_answer("market")["answer"]
        self.assertIn("at least 5% below its highest close", market)  # the D2 target definition
        self.assertTrue(states_the_selection_rule(cited_sentence(agent_tools.context_answer("market"), "D13.")))

    def test_market_explanation_states_how_the_primary_model_is_selected(self):
        """Regression (D30): the D13 entry quoted the secondary-metric rule after its passage lost its lead-in."""
        agent_tools, _ = runtime.make_tools(SETTINGS, use_llm=False)
        sentence = cited_sentence(agent_tools.context_answer("market"), "D13.")
        self.assertTrue(states_the_selection_rule(sentence), sentence)
        self.assertNotRegex(sentence.lower(), r"\bsecondary\b")
        state = g.build_graph(agent_tools).invoke({"question": "What is the current market stress level?"})
        self.assertIn(sentence, state["answer"])  # the workflow answer carries the same explanation


def cited_sentence(result: dict, section: str) -> str:
    """The sentence of a context answer that cites the passage from `section`."""
    [marker] = [c["marker"] for c in result["citations"] if section in c["section"]]
    sentences = {int(n): s.strip() for s, n in g.re.findall(r"(.*?)\s*\[(\d+)\]", result["answer"])}
    return sentences[marker]


def states_the_selection_rule(sentence: str) -> bool:
    """Says that the primary model is the one with the highest pooled walk-forward average precision (D13)."""
    text = sentence.lower()
    return all(g.re.search(p, text) for p in (r"\bprimary\b", r"\bhighest\b", r"average precision", r"walk-forward|pooled"))


if __name__ == "__main__":
    unittest.main()
