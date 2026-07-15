"""Generate a detailed HTML report from eval-report.json."""

import json
import os
from pathlib import Path

_parent = Path(__file__).parent
REPORT_JSON = Path(os.environ.get("EVAL_REPORT_PATH", _parent / "eval-report.json"))
REPORT_HTML = Path(os.environ.get("EVAL_REPORT_HTML", _parent / "eval-report.html"))
BASELINE_JSON = _parent / "eval-baseline.json"

CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1100px; margin: 0 auto; padding: 20px;
  background: #f8f9fa; color: #1a1a1a;
}
h1 { border-bottom: 2px solid #dee2e6; padding-bottom: 10px; }
.metrics { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }
.metric {
  background: white; border: 1px solid #dee2e6;
  border-radius: 8px; padding: 16px 24px; text-align: center;
  min-width: 120px;
}
.metric .value { font-size: 28px; font-weight: 700; }
.metric .label { font-size: 13px; color: #666; margin-top: 4px; }
.metric.pass .value { color: #28a745; }
.metric.warn .value { color: #ffc107; }
.delta { font-size: 13px; font-weight: 600; margin-top: 2px; }
.delta.up { color: #28a745; }
.delta.down { color: #dc3545; }
.delta.flat { color: #6c757d; }
.delta.none { color: #999; font-style: italic; }
.comparison {
  background: white; border: 2px solid #1a1a2e;
  border-radius: 8px; margin: 24px 0; overflow: hidden;
}
.comparison-header {
  padding: 14px 16px; background: #1a1a2e; color: white;
  font-weight: 600; font-size: 15px;
  display: flex; justify-content: space-between; align-items: center;
}
.comparison-header .badge {
  background: #4caf50; color: white; padding: 2px 8px;
  border-radius: 4px; font-size: 12px; font-weight: 600;
}
.comparison table {
  width: 100%; border-collapse: collapse; font-size: 14px;
}
.comparison th {
  padding: 10px 16px; text-align: left; background: #e9ecef;
  font-weight: 600; border-bottom: 2px solid #dee2e6;
}
.comparison td {
  padding: 10px 16px; border-bottom: 1px solid #eee;
}
.comparison tr:last-child td { border-bottom: none; }
.comparison .delta-up { color: #28a745; font-weight: 600; }
.comparison .delta-down { color: #dc3545; font-weight: 600; }
.comparison .delta-flat { color: #6c757d; }
.comparison .summary {
  padding: 12px 16px; background: #f0f7ff; border-top: 1px solid #dee2e6;
  font-size: 13px; color: #333;
}
.query-card {
  background: white; border: 1px solid #dee2e6;
  border-radius: 8px; margin: 20px 0; overflow: hidden;
}
.query-header {
  padding: 14px 16px; background: #e9ecef; font-weight: 600; font-size: 15px;
}
.query-meta {
  padding: 8px 16px; font-size: 13px; color: #555;
  border-bottom: 1px solid #eee;
}
.query-metrics {
  display: flex; gap: 12px; padding: 8px 16px;
  font-size: 12px; color: #666; border-bottom: 1px solid #eee;
  flex-wrap: wrap;
}
.query-metrics span { white-space: nowrap; }
.query-metrics .good { color: #28a745; font-weight: 600; }
.query-metrics .bad { color: #dc3545; font-weight: 600; }
.results-table {
  width: 100%; border-collapse: collapse; font-size: 13px;
}
.results-table th {
  padding: 8px 10px; text-align: left; background: #f8f9fa;
  font-weight: 600; font-size: 12px; border-bottom: 2px solid #dee2e6;
  white-space: nowrap;
}
.results-table td {
  padding: 7px 10px; border-bottom: 1px solid #f0f0f0;
  vertical-align: top;
}
.results-table tr:last-child td { border-bottom: none; }
.results-table .rank-col { text-align: center; font-weight: 700; width: 30px; }
.results-table .before-col { text-align: center; color: #666; width: 40px; }
.results-table .delta-col { text-align: center; width: 40px; font-weight: 600; }
.results-table .delta-col.up { color: #28a745; }
.results-table .delta-col.down { color: #dc3545; }
.results-table .delta-col.flat { color: #999; }
.results-table .source-col {
  font-family: monospace; font-size: 11px; color: #555;
  white-space: nowrap; max-width: 200px; overflow: hidden;
  text-overflow: ellipsis;
}
.results-table .score-col {
  font-family: monospace; font-size: 11px; color: #666;
  white-space: nowrap;
}
.results-table .status-col { width: 60px; }
.results-table .text-col {
  font-size: 12px; color: #444; max-width: 300px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.tag {
  display: inline-block; padding: 1px 6px; border-radius: 4px;
  font-size: 10px; font-weight: 600;
}
.tag.rel { background: #28a745; color: white; }
.tag.not-rel { background: #dc3545; color: white; }
tr.selected-k2 { background: #e8f5e9; border-left: 3px solid #28a745; }
tr.selected-k4 { background: #e0f2f1; border-left: 3px solid #009688; }
tr.selected-k2 td:first-child, tr.selected-k4 td:first-child {
  padding-left: 7px;
}
.selection-legend {
  padding: 6px 16px; font-size: 11px; color: #666;
  background: #f8f9fa; border-top: 1px solid #eee;
  display: flex; gap: 16px; align-items: center;
}
.selection-legend .swatch {
  display: inline-block; width: 12px; height: 12px;
  border-radius: 2px; margin-right: 4px; vertical-align: middle;
}
.guide {
  background: white; border: 1px solid #dee2e6;
  border-radius: 8px; margin: 24px 0; overflow: hidden;
}
.guide-header {
  padding: 14px 16px; background: #f8f9fa; border-bottom: 1px solid #dee2e6;
  font-weight: 600; font-size: 15px; cursor: pointer;
}
.guide-header:hover { background: #e9ecef; }
.guide-body {
  padding: 16px; font-size: 13px; line-height: 1.7; color: #333;
}
.guide-body h3 { font-size: 14px; margin: 16px 0 8px 0; color: #1a1a2e; }
.guide-body h3:first-child { margin-top: 0; }
.guide-body ul { margin: 4px 0 12px 0; padding-left: 20px; }
.guide-body li { margin-bottom: 4px; }
.guide-body code {
  background: #f0f0f0; padding: 1px 5px; border-radius: 3px;
  font-family: monospace; font-size: 12px;
}
"""


def _escape_html(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _load_baseline():
    if not BASELINE_JSON.exists():
        return None
    try:
        data = json.loads(BASELINE_JSON.read_text())
        if "metrics" in data:
            return data["metrics"]
        if "recall_at_2" in data:
            return data
        return None
    except (json.JSONDecodeError, KeyError):
        return None


def _delta_html(current, baseline_val, key):
    if baseline_val is None:
        return '<div class="delta none">no baseline</div>'
    diff = current - baseline_val
    if abs(diff) < 0.005:
        return '<div class="delta flat">=&nbsp;0.00</div>'
    sign = "+" if diff > 0 else ""
    cls = "up" if diff > 0 else "down"
    return f'<div class="delta {cls}">{sign}{diff:.2f}</div>'


def _pipeline_comparison_html(pipeline):
    if not pipeline or "before" not in pipeline or "after" not in pipeline:
        return ""

    before = pipeline["before"]
    after = pipeline["after"]

    improvements = []
    degradations = []
    for key, label in [
        ("recall_at_2", "Recall@2"),
        ("recall_at_4", "Recall@4"),
        ("precision_at_2", "Precision@2"),
        ("precision_at_4", "Precision@4"),
        ("mrr", "MRR"),
    ]:
        b = before.get(key, 0)
        a = after.get(key, 0)
        d = a - b
        if d > 0.01:
            improvements.append(f"{label} +{d:.0%}")
        elif d < -0.01:
            degradations.append(f"{label} {d:.0%}")

    parts = []
    if improvements:
        parts.append(f"Improves: {', '.join(improvements)}.")
    if degradations:
        parts.append(f"Degrades: {', '.join(degradations)}.")
    if not parts:
        parts.append("No significant change.")
    summary_text = " ".join(parts)

    html = (
        '<div class="comparison">\n'
        '  <div class="comparison-header">\n'
        "    <span>Bi-Encoder \u2192 Cross-Encoder Reranking</span>\n"
        '    <span class="badge">RERANK ENABLED</span>\n'
        "  </div>\n"
        "  <table>\n"
        "    <thead><tr>"
        "<th>Metric</th>"
        "<th>Bi-Encoder (before)</th>"
        "<th>+Rerank (after)</th>"
        "<th>Delta</th>"
        "</tr></thead>\n"
        "    <tbody>\n"
    )

    for key, label in [
        ("recall_at_2", "Recall@2"),
        ("recall_at_4", "Recall@4"),
        ("precision_at_2", "Precision@2"),
        ("precision_at_4", "Precision@4"),
        ("mrr", "MRR"),
    ]:
        b = before.get(key, 0)
        a = after.get(key, 0)
        d = a - b
        if abs(d) < 0.005:
            delta_cls, delta_text = "delta-flat", "\u2014"
        elif d > 0:
            delta_cls, delta_text = "delta-up", f"+{d:.2f}"
        else:
            delta_cls, delta_text = "delta-down", f"{d:.2f}"
        html += (
            f"      <tr><td><strong>{label}</strong></td>"
            f"<td>{b:.2f}</td><td>{a:.2f}</td>"
            f'<td class="{delta_cls}">{delta_text}</td></tr>\n'
        )

    html += f"    </tbody>\n  </table>\n  <div class=\"summary\">{summary_text}</div>\n</div>\n\n"
    return html


def _guide_html():
    return """
<div class="guide">
  <div class="guide-header" onclick="this.nextElementSibling.style.display =
    this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
    &#9432; How to read this report
  </div>
  <div class="guide-body" style="display:none">
    <h3>What this report shows</h3>
    <p>For each query, you see the <strong>final reranked results</strong> in one
    table. The "Before" column tells you where each item was ranked by the
    bi-encoder <em>before</em> reranking, so you can see exactly how the
    cross-encoder reordered the results.</p>

    <h3>Selection highlights</h3>
    <ul>
      <li><strong>Green rows</strong> (top 2) &mdash; These are the results
        returned to the user when k=2.</li>
      <li><strong>Teal rows</strong> (rows 3-4) &mdash; Additional results
        included when k=4.</li>
      <li>Rows below position 4 are <em>not selected</em> at either k value.</li>
    </ul>

    <h3>Columns</h3>
    <ul>
      <li><strong>#</strong> &mdash; Final rank after cross-encoder reranking.</li>
      <li><strong>Before</strong> &mdash; Rank in the bi-encoder results.
        &mdash; means the item was not in the bi-encoder top candidates.</li>
      <li><strong>&Delta;</strong> &mdash; Movement: &#8593;N = promoted N
        places, &#8595;N = demoted, &mdash; = unchanged.</li>
      <li><strong>bi:score / ce:score</strong> &mdash; Bi-encoder cosine
        similarity and cross-encoder relevance score.</li>
      <li><strong>REL / NOT REL</strong> &mdash; Whether the chunk is from a
        relevant document (per benchmark labels).</li>
    </ul>

    <h3>Metrics</h3>
    <ul>
      <li><code>Recall@K</code> &mdash; Of all relevant documents, how many
        appear in the top K?</li>
      <li><code>Precision@K</code> &mdash; Of the top K results, what fraction
        are relevant?</li>
      <li><code>MRR</code> &mdash; Position of the first relevant result.</li>
    </ul>
  </div>
</div>
"""


def _unified_results_table(detail, relevant_documents):
    """Build a single unified results table from rerank_detail."""
    if not detail:
        return ""

    reranked_top8 = detail.get("reranked_top8", [])
    rank_changes = detail.get("rank_changes", [])
    bi_top8 = detail.get("bi_encoder_top8", [])

    if not reranked_top8:
        return ""

    # Build lookup: source_file → rank_change info (using after-rank as key)
    rc_by_after = {}
    for rc in rank_changes:
        rc_by_after[rc["after"]] = rc

    # Build lookup: source_file → bi-encoder rank (from bi_top8)
    bi_rank_by_source = {}
    for frag in bi_top8:
        src = frag["source_file"]
        if src not in bi_rank_by_source:
            bi_rank_by_source[src] = frag["rank"]

    html = (
        '<table class="results-table">\n'
        "  <thead><tr>"
        "<th class='rank-col'>#</th>"
        "<th class='before-col'>Before</th>"
        "<th class='delta-col'>&Delta;</th>"
        "<th class='source-col'>Source</th>"
        "<th class='score-col'>bi:score</th>"
        "<th class='score-col'>ce:score</th>"
        "<th class='status-col'>Status</th>"
        "<th class='text-col'>Text preview</th>"
        "</tr></thead>\n"
        "  <tbody>\n"
    )

    for frag in reranked_top8:
        after_rank = frag["rank"]
        source = frag["source_file"]
        is_rel = frag["is_relevant"]

        # Find this item's position before reranking
        rc = rc_by_after.get(after_rank)
        if rc:
            before_rank = rc["before"]
            delta = rc["delta"]
        else:
            # Fallback: check bi_top8
            before_rank = bi_rank_by_source.get(source, None)
            delta = 0

        # Row class based on selection
        if after_rank <= 2:
            row_cls = "selected-k2"
        elif after_rank <= 4:
            row_cls = "selected-k4"
        else:
            row_cls = ""

        # Before column
        if before_rank is not None:
            before_text = str(before_rank)
        else:
            before_text = "\u2014"

        # Delta column
        if before_rank is not None and delta != 0:
            if delta > 0:
                delta_cls = "up"
                delta_text = f"\u2191{delta}"
            else:
                delta_cls = "down"
                delta_text = f"\u2193{abs(delta)}"
        else:
            delta_cls = "flat"
            delta_text = "\u2014"

        # Status tag
        tag_cls = "rel" if is_rel else "not-rel"
        tag_text = "REL" if is_rel else "NOT REL"

        # Text preview
        text = frag.get("text_preview", "")

        html += (
            f'  <tr class="{row_cls}">'
            f"<td class='rank-col'>{after_rank}</td>"
            f"<td class='before-col'>{before_text}</td>"
            f"<td class='delta-col {delta_cls}'>{delta_text}</td>"
            f"<td class='source-col' title='{_escape_html(source)}'>"
            f"{_escape_html(source)}</td>"
            f"<td class='score-col'>{frag['bi_score']:.4f}</td>"
            f"<td class='score-col'>{frag.get('ce_score', 0) or 0:.4f}</td>"
            f"<td class='status-col'><span class='tag {tag_cls}'>{tag_text}</span></td>"
            f"<td class='text-col' title='{_escape_html(text)}'>"
            f"{_escape_html(text)}</td>"
            f"</tr>\n"
        )

    html += "  </tbody>\n</table>\n"

    # Legend
    html += (
        '<div class="selection-legend">'
        '<span><span class="swatch" style="background:#e8f5e9;border:1px solid #28a745"></span>'
        "Top 2 (selected for k=2)</span>"
        '<span><span class="swatch" style="background:#e0f2f1;border:1px solid #009688"></span>'
        "Top 3-4 (selected for k=4)</span>"
        "</div>\n"
    )

    return html


def generate():
    if not REPORT_JSON.exists():
        print(f"No {REPORT_JSON} found \u2014 skipping report generation.")
        return

    data = json.loads(REPORT_JSON.read_text())
    metrics = data["metrics"]
    baseline = _load_baseline()
    rerank_detail_list = data.get("rerank_detail", [])
    rerank_detail_by_query = {d["query"]: d for d in rerank_detail_list}

    def metric_class(key, threshold):
        return "pass" if metrics[key] >= threshold else "warn"

    bl = baseline or {}

    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>Eval Report &mdash; DOC RAG</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n"
        "<h1>DOC RAG &mdash; Eval Report</h1>\n\n"
    )

    # Guide
    html += _guide_html()

    # Pipeline comparison
    html += _pipeline_comparison_html(data.get("pipeline_comparison"))

    # Metric cards
    html += '<div class="metrics">\n'
    for key, label, thresh in [
        ("recall_at_2", "Recall@2", 0.80),
        ("recall_at_4", "Recall@4", 0.85),
        ("precision_at_2", "Precision@2", 0.70),
        ("precision_at_4", "Precision@4", 0.65),
        ("mrr", "MRR", 0.80),
    ]:
        val = metrics.get(key, 0)
        cls = metric_class(key, thresh)
        bl_val = bl.get(key)
        html += (
            f'  <div class="metric {cls}">\n'
            f'    <div class="value">{val:.2f}</div>\n'
            f"    {_delta_html(val, bl_val, key)}\n"
            f'    <div class="label">{label}</div>\n  </div>\n'
        )
    n_ans = metrics.get("n_answerable", 0)
    n_na = metrics.get("n_no_answer", 0)
    html += (
        '  <div class="metric pass">\n'
        f'    <div class="value">{n_ans + n_na}</div>\n'
        '    <div class="label">Queries</div>\n  </div>\n'
        '  <div class="metric pass">\n'
        f'    <div class="value">{n_ans}</div>\n'
        '    <div class="label">Answerable</div>\n  </div>\n'
        '  <div class="metric warn">\n'
        f'    <div class="value">{n_na}</div>\n'
        '    <div class="label">No Answer</div>\n  </div>\n'
        "</div>\n"
    )

    # Per-query results with unified table
    for q in data.get("queries", []):
        query = q["query"]
        relevant_docs = q.get("relevant_documents", [])
        category = q.get("category", "single_passage")

        # Skip _reranked suffixed queries (they're duplicates for metric collection)
        if query.endswith("_reranked"):
            continue
        # Skip (pipeline) suffixed queries too
        if query.endswith(" (pipeline)"):
            continue

        html += '<div class="query-card">\n'
        html += f'  <div class="query-header">&ldquo;{_escape_html(query)}&rdquo;</div>\n'
        html += (
            '  <div class="query-meta">'
            f"Category: {category} | "
            f"Relevant: {', '.join(relevant_docs)}"
            "</div>\n"
        )

        # Per-query metrics
        r2 = q.get("recall_at_2", 0)
        p2 = q.get("precision_at_2", 0)
        r4 = q.get("recall_at_4", 0)
        p4 = q.get("precision_at_4", 0)
        rr = q.get("reciprocal_rank", 0)

        def mcls(val, thresh):
            return "good" if val >= thresh else "bad"

        if category != "no_answer":
            html += (
                '  <div class="query-metrics">'
                f'<span class="{mcls(r2, 0.8)}">R@2={r2:.2f}</span>'
                f'<span class="{mcls(r4, 0.85)}">R@4={r4:.2f}</span>'
                f'<span class="{mcls(p2, 0.5)}">P@2={p2:.2f}</span>'
                f'<span class="{mcls(p4, 0.4)}">P@4={p4:.2f}</span>'
                f'<span class="{mcls(rr, 0.7)}">MRR={rr:.2f}</span>'
                "</div>\n"
            )
        else:
            html += (
                '  <div class="query-metrics">'
                '<span class="bad">No Answer Query (metrics N/A)</span>'
                "</div>\n"
            )

        # Unified results table (from rerank_detail if available)
        detail = rerank_detail_by_query.get(query)
        if detail:
            html += _unified_results_table(detail, relevant_docs)
        else:
            # Fallback: show bi-encoder only results
            html += _bi_encoder_only_table(q)

        html += "</div>\n"

    html += "</body></html>"
    REPORT_HTML.write_text(html)
    (REPORT_HTML.parent / "index.html").write_text(html)
    print(f"Report written to {REPORT_HTML}")


def _bi_encoder_only_table(q):
    """Fallback table when no rerank_detail is available."""
    frags = q.get("retrieved_fragments", [])
    if not frags:
        return '<p style="padding:16px;color:#999">No results</p>\n'

    html = (
        '<table class="results-table">\n'
        "  <thead><tr>"
        "<th class='rank-col'>#</th>"
        "<th class='source-col'>Source</th>"
        "<th class='score-col'>Score</th>"
        "<th class='status-col'>Status</th>"
        "<th class='text-col'>Text preview</th>"
        "</tr></thead>\n"
        "  <tbody>\n"
    )

    for frag in frags:
        is_rel = frag["is_relevant"]
        tag_cls = "rel" if is_rel else "not-rel"
        tag_text = "REL" if is_rel else "NOT REL"
        text = frag["text"][:200]
        src = frag["source_file"]
        row_cls = "selected-k2" if frag["rank"] <= 2 else ""
        html += (
            f'  <tr class="{row_cls}">'
            f"<td class='rank-col'>{frag['rank']}</td>"
            f"<td class='source-col'>{_escape_html(src)}</td>"
            f"<td class='score-col'>{frag['score']:.4f}</td>"
            f"<td class='status-col'><span class='tag {tag_cls}'>{tag_text}</span></td>"
            f"<td class='text-col'>{_escape_html(text)}</td>"
            f"</tr>\n"
        )

    html += "  </tbody>\n</table>\n"
    return html


if __name__ == "__main__":
    generate()
