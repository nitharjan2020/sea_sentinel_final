import pytest

from sea_sentinel import report


def test_render_fills_placeholders():
    assert report.render("n = {{n}}, m = {{ m }}", {"n": "5", "m": 7}) == "n = 5, m = 7"


def test_render_fails_on_undefined_value():
    with pytest.raises(KeyError, match="missing_value"):
        report.render("{{missing_value}}", {})


def test_template_header_is_replaced(tmp_path):
    template = tmp_path / "t.md"
    template.write_text("<!-- edit me -->\n# Title {{x}}\n", encoding="utf-8")
    out = tmp_path / "r.md"
    original = report.build_values
    report.build_values = lambda ctx: {"x": "1"}
    try:
        report.write({}, template, out)
    finally:
        report.build_values = original
    text = out.read_text(encoding="utf-8")
    assert text.startswith(report.GENERATED_NOTICE)
    assert "edit me" not in text and "# Title 1" in text


def test_every_template_placeholder_is_a_known_key():
    """Catch typos in the template without running the pipeline: every placeholder must appear as a
    quoted key somewhere in report.py."""
    import inspect
    import re

    source = inspect.getsource(report)
    keys = set(re.findall(r'"(\w+)":', source)) | set(re.findall(r'f"(\w+)_\{', source))
    dynamic = {f"{p}_{d}" for p in ("n_outbreaks", "obs_dated", "delay_median") for d in ("hpai", "asf")}
    template = re.sub(r"\A<!--.*?-->", "", report.TEMPLATE.read_text(encoding="utf-8"), flags=re.S)
    unknown = set(report.PLACEHOLDER.findall(template)) - keys - dynamic
    assert not unknown, unknown


@pytest.mark.parametrize("x,expected", [(0.645, "65%"), (0.6449, "64%"), (0.125, "13%"), (1.0, "100%")])
def test_pct_rounds_half_up(x, expected):
    assert report.pct(x) == expected


def test_formatting_helpers():
    assert report.signed(-0.2) == "−0.20" and report.signed(0.085, 2) == "+0.09"
    assert report.pct_range([0.48, 0.52]) == "48–52%" and report.pct_range([0.5, 0.5]) == "50%"
    assert report.join_words(["a"]) == "a" and report.join_words(["a", "b", "c"]) == "a, b and c"
    assert report.md_table(["A", "B"], [[1, 2]]) == "| A | B |\n|---|---|\n| 1 | 2 |"


def test_readme_links_point_to_existing_report_sections():
    """README results link into the generated report; renaming a heading must not break them silently."""
    import re

    readme = (report.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    template = report.TEMPLATE.read_text(encoding="utf-8")

    def slug(heading: str) -> str:  # GitHub heading-anchor rules
        return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")

    anchors = {slug(h) for h in re.findall(r"^#{1,6} (.+)$", template, re.M)}
    links = re.findall(r"\]\(docs/technical_report\.md#([^)]+)\)", readme)
    assert links, "expected README links into the report"
    assert not [a for a in links if a not in anchors]
