"""Tests specific to the bigquery dialect."""

import hypothesis.strategies as st
import pytest
from hypothesis import example, given, note, settings

from sqlfluff.core import FluffConfig, Linter
from sqlfluff.core.dialects.common import (
    get_from_clause_aliases,
    iter_raw_references,
)
from sqlfluff.core.parser import Lexer, Parser


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.tuples(st.sampled_from(["<", "=", ">"]), st.sampled_from(["AND", "OR"])),
        min_size=1,
        max_size=30,
    )
)
@example(data=[("<", "AND")])
@example(data=[(">", "AND")])
@example(data=[("<", "AND"), (">", "AND")])
@example(data=[("<", "AND"), ("=", "AND"), (">", "AND")])
@example(data=[(">", "AND"), ("<", "AND")])
@example(data=[("<", "AND"), ("<", "AND"), (">", "AND")])
@example(data=[(">", "AND"), (">", "AND"), ("<", "AND")])
def test_bigquery_relational_operator_parsing(data):
    """Tests queries with a diverse mixture of relational operators."""
    # Generate a simple SELECT query with relational operators and conjunctions
    # as specified in 'data'. Note the conjunctions are used as separators
    # between comparisons, sn the conjunction in the first item is not used.
    filter = []
    for i, (relation, conjunction) in enumerate(data):
        if i:
            filter.append(f" {conjunction} ")
        filter.append(f"a {relation} b")
    raw = f"SELECT * FROM t WHERE {''.join(filter)}"
    note(f"query: {raw}")
    # Load the right dialect
    config = FluffConfig(overrides=dict(dialect="bigquery"))
    tokens, lex_vs = Lexer(config=config).lex(raw)
    # From just the initial parse, check we're all there
    assert "".join(token.raw for token in tokens) == raw
    # Check we don't have lexing issues
    assert not lex_vs

    # Do the parse WITHOUT lots of logging
    # The logs get too long here to be useful. We should use
    # specific segment tests if we want to debug logs.
    parsed = Parser(config=config).parse(tokens)
    print(f"Post-parse structure: {parsed.to_tuple(show_raw=True)}")
    print(f"Post-parse structure: {parsed.stringify()}")
    # Check we're all there.
    assert parsed.raw == raw
    # Check that there's nothing un parsable
    typs = parsed.type_set()
    assert "unparsable" not in typs


@pytest.mark.parametrize(
    "table_reference, reference_parts",
    [
        (
            "bigquery-public-data.pypi.file_downloads",
            ["bigquery-public-data", "pypi", "file_downloads"],
        ),
        (
            "`bigquery-public-data.pypi.file_downloads`",
            ["bigquery-public-data", "pypi", "file_downloads"],
        ),
        ("foo.far.bar", ["foo", "far", "bar"]),
        ("`foo.far.bar`", ["foo", "far", "bar"]),
        ("a-b.c-d.e-f", ["a-b", "c-d", "e-f"]),
    ],
)
def test_bigquery_table_reference_segment_iter_raw_references(
    table_reference, reference_parts
):
    """Tests BigQuery handling of iter_raw_references() for table references.

    The BigQuery implementation is more complex, handling:
    - hyphenated table references
    - quoted or not quoted table references

    Exercises both the new free function (the recommended API) and the
    deprecated ``TableReferenceSegment.iter_raw_references()`` wrapper, which
    must still return the same result and emit a ``DeprecationWarning``.
    """
    query = f"SELECT bar.user_id FROM {table_reference}"
    config = FluffConfig(overrides=dict(dialect="bigquery"))
    tokens, lex_vs = Lexer(config=config).lex(query)
    parsed = Parser(config=config).parse(tokens)
    for table_reference in parsed.recursive_crawl("table_reference"):
        # Recommended API: the free function.
        actual_reference_parts = [
            orp.part for orp in iter_raw_references(table_reference, "bigquery")
        ]
        assert reference_parts == actual_reference_parts
        # Deprecated wrapper: still works, but warns.
        with pytest.warns(DeprecationWarning):
            deprecated_parts = [
                orp.part for orp in table_reference.iter_raw_references()
            ]
        assert reference_parts == deprecated_parts


@pytest.mark.parametrize(
    "from_clause, expected_ref_str",
    [
        # Unaliased, hyphenated, multi-part table name. The eventual alias is
        # the final part, which requires BigQuery's hyphen-aware handling
        # ("tbl-name", not "name").
        ("project.dataset.tbl-name", "tbl-name"),
        # Quoted multi-part name.
        ("`bigquery-public-data.pypi.file_downloads`", "file_downloads"),
        # Explicit alias is returned verbatim.
        ("foo.bar AS baz", "baz"),
    ],
)
def test_bigquery_from_clause_get_eventual_aliases_back_compat(
    from_clause, expected_ref_str
):
    """Back-compat for ``FromClauseSegment.get_eventual_aliases()`` on BigQuery.

    This is a behavior-parity check: the deprecated wrapper and the replacement
    free function ``get_from_clause_aliases`` must agree, and both must match
    the legacy (pre-refactor) implementation. It guards the back-compat path
    (``dialect_name=None``), which must dispatch via the segment's own class so
    the legacy behavior is unchanged.
    """
    query = f"SELECT 1 FROM {from_clause}"
    config = FluffConfig(overrides=dict(dialect="bigquery"))
    tokens, _ = Lexer(config=config).lex(query)
    parsed = Parser(config=config).parse(tokens)
    fc = next(parsed.recursive_crawl("from_clause"))

    # Recommended API: the free function with an explicit dialect.
    free_aliases = get_from_clause_aliases(fc, "bigquery")
    assert free_aliases[0][1].ref_str == expected_ref_str

    # Segment method must agree with the free function (and the legacy impl).
    wrapper_aliases = fc.get_eventual_aliases()
    assert wrapper_aliases[0][1].ref_str == expected_ref_str


def test_cast_as_float_fails():
    """CAST(... AS FLOAT) should fail for BigQuery (only FLOAT64 allowed)."""
    sql = "SELECT CAST('4.0' AS FLOAT)"
    parsed = Linter(dialect="bigquery").parse_string(sql)
    # Parsing produces a parse error for the unsupported `FLOAT` type.
    assert parsed.violations


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT x.y.*.z FROM t",
        "SELECT results[0].*.z FROM t",
        # The wildcard is equally terminal when the chain hangs off a function.
        "SELECT testFunction(a).b.*.z FROM t",
        "SELECT testFunction(a).*.z FROM t",
        "SELECT testFunction(a)[0].b.*.z FROM t",
    ],
)
def test_semi_structured_wildcard_is_terminal(sql):
    """A wildcard may only be the final element of an accessor chain."""
    parsed = Linter(dialect="bigquery").parse_string(sql)
    # Nothing may be chained after the `.*`, so these are a parse error.
    assert parsed.violations


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT results[0].* FROM t",
        "SELECT results[0].* EXCEPT (cola) FROM t",
        "SELECT results[0].* REPLACE (1 AS cola) FROM t",
        "SELECT s.arr[0].field.* EXCEPT (cola) FROM t",
        "SELECT testFunction(a).b.* FROM t",
    ],
)
def test_semi_structured_wildcard_parses(sql):
    """A wildcard ending an accessor chain parses, with EXCEPT/REPLACE."""
    parsed = Linter(dialect="bigquery").parse_string(sql)
    assert not parsed.violations


@pytest.mark.parametrize(
    "sql",
    [
        # A SELECT statement followed by a chain of pipe operators.
        "SELECT a, COUNT(*) AS c FROM t\n"
        "|> WHERE a > 10\n"
        "|> GROUP BY a\n"
        "|> ORDER BY c DESC\n",
        # A FROM-first query with a standalone GROUP BY pipe operator.
        "FROM Produce\n|> GROUP BY item\n",
        # GROUP BY with multiple keys, followed by more operators.
        "FROM Produce\n|> GROUP BY category, item\n|> ORDER BY category DESC, item\n",
        # GROUP AND ORDER BY as a standalone pipe operator.
        "FROM Produce\n|> GROUP AND ORDER BY item, category DESC\n",
        # AGGREGATE combined with GROUP BY still parses.
        "FROM Produce\n"
        "|> AGGREGATE SUM(sales) AS total_sales GROUP BY item\n"
        "|> ORDER BY total_sales DESC\n",
    ],
)
def test_bigquery_pipe_statement_parses(sql):
    """Pipe operator chains parse natively in the BigQuery dialect."""
    parsed = Linter(dialect="bigquery").parse_string(sql)
    assert not parsed.violations
    tree = parsed.tree
    assert tree
    # The whole query is a single pipe statement.
    pipe_statements = list(tree.recursive_crawl("pipe_statement"))
    assert len(pipe_statements) == 1
    # One clause per `|>` operator in the source.
    pipe_clauses = list(tree.recursive_crawl("pipe_operator_clause"))
    assert len(pipe_clauses) == sql.count("|>")
    # Each operator clause retains its position in the source file.
    for clause in pipe_clauses:
        assert clause.pos_marker is not None
    assert [clause.pos_marker.line_no for clause in pipe_clauses] == [
        line_no
        for line_no, line in enumerate(sql.splitlines(), start=1)
        if line.startswith("|>")
    ]


def test_bigquery_pipe_statement_group_by_clause_type():
    """A standalone `|> GROUP BY` parses as a group_and_orderby_clause."""
    sql = "SELECT a FROM t\n|> GROUP BY a\n"
    parsed = Linter(dialect="bigquery").parse_string(sql)
    assert not parsed.violations
    tree = parsed.tree
    assert tree
    group_clauses = list(tree.recursive_crawl("group_and_orderby_clause"))
    assert len(group_clauses) == 1
    # The clause sits inside a pipe operator clause, keeping position info.
    assert group_clauses[0].pos_marker is not None
    assert group_clauses[0].pos_marker.line_no == 2


def test_bigquery_pipe_statement_rules():
    """Existing layout and aliasing rules give sensible results on pipes."""
    sql = (
        "SELECT a, COUNT(*) AS c\n"
        "FROM very_long_table_name AS this_alias_is_way_too_long\n"
        "|> WHERE a > 10\n"
        "|> GROUP BY a\n"
        "|> ORDER BY c DESC\n"
    )
    config = FluffConfig(
        configs={"rules": {"aliasing.length": {"max_alias_length": 5}}},
        overrides={"dialect": "bigquery", "rules": "LT02,AL01,AL02,AL06"},
    )
    violations = Linter(config=config).lint_string(sql).get_violations()
    # The only violation is the over-length table alias (AL06).
    assert [(v.rule_code(), v.line_no) for v in violations] == [("AL06", 2)]

    # A mis-indented pipe operator is flagged by the indentation rule.
    misindented_sql = sql.replace("|> GROUP BY a", "  |> GROUP BY a")
    violations = Linter(config=config).lint_string(misindented_sql).get_violations()
    assert ("LT02", 4) in [(v.rule_code(), v.line_no) for v in violations]
