"""
RiTa Agent Tools — DataHub / Dataverse compliance tools for the LangGraph agent.

Tools:
  1. fetch_pr_sql_files           — Mock Azure DevOps API: list & fetch .sql files in a PR.
  2. extract_sql_columns_and_tables — Parse SQL to extract columns and source tables using sqlglot.
  3. get_datahub_column_tags       — Mock DataHub/Dataverse lookup for DPDP compliance tags.
  4. rewrite_sql_ast               — sqlglot-powered AST rewrite to inject masking expressions.
  5. block_azure_devops_pr         — Mock Azure DevOps PR-block action with Rich output.
"""

from __future__ import annotations

import sqlglot
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# ──────────────────────────────────────────────────────────────
# 1. Azure DevOps PR File Fetcher (Mock)
# ──────────────────────────────────────────────────────────────

# Simulated PR file contents — keyed by (pr_id, repo) → list of file dicts.
# Each file dict has: path, change_type, and content (for .sql files).
_PR_FILES_MOCK: dict[str, list[dict]] = {
    "442": [
        {
            "path": "dbt/models/analytics/vw_credit_risk.sql",
            "change_type": "add",
            "content": (
                "CREATE VIEW analytics_dw.vw_credit_risk AS\n"
                "SELECT\n"
                "    customer_id,\n"
                "    pan_no,\n"
                "    phone_num\n"
                "FROM stg_lake.raw_customers;"
            ),
        },
        {
            "path": "dbt/models/analytics/vw_loan_summary.sql",
            "change_type": "add",
            "content": (
                "CREATE VIEW analytics_dw.vw_loan_summary AS\n"
                "SELECT\n"
                "    loan_id,\n"
                "    customer_id,\n"
                "    loan_amount,\n"
                "    tenure_months\n"
                "FROM stg_lake.raw_loans;"
            ),
        },
        {
            "path": "README.md",
            "change_type": "edit",
            "content": None,  # Non-SQL files have no SQL content
        },
    ],
    "501": [
        {
            "path": "dbt/models/risk/vw_kyc_report.sql",
            "change_type": "add",
            "content": (
                "CREATE VIEW analytics_dw.vw_kyc_report AS\n"
                "SELECT\n"
                "    customer_id,\n"
                "    pan_no,\n"
                "    phone_num,\n"
                "    email_address\n"
                "FROM stg_lake.raw_customers\n"
                "WHERE kyc_status = 'PENDING';"
            ),
        },
    ],
}


@tool
def fetch_pr_sql_files(pr_id: str) -> list[dict]:
    """Fetch all SQL files changed in an Azure DevOps Pull Request.

    Mocks the Azure DevOps REST API:
      GET /{org}/{project}/_apis/git/repositories/{repo}/pullRequests/{pr_id}/iterations/.../changes

    Filters for files ending in .sql and returns their full content.

    Args:
        pr_id: The Pull Request identifier (e.g. '442').

    Returns:
        A list of dicts, each with keys:
          - path (str): File path in the repository.
          - change_type (str): 'add', 'edit', or 'delete'.
          - content (str): Full SQL file content.
        Returns an empty list if the PR is not found or has no SQL files.
    """
    all_files = _PR_FILES_MOCK.get(pr_id, [])
    sql_files = [
        f for f in all_files
        if f["path"].endswith(".sql") and f["content"] is not None
    ]
    return sql_files


# ──────────────────────────────────────────────────────────────
# 2. SQL Column & Table Extractor (sqlglot)
# ──────────────────────────────────────────────────────────────


@tool
def extract_sql_columns_and_tables(sql: str) -> list[dict[str, str]]:
    """Parse a SQL statement and extract all projected columns along with their source tables.

    Uses sqlglot to parse the SQL AST (Redshift dialect) and resolves which table each column
    in the SELECT clause originates from, handling table aliases and CREATE VIEW statements.

    Args:
        sql: The raw SQL content to parse.

    Returns:
        A list of dicts with keys "table" and "column".
        Example: [{"table": "stg_lake.raw_customers", "column": "pan_no"}, ...]
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="redshift")
    except Exception as exc:
        return [{"error": f"Failed to parse SQL: {exc}"}]

    # Determine view/create target table to avoid treating the target view as a source table
    create_target = None
    if isinstance(tree, sqlglot.exp.Create):
        if tree.this:
            create_target = tree.this.sql(dialect="redshift").lower()

    # Find all source tables in the query
    # Map alias (and table name) -> full table sql representation
    table_map: dict[str, str] = {}
    source_tables: list[str] = []

    for tbl in tree.find_all(sqlglot.exp.Table):
        clean_tbl = tbl.copy()
        clean_tbl.set("alias", None)
        tbl_sql = clean_tbl.sql(dialect="redshift")
        if create_target and tbl_sql.lower() == create_target:
            continue
        source_tables.append(tbl_sql)
        table_map[tbl_sql.lower()] = tbl_sql
        if tbl.name:
            table_map[tbl.name.lower()] = tbl_sql
        if tbl.alias:
            table_map[tbl.alias.lower()] = tbl_sql

    # Find the SELECT statements
    selects = list(tree.find_all(sqlglot.exp.Select))
    if not selects:
        select_expressions = list(tree.find_all(sqlglot.exp.Column))
    else:
        select_expressions = []
        for sel in selects:
            for expr in sel.expressions:
                for col in expr.find_all(sqlglot.exp.Column):
                    select_expressions.append(col)

    results: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    default_table = source_tables[0] if source_tables else "unknown"

    for col in select_expressions:
        col_name = col.name
        table_ref = col.table.lower() if col.table else ""
        resolved_table = table_map.get(table_ref, default_table)

        pair = (resolved_table, col_name)
        if pair not in seen:
            seen.add(pair)
            results.append({"table": resolved_table, "column": col_name})

    return results


# ──────────────────────────────────────────────────────────────
# 3. DataHub / Dataverse Mock
# ──────────────────────────────────────────────────────────────

# Simulated DataHub metadata store — extend this dict to add more tables/columns.
_DATAHUB_MOCK: dict[str, dict[str, list[str]]] = {
    "stg_lake.raw_customers": {
        "pan_no": ["DPDP: PAN"],
        "phone_num": ["DPDP: Mobile"],
    },
}



@tool
def get_datahub_column_tags(table_name: str, column_name: str) -> list[str]:
    """Query DataHub for compliance/governance tags on a specific column.

    Args:
        table_name:  Fully-qualified table name (e.g. 'stg_lake.raw_customers').
        column_name: Column name to look up (e.g. 'pan_no').

    Returns:
        A list of tag strings (e.g. ['DPDP: PAN']) or an empty list if no tags exist.
    """
    #Replace with actual call to Dataverse API
    tags = _DATAHUB_MOCK.get(table_name, {}).get(column_name, [])
    return tags


# ──────────────────────────────────────────────────────────────
# 2. SQL AST Rewriter (sqlglot)
# ──────────────────────────────────────────────────────────────


@tool
def rewrite_sql_ast(raw_sql: str, mask_instructions: dict[str, str]) -> str:
    """Rewrite a Redshift SQL statement by injecting masking expressions.

    Uses sqlglot to parse the SQL into an AST, apply column-level transformations,
    and regenerate clean Redshift SQL — preserving aliases, joins, and structure.

    Args:
        raw_sql:           The original SQL statement (Redshift dialect).
        mask_instructions:  A dict mapping column names to masking SQL expressions.
                           Use ``{col}`` as a placeholder for the original column ref.
                           Example: ``{"pan_no": "SHA2({col}, 256)"}``

    Returns:
        The rewritten SQL string with masking applied.
    """
    tree = sqlglot.parse_one(raw_sql, dialect="redshift")

    def _transformer(node: sqlglot.exp.Expression) -> sqlglot.exp.Expression:
        if isinstance(node, sqlglot.exp.Column):
            col_name = node.name.lower()
            if col_name in mask_instructions:
                mask_expr_template = mask_instructions[col_name]
                # Build the column reference as sqlglot would render it
                col_sql = node.sql(dialect="redshift")
                # Substitute {col} placeholder with the actual column reference
                mask_sql = mask_expr_template.replace("{col}", col_sql)
                # Parse the masking expression and alias it to the original column name
                new_node = sqlglot.parse_one(mask_sql, dialect="redshift")
                return sqlglot.exp.Alias(
                    this=new_node,
                    alias=sqlglot.exp.to_identifier(col_name),
                )
        return node

    transformed = tree.transform(_transformer)
    return transformed.sql(dialect="redshift", pretty=True)


# # ──────────────────────────────────────────────────────────────
# # 3. Azure DevOps PR Blocker (Mock)
# # ──────────────────────────────────────────────────────────────

# _console = Console()


# @tool
# def block_azure_devops_pr(
#     pr_id: str,
#     violation_summary: str,
#     suggested_sql: str,
# ) -> str:
#     """Block an Azure DevOps Pull Request due to a DPDP compliance violation.

#     This mocks the Azure DevOps REST API call that would set the PR status to
#     'blocked' and post a comment with the violation details and suggested fix.

#     Args:
#         pr_id:              The PR identifier (e.g. '442').
#         violation_summary:  Human-readable summary of the compliance violation(s).
#         suggested_sql:      The rewritten SQL that should replace the original.

#     Returns:
#         A confirmation string indicating the PR was blocked successfully.
#     """
#     # ── Rich console output ──
#     table = Table(title="🛡️  DPDP Compliance Violation Report", show_lines=True)
#     table.add_column("Field", style="bold cyan", width=20)
#     table.add_column("Value", style="white")

#     table.add_row("PR ID", f"#{pr_id}")
#     table.add_row("Status", "[bold red]BLOCKED[/bold red]")
#     table.add_row("Violation", violation_summary)

#     _console.print()
#     _console.print(table)
#     _console.print()
#     _console.print(
#         Panel(
#             suggested_sql,
#             title="✅ Suggested Masked SQL",
#             border_style="green",
#             padding=(1, 2),
#         )
#     )
#     _console.print()

#     return (
#         f"PR #{pr_id} has been BLOCKED. "
#         f"Violation: {violation_summary}. "
#         f"Suggested SQL posted as a PR comment."
#     )
