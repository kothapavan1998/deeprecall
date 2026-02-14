"""
End-to-end test: Real CRE underwriting dataset from LenderBox.
Ingests PDFs + CSVs into persistent ChromaDB, then runs multi-hop
underwriting questions through DeepRecall's RLM engine.
"""

from __future__ import annotations

import csv
import os
import shutil
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "/Users/kothapavan/Downloads/lenderbox_dummy_dataset"
CHROMA_DIR = "/tmp/deeprecall_cre_test_db"

# ---------------------------------------------------------------------------
# Document loading helpers
# ---------------------------------------------------------------------------


def load_pdf_text(path: str) -> str:
    """Read a text-based PDF (the lenderbox PDFs are simple text)."""
    with open(path, "rb") as f:
        try:
            import PyPDF2  # noqa: F811

            reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            pass

    # Fallback: these PDFs are tiny and mostly plain text extractable
    with open(path, errors="ignore") as f:
        raw = f.read()
    # Strip binary noise
    lines = [ln for ln in raw.splitlines() if ln.isprintable()]
    return "\n".join(lines)


def load_csv_rows(path: str) -> list[dict]:
    """Load CSV into a list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


def chunk_loan_applications(rows: list[dict]) -> list[tuple[str, dict]]:
    """One document per loan application."""
    docs = []
    for r in rows:
        text = (
            f"Loan Application - {r['loan_id']}\n"
            f"Borrower: {r['borrower_name']}, Credit Score: {r['borrower_credit_score']}\n"
            f"Loan Amount: ${float(r['loan_amount']):,.2f}\n"
            f"Interest Rate: {r['interest_rate']}%, Term: {r['loan_term_years']}yr, "
            f"Amortization: {r['loan_amortization_years']}yr\n"
            f"Property: {r['property_type']} in {r['property_city']}\n"
            f"Status: {r['approval_status']} ({r['approval_notes']})"
        )
        meta = {
            "source": "loan_applications.csv",
            "loan_id": r["loan_id"],
            "type": "loan_application",
            "city": r["property_city"],
            "property_type": r["property_type"],
        }
        docs.append((text, meta))
    return docs


def chunk_loan_performance(rows: list[dict]) -> list[tuple[str, dict]]:
    """Group payment history by loan_id -> one doc per loan."""
    from collections import defaultdict

    by_loan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_loan[r["loan_id"]].append(r)

    docs = []
    for loan_id, payments in by_loan.items():
        total_due = sum(float(p["payment_due"]) for p in payments)
        total_paid = sum(float(p["payment_made"]) for p in payments)
        delinquent_months = sum(1 for p in payments if p["delinquent"] == "1")

        lines = [f"Payment History - {loan_id}"]
        lines.append(f"Period: {payments[0]['month']} to {payments[-1]['month']}")
        lines.append(f"Total Due: ${total_due:,.2f}, Total Paid: ${total_paid:,.2f}")
        lines.append(
            f"Collection Rate: {total_paid / total_due * 100:.1f}%"
            if total_due > 0
            else "Collection Rate: N/A"
        )
        lines.append(f"Delinquent Months: {delinquent_months} of {len(payments)}")
        lines.append("")
        lines.append("Month-by-month:")
        for p in payments:
            flag = " ** DELINQUENT **" if p["delinquent"] == "1" else ""
            lines.append(
                f"  {p['month']}: Due ${float(p['payment_due']):,.2f}, "
                f"Paid ${float(p['payment_made']):,.2f}{flag}"
            )

        meta = {
            "source": "loan_performance.csv",
            "loan_id": loan_id,
            "type": "payment_history",
            "delinquent_months": str(delinquent_months),
        }
        docs.append(("\n".join(lines), meta))
    return docs


def chunk_property_financials(rows: list[dict]) -> list[tuple[str, dict]]:
    """One document per loan's property financials."""
    docs = []
    for r in rows:
        text = (
            f"Property Financials - {r['loan_id']}\n"
            f"NOI: ${float(r['net_operating_income']):,.2f}\n"
            f"Occupancy Rate: {r['occupancy_rate_pct']}%\n"
            f"Annual Rent Roll: ${float(r['annual_rent_roll']):,.2f}\n"
            f"Annual Expenses: ${float(r['annual_expenses']):,.2f}\n"
            f"DSCR: {r['debt_service_coverage_ratio']}\n"
            f"LTV: {r['loan_to_value_pct']}%"
        )
        meta = {
            "source": "property_financials.csv",
            "loan_id": r["loan_id"],
            "type": "property_financial",
        }
        docs.append((text, meta))
    return docs


def chunk_market_data(rows: list[dict]) -> list[tuple[str, dict]]:
    """Group market data by city + property_type -> one doc per combo."""
    from collections import defaultdict

    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = f"{r['city']}_{r['property_type']}"
        by_key[key].append(r)

    docs = []
    for _key, months in by_key.items():
        city = months[0]["city"]
        ptype = months[0]["property_type"]

        avg_rent = sum(float(m["avg_rent_per_sqft"]) for m in months) / len(months)
        avg_cap = sum(float(m["cap_rate_pct"]) for m in months) / len(months)
        avg_vac = sum(float(m["vacancy_rate_pct"]) for m in months) / len(months)
        avg_price = sum(float(m["avg_price_per_sqft"]) for m in months) / len(months)

        lines = [f"Market Data - {city} {ptype} (2025)"]
        lines.append(f"Avg Rent/SqFt: ${avg_rent:.2f}")
        lines.append(f"Avg Cap Rate: {avg_cap:.2f}%")
        lines.append(f"Avg Vacancy Rate: {avg_vac:.2f}%")
        lines.append(f"Avg Price/SqFt: ${avg_price:.2f}")
        lines.append("")
        lines.append("Monthly breakdown:")
        for m in months:
            lines.append(
                f"  {m['month']}: Rent ${m['avg_rent_per_sqft']}/sqft, "
                f"Cap {m['cap_rate_pct']}%, "
                f"Vacancy {m['vacancy_rate_pct']}%, "
                f"Price ${m['avg_price_per_sqft']}/sqft"
            )

        meta = {
            "source": "market_data.csv",
            "type": "market_data",
            "city": city,
            "property_type": ptype,
        }
        docs.append(("\n".join(lines), meta))
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    from deeprecall import (
        ConsoleCallback,
        DeepRecall,
        DeepRecallConfig,
        InMemoryCache,
        QueryBudget,
    )
    from deeprecall.core.callbacks import UsageTrackingCallback
    from deeprecall.vectorstores.chroma import ChromaStore

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY")
        return

    print("=" * 70)
    print("DeepRecall -- CRE Underwriting E2E Test (LenderBox Dataset)")
    print("=" * 70)

    # --- 1. Ingest into persistent ChromaDB ---
    print("\n[1/5] Ingesting documents into persistent ChromaDB...")

    # Clean slate
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    store = ChromaStore(
        collection_name="lenderbox_cre",
        persist_directory=CHROMA_DIR,
    )

    all_docs: list[str] = []
    all_metas: list[dict] = []

    # PDFs
    pdf_files = {
        "appraisal_report.pdf": "appraisal",
        "lease_agreement.pdf": "lease",
        "loan_term_sheet.pdf": "loan_terms",
        "loan_application_form.pdf": "loan_application_form",
        "payment_history_statement.pdf": "payment_statement",
    }
    for fname, dtype in pdf_files.items():
        path = os.path.join(DATA_DIR, fname)
        text = load_pdf_text(path)
        if text.strip():
            all_docs.append(text)
            all_metas.append({"source": fname, "type": dtype})
            print(f"  PDF: {fname} ({len(text)} chars)")

    # CSVs -- intelligently chunked
    loan_apps = load_csv_rows(os.path.join(DATA_DIR, "loan_applications.csv"))
    chunks = chunk_loan_applications(loan_apps)
    for doc, meta in chunks:
        all_docs.append(doc)
        all_metas.append(meta)
    print(f"  CSV: loan_applications.csv -> {len(chunks)} loan docs")

    perf_rows = load_csv_rows(os.path.join(DATA_DIR, "loan_performance.csv"))
    chunks = chunk_loan_performance(perf_rows)
    for doc, meta in chunks:
        all_docs.append(doc)
        all_metas.append(meta)
    print(f"  CSV: loan_performance.csv -> {len(chunks)} payment history docs")

    fin_rows = load_csv_rows(os.path.join(DATA_DIR, "property_financials.csv"))
    chunks = chunk_property_financials(fin_rows)
    for doc, meta in chunks:
        all_docs.append(doc)
        all_metas.append(meta)
    print(f"  CSV: property_financials.csv -> {len(chunks)} financial docs")

    mkt_rows = load_csv_rows(os.path.join(DATA_DIR, "market_data.csv"))
    chunks = chunk_market_data(mkt_rows)
    for doc, meta in chunks:
        all_docs.append(doc)
        all_metas.append(meta)
    print(f"  CSV: market_data.csv -> {len(chunks)} market docs")

    ids = store.add_documents(documents=all_docs, metadatas=all_metas)
    print(f"\n  Total documents ingested: {len(ids)}")
    print(f"  ChromaDB persisted at: {CHROMA_DIR}")

    # --- 2. Setup engine ---
    print("\n[2/5] Configuring DeepRecall engine...")

    usage = UsageTrackingCallback()
    config = DeepRecallConfig(
        backend="openai",
        backend_kwargs={"model_name": "gpt-4o-mini", "api_key": api_key},
        max_iterations=12,
        top_k=8,
        verbose=False,
        cache=InMemoryCache(max_size=50, default_ttl=600),
        callbacks=[
            ConsoleCallback(show_code=True, show_output=True),
            usage,
        ],
    )
    engine = DeepRecall(vectorstore=store, config=config)
    print(f"  Engine ready: {engine}")

    # --- 3. Multi-hop underwriting queries ---

    queries = [
        {
            "name": "Cross-doc deal analysis",
            "query": (
                "For the industrial property at 789 Commerce Blvd Houston: "
                "What is the appraised value, the lease terms (tenant, rent, "
                "escalation), and the loan terms (amount, rate, LTV, DSCR)? "
                "Does the loan amount make sense given the appraised value and LTV?"
            ),
            "budget": QueryBudget(max_time_seconds=120),
        },
        {
            "name": "Risk assessment with payment history",
            "query": (
                "Which loans in the portfolio have the highest risk? "
                "Consider delinquency history, DSCR below 1.25, LTV above 75%, "
                "and low credit scores. List the top 5 riskiest loans with "
                "specific numbers and explain why each is risky."
            ),
            "budget": QueryBudget(max_time_seconds=120),
        },
        {
            "name": "Market comparison for underwriting",
            "query": (
                "Compare the Houston Industrial market conditions to the lease "
                "terms for the property at 789 Commerce Blvd. Is the $22/sqft "
                "rent above or below market? What does the cap rate environment "
                "suggest about the property's $5.75M appraised value? "
                "Is this a good deal for the lender?"
            ),
            "budget": QueryBudget(max_time_seconds=120),
        },
    ]

    for i, q in enumerate(queries, start=3):
        print(f"\n{'=' * 70}")
        print(f"[{i}/5] Query: {q['name']}")
        print("=" * 70)

        t0 = time.perf_counter()
        result = engine.query(q["query"], budget=q["budget"])
        elapsed = time.perf_counter() - t0

        print("\n--- Answer ---")
        print(result.answer[:800])
        if len(result.answer) > 800:
            print(f"... ({len(result.answer)} chars total)")
        print("\n--- Stats ---")
        print(f"  Sources: {len(result.sources)}")
        print(f"  Reasoning steps: {len(result.reasoning_trace)}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Confidence: {result.confidence}")
        if result.budget_status:
            print(f"  Budget exceeded: {result.budget_status.get('budget_exceeded')}")
        if result.error:
            print(f"  Error: {result.error}")
        print()

    # --- Summary ---
    print("=" * 70)
    print("USAGE SUMMARY")
    print("=" * 70)
    summary = usage.summary()
    print(f"  Total queries: {summary['total_queries']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Total searches: {summary['total_searches']}")
    print(f"  Total time: {summary['total_time']}s")
    print(f"  Errors: {summary['errors']}")
    print(f"\n  ChromaDB data persisted at: {CHROMA_DIR}")
    print("  (Data survives restarts -- re-run queries without re-ingesting)")

    print("\n" + "=" * 70)
    print("ALL QUERIES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
