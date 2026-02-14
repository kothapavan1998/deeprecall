"""End-to-end test of DeepRecall v0.2 features with complex documents."""

from __future__ import annotations

import os
import time

# --- Complex multi-document dataset requiring multi-hop reasoning ---

DOCUMENTS = [
    # Document 1: Company overview
    """TechNova Inc. Annual Report 2025 - Executive Summary

TechNova Inc. was founded in 2019 by Dr. Sarah Chen and Marcus Rivera in Austin, Texas.
The company specializes in autonomous warehouse robotics and AI-powered logistics optimization.

In fiscal year 2025, TechNova reported revenue of $847 million, a 43% increase from
$592 million in FY2024. Net income reached $127 million, compared to a net loss of
$34 million the previous year. The company achieved profitability for the first time
in Q2 2025.

TechNova operates in three business segments:
1. Warehouse Robotics (62% of revenue) - autonomous picking and packing robots
2. Fleet Intelligence (24% of revenue) - delivery route optimization software
3. Supply Chain Analytics (14% of revenue) - demand forecasting and inventory management

The company has 2,847 employees across 12 offices globally, with major R&D centers
in Austin (HQ), Berlin, and Singapore.""",
    # Document 2: Product details
    """TechNova Product Line - Technical Specifications

NovaPick X3 (Warehouse Robotics):
- 6-axis robotic arm with 99.7% pick accuracy
- Processes 1,200 items per hour (3x industry average)
- Uses proprietary VisionAI system with 12 cameras
- Price: $185,000 per unit, 3-year warranty
- Deployed at: Amazon, Walmart, DHL, FedEx
- Total units deployed: 4,200+ across 180 warehouses

FleetMind Pro (Fleet Intelligence):
- Real-time route optimization for delivery fleets
- Reduces fuel costs by average 28%
- Handles up to 10,000 vehicles per instance
- SaaS pricing: $12 per vehicle per month
- Customers: UPS, Maersk, XPO Logistics
- Processes 2.1 million route optimizations daily

ChainSight (Supply Chain Analytics):
- ML-powered demand forecasting with 94% accuracy
- Inventory optimization reducing overstock by 35%
- Integration with SAP, Oracle, and Microsoft Dynamics
- Enterprise pricing: $50,000 - $500,000 annually
- Notable clients: Procter & Gamble, Unilever, Nestlé""",
    # Document 3: Financial details
    """TechNova Financial Performance - Detailed Breakdown

Revenue by Segment (FY2025):
- Warehouse Robotics: $525M (grew 51% YoY)
- Fleet Intelligence: $203M (grew 37% YoY)
- Supply Chain Analytics: $119M (grew 28% YoY)

Gross Margin: 64.2% (up from 58.1% in FY2024)
Operating Margin: 18.3% (vs -3.2% in FY2024)
R&D Spending: $156M (18.4% of revenue)

Key Financial Metrics:
- Cash and equivalents: $423M
- Total debt: $180M (paid down from $340M)
- Free cash flow: $198M
- Customer retention rate: 97.2%

The improvement in margins was driven by:
1. Economies of scale in robot manufacturing (unit cost down 22%)
2. Higher SaaS attach rates on FleetMind (now 78% of fleet customers)
3. Reduced cloud infrastructure costs through custom chip deployment

Capital expenditure was $89M, primarily for the new Berlin R&D center
and expansion of the Austin manufacturing facility.""",
    # Document 4: Competitive landscape
    """TechNova Competitive Analysis - Market Position

Warehouse Robotics Market (TAM: $18.7B by 2027):
- TechNova: 12% market share (up from 8% in 2024)
- Amazon Robotics: 28% (primarily internal use)
- Berkshire Grey: 9%
- Locus Robotics: 7%
- Fetch Robotics (Zebra): 6%

TechNova's key differentiators:
1. Pick accuracy (99.7% vs industry avg 97.2%)
2. Speed (1,200 items/hr vs avg 400 items/hr)
3. Integration flexibility (works with 40+ WMS platforms)
4. Total cost of ownership (ROI in 14 months vs industry avg 24 months)

Fleet Intelligence Market (TAM: $8.2B by 2027):
- TechNova FleetMind: 6% market share
- Samsara: 15%
- Geotab: 12%
- Verizon Connect: 10%

Supply Chain Analytics Market (TAM: $12.4B by 2027):
- Blue Yonder: 18%
- Kinaxis: 11%
- TechNova ChainSight: 3%
- o9 Solutions: 5%""",
    # Document 5: Leadership and strategy
    """TechNova Leadership and Strategic Outlook

Board of Directors:
- Dr. Sarah Chen, CEO & Co-founder (PhD MIT, prev. at Boston Dynamics)
- Marcus Rivera, CTO & Co-founder (prev. VP Engineering at Waymo)
- Jennifer Walsh, CFO (prev. CFO at Datadog)
- Dr. Raj Patel, Chief AI Officer (prev. DeepMind, 47 publications)
- Lisa Kim, COO (prev. SVP Operations at Shopify)

Strategic Priorities for FY2026:
1. Launch NovaPick X4 with dual-arm manipulation (expected Q2 2026)
2. Expand FleetMind into autonomous vehicle coordination
3. Achieve $1.2B revenue target (42% growth)
4. Enter Asian market with Singapore hub (targeting $100M revenue)
5. Strategic acquisition in computer vision ($50-150M budget)

Risk Factors:
- Dependence on semiconductor supply chain (NVIDIA GPUs)
- Increasing competition from Amazon Robotics
- Regulatory uncertainty around autonomous systems
- Key person risk (Dr. Chen and Rivera)
- Geopolitical risks in Singapore expansion

Recent Partnerships:
- NVIDIA: Preferred partner for Jetson robotics platform
- Microsoft: Azure integration for ChainSight
- Samsung SDS: Distribution partner for Asian markets""",
    # Document 6: Technical deep-dive
    """TechNova AI Architecture - Technical White Paper (Summary)

VisionAI System (powers NovaPick):
The proprietary VisionAI system uses a novel architecture called HARPS
(Hierarchical Attention for Robotic Pick and Sort). Key innovations:

1. Multi-scale object detection using 12 synchronized cameras
2. Real-time 3D reconstruction at 60fps
3. Grasp planning neural network trained on 50M simulated grasps
4. Transfer learning from sim-to-real with only 1,000 real-world examples
5. Handles deformable objects (bags, clothing) with 98.1% accuracy

The system runs on NVIDIA Jetson AGX Orin, consuming only 45W of power.
Inference latency is under 50ms per pick decision.

FleetMind Optimization Engine:
Uses a combination of:
- Graph neural networks for route topology
- Reinforcement learning for dynamic re-routing
- Transformer-based demand prediction
- Quantum-inspired optimization for vehicle scheduling

Processing infrastructure: 2,400 NVIDIA A100 GPUs across 3 data centers.
Average optimization time: 230ms for a 10,000-vehicle fleet.

Patent Portfolio: TechNova holds 87 patents (34 granted, 53 pending),
covering robotic manipulation, route optimization, and demand forecasting.""",
]

METADATAS = [
    {"section": "executive_summary", "type": "annual_report", "year": "2025"},
    {"section": "products", "type": "technical", "year": "2025"},
    {"section": "financials", "type": "annual_report", "year": "2025"},
    {"section": "competition", "type": "analysis", "year": "2025"},
    {"section": "leadership", "type": "annual_report", "year": "2025"},
    {"section": "technology", "type": "white_paper", "year": "2025"},
]


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
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return

    print("=" * 70)
    print("DeepRecall v0.2 End-to-End Test")
    print("=" * 70)

    # --- Setup ---
    print("\n[1/6] Setting up vector store and engine...")

    store = ChromaStore(collection_name="technova_test")
    ids = store.add_documents(documents=DOCUMENTS, metadatas=METADATAS)
    print(f"  Ingested {len(ids)} documents ({store.count()} total)")

    # Create usage tracker to aggregate across tests
    usage_tracker = UsageTrackingCallback()

    config = DeepRecallConfig(
        backend="openai",
        backend_kwargs={"model_name": "gpt-4o-mini", "api_key": api_key},
        max_iterations=10,
        top_k=5,
        verbose=False,
        cache=InMemoryCache(max_size=100, default_ttl=300),
        callbacks=[
            ConsoleCallback(show_code=True, show_output=True),
            usage_tracker,
        ],
    )
    engine = DeepRecall(vectorstore=store, config=config)
    print(f"  Engine: {engine}")

    # --- Test 1: Basic query with reasoning trace ---
    print("\n" + "=" * 70)
    print("[2/6] Test: Multi-hop reasoning query")
    print("=" * 70)

    result = engine.query(
        "What is TechNova's most profitable business segment, and what "
        "specific technology makes it successful? Include financial numbers."
    )

    print(f"\n  Answer: {result.answer[:300]}...")
    print(f"  Sources: {len(result.sources)}")
    print(f"  Reasoning steps: {len(result.reasoning_trace)}")
    print(f"  Execution time: {result.execution_time:.2f}s")
    print(f"  Confidence: {result.confidence}")
    print(f"  Tokens: {result.usage.total_input_tokens + result.usage.total_output_tokens}")

    if result.reasoning_trace:
        print(f"\n  --- Reasoning Trace ({len(result.reasoning_trace)} steps) ---")
        for step in result.reasoning_trace:
            searches = [s.get("query", "?") for s in step.searches]
            print(
                f"  Step {step.iteration}: {step.action} "
                f"| searches={len(step.searches)} "
                f"| sub_llm={step.sub_llm_calls} "
                f"| time={step.iteration_time or 0:.1f}s"
            )
            if searches:
                print(f"    Searched: {searches}")

    assert result.answer, "Expected non-empty answer"
    assert result.reasoning_trace, "Expected non-empty reasoning trace"
    print("\n  PASSED")

    # --- Test 2: Budget guardrails ---
    print("\n" + "=" * 70)
    print("[3/6] Test: Budget guardrails (max 3 searches)")
    print("=" * 70)

    result2 = engine.query(
        "Compare all three of TechNova's business segments in detail.",
        budget=QueryBudget(max_search_calls=3, max_time_seconds=60),
    )

    print(f"\n  Answer length: {len(result2.answer)}")
    print(f"  Budget status: {result2.budget_status}")
    print(f"  Error: {result2.error}")
    print(f"  Search calls used: {result2.budget_status.get('search_calls_used', 0)}")

    assert result2.budget_status is not None, "Expected budget status"
    print("\n  PASSED")

    # --- Test 3: Caching ---
    print("\n" + "=" * 70)
    print("[4/6] Test: Query caching")
    print("=" * 70)

    # Same query should hit cache
    t1 = time.perf_counter()
    result3a = engine.query(
        "What is TechNova's most profitable business segment, and what "
        "specific technology makes it successful? Include financial numbers."
    )
    time_first = time.perf_counter() - t1

    t2 = time.perf_counter()
    result3b = engine.query(
        "What is TechNova's most profitable business segment, and what "
        "specific technology makes it successful? Include financial numbers."
    )
    time_cached = time.perf_counter() - t2

    cache_stats = config.cache.stats()
    print(f"  First query time: {time_first:.2f}s")
    print(f"  Cached query time: {time_cached:.4f}s")
    print(f"  Speedup: {time_first / max(time_cached, 0.001):.0f}x")
    print(f"  Cache stats: {cache_stats}")
    print(f"  Answers match: {result3a.answer == result3b.answer}")

    assert time_cached < time_first, "Cached query should be faster"
    assert cache_stats["hits"] >= 1, "Expected at least 1 cache hit"
    print("\n  PASSED")

    # --- Test 4: Complex multi-hop question ---
    print("\n" + "=" * 70)
    print("[5/6] Test: Complex multi-hop reasoning")
    print("=" * 70)

    result4 = engine.query(
        "If TechNova achieves its FY2026 revenue target, what would be the "
        "implied growth rate, and how does this compare to their FY2025 growth? "
        "Also, what are the biggest risks to achieving this target?"
    )

    print(f"\n  Answer: {result4.answer[:400]}...")
    print(f"  Steps: {len(result4.reasoning_trace)}")
    print(f"  Sources used: {len(result4.sources)}")

    assert result4.answer, "Expected non-empty answer"
    print("\n  PASSED")

    # --- Test 5: Usage summary ---
    print("\n" + "=" * 70)
    print("[6/6] Test: Usage tracking across all queries")
    print("=" * 70)

    summary = usage_tracker.summary()
    print(f"  Total queries: {summary['total_queries']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Total searches: {summary['total_searches']}")
    print(f"  Total time: {summary['total_time']}s")
    print(f"  Errors: {summary['errors']}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
