import argparse
import time
from collections import defaultdict
from tabulate import tabulate
from voicebot.turn_detector import TurnDetector, TurnDecision


def run_benchmarks(dialect_filter: str | None = None):
    # Each case has:
    #   text                  — the transcript snippet
    #   silence               — silence duration in ms
    #   expected              — the correct TurnDecision
    #   dialect               — "neutral" | "southern" | "sandiego"
    #   min_silence_override  — (optional) create a detector with this threshold
    #                           instead of the default 800ms.
    #                           This lets us test whether dialect-specific thresholds
    #                           produce correct decisions that the default threshold
    #                           would get wrong.
    test_cases = [

        # ── NEUTRAL: CLASSIC DANGLERS — ends on preposition/conjunction/det (WAIT) ──
        {"text": "I'm looking for a truck that's got", "silence": 900, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "Can you check if the service department is", "silence": 1100, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I was thinking about coming in on", "silence": 850, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "Does the Ford F-150 come with", "silence": 1200, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I'd like to get a vehicle that has", "silence": 950, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "The car I'm interested in is a", "silence": 1000, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I need a truck that can tow up to", "silence": 1050, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I was wondering if you had anything in", "silence": 900, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "Is the black one parked next to", "silence": 880, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "My budget is around forty thousand and", "silence": 1100, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I'd prefer something with leather seats and", "silence": 950, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "Are there any deals going on for", "silence": 1150, "expected": TurnDecision.WAIT, "dialect": "neutral"},

        # ── NEUTRAL: COMPLETE SINGLE-SENTENCE QUERIES (TALK) ──
        {"text": "Is the service center open today?", "silence": 850, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I want to see the Ford F-150.", "silence": 900, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "What are your hours on Sunday?", "silence": 800, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I'd like to book a test drive for tomorrow.", "silence": 950, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Do you have any trucks in stock?", "silence": 850, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "What's the price on that Silverado?", "silence": 900, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I need an oil change today.", "silence": 1000, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Can I schedule an appointment for Saturday?", "silence": 950, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "How much is the extended warranty?", "silence": 880, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I'm interested in the white Chevy Tahoe.", "silence": 1100, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "My trade-in is a 2019 Honda Accord.", "silence": 950, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I want something with good gas mileage.", "silence": 1200, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Is financing available on that vehicle?", "silence": 900, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I'd like to see the inventory for SUVs.", "silence": 850, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Can you tell me about the warranty coverage?", "silence": 1000, "expected": TurnDecision.TALK, "dialect": "neutral"},

        # ── NEUTRAL: MULTI-SENTENCE WITH COMPLETE LAST SENTENCE (TALK) ──
        {"text": "The truck looks great. I want to buy it.", "silence": 850, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I saw the silver one. What's the price?", "silence": 900, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "We've been looking for a while. This is exactly what we need.", "silence": 950, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "My wife drives a sedan. I prefer an SUV though.", "silence": 1000, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I was here last week. I'd like to test drive the Explorer.", "silence": 850, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "The salesperson showed us two options. I like the blue one.", "silence": 900, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "We need something safe for our kids. The Expedition looks perfect.", "silence": 1100, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I've done my research online. I'm ready to make a deal.", "silence": 950, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "My current car has high mileage. I need a reliable replacement.", "silence": 880, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I drove the Tacoma yesterday. I want to buy that one.", "silence": 1000, "expected": TurnDecision.TALK, "dialect": "neutral"},

        # ── NEUTRAL: MULTI-SENTENCE WITH INCOMPLETE LAST SENTENCE (WAIT) ──
        {"text": "The truck looks great. But I was wondering if", "silence": 900, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I saw the silver one on the lot. Does it", "silence": 1000, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I love the color. But I really need something with", "silence": 950, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "The price is decent. Could you check if it comes with", "silence": 1100, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "My husband wants a truck. I personally prefer a", "silence": 850, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I talked to someone yesterday. They said the deal was good but", "silence": 1000, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "This is our third dealership today. We're looking for something that", "silence": 900, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I appreciate the offer. I just need to think about", "silence": 950, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "We've seen a few models. I was hoping to find one that has", "silence": 880, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "That sounds reasonable. But can you also check on", "silence": 1050, "expected": TurnDecision.WAIT, "dialect": "neutral"},

        # ── NEUTRAL: HARD LIMIT — silence >= 2000ms always forces TALK ──
        # These bypass the NLP entirely. Even a dangler should return TALK.
        # Why test this? To confirm the hard_limit branch works independently of syntax.
        {"text": "I need a car.", "silence": 2000, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Looking for a truck with", "silence": 2500, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "My budget is", "silence": 3000, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I want something that has good", "silence": 2200, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Could you check if the deal includes", "silence": 2000, "expected": TurnDecision.TALK, "dialect": "neutral"},

        # ── NEUTRAL: TOO-SHORT SILENCE — silence < 800ms always forces WAIT ──
        # Even a complete sentence returns WAIT. Tests the min_silence branch.
        {"text": "Is the service center open today?", "silence": 500, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I want to see the Ford F-150.", "silence": 300, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "What are your hours?", "silence": 700, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I'd like to schedule an appointment.", "silence": 400, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "Do you have any discounts available?", "silence": 600, "expected": TurnDecision.WAIT, "dialect": "neutral"},

        # ── NEUTRAL: COMPLETE QUESTIONS (TALK) ──
        {"text": "Do you offer military discounts?", "silence": 900, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "How long does a typical oil change take?", "silence": 850, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Is the Raptor available in blue?", "silence": 950, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Can I trade in my current vehicle?", "silence": 1000, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Are there any zero-percent financing options?", "silence": 880, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "What's included in the maintenance package?", "silence": 1100, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Does this model come in all-wheel drive?", "silence": 900, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "How many miles does this truck have?", "silence": 950, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Is there a certified pre-owned program?", "silence": 1000, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "Can you run my credit today?", "silence": 850, "expected": TurnDecision.TALK, "dialect": "neutral"},

        # ── NEUTRAL: ENDS ON A VERB — hungry-verb check catches these (WAIT) ──
        {"text": "My mechanic said the engine needs", "silence": 900, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I'm only here because my wife convinced", "silence": 850, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "The salesman told me to come back and", "silence": 950, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "My mechanic said the car needs", "silence": 900, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I've been meaning to come in and", "silence": 950, "expected": TurnDecision.WAIT, "dialect": "neutral"},

        # ── NEUTRAL: ADDITIONAL DANGLERS — ends with ADP/SCONJ (WAIT) ──
        {"text": "I'm looking at the inventory for", "silence": 900, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I was hoping to get something before", "silence": 850, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "Is the truck available in", "silence": 900, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I was told to ask about", "silence": 1000, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I heard you were running a special on", "silence": 1000, "expected": TurnDecision.WAIT, "dialect": "neutral"},

        # ── NEUTRAL: ADDITIONAL COMPLETE STATEMENTS (TALK) ──
        {"text": "The salesman was very helpful.", "silence": 950, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "My trade-in has about eighty thousand miles.", "silence": 1000, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I'm pre-approved for fifty thousand dollars.", "silence": 950, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "The Expedition is perfect for my family.", "silence": 850, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I just want something reliable.", "silence": 900, "expected": TurnDecision.TALK, "dialect": "neutral"},

        # ── NEUTRAL: MULTI-SENTENCE COMPLETE FOLLOW-UPS (TALK) ──
        {"text": "I came in yesterday. Is the blue one still available?", "silence": 900, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I checked your website. The deal online looks better.", "silence": 950, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "My lease is up in June. I need to find something soon.", "silence": 1000, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "The salesman gave me a quote. Can we do better than that?", "silence": 850, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I test drove the Tundra last week. I'm ready to make a decision.", "silence": 900, "expected": TurnDecision.TALK, "dialect": "neutral"},

        # ── NEUTRAL: BORDERLINE / STRESS CASES ──
        {"text": "I need something before my warranty expires and", "silence": 800, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "The interior is nice but", "silence": 1100, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "I don't know, I was kind of thinking maybe something like", "silence": 850, "expected": TurnDecision.WAIT, "dialect": "neutral"},
        {"text": "Probably around next month or so", "silence": 1000, "expected": TurnDecision.TALK, "dialect": "neutral"},
        {"text": "I need that truck because", "silence": 950, "expected": TurnDecision.WAIT, "dialect": "neutral"},

        # ════════════════════════════════════════════════════════════════
        # SOUTHERN DIALECT CASES
        # These use min_silence_override=1200 to test the Southern profile
        # threshold (1200ms). A Southern speaker pauses longer mid-sentence,
        # so the default 800ms detector would incorrectly fire TALK on
        # silences like 1050ms that are actually mid-sentence pauses.
        # ════════════════════════════════════════════════════════════════

        # ── SOUTHERN: CLASSIC FILLERS AND DANGLERS (WAIT with 1200ms detector) ──
        # Key test: silence is 1050–1150ms — above default 800ms but below 1200ms.
        # With 800ms detector: would TALK (false positive — cuts the speaker off).
        # With 1200ms detector: correctly WAIT.
        {"text": "Well I reckon I'm fixin to", "silence": 1050, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "I was just over there by the", "silence": 1100, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "Shoot I just want something that'll", "silence": 1000, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "Y'all got anything that might be good for", "silence": 1050, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "Bless your heart I was just wondering about", "silence": 1150, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "Lord have mercy I need a truck that can", "silence": 1100, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "Well now I'm not sure but I think it was", "silence": 1050, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "I tell you what, if you can get me something with", "silence": 1150, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "Mama always said I needed something that", "silence": 1050, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "I been drivin that old pickup and it just ain't got the", "silence": 1100, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},

        # ── SOUTHERN: COMPLETE UTTERANCES (TALK — silence well above 1200ms) ──
        # These are genuinely complete sentences. Silence is 1300ms+, so both
        # the 800ms and 1200ms detectors agree: TALK.
        {"text": "Y'all got any trucks left over from last year's model?", "silence": 1400, "expected": TurnDecision.TALK, "dialect": "southern", "min_silence_override": 1200},
        {"text": "I tell you what, that Silverado right there is exactly what I need.", "silence": 1350, "expected": TurnDecision.TALK, "dialect": "southern", "min_silence_override": 1200},
        {"text": "My daddy always drove a Ford and I ain't about to change that.", "silence": 1500, "expected": TurnDecision.TALK, "dialect": "southern", "min_silence_override": 1200},
        {"text": "We been looking all week and this is the best deal we seen.", "silence": 1400, "expected": TurnDecision.TALK, "dialect": "southern", "min_silence_override": 1200},
        {"text": "I reckon I'll take the red one if the price is right.", "silence": 1450, "expected": TurnDecision.TALK, "dialect": "southern", "min_silence_override": 1200},
        {"text": "Can y'all throw in floor mats?", "silence": 1300, "expected": TurnDecision.TALK, "dialect": "southern", "min_silence_override": 1200},
        {"text": "Do you folks offer any kind of military discount?", "silence": 1350, "expected": TurnDecision.TALK, "dialect": "southern", "min_silence_override": 1200},
        {"text": "I need something that can haul my trailer on the weekends.", "silence": 1400, "expected": TurnDecision.TALK, "dialect": "southern", "min_silence_override": 1200},

        # ── SOUTHERN: MULTI-SENTENCE WITH INCOMPLETE LAST SENTENCE (WAIT) ──
        {"text": "That truck looks real nice. But I was wondering if it comes with", "silence": 1050, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "My wife likes the color. I just need to make sure it has", "silence": 1100, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},
        {"text": "I talked to Bobby down at the other lot. He said you might have something that", "silence": 1150, "expected": TurnDecision.WAIT, "dialect": "southern", "min_silence_override": 1200},

        # ── SOUTHERN: HARD LIMIT (silence >= 3000ms — Southern hard_limit) ──
        # Southern hard_limit is 3000ms. Even incomplete text → TALK.
        {"text": "Well I was just thinking maybe something with", "silence": 3000, "expected": TurnDecision.TALK, "dialect": "southern", "min_silence_override": 1200},
        {"text": "I reckon I need a truck that", "silence": 3500, "expected": TurnDecision.TALK, "dialect": "southern", "min_silence_override": 1200},

        # ════════════════════════════════════════════════════════════════
        # SAN DIEGO DIALECT CASES
        # These use min_silence_override=650 to test the SD profile threshold.
        # SD speakers are faster and more clipped — a complete thought can land
        # in under 700ms of silence. The default 800ms would be too slow (WAIT
        # when it should TALK). The 650ms threshold responds faster.
        # ════════════════════════════════════════════════════════════════

        # ── SANDIEGO: COMPLETE CLIPPED SENTENCES (TALK — silence 680–700ms) ──
        # Key test: silence is just above 650ms but below the 800ms default.
        # With 800ms detector: WAIT (misses a complete thought).
        # With 650ms detector: correctly TALK.
        {"text": "So like, what's the price on that one?", "silence": 680, "expected": TurnDecision.TALK, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "Totally want the blue one.", "silence": 700, "expected": TurnDecision.TALK, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "Does it come in black?", "silence": 670, "expected": TurnDecision.TALK, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "Can I do a test drive today?", "silence": 690, "expected": TurnDecision.TALK, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "I'm basically looking for an SUV.", "silence": 680, "expected": TurnDecision.TALK, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "What's the monthly payment on that?", "silence": 700, "expected": TurnDecision.TALK, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "I saw it online and it looks super clean.", "silence": 690, "expected": TurnDecision.TALK, "dialect": "sandiego", "min_silence_override": 650},

        # ── SANDIEGO: TOO-SHORT SILENCE — still WAIT even with 650ms detector ──
        # Validates the lower threshold doesn't fire too eagerly.
        {"text": "Does it come in red?", "silence": 580, "expected": TurnDecision.WAIT, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "I want the sunroof version.", "silence": 600, "expected": TurnDecision.WAIT, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "How much is it?", "silence": 550, "expected": TurnDecision.WAIT, "dialect": "sandiego", "min_silence_override": 650},

        # ── SANDIEGO: MID-SENTENCE FRAGMENTS (WAIT) ──
        {"text": "Like, I'm looking for something that's", "silence": 680, "expected": TurnDecision.WAIT, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "Dude I just need something with", "silence": 690, "expected": TurnDecision.WAIT, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "It's like totally the right vibe but I need it to have", "silence": 670, "expected": TurnDecision.WAIT, "dialect": "sandiego", "min_silence_override": 650},

        # ── SANDIEGO: MULTI-SENTENCE COMPLETE (TALK) ──
        {"text": "I looked at the website. That deal is super legit.", "silence": 690, "expected": TurnDecision.TALK, "dialect": "sandiego", "min_silence_override": 650},
        {"text": "My roommate has one. I want the same color.", "silence": 680, "expected": TurnDecision.TALK, "dialect": "sandiego", "min_silence_override": 650},
    ]

    if dialect_filter:
        test_cases = [c for c in test_cases if c["dialect"] == dialect_filter]

    # Build one TurnDetector per unique (min_silence, hard_limit) combination
    # so we don't reload spaCy on every case.
    # Key insight: the default detector uses (800, 2000). Dialect overrides use
    # whatever min_silence_override specifies, keeping hard_limit at its default.
    _detector_cache: dict[int, TurnDetector] = {}

    def get_detector(case: dict) -> TurnDetector:
        threshold = case.get("min_silence_override", 800)
        if threshold not in _detector_cache:
            _detector_cache[threshold] = TurnDetector(min_silence=threshold)
        return _detector_cache[threshold]

    results = []
    dialect_stats: dict[str, dict] = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})
    total_latency = 0.0
    failures = []

    for case in test_cases:
        detector = get_detector(case)

        start = time.perf_counter()
        result = detector.evaluate(case["text"], case["silence"])
        latency_ms = (time.perf_counter() - start) * 1000

        total_latency += latency_ms
        passed = result == case["expected"]
        status = "✅ PASS" if passed else "❌ FAIL"
        dialect = case["dialect"]

        dialect_stats[dialect]["total"] += 1
        if passed:
            dialect_stats[dialect]["passed"] += 1
        else:
            dialect_stats[dialect]["failed"] += 1
            failures.append([
                dialect,
                f'"{case["text"][:45]}..."',
                result.name,
                case["expected"].name,
            ])

        results.append([status, f"{latency_ms:.2f}ms", case["text"][:40] + "...", result.name, dialect])

    n = len(test_cases)

    # ── Aggregate table ──
    print("\n=== AGGREGATE ===")
    print(tabulate(results, headers=["Status", "Latency", "Input Snippet", "Decision", "Dialect"]))

    total_passed = sum(s["passed"] for s in dialect_stats.values())
    avg_latency = total_latency / n if n else 0
    print(f"\nPassed: {total_passed}/{n}  |  Failed: {n - total_passed}/{n}  |  Avg Latency: {avg_latency:.2f}ms")

    # ── Per-dialect summary ──
    print("\n=== BY DIALECT ===")
    dialect_rows = []
    for d, s in sorted(dialect_stats.items()):
        rate = f"{100 * s['passed'] / s['total']:.1f}%" if s["total"] else "N/A"
        dialect_rows.append([d, s["passed"], s["failed"], s["total"], rate])
    print(tabulate(dialect_rows, headers=["Dialect", "Passed", "Failed", "Total", "Pass Rate"]))

    # ── Failures only ──
    if failures:
        print("\n=== FAILURES ===")
        print(tabulate(failures, headers=["Dialect", "Input Snippet", "Got", "Expected"]))
    else:
        print("\nAll cases passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dialect",
        choices=["neutral", "southern", "sandiego"],
        default=None,
        help="Run only cases for a specific dialect",
    )
    args = parser.parse_args()
    run_benchmarks(dialect_filter=args.dialect)
