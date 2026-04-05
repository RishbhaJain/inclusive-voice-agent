import time
from tabulate import tabulate # pip install tabulate
from voicebot.turn_detector import TurnDetector, TurnDecision

def run_benchmarks():
    detector = TurnDetector()
    
    # 100 Test Cases categorized by intent
    test_cases = [
        # --- CLASSIC DANGLERS — ends on preposition/conjunction/det (Should WAIT) ---
        {"text": "I'm looking for a truck that's got", "silence": 900, "expected": TurnDecision.WAIT},
        {"text": "Can you check if the service department is", "silence": 1100, "expected": TurnDecision.WAIT},
        {"text": "I was thinking about coming in on", "silence": 850, "expected": TurnDecision.WAIT},
        {"text": "Does the Ford F-150 come with", "silence": 1200, "expected": TurnDecision.WAIT},
        {"text": "I'd like to get a vehicle that has", "silence": 950, "expected": TurnDecision.WAIT},
        {"text": "The car I'm interested in is a", "silence": 1000, "expected": TurnDecision.WAIT},
        {"text": "I need a truck that can tow up to", "silence": 1050, "expected": TurnDecision.WAIT},
        {"text": "I was wondering if you had anything in", "silence": 900, "expected": TurnDecision.WAIT},
        {"text": "Is the black one parked next to", "silence": 880, "expected": TurnDecision.WAIT},
        {"text": "My budget is around forty thousand and", "silence": 1100, "expected": TurnDecision.WAIT},
        {"text": "I'd prefer something with leather seats and", "silence": 950, "expected": TurnDecision.WAIT},
        {"text": "Are there any deals going on for", "silence": 1150, "expected": TurnDecision.WAIT},

        # --- COMPLETE SINGLE-SENTENCE QUERIES (Should TALK) ---
        {"text": "Is the service center open today?", "silence": 850, "expected": TurnDecision.TALK},
        {"text": "I want to see the Ford F-150.", "silence": 900, "expected": TurnDecision.TALK},
        {"text": "What are your hours on Sunday?", "silence": 800, "expected": TurnDecision.TALK},
        {"text": "I'd like to book a test drive for tomorrow.", "silence": 950, "expected": TurnDecision.TALK},
        {"text": "Do you have any trucks in stock?", "silence": 850, "expected": TurnDecision.TALK},
        {"text": "What's the price on that Silverado?", "silence": 900, "expected": TurnDecision.TALK},
        {"text": "I need an oil change today.", "silence": 1000, "expected": TurnDecision.TALK},
        {"text": "Can I schedule an appointment for Saturday?", "silence": 950, "expected": TurnDecision.TALK},
        {"text": "How much is the extended warranty?", "silence": 880, "expected": TurnDecision.TALK},
        {"text": "I'm interested in the white Chevy Tahoe.", "silence": 1100, "expected": TurnDecision.TALK},
        {"text": "My trade-in is a 2019 Honda Accord.", "silence": 950, "expected": TurnDecision.TALK},
        {"text": "I want something with good gas mileage.", "silence": 1200, "expected": TurnDecision.TALK},
        {"text": "Is financing available on that vehicle?", "silence": 900, "expected": TurnDecision.TALK},
        {"text": "I'd like to see the inventory for SUVs.", "silence": 850, "expected": TurnDecision.TALK},
        {"text": "Can you tell me about the warranty coverage?", "silence": 1000, "expected": TurnDecision.TALK},

        # --- MULTI-SENTENCE WITH COMPLETE LAST SENTENCE (Should TALK) ---
        {"text": "The truck looks great. I want to buy it.", "silence": 850, "expected": TurnDecision.TALK},
        {"text": "I saw the silver one. What's the price?", "silence": 900, "expected": TurnDecision.TALK},
        {"text": "We've been looking for a while. This is exactly what we need.", "silence": 950, "expected": TurnDecision.TALK},
        {"text": "My wife drives a sedan. I prefer an SUV though.", "silence": 1000, "expected": TurnDecision.TALK},
        {"text": "I was here last week. I'd like to test drive the Explorer.", "silence": 850, "expected": TurnDecision.TALK},
        {"text": "The salesperson showed us two options. I like the blue one.", "silence": 900, "expected": TurnDecision.TALK},
        {"text": "We need something safe for our kids. The Expedition looks perfect.", "silence": 1100, "expected": TurnDecision.TALK},
        {"text": "I've done my research online. I'm ready to make a deal.", "silence": 950, "expected": TurnDecision.TALK},
        {"text": "My current car has high mileage. I need a reliable replacement.", "silence": 880, "expected": TurnDecision.TALK},
        {"text": "I drove the Tacoma yesterday. I want to buy that one.", "silence": 1000, "expected": TurnDecision.TALK},

        # --- MULTI-SENTENCE WITH INCOMPLETE LAST SENTENCE (Should WAIT) ---
        {"text": "The truck looks great. But I was wondering if", "silence": 900, "expected": TurnDecision.WAIT},
        {"text": "I saw the silver one on the lot. Does it", "silence": 1000, "expected": TurnDecision.WAIT},
        {"text": "I love the color. But I really need something with", "silence": 950, "expected": TurnDecision.WAIT},
        {"text": "The price is decent. Could you check if it comes with", "silence": 1100, "expected": TurnDecision.WAIT},
        {"text": "My husband wants a truck. I personally prefer a", "silence": 850, "expected": TurnDecision.WAIT},
        {"text": "I talked to someone yesterday. They said the deal was good but", "silence": 1000, "expected": TurnDecision.WAIT},
        {"text": "This is our third dealership today. We're looking for something that", "silence": 900, "expected": TurnDecision.WAIT},
        {"text": "I appreciate the offer. I just need to think about", "silence": 950, "expected": TurnDecision.WAIT},
        {"text": "We've seen a few models. I was hoping to find one that has", "silence": 880, "expected": TurnDecision.WAIT},
        {"text": "That sounds reasonable. But can you also check on", "silence": 1050, "expected": TurnDecision.WAIT},

        # --- REGIONAL / TEXAS SPEECH FILLERS (Should WAIT) ---
        {"text": "Well I reckon I'm fixin to", "silence": 1000, "expected": TurnDecision.WAIT},
        {"text": "I was just over there by the", "silence": 900, "expected": TurnDecision.WAIT},
        {"text": "Shoot I just want something that'll", "silence": 950, "expected": TurnDecision.WAIT},
        {"text": "Y'all got anything that might be good for", "silence": 850, "expected": TurnDecision.WAIT},
        {"text": "Bless your heart I was just wondering about", "silence": 900, "expected": TurnDecision.WAIT},
        {"text": "Lord have mercy I need a truck that can", "silence": 1100, "expected": TurnDecision.WAIT},
        {"text": "Well now I'm not sure but I think it was", "silence": 950, "expected": TurnDecision.WAIT},
        {"text": "I tell you what, if you can get me something with", "silence": 1000, "expected": TurnDecision.WAIT},

        # --- HARD LIMIT CASES — silence >= 2000ms always forces TALK ---
        # These test the hard_limit branch, NOT the NLP. Even danglers should return TALK.
        {"text": "I need a car.", "silence": 2000, "expected": TurnDecision.TALK},
        {"text": "Looking for a truck with", "silence": 2500, "expected": TurnDecision.TALK},
        {"text": "My budget is", "silence": 3000, "expected": TurnDecision.TALK},
        {"text": "I want something that has good", "silence": 2200, "expected": TurnDecision.TALK},
        {"text": "Could you check if the deal includes", "silence": 2000, "expected": TurnDecision.TALK},

        # --- TOO-SHORT SILENCE — silence < 800ms always forces WAIT ---
        # These test the min_silence branch. Even complete sentences should return WAIT.
        {"text": "Is the service center open today?", "silence": 500, "expected": TurnDecision.WAIT},
        {"text": "I want to see the Ford F-150.", "silence": 300, "expected": TurnDecision.WAIT},
        {"text": "What are your hours?", "silence": 700, "expected": TurnDecision.WAIT},
        {"text": "I'd like to schedule an appointment.", "silence": 400, "expected": TurnDecision.WAIT},
        {"text": "Do you have any discounts available?", "silence": 600, "expected": TurnDecision.WAIT},

        # --- COMPLETE QUESTIONS (Should TALK) ---
        {"text": "Do you offer military discounts?", "silence": 900, "expected": TurnDecision.TALK},
        {"text": "How long does a typical oil change take?", "silence": 850, "expected": TurnDecision.TALK},
        {"text": "Is the Raptor available in blue?", "silence": 950, "expected": TurnDecision.TALK},
        {"text": "Can I trade in my current vehicle?", "silence": 1000, "expected": TurnDecision.TALK},
        {"text": "Are there any zero-percent financing options?", "silence": 880, "expected": TurnDecision.TALK},
        {"text": "What's included in the maintenance package?", "silence": 1100, "expected": TurnDecision.TALK},
        {"text": "Does this model come in all-wheel drive?", "silence": 900, "expected": TurnDecision.TALK},
        {"text": "How many miles does this truck have?", "silence": 950, "expected": TurnDecision.TALK},
        {"text": "Is there a certified pre-owned program?", "silence": 1000, "expected": TurnDecision.TALK},
        {"text": "Can you run my credit today?", "silence": 850, "expected": TurnDecision.TALK},

        # --- ENDS ON A VERB — the "hungry verb" check should catch these (Should WAIT) ---
        {"text": "My mechanic said the engine needs", "silence": 900, "expected": TurnDecision.WAIT},
        {"text": "I'm only here because my wife convinced", "silence": 850, "expected": TurnDecision.WAIT},
        {"text": "The salesman told me to come back and", "silence": 950, "expected": TurnDecision.WAIT},
        {"text": "I heard the promotion ended but maybe it continued", "silence": 1000, "expected": TurnDecision.WAIT},
        {"text": "I've been meaning to come in and", "silence": 950, "expected": TurnDecision.WAIT},

        # --- ADDITIONAL DANGLERS — ends with ADP/SCONJ (Should WAIT) ---
        {"text": "I'm looking at the inventory for", "silence": 900, "expected": TurnDecision.WAIT},
        {"text": "I was hoping to get something before", "silence": 850, "expected": TurnDecision.WAIT},
        {"text": "Is the truck available in", "silence": 900, "expected": TurnDecision.WAIT},
        {"text": "I was told to ask about", "silence": 1000, "expected": TurnDecision.WAIT},
        {"text": "I heard you were running a special on", "silence": 1000, "expected": TurnDecision.WAIT},

        # --- ADDITIONAL COMPLETE STATEMENTS (Should TALK) ---
        {"text": "The salesman was very helpful.", "silence": 950, "expected": TurnDecision.TALK},
        {"text": "My trade-in has about eighty thousand miles.", "silence": 1000, "expected": TurnDecision.TALK},
        {"text": "I'm pre-approved for fifty thousand dollars.", "silence": 950, "expected": TurnDecision.TALK},
        {"text": "The Expedition is perfect for my family.", "silence": 850, "expected": TurnDecision.TALK},
        {"text": "I just want something reliable.", "silence": 900, "expected": TurnDecision.TALK},

        # --- MULTI-SENTENCE COMPLETE FOLLOW-UPS (Should TALK) ---
        {"text": "I came in yesterday. Is the blue one still available?", "silence": 900, "expected": TurnDecision.TALK},
        {"text": "I checked your website. The deal online looks better.", "silence": 950, "expected": TurnDecision.TALK},
        {"text": "My lease is up in June. I need to find something soon.", "silence": 1000, "expected": TurnDecision.TALK},
        {"text": "The salesman gave me a quote. Can we do better than that?", "silence": 850, "expected": TurnDecision.TALK},
        {"text": "I test drove the Tundra last week. I'm ready to make a decision.", "silence": 900, "expected": TurnDecision.TALK},

        # --- BORDERLINE / STRESS CASES (Should WAIT) ---
        # Short silence right at the boundary
        {"text": "I need something before my warranty expires and", "silence": 800, "expected": TurnDecision.WAIT},
        # Sounds complete but is a trailing conjunction
        {"text": "The interior is nice but", "silence": 1100, "expected": TurnDecision.WAIT},
        # Mid-thought filler at boundary
        {"text": "I don't know, I was kind of thinking maybe something like", "silence": 850, "expected": TurnDecision.WAIT},
        # Fragment — no subject
        {"text": "Probably around next month or so", "silence": 1000, "expected": TurnDecision.TALK},
        # Ends with SCONJ "because"
        {"text": "I need that truck because", "silence": 950, "expected": TurnDecision.WAIT},
    ]

    results = []
    total_latency = 0

    for case in test_cases:
        start_time = time.perf_counter()
        
        # The actual logic call
        result = detector.evaluate(case["text"], case["silence"])
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        total_latency += latency_ms
        
        status = "✅ PASS" if result == case["expected"] else "❌ FAIL"
        results.append([status, f"{latency_ms:.2f}ms", case["text"][:40] + "...", result.name])

    # Print Summary Table
    print(tabulate(results, headers=["Status", "Latency", "Input Snippet", "Decision"]))
    n = len(test_cases)
    passed = sum(1 for r in results if r[0].startswith("✅"))
    print(f"\nPassed: {passed}/{n}  |  Failed: {n - passed}/{n}  |  Avg Latency: {total_latency/n:.2f}ms")

if __name__ == "__main__":
    run_benchmarks()