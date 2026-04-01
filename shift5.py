# imports
import ollama
import pandas as pd
import sys
import time

# Data sets containing fact and opinion statements
facts = pd.read_csv("fact.csv")["claim"].tolist()     # Compiled and reformated by Claude.ai from FAViQ
opins = pd.read_csv("opinion.csv")["claim"].tolist()  # Created by Claude.ai

texts = {
    "comment" : "Generate a comment based on the requirements",
    "paraphrase" : "Generate a paraphrase based on the requirements"
}
centaur_framings = {
    "gain": "Frame your response around benefits, positive outcomes, and what can be gained",
    "loss": "Frame your response around risks, negative consequences, and what could be lost",
    "none": "Respond naturally without any specific framing direction"
}

centaur_refutations = {
    "one-sided": "Present only arguments that support your position. Do not mention, acknowledge, or address any counterarguments or opposing views.",
    "two-sided": "First acknowledge the strongest counterargument against your position, then refute it with evidence before presenting your main argument.",
    "none": "Respond naturally without any specific argumentative structure."
}

# ============================================================
# Centaur experiment mode (Zhile's framing × refutation study)
# ============================================================

def centaur_respond(claim, text_type, framing, refutation):
    """Generate AI persuasive text for centaur experiment."""
    return ollama.generate(
        model="llama3.1:8b-instruct-q4_K_M",
        prompt=(
            f"Generate a response to this claim: {claim}. "
            f"Type: {text_type}, {texts[text_type]}. "
            f"Framing: {centaur_framings[framing]}. "
            f"Refutation style: {centaur_refutations[refutation]}. "
            f"Do not include requirements in your response. Provide your response now:"
        ),
        options={"temperature": 0.25, "num_predict": 250}
    )["response"]


def centaur_evaluate(history):
    """Get belief rating from Centaur model (temperature=0 for determinism)."""
    for attempt in range(3):
        raw = ollama.chat(
            model="centaur",
            messages=history,
            options={"temperature": 0, "num_predict": 1}
        )['message']['content']
        for char in raw.strip():
            if char in "01234":
                return int(char)

    print(f"\tWARNING: No valid digit after 3 attempts (last: '{raw}'), defaulting to 2 (neutral)")
    return 2


def centaur_run_trial(claim, text_type, framing, refutation):
    """Run one experimental trial: initial belief -> show AI text -> final belief."""
    history = [{"role": "user",
        "content": (
            f"You are presented with this claim: {claim} "
            "Rate your belief: 0=strongly disagree, 1=disagree, 2=neutral, "
            "3=agree, 4=strongly agree. Respond with ONLY a single digit."
        )}]
    initial_belief = centaur_evaluate(history)
    gen_text = centaur_respond(claim, text_type, framing, refutation)
    history.append({"role": "assistant", "content": str(initial_belief)})
    history.append({"role": "user",
        "content": (
            f"Now consider this AI {text_type}: {gen_text} "
            "Rate your belief again: 0=strongly disagree, 1=disagree, 2=neutral, "
            "3=agree, 4=strongly agree. Respond with ONLY a single digit."
        )})
    final_belief = centaur_evaluate(history)
    return initial_belief, final_belief, final_belief - initial_belief, gen_text


def centaur_run_control(claim):
    """Control trial: ask belief twice with no AI text (shift should be 0)."""
    history = [{"role": "user",
        "content": (
            f"You are presented with this claim: {claim} "
            "Rate your belief: 0=strongly disagree, 1=disagree, 2=neutral, "
            "3=agree, 4=strongly agree. Respond with ONLY a single digit."
        )}]
    initial_belief = centaur_evaluate(history)
    history.append({"role": "assistant", "content": str(initial_belief)})
    history.append({"role": "user",
        "content": (
            f"Rate your belief on the same claim again: {claim} "
            "0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree. "
            "Respond with ONLY a single digit."
        )})
    final_belief = centaur_evaluate(history)
    return initial_belief, final_belief, final_belief - initial_belief


def centaur_run_all(n=None):
    """Run the full centaur experiment across all claims and conditions."""
    facts_df = pd.read_csv("fact.csv")
    opins_df = pd.read_csv("opinion.csv")
    claims = []
    for _, row in facts_df.iterrows():
        claims.append((row["claim"], "fact"))
    for _, row in opins_df.iterrows():
        claims.append((row["claim"], "opinion"))
    if n is not None:
        claims = claims[:n]

    results = []
    gen_texts = []
    total = len(claims) * 19
    count = 0

    for claim, claim_type in claims:
        print(f"\nClaim ({claim_type}): {claim}")
        # Control condition (no AI text)
        count += 1
        print(f"\t[{count}/{total}] control (no AI text)")
        init, final, shift = centaur_run_control(claim)
        results.append({
            "claim": claim, "claim_type": claim_type,
            "text_type": "none", "framing": "none", "refutation": "none",
            "initial_belief": init, "final_belief": final, "shift": shift
        })
        if shift != 0:
            print(f"\t  WARNING: control shift={shift} (expected 0)")
        # Experimental conditions: 2 text_types × 3 framings × 3 refutations = 18
        for text_type in texts:
            for framing in centaur_framings:
                for refutation in centaur_refutations:
                    count += 1
                    print(f"\t[{count}/{total}] {text_type} | {framing} | {refutation}")
                    init, final, shift, gen_text = centaur_run_trial(
                        claim, text_type, framing, refutation)
                    results.append({
                        "claim": claim, "claim_type": claim_type,
                        "text_type": text_type, "framing": framing,
                        "refutation": refutation,
                        "initial_belief": init, "final_belief": final, "shift": shift
                    })
                    gen_texts.append({
                        "claim": claim, "text_type": text_type,
                        "framing": framing, "refutation": refutation,
                        "generated_text": gen_text
                    })

    pd.DataFrame(results).to_csv("centaur_results.csv", index=False)
    pd.DataFrame(gen_texts).to_csv("centaur_generated_texts.csv", index=False)
    print(f"\nResults saved to centaur_results.csv ({len(results)} rows)")
    print(f"Generated texts saved to centaur_generated_texts.csv ({len(gen_texts)} rows)")


def centaur_main(args):
    """Entry point for centaur mode CLI routing."""
    if '-test' in args:
        print("Running centaur sanity check (2 claims)...")
        s = time.time()
        centaur_run_all(n=2)
        print(f"\nSanity check completed in {time.time()-s:.1f}s")
        return
    n = None
    if '-n' in args:
        n = int(args[args.index('-n') + 1])
    print(f"Running centaur experiment{f' (first {n} claims)' if n else ' (all claims)'}...")
    s = time.time()
    centaur_run_all(n=n)
    print(f"\nExperiment completed in {time.time()-s:.1f}s")


# ============================================================
# Two-phase centaur execution (--persuader support)
# ============================================================

CENTAUR_PERSUADER_DEFAULT = "llama3.1:8b-instruct-q4_K_M"

CENTAUR_PERSUADER_ALIASES = {
    "70b": "llama3.1:70b",
    "8b": "llama3.1:8b-instruct-q4_K_M",
}


def centaur_respond_with_model(claim, text_type, framing, refutation, model):
    """Generate AI persuasive text using a specified model."""
    return ollama.generate(
        model=model,
        prompt=(
            f"Generate a response to this claim: {claim}. "
            f"Type: {text_type}, {texts[text_type]}. "
            f"Framing: {centaur_framings[framing]}. "
            f"Refutation style: {centaur_refutations[refutation]}. "
            f"Do not include requirements in your response. Provide your response now:"
        ),
        options={"temperature": 0.25, "num_predict": 250}
    )["response"]


def centaur_eval_with_text(claim, text_type, gen_text):
    """Evaluate belief shift using pre-generated AI text."""
    history = [{"role": "user",
        "content": (
            f"You are presented with this claim: {claim} "
            "Rate your belief: 0=strongly disagree, 1=disagree, 2=neutral, "
            "3=agree, 4=strongly agree. Respond with ONLY a single digit."
        )}]
    initial_belief = centaur_evaluate(history)
    history.append({"role": "assistant", "content": str(initial_belief)})
    history.append({"role": "user",
        "content": (
            f"Now consider this AI {text_type}: {gen_text} "
            "Rate your belief again: 0=strongly disagree, 1=disagree, 2=neutral, "
            "3=agree, 4=strongly agree. Respond with ONLY a single digit."
        )})
    final_belief = centaur_evaluate(history)
    return initial_belief, final_belief, final_belief - initial_belief


def centaur_phase1_generate(n=None, persuader_model=CENTAUR_PERSUADER_DEFAULT):
    """Phase 1: Generate all persuasive texts and save to CSV."""
    facts_df = pd.read_csv("fact.csv")
    opins_df = pd.read_csv("opinion.csv")
    claims = []
    for _, row in facts_df.iterrows():
        claims.append((row["claim"], "fact"))
    for _, row in opins_df.iterrows():
        claims.append((row["claim"], "opinion"))
    if n is not None:
        claims = claims[:n]

    gen_texts = []
    total = len(claims) * len(texts) * len(centaur_framings) * len(centaur_refutations)
    count = 0

    for claim, claim_type in claims:
        print(f"\nClaim ({claim_type}): {claim}")
        for text_type in texts:
            for framing in centaur_framings:
                for refutation in centaur_refutations:
                    count += 1
                    print(f"\t[{count}/{total}] {text_type} | {framing} | {refutation}")
                    gen_text = centaur_respond_with_model(
                        claim, text_type, framing, refutation, persuader_model)
                    gen_texts.append({
                        "claim": claim, "text_type": text_type,
                        "framing": framing, "refutation": refutation,
                        "generated_text": gen_text
                    })

    pd.DataFrame(gen_texts).to_csv("centaur_generated_texts.csv", index=False)
    print(f"\nPhase 1 complete: {len(gen_texts)} texts saved to centaur_generated_texts.csv")


def centaur_phase2_evaluate(n=None):
    """Phase 2: Load generated texts and run Centaur evaluation."""
    gen_df = pd.read_csv("centaur_generated_texts.csv")

    facts_df = pd.read_csv("fact.csv")
    opins_df = pd.read_csv("opinion.csv")
    claims = []
    for _, row in facts_df.iterrows():
        claims.append((row["claim"], "fact"))
    for _, row in opins_df.iterrows():
        claims.append((row["claim"], "opinion"))
    if n is not None:
        claims = claims[:n]

    results = []
    total = len(claims) * 19
    count = 0

    for claim, claim_type in claims:
        print(f"\nClaim ({claim_type}): {claim}")
        # Control condition
        count += 1
        print(f"\t[{count}/{total}] control (no AI text)")
        init, final, shift = centaur_run_control(claim)
        results.append({
            "claim": claim, "claim_type": claim_type,
            "text_type": "none", "framing": "none", "refutation": "none",
            "initial_belief": init, "final_belief": final, "shift": shift
        })
        if shift != 0:
            print(f"\t  WARNING: control shift={shift} (expected 0)")
        # Experimental conditions using pre-generated texts
        for text_type in texts:
            for framing in centaur_framings:
                for refutation in centaur_refutations:
                    count += 1
                    print(f"\t[{count}/{total}] {text_type} | {framing} | {refutation}")
                    match = gen_df[
                        (gen_df["claim"] == claim) &
                        (gen_df["text_type"] == text_type) &
                        (gen_df["framing"] == framing) &
                        (gen_df["refutation"] == refutation)
                    ]
                    gen_text = match.iloc[0]["generated_text"]
                    init, final, shift = centaur_eval_with_text(
                        claim, text_type, gen_text)
                    results.append({
                        "claim": claim, "claim_type": claim_type,
                        "text_type": text_type, "framing": framing,
                        "refutation": refutation,
                        "initial_belief": init, "final_belief": final, "shift": shift
                    })

    pd.DataFrame(results).to_csv("centaur_results.csv", index=False)
    print(f"\nPhase 2 complete: {len(results)} results saved to centaur_results.csv")


def centaur_main_v2(args):
    """Entry point for centaur mode with --persuader support."""
    persuader_model = CENTAUR_PERSUADER_DEFAULT
    if '--persuader' in args:
        idx = args.index('--persuader')
        raw = args[idx + 1]
        persuader_model = CENTAUR_PERSUADER_ALIASES.get(raw, raw)

    # Default persuader: use original single-phase approach
    if persuader_model == CENTAUR_PERSUADER_DEFAULT:
        centaur_main(args)
        return

    # Non-default persuader: two-phase approach
    n = None
    if '-n' in args:
        n = int(args[args.index('-n') + 1])
    if '-test' in args:
        n = 2

    print(f"Persuader model: {persuader_model}")
    print(f"Running two-phase centaur experiment{f' (first {n} claims)' if n else ' (all claims)'}...")
    s = time.time()

    print("\n=== Phase 1: Generating persuasive texts ===")
    centaur_phase1_generate(n=n, persuader_model=persuader_model)

    print("\n=== Phase 2: Evaluating with Centaur ===")
    centaur_phase2_evaluate(n=n)

    print(f"\nExperiment completed in {time.time()-s:.1f}s")

def main():
    args = sys.argv[1:]
    nargs = len(args)

    mode_idx = args.index('--mode')
    if mode_idx + 1 < nargs and args[mode_idx + 1] == 'centaur':
        centaur_main_v2(args)

if __name__ == "__main__":
    main()