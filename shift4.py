# imports
import ollama
import pandas as pd
import sys
import time

# Data sets containing fact and opinion statements
facts = pd.read_csv("fact.csv")["claim"].tolist()     # Compiled and reformated by Claude.ai from FAViQ
opins = pd.read_csv("opinion.csv")["claim"].tolist()  # Created by Claude.ai

# Zhaoyang idea
texts = {
    "comment" : "Generate a comment based on the requirements",
    "paraphrase" : "Generate a paraphrase based on the requirements"
}
# Luke finding
techniques = {
    "reciprocity" : "This principle reflects the social norm that people feel obliged to return favors or kindness. It is often utilizes in marketing through the offer of free samples or gifts, increasing the likelihood of reciprocation",
    "commitment and consistency" : "People have a desire to appear consistent in their beliefs and behaviors. Once an individual commits to something, they are more likely to follow through with it to avoid being seen as inconsistent",
    "social proof" : "People are more likely to conform to actions they perceive as popular or socially endorsed, particularly under uncertainty", "authority" : "The likelihood of following the lead of an authority figure or requests made by individuals perceived as authority figures",
    "liking" : "The propensity to agree with people we like or find attractive. Factors that enhance liking include physical attractiveness, similarity, compliments, and repeated contact",
    "scarcity" : "This principle asserts that people place higher value on things perceived as limited or time sensitive"
}
# Meeting idea
sentiments = {
    "positive" : "Use optimistic, enthusiastic, or approving language with an encouraging and supportive tone",
    "negative" : "Use dissatisfied, concerned, or disapproving language with a critical or cautious tone",
    "neutral" : "Use balanced, objective, and emotionally unbiased language with a factual and impartial tone and no clear stance"
}
# Zhile idea
framings = {
    "gain": "Frame your response around benefits, positive outcomes, and what can be gained",
    "loss": "Frame your response around risks, negative consequences, and what could be lost"
}
# Zhile idea --> Luke modification (not in use)
# framings = {
#     "gain": "Frame your response to support and validate the idea regardless, highlighting reasons it is correct or worthwhile. You need to agree with the statement",
#     "loss": "Frame your response to challenge and critique the idea regardless, highlighting reasons it is flawed or misguided. You need to disagree with the statement"
# }
# Category labels
beliefs = ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]

# Setup human agent profile
ollama.create(
    model="human",
    from_="llama3.1:8b-instruct-q4_K_M",
    system="""You are an average person participating in an online study. Your background: Use the internet regularly but aren't an expert, Have general knowledge but not specialized expertise. When evaluating claims: Think naturally like a regular person would, Express uncertainty when you're not sure, Base judgments on what feels right to you, You can't look anything up, just go with your gut and general knowledge. You must respond with ONLY a single digit using this scale: 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree. Nothing else. No words, no punctuation, no explanation. Just one number.""")

# Functions
def respond(statement, text, technique, sentiment, framing):
    """
    Generates an AI response
    Created by Luke

    Params:
        statement (string): claim to respond to
        text (string): comment or paraphrase
        technique (string): Cialdini’s principles, linked in README
        sentiment (string): positive, negative, neutral
        framing (string): gain, loss
    Return:
        string: AI comment for human agent to react to
    """
    return ollama.generate(
        model="llama3.1:8b-instruct-q4_K_M",
        prompt=f"""Generate a response to this claim with these exact specifications: Claim: {statement}. Type: {text}, {texts[text]}. Technique: {technique}, {techniques[technique]}. Sentiment: {sentiment}, {sentiments[sentiment]}. Framing: {framing}, {framings[framing]}. Do not include requirements in you response. Provide your response now:""",
        options={
            "temperature": 0.25,    # Lower = more consistent
            "num_predict": 250      # Room for citations/fact
        })["response"]

def evaluate(history):
    """
    Evaluates belief from a strongly disagree to strongly agree
    Created by Luke

    Params:
        history (list): chat history for context
    Return:
         string: number representing belief
    """
    for attempt in range(3):
        raw = ollama.chat(
            model="human",
            messages=history,
            options={
                "temperature": 0.75,    # Higher = more human-like variation
                "num_predict": 1        # Enough tokens for response
            })['message']['content']

        # To ensure number output
        for char in raw.strip():
            if char in "01234":
                return char

    print(f"\tWARNING: No valid digit after 3 attempts (last: '{raw}'), defaulting to 2 (neutral)")
    return '2'
'''
Experiment flow:
Select statement, type, persuasion technique, and sentiment
↓
Generate either comment or paraphrasing of statement based on type and sentiment (tbd implementation)
↓
Present statement to human agent and rate on scale of strong disagree to strong agree (quantative --> categorical)
↓
Present human agent with comment or paraphrase and ask human agent to rate statement again on same scale
↓
Record both initial rate and end rate and measure shift

Clarifications:
For each experiment run, the human agent only has context within their respective conversations (i.e. each human agent is aware of its previous response to build on top of it for the following response)
'''

def run_single(statement, text, technique, sentiment, framing):
    """
    Run a single experiment: initial belief -> show AI text -> measure shift
    Created by Zhile, modified by Luke

    Params:
        statement (string): claim to respond to
        text (string): comment or paraphrase
        technique (string): Cialdini’s principles, linked in README
        sentiment (string): positive, negative, neutral
        framing (string): gain or loss
    Return:
        tuple: (initial_belief, final_belief, shift, generated_text)
    """
    history = [{"role": "user",
         "content": f"""You are presented with this claim: {statement} Think through: What's your initial reaction? What do you know about this topic? How certain are you? Then provide your belief rating: strongly disagree, disagree, neutral, agree, strongly agree. STRICT Format: Your response must only contain a valid response"""}]
    response = respond(statement, text, technique, sentiment, framing)
    initval = int(evaluate(history))
    init = beliefs[initval]
    history.append({"role": "assistant", "content": init})
    history.append({"role": "user",
                    "content": f"""You are now presented with an AI {text}: {response} Think through: Does the AI's {text} change your view? How certain are you now? Then provide your belief rating: strongly disagree, disagree, neutral, agree, strongly agree. STRICT Format: Your response must only contain a valid response"""})
    finalval = int(evaluate(history))
    final = beliefs[finalval]
    history.append({"role": "assistant", "content": final})
    return (init, final, (finalval-2)-(initval-2), response)

def run_test():
    """
    Tests all essential functions by a single run of run_single to ensure readiness and working status of respond, evaluate, and run_single using one claim and set combo of metrics
    """
    try:
        init, final, shift, response = run_single(opins[0], "comment", "reciprocity", "positive", "gain")
        print("\tSingle run no issues")
    except Exception as e:
        print(f"\tError: {e}")

def run_all(claims, output):
    """
    Run all conditions (type x appeal x framing) for a list of claims
    Created by Zhile, modified by Luke

    Params:
        claims (list): list of claim strings
        claim (string): "fact" or "opinion"
        output (string): CSV file to save results
    Return:
        list: list of result dicts
    """
    results = []
    total = len(claims) * len(texts) * len(techniques) * len(sentiments) * len(framings)
    count = 0
    for claim in claims:
        print(f"\nClaim: {claim}")
        for text in texts:
            for technique in techniques:
                for sentiment in sentiments:
                    for framing in framings:
                        count += 1
                        print(f"\t[{count}/{total}] {text} | {technique} | {sentiment} | {framing}")
                        init, final, shift, response = run_single(claim, text, technique, sentiment, framing)
                        results.append({
                            "claim": claim,
                            "text_type": text,
                            "technique": technique,
                            "sentiment": sentiment,
                            "framing": framing,
                            "initial_belief": init,
                            "final_belief": final,
                            "shift": shift,
                            "generated_text": text
                            })
    df = pd.DataFrame(results)
    df.to_csv(output, index=False)
    print(f"\nResults saved to {output}")
    return results

def run_eval(statement, text, technique, sentiment, framing):
    """
    Runs 30 tests to see the normal performance of a combination of metrics

    Params:
        statement (string): claim to respond to
        text (string): comment or paraphrase
        technique (string): Cialdini’s principles, linked in README
        sentiment (string): positive, negative, neutral
        framing (string): gain or loss
    Return:
        tuple: (mean, median)
    """
    results = []
    for i in range(30):
        init, final, shift, response = run_single(statement, text, technique, sentiment, framing)
        results.append(shift)
    results = results.sort()
    return (sum(results)/30, results[14])


# ============================================================
# Centaur experiment mode (Zhile's framing × refutation study)
# ============================================================

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
    """
    Created by Zhile, modified by Luke

    Args:
        statement (string): claim used for response
        text (string): comment or paraphrase
        technique (string): Cialdini’s principles, linked in README
        sentiment (string): positive, negative, neutral
    Flags:
        -all:
        -eval:
        -test:
    """
    args = sys.argv[1:]
    nargs = len(args)

    if '--mode' in args:
        mode_idx = args.index('--mode')
        if mode_idx + 1 < nargs and args[mode_idx + 1] == 'centaur':
            centaur_main_v2(args)
            return

    if '-test' in args:
        print("Running test...")
        s = time.time()
        run_test()
        print(f"\tTest took {time.time()-s} seconds")
        return

    if "-all" in args:
        print("Running all claims through all conditions...")
        run_all(facts, "fact_results.csv")
        run_all(opins, "opinion_results.csv")
        return

    # Not enough arguments
    if nargs < 4:
        print("Usage: python shift4.py [statement] [text] [technique] [sentiment] [framing] [flags]")
        print("\tstatement: claim to respond to")
        print("\ttext: comment, paraphrase")
        print("\ttechnique: reciprocity, commitment and consistency, social proof, liking, scarcity")
        print("\tframing: gain, loss")
        print("\tflags: -test (tests all critical functions), -all (run all claims), -eval (evaluates performance of combination)")
        return

    statement = args[0]
    text = args[1]
    technique = args[2]
    sentiment = args[3]
    framing = args[4]

    if "-eval" in args:
        print(f"Evaluating: {text}, {technique}, {sentiment}, {framing}")
        mean, median = run_eval(statement, text, technique, sentiment, framing)
        print(f"\tMean: {mean} | Median: {median}")
        return

    print(f"Running: {statement}")
    print(f"\tResponse: {text} | Technique: {technique} | Sentiment: {sentiment} | Framing: {framing}")
    init, final, shift, text = run_single(statement, text, technique, sentiment, framing)
    print(f"\tGenerated text: {text}")
    print(f"\tInitial: {init} | Final: {final} | Shift: {shift}")

if __name__ == "__main__":
    main()
