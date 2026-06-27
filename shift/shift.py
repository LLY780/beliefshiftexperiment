# IMPORTS
import os
import sys
import time

import ollama
from ollama import ResponseError
import pandas as pd
from pydantic import BaseModel

# GLOBAL VARS
model = "none"
claims = []
n = 30

# JSON Schema for structured output
class Belief(BaseModel):
    rating : int

# RESPONSE METRICS
COMMENT = "Write a direct reaction or opinion about the claim. Do not propose solutions, alternatives, or advice. Simply express a view on the claim as stated."

# Cialdani's Principles of Persuasion
TECHNIQUES = {
    "reciprocity": "Apply reciprocity by acknowledging the reader's perspective or conceding a point before presenting your position, creating a sense of give-and-take",
    "commitment and consistency": "Apply commitment and consistency by referencing positions or values the reader likely already holds, then showing how the claim aligns with those existing beliefs",
    "social proof": "Apply social proof by referencing what most people think, believe, or do in relation to the claim, implying that the popular view supports your position",
    "authority": "Apply authority by framing your response as informed or expert, citing reasoning or knowledge that positions your response as credible and well-founded",
    "liking": "Apply liking by using warm, relatable language that builds rapport with the reader before presenting your position",
    "scarcity": "Apply scarcity by framing the claim in terms of urgency or limited opportunity, suggesting that acting on or accepting the claim is time-sensitive",
    "none": "Do not apply any persuasion technique. Simply present the information plainly"
}

# Suggested by Professor Reza Zafarani to expand sentiments to have levels
SENTIMENTS = {
    "extremely positive": "Use euphoric, deeply passionate, or overwhelmingly appreciative language — bold superlatives, exclamations, and intense emotional warmth that leaves no doubt about enthusiasm",
    "moderately positive": "Use optimistic, enthusiastic, or approving language with an encouraging and supportive tone — clearly positive but measured and grounded",
    "mildly positive": "Use gently favorable or quietly approving language with a warm, understated tone — a slight lean toward the positive without overt enthusiasm",
    "neutral": "Use balanced, objective, and emotionally unbiased language with a factual and impartial tone and no clear stance",
    "mildly negative": "Use slightly dissatisfied or gently cautionary language with a mild critical tone — a subtle lean toward concern without strong condemnation",
    "moderately negative": "Use dissatisfied, concerned, or disapproving language with a clear critical or cautious tone — noticeably negative but not alarmist",
    "extremely negative": "Use intensely alarmed, strongly disapproving, or deeply frustrated language with an urgent, blunt tone — pointed criticism and emotional weight that signals serious concern"
}

# Measurable effect on opinion with a goal-oriented opinion
GOALS = {
    "pro": "You MUST argue in favor of the claim. Do not acknowledge opposing views or concede any points against it. Support the claim unconditionally.",
    "counter": "You MUST argue against the claim. Do not acknowledge supporting views or concede any points for it. Oppose the claim unconditionally.",
    "none": "You MUST take no position on the claim. Do not support or oppose the claim."
}

EVALUATE_SYSTEM = """You are a belief evaluator. You will be presented with a claim and asked to rate your belief in it on the following scale:
- 0 = completely certain it is FALSE / completely DISAGREE
- 50 = genuinely uncertain, equally likely to be true or false / neither AGREE nor DISAGREE
- 100 = completely certain it is TRUE / completely AGREE

Rules:
- Respond ONLY with a JSON object containing a single integer field "rating"
- Your rating must reflect your genuine position on the claim
- Do not explain your reasoning"""

# FUNCTIONS
def get_avgtime():
    times = 0.0
    response = respond("Heavy social media use is associated with increased rates of loneliness among adults.",
                       "social proof", "moderately positive", "pro")
    run_single("Heavy social media use is associated with increased rates of loneliness among adults.",response) # warm-up run
    for i in range(30):
        try:
            s = time.time()
            fuck, *shit = run_single("Heavy social media use is associated with increased rates of loneliness among adults.",response)
            curr = time.time()-s
            print(f"\tRun {i+1} took {two_decimal(curr)} seconds.")
            times += curr
        except Exception as e:
            print(f"\tError: {e}")
    return two_decimal(times/30) # Formats to 2 decimals

def model_check(model):
    try:
        ollama.show(model)
        return True # Model is found
    except ResponseError as e:
        if e.status_code == 404:
            print(f"{model} does not exist. Re-pull and check spelling.")
            raise # Model is not found
        print("Please ensure Ollama is running and installed properly.")
        raise e # Connection/API error

def two_decimal(num):
    return ((num*100)//1)/100

def respond(claim, technique, sentiment, goal):
    return ollama.generate(
        model=model,
        prompt=f"Your primary directive is the goal: {goal}, {GOALS[goal]}. Generate a response to this claim following this goal above all else. {COMMENT} Claim: {claim}. Technique: {technique}, {TECHNIQUES[technique]}. Sentiment: {sentiment}, {SENTIMENTS[sentiment]}. Your response must address the claim and only the claim. You may use framing, context, or rhetorical techniques to respond. You are forbidden from introducing unrelated topics, solutions, or information beyond the claim itself. Do not include requirements in your response. Provide your response now:",
        options={
            "num_predict": 350  # Room for citations/fact
        })["response"]

def evaluate(history):
    while True:
        try:
            result = ollama.chat(
                model=model,
                messages=history,
                format=Belief.model_json_schema(),
                options={
                    "num_predict": 20  # Enough tokens for JSON schema
                }

            )
            num = Belief.model_validate_json(result.message.content).rating
            if num in range(0, 101):
                return num
        except Exception:
            continue

def run_test():
    try:
        response = respond("Heavy social media use is associated with increased rates of loneliness among adults.","social proof","moderately positive","pro")
        init, final, shift, response = run_single("Heavy social media use is associated with increased rates of loneliness among adults.",response)
        print("Output:")
        print(f"\tinit:{init}")
        print(f"\tfinal:{final}")
        print(f"\tshift:{shift}")
        print(f"{response}")
    except Exception as e:
        print(f"\terror: {e}")

def run_single(claim, response):
    chat = [
        {"role":"system","content":EVALUATE_SYSTEM},
        {"role":"user","content":f"You are presented with this claim: {claim}. Rate your belief in the claim itself on this scale: 0 = completely certain it's FALSE/completely DISAGREE, 100 = completely certain it's TRUE/completely AGREE, 50 = equally likely to be TRUE or FALSE/uncertain. STRICT Format: Your response must only contain a single number 0-100."}
    ]
    init = evaluate(chat)
    chat.append({"role": "assistant", "content": str(init)})
    chat.append({"role":"user", "content":f"You are then presented with this response: {response}. After reading the response, rate your belief in the claim itself on this scale: 0 = completely certain it's FALSE/completely DISAGREE, 100 = completely certain it's TRUE/completely AGREE, 50 = equally likely to be TRUE or FALSE/uncertain. STRICT Format: Your response must only contain a single number 0-100."})
    final = evaluate(chat)
    shift = final-init
    return init, final, shift, response

def run_eval(claim, technique, sentiment, goal):
    inits = []
    finals = []
    shifts = []
    responses = []
    print("Evaluating: ",end="",flush=True)
    response = respond(claim,technique,sentiment,goal)
    for i in range(n):
        init, final, shift, response = run_single(claim,response)
        print("#", end="", flush=True)
        inits.append(init)
        finals.append(final)
        shifts.append(shift)
    print()
    abs_inits = [abs(j) for j in inits]
    abs_finals = [abs(j) for j in finals]
    abs_shifts = [abs(j) for j in shifts]
    return inits, finals, shifts, abs_inits, abs_finals, abs_shifts, response

def run_all(export):
    # Run vars
    results = [] # List of dicts representing a csv entry
    stats = [] # List of dicts representing a csv entry
    total = len(claims)*len(TECHNIQUES)*len(SENTIMENTS)*len(GOALS)
    count = 0
    clean = export.replace(":", "_").replace("/", "_")
    results_file = clean + ".results.csv"
    stats_file = clean + ".stats.csv"

    # Checkpoint system
    completed = set()
    if os.path.exists(results_file):
        existing = pd.read_csv(results_file)
        results = existing.to_dict("records")
        for _, row in existing.iterrows():
            completed.add((row["claim"],row["technique"],row["sentiment"],row["goal"]))
        print(f"\tResuming from {len(completed)} completed combinations")
    if os.path.exists(stats_file):
        existing_stats = pd.read_csv(stats_file)
        stats = existing_stats.to_dict("records")

    # Start testing claims
    for claim in claims:
        print(f"\nClaim: {claim}")
        for technique in TECHNIQUES:
            for sentiment in SENTIMENTS:
                for goal in GOALS:
                    count += 1
                    # If combo already tested
                    if (claim,technique,sentiment,goal) in completed:
                        print(f"\t[{count}/{total}] Completed - {technique} | {sentiment} | {goal}")
                        continue
                    # Else resume testing
                    print(f"\t[{count}/{total}] | {technique} | {sentiment} | {goal}", end="\n\t")
                    inits, finals, shifts, abs_inits, abs_finals, abs_shifts, response = run_eval(claim,technique,sentiment,goal)
                    # Results compiling
                    for i in range(n):
                        results.append({
                            "claim":claim,
                            "technique":technique,
                            "sentiment":sentiment,
                            "goal":goal,
                            "init":inits[i],
                            "final":finals[i],
                            "shift":shifts[i],
                            "response":response
                        })
                    # Stats compiling
                    stats.append({
                        "claim":claim,
                        "technique":technique,
                        "sentiment":sentiment,
                        "goal":goal,
                        "mean_init":sum(inits)/n,
                        "mean_final":sum(finals)/n,
                        "mean_shift":sum(shifts)/n,
                        "abs_mean_init":sum(inits)/n,
                        "abs_mean_final":sum(finals)/n,
                        "abs_mean_shift":sum(shifts)/n
                    })
                    # Incremental save after every combination
                    pd.DataFrame(results).to_csv(results_file, index=False)
                    pd.DataFrame(stats).to_csv(stats_file, index=False)

    # End of run. All progress compiled and saved
    print(f"\nResults saved to {results_file}")
    print(f"\nStats saved to {stats_file}")

def main():
    args = sys.argv[1:]
    nargs = len(args)
    # Usage: python(3) shift.py [claims csv] [ollama model] [run type] [optional flags]

    # Help
    if "-h" in args or "--help" in args:
        print("Usage: python(3) shift.py [claims] [ollama model] [run type] [run flags] [optional flags]")
        print("Run Types:\n\ttest: none\n\ttime: none\n\tsingle: claim, technique, sentiment, goal\n\teval: claim, technique, sentiment, goal\n\tall: output name")
        print("Optional flags:\n\t-c, --claims [start] [end]: Select the range of claims used in testing. (i.e. 23 to 75)\n\t-h, --help: Print the usage.\n\t-n, --runs [n]: Set the amount of runs used in evaluations. (default is 30)\n\t-s, --sample [seed] [n]: Sample a number of claims using a random seed. (i.e. 42 & 10)")
        print("\tNote: if sample and claims are used together, claims will be used first.\n")

    # Required args
    if nargs < 3:
        print(f"Current args: {args}\nMissing required args. Use -h, --help to view the usage.")
        return
    global model, claims, n
    fclaims, model, run, *flags = args
    df = pd.read_csv(fclaims)
    claims = df["claim"].tolist()
    model_check(model)

    # Claims check
    if len(claims) < 1:
        print("Given claims file is empty!")
        return

    # Flag checks
    if "-c" in flags or "--claims" in flags:
        if "-c" in flags:
            idx = flags.index("-c")
        else:
            idx = flags.index("--claims")
        start = int(flags[idx+1])
        end = int(flags[idx+2])+1
        claims = claims[start:end]

    if "-n" in flags or "--runs" in flags:
        if "-n" in flags:
            idx = flags.index("-n")
        else:
            idx = flags.index("--runs")
        n = int(flags[idx+1])

    if "-s" in flags or "--sample" in flags:
        if "-s" in flags:
            idx = flags.index("-s")
        else:
            idx = flags.index("--sample")
        seed = int(flags[idx+1])
        size = int(flags[idx+2])
        sampled = df.sample(size,random_state=seed)
        claims = sampled["claim"].tolist()

    # Run types
    if "test" == run:
        print("Testing response and evaluation...")
        run_test()

    if "time" == run:
        print("Getting the average runtime...")
        t = get_avgtime()
        print(f"The average runtime is {t}s")

    if "single" == run:
        claim = flags[0]
        technique = flags[1]
        sentiment = flags[2]
        goal = flags[3]
        print("Running single combo...")
        response = respond(claim,technique,sentiment,goal)
        run_single(claim,response)

    if "eval" == run:
        claim = flags[0]
        technique = flags[1]
        sentiment = flags[2]
        goal = flags[3]
        print("Evaluating current combo...")
        run_eval(claim,technique,sentiment,goal)

    if "all" == run:
        export = flags[0]
        runtime = get_avgtime()
        totaltime = (int(runtime)*len(claims)*len(TECHNIQUES)*len(SENTIMENTS)*len(GOALS)*n)/60/60
        print(f"Estimated total time: {two_decimal(totaltime)} hours.")
        print("Evaluating all combos against all claims...")
        run_all(export)

if __name__ == "__main__":
    main()

#todo
# Add descriptions to all functions