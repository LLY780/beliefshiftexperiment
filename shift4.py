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
# Category labels
beliefs = ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]

# Setup human agent profile
ollama.create(
    model="human",
    from_="llama3.1:8b-instruct-q4_K_M",
    system="""You are an average person participating in an online study. Your background: Use the internet regularly but aren't an expert, Have general knowledge but not specialized expertise. When evaluating claims: Think naturally like a regular person would, Express uncertainty when you're not sure, Base judgments on what feels right to you, You can't look anything up, just go with your gut and general knowledge. You can only respond in five different ways: strongly disagree, disagree, neutral, agree, or strongly agree and your response must only contain numbers 0 through 4 representing strongly disagree to strongly agree""")

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
    return ollama.chat(
        model="human",
        messages=history,
        options={
            "temperature": 0.75,    # Higher = more human-like variation
            "num_predict": 1        # Enough tokens for response
        })['message']['content']

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
        #todo

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

    # Not enough arguments
    if nargs < 4:
        print("Usage: python3 shift4.py [statement] [text] [technique] [sentiment] [framing] [flags]")
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

    if "-test" in args or nargs == 0:
        print("Running test...")
        s = time.time()
        run_test()
        print(f"Test took {time.time()-s} seconds")
        return

    if "-all" in args:
        print("Running all claims through all conditions...")
        run_all(facts, "fact_results.csv")
        run_all(opins, "opinion_results.csv")
        return

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