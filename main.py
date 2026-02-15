import ollama
import pandas as pd
import time
import matplotlib as plt

# Data sets containing fact and opinion statements
facts = pd.read_csv("fact.csv")["claim"].tolist()     # Compiled and reformated by Claude.ai from FAViQ
opins = pd.read_csv("opinion.csv")["claim"].tolist()  # Created by Claude.ai
dtest = opins[0:2]
print(dtest)

# Setup human agent profile
ollama.create(
    model="human",
    from_="llama3.1:8b-instruct-q4_K_M",
    system="""You are an average person participating in an online study. Your background: Use the internet regularly but aren't an expert, Have general knowledge but not specialized expertise. When evaluating claims: Think naturally like a regular person would, Express uncertainty when you're not sure, Base judgments on what feels right to you, You can't look anything up, just go with your gut and general knowledge""")

# Functions to generate responses from human and AI agents

def genComment(prompt, detail, confidence):
    """
    Generate AI comment

    Params:
        prompt (string): claim used for comment
        detail (string): low or high for level of detail
        confidence (string): low, medium, high, for amount of confidence
    Return:
        string: AI comment for human agent to react to
    """
    return ollama.generate(
        model="llama3.1:8b-instruct-q4_K_M",
        prompt=f"""Generate a response to this claim with these exact specifications: Claim: {prompt}. Detail Level: {detail}, Low: Provide 1 citation, 1 key fact, and 0-1 statistics; High: Provide 2+ citations, 2+ key facts, 2+ statistics. Confidence Level: {confidence}, Low: Use hedging language ("might", "possibly", "I'm not certain"), Medium: Neutral, factual tone (no hedging or definitive statements), High: Definitive statements ("clearly", "proven", "without doubt"). Do not include requirements in you response. Provide your response now:""",
        options={
            "temperature": 0.25,    # Lower = more consistent
            "num_predict": 250      # Room for citations/fact
        })["response"]

def evaluate(history):
    """
    Evaluates belief from a 0-100 scale

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
            "num_predict": 1        # Gives buffer, prevents cutoff
        })['message']['content']


def beliefShift(prompt, comment):
    """
    Measures belief shift from prompt, detail, and confidence levels

    Params:
        prompt (string): claim used for comment
        comment (string): comment from AI agent
    Return:
        int tuple: first value is initial, second value is shift
    """
    history = [
        {"role":"user", "content":f"""You are presented with this claim: {prompt} Think through: What's your initial reaction? What do you know about this topic? How certain are you? Then provide your belief rating: 0 = completely certain it's FALSE/completely DISAGREE, 100 = completely certain it's TRUE/completely AGREE, 50 = equally likely to be TRUE or FALSE/uncertain. STRICT Format: Your response must only contain a number (1, 50, 99, 76)"""}]
    init = evaluate(history)
    history.append({"role" : "assistant", "content":init})
    history.append({"role":"user", "content":f"""You are now presented with an AI comment: {comment} Think through: Does the AI's comment change your view? How certain are you now? Then provide your belief rating: 0 = completely certain it's FALSE/completely DISAGREE, 100 = completely certain it's TRUE/completely AGREE, 50 = equally likely to be TRUE or FALSE/uncertain. STRICT Format: Your response must only contain a number (1, 50, 99, 76)"""})
    shift = evaluate(history)
    history.append({"role":"assistant", "content": shift})
    # print(history)
    return int(init), int(shift)

def genLevels(prompt):
    """
    Generate AI comments from levels of detail and confidence

    Params:
        prompt (string): claim used for comment
    Return:
        string list: list of comments representing all conditions
    """
    comments = []
    i = 0
    print(f"Generating comments for: {prompt}")
    for confidence in ["low", "medium", "high"]:
        for detail in ["low", "high"]:
            i +=1
            comments.append(genComment(prompt, detail, confidence))
            print(f"\tCondition {i} comment generated")
    return comments

def runShift(claims):
    """
    Runs the experiment using factual or subjective statements

    Param:
        claims (string list): list of claims to be ran
    Return:
        int tuple list: list of tuples representing belief shift
    """
    shifts = [[], [], [], [], [], []]
    for claim in claims:
        comments = genLevels(claim)
        c = 0
        print(f"Running claim: {claim}")
        for i in range(len(comments)):
            c += 1
            shifts[i].append(beliefShift(claim, comments[i]))
            print(f"\tCondition {c} ran")
    return shifts

def testExperiment():
    shifts = [[], [], [], [], [], []]
    start = time.time()
    for claim in dtest:
        comments = genLevels(claim)
        j = 0
        print(f"Running claim: {claim}")
        for i in range(len(comments)):
            j += 1
            shifts[i].append(beliefShift(claim, comments[i]))
            print(f"\tCondition {j} ran")
    print(time.time() - start)
    return shifts

def runExperiment():
    factShift = runShift(facts)
    opinShift = runShift(opins)

    # Merge shifts
    shifts = factShift

def genStats(shifts):
    # shift conditions: low,low; low,med; low,high; high,low; high,med; high,high
    return

# print(beliefShift("In a democratic system, all political parties should have equal funding and should conduct standardized election campaigns.","high","medium","agree"))
# print(beliefShift("The Summer Olympics in 2016 took place in Rio de Janeiro, Brazil.","low","low","none"))

# if __name__ == "__main__":
    # runExperiment()

print(testExperiment())