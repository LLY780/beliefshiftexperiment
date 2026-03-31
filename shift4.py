# imports
from unittest import removeResult

import ollama
import pandas as pd
import sys

# Data sets containing fact and opinion statements
facts = pd.read_csv("fact.csv")["claim"].tolist()     # Compiled and reformated by Claude.ai from FAViQ
opins = pd.read_csv("opinion.csv")["claim"].tolist()  # Created by Claude.ai
# dtest = opins[0:2]
# print(dtest)

types = {
    "comment" : "Generate a comment based on the requirements",
    "paraphrase" : "Generate a paraphrase based on the requirements"
}
techniques = {
    "reciprocity" : "This principle reflects the social norm that people feel obliged to return favors or kindness. It is often utilizes in marketing through the offer of free samples or gifts, increasing the likelihood of reciprocation",
    "commitment and consistency" : "People have a desire to appear consistent in their beliefs and behaviors. Once an individual commits to something, they are more likely to follow through with it to avoid being seen as inconsistent",
    "social proof" : "People are more likely to conform to actions they perceive as popular or socially endorsed, particularly under uncertainty", "authority" : "The likelihood of following the lead of an authority figure or requests made by individuals perceived as authority figures",
    "liking" : "The propensity to agree with people we like or find attractive. Factors that enhance liking include physical attractiveness, similarity, compliments, and repeated contact",
    "scarcity" : "This principle asserts that people place higher value on things perceived as limited or time sensitive"
}
sentiments = {
    "positive" : "Use optimistic, enthusiastic, or approving language with an encouraging and supportive tone",
    "negative" : "Use dissatisfied, concerned, or disapproving language with a critical or cautious tone", "neutral" : "Use balanced, objective, and emotionally unbiased language with a factual and impartial tone and no clear stance"
}
beliefs = ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]

# Setup human agent profile
ollama.create(
    model="human",
    from_="llama3.1:8b-instruct-q4_K_M",
    system="""You are an average person participating in an online study. Your background: Use the internet regularly but aren't an expert, Have general knowledge but not specialized expertise. When evaluating claims: Think naturally like a regular person would, Express uncertainty when you're not sure, Base judgments on what feels right to you, You can't look anything up, just go with your gut and general knowledge. You can only respond in five different ways: strongly disagree, disagree, neutral, agree, strongly agreee.""")

# Functions
def respond(statement, type, technique, sentiment):
    """
    Generate AI response

    Params:
        prompt (string): claim used for comment
        detail (string): low or high for level of detail
        confidence (string): low, medium, high, for amount of confidence
    Return:
        string: AI comment for human agent to react to
    """
    return ollama.generate(
        model="llama3.1:8b-instruct-q4_K_M",
        prompt=f"""Generate a response to this claim with these exact specifications: Claim: {statement}. Type: {type}, {types[type]}. Technique: {technique}, {techniques[technique]}. Sentiment: {sentiment}, {sentiments[sentiment]}. Do not include requirements in you response. Provide your response now:""",
        options={
            "temperature": 0.25,    # Lower = more consistent
            "num_predict": 250      # Room for citations/fact
        })["response"]

def evaluate(history):
    """
    Evaluates belief from a strongly disagree to strongly agree

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
            "num_predict": 5        # Enough tokens for response
        })['message']['content']

# def default(statement, type, technique, sentiment):
#     history = history = [
#         {"role": "user",
#          "content": f"""You are presented with this claim: {statement} Think through: What's your initial reaction? What do you know about this topic? How certain are you? Then provide your belief rating: strongly disagree, disagree, neutral, agree, strongly agreee. STRICT Format: Your response must only contain a valid response"""}]
#     response = respond(statement, type, technique, sentiment)
#     init = evaluate(history)
#     history.append({"role": "assistant", "content": init})
#     history.append({"role": "user",
#                     "content": f"""You are now presented with an AI comment: {response} Think through: Does the AI's comment change your view? How certain are you now? Then provide your belief rating: strongly disagree, disagree, neutral, agree, strongly agreee. STRICT Format: Your response must only contain a valid response"""})
#     shift = evaluate(history)
#     history.append({"role": "assistant", "content": shift})
#     return (beliefs.index(init.lower())-2, beliefs.index(shift.lower())-2, response)
#
# def all(function, type, technique, sentiment):
#     results = {}
#     for statement in facts:
#         results[statement] = function(statement, type, technique, sentiment)
#     for statement in opins:
#         results[statement] = function(statement, type, technique, sentiment)
#     return results.values()
#
# def compare(statement, type, technique, sentiment):
#     # todo
#     return
#
# def t1(statement, type, technique, sentiment):
#     # todo
#     return
#
# def t2(statement, type, technique, sentiment):
#     # todo
#     return
#
# def t3(statement, type, technique, sentiment):
#     print(statement)
#     # type, technique, sentiment not needed. just there to use all() on
#     results = {}
#     for type in types.keys():
#         for technique in techniques.keys():
#             for sentiment in sentiments.keys():
#                 print("\t",type, technique, sentiment)
#                 results[(type, technique, sentiment)] = default(statement, type, technique, sentiment)
#     print(results) # prints to see responses if needed
#
#     shifts = {}
#     for key in results:
#         init, shift, response = results[key]
#         shifts[key] = shift - init
#     return shifts.values()




# Experiment flow
# Select statement, type, persuasion technique, and sentiment
# ↓
# Generate either comment or paraphrasing of statement based on type and sentiment (tbd implementation)
# ↓
# Present statement to human agent and rate on scale of strong disagree to strong agree (quantative --> categorical)
# ↓
# Present human agent with comment or paraphrase and ask human agent to rate statement again on same scale
# ↓
# Record both initial rate and end rate and measure shift

def main():
    args = sys.argv[1:]

    if len(args) == 0 or "-test" in args:
        dtest()
        return

    if len(args) < 4:
        print("Usage: python myshift.py [claim] [type] [appeal] [framing]")
        print("  type: comment, paraphrase")
        print("  appeal: emotional, logical")
        print("  framing: gain, loss")
        print("  Flags: -test (run test on 2 claims), -all (run all claims)")
        return

    statement = args[0]
    type = args[1]
    appeal = args[2]
    framing = args[3]

    if "-all" in args:
        print("Running all claims through all conditions...")
        run_all_conditions(facts, "fact", "fact_results.csv")
        run_all_conditions(opins, "opinion", "opinion_results.csv")
        return

    print(f"Running: {statement}")
    print(f"  Type: {type} | Appeal: {appeal} | Framing: {framing}")
    init, final, shift, text = run_single(statement, type, appeal, framing)
    print(f"  Generated text: {text}")
    print(f"  Initial: {init} | Final: {final} | Shift: {shift}")

if __name__ == "__main__":
    main()


# def main():
#     # Arguments: statement, type, technique, sentiment
#         # statement: statement to be evalauted on
#         # type: # comment or paraphrase
#         # technique: reciprocity, commitment and consistency, social proof, authority, liking, scarcity, none
#         # sentiment: positive, negative, neutral, none
#     # Flags: -h (history) -c (compare) -a (all) -t1 (test1) -t2 (test2)
#         # compare is used to compare two metrics against each other in a one-run evaluation
#         # all is used to run all statements through test
#         # test1 is a comparative test with one agent trying all different methods
#         # test2 is a evaluation test to see the normal performance of a combination of metrics, with 30 evaluations across different agents
#         # test3 is a comprehensive evaluation of all metrics
#     # Default behavior: run a test with the given arguments and display the shift
#     args = sys.argv[1:]
#     statement = args[0]
#     type = args[1]
#     technique = args[2]
#     sentiment = args[3]
#     history = history = [
#         {"role" : "user", "content" : f"""You are presented with this claim: {statement} Think through: What's your initial reaction? What do you know about this topic? How certain are you? Then provide your belief rating: strongly disagree, disagree, neutral, agree, strongly agreee. STRICT Format: Your response must only contain a valid response"""}]
#     nargs = len(args)
#     if nargs < 4:
#         # Not enough args
#         print("Please add all arguments")
#     elif nargs == 4:
#         # Default test
#         if "-a" in args:
#             print(all(default, type, technique, sentiment))
#             return
#         print(default(statement, type, technique, sentiment))
#     else:
#         # Flags
#         # if "-h" in args:
#         #     history = [args.index("-h")+1]
#             # will implement later
#             # supposed to carry over any chat history
#         # if "-c" in args:
#         #     if "-a" in args:
#         #         all(compare, type, technique, sentiment)
#         #         return
#         #     print(compare())
#             # will implement later
#             # too busy to create or figure out how to repurpose all() for compare()
#         if "-t1" in args:
#             if "-a" in args:
#                 all(t1, type, technique, sentiment)
#                 return
#             print(t1(statement, type, technique, sentiment))
#         elif "-t2" in args:
#             if "-a" in args:
#                 all(t2, type, technique, sentiment)
#                 return
#             print(t2(statement, type, technique, sentiment))
#         elif "-t3" in args:
#             if "-a" in args:
#                 all(t3, type, technique, sentiment)
#                 return
#             print(t3(statement, type, technique, sentiment))
#         elif "-test" in args:
#             print("test flag")
#
# if (__name__ == "__main__"):
#     main()

# notes for reimplementation
    # for command line, make vars dependent on which test
    # unreasonable to add filler data for test that doesnt use it