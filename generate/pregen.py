# IMPORTS
import os
import sys

import ollama
from ollama import ResponseError
import pandas as pd
from pydantic import BaseModel
from typing import Literal

# GLOBAL VARS
model = "none"
per = 8  # 8 empirical, 8 normative
output = "preset_claims.csv"
seed = 42

# JSON Schema for structured output
class Claim(BaseModel):
    claim: str
    type: Literal["empirical", "normative"]
    dimension: str

class ClaimList(BaseModel):
    claims: list[Claim]

# Domains and Subtopics
DOMAINS = {
    "politics": ["Electoral systems","Political parties","Immigration policy","Freedom of speech","Government surveillance","Campaign finance","Voter participation","National security","International relations","Separation of church and state","Civil liberties","Lobbying","Term limits","Media and politics","Populism","Political polarization","Protest and civil disobedience","Constitutionalism","Foreign aid","Political corruption"],
    "ethics": ["Capital punishment","Euthanasia","Animal rights","Genetic engineering","Privacy","Wealth inequality","Affirmative action","Whistleblowing","Cultural appropriation","Moral responsibility","Abortion","Sex work","Drug legalization","Cloning","Obligations to future generations","Torture","Corporate ethics","Charity and obligation","Lying and deception","Punishment and justice"],
    "science/technology": ["Artificial intelligence","Genetic modification of food","Climate science","Vaccine policy","Nuclear energy","Space exploration","Surveillance technology","Automation and labor","Social media algorithms","Data privacy","Biotechnology","Cryptocurrency","Internet access","Human enhancement","Autonomous weapons","Scientific funding","Open source technology","Gene editing in humans","Big tech monopolies","Digital addiction"],
    "society/culture": ["Social media","Gender roles","Religion in public life","Marriage and family structure","Education systems","Cancel culture","Immigration and identity","Multiculturalism","Celebrity influence","Urbanization","Consumerism","Aging populations","Youth culture","Sports and society","Art and censorship","Class and mobility","Individualism vs collectivism","Tradition vs progress","Language and identity","Mental health stigma"],
    "environment": ["Climate change","Renewable energy","Fossil fuels","Deforestation","Plastic pollution","Animal agriculture","Nuclear power","Carbon taxation","Biodiversity loss","Ocean conservation","Urban development","Water scarcity","Endangered species protection","Environmental justice","Geoengineering","Single use plastics","Rewilding","Green technology","Individual vs corporate responsibility","International climate agreements"],
    "health": ["Universal healthcare","Mental health treatment","Drug policy","Obesity and public health","Pharmaceutical pricing","Vaccination","Reproductive health","End of life care","Alternative medicine","Healthcare rationing","Addiction treatment","Genetic testing","Food regulation","Exercise and personal responsibility","Healthcare privatization","Antibiotic resistance","Cosmetic surgery","Elderly care","Health insurance","Medical research ethics"],
}

# Types and Dimensions
EMPIRICAL_DIMS = ["causation","magnitude","prevalence","comparison","calibration","consensus"]
NORMATIVE_DIMS = ["moral valence","responsibility","policy response","value tradeoff","loyalty","authority","sanctity","consensus","trolley problem"]

# System Prompts
EMPR_PROMPT = """You are a claim generation engine. Your sole purpose is to generate rateable opinion claims on a given topic. A rateable claim is a declarative statement that a person can position themselves on a scale from 0 (strongly disagree) to 100 (strongly agree).

You must follow these rules without exception:
- Every claim is a single declarative sentence
- Every claim addresses exactly one issue — no compound claims
- Every claim uses neutral language — no loaded terms, emotional language, or framing that pushes toward agreement or disagreement
- Every claim must be genuinely contestable — a reasonable, informed person could disagree with it
- Never generate questions, calls to action, or policy prescriptions framed as facts
- Never embed evaluative adjectives within claims — words like harmful, dangerous, irresponsible, beneficial, or destructive pre-load a verdict and must be avoided. State the claim without prejudging its valence.

You will generate this type of claim:

Empirical claims assert something about the state of the world that is in principle testable by evidence. Draw from these dimensions as appropriate:
- causation — what does this topic cause or produce?
- magnitude — how significant, widespread, or severe is it?
- prevalence — how common or widespread is it?
- comparison — how does it compare to alternatives or prior states?
- calibration — where one answer is more evidentially supported even if publicly contested
- consensus — where one answer is near-universally agreed upon but a theoretically possible opposing position exists"""

NORM_PROMPT = """You are a claim generation engine. Your sole purpose is to generate rateable opinion claims on a given topic. A rateable claim is a declarative statement that a person can position themselves on a scale from 0 (strongly disagree) to 100 (strongly agree).

You must follow these rules without exception:
- Every claim is a single declarative sentence
- Every claim addresses exactly one issue — no compound claims
- Every claim uses neutral language — no loaded terms, emotional language, or framing that pushes toward agreement or disagreement
- Every claim must be genuinely contestable — a reasonable, informed person could disagree with it
- Never generate questions, calls to action, or policy prescriptions framed as facts
- Never embed evaluative adjectives within claims — words like harmful, dangerous, irresponsible, beneficial, or destructive pre-load a verdict and must be avoided. State the claim without prejudging its valence.

You will generate this type of claim:

Normative claims assert that something is good, right, or ought to be the case. Draw from these dimensions as appropriate:
- moral valence — is this thing good, bad, right, or wrong?
- responsibility — who is accountable for this, and to what degree?
- policy response — what should be done about it?
- value tradeoff — what does prioritizing this come at the expense of?
- loyalty — does this strengthen or weaken bonds between people, groups, or institutions?
- authority — does this support or undermine legitimate structures, rules, or hierarchies?
- sanctity — does this preserve or violate deeply held standards of dignity, purity, or tradition?
- consensus — where one answer is near-universally agreed upon but a theoretically possible opposing position exists
- trolley problem — where a direct harmful action produces a better aggregate outcome than inaction"""

# FUNCTIONS
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

def valid_claim(claim, seen):
    if not claim.get("claim") or len(claim["claim"].strip()) < 20: return False # is more than 20 chars?
    if claim.get("type") not in ("empirical", "normative"): return False # is empirical or normative?
    if not (claim.get("dimension") in EMPIRICAL_DIMS or claim.get("dimension") in NORMATIVE_DIMS): return False # is valid dimension?
    if "?" in claim["claim"]: return False # is not question?
    if claim["claim"].strip().lower() in seen: return False # is not a duplicate?
    return True # if all above true/yes, is valid

def valid_batch(claims, seen):
    if len(claims) != per: return False # is there n claims?
    if not all(valid_claim(c, seen) for c in claims): return False # are they all valid?
    return True # if all above true/yes, is valid

def generate(domain, subtopic, count, seen):
    while True:
        try:
            attempt = 0
            # Generate empirical claims
            while True:
                print("\tGenerating empirical claims...",end=" ")
                empirical = ollama.chat(
                    model=model,
                    messages=[
                        {"role":"system", "content": EMPR_PROMPT},
                        {"role":"user",
                         "content":f"Generate {per} empirical claims about {subtopic}. Domain is {domain}. Use only these dimensions: {', '.join(EMPIRICAL_DIMS)}"}
                    ],
                    format=ClaimList.model_json_schema(),
                    options={"seed":seed+count+attempt,"num_predict": per*100}
                )
                parsed_empirical = [{**c.model_dump(),"domain":domain,"subtopic":subtopic,"dimension":c.dimension.strip().lower()} for c in ClaimList.model_validate_json(empirical.message.content).claims]
                if valid_batch(parsed_empirical,seen):
                    print("Success!")
                    break
                print("Failed!. Trying again...")
                attempt += 1
            attempt = 0
            # Generate normative claims
            while True:
                print("\tGenerating normative claims...",end=" ")
                normative = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system","content":NORM_PROMPT},
                        {"role":"user",
                         "content": f"Generate {per} normative claims about {subtopic}. Domain is {domain}. Use only these dimensions: {', '.join(NORMATIVE_DIMS)}"}
                    ],
                    format=ClaimList.model_json_schema(),
                    options={"seed":seed+count+attempt+1000,"num_predict":per*100}
                )
                parsed_normative = [{**c.model_dump(),"domain":domain,"subtopic":subtopic,"dimension":c.dimension.strip().lower()} for c in ClaimList.model_validate_json(normative.message.content).claims]
                if valid_batch(parsed_normative,seen):
                    print("Success!")
                    break
                print("Failed!. Trying again...")
                attempt += 1
            return parsed_empirical + parsed_normative
        except Exception:
            continue

def generate_claims():
    # Run vars
    claims = []
    seen = set()
    completed = set()
    count = 0
    total = len(DOMAINS.keys())*len(DOMAINS["politics"])

    # Checkpoint system
    if os.path.exists(output):
        existing = pd.read_csv(output)
        claims = existing.to_dict("records")
        for _, row in existing.iterrows():
            completed.add((row["domain"], row["subtopic"]))
            seen.add(row["claim"].strip().lower())
        print(f"Resuming from {len(completed)} completed subtopics")

    # Generate claims
    for domain in DOMAINS:
        for subtopic in DOMAINS[domain]:
            count += 1
            # If claim already generated
            if (domain, subtopic) in completed:
                print(f"\t[{count}/{total}] Completed - {domain} | {subtopic}")
                continue
            # Else continue generating
            curr = generate(domain,subtopic,count,seen)
            for c in curr:
                seen.add(c["claim"].strip().lower())
            claims.extend(curr)
            pd.DataFrame(claims).to_csv(output, index=False)
            print(f"\t[{count}/{total}] | {domain} | {subtopic} — {len(curr)} claims ({len(claims)} total)")

    # End of run. All progress compiled and saved
    print(f"Claims saved to {output}")

def main():
    args = sys.argv[1:]
    nargs = len(args)
    # Usage: python(3) shift.py [ollama model] [output file] [optional flags]

    # Help
    if "-h" in args or "--help" in args:
        print("Usage: python shift.py [ollama model] [output file] [optional flags]")
        print("Optional flags:\n\t-c, --claims [n]: Set the amount of claims per type. (default is 8) (i.e. 10 empirical, 10 normative)\n\t-h, --help: Print the usage.\n\t-s, --seed [seed] [n]: Set the generation seed. (default is 42)")
        print()

    # Required args
    if nargs < 2:
        print(f"Current args: {args}\nMissing required args. Use -h, --help to view the usage.")
        return
    global model, output, per, seed
    model, out, *flags = args
    model_check(model)
    output = out.replace(":", "_").replace("/", "_")

    # Flag checks
    if "-c" in flags or "--claims" in flags:
        if "-c" in flags:
            idx = flags.index("-c")
        else:
            idx = flags.index("--claims")
        per = int(flags[idx+1])

    if "-s" in flags or "--seed" in flags:
        if "-s" in flags:
            idx = flags.index("-s")
        else:
            idx = flags.index("--seed")
        seed = int(flags[idx + 1])

    # Generate claims
    print(f"Generating {len(DOMAINS.keys())*len(DOMAINS['politics'])*(per*2)} claims...")
    generate_claims()

if __name__ == "__main__":
    main()