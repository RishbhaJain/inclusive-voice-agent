from enum import Enum
import spacy

import asyncio
import time

from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  
client = AsyncOpenAI() 


class TurnDecision(Enum):
        WAIT = 0
        RECONFIRM = 1
        TALK = 2

class TurnDetector:
    def __init__(self, min_silence: int = 800, hard_limit: int = 2000):
        self.nlp = spacy.load("en_core_web_sm")
        self.min_silence = min_silence
        self.hard_limit = hard_limit

    def is_syntactically_complete(self, text: str) -> bool:
        if not text.strip():
            return False
        
        doc = self.nlp(text)
        last_token = doc[-1]

        # --- 1. THE DANGLER CHECK (Functional Incompleteness) ---
        # These Parts of Speech (POS) literally cannot end a sentence.
        # DET: "a", "the" | ADP: "with", "for" | CCONJ: "and" | PART: "to"
        if last_token.pos_ in ["DET", "ADP", "CCONJ", "SCONJ", "PART"]:
            return False

        # --- 2. THE STRUCTURAL CHECK (The "Bones") ---
        has_subject = any(t.dep_ in ["nsubj", "nsubjpass", "expl"] for t in doc)
        has_root = any(t.dep_ == "ROOT" for t in doc)
        
        if not (has_subject and has_root):
            return False

        # --- 3. THE "HUNGRY VERB" CHECK (Transitivity) ---
        # In "that's got", the last token is "got" (VERB). 
        # If a sentence ends on a VERB, it's almost always a pause in a dealership context.
        if last_token.pos_ == "VERB":
            return False

        # --- 4. THE SATISFACTION CHECK ---
        # A thought is complete if the ROOT is satisfied by an object, attribute, or adverb.
        # "open" = acomp/attr | "truck" = dobj | "now" = advmod
        completion_deps = {
            "dobj", "pobj", "attr", "acomp", "xcomp", "ccomp", "advmod"
        }
        
        # We check if the doc contains any of these completion signals
        is_satisfied = any(t.dep_ in completion_deps for t in doc)
        
        return is_satisfied
    
    def get_last_sentence(self, text:str)->str:
        doc = self.nlp(text)
        sentences = list(doc.sents)
        return sentences[-1].text.strip() if sentences else text


    def evaluate(self, transcript: str, silence_ms: int) -> TurnDecision:
        # Check how much youre sending in transcript
        if silence_ms < self.min_silence:
            return TurnDecision.WAIT
        if silence_ms>=self.hard_limit:
            return TurnDecision.TALK
        last_utterance = self.get_last_sentence(transcript)
        if self.is_syntactically_complete(last_utterance):
            return TurnDecision.TALK
        return TurnDecision.WAIT
        
    
        