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

class SpacyTurnDetector:
    """
    LiveKit turn_detector plugin using spaCy syntactic completeness.

    LiveKit calls predict_end_of_turn() after VAD detects silence and
    min_endpointing_delay has passed. We return 1.0 if the last user
    message looks syntactically complete, 0.0 if it's a fragment.
    LiveKit compares this against unlikely_threshold (0.5):
      score >= 0.5 → commit turn → agent responds
      score <  0.5 → hold turn  → keep waiting up to max_endpointing_delay

    This prevents the agent from cutting in on "To", "But", "I was", etc.
    """

    def __init__(self) -> None:
        # Load spaCy once at construction — not on every predict_end_of_turn call.
        # spacy.load() takes ~100ms; calling it per-check would add latency and
        # could cause timeouts in LiveKit's EOU detection pipeline.
        self._td = TurnDetector()

    @property
    def model(self) -> str:
        return "en_core_web_sm"

    @property
    def provider(self) -> str:
        return "spacy"

    async def unlikely_threshold(self, language=None) -> float | None:
        # Scores below 0.5 → hold the turn
        return 0.5

    async def supports_language(self, language=None) -> bool:
        return True  # spaCy en_core_web_sm handles English; neutral for other languages

    async def predict_end_of_turn(self, chat_ctx, *, timeout=None) -> float:
        """
        Extract the last user message from the chat context and run the
        syntactic completeness check. Returns 1.0 (complete) or 0.0 (fragment).

        LiveKit adds the current partial transcript as the last user message
        in chat_ctx before calling this, so we always see the live utterance.
        """
        messages = chat_ctx.messages()
        last_user_text = ""
        for msg in reversed(messages):
            if getattr(msg, "role", None) == "user":
                content = msg.content
                if isinstance(content, str):
                    last_user_text = content
                elif isinstance(content, list):
                    last_user_text = " ".join(
                        c if isinstance(c, str) else getattr(c, "text", "")
                        for c in content
                    )
                break

        if not last_user_text.strip():
            return 0.0

        last_sentence = self._td.get_last_sentence(last_user_text)
        return 1.0 if self._td.is_syntactically_complete(last_sentence) else 0.0


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

        # Surface-form check for articles that can never end a sentence.
        # spaCy sometimes mislabels "a"/"an" as PRON when the noun is missing
        # (e.g. "I am looking for a" → "a" tagged as pobj PRON). These words
        # are always determiners in practice at sentence-end position.
        if last_token.text.lower() in {"a", "an", "the"}:
            return False

        # --- 2. THE STRUCTURAL CHECK (The "Bones") ---
        has_subject = any(t.dep_ in ["nsubj", "nsubjpass", "expl"] for t in doc)
        has_root = any(t.dep_ == "ROOT" for t in doc)
        
        if not (has_subject and has_root):
            return False

        # --- 3. THE "HUNGRY VERB" CHECK (Transitivity) ---
        # "I want", "I am looking", "that's got" end on a bare VERB that needs
        # an object → incomplete. Exception: negated verbs like "I do not know"
        # or "I cannot see" are genuinely complete sentences. We detect negation
        # via the `neg` dependency (attached to the verb by spaCy).
        has_negation = any(t.dep_ == "neg" for t in doc)
        if last_token.pos_ in ("VERB", "AUX") and not has_negation:
            return False

        # --- 3b. RELATIVE CLAUSE STARTER CHECK ---
        # Words like "which", "that", "who", "where" at sentence-end always
        # open a relative clause that has no body yet.
        # spaCy sometimes tags "which" as PRON/dobj, bypassing the DET/SCONJ check.
        if last_token.text.lower() in {"which", "that", "who", "whom", "where", "when", "how", "why"}:
            return False

        return True
    
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
        
    
        