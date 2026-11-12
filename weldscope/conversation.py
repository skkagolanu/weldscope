"""Conversation routing and bounded history, separate from document retrieval."""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ConversationPlan:
    kind: str
    answer: str = ""
    retrieval_query: str = ""


DOMAIN = re.compile(
    r"\b(fsw|friction.stir|weld\w*|wormholes?|tunnel defects?|heat input|"
    r"tool pins?|tool shoulders?|rotation(?:al)? speed|traverse speed|travel speed|rpm|"
    r"sensor(?:s| log| data)?|stir zone|nugget zone|flash defects?|kissing bonds?|"
    r"tmaz|haz|axial force|anvil|travel speed|shoulder scroll|bobbin|retreating side|advancing side|"
    r"dynamically recrystallized zone|solid.state joining|plunge depth|dwell time|faying surfaces?)\b",
    re.I,
)
FOLLOW_UP = re.compile(
    r"^(?:and\b|what about\b|how about\b|why\b|tell me more\b|explain (?:that|this|more)\b)"
    r"|\b(?:it|its|that|these|those|they|them|the same)\b",
    re.I,
)


SCOPE_REPLY = (
    "I can answer questions supported by the bundled friction stir welding documents. "
    "I can't assess a real vessel's condition, determine whether it is safe, or advise "
    "on a live operational incident. Please ask a general question about the corpus."
)


def is_recipe_request(text):
    recipe = re.search(r"\b(?:recipes?|meal plans?)\b", text)
    food = re.search(
        r"\b(?:cook\w*|bak\w*|meals?|dinner|lunch|breakfast|food|pizza|pasta|curry)\b", text
    )
    cooking_instructions = re.search(
        r"\b(?:how to|how (?:do|can|should) i|help me|teach me|instructions to)\b.{0,80}\b(?:cook|bake)\b"
        r"|\b(?:please |can you |could you )?(?:cook|bake)\s+(?:me |us |a |some |my )",
        text,
    )
    return bool(cooking_instructions or (recipe and (food or not DOMAIN.search(text))))


def is_live_operational_request(text):
    vessel = re.search(r"\b(?:ship|boat|vessel|yacht|ferry|hull)\b", text)
    live = re.search(
        r"\b(?:i am on|i'm on|we are on|we're on|aboard|onboard|my ship|our ship|my boat|our boat|this vessel|right now|currently|soon)\b",
        text,
    )
    failure = re.search(
        r"\b(?:sink|sinking|crack\w*|coming apart|coming loose|taking on water|flooding|failing|seaworth\w*|safe to sail)\b",
        text,
    )
    assessment = re.search(r"\b(?:is (?:my|our|this) .{0,35}safe|can (?:we|i) .{0,30}sail)\b", text)
    direct_incident = re.search(
        r"\b(?:we|i)(?:'re| are|'m| am)?(?: might| may| could| will| are about to| am about to)?\s+(?:sink|sinking|taking on water)\b",
        text,
    )
    return bool(direct_incident or (vessel and ((live and failure) or assessment)))


def completed_history(messages, max_turns=10, character_budget=6000):
    """Keep complete successful exchanges; failed calls are never model context."""
    pairs = []
    pending = None
    for message in messages:
        if message["role"] == "user":
            pending = message
        elif message["role"] == "assistant" and pending is not None:
            if message.get("route") != "error":
                pairs.append((pending, message))
            pending = None
    selected = []
    used = 0
    for user, assistant in reversed(pairs[-max_turns:]):
        pair = [
            dict(user, content=user["content"][:1000]),
            dict(assistant, content=assistant["content"][:1500]),
        ]
        size = sum(len(m["content"]) for m in pair)
        if used + size > character_budget:
            break
        selected[0:0] = pair
        used += size
    return selected


def about_weldscope():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    papers = sorted(p.name for p in data_dir.glob("*.pdf"))
    corpus = ", ".join(f"`{name}`" for name in papers) or "the configured welding PDFs"
    return (
        "I'm **WeldScope**, a research assistant for **friction stir welding (FSW)**. "
        "I help explore welding defects, possible causes, process parameters, and corrective actions.\n\n"
        f"My bundled document collection contains {corpus}. I combine keyword search (BM25) "
        "with semantic vector search, rerank the passages, and pass relevant evidence to a language "
        "model to write an answer with source passages. RAG means **retrieval-augmented generation**: "
        "the documents are searched at question time; they aren't used to train the model.\n\n"
        "There's also a bundled example sensor CSV, used when you ask about sensor readings; "
        "it isn't a live feed. I use recent completed exchanges for follow-ups within this session. "
        "Clear Conversation resets that context.\n\n"
        "Try: **What causes tunnel defects in FSW?** or **What do the sensor readings show?**"
    )


def _active_topic(history):
    for message in reversed(history):
        if message["role"] != "assistant":
            continue
        if message.get("route") in {"chat", "out_of_scope"}:
            return None
        if message.get("route") == "research":
            return message.get("retrieval_query")
    return None


def plan_conversation(question, history=None):
    """Handle known social/product questions locally; retrieve only for research."""
    text = re.sub(r"\s+", " ", question.lower()).strip().rstrip(".!? ")
    text = text.replace("what's", "what is").replace("who's", "who is")
    # Current task intent wins over both domain keywords and prior conversation.
    if re.search(
        r"\b(?:tell me a joke|write (?:python |javascript )?code|sorting algorithm|"
        r"(?:private|access|my) .{0,35}password|remaining (?:fatigue )?life|safe to sail)\b",
        text,
    ):
        return ConversationPlan(
            "out_of_scope",
            "That request is outside the bundled FSW research corpus. I can discuss the documents, but cannot provide private data, unrelated content or equipment-specific assessments.",
        )
    if is_recipe_request(text):
        return ConversationPlan(
            "out_of_scope",
            "Recipes and cooking instructions are outside this chatbot's corpus. I can answer questions grounded in the bundled friction stir welding documents.",
        )
    if is_live_operational_request(text):
        return ConversationPlan("out_of_scope", SCOPE_REPLY)
    if re.fullmatch(
        r"(?:hi|hey|hello|hiya|good (?:morning|afternoon|evening))(?: there| weldscope)?(?:[,! ]+how are you)?",
        text,
    ):
        return ConversationPlan(
            "local",
            "Hi! I'm WeldScope. Ask me about friction stir welding, "
            "the documents I use, or how this assistant works.",
        )
    if re.fullmatch(r"(?:thanks|thank you|cheers|thanks a lot|thank you very much)", text):
        return ConversationPlan("local", "You're welcome! What would you like to explore next?")
    if text in {"how are you", "how are you doing", "how's it going"}:
        return ConversationPlan("local", "I'm ready to help. What are you working on?")
    if text in {"bye", "goodbye", "see you", "good night"}:
        return ConversationPlan(
            "local", "See you! Come back whenever you'd like to explore a welding question."
        )
    about = (
        r"\b(?:what (?:is|was) (?:this (?:chatbot|app)|weldscope) (?:built|designed|made|used) for|what (?:is|was) your purpose)\b",
        r"\b(?:what|who) (?:is|are) (?:you|weldscope|this (?:app|rag|chatbot|assistant|project)|(?:a |the )?rag)\b",
        r"\b(?:what|which) (?:can you|do you|are you) (?:do|use|help|based|trained)\b",
        r"\b(?:how do you work|how does (?:this|weldscope|the rag) work|what can (?:this|weldscope) do)\b",
        r"\b(?:what|which) (?:documents|papers|sources|data|corpus|model|models) (?:do you|are you|does (?:this|weldscope)|is (?:this|the rag))\b",
        r"\b(?:your (?:sources|knowledge base|capabilities)|what is (?:this|the rag|your rag|your model|weldscope) based on)\b",
        r"^(?:tell me about (?:yourself|weldscope|this app|this rag)|what is your model)",
    )
    if text in {"help", "what can you do", "what is rag", "how does rag work"} or any(
        re.search(pattern, text) for pattern in about
    ):
        return ConversationPlan("local", about_weldscope())

    # A self-contained domain question takes precedence over incidental pronouns
    # such as 'that' in a new question. History only resolves implicit follow-ups.
    topic = _active_topic(history or [])
    explicit_followup = re.search(
        r"^(?:and\b|what about\b|how about\b|tell me more\b|"
        r"how (?:is|does|do) (?:that|it|this|they|those)\b)",
        text,
    )
    if topic and explicit_followup:
        return ConversationPlan("research", retrieval_query=f"{topic[:500]}\nFollow-up: {question}")
    if DOMAIN.search(text):
        return ConversationPlan("research", retrieval_query=question)
    if topic and FOLLOW_UP.search(text):
        return ConversationPlan("research", retrieval_query=f"{topic[:500]}\nFollow-up: {question}")
    if re.fullmatch(r"(?:why|why is that|tell me more|explain that|what about it)", text):
        return ConversationPlan(
            "local", "Which welding process or defect would you like me to explain?"
        )
    return ConversationPlan(
        "out_of_scope",
        "I can only answer questions grounded in the bundled friction stir welding documents. Ask about the corpus, its sources, or how WeldScope works.",
    )


def needs_sensor_context(question):
    return bool(
        re.search(r"\b(?:sensors?|readings?|log|csv|recorded|measurements?)\b", question, re.I)
    )
