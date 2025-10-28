from typing import List, Dict, Any
from keybert import KeyBERT
from transformers import pipeline
import math


class AspectSentimentAnalyzer:
    """Extract aspects (keyphrases) and compute sentiment per aspect from reviews/text.

    Uses KeyBERT for keyphrase extraction and a transformers sentiment pipeline for labeling.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        # KeyBERT uses a sentence-transformers model under the hood; we have sentence-transformers in reqs
        self.kb = KeyBERT()
        # Sentiment analysis pipeline (light and fast)
        self.sent = pipeline("sentiment-analysis", model=model_name)

    def extract_aspects(self, text: str, top_n: int = 5) -> List[str]:
        if not text or not text.strip():
            return []
        try:
            kws = self.kb.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n, use_mmr=True)
            return [kw for kw, _ in kws]
        except Exception:
            # fallback: very simple heuristic split
            tokens = [t.strip() for t in text.split('.') if len(t.strip()) > 4]
            return tokens[:top_n]

    def aspect_sentiments(self, text: str, aspects: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return sentiment label and score for each aspect.

        Returns dict: { aspect: {label: str, score: float} }
        """
        res = {}
        if not text or not aspects:
            return res

        for a in aspects:
            snippet = a
            # Prefer short snippet that includes aspect + some context
            if a.lower() not in text.lower():
                snippet = f"{a}. {text[:300]}"
            else:
                # extract small window around first occurrence
                idx = text.lower().find(a.lower())
                start = max(0, idx - 120)
                end = min(len(text), idx + 120)
                snippet = text[start:end]

            try:
                out = self.sent(snippet[:512])[0]
                label = out.get("label", "NEUTRAL")
                score = float(out.get("score", 0.0))
            except Exception:
                label = "NEUTRAL"
                score = 0.0

            res[a] = {"label": label, "score": score}

        return res

    def extract_pros_cons(self, text: str, top_n: int = 5, positive_threshold: float = 0.6, negative_threshold: float = 0.6) -> Dict[str, Any]:
        """Extract key aspects and separate them into pros and cons based on sentiment.

        Returns: { "aspects": [...], "pros": [...], "cons": [...], "aspect_sentiments": {...} }
        """
        aspects = self.extract_aspects(text, top_n=top_n)
        sentiments = self.aspect_sentiments(text, aspects)

        pros = []
        cons = []
        for a, info in sentiments.items():
            label = info.get("label", "NEUTRAL")
            score = info.get("score", 0.0)
            if label.upper() == "POSITIVE" and score >= positive_threshold:
                pros.append(a)
            elif label.upper() == "NEGATIVE" and score >= negative_threshold:
                cons.append(a)

        return {
            "aspects": aspects,
            "pros": pros,
            "cons": cons,
            "aspect_sentiments": sentiments,
        }


if __name__ == "__main__":
    sample = (
        "The food was delicious and the biryani stood out, but the service was slow and the place was noisy."
    )
    a = AspectSentimentAnalyzer()
    print(a.extract_pros_cons(sample))
