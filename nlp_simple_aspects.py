import nltk
from typing import List, Dict, Any
from collections import Counter
import re

# Ensure necessary NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

STOP = set(stopwords.words('english'))


class SimpleAspectAnalyzer:
    """Lightweight aspect extractor + VADER-based pros/cons.

    Works with only NLTK (already in requirements) and aims to provide quick
    aspect suggestions and pros/cons from reviews/descriptions without heavy deps.
    """

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def _candidate_keywords(self, text: str, top_n: int = 6) -> List[str]:
        # normalize and extract words/bigrams, filter stopwords and short tokens
        text = re.sub(r"[^\w\s]", ' ', text.lower())
        # Use simple regex tokenization to avoid heavy punkt dependency issues
        tokens = [t for t in re.findall(r"\b[a-zA-Z]+\b", text) if t not in STOP and len(t) > 2]
        # build unigrams and bigrams
        unigrams = tokens
        bigrams = [f"{unigrams[i]} {unigrams[i+1]}" for i in range(len(unigrams)-1)]
        counts = Counter(unigrams + bigrams)
        most = [kw for kw, _ in counts.most_common(top_n*2)]
        # prefer bigrams if present
        prefs = [w for w in most if ' ' in w]
        prefs += [w for w in most if ' ' not in w]
        # dedupe preserving order
        seen = set()
        res = []
        for w in prefs:
            if w in seen:
                continue
            seen.add(w)
            res.append(w)
            if len(res) >= top_n:
                break
        return res

    def extract_pros_cons(self, text: str, top_n: int = 6, pos_thresh: float = 0.3, neg_thresh: float = -0.3) -> Dict[str, Any]:
        """Return aspects, pros and cons using simple heuristics and VADER.

        Output: { 'aspects': [...], 'pros': [...], 'cons': [...], 'aspect_sentiments': {asp: avg_compound} }
        """
        if not text or not text.strip():
            return {"aspects": [], "pros": [], "cons": [], "aspect_sentiments": {}}

        aspects = self._candidate_keywords(text, top_n=top_n)

        # map aspect -> list of compound sentiment scores from sentences that mention it
        asp_scores: Dict[str, List[float]] = {a: [] for a in aspects}
        # Sentence splitting: prefer nltk sent_tokenize, but fall back to regex if punkt not available
        try:
            sents = sent_tokenize(text)
        except Exception:
            sents = re.split(r'[\.!?]+\s*', text)
        for s in sents:
            score = self.vader.polarity_scores(s).get('compound', 0.0)
            ls = s.lower()
            for a in aspects:
                if a in ls:
                    asp_scores[a].append(score)

        aspect_sentiments: Dict[str, float] = {}
        pros = []
        cons = []
        for a, scores in asp_scores.items():
            if not scores:
                avg = 0.0
            else:
                avg = sum(scores) / len(scores)
            aspect_sentiments[a] = avg
            if avg >= pos_thresh:
                pros.append(a)
            elif avg <= neg_thresh:
                cons.append(a)

        return {
            "aspects": aspects,
            "pros": pros,
            "cons": cons,
            "aspect_sentiments": aspect_sentiments,
        }


if __name__ == "__main__":
    s = SimpleAspectAnalyzer()
    sample = "The food was excellent, the biryani and paneer were amazing. Service was slow and the place noisy."
    print(s.extract_pros_cons(sample))
