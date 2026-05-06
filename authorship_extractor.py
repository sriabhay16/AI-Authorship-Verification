import re
import numpy as np
from collections import Counter
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# NLTK STOPWORDS (with fallback)
# ============================================================================

def _load_stopwords():
    fallback = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'am', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'what', 'which', 'who', 'whom', 'why', 'how',
        'as', 'if', 'when', 'where', 'because', 'whether', 'while', 'although'
    }

    try:
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords

        try:
            words = nltk_stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = nltk_stopwords.words("english")

        return set(words)

    except Exception:
        return fallback


NLTK_STOPWORDS = _load_stopwords()

# ============================================================================
# AUTHORSHIP METRICS EXTRACTOR - 40 KEY METRICS
# ============================================================================

class AuthorshipMetricsExtractor:
    """
    Comprehensive metrics extraction system for authorship attribution.
    Preserves all 40 metrics while using NLTK stopwords.
    """

    def __init__(self, text: str):
        self.text = text

        # Tokenization
        self.sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        self.tokens = re.findall(r"\b\w+\b|\S", text.lower())
        self.tokens_no_punct = [t for t in self.tokens if t.isalpha()]

        # Basic stats
        self.num_tokens = len(self.tokens)
        self.num_tokens_no_punct = len(self.tokens_no_punct)
        self.num_sentences = len(self.sentences)
        self.num_types = len(set(self.tokens_no_punct))

        # Linguistic categories
        self.stop_words = NLTK_STOPWORDS
        self.modals = {"can", "could", "may", "might", "must", "should", "will", "would"}
        self.pronouns = {
            "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "her", "its", "our", "their"
        }

    # ========== TIER 1: FOUNDATIONAL METRICS (15 metrics) ==========

    def metric_01_average_word_length(self) -> float:
        if not self.tokens_no_punct:
            return 0.0
        return float(np.mean([len(t) for t in self.tokens_no_punct]))

    def metric_02_type_token_ratio(self) -> float:
        if self.num_tokens_no_punct == 0:
            return 0.0
        return float(self.num_types / self.num_tokens_no_punct)

    def metric_03_yules_k(self) -> float:
        if self.num_tokens_no_punct <= 1:
            return 0.0
        word_freq = Counter(self.tokens_no_punct)
        N = self.num_tokens_no_punct
        freq_dist = Counter(word_freq.values())
        sum_m_sq_v = sum(m**2 * freq_dist[m] for m in freq_dist)
        k = 10000 * (sum_m_sq_v - N) / (N**2 * (N - 1))
        return float(max(k, 0.0))

    def metric_04_hapax_legomena(self) -> float:
        word_freq = Counter(self.tokens_no_punct)
        hapax_count = sum(1 for count in word_freq.values() if count == 1)
        return float(hapax_count / self.num_types) if self.num_types > 0 else 0.0

    def metric_05_lexical_density(self) -> float:
        content_count = sum(1 for t in self.tokens_no_punct if t not in self.stop_words)
        function_count = sum(1 for t in self.tokens_no_punct if t in self.stop_words)
        return float(content_count / function_count) if function_count > 0 else 0.0

    def metric_06_average_sentence_length(self) -> float:
        if self.num_sentences == 0:
            return 0.0
        sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in self.sentences]
        return float(np.mean(sentence_lengths)) if sentence_lengths else 0.0

    def metric_07_sentence_length_std(self) -> float:
        if self.num_sentences <= 1:
            return 0.0
        sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in self.sentences]
        return float(np.std(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0

    def metric_08_complex_sentence_ratio(self) -> float:
        subordinators = {
            "because", "although", "if", "when", "while", "since",
            "unless", "until", "after", "before", "that", "which"
        }

        complex_count = 0
        for sentence in self.sentences:
            sent_lower = sentence.lower()
            if any(f" {sub} " in f" {sent_lower} " for sub in subordinators):
                complex_count += 1

        return float(complex_count / self.num_sentences) if self.num_sentences > 0 else 0.0

    def metric_09_average_clause_depth(self) -> float:
        max_depth = 0
        current_depth = 0

        for char in self.text:
            if char in "([{":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ")]}":
                current_depth = max(0, current_depth - 1)

        return float(max_depth) if max_depth > 0 else 1.0

    def metric_10_punctuation_distribution(self) -> Dict[str, float]:
        punct_chars = {".": 0, ",": 0, ";": 0, "!": 0, "?": 0}
        for char in self.text:
            if char in punct_chars:
                punct_chars[char] += 1

        total = sum(punct_chars.values())
        return {k: v / total if total > 0 else 0.0 for k, v in punct_chars.items()}

    def metric_11_punctuation_consistency(self) -> float:
        punct_dist = self.metric_10_punctuation_distribution()
        values = list(punct_dist.values())
        return float(np.std(values)) if len(values) > 0 else 0.0

    def metric_12_dash_semicolon_ratio(self) -> float:
        dash_count = self.text.count("-") + self.text.count("–") + self.text.count("—")
        semi_count = self.text.count(";")
        total = dash_count + semi_count
        return float(dash_count / total) if total > 0 else 0.0

    def metric_13_burrows_delta(self, text2_tokens: List[str], mfw_count: int = 100) -> float:
        all_tokens = self.tokens_no_punct + text2_tokens
        word_freq = Counter(all_tokens)
        mfw_list = [w for w, _ in word_freq.most_common(mfw_count)]

        if not mfw_list:
            return 0.0

        text1_freq = Counter(self.tokens_no_punct)
        text2_freq = Counter(text2_tokens)
        text1_total = len(self.tokens_no_punct)
        text2_total = len(text2_tokens)

        delta = 0.0
        for word in mfw_list:
            freq1 = (text1_freq.get(word, 0) / text1_total * 100) if text1_total > 0 else 0
            freq2 = (text2_freq.get(word, 0) / text2_total * 100) if text2_total > 0 else 0
            mfw_mean = (freq1 + freq2) / 2
            mfw_std = np.std([freq1, freq2])

            if mfw_std > 0:
                z1 = (freq1 - mfw_mean) / mfw_std
                z2 = (freq2 - mfw_mean) / mfw_std
                delta += abs(z1 - z2)

        return float(delta / len(mfw_list))

    def metric_14_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def metric_15_jensen_shannon_divergence(self, dist1: List[float], dist2: List[float]) -> float:
        d1 = np.array(dist1) / (np.sum(dist1) + 1e-10)
        d2 = np.array(dist2) / (np.sum(dist2) + 1e-10)
        m = 0.5 * (d1 + d2)

        kl1 = np.sum(d1 * np.log((d1 + 1e-10) / (m + 1e-10)))
        kl2 = np.sum(d2 * np.log((d2 + 1e-10) / (m + 1e-10)))
        js = 0.5 * kl1 + 0.5 * kl2

        return float(np.sqrt(max(js, 0.0)))

    # ========== TIER 2: INTERMEDIATE METRICS (15 metrics) ==========

    def metric_16_pos_bigrams_distribution(self) -> Dict[str, float]:
        pos_tags = []
        for token in self.tokens:
            if token in self.modals:
                pos_tags.append("MD")
            elif token in self.pronouns:
                pos_tags.append("PRP")
            elif token in self.stop_words:
                pos_tags.append("IN")
            else:
                pos_tags.append("NN")

        pos_bigrams = [f"{pos_tags[i]}-{pos_tags[i+1]}" for i in range(len(pos_tags) - 1)]
        bigram_freq = Counter(pos_bigrams)
        top_bigrams = dict(bigram_freq.most_common(10))
        total = sum(top_bigrams.values())

        return {k: v / total for k, v in top_bigrams.items()} if total > 0 else {}

    def metric_17_pronoun_frequency(self) -> Dict[str, float]:
        first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
        second_person = {"you", "your", "yours"}
        third_person = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"}

        first = sum(1 for t in self.tokens_no_punct if t in first_person)
        second = sum(1 for t in self.tokens_no_punct if t in second_person)
        third = sum(1 for t in self.tokens_no_punct if t in third_person)
        total_pronouns = first + second + third

        return {
            "first_person": first / total_pronouns if total_pronouns > 0 else 0.0,
            "second_person": second / total_pronouns if total_pronouns > 0 else 0.0,
            "third_person": third / total_pronouns if total_pronouns > 0 else 0.0,
        }

    def metric_18_modal_frequency(self) -> float:
        modal_count = sum(1 for t in self.tokens_no_punct if t in self.modals)
        return float(modal_count / self.num_tokens_no_punct) if self.num_tokens_no_punct > 0 else 0.0

    def metric_19_character_ngram_diversity(self) -> float:
        text_clean = "".join(c for c in self.text if c.isalpha())
        char_trigrams = [text_clean[i:i+3] for i in range(len(text_clean) - 2)]

        if not char_trigrams:
            return 0.0

        unique_trigrams = len(set(char_trigrams))
        return float(unique_trigrams / len(char_trigrams))

    def metric_20_word_bigram_entropy(self) -> float:
        word_bigrams = [
            (self.tokens_no_punct[i], self.tokens_no_punct[i + 1])
            for i in range(len(self.tokens_no_punct) - 1)
        ]

        if not word_bigrams:
            return 0.0

        bigram_freq = Counter(word_bigrams)
        probs = np.array(list(bigram_freq.values())) / len(word_bigrams)
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    def metric_21_frequent_word_bigrams(self) -> Dict[str, float]:
        word_bigrams = [
            (self.tokens_no_punct[i], self.tokens_no_punct[i + 1])
            for i in range(len(self.tokens_no_punct) - 1)
        ]

        bigram_freq = Counter(word_bigrams)
        top_50 = dict(bigram_freq.most_common(50))
        total = sum(top_50.values())

        top_10 = dict(sorted(top_50.items(), key=lambda x: x[1], reverse=True)[:10])
        return {str(k): v / total for k, v in top_10.items()} if total > 0 else {}

    def metric_22_rare_ngram_coverage(self) -> float:
        word_bigrams = [
            (self.tokens_no_punct[i], self.tokens_no_punct[i + 1])
            for i in range(len(self.tokens_no_punct) - 1)
        ]

        if not word_bigrams:
            return 0.0

        bigram_freq = Counter(word_bigrams)
        rare_ngrams = sum(1 for count in bigram_freq.values() if count == 1)
        return float(rare_ngrams / len(bigram_freq)) if len(bigram_freq) > 0 else 0.0

    def metric_23_perplexity_approximation(self) -> float:
        word_freq = Counter(self.tokens_no_punct)
        total = len(self.tokens_no_punct)

        log_prob_sum = 0.0
        for token in self.tokens_no_punct:
            prob = (word_freq[token] + 1) / (total + len(word_freq))
            log_prob_sum += np.log2(prob)

        perplexity = 2 ** (-log_prob_sum / total) if total > 0 else 0.0
        return float(perplexity)

    def metric_24_shannon_entropy(self) -> float:
        word_freq = Counter(self.tokens_no_punct)
        if self.num_tokens_no_punct == 0:
            return 0.0
        probs = np.array(list(word_freq.values())) / self.num_tokens_no_punct
        entropy_val = -np.sum(probs * np.log2(probs + 1e-10))
        return float(entropy_val)

    def metric_25_cross_entropy_loss(self) -> float:
        word_freq = Counter(self.tokens_no_punct)
        if self.num_tokens_no_punct == 0:
            return 0.0
        probs = np.array(list(word_freq.values())) / self.num_tokens_no_punct
        uniform_probs = np.ones_like(probs) / len(probs)
        cross_entropy = -np.sum(uniform_probs * np.log2(probs + 1e-10))
        return float(cross_entropy)

    def metric_26_flesch_kincaid_grade(self) -> float:
        if self.num_sentences == 0 or self.num_tokens_no_punct == 0:
            return 0.0

        def count_syllables(word):
            word = word.lower()
            syllables = 0
            vowels = "aeiouy"
            prev_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = is_vowel

            if word.endswith("e"):
                syllables -= 1
            if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
                syllables += 1

            return max(1, syllables)

        total_syllables = sum(count_syllables(w) for w in self.tokens_no_punct)

        grade = (
            0.39 * (self.num_tokens_no_punct / self.num_sentences)
            + 11.8 * (total_syllables / self.num_tokens_no_punct)
            - 15.59
        )

        return float(max(0, grade))

    def metric_27_flesch_reading_ease(self) -> float:
        if self.num_sentences == 0 or self.num_tokens_no_punct == 0:
            return 0.0

        def count_syllables(word):
            word = word.lower()
            syllables = 0
            vowels = "aeiouy"
            prev_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = is_vowel

            if word.endswith("e"):
                syllables -= 1
            if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
                syllables += 1

            return max(1, syllables)

        total_syllables = sum(count_syllables(w) for w in self.tokens_no_punct)

        fre = (
            206.835
            - 1.015 * (self.num_tokens_no_punct / self.num_sentences)
            - 84.6 * (total_syllables / self.num_tokens_no_punct)
        )

        return float(max(0, min(100, fre)))

    def metric_28_gunning_fog_index(self) -> float:
        if self.num_sentences == 0 or self.num_tokens_no_punct == 0:
            return 0.0

        def count_syllables(word):
            word = word.lower()
            syllables = 0
            vowels = "aeiouy"
            prev_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = is_vowel

            if word.endswith("e"):
                syllables -= 1
            if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
                syllables += 1

            return max(1, syllables)

        complex_words = sum(1 for w in self.tokens_no_punct if count_syllables(w) >= 3)

        gunning_fog = 0.4 * (
            (self.num_tokens_no_punct / self.num_sentences)
            + 100 * (complex_words / self.num_tokens_no_punct)
        )

        return float(max(0, gunning_fog))

    def metric_29_discourse_marker_frequency(self) -> float:
        discourse_markers = {
            "however", "therefore", "furthermore", "moreover", "besides", "also",
            "indeed", "in fact", "for example", "for instance", "as a result",
            "consequently", "thus", "hence", "nonetheless", "meanwhile"
        }

        text_lower = self.text.lower()
        marker_count = 0
        for marker in discourse_markers:
            marker_count += text_lower.count(marker)

        return float(marker_count / self.num_sentences) if self.num_sentences > 0 else 0.0

    def metric_30_connective_diversity(self) -> float:
        connectives = {
            "and", "but", "or", "nor", "yet", "so", "because", "since",
            "although", "though", "if", "unless", "while", "when", "before",
            "after", "until", "thus", "therefore", "however", "moreover"
        }

        connective_usage = [t for t in self.tokens_no_punct if t in connectives]
        unique_connectives = len(set(connective_usage))

        return float(unique_connectives / len(connective_usage)) if len(connective_usage) > 0 else 0.0

    # ========== TIER 3: ADVANCED METRICS (10 metrics) ==========

    def metric_31_liwc_authenticity_proxy(self) -> float:
        first_person_sing = {"i", "me", "my", "mine"}
        third_person = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}

        insight = {"think", "know", "consider", "realize", "understand", "guess", "suppose", "learn"}
        differ = {"but", "however", "although", "though", "whereas", "yet"}
        relativ = {"in", "on", "at", "over", "under", "between", "around", "near", "above", "below"}
        discrep = {"should", "would", "could", "might", "may", "must"}

        if self.num_tokens_no_punct == 0:
            return 0.0

        toks = self.tokens_no_punct

        i_cnt = sum(1 for t in toks if t in first_person_sing)
        insight_cnt = sum(1 for t in toks if t in insight)
        differ_cnt = sum(1 for t in toks if t in differ)
        relativ_cnt = sum(1 for t in toks if t in relativ)
        discrep_cnt = sum(1 for t in toks if t in discrep)
        shehe_cnt = sum(1 for t in toks if t in third_person)

        scale = 100.0 / self.num_tokens_no_punct
        raw = (i_cnt + insight_cnt + differ_cnt + relativ_cnt - discrep_cnt - shehe_cnt) * scale
        score = 100.0 / (1.0 + np.exp(-0.25 * raw))
        return float(score)

    def metric_32_coreference_pattern_frequency(self) -> float:
        pronouns_count = sum(1 for t in self.tokens_no_punct if t in self.pronouns)
        nouns_count = sum(1 for t in self.tokens_no_punct if t not in self.stop_words and t not in self.modals)

        coreference_ratio = (
            pronouns_count / (nouns_count + pronouns_count)
            if (nouns_count + pronouns_count) > 0 else 0.0
        )
        return float(coreference_ratio)

    def metric_33_semantic_relatedness(self) -> float:
        if len(self.sentences) <= 1:
            return 0.0

        overlaps = []
        for i in range(len(self.sentences) - 1):
            sent1_words = set(re.findall(r"\b\w+\b", self.sentences[i].lower()))
            sent2_words = set(re.findall(r"\b\w+\b", self.sentences[i + 1].lower()))

            if sent1_words or sent2_words:
                overlap = len(sent1_words & sent2_words) / len(sent1_words | sent2_words)
                overlaps.append(overlap)

        return float(np.mean(overlaps)) if overlaps else 0.0

    def metric_34_mattr_moving_average_ttr(self, window_size: int = 100) -> float:
        if len(self.tokens_no_punct) < window_size:
            return self.metric_02_type_token_ratio()

        ratios = []
        for i in range(len(self.tokens_no_punct) - window_size):
            window = self.tokens_no_punct[i:i + window_size]
            window_types = len(set(window))
            window_ttr = window_types / window_size
            ratios.append(window_ttr)

        return float(np.mean(ratios)) if ratios else 0.0

    def metric_35_honores_r(self) -> float:
        if self.num_tokens_no_punct == 0:
            return 0.0

        word_freq = Counter(self.tokens_no_punct)
        hapax_count = sum(1 for count in word_freq.values() if count == 1)

        if (1 - hapax_count / self.num_types) == 0:
            return 0.0

        r = 100 * np.log(self.num_tokens_no_punct) / (1 - hapax_count / self.num_types)
        return float(max(0, r))

    def metric_36_zipfian_distribution(self) -> Dict[str, float]:
        word_freq = Counter(self.tokens_no_punct)
        frequencies = sorted(word_freq.values(), reverse=True)

        if len(frequencies) < 2:
            return {"alpha": 0.0, "r_squared": 0.0}

        ranks = np.arange(1, len(frequencies) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)

        try:
            alpha = -np.corrcoef(log_ranks, log_freqs)[0, 1]
            r_squared = np.corrcoef(log_ranks, log_freqs)[0, 1] ** 2
        except Exception:
            alpha = 0.0
            r_squared = 0.0

        return {"alpha": float(alpha), "r_squared": float(r_squared)}

    def metric_37_colocation_patterns(self) -> float:
        collocations = [
            ("said", "that"), ("made", "of"), ("great", "deal"), ("long", "time"),
            ("high", "school"), ("good", "time"), ("way", "to"), ("lot", "of")
        ]

        text_lower = self.text.lower()
        colocation_count = 0
        for word1, word2 in collocations:
            pattern = f"{word1} {word2}"
            colocation_count += text_lower.count(pattern)

        return float(colocation_count / len(self.sentences)) if self.num_sentences > 0 else 0.0

    def metric_38_grammar_error_rate(self) -> float:
        errors = 0
        total_checks = len(self.sentences)

        for sentence in self.sentences:
            sent_tokens = re.findall(r"\b\w+\b", sentence.lower())

            if "is" in sent_tokens and any(t in ["are", "am", "were"] for t in sent_tokens):
                errors += 1
            if "are" in sent_tokens and any(t in ["is", "am", "was"] for t in sent_tokens):
                errors += 1

        error_rate = errors / total_checks if total_checks > 0 else 0.0
        return float(error_rate)

    def metric_39_sentence_construction_uniformity(self) -> float:
        if len(self.sentences) <= 1:
            return 0.0

        sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in self.sentences]
        mean_len = np.mean(sentence_lengths)
        std_len = np.std(sentence_lengths)

        cv = (std_len / mean_len * 100) if mean_len > 0 else 0.0
        uniformity = 1 / (1 + cv / 100) if cv > 0 else 1.0

        return float(uniformity)

    def metric_40_token_predictability_variance(self) -> float:
        word_freq = Counter(self.tokens_no_punct)
        total = self.num_tokens_no_punct

        predictabilities = []
        for token in self.tokens_no_punct:
            prob = word_freq[token] / total
            predictabilities.append(prob)

        variance = np.var(predictabilities)
        return float(variance)

    # ========== MAIN EXTRACTION METHOD ==========

    def extract_all_metrics(self, text2_tokens: Optional[List[str]] = None) -> Dict[str, float]:
        metrics_dict = {}

        # TIER 1
        metrics_dict["m01_average_word_length"] = self.metric_01_average_word_length()
        metrics_dict["m02_type_token_ratio"] = self.metric_02_type_token_ratio()
        metrics_dict["m03_yules_k"] = self.metric_03_yules_k()
        metrics_dict["m04_hapax_legomena"] = self.metric_04_hapax_legomena()
        metrics_dict["m05_lexical_density"] = self.metric_05_lexical_density()
        metrics_dict["m06_average_sentence_length"] = self.metric_06_average_sentence_length()
        metrics_dict["m07_sentence_length_std"] = self.metric_07_sentence_length_std()
        metrics_dict["m08_complex_sentence_ratio"] = self.metric_08_complex_sentence_ratio()
        metrics_dict["m09_average_clause_depth"] = self.metric_09_average_clause_depth()

        punct_dist = self.metric_10_punctuation_distribution()
        for k, v in punct_dist.items():
            metrics_dict[f"m10_punct_{k}"] = v

        metrics_dict["m11_punctuation_consistency"] = self.metric_11_punctuation_consistency()
        metrics_dict["m12_dash_semicolon_ratio"] = self.metric_12_dash_semicolon_ratio()

        if text2_tokens is not None:
            metrics_dict["m13_burrows_delta"] = self.metric_13_burrows_delta(text2_tokens)

        # TIER 2
        pos_bigrams = self.metric_16_pos_bigrams_distribution()
        for i, (k, v) in enumerate(list(pos_bigrams.items())[:3]):
            metrics_dict[f"m16_pos_bigram_{i}"] = v

        pronouns = self.metric_17_pronoun_frequency()
        metrics_dict["m17_pronoun_first_person"] = pronouns["first_person"]
        metrics_dict["m17_pronoun_second_person"] = pronouns["second_person"]
        metrics_dict["m17_pronoun_third_person"] = pronouns["third_person"]

        metrics_dict["m18_modal_frequency"] = self.metric_18_modal_frequency()
        metrics_dict["m19_character_ngram_diversity"] = self.metric_19_character_ngram_diversity()
        metrics_dict["m20_word_bigram_entropy"] = self.metric_20_word_bigram_entropy()

        word_bigrams = self.metric_21_frequent_word_bigrams()
        metrics_dict["m21_top_word_bigrams_count"] = len(word_bigrams)

        metrics_dict["m22_rare_ngram_coverage"] = self.metric_22_rare_ngram_coverage()
        metrics_dict["m23_perplexity_approximation"] = self.metric_23_perplexity_approximation()
        metrics_dict["m24_shannon_entropy"] = self.metric_24_shannon_entropy()
        metrics_dict["m25_cross_entropy_loss"] = self.metric_25_cross_entropy_loss()
        metrics_dict["m26_flesch_kincaid_grade"] = self.metric_26_flesch_kincaid_grade()
        metrics_dict["m27_flesch_reading_ease"] = self.metric_27_flesch_reading_ease()
        metrics_dict["m28_gunning_fog_index"] = self.metric_28_gunning_fog_index()
        metrics_dict["m29_discourse_marker_frequency"] = self.metric_29_discourse_marker_frequency()
        metrics_dict["m30_connective_diversity"] = self.metric_30_connective_diversity()

        # TIER 3
        metrics_dict["m31_liwc_authenticity_proxy"] = self.metric_31_liwc_authenticity_proxy()
        metrics_dict["m32_coreference_pattern_frequency"] = self.metric_32_coreference_pattern_frequency()
        metrics_dict["m33_semantic_relatedness"] = self.metric_33_semantic_relatedness()
        metrics_dict["m34_mattr"] = self.metric_34_mattr_moving_average_ttr()
        metrics_dict["m35_honores_r"] = self.metric_35_honores_r()

        zipfian = self.metric_36_zipfian_distribution()
        metrics_dict["m36_zipfian_alpha"] = zipfian["alpha"]
        metrics_dict["m36_zipfian_r_squared"] = zipfian["r_squared"]

        metrics_dict["m37_colocation_patterns"] = self.metric_37_colocation_patterns()
        metrics_dict["m38_grammar_error_rate"] = self.metric_38_grammar_error_rate()
        metrics_dict["m39_sentence_construction_uniformity"] = self.metric_39_sentence_construction_uniformity()
        metrics_dict["m40_token_predictability_variance"] = self.metric_40_token_predictability_variance()

        # Optional pairwise metrics
        if text2_tokens is not None and len(text2_tokens) > 0:
            vec1 = np.array([
                metrics_dict.get("m01_average_word_length", 0.0),
                metrics_dict.get("m02_type_token_ratio", 0.0),
                metrics_dict.get("m03_yules_k", 0.0),
                metrics_dict.get("m06_average_sentence_length", 0.0),
                metrics_dict.get("m07_sentence_length_std", 0.0),
                metrics_dict.get("m18_modal_frequency", 0.0),
                metrics_dict.get("m20_word_bigram_entropy", 0.0),
                metrics_dict.get("m23_perplexity_approximation", 0.0),
                metrics_dict.get("m24_shannon_entropy", 0.0),
                metrics_dict.get("m26_flesch_kincaid_grade", 0.0),
            ], dtype=float)

            text2 = " ".join(text2_tokens)
            ext2 = AuthorshipMetricsExtractor(text2)
            m2 = ext2.extract_all_metrics(text2_tokens=None)

            vec2 = np.array([
                m2.get("m01_average_word_length", 0.0),
                m2.get("m02_type_token_ratio", 0.0),
                m2.get("m03_yules_k", 0.0),
                m2.get("m06_average_sentence_length", 0.0),
                m2.get("m07_sentence_length_std", 0.0),
                m2.get("m18_modal_frequency", 0.0),
                m2.get("m20_word_bigram_entropy", 0.0),
                m2.get("m23_perplexity_approximation", 0.0),
                m2.get("m24_shannon_entropy", 0.0),
                m2.get("m26_flesch_kincaid_grade", 0.0),
            ], dtype=float)

            metrics_dict["m14_cosine_similarity"] = self.metric_14_cosine_similarity(vec1, vec2)

            f1 = Counter(self.tokens_no_punct)
            f2 = Counter(text2_tokens)
            vocab = sorted(set(f1.keys()) | set(f2.keys()))
            dist1 = [f1.get(w, 0) for w in vocab]
            dist2 = [f2.get(w, 0) for w in vocab]
            metrics_dict["m15_jensen_shannon_divergence"] = self.metric_15_jensen_shannon_divergence(dist1, dist2)

        return metrics_dict


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    sample_text = """
    Artificial intelligence is transforming our world. Machine learning models can now understand
    natural language with remarkable accuracy. However, there are important differences between
    AI-generated content and authentic human writing. When we examine stylistic patterns carefully,
    we can identify subtle markers that distinguish original authorship from computational generation.
    """

    extractor = AuthorshipMetricsExtractor(sample_text)
    metrics = extractor.extract_all_metrics()

    print("Sample Metrics Extraction:")
    print(f"  Type-Token Ratio: {metrics['m02_type_token_ratio']:.4f}")
    print(f"  Yule's K: {metrics['m03_yules_k']:.4f}")
    print(f"  Flesch-Kincaid Grade: {metrics['m26_flesch_kincaid_grade']:.4f}")
    print(f"  Perplexity: {metrics['m23_perplexity_approximation']:.4f}")
    print(f"  Total metrics extracted: {len(metrics)}")
