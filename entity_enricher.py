import json
import os
import re
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

import requests


class WikiEntityEnricher:
    def __init__(self, cache_path: str = "cache/wiki_sentence_cache.json", timeout: int = 4):
        self.timeout = timeout
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.tagme_token = os.environ.get("TAGME_TOKEN", "").strip()
        self.cache: Dict[str, str] = self._load_cache()

        try:
            import jieba  # type: ignore
            self.jieba = jieba
        except Exception:
            self.jieba = None

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _first_sentence(self, text: str) -> str:
        if not text:
            return ""
        parts = re.split(r"(?<=[。！？.!?])\s+", text.strip())
        sent = parts[0].strip() if parts else text.strip()
        if len(sent) > 180:
            sent = sent[:180].rstrip() + "..."
        return sent

    def _detect_language(self, text: str) -> str:
        if not text:
            return "zh"
        zh_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        en_count = len(re.findall(r"[A-Za-z]", text))
        if en_count > zh_count:
            return "en"
        return "zh"

    def _extract_entities_jieba(self, text: str, top_k: int = 5) -> List[str]:
        if not text:
            return []
        if self.jieba is None:
            tokens = re.findall(r"[A-Za-z][A-Za-z\-]+|[\u4e00-\u9fff]{2,}", text)
            uniq = []
            for token in tokens:
                if token not in uniq:
                    uniq.append(token)
            return uniq[:top_k]

        words = self.jieba.cut(text)
        candidates = []
        stop = {"我们", "你们", "他们", "这个", "那个", "以及", "但是", "因为", "所以"}
        for word in words:
            token = word.strip()
            if len(token) < 2:
                continue
            if token in stop:
                continue
            if re.fullmatch(r"\d+", token):
                continue
            candidates.append(token)

        uniq = []
        for token in candidates:
            if token not in uniq:
                uniq.append(token)
        return uniq[:top_k]

    def _extract_entities_tagme(self, text: str, top_k: int = 5) -> List[str]:
        if not text or not self.tagme_token:
            return []
        try:
            url = "https://tagme.d4science.org/tagme/tag"
            params = {
                "gcube-token": self.tagme_token,
                "text": text[:800],
                "lang": "en",
                "include_abstract": "false",
            }
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            annotations = data.get("annotations", []) if isinstance(data, dict) else []
            entities = []
            for ann in annotations:
                title = ann.get("title")
                rho = ann.get("rho", 0)
                if title and rho >= 0.1:
                    entities.append(title)
            uniq = []
            for e in entities:
                if e not in uniq:
                    uniq.append(e)
            return uniq[:top_k]
        except Exception:
            return []

    def _wiki_first_sentence(self, entity: str, lang: str = "zh") -> str:
        key = f"{lang}:{entity}"
        if key in self.cache:
            return self.cache[key]

        try:
            search_url = f"https://{lang}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": entity,
                "format": "json",
                "srlimit": 1,
            }
            search_resp = self.session.get(search_url, params=params, timeout=self.timeout)
            search_resp.raise_for_status()
            search_data = search_resp.json()
            search_list = search_data.get("query", {}).get("search", [])
            if not search_list:
                self.cache[key] = ""
                return ""

            title = search_list[0].get("title", "")
            if not title:
                self.cache[key] = ""
                return ""

            summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
            summary_resp = self.session.get(summary_url, timeout=self.timeout)
            summary_resp.raise_for_status()
            summary_data = summary_resp.json()
            extract = summary_data.get("extract", "") if isinstance(summary_data, dict) else ""
            sent = self._first_sentence(extract)
            self.cache[key] = sent
            return sent
        except Exception:
            self.cache[key] = ""
            return ""
        finally:
            if len(self.cache) % 20 == 0:
                self._save_cache()

    def build_background_text(self, text: str, max_entities: int = 3) -> str:
        lang = self._detect_language(text)

        if lang == "zh":
            merged = self._extract_entities_jieba(text, top_k=max_entities)
        else:
            merged = self._extract_entities_tagme(text, top_k=max_entities)

        if not merged:
            return ""

        sentences = []
        for entity in merged:
            sentence = self._wiki_first_sentence(entity, lang=lang)
            if not sentence:
                back_lang = "en" if lang == "zh" else "zh"
                sentence = self._wiki_first_sentence(entity, lang=back_lang)
            if sentence:
                sentences.append(sentence)

        if not sentences:
            return ""

        merged_text = " ".join(sentences)
        return merged_text[:512]
