from typing import Any
import asyncio
import datetime
import json
import pathlib

import discord

from .models import MemoryCategory

MEMORY_DIR = pathlib.Path.home() / ".voxbot" / "soul"
PEOPLE_FILE = MEMORY_DIR / "people.json"


class MemoryService:
    def __init__(self, people_file: pathlib.Path = PEOPLE_FILE) -> None:
        self.people_file = people_file
        self._lock = asyncio.Lock()

    def _empty_memory(self) -> dict[str, Any]:
        return {"users": {}}

    def _read_memory(self) -> dict[str, Any]:
        if not self.people_file.exists():
            return self._empty_memory()

        with self.people_file.open(encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            msg = f"{self.people_file} must contain a JSON object"
            raise ValueError(msg)

        users = data.setdefault("users", {})
        if not isinstance(users, dict):
            msg = f"{self.people_file} field 'users' must contain a JSON object"
            raise ValueError(msg)

        return data

    def _write_memory(self, data: dict[str, Any]) -> None:
        self.people_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.people_file.with_suffix(".json.tmp")

        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

        tmp_file.replace(self.people_file)

    @staticmethod
    def _normalize_person_name(name: str) -> str:
        return " ".join(name.casefold().split())

    def _resolve_person(
        self,
        message: discord.Message,
        person_id: str | None,
        person_name: str | None,
    ) -> tuple[str, str]:
        if person_id:
            return str(person_id), person_name or str(person_id)

        if person_name:
            normalized = self._normalize_person_name(person_name)
            candidates = [message.author, *message.mentions]
            for user in candidates:
                names = [user.name, user.display_name]
                if global_name := getattr(user, "global_name", None):
                    names.append(global_name)

                if normalized in {self._normalize_person_name(name) for name in names if name}:
                    return str(user.id), user.display_name

            return f"name:{normalized}", person_name

        return str(message.author.id), message.author.display_name

    @staticmethod
    def _ensure_user(data: dict[str, Any], person_key: str, display_name: str) -> dict[str, Any]:
        users = data.setdefault("users", {})
        user = users.setdefault(person_key, {})
        user.setdefault("display_names", [])
        user.setdefault("facts", [])

        if display_name and display_name not in user["display_names"]:
            user["display_names"].append(display_name)

        return user

    def summary(self, message: discord.Message | None) -> str:
        if message is None:
            return "- No current author; this is a background identity check."

        try:
            data = self._read_memory()
        except (OSError, ValueError, json.JSONDecodeError) as e:
            return f"- Memory unavailable: {e}"

        user = data.get("users", {}).get(str(message.author.id))
        if not user:
            return "- No saved memories for this user."

        facts = user.get("facts", [])
        if not isinstance(facts, list) or not facts:
            return "- No saved memories for this user."

        lines = []
        for fact in facts[-20:]:
            if not isinstance(fact, dict):
                continue

            category = fact.get("category", "other")
            text = fact.get("fact")
            if text:
                lines.append(f"- {category}: {text}")

        return "\n".join(lines) or "- No saved memories for this user."

    async def remember(
        self,
        message: discord.Message | None,
        fact: str,
        category: MemoryCategory = "other",
        person_id: str | None = None,
        person_name: str | None = None,
    ) -> str:
        fact = " ".join(fact.strip().split())
        if not fact:
            return "No memory stored because the fact was empty."

        if len(fact) > 300:
            return "No memory stored because the fact was too long. Store a shorter, specific fact."

        if message is None:
            return "No memory stored because there is no current Discord message."

        person_key, display_name = self._resolve_person(message, person_id, person_name)
        now = datetime.datetime.now(datetime.UTC).isoformat()

        try:
            async with self._lock:
                data = self._read_memory()
                user = self._ensure_user(data, person_key, display_name)
                facts = user.setdefault("facts", [])
                if not isinstance(facts, list):
                    facts = []
                    user["facts"] = facts

                normalized_fact = fact.casefold()
                for existing in facts:
                    if not isinstance(existing, dict):
                        continue

                    same_category = existing.get("category") == category
                    same_fact = str(existing.get("fact", "")).casefold() == normalized_fact
                    if same_category and same_fact:
                        existing["updated_at"] = now
                        self._write_memory(data)
                        return f"Memory already existed for {display_name}; refreshed it."

                facts.append(
                    {
                        "category": category,
                        "fact": fact,
                        "confidence": "explicit",
                        "source": "discord",
                        "source_message_id": str(message.id),
                        "channel_id": str(message.channel.id),
                        "guild_id": str(message.guild.id) if message.guild else None,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                self._write_memory(data)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            return f"Failed to store memory: {e}"

        return f"Remembered for {display_name}: {fact}"

    async def forget(
        self,
        message: discord.Message | None,
        fact_fragment: str = "",
        category: MemoryCategory | None = None,
        person_id: str | None = None,
        person_name: str | None = None,
    ) -> str:
        fact_fragment = " ".join(fact_fragment.strip().split()).casefold()
        if not fact_fragment and category is None:
            return "No memory forgotten because neither a fact fragment nor category was provided."

        if message is None:
            return "No memory forgotten because there is no current Discord message."

        person_key, display_name = self._resolve_person(message, person_id, person_name)

        try:
            async with self._lock:
                data = self._read_memory()
                users = data.get("users", {})
                if not isinstance(users, dict) or person_key not in users:
                    return f"No memories found for {display_name}."

                user = users[person_key]
                facts = user.get("facts", [])
                if not isinstance(facts, list):
                    return f"No memories found for {display_name}."

                kept = []
                removed = 0
                for existing in facts:
                    if not isinstance(existing, dict):
                        kept.append(existing)
                        continue

                    category_matches = category is None or existing.get("category") == category
                    fact_matches = not fact_fragment or fact_fragment in str(existing.get("fact", "")).casefold()
                    if category_matches and fact_matches:
                        removed += 1
                    else:
                        kept.append(existing)

                if removed == 0:
                    return f"No matching memories found for {display_name}."

                user["facts"] = kept
                self._write_memory(data)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            return f"Failed to forget memory: {e}"

        return f"Forgot {removed} memory item(s) for {display_name}."
