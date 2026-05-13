import unicodedata

import discord


class IdentityService:
    """ """

    def __init__(self, home_guild_id: str | None = None) -> None:
        self.home_guild_id = home_guild_id

    def summary(self, bot: discord.Client) -> str:
        configured_guild_id = self.home_guild_id
        if not configured_guild_id:
            return "- Home guild is not configured."

        try:
            guild_id = int(configured_guild_id)
        except ValueError:
            return f"- Home guild id is invalid: {configured_guild_id}"

        if (guild := bot.get_guild(guild_id)) is None:
            return f"- Home guild {guild_id} is not currently available."

        return f"- Home guild: {guild.name} ({guild.id}); current display name: {guild.me.display_name}"

    @staticmethod
    def normalize_display_name(display_name: str) -> str:
        if any(unicodedata.category(char) == "Cc" for char in display_name):
            msg = "display name cannot contain control characters or newlines"
            raise ValueError(msg)

        normalized = " ".join(display_name.strip().split())
        if not normalized:
            msg = "display name cannot be empty"
            raise ValueError(msg)

        if len(normalized) > 32:
            msg = "display name must be 32 characters or fewer"
            raise ValueError(msg)

        return normalized

    async def change_display_name(
        self,
        bot: discord.Client,
        display_name: str,
        reason: str | None = None,
    ) -> str:
        try:
            normalized_name = self.normalize_display_name(display_name)
        except ValueError as e:
            return f"Name not changed: {e}."

        configured_guild_id = self.home_guild_id
        if not configured_guild_id:
            return "Name not changed: SOUL_HOME_GUILD_ID is not configured."

        try:
            guild_id = int(configured_guild_id)
        except ValueError:
            return "Name not changed: SOUL_HOME_GUILD_ID must be a Discord guild id."

        guild = bot.get_guild(guild_id)
        if guild is None:
            return f"Name not changed: home guild {guild_id} is not available."

        bot_user = bot.user
        if bot_user is None:
            return "Name not changed: bot user is not ready."

        member = guild.me or guild.get_member(bot_user.id)
        if member is None:
            try:
                member = await guild.fetch_member(bot_user.id)
            except discord.Forbidden:
                return "Name not changed: missing permission to find Voxbot in the home guild."
            except discord.HTTPException:
                return "Name not changed: could not find Voxbot in the home guild."

        audit_reason = "Voxbot self-renamed."
        if reason:
            clean_reason = " ".join(reason.strip().split())
            if clean_reason:
                audit_reason = f"Voxbot self-renamed: {clean_reason[:480]}"

        try:
            await member.edit(nick=normalized_name, reason=audit_reason)
        except discord.Forbidden:
            return "Name not changed: missing permission to change Voxbot's nickname in the home guild."
        except discord.HTTPException as e:
            return f"Name not changed: Discord rejected the nickname update ({e})."

        return f"Changed Voxbot's home-guild display name to {normalized_name}."
