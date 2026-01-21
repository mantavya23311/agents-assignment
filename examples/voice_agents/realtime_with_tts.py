import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, room_io
from livekit.agents.llm import function_tool
from livekit.plugins import google, openai  # noqa: F401

logger = logging.getLogger("realtime-with-tts")
logger.setLevel(logging.INFO)

load_dotenv()

# ---------------------------
# NEW: interruption logic
# ---------------------------

agent_is_speaking = False

IGNORE_WORDS = {
    "yeah", "ok", "okay", "hmm", "uh-huh", "right"
}

INTERRUPT_WORDS = {
    "stop", "wait", "no", "cancel"
}


def should_interrupt(text: str, agent_speaking: bool) -> bool:
    text = text.lower().strip()
    words = text.split()

    has_interrupt = any(w in INTERRUPT_WORDS for w in words)
    only_ignore = all(w in IGNORE_WORDS for w in words)

    if agent_speaking:
        if has_interrupt:
            return True          # explicit command
        if only_ignore:
            return False         # passive acknowledgement
        return True              # real sentence
    else:
        return True              # agent silent → respond normally


# ---------------------------
# Agent definition
# ---------------------------

class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            llm=openai.realtime.RealtimeModel(modalities=["text"]),
            # llm=google.beta.realtime.RealtimeModel(modalities=[Modality.TEXT]),
            tts=openai.TTS(voice="ash"),
        )

    async def say(self, *args, **kwargs):
        """
        Override say() to track speaking state
        """
        global agent_is_speaking
        agent_is_speaking = True
        try:
            return await super().say(*args, **kwargs)
        finally:
            agent_is_speaking = False

    async def on_user_transcript(self, text: str):
        """
        NEW: handle user speech with context-aware interruption logic
        """
        global agent_is_speaking

        logger.info(f"User said: {text}")

        if should_interrupt(text, agent_is_speaking):
            if agent_is_speaking:
                logger.info("Interrupting agent speech")
                await self.interrupt()

            await self.handle_user_input(text)
        else:
            # Ignore filler words completely
            logger.info("Ignoring filler input while agent is speaking")

    @function_tool
    async def get_weather(self, location: str):
        """Called when the user asks about the weather."""
        logger.info(f"getting weather for {location}")
        return f"The weather in {location} is sunny, and the temperature is 20 degrees Celsius."


# ---------------------------
# Server setup
# ---------------------------

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession()

    await session.start(
        agent=WeatherAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_output=True,
            audio_output=True,
        ),
    )

    # Initial greeting
    session.generate_reply(instructions="say hello to the user in English")


if __name__ == "__main__":
    cli.run_app(server)
