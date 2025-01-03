import asyncio
import io
import json

from openai import AsyncOpenAI
from pydub import AudioSegment

aclient = AsyncOpenAI()
max_chunk_size = 25 * 1024 * 1024  # 25 MB


def audio_segment_to_file(segment, format="mp3"):
    buffer = io.BytesIO()
    segment.export(buffer, format=format)
    buffer.seek(0)
    buffer.name = f"chunk.{format}"  # Assign a name with the correct extension
    return buffer


async def atranscribe_audio(file, cutoff_time=None):
    kwargs = {"response_format": "verbose_json", "timestamp_granularities": ["word"], "language": "en", "prompt": ""}
    # Load the audio file
    audio = AudioSegment.from_mp3(file)
    if cutoff_time:
        audio = audio[: cutoff_time * 1000]
    # Calculate chunk length in milliseconds
    chunk_length_ms = int((max_chunk_size / len(audio.raw_data)) * len(audio))

    async def transcribe_chunk(chunk, chunk_number, start_time):
        chunk_file = audio_segment_to_file(chunk, format="mp3")
        try:
            transcription = await aclient.audio.transcriptions.create(model="whisper-1", file=chunk_file, **kwargs)
            print(f"Processed chunk {chunk_number}")
            return transcription, start_time
        except Exception as e:
            print(f"Error processing chunk {chunk_number}: {e}")
            return None, start_time

    async def transcribe_audio():
        if len(audio) > chunk_length_ms:
            chunks = [audio[i : i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

            tasks = [transcribe_chunk(chunk, i + 1, i * chunk_length_ms / 1000) for i, chunk in enumerate(chunks)]
            results = await asyncio.gather(*tasks)

            full_transcription = []
            for result, start_time in results:
                if result:
                    for word in result.words:
                        word["start"] += start_time
                        word["end"] += start_time
                    full_transcription.extend(result.words)
        else:
            audio_file = audio_segment_to_file(audio, format="mp3")
            try:
                transcription = await aclient.audio.transcriptions.create(model="whisper-1", file=audio_file, **kwargs)
                full_transcription = transcription.words
            except Exception as e:
                print(f"Error processing audio file: {e}")
                full_transcription = []

        return full_transcription

    return await transcribe_audio()


def format_transcript(words, chunk_size: int = 10):
    transcript = []
    current_speaker = "Speaker 1"
    current_text = []
    current_start = None
    current_end = None
    current_duration = 0.0  # Track the duration of the current chunk

    for word in words:
        if current_start is None:
            current_start = word["start"]  # Set the start time for the first word in the chunk

        current_text.append(word["word"])
        current_end = word["end"]
        current_duration = current_end - current_start  # Calculate the duration of the current chunk

        # Check if the current duration exceeds the specified chunk size
        if current_duration >= chunk_size:
            transcript.append(
                {"speaker": current_speaker, "start": current_start, "end": current_end, "text": " ".join(current_text)}
            )
            # Reset for the next chunk
            current_text = []
            current_start = None
            current_duration = 0.0  # Reset duration

    # Add any remaining text if it exists
    if current_text:
        transcript.append(
            {"speaker": current_speaker, "start": current_start, "end": current_end, "text": " ".join(current_text)}
        )

    return transcript


if __name__ == "__main__":
    file = "/Users/pranaviyer/Downloads/Marcos Marino Beiras - Resurgence and non perturbative topological strings.mp3"
    words = asyncio.run(atranscribe_audio(file, cutoff_time=100))
    transcript = format_transcript(words, chunk_size=3)

    # save to json
    with open("transcription.json", "w") as f:
        json.dump({"transcript": transcript}, f, indent=2)
