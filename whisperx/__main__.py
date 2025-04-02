import argparse
import os
import requests
import torch
import whisperx

def download_audio(url, auth_details=None):
    try:
        if auth_details:
            response = requests.get(url, auth=(auth_details["username"], auth_details["password"]))
        else:
            response = requests.get(url)
        response.raise_for_status()
        temp_path = os.path.join(os.getcwd(), "temp_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(response.content)
        return temp_path
    except Exception as e:
        raise Exception(f"Failed to download audio: {e}")

def transcribe_with_whisperx(input_source, auth_details=None):
    # Handle URL input
    if input_source.startswith("http"):
        audio_path = download_audio(input_source, auth_details)
    else:
        audio_path = input_source

    # Load WhisperX model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("large-v2", device)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    # Add speaker diarization
    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)
    diarization = whisperx.DiarizationPipeline(use_auth_token=None, device=device)  # Add HF token if needed
    diarize_segments = diarization(audio)
    result = whisperx.assign_word_speakers(diarization, result)

    # Clean up temporary file if downloaded
    if input_source.startswith("http"):
        os.remove(audio_path)

    # Format the result for output
    output = []
    for segment in result["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "")
        output.append(f"Speaker {speaker}: {text}")
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="WhisperX Transcription with URL Support")
    parser.add_argument("input", help="Path to audio file or URL")
    parser.add_argument("--username", help="Username for basic auth", default=None)
    parser.add_argument("--password", help="Password for basic auth", default=None)
    args = parser.parse_args()

    auth_details = None
    if args.username and args.password:
        auth_details = {"username": args.username, "password": args.password}

    try:
        result = transcribe_with_whisperx(args.input, auth_details)
        print(result)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()