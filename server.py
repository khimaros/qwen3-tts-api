from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import time
from werkzeug.exceptions import HTTPException

import config
import tts

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": config.CORS_ALLOWED_ORIGIN}})


@app.route("/v1/audio/speech", methods=["POST"])
def speech_api():
    """
    OpenAI API compatible endpoint for generating speech from text.
    It supports a limited set of parameters.
    """
    data = request.get_json()

    # Extract parameters from the request
    # OpenAI API compatible parameters only
    text = data.get("input")
    model = data.get("model", config.MODEL)
    voice = data.get("voice")
    response_format = data.get("response_format", "wav")

    print(f"Got request: {data}")

    # Validate parameters
    if not text:
        return (
            jsonify({"error": "Input text is required."}),
            400,
        )
    if voice not in config.SUPPORTED_VOICES:
        return jsonify({"error": "Unsupported voice specified."}), 400
    if response_format not in config.SUPPORTED_RESPONSE_FORMATS:
        return (
            jsonify(
                {
                    "error": "Unsupported response format specified. Got: "
                    + response_format
                }
            ),
            400,
        )

    # Generate audio from the text
    start_time = time.time()
    audio_data = tts.generate_audio(text, voice, response_format, model)
    elapsed = time.time() - start_time
    print(f"generation took {elapsed:.2f}s ({len(text)} chars)")

    # Create a BytesIO object for the response
    audio_io = io.BytesIO(audio_data)
    audio_io.seek(0)

    # Set the appropriate MIME type based on the requested response format
    mime_type = "audio/" + response_format

    return send_file(
        audio_io,
        mimetype=mime_type,
        as_attachment=True,
        download_name=f"speech.{response_format}",
    )


@app.route("/tts", methods=["POST"])
def tts_api():
    """
    Endpoint for generating audio from text using the TTS model.
    This endpoint accepts more parameters used by the model.
    """
    data = request.get_json()

    # Extract parameters from the request
    text = data.get("text")
    model = data.get("model", config.MODEL)
    voice = data.get("predefined_voice_id")
    speed = float(data.get("speed_factor", 1.0))
    seed = data.get("seed", config.SEED)
    response_format = data.get("response_format", "wav")
    params = {}

    if "temperature" in data and data.get("temperature") is not None:
        params["temperature"] = data.get("temperature")

    if "instruct" in data and data.get("instruct") is not None:
        params["instruct"] = data.get("instruct")

    print(f"Got request: {data}")
    chunk_size = data.get("chunk_size", 10000)

    # Validate parameters
    if not text:
        return (
            jsonify({"error": "Input text is required."}),
            400,
        )
    if voice not in config.SUPPORTED_VOICES:
        return jsonify({"error": "Unsupported voice specified."}), 400
    if response_format not in config.SUPPORTED_RESPONSE_FORMATS:
        return (
            jsonify(
                {
                    "error": "Unsupported response format specified. Got: "
                    + response_format
                }
            ),
            400,
        )

    if chunk_size < 1:
        return jsonify({"error": "Chunk size must be greater than 0."}), 400

    # Generate audio from the text
    start_time = time.time()
    audio_data = tts.generate_audio(
        text, voice, response_format, model, speed, chunk_size, seed, params
    )
    elapsed = time.time() - start_time
    print(f"generation took {elapsed:.2f}s ({len(text)} chars)")

    # Create a BytesIO object for the response
    audio_io = io.BytesIO(audio_data)
    audio_io.seek(0)

    # Set the appropriate MIME type based on the requested response format
    mime_type = "audio/" + response_format

    return send_file(
        audio_io,
        mimetype=mime_type,
        as_attachment=True,
        download_name=f"speech.{response_format}",
    )


@app.route("/health", methods=["GET"])
def health_api():
    return "", 200


@app.route("/voices", methods=["GET"])
@app.route("/v1/audio/voices", methods=["GET"])
def get_voices_api():
    return jsonify({"voices": config.SUPPORTED_VOICES})


@app.route("/models", methods=["GET"])
def get_models_api():
    from engine import index

    return jsonify({"models": list(index.get_available_model_engines().keys())})


@app.get("/get_predefined_voices")
def get_predefined_voices_api():
    """Returns a list of predefined voices with display names and filenames."""
    """List of dictionaries: [{"display_name": "Formatted Name", "filename": "original_file.wav"}, ...]"""
    try:
        response = jsonify(
            [
                {"display_name": voice, "filename": voice}
                for voice in config.SUPPORTED_VOICES
            ]
        )
        return response

    except Exception:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices list."
        )


# These are strictly for ST to avoid errors
@app.get("/get_reference_files")
def get_reference_files_api():
    # do nothing
    return "[]"


@app.get("/api/ui/initial-data")
def get_api_ui_initial_data():
    # do nothing
    return "{}"
