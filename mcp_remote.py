"""
Video Watch - Remote MCP Server on Modal
Fully cloud-hosted MCP server that lets Claude "watch" videos.
"""

import modal
import base64
import json
import subprocess
import tempfile
from pathlib import Path

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl")
    .pip_install(
        "yt-dlp",
        "curl_cffi",  # For browser impersonation (TikTok, Instagram, etc.)
        "brotli",     # For compression support
        "openai-whisper",
        "torch",
        "mcp[cli]",
        "starlette",
        "sse-starlette",
        "uvicorn",
    )
)

app = modal.App("video-watch-mcp", image=image)


def extract_frames(video_path: str, output_dir: str, fps: float = 0.5) -> list[str]:
    """Extract frames from video at specified fps."""
    output_pattern = f"{output_dir}/frame_%04d.jpg"
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps},drawtext=text='%{{pts\\:hms}}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5",
        "-q:v", "2",
        output_pattern
    ], capture_output=True, check=True)
    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video as wav file."""
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path
    ], capture_output=True, check=True)
    return output_path


@app.function(gpu="T4", timeout=300)
def process_video(url: str, fps: float = 0.5, max_frames: int = 10):
    """Download and process a video URL."""
    import whisper

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/video.mp4"
        audio_path = f"{tmpdir}/audio.wav"
        frames_dir = f"{tmpdir}/frames"
        Path(frames_dir).mkdir()

        # Download video (with browser impersonation for tricky platforms)
        # Format: try 720p or lower, fallback to best available
        result = subprocess.run([
            "yt-dlp",
            "-f", "best[height<=720]/best",
            "-o", video_path,
            "--no-playlist",
            "--impersonate", "chrome",
            url
        ], capture_output=True, text=True)

        if result.returncode != 0:
            return {"success": False, "error": f"Download failed: {result.stderr}"}

        # Get duration
        probe = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", video_path
        ], capture_output=True, text=True)

        duration = 0
        try:
            duration = float(json.loads(probe.stdout)["format"]["duration"])
        except:
            pass

        # Extract frames
        frame_paths = extract_frames(video_path, frames_dir, fps)

        if len(frame_paths) > max_frames:
            step = len(frame_paths) / max_frames
            frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

        frames_b64 = []
        for fp in frame_paths:
            with open(fp, "rb") as f:
                frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))

        # Transcribe
        transcript = ""
        try:
            extract_audio(video_path, audio_path)
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            transcript = result["text"]
        except Exception as e:
            transcript = f"[Transcription failed: {e}]"

        return {
            "success": True,
            "duration_seconds": duration,
            "frame_count": len(frames_b64),
            "frames": frames_b64,
            "transcript": transcript,
            "url": url
        }


@app.function()
@modal.asgi_app()
def mcp_server():
    """ASGI app that serves MCP over SSE."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    from sse_starlette.sse import EventSourceResponse
    import asyncio

    async def handle_sse(request):
        """Handle SSE connection for MCP."""
        async def event_generator():
            # Send server info
            yield {
                "event": "endpoint",
                "data": json.dumps({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                })
            }

        return EventSourceResponse(event_generator())

    async def handle_mcp(request):
        """Handle MCP JSON-RPC requests."""
        body = await request.json()
        method = body.get("method", "")
        params = body.get("params", {})
        request_id = body.get("id")

        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "video-watch",
                        "version": "1.0.0"
                    }
                }
            })

        elif method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [{
                        "name": "watch_video",
                        "description": "Process a video URL so Claude can 'watch' it. Downloads the video, extracts key frames as images, and transcribes the audio. Works with TikTok, YouTube, Instagram, Twitter/X, and most video platforms.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "The video URL (TikTok, YouTube, etc.)"
                                },
                                "max_frames": {
                                    "type": "integer",
                                    "description": "Maximum frames to extract (default 10)",
                                    "default": 10
                                }
                            },
                            "required": ["url"]
                        }
                    }]
                }
            })

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})

            if tool_name == "watch_video":
                url = args.get("url")
                max_frames = min(args.get("max_frames", 10), 20)

                # Call the GPU function
                result = process_video.remote(url, 0.5, max_frames)

                if not result.get("success"):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": f"Error: {result.get('error', 'Unknown error')}"
                            }]
                        }
                    })

                # Build response
                duration = result.get("duration_seconds", 0)
                minutes = int(duration // 60)
                seconds = int(duration % 60)

                content = [{
                    "type": "text",
                    "text": f"**Video:** {url}\n**Duration:** {minutes}:{seconds:02d}\n**Frames:** {result.get('frame_count', 0)}\n\n**Transcript:**\n{result.get('transcript', '[No transcript]')}"
                }]

                # Add frames as images
                for frame_b64 in result.get("frames", []):
                    content.append({
                        "type": "image",
                        "data": frame_b64,
                        "mimeType": "image/jpeg"
                    })

                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": content}
                })

        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        })

    async def health(request):
        return JSONResponse({"status": "ok", "service": "video-watch-mcp"})

    routes = [
        Route("/", handle_mcp, methods=["POST"]),
        Route("/sse", handle_sse),
        Route("/health", health),
    ]

    return Starlette(routes=routes)
