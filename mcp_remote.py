"""
Video Watch - Remote MCP Server on Modal
Fully cloud-hosted MCP server that lets Claude "watch" videos.

Three tools:
- video_listen: Transcript only (lightweight)
- video_see: Frames only (visual content)
- watch_video: Both (full experience)
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
    .apt_install("ffmpeg", "curl", "unzip")
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
    .run_commands(
        "curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh",
    )
)

app = modal.App("video-watch-mcp", image=image)


def download_video(url: str, video_path: str) -> dict:
    """Download video, return success/error."""
    result = subprocess.run([
        "yt-dlp",
        "-f", "best[height<=720]/best",
        "-o", video_path,
        "--no-playlist",
        "--impersonate", "chrome",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        url
    ], capture_output=True, text=True)

    if result.returncode != 0:
        return {"success": False, "error": result.stderr}
    return {"success": True}


def get_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", video_path
    ], capture_output=True, text=True)

    try:
        return float(json.loads(probe.stdout)["format"]["duration"])
    except:
        return 0


def extract_frames(video_path: str, output_dir: str, fps: float = 0.5, max_frames: int = 5) -> list[str]:
    """Extract frames from video, return as base64 list."""
    output_pattern = f"{output_dir}/frame_%04d.jpg"

    # Smaller frames (480px width), more compression
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps},scale=480:-1,drawtext=text='%{{pts\\:hms}}':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5",
        "-q:v", "6",
        output_pattern
    ], capture_output=True, check=True)

    frame_paths = sorted(Path(output_dir).glob("frame_*.jpg"))

    # Limit frames
    if len(frame_paths) > max_frames:
        step = len(frame_paths) / max_frames
        frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

    frames_b64 = []
    for fp in frame_paths:
        with open(fp, "rb") as f:
            frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))

    return frames_b64


def transcribe_audio(video_path: str, audio_path: str) -> str:
    """Extract and transcribe audio."""
    import whisper

    # Extract audio
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ], capture_output=True, check=True)

    # Transcribe
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


@app.function(gpu="T4", timeout=300)
def process_listen(url: str):
    """Audio/transcript only - lightweight."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/video.mp4"
        audio_path = f"{tmpdir}/audio.wav"

        dl = download_video(url, video_path)
        if not dl["success"]:
            return {"success": False, "error": dl["error"]}

        duration = get_duration(video_path)

        try:
            transcript = transcribe_audio(video_path, audio_path)
        except Exception as e:
            transcript = f"[Transcription failed: {e}]"

        return {
            "success": True,
            "duration_seconds": duration,
            "transcript": transcript,
            "url": url
        }


@app.function(timeout=300)
def process_see(url: str, max_frames: int = 5):
    """Frames only - no GPU needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/video.mp4"
        frames_dir = f"{tmpdir}/frames"
        Path(frames_dir).mkdir()

        dl = download_video(url, video_path)
        if not dl["success"]:
            return {"success": False, "error": dl["error"]}

        duration = get_duration(video_path)
        frames = extract_frames(video_path, frames_dir, fps=0.5, max_frames=max_frames)

        return {
            "success": True,
            "duration_seconds": duration,
            "frame_count": len(frames),
            "frames": frames,
            "url": url
        }


@app.function(gpu="T4", timeout=300)
def process_watch(url: str, max_frames: int = 5):
    """Full experience - frames + transcript."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/video.mp4"
        audio_path = f"{tmpdir}/audio.wav"
        frames_dir = f"{tmpdir}/frames"
        Path(frames_dir).mkdir()

        dl = download_video(url, video_path)
        if not dl["success"]:
            return {"success": False, "error": dl["error"]}

        duration = get_duration(video_path)
        frames = extract_frames(video_path, frames_dir, fps=0.5, max_frames=max_frames)

        try:
            transcript = transcribe_audio(video_path, audio_path)
        except Exception as e:
            transcript = f"[Transcription failed: {e}]"

        return {
            "success": True,
            "duration_seconds": duration,
            "frame_count": len(frames),
            "frames": frames,
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

    async def handle_sse(request):
        async def event_generator():
            yield {
                "event": "endpoint",
                "data": json.dumps({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                })
            }
        return EventSourceResponse(event_generator())

    def format_duration(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    async def handle_mcp(request):
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
                    "serverInfo": {"name": "video-watch", "version": "2.0.0"}
                }
            })

        elif method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "video_listen",
                            "description": "Get transcript of a video. Lightweight - returns only the spoken/audio content as text. Best for: talking head videos, podcasts, commentary, interviews, tutorials with narration.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string", "description": "Video URL"}
                                },
                                "required": ["url"]
                            }
                        },
                        {
                            "name": "video_see",
                            "description": "Get visual frames from a video. Returns key frames as images, no audio transcription. Best for: dance videos, visual art, scenery, silent clips, memes, anything where visuals matter more than audio.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string", "description": "Video URL"},
                                    "max_frames": {"type": "integer", "description": "Max frames (default 5)", "default": 5}
                                },
                                "required": ["url"]
                            }
                        },
                        {
                            "name": "watch_video",
                            "description": "Full video experience - frames AND transcript. Uses more context but gives complete picture. Use when both visuals and audio matter.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string", "description": "Video URL"},
                                    "max_frames": {"type": "integer", "description": "Max frames (default 5)", "default": 5}
                                },
                                "required": ["url"]
                            }
                        }
                    ]
                }
            })

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            url = args.get("url")

            if not url:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "Error: No URL provided"}]}
                })

            # Route to appropriate processor
            if tool_name == "video_listen":
                result = process_listen.remote(url)

                if not result.get("success"):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"Error: {result.get('error')}"}]}
                    })

                content = [{
                    "type": "text",
                    "text": f"**Video:** {url}\n**Duration:** {format_duration(result.get('duration_seconds', 0))}\n\n**Transcript:**\n{result.get('transcript', '[No transcript]')}"
                }]

            elif tool_name == "video_see":
                max_frames = min(args.get("max_frames", 5), 10)
                result = process_see.remote(url, max_frames)

                if not result.get("success"):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"Error: {result.get('error')}"}]}
                    })

                content = [{
                    "type": "text",
                    "text": f"**Video:** {url}\n**Duration:** {format_duration(result.get('duration_seconds', 0))}\n**Frames:** {result.get('frame_count', 0)}"
                }]

                for frame_b64 in result.get("frames", []):
                    content.append({"type": "image", "data": frame_b64, "mimeType": "image/jpeg"})

            elif tool_name == "watch_video":
                max_frames = min(args.get("max_frames", 5), 10)
                result = process_watch.remote(url, max_frames)

                if not result.get("success"):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"Error: {result.get('error')}"}]}
                    })

                content = [{
                    "type": "text",
                    "text": f"**Video:** {url}\n**Duration:** {format_duration(result.get('duration_seconds', 0))}\n**Frames:** {result.get('frame_count', 0)}\n\n**Transcript:**\n{result.get('transcript', '[No transcript]')}"
                }]

                for frame_b64 in result.get("frames", []):
                    content.append({"type": "image", "data": frame_b64, "mimeType": "image/jpeg"})
            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
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
        return JSONResponse({"status": "ok", "service": "video-watch-mcp", "version": "2.0.0"})

    routes = [
        Route("/", handle_mcp, methods=["POST"]),
        Route("/sse", handle_sse),
        Route("/health", health),
    ]

    return Starlette(routes=routes)
