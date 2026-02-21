"""
Media Fetch - Remote MCP Server on Modal
Fully cloud-hosted MCP server for consuming media: videos, podcasts, articles.

Video tools:
- video_listen: Transcript only (lightweight)
- video_see: Frames only (visual content)
- watch_video: Both (full experience)

Substack tools:
- substack_list_posts: List posts from any Substack publication
- substack_get_article: Fetch full article text
- substack_get_podcast: Transcribe podcast episodes via Whisper
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
        "feedparser",       # For Substack RSS parsing
        "beautifulsoup4",   # For article content extraction
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


# --- Substack helpers ---

def fetch_substack_rss(substack_slug: str) -> list[dict]:
    """Fetch and parse RSS feed for a Substack publication."""
    import feedparser
    from curl_cffi import requests as curl_requests

    feed_url = f"https://{substack_slug}.substack.com/feed"
    # curl_cffi impersonates a real browser to bypass Cloudflare challenges
    resp = curl_requests.get(feed_url, impersonate="chrome")
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code} fetching {feed_url}")
    raw = resp.content  # bytes for feedparser
    if not raw or len(raw) < 100:
        raise Exception(f"Empty or tiny response ({len(raw) if raw else 0} bytes)")
    # Check if we got HTML instead of XML (Cloudflare challenge)
    raw_start = raw[:200].decode(errors="replace").lower()
    if "<!doctype html" in raw_start or "just a moment" in raw_start:
        raise Exception(f"Got Cloudflare challenge instead of RSS ({len(raw)} bytes)")
    feed = feedparser.parse(raw)
    if feed.bozo and not feed.entries:
        raise Exception(f"feedparser error: {feed.bozo_exception}")

    posts = []
    for entry in feed.entries:
        # Check for audio enclosure (podcast episode)
        audio_url = None
        for link in getattr(entry, "enclosures", []):
            if link.get("type", "").startswith("audio/"):
                audio_url = link.get("href")
                break

        posts.append({
            "title": entry.get("title", "Untitled"),
            "url": entry.get("link", ""),
            "published": entry.get("published", ""),
            "summary": entry.get("summary", "")[:300],
            "has_audio": audio_url is not None,
            "audio_url": audio_url,
        })

    return posts


def fetch_substack_article(url: str) -> dict:
    """Fetch full article content from a Substack post URL."""
    from bs4 import BeautifulSoup
    from curl_cffi import requests as curl_requests

    resp = curl_requests.get(url, impersonate="chrome")
    if resp.status_code != 200:
        return {"success": False, "error": f"HTTP {resp.status_code}"}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title
    title_el = soup.find("h1", class_="post-title")
    if not title_el:
        title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else "Untitled"

    # Extract subtitle
    subtitle_el = soup.find("h3", class_="subtitle")
    subtitle = subtitle_el.get_text(strip=True) if subtitle_el else None

    # Extract author
    author_el = soup.find("a", class_="frontend-pencraft-Text-module__decoration-hover-underline--BEYAn")
    if not author_el:
        # Fallback: look in meta tags
        meta_author = soup.find("meta", {"name": "author"})
        author = meta_author["content"] if meta_author else "Unknown"
    else:
        author = author_el.get_text(strip=True)

    # Extract main content
    body = soup.find("div", class_="body")
    if not body:
        body = soup.find("div", class_="available-content")
    if not body:
        body = soup.find("article")

    if body:
        # Remove script/style tags
        for tag in body.find_all(["script", "style", "button"]):
            tag.decompose()
        content = body.get_text(separator="\n", strip=True)
    else:
        content = "[Could not extract article content]"

    # Check for audio
    audio_el = soup.find("audio")
    audio_url = None
    if audio_el and audio_el.get("src"):
        audio_url = audio_el["src"]
    else:
        source_el = soup.find("source", {"type": "audio/mpeg"})
        if source_el:
            audio_url = source_el.get("src")

    return {
        "success": True,
        "title": title,
        "subtitle": subtitle,
        "author": author,
        "content": content,
        "audio_url": audio_url,
        "url": url,
    }


def download_audio(url: str, audio_path: str) -> dict:
    """Download audio file from URL."""
    from curl_cffi import requests as curl_requests

    resp = curl_requests.get(url, impersonate="chrome")
    if resp.status_code != 200:
        return {"success": False, "error": f"HTTP {resp.status_code}"}

    with open(audio_path, "wb") as f:
        f.write(resp.content)

    return {"success": True}


# --- Substack Modal functions ---

@app.function(timeout=120)
def process_substack_list(substack_slug: str, episodes_only: bool = False):
    """List posts from a Substack RSS feed."""
    try:
        posts = fetch_substack_rss(substack_slug)
        if episodes_only:
            posts = [p for p in posts if p["has_audio"]]
        return {"success": True, "posts": posts, "count": len(posts)}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}


@app.function(timeout=120)
def process_substack_article(url: str):
    """Fetch article content from a Substack URL."""
    try:
        return fetch_substack_article(url)
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.function(gpu="T4", timeout=600)
def process_substack_podcast(audio_url: str):
    """Download and transcribe a podcast episode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mp3_path = f"{tmpdir}/episode.mp3"
        audio_path = f"{tmpdir}/audio.wav"

        # Download the MP3
        dl = download_audio(audio_url, mp3_path)
        if not dl["success"]:
            return {"success": False, "error": dl["error"]}

        # Get duration
        duration = get_duration(mp3_path)

        # Convert to WAV and transcribe using existing Whisper pipeline
        try:
            transcript = transcribe_audio(mp3_path, audio_path)
        except Exception as e:
            transcript = f"[Transcription failed: {e}]"

        return {
            "success": True,
            "duration_seconds": duration,
            "transcript": transcript,
            "audio_url": audio_url,
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
                    "serverInfo": {"name": "media-fetch", "version": "3.0.0"}
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
                        },
                        {
                            "name": "substack_list_posts",
                            "description": "List recent posts from any Substack publication. Returns titles, dates, summaries, and whether each post has audio (podcast episode). Use the substack slug from the URL (e.g. 'cindieknzz' from cindieknzz.substack.com).",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "substack": {"type": "string", "description": "Substack slug (e.g. 'cindieknzz', 'platformer')"},
                                    "episodes_only": {"type": "boolean", "description": "Only return podcast episodes with audio (default false)", "default": False}
                                },
                                "required": ["substack"]
                            }
                        },
                        {
                            "name": "substack_get_article",
                            "description": "Fetch the full text content of a Substack article or post. Returns title, author, subtitle, and full article text.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string", "description": "Full Substack post URL"}
                                },
                                "required": ["url"]
                            }
                        },
                        {
                            "name": "substack_get_podcast",
                            "description": "Fetch and transcribe a Substack podcast episode using Whisper. Provide either the direct audio URL (from substack_list_posts) or the post URL. Returns full transcript and duration.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "audio_url": {"type": "string", "description": "Direct MP3 audio URL (from substack_list_posts audio_url field)"},
                                    "post_url": {"type": "string", "description": "Substack post URL (will extract audio URL from the page)"}
                                }
                            }
                        }
                    ]
                }
            })

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            url = args.get("url")

            # Video tools require URL
            if tool_name in ("video_listen", "video_see", "watch_video") and not url:
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
            # --- Substack tools ---

            elif tool_name == "substack_list_posts":
                substack = args.get("substack")
                if not substack:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": "Error: No substack slug provided"}]}
                    })

                episodes_only = args.get("episodes_only", False)
                result = process_substack_list.remote(substack, episodes_only)

                if not result.get("success"):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"Error: {result.get('error')}"}]}
                    })

                posts = result.get("posts", [])
                lines = [f"**{substack}.substack.com** — {result.get('count', 0)} posts\n"]
                for i, post in enumerate(posts, 1):
                    audio_badge = " [PODCAST]" if post.get("has_audio") else ""
                    lines.append(f"{i}. **{post['title']}**{audio_badge}")
                    lines.append(f"   {post.get('published', '')}")
                    lines.append(f"   {post.get('url', '')}")
                    if post.get("audio_url"):
                        lines.append(f"   Audio: {post['audio_url']}")
                    lines.append("")

                content = [{"type": "text", "text": "\n".join(lines)}]

            elif tool_name == "substack_get_article":
                if not url:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": "Error: No URL provided"}]}
                    })

                result = process_substack_article.remote(url)

                if not result.get("success"):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"Error: {result.get('error')}"}]}
                    })

                header = f"**{result.get('title', 'Untitled')}**"
                if result.get("subtitle"):
                    header += f"\n*{result['subtitle']}*"
                header += f"\nBy {result.get('author', 'Unknown')}"
                header += f"\n{result.get('url', '')}"
                if result.get("audio_url"):
                    header += f"\nAudio: {result['audio_url']}"
                header += f"\n\n---\n\n{result.get('content', '[No content]')}"

                content = [{"type": "text", "text": header}]

            elif tool_name == "substack_get_podcast":
                audio_url = args.get("audio_url")
                post_url = args.get("post_url")

                # If we have a post URL but no audio URL, fetch the article to find it
                if not audio_url and post_url:
                    article = process_substack_article.remote(post_url)
                    if article.get("success") and article.get("audio_url"):
                        audio_url = article["audio_url"]
                    else:
                        return JSONResponse({
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {"content": [{"type": "text", "text": "Error: Could not find audio URL. Try using substack_list_posts first to get the direct audio URL."}]}
                        })

                if not audio_url:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": "Error: Provide either audio_url or post_url"}]}
                    })

                result = process_substack_podcast.remote(audio_url)

                if not result.get("success"):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"Error: {result.get('error')}"}]}
                    })

                content = [{
                    "type": "text",
                    "text": f"**Podcast Episode**\n**Duration:** {format_duration(result.get('duration_seconds', 0))}\n**Audio:** {audio_url}\n\n**Transcript:**\n{result.get('transcript', '[No transcript]')}"
                }]

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
        return JSONResponse({"status": "ok", "service": "media-fetch-mcp", "version": "3.0.0"})

    routes = [
        Route("/", handle_mcp, methods=["POST"]),
        Route("/sse", handle_sse),
        Route("/health", health),
    ]

    return Starlette(routes=routes)
