"""Low-memory attachment processing for the chat API.

Raw uploads are streamed to disk and removed immediately after conversion.
Images are resized JPEGs; documents are bounded plain-text extracts.
"""
from __future__ import annotations

import base64
import ctypes
import gc
import io
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import settings


MAX_ATTACHMENTS = 10
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_FILE_TEXT_CHARS = 30_000
MAX_TOTAL_TEXT_CHARS = 80_000
MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_SIDE = 1600
MAX_PROCESSED_IMAGE_BYTES = 1_500_000
ATTACHMENT_TTL_SECONDS = 24 * 60 * 60
ATTACHMENTS_DIR = settings.data_dir / "attachments"

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml",
    ".yaml", ".yml", ".log", ".ini", ".cfg", ".conf", ".py", ".js", ".ts",
    ".jsx", ".tsx", ".html", ".htm", ".css", ".scss", ".sql", ".sh", ".bash",
    ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb",
    ".php", ".swift", ".vue", ".svelte",
}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


class AttachmentError(ValueError):
    pass


def _release_large_buffers() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def safe_filename(value: str) -> str:
    name = Path(str(value or "attachment")).name.replace("\x00", "")
    name = re.sub(r"[\r\n\t]+", " ", name).strip()
    return name[:180] or "attachment"


def attachment_path(user_id: int, attachment_id: str, suffix: str) -> Path:
    directory = ATTACHMENTS_DIR / str(int(user_id))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory / f"{attachment_id}{suffix}"


def _check_zip(archive: zipfile.ZipFile) -> None:
    total = 0
    for item in archive.infolist():
        total += max(0, item.file_size)
        if item.file_size > 50 * 1024 * 1024 or total > 150 * 1024 * 1024:
            raise AttachmentError("Office 文件解压后过大，已拒绝处理")


def _iter_xml_text(stream: io.BufferedReader, tag_suffix: str) -> Iterable[str]:
    try:
        for _, element in ElementTree.iterparse(stream, events=("end",)):
            if element.tag.endswith(tag_suffix) and element.text:
                yield element.text
            element.clear()
    except ElementTree.ParseError as exc:
        raise AttachmentError("文档 XML 已损坏") from exc


def _extract_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        _check_zip(archive)
        try:
            stream = archive.open("word/document.xml")
        except KeyError as exc:
            raise AttachmentError("不是有效的 Word .docx 文件") from exc
        parts: list[str] = []
        used = 0
        with stream:
            for value in _iter_xml_text(stream, "}t"):
                value = value.strip()
                if not value:
                    continue
                remaining = MAX_FILE_TEXT_CHARS - used
                if remaining <= 0:
                    break
                parts.append(value[:remaining])
                used += min(len(value), remaining) + 1
        return "\n".join(parts)


def _extract_xlsx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        _check_zip(archive)
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            with archive.open("xl/sharedStrings.xml") as stream:
                current: list[str] = []
                shared_chars = 0
                for event, element in ElementTree.iterparse(stream, events=("start", "end")):
                    if event == "end" and element.tag.endswith("}t") and element.text:
                        remaining = 2_000_000 - shared_chars
                        if remaining > 0:
                            value = element.text[:remaining]
                            current.append(value)
                            shared_chars += len(value)
                    elif event == "end" and element.tag.endswith("}si"):
                        shared.append("".join(current))
                        current = []
                        if len(shared) >= 20_000 or shared_chars >= 2_000_000:
                            break
                    if event == "end":
                        element.clear()

        sheets = sorted(
            name for name in archive.namelist()
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        )[:5]
        lines: list[str] = []
        used = 0
        for sheet_index, sheet_name in enumerate(sheets, start=1):
            if used >= MAX_FILE_TEXT_CHARS:
                break
            lines.append(f"[工作表 {sheet_index}]")
            with archive.open(sheet_name) as stream:
                row_values: list[str] = []
                cell_type = ""
                cell_value = ""
                rows_seen = 0
                for event, element in ElementTree.iterparse(stream, events=("start", "end")):
                    suffix = element.tag.rsplit("}", 1)[-1]
                    if event == "start" and suffix == "c":
                        cell_type = element.attrib.get("t", "")
                        cell_value = ""
                    elif event == "end" and suffix in ("v", "t") and element.text:
                        cell_value += element.text
                    elif event == "end" and suffix == "c":
                        value = cell_value
                        if cell_type == "s":
                            try:
                                value = shared[int(cell_value)]
                            except (ValueError, IndexError):
                                pass
                        row_values.append(value)
                    elif event == "end" and suffix == "row":
                        line = "\t".join(row_values).rstrip()
                        row_values = []
                        rows_seen += 1
                        if line:
                            remaining = MAX_FILE_TEXT_CHARS - used
                            if remaining <= 0:
                                break
                            lines.append(line[:remaining])
                            used += min(len(line), remaining) + 1
                        if rows_seen >= 2000:
                            break
                    if event == "end":
                        element.clear()
        return "\n".join(lines)


def _extract_pdf(path: Path) -> str:
    executable = shutil.which("pdftotext")
    limiter = shutil.which("prlimit")
    if not executable or not limiter:
        raise AttachmentError("服务器缺少 PDF 文本提取组件，请重新运行安装脚本")
    output = path.with_suffix(".extracted.txt")
    try:
        completed = subprocess.run(
            [
                limiter, "--as=100663296", "--cpu=35", "--fsize=524288", "--",
                executable, "-f", "1", "-l", "30", "-layout", "-enc", "UTF-8",
                str(path), str(output),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )
        text = output.read_text(encoding="utf-8", errors="replace") if output.is_file() else ""
        if not text.strip():
            message = completed.stderr.decode("utf-8", errors="replace").strip().lower()
            if "password" in message or "encrypted" in message:
                raise AttachmentError("PDF 已加密，请先解除密码")
            if completed.returncode:
                raise AttachmentError("PDF 解析失败或超过了这台服务器的安全资源上限")
            raise AttachmentError("PDF 没有可提取文字；扫描版 PDF 暂不支持 OCR")
        return text.strip()[:MAX_FILE_TEXT_CHARS]
    except subprocess.TimeoutExpired as exc:
        raise AttachmentError("PDF 处理超过 45 秒，已停止") from exc
    finally:
        output.unlink(missing_ok=True)


def _extract_plain_text(path: Path) -> str:
    data = bytearray()
    maximum = MAX_FILE_TEXT_CHARS * 4
    with path.open("rb") as stream:
        while len(data) < maximum:
            chunk = stream.read(min(64 * 1024, maximum - len(data)))
            if not chunk:
                break
            data.extend(chunk)
    if b"\x00" in data[:4096]:
        raise AttachmentError("文件不是可识别的纯文本格式")
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            return bytes(data).decode(encoding)[:MAX_FILE_TEXT_CHARS]
        except UnicodeDecodeError:
            continue
    return bytes(data).decode("utf-8", errors="replace")[:MAX_FILE_TEXT_CHARS]


def _process_image(source: Path, destination: Path) -> dict[str, Any]:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(source) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise AttachmentError("图片像素过大；这台服务器最多处理约 1600 万像素的图片")
            if image.format == "JPEG":
                image.draft("RGB", (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            image.seek(0)
            try:
                orientation = int(image.getexif().get(274, 1))
            except (TypeError, ValueError):
                orientation = 1
            if orientation not in (0, 1):
                image = ImageOps.exif_transpose(image)
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            if image.mode != "RGB":
                background = Image.new("RGB", image.size, "white")
                if "A" in image.getbands():
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image.convert("RGB"))
                image = background
            quality = 84
            while True:
                image.save(destination, format="JPEG", quality=quality, optimize=True, progressive=True)
                if destination.stat().st_size <= MAX_PROCESSED_IMAGE_BYTES:
                    break
                if quality > 62:
                    quality -= 10
                    continue
                new_size = (max(320, int(image.width * 0.82)), max(320, int(image.height * 0.82)))
                if new_size == image.size:
                    raise AttachmentError("图片压缩后仍然过大")
                image.thumbnail(new_size, Image.Resampling.LANCZOS)
        os.chmod(destination, 0o600)
    except AttachmentError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise AttachmentError("图片格式无效或图片已损坏") from exc
    return {
        "kind": "image",
        "media_type": "image/jpeg",
        "stored_path": str(destination),
        "processed_size": destination.stat().st_size,
    }


def process_upload(source: Path, user_id: int, attachment_id: str, filename: str, media_type: str) -> dict[str, Any]:
    """Convert an upload and always remove the raw input file."""
    name = safe_filename(filename)
    extension = Path(name).suffix.lower()
    destination: Path | None = None
    try:
        if extension in _IMAGE_EXTENSIONS or media_type.startswith("image/"):
            destination = attachment_path(user_id, attachment_id, ".jpg")
            return _process_image(source, destination)
        if extension == ".pdf" or media_type == "application/pdf":
            text = _extract_pdf(source)
        elif extension == ".docx":
            text = _extract_docx(source)
        elif extension == ".xlsx":
            text = _extract_xlsx(source)
        elif extension in _TEXT_EXTENSIONS or media_type.startswith("text/"):
            text = _extract_plain_text(source)
        elif extension in (".doc", ".xls"):
            raise AttachmentError("旧版 .doc/.xls 暂不支持，请另存为 .docx/.xlsx")
        else:
            raise AttachmentError("不支持此文件格式；支持图片、PDF、DOCX、XLSX 和常见文本/代码文件")
        text = text.strip()
        if not text:
            raise AttachmentError("文件中没有可读取的文字")
        destination = attachment_path(user_id, attachment_id, ".txt")
        destination.write_text(text[:MAX_FILE_TEXT_CHARS], encoding="utf-8")
        os.chmod(destination, 0o600)
        return {
            "kind": "document",
            "media_type": "text/plain",
            "stored_path": str(destination),
            "processed_size": destination.stat().st_size,
        }
    except (zipfile.BadZipFile, KeyError) as exc:
        raise AttachmentError("Office 文件无效或已损坏") from exc
    except Exception:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise
    finally:
        source.unlink(missing_ok=True)
        _release_large_buffers()


def build_model_messages(messages: list[dict[str, Any]], records: list[dict[str, Any]], allow_images: bool) -> list[dict[str, Any]]:
    """Add bounded extracts and OpenAI-compatible image_url parts to the last user turn."""
    output = [dict(item) for item in messages]
    user_index = next((index for index in range(len(output) - 1, -1, -1) if output[index].get("role") == "user"), -1)
    if user_index < 0:
        raise AttachmentError("附件必须随用户消息一起发送")
    original = str(output[user_index].get("content") or "")
    document_parts: list[str] = []
    images: list[dict[str, Any]] = []
    text_used = 0
    image_bytes = 0
    for record in records:
        path = Path(record["stored_path"])
        if not path.is_file():
            raise AttachmentError(f"附件已过期或丢失：{record['original_name']}")
        if record["kind"] == "document":
            remaining = MAX_TOTAL_TEXT_CHARS - text_used
            if remaining <= 0:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")[:remaining]
            text_used += len(text)
            document_parts.append(f"\n\n--- 附件：{record['original_name']}（提取文本）---\n{text}")
        elif record["kind"] == "image":
            if not allow_images:
                raise AttachmentError("当前 DeepSeek Responses 路由不接受图片，请改用支持视觉输入的 Custom 模型")
            raw = path.read_bytes()
            image_bytes += len(raw)
            if image_bytes > MAX_PROCESSED_IMAGE_BYTES * MAX_ATTACHMENTS:
                raise AttachmentError("处理后的图片总量过大")
            images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{record['media_type']};base64,{base64.b64encode(raw).decode('ascii')}",
                    "detail": "auto",
                },
            })
    text_content = original + "".join(document_parts)
    output[user_index]["content"] = ([{"type": "text", "text": text_content or "请分析这些图片。"}, *images] if images else text_content)
    return output


def delete_files(records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        try:
            Path(str(record.get("stored_path") or "")).unlink(missing_ok=True)
        except OSError:
            pass
    _release_large_buffers()


def cleanup_orphan_files(active_paths: set[str], max_age_seconds: int = ATTACHMENT_TTL_SECONDS) -> None:
    if not ATTACHMENTS_DIR.is_dir():
        return
    cutoff = time.time() - max(3600, int(max_age_seconds))
    for path in ATTACHMENTS_DIR.glob("*/*"):
        try:
            if path.is_file() and str(path) not in active_paths and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            pass
