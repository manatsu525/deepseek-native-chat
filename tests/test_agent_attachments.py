import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from starlette.requests import Request
from app import attachments, main, workspace
from app.db import Database


class AgentAttachmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_original_publish_and_retry_without_overwriting_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = Database(root / 'test.db')
            database.init()
            user_id = database.run('INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)', ('test', 'hash', 1))
            raw = b'PK\x03\x04\x00binary archive payload\xff'
            async def receive():
                return {'type': 'http.request', 'body': raw, 'more_body': False}
            request = Request({'type': 'http', 'headers': [(b'content-type', b'application/zip')]}, receive)
            with patch.object(main, 'db', database), patch.object(main, 'attachment_upload_locks', {}), patch.object(attachments, 'ATTACHMENTS_DIR', root / 'attachments'), patch.object(workspace, 'AGENT_WORKSPACE_ROOT', root / 'share'):
                public = await main.upload_attachment(request, 'a' * 32, filename='project.zip', chat_mode='agent', user={'id': user_id})
                self.assertEqual(public['kind'], 'agent_file')
                record = database.one('SELECT * FROM attachments WHERE id=?', (public['id'],))
                self.assertEqual(Path(record['stored_path']).read_bytes(), raw)
                target = attachments.agent_attachment_path(record)
                self.assertFalse(target.exists())
                messages = attachments.build_agent_messages([{'role': 'user', 'content': 'unzip'}], [record])
                self.assertEqual(target.read_bytes(), raw)
                self.assertIn(str(target), messages[0]['content'])
                target.write_bytes(b'user edited')
                attachments.build_agent_messages([{'role': 'user', 'content': 'retry'}], [record])
                self.assertEqual(target.read_bytes(), b'user edited')
                with self.assertRaises(attachments.AttachmentError):
                    attachments.build_model_messages([{'role': 'user', 'content': 'read'}], [record], True)

    async def test_any_extension_and_empty_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(attachments, 'ATTACHMENTS_DIR', Path(directory) / 'data'):
            for name, content in [('data.bin', b'\x00\xff'), ('old.doc', b'old binary'), ('audio.mp3', b'audio'), ('no_extension', b''), ('main.rs', b'a' * 100000)]:
                source = Path(directory) / 'incoming'
                source.write_bytes(content)
                result = attachments.process_upload(source, 1, name.replace('.', '_'), name, 'application/octet-stream', agent_mode=True)
                self.assertEqual(Path(result['stored_path']).read_bytes(), content)
                self.assertFalse(source.exists())

    async def test_standard_still_rejects_archives_and_extracts_text(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(attachments, 'ATTACHMENTS_DIR', Path(directory) / 'data'):
            source = Path(directory) / 'incoming'
            source.write_bytes(b'zip bytes')
            with self.assertRaises(attachments.AttachmentError):
                attachments.process_upload(source, 1, 'id', 'project.zip', 'application/zip')
            source.write_text('hello')
            result = attachments.process_upload(source, 1, 'text', 'readme.txt', 'text/plain')
            self.assertEqual(result['kind'], 'document')

    async def test_image_has_preview_and_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            Image.new('RGBA', (2, 2), (0, 0, 255, 100)).save(source)
            raw = source.read_bytes()
            with patch.object(attachments, 'ATTACHMENTS_DIR', root / 'data'), patch.object(workspace, 'AGENT_WORKSPACE_ROOT', root / 'share'):
                result = attachments.process_upload(source, 1, 'image', 'source.png', 'image/png', agent_mode=True)
                record = {**result, 'id': 'image', 'original_name': 'source.png'}
                messages = attachments.build_agent_messages([{'role': 'user', 'content': 'look'}], [record])
                self.assertEqual(attachments.agent_attachment_path(record).read_bytes(), raw)
                self.assertEqual(messages[0]['content'][1]['type'], 'image_url')


if __name__ == '__main__':
    unittest.main()
