from __future__ import annotations

import asyncio
import copy
import json
import shlex
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import agent as agent_module
from app.agent import AgentRuntime
from app import mimo_local
from app.file_knowledge import FileKnowledge
from app.context import checkpoint_payload


class HostReadTests(unittest.TestCase):
    def read(self, runtime, knowledge, path, **arguments):
        text = runtime.execute('host_read_file', {'path': str(path), **arguments})
        replacement = knowledge.record_read(json.loads(text))
        return json.loads(replacement or text), replacement is not None

    def test_host_read_returns_whole_file_and_splits_only_when_too_large(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = AgentRuntime(None, 1, 'test')
            small = Path(directory) / 'small.js'
            small.write_text(''.join(f'line {n}\n' for n in range(1, 101)))
            result = json.loads(runtime.execute('host_read_file', {'path': str(small), 'start_line': 40, 'end_line': 60}))
            self.assertEqual((result['from_line'], result['through_line'], result['truncated']), (1, 100, False))
            self.assertIn('1|line 1\n', result['content'])
            big = Path(directory) / 'big.txt'
            big.write_text(''.join(f'{"y" * 99}\n' for _ in range(2000)))
            result = json.loads(runtime.execute('host_read_file', {'path': str(big)}))
            self.assertTrue(result['truncated'])
            self.assertLessEqual(len(result['content']), agent_module.HOST_READ_MAX_CHARS)
            self.assertEqual(result['next_start_line'], result['through_line'] + 1)
            self.assertEqual(result['line_count'], 2000)
            self.assertTrue(result['revision'])
            schema = next(item for item in runtime.tool_definitions if item['function']['name'] == 'host_read_file')
            self.assertEqual(set(schema['function']['parameters']['properties']), {'path', 'start_line'})

    def test_unchanged_host_reads_are_short_until_the_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'app.js'
            path.write_text(''.join(f'line {n}\n' for n in range(1, 101)))
            runtime, knowledge = AgentRuntime(None, 1, 'test'), FileKnowledge()
            first, duplicate = self.read(runtime, knowledge, path)
            self.assertFalse(duplicate)
            self.assertIn('1|line 1', first['content'])
            again, duplicate = self.read(runtime, knowledge, path)
            self.assertTrue(duplicate)
            self.assertTrue(again['unchanged'])
            self.assertNotIn('content', again)
            # Asking again right away means the model lost track: give it the file.
            third, duplicate = self.read(runtime, knowledge, path)
            self.assertFalse(duplicate)
            self.assertIn('1|line 1', third['content'])
            fourth, duplicate = self.read(runtime, knowledge, path)
            self.assertTrue(duplicate)
            self.assertIn('warning', fourth)
            path.write_text('changed\n')
            changed, duplicate = self.read(runtime, knowledge, path)
            self.assertFalse(duplicate)
            self.assertIn('1|changed', changed['content'])

    def test_own_host_edit_keeps_the_file_known(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mod.json'
            path.write_text('{\r\n  "spell": "old"\r\n}\r\n')
            runtime, knowledge = AgentRuntime(None, 1, 'test'), FileKnowledge(validate=mimo_local._host_revision)
            self.read(runtime, knowledge, path)
            result = json.loads(runtime.execute('host_edit_file', {'path': str(path), 'edits': [
                {'old_text': '"spell": "old"', 'new_text': '"spell": "new"'}]}))
            self.assertTrue(result['ok'], result)
            self.assertEqual(path.read_bytes(), b'{\r\n  "spell": "new"\r\n}\r\n')
            knowledge.record_own_change(mimo_local._host_read_snapshot(str(path)), visible=True)
            after, duplicate = self.read(runtime, knowledge, path)
            self.assertTrue(duplicate)
            self.assertIn('updated_excerpt', after['message'])
            # A change made outside the model's own edits is detected at checkpoint time.
            path.write_text('other\n')
            self.assertEqual(knowledge.snapshots(10_000), [])
            self.assertFalse(knowledge.known(str(path)))

    def test_host_apply_patch_alias_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'a.txt'
            path.write_text('one\ntwo\n')
            runtime = AgentRuntime(None, 1, 'test')
            alias = json.loads(runtime.execute('host_apply_patch', {'path': str(path), 'old_text': 'one', 'new_text': 'ONE'}))
            self.assertTrue(alias['ok'], alias)
            failed = json.loads(runtime.execute('host_edit_file', {'path': str(path), 'edits': [
                {'old_text': 'two', 'new_text': 'TWO'}, {'old_text': 'missing', 'new_text': 'x'}]}))
            self.assertFalse(failed['ok'])
            self.assertIn('整个批次未修改', failed['error'])
            self.assertEqual(path.read_text(), 'ONE\ntwo\n')


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_kills_shell_children_before_returning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, late = root / 'ready', root / 'late'
            runtime = AgentRuntime(None, 1, 'test')
            command = f"touch {shlex.quote(str(ready))}; (sleep 1; touch {shlex.quote(str(late))}) & wait"
            state = {'id': 'job', 'status': 'running', 'stop_requested': 0}
            async def job():
                try:
                    await runtime.execute_async('host_run_command', {'command': command, 'cwd': directory})
                except asyncio.CancelledError:
                    state['status'] = 'stopped'
                    raise
            task = asyncio.create_task(job())
            try:
                for _ in range(200):
                    if ready.exists(): break
                    await asyncio.sleep(.01)
                self.assertTrue(ready.exists())
                from app import main
                fake_db = SimpleNamespace(one=lambda *_: dict(state), update_job=lambda _, **values: state.update(values))
                with patch.object(main, 'db', fake_db), patch.object(main, 'tasks', {'job': task}):
                    await main.stop_job('job', {'id': 1})
                    self.assertEqual(state['status'], 'running')
                    self.assertEqual(state['stop_requested'], 1)
                    await main.stop_job('job', {'id': 1})
            finally:
                if not state['stop_requested']:
                    task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)
            self.assertEqual(state['status'], 'stopped')
            await asyncio.sleep(1.2)
            self.assertFalse(late.exists())
            self.assertTrue(json.loads(runtime.execute('host_write_file', {'path': str(late), 'content': 'bad'}))['cancelled'])
            self.assertFalse(late.exists())

    async def test_command_timeout_kills_child(self):
        with tempfile.TemporaryDirectory() as directory:
            late = Path(directory) / 'late'
            runtime = AgentRuntime(None, 1, 'test')
            result = json.loads(await runtime.execute_async('host_run_command', {
                'command': f'(sleep 2; touch {shlex.quote(str(late))}) & wait', 'cwd': directory, 'timeout_seconds': 1}))
            self.assertTrue(result['timeout'])
            await asyncio.sleep(1.2)
            self.assertFalse(late.exists())

    @unittest.skipUnless(shutil.which('node'), 'Node is required')
    async def test_html_checks_inline_handlers_and_local_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / 'index.html'
            runtime = AgentRuntime(None, 1, 'test')
            for html in ('<html><script>const broken = ;</script></html>',
                         '<html><button onclick="const x = ;">x</button></html>',
                         '<html><script src="app.js"></script></html>'):
                page.write_text(html)
                (root / 'app.js').write_text('const broken = ;')
                result = json.loads(await runtime.execute_async('frontend_validate_page', {'path': str(page)}))
                self.assertFalse(result['ok'], result)
                self.assertTrue(result['errors'])
            page.write_text('<html><button onclick="return false">x</button><script type="module">export const x = 1;</script><script src="app.js?v=1"></script></html>')
            (root / 'app.js').write_text('const x = 1;')
            result = json.loads(await runtime.execute_async('frontend_validate_page', {'path': str(page)}))
            self.assertTrue(result['ok'], result)
            self.assertEqual(len(result['checked_scripts']), 3)

    async def test_failed_host_tool_is_failed_in_live_trace_and_checkpoint(self):
        requests = []
        call = {'index': 0, 'id': 'p', 'type': 'function', 'function': {
            'name': 'host_apply_patch', 'arguments': json.dumps({'path': '/nonexistent-reliability-test', 'old_text': 'x', 'new_text': 'y'})}}
        rounds = [{'choices': [{'delta': {'tool_calls': [call]}}]}, {'choices': [{'delta': {'content': 'failed as expected'}}]}]
        class Response:
            status_code = 200
            def __init__(self, event): self.event = event
            async def aiter_lines(self):
                yield 'data: ' + json.dumps(self.event)
                yield 'data: [DONE]'
            async def aread(self): return b''
            async def __aenter__(self): return self
            async def __aexit__(self, *_): return False
        class Client:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *_): return False
            def stream(self, *args, **kwargs):
                requests.append(copy.deepcopy(kwargs.get('json')))
                return Response(rounds.pop(0))
        async def update(state): pass
        runtime = AgentRuntime(None, 1, 'test')
        with patch.object(mimo_local.httpx, 'AsyncClient', Client):
            result = await mimo_local.stream_response(
                base_url='https://example.test/v1', api_key='test', model='test', messages=[{'role': 'user', 'content': 'fix'}],
                timeout=30, stopped=lambda: False, update=update, settings={'thinking': 'disabled', 'context_budget_chars': 40_000},
                agent_mode=True, web_enabled=False, workspace=None, workspace_access='none',
                extra_tools=runtime.tool_definitions, extra_tool_handler=runtime.execute_async)
        self.assertEqual(result['tool_trace'][0]['status'], 'failed')
        self.assertIn('文件不存在', result['searches'][0]['error'])
        # The failed result itself is what the model sees next round.
        self.assertIn('文件不存在', requests[1]['messages'][-1]['content'])

    async def test_repeated_host_read_survives_checkpoint_without_rereading(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'mod.json'
            # ~10K chars numbered: fits the snapshot share of a 40K budget.
            target.write_text('{"spell": "old"}\n' + '// padding\n' * 700)
            requests = []
            def read_call(call_id):
                return {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': call_id, 'type': 'function', 'function': {
                    'name': 'host_read_file', 'arguments': json.dumps({'path': str(target)})}}]}}]}
            rounds = [read_call('r1'), read_call('r2'), {'choices': [{'delta': {'content': 'done'}}]}]
            class Response:
                status_code = 200
                def __init__(self, event): self.event = event
                async def aiter_lines(self):
                    yield 'data: ' + json.dumps(self.event)
                    yield 'data: [DONE]'
                async def aread(self): return b''
                async def __aenter__(self): return self
                async def __aexit__(self, *_): return False
            class Client:
                def __init__(self, **kwargs): pass
                async def __aenter__(self): return self
                async def __aexit__(self, *_): return False
                def stream(self, *args, **kwargs):
                    requests.append(copy.deepcopy(kwargs.get('json')))
                    return Response(rounds.pop(0))
            async def update(state): pass
            runtime = AgentRuntime(None, 1, 'test')
            # A large prompt plus the read pushes the request over the smallest
            # budget, so the first round is compacted before the second request.
            with patch.object(mimo_local.httpx, 'AsyncClient', Client):
                result = await mimo_local.stream_response(
                    base_url='https://example.test/v1', api_key='test', model='test', messages=[{'role': 'user', 'content': 'fix'}],
                    timeout=30, stopped=lambda: False, update=update, settings={'thinking': 'disabled', 'context_budget_chars': 40_000},
                    system_addendum='p' * 33_000,
                    agent_mode=True, web_enabled=False, workspace=None, workspace_access='none',
                    extra_tools=runtime.tool_definitions, extra_tool_handler=runtime.execute_async)
            self.assertEqual(result['answer'], 'done')
            self.assertEqual([item['status'] for item in result['tool_trace']], ['completed', 'skipped'])
            # After the first round was compacted, the content lives in the checkpoint.
            checkpoint = checkpoint_payload(requests[1]['messages'])
            self.assertIn('"spell": "old"', checkpoint['file_snapshots'][0]['content'])
            self.assertTrue(result['round_stats'][0]['compacted_after'])
            second_result = json.loads(requests[2]['messages'][-1]['content'])
            self.assertTrue(second_result['unchanged'])


if __name__ == '__main__':
    unittest.main()
