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
from app.mimo_local import (
    _maybe_compact_agent_context, _process_host_read, _remember_host_evidence,
    AGENT_CONTEXT_COMPACT_THRESHOLD, CHECKPOINT_READ_EVIDENCE_CHARS,
)


class CheckpointTests(unittest.TestCase):
    def test_multi_call_exchange_and_provider_items_survive_repeated_checkpoints(self):
        evidence = []
        _remember_host_evidence(evidence, 'host_apply_patch',
                                {'path': '/home/share/app.js', 'old_text': 'old', 'new_text': 'new'},
                                '{"ok":true,"replacements":1}', 'completed')
        base = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'fix'}]
        latest = [{'role': 'assistant', 'tool_calls': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}],
                   'responses_output_items': [{'type': 'reasoning', 'id': 'r'}]}]
        latest += [{'role': 'tool', 'tool_call_id': key, 'content': key} for key in ('a', 'b', 'c')]
        history = base + [{'role': 'assistant', 'content': 'x' * (AGENT_CONTEXT_COMPACT_THRESHOLD + 1)}] + copy.deepcopy(latest)
        kwargs = dict(base_message_count=2, workspace=None, sources={}, tool_trace=[], host_evidence=evidence)
        self.assertTrue(_maybe_compact_agent_context(history, **kwargs))
        self.assertEqual(history[2:], latest)
        checkpoint = json.loads(history[0]['content'].split('CONTEXT CHECKPOINT:\n')[1])
        self.assertEqual(checkpoint['host_operation_evidence'][0]['arguments']['new_text'], 'new')
        history[2]['content'] = 'x' * (AGENT_CONTEXT_COMPACT_THRESHOLD + 1)
        self.assertTrue(_maybe_compact_agent_context(history, **kwargs))
        self.assertEqual([m.get('tool_call_id') for m in history[3:]], ['a', 'b', 'c'])
        self.assertEqual(history[2]['responses_output_items'], latest[0]['responses_output_items'])

    def test_host_evidence_is_bounded_and_oversized_content_is_explicit(self):
        evidence = []
        for i in range(40):
            _remember_host_evidence(evidence, 'host_read_file', {'path': str(i)}, 'x' * 3000, 'completed')
        self.assertLessEqual(len(json.dumps(evidence, ensure_ascii=False)), CHECKPOINT_READ_EVIDENCE_CHARS)
        _remember_host_evidence(evidence, 'host_write_file', {'path': 'big', 'content': 'x' * 90000}, '{}', 'completed')
        self.assertIn('evidence_omitted', evidence[-1])

    def test_checkpoint_keeps_recent_small_exchanges(self):
        base = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'fix'}]
        old = [{'role': 'assistant', 'tool_calls': [{'id': 'old'}]},
               {'role': 'tool', 'tool_call_id': 'old', 'content': 'x' * (AGENT_CONTEXT_COMPACT_THRESHOLD + 1)}]
        recent = [{'role': 'assistant', 'tool_calls': [{'id': 'r'}]}, {'role': 'tool', 'tool_call_id': 'r', 'content': 'recent'}]
        latest = [{'role': 'assistant', 'tool_calls': [{'id': 'l'}]}, {'role': 'tool', 'tool_call_id': 'l', 'content': 'latest'}]
        history = base + old + recent + latest
        self.assertTrue(_maybe_compact_agent_context(history, base_message_count=2, workspace=None, sources={}, tool_trace=[]))
        self.assertEqual(history[2:], recent + latest)

    def test_checkpoint_carries_host_read_snapshots_within_budget(self):
        evidence = {
            ('/a', 1, None): {'path': '/a', 'revision': 'r1', 'numbered_content': 'x' * (CHECKPOINT_READ_EVIDENCE_CHARS - 60)},
            ('/b', 1, None): {'path': '/b', 'revision': 'r2', 'numbered_content': '1|keep'},
        }
        history = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'fix'},
                   {'role': 'assistant', 'content': 'x' * (AGENT_CONTEXT_COMPACT_THRESHOLD + 1)}]
        self.assertTrue(_maybe_compact_agent_context(
            history, base_message_count=2, workspace=None, sources={}, tool_trace=[],
            host_evidence=[], host_read_evidence=evidence))
        checkpoint = json.loads(history[0]['content'].split('CONTEXT CHECKPOINT:\n')[1])
        # Newest wins; the oversized older one no longer fits and is forgotten.
        self.assertEqual([item['path'] for item in checkpoint['host_read_snapshots']], ['/b'])
        self.assertEqual(list(evidence), [('/b', 1, None)])


class HostReadTests(unittest.TestCase):
    def test_host_read_is_bounded_and_paginated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'big.txt'
            path.write_text(''.join(f'{"y" * 99}\n' for _ in range(2000)))
            result = json.loads(AgentRuntime(None, 1, 'test').execute('host_read_file', {'path': str(path)}))
            self.assertTrue(result['truncated'])
            self.assertLessEqual(len(result['content']), agent_module.HOST_READ_MAX_CHARS)
            self.assertEqual(result['next_start_line'], result['end_line'] + 1)
            self.assertEqual(result['line_count'], 2000)
            self.assertTrue(result['revision'])

    def test_unchanged_host_reads_are_deduplicated_until_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'app.js'
            path.write_text(''.join(f'line {n}\n' for n in range(1, 101)))
            runtime = AgentRuntime(None, 1, 'test')
            evidence, revisions, counts = {}, {}, {}

            def read(**arguments):
                arguments['path'] = str(path)
                text, record, duplicate = _process_host_read(
                    runtime.execute('host_read_file', arguments), arguments, evidence, revisions, counts)
                self.assertNotIn('line 1', record)
                return json.loads(text), duplicate

            first, duplicate = read()
            self.assertFalse(duplicate)
            self.assertIn('1|line 1', first['content'])
            again, duplicate = read()
            self.assertTrue(duplicate)
            self.assertTrue(again['unchanged'])
            self.assertNotIn('content', again)
            subrange, duplicate = read(start_line=10, end_line=20)
            self.assertTrue(duplicate)
            self.assertIn('warning', subrange)
            path.write_text('changed\n')
            changed, duplicate = read()
            self.assertFalse(duplicate)
            self.assertIn('1|changed', changed['content'])
            self.assertEqual(len(evidence), 1)
            partial, duplicate = read(start_line=1, end_line=1)
            self.assertTrue(duplicate)

    def test_partial_host_read_does_not_hide_other_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'app.js'
            path.write_text(''.join(f'line {n}\n' for n in range(1, 101)))
            runtime = AgentRuntime(None, 1, 'test')
            evidence, revisions, counts = {}, {}, {}
            for start, end, expected_duplicate in ((1, 50, False), (40, 60, False), (45, 55, True), (1, 60, False)):
                arguments = {'path': str(path), 'start_line': start, 'end_line': end}
                _, _, duplicate = _process_host_read(
                    runtime.execute('host_read_file', arguments), arguments, evidence, revisions, counts)
                self.assertEqual(duplicate, expected_duplicate, (start, end))
            # The 1-60 read supersedes both narrower snapshots.
            self.assertEqual(list(evidence), [(str(path), 1, 60)])

    def test_workspace_subrange_is_covered_by_kept_full_read(self):
        evidence = {('app.js', 1, None): {'path': 'app.js', 'revision': 'r', 'line_count': 100,
                                          'returned_from_line': 1, 'returned_through_line': 100,
                                          'numbered_content': '...'}}
        self.assertIsNotNone(mimo_local._find_covering_read(evidence, 'app.js', None, 30, 60))
        self.assertIsNotNone(mimo_local._find_covering_read(evidence, 'app.js', None, 90, None))
        self.assertIsNone(mimo_local._find_covering_read(evidence, 'other.js', None, 30, 60))
        evidence[('app.js', 1, None)]['returned_through_line'] = 50
        self.assertIsNone(mimo_local._find_covering_read(evidence, 'app.js', None, 30, 60))


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
        with patch.object(mimo_local.httpx, 'AsyncClient', Client), patch.object(mimo_local, 'AGENT_CONTEXT_COMPACT_THRESHOLD', 1):
            result = await mimo_local.stream_response(
                base_url='https://example.test/v1', api_key='test', model='test', messages=[{'role': 'user', 'content': 'fix'}],
                timeout=30, stopped=lambda: False, update=update, settings={'thinking': 'disabled'},
                agent_mode=True, web_enabled=False, workspace=None, workspace_access='none',
                extra_tools=runtime.tool_definitions, extra_tool_handler=runtime.execute_async)
        self.assertEqual(result['tool_trace'][0]['status'], 'failed')
        self.assertIn('文件不存在', result['searches'][0]['error'])
        checkpoint = json.loads(requests[1]['messages'][0]['content'].split('CONTEXT CHECKPOINT:\n')[1])
        self.assertEqual(checkpoint['host_operation_evidence'][0]['status'], 'failed')
        self.assertIn('文件不存在', checkpoint['host_operation_evidence'][0]['result'])

    async def test_repeated_host_read_survives_checkpoint_without_rereading(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'mod.json'
            target.write_text('{"spell": "old"}\n')
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
            with patch.object(mimo_local.httpx, 'AsyncClient', Client), patch.object(mimo_local, 'AGENT_CONTEXT_COMPACT_THRESHOLD', 1):
                result = await mimo_local.stream_response(
                    base_url='https://example.test/v1', api_key='test', model='test', messages=[{'role': 'user', 'content': 'fix'}],
                    timeout=30, stopped=lambda: False, update=update, settings={'thinking': 'disabled'},
                    agent_mode=True, web_enabled=False, workspace=None, workspace_access='none',
                    extra_tools=runtime.tool_definitions, extra_tool_handler=runtime.execute_async)
            self.assertEqual(result['answer'], 'done')
            self.assertEqual([item['status'] for item in result['tool_trace']], ['completed', 'skipped'])
            # After the first round was checkpointed, the content is still available.
            checkpoint = json.loads(requests[1]['messages'][0]['content'].split('CONTEXT CHECKPOINT:\n')[1])
            self.assertIn('"spell": "old"', checkpoint['host_read_snapshots'][0]['numbered_content'])
            self.assertNotIn('"spell"', json.dumps(checkpoint['host_operation_evidence'], ensure_ascii=False))
            second_result = json.loads(requests[2]['messages'][-1]['content'])
            self.assertTrue(second_result['unchanged'])


if __name__ == '__main__':
    unittest.main()
