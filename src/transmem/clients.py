import os
from urllib.parse import urlparse
import json
import logging
import asyncio
import time
import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from graphiti_core.prompts.models import Message
import openai
from openai.types.chat import ChatCompletionMessageParam
import json_repair
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import ModelSize, LLMConfig
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.extract_edges import Edge as ExtractedEdgeModel
from graphiti_core.prompts.extract_edges import ExtractedEdges as ExtractedEdgesModel
from .runtime_api import apply_completion_limit, is_remote_gpt5_family

class KGClient(OpenAIGenericClient):
    """
    Simplified KG Client.
    Manually controlled via 'disable_thinking' flag to suppress CoT output.
    """
    CONTEXT_PRECLAMP_BUFFER = 1000
    CONTEXT_RETRY_BUFFER = 200

    def __init__(self, config: LLMConfig, disable_thinking: bool=False, **kwargs):
        kwargs.setdefault('max_tokens', config.max_tokens)
        super().__init__(config, **kwargs)
        self.disable_thinking = disable_thinking
        self.model_max_context = self._resolve_model_max_context()
        if self.disable_thinking:
            logger.info('Thinking disabled for construction requests')

    async def summarize(self, prompt: str) -> str:
        """Bound optional summary maintenance independently of graph extraction."""
        kwargs = apply_completion_limit({
            'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.3,
        }, self.config.base_url, 1024)
        if self.disable_thinking and self.config.base_url and urlparse(self.config.base_url).hostname != 'api.openai.com':
            kwargs['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
        started = time.monotonic()
        logger.info('LLM start stage=summary max_tokens=1024')
        response = await asyncio.wait_for(
            self.client.with_options(timeout=90, max_retries=0).chat.completions.create(**kwargs),
            timeout=100,
        )
        logger.info('LLM end stage=summary elapsed=%.2fs finish_reason=%s',
                    time.monotonic()-started, self._response_finish_reason(response))
        result = re.sub('<think>.*?</think>', '', response.choices[0].message.content or '', flags=re.DOTALL).strip()
        if self._response_finish_reason(response) != 'stop' or len(result) < 5:
            raise ValueError('Summary was empty, truncated, or did not finish normally')
        return result

    def _resolve_model_max_context(self) -> int | None:
        for env_name in ('KG_MODEL_MAX_CONTEXT', 'OPENAI_MODEL_MAX_CONTEXT', 'MODEL_MAX_CONTEXT'):
            raw = os.getenv(env_name)
            if raw is None or not raw.strip():
                continue
            try:
                value = int(raw)
            except ValueError:
                logger.warning('[KGClient] Ignoring invalid %s=%r', env_name, raw)
                continue
            if value > 0:
                return value
        # The completion budget is not the model's total context window.
        return None

    def _parse_context_overflow_details(self, err_str: str) -> tuple[int | None, int | None]:
        context_match = re.search('maximum context length is\\s+(\\d+)\\s+tokens', err_str)
        context_limit = int(context_match.group(1)) if context_match else self.model_max_context
        input_candidates: list[int] = []
        for pattern in ('(\\d+)\\s+in the messages', 'prompt contains at least\\s+(\\d+)\\s+input tokens', 'parameter=input_tokens,\\s*value=(\\d+)'):
            input_candidates.extend((int(m) for m in re.findall(pattern, err_str)))
        actual_input_tokens = max(input_candidates) if input_candidates else None
        return (context_limit, actual_input_tokens)

    def _is_context_overflow_error(self, err: Exception | str) -> bool:
        text = str(err).lower()
        return 'maximum context length' in text or 'parameter=input_tokens' in text

    def _head_tail_trim(self, text: str, keep_chars: int) -> str:
        if keep_chars <= 0:
            return ''
        if len(text) <= keep_chars:
            return text
        marker = '\n...[TRUNCATED FOR CONTEXT]...\n'
        if keep_chars <= len(marker) + 16:
            return text[:keep_chars]
        head = max((keep_chars - len(marker)) // 2, 1)
        tail = max(keep_chars - len(marker) - head, 1)
        return text[:head] + marker + text[-tail:]

    def _trim_tagged_block(self, text: str, tag: str, keep_chars: int) -> tuple[str, int]:
        pattern = re.compile(f'<{tag}>\\s*(.*?)\\s*</{tag}>', flags=re.DOTALL)
        match = pattern.search(text)
        if not match:
            return (text, 0)
        original_body = match.group(1)
        trimmed_body = self._head_tail_trim(original_body, keep_chars)
        if trimmed_body == original_body:
            return (text, 0)
        replaced = text[:match.start(1)] + trimmed_body + text[match.end(1):]
        return (replaced, max(0, len(original_body) - len(trimmed_body)))

    def _trim_prompt_text_for_context(self, text: str, chars_to_trim: int) -> str:
        if chars_to_trim <= 0 or not text:
            return text
        updated = text
        remaining = chars_to_trim
        for (tag, min_keep) in (('PREVIOUS MESSAGES', 1200), ('EXISTING ENTITIES', 2000), ('ENTITIES', 1600), ('CURRENT MESSAGE', 4000)):
            if remaining <= 0:
                break
            body_match = re.search(f'<{tag}>\\s*(.*?)\\s*</{tag}>', updated, flags=re.DOTALL)
            if not body_match:
                continue
            body = body_match.group(1)
            keep_chars = max(min_keep, len(body) - remaining)
            (updated, removed) = self._trim_tagged_block(updated, tag, keep_chars)
            remaining = max(0, remaining - removed)
        if remaining > 0:
            updated = self._head_tail_trim(updated, max(1024, len(updated) - remaining))
        return updated

    def _trim_api_kwargs_for_context(self, api_kwargs: Dict[str, Any], context_limit: int, actual_input_tokens: int) -> Dict[str, Any]:
        current_completion = api_kwargs.get('max_tokens') or api_kwargs.get('max_completion_tokens') or self.max_tokens
        target_input_tokens = max(1024, context_limit - int(current_completion) - self.CONTEXT_PRECLAMP_BUFFER)
        excess_tokens = max(0, actual_input_tokens - target_input_tokens)
        chars_to_trim = max(1200, excess_tokens * 6)
        trimmed_messages = []
        for message in api_kwargs.get('messages', []):
            trimmed_messages.append(dict(message))
        user_indices = sorted([idx for (idx, message) in enumerate(trimmed_messages) if message.get('role') == 'user' and isinstance(message.get('content'), str)], key=lambda idx: len(trimmed_messages[idx]['content']), reverse=True)
        for idx in user_indices:
            if chars_to_trim <= 0:
                break
            before = trimmed_messages[idx]['content']
            after = self._trim_prompt_text_for_context(before, chars_to_trim)
            trimmed_messages[idx]['content'] = after
            chars_to_trim = max(0, chars_to_trim - max(0, len(before) - len(after)))
        updated_kwargs = dict(api_kwargs)
        updated_kwargs['messages'] = trimmed_messages
        return updated_kwargs

    def _response_finish_reason(self, response: Any | None) -> str | None:
        if response is None:
            return None
        choices = getattr(response, 'choices', None) or []
        if not choices:
            return None
        return getattr(choices[0], 'finish_reason', None)

    def _log_response_meta(self, response: Any | None, *, prefix: str, level: int=logging.WARNING, model_name: str | None=None) -> None:
        if response is None:
            return
        logger.log(level, '%s model=%s finish_reason=%s', prefix,
                   model_name or self.model, self._response_finish_reason(response))

    def _salvage_extracted_edges_payload(self, payload: Any, *, response: Any | None=None, model_name: str | None=None) -> Dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        raw_edges = payload.get('edges')
        if not isinstance(raw_edges, list):
            return None
        valid_edges: list[dict[str, Any]] = []
        dropped = 0
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                dropped += 1
                continue
            candidate = dict(raw_edge)
            for field_name in ('source_entity_id', 'target_entity_id'):
                value = candidate.get(field_name)
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped.isdigit():
                        candidate[field_name] = int(stripped)
            try:
                validated = ExtractedEdgeModel(**candidate)
            except Exception:
                dropped += 1
                continue
            valid_edges.append(validated.model_dump())
        if not valid_edges:
            return None
        salvaged_payload = {'edges': valid_edges}
        ExtractedEdgesModel(**salvaged_payload)
        logger.warning('[KGClient] Partial salvage for ExtractedEdges succeeded: kept=%d dropped=%d raw=%d', len(valid_edges), dropped, len(raw_edges))
        self._log_response_meta(response, prefix='[KGClient] Salvaged response meta', model_name=model_name)
        return salvaged_payload

    def _validate_or_salvage_payload(self, payload: Any, response_model: type[BaseModel] | None, *, response: Any | None=None, model_name: str | None=None) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError(f'Structured response must be a dict, got {type(payload).__name__}')
        if response_model is None:
            return payload
        try:
            response_model(**payload)
            return payload
        except Exception:
            is_extract_edges = response_model is ExtractedEdgesModel or getattr(response_model, '__name__', '') == ExtractedEdgesModel.__name__
            if is_extract_edges:
                salvaged_payload = self._salvage_extracted_edges_payload(payload, response=response, model_name=model_name)
                if salvaged_payload is not None:
                    return salvaged_payload
            raise

    def _parse_structured_result(self, result: str, response_model: type[BaseModel] | None, *, response: Any | None=None, model_name: str | None=None) -> Dict[str, Any]:
        try:
            parsed_json = json.loads(result)
            return self._validate_or_salvage_payload(parsed_json, response_model, response=response, model_name=model_name)
        except Exception as initial_error:
            if json_repair:
                try:
                    logger.warning('[KGClient] Initial JSON/validation failed. Attempting repair... Error: %s', initial_error)
                    repaired_json = json_repair.repair_json(result, return_objects=True)
                    return self._validate_or_salvage_payload(repaired_json, response_model, response=response, model_name=model_name)
                except Exception as repair_error:
                    logger.error(f'[KGClient] json_repair also failed: {repair_error}')
            raise initial_error

    async def _generate_response(self, messages: list[Message], response_model: type[BaseModel] | None=None, max_tokens: int | None=None, model_size: ModelSize=ModelSize.medium) -> dict[str, Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens
        # Graphiti edge extraction explicitly requests 16K; respect the release cap.
        max_tokens = min(max_tokens, self.max_tokens)
        if self.model_max_context is not None:
            max_tokens = min(max_tokens, max(512, self.model_max_context - self.CONTEXT_PRECLAMP_BUFFER))
        model_name = self._get_model_for_size(model_size)
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            content = m.content.strip()
            if hasattr(self, '_clean_input'):
                content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': content})
        try:
            response_format: dict[str, Any] = {'type': 'json_object'}
            if response_model is not None:
                schema_name = getattr(response_model, '__name__', 'structured_response')
                try:
                    json_schema = response_model.model_json_schema()
                except AttributeError:
                    json_schema = response_model.schema()
                response_format = {'type': 'json_schema', 'json_schema': {'name': schema_name, 'schema': json_schema}}
            api_kwargs = {'model': model_name, 'messages': openai_messages, 'response_format': response_format}
            if not is_remote_gpt5_family(self.config.base_url, model_name):
                api_kwargs['temperature'] = self.temperature
                api_kwargs['top_p'] = 0.8
            api_kwargs = apply_completion_limit(api_kwargs, self.config.base_url, max_tokens)
            if self.config.base_url and urlparse(self.config.base_url).hostname != 'api.openai.com':
                extra_body = {'top_k': 20}
                if self.disable_thinking:
                    extra_body['chat_template_kwargs'] = {'enable_thinking': False}
                api_kwargs['extra_body'] = extra_body
            started = time.monotonic()
            logger.info('LLM start schema=%s max_tokens=%s',
                        getattr(response_model, '__name__', 'json'), max_tokens)
            response = await self.client.chat.completions.create(**api_kwargs)
            logger.info('LLM end schema=%s elapsed=%.2fs finish_reason=%s',
                        getattr(response_model, '__name__', 'json'), time.monotonic()-started,
                        self._response_finish_reason(response))
            result = response.choices[0].message.content or ''
            if '<think>' in result:
                result = re.sub('<think>.*?</think>', '', result, flags=re.DOTALL).strip()
            try:
                return self._parse_structured_result(result, response_model, response=response, model_name=model_name)
            except Exception as e:
                self._log_response_meta(response, prefix='[KGClient] Invalid structured response meta', level=logging.ERROR, model_name=model_name)
                raise e
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except openai.BadRequestError as e:
            err_str = str(e)
            (context_limit, actual_input_tokens) = self._parse_context_overflow_details(err_str)
            if actual_input_tokens is not None and context_limit is not None:
                safe_max = max(512, min(max_tokens, context_limit - actual_input_tokens - self.CONTEXT_PRECLAMP_BUFFER))
                logger.warning('[KGClient] ⚠️ Context overflow detected. context_limit=%s input=%s tokens. Retrying with max_tokens=%s (was %s).', context_limit, actual_input_tokens, safe_max, max_tokens)
                try:
                    api_kwargs.pop('max_tokens', None)
                    api_kwargs.pop('max_completion_tokens', None)
                    api_kwargs = apply_completion_limit(api_kwargs, self.config.base_url, safe_max)
                    response = await self.client.chat.completions.create(**api_kwargs)
                    result = response.choices[0].message.content or ''
                    if '<think>' in result:
                        result = re.sub('<think>.*?</think>', '', result, flags=re.DOTALL).strip()
                    return self._parse_structured_result(result, response_model, response=response, model_name=model_name)
                except openai.BadRequestError as retry_e:
                    (retry_limit, retry_input_tokens) = self._parse_context_overflow_details(str(retry_e))
                    if self._is_context_overflow_error(retry_e) and retry_limit is not None and (retry_input_tokens is not None):
                        logger.warning('[KGClient] Prompt still too large after lowering completion budget. Preclamping prompt content and retrying once more. context_limit=%s input=%s model=%s', retry_limit, retry_input_tokens, model_name)
                        trimmed_kwargs = self._trim_api_kwargs_for_context(api_kwargs, retry_limit, retry_input_tokens)
                        response = await self.client.chat.completions.create(**trimmed_kwargs)
                        result = response.choices[0].message.content or ''
                        if '<think>' in result:
                            result = re.sub('<think>.*?</think>', '', result, flags=re.DOTALL).strip()
                        return self._parse_structured_result(result, response_model, response=response, model_name=model_name)
                    logger.error(f'[KGClient] ❌ Retry also failed: {retry_e}')
                    raise retry_e
                except Exception as retry_e:
                    logger.error(f'[KGClient] ❌ Retry also failed: {retry_e}')
                    raise retry_e
            else:
                logger.error(f'[KGClient] BadRequestError (non-context): {e}')
                raise
        except Exception as e:
            self._log_response_meta(response if 'response' in locals() else None, prefix='[KGClient] Exception response meta', level=logging.ERROR, model_name=model_name)
            logger.exception('Error generating structured response model=%s', model_name)
            raise

class Person(BaseModel):
    pass

class Organization(BaseModel):
    pass

class Location(BaseModel):
    pass

class Concept(BaseModel):
    pass

class Activity(BaseModel):
    pass
GENERIC_SCHEMA = {'Person': Person, 'Organization': Organization, 'Location': Location, 'Concept': Concept, 'Activity': Activity}

class ShiftEvent(BaseModel):
    event_type: str = Field(..., description="Category of the change (e.g., 'Preference Shift', 'Location Change')")
    old_state: str = Field(..., description='Explicit description of the state BEFORE the change')
    new_state: str = Field(..., description='Explicit description of the state AFTER the change')
    trigger: str = Field(..., description='The specific cause or trigger for this change')
    valid_at: str = Field(..., description='The best-guess date when this change became effective (YYYY-MM-DD)')

class ShiftEventList(BaseModel):
    shifts: List[ShiftEvent]
logger = logging.getLogger(__name__)
