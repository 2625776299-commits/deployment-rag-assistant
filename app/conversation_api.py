"""Authenticated HTTP adapter for the conversation service; dependencies explicit."""
import logging
from hmac import compare_digest
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
import psycopg

from app.conversation_service import public_turn
from app.provider_errors import ModelProviderError, provider_public_reply


logger = logging.getLogger(__name__)


def bearer_authenticator(tokens):
    """Local/demo token -> server-owned identity mapping; tokens stay off the wire logs."""
    if (not isinstance(tokens, dict) or not tokens
            or any(not isinstance(k, str) or not k.isascii() or not k.strip()
                   or not isinstance(v, str) or not v.strip() for k, v in tokens.items())):
        raise ValueError('require_server_token_identity_mapping')
    identities = tuple(tokens.items())
    def authenticate(request):
        scheme, _, supplied = request.headers.get('Authorization', '').partition(' ')
        if scheme.lower() == 'bearer' and supplied.isascii():
            for token, owner_id in identities:
                if compare_digest(token, supplied):
                    return owner_id
        raise HTTPException(status_code=401, detail='请提供有效的登录凭证。')
    return authenticate


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    message_id: str = Field(min_length=1)
    content: str = Field(min_length=1)

    @field_validator('message_id', 'content')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('message_fields_must_not_be_blank')
        # Preserve canonical raw text, including YAML/code indentation.
        return value


def install_conversation_routes(app, service, authenticate):
    if not callable(authenticate):
        raise ValueError('conversation_authentication_must_be_callable')

    def invoke(request, operation):
        request_id = uuid4().hex
        headers = {'X-Request-ID': request_id}
        try:
            owner_id = authenticate(request)
            if not isinstance(owner_id, str) or not owner_id.strip():
                raise HTTPException(status_code=401, detail='请先登录。')
            status, body = operation(owner_id)
        except HTTPException as error:
            status, body = error.status_code, {'status': 'failed', 'message': str(error.detail)}
        except ModelProviderError as error:
            status, body = 503, provider_public_reply(error.detail('prepare'))
            logger.warning('conversation_provider_unavailable request_id=%s code=%s', request_id, error.code)
        except ValueError as error:
            code = str(error)
            status = 404 if code in ('conversation_not_found', 'message_not_found') else 409 if code == 'message_id_conflict' else 500
            body = {'status': 'failed', 'message': ('未找到可访问的会话或消息。' if status == 404
                    else '同一个消息标识不能用于不同内容。' if status == 409 else '本轮处理未能完成，请提供请求编号以便排查。')}
            if status == 500:
                logger.warning('conversation_invalid_result request_id=%s code=%s', request_id, code)
        except (TimeoutError, psycopg.OperationalError):
            status, body = 503, {'status': 'failed', 'message': '服务暂时未能完成处理，消息可按原标识重试。'}
            logger.warning('conversation_unavailable request_id=%s', request_id)
        except Exception:
            status, body = 500, {'status': 'failed', 'message': '服务处理异常，请提供请求编号以便排查。'}
            logger.error('conversation_unexpected request_id=%s', request_id)
        return JSONResponse(status_code=status, content={**body, 'request_id': request_id}, headers=headers)

    @app.post('/conversations')
    def create_conversation(request: Request):
        return invoke(request, lambda owner: (201, {'conversation_id': service.store.create(owner)}))

    @app.post('/conversations/{conversation_id}/messages')
    def send_message(conversation_id: str, message: ConversationMessage, request: Request):
        def submit(owner):
            record, replayed = service.submit(owner, conversation_id, message.message_id, message.content)
            return public_turn(record, replayed=replayed)
        return invoke(request, submit)

    @app.get('/conversations/{conversation_id}/messages/{message_id}')
    def read_message(conversation_id: str, message_id: str, request: Request):
        return invoke(request, lambda owner: public_turn(
            service.store.read_message_turn(owner, conversation_id, message_id), historical=True))

    @app.get('/conversations/{conversation_id}/messages')
    def read_messages(conversation_id: str, request: Request):
        return invoke(request, lambda owner: (200, {'conversation_id': conversation_id, 'turns': [
            {'user_message': r['content'], **public_turn(r, historical=True)[1]}
            for r in service.history(owner, conversation_id)]}))
