'use strict';

function fxbxHttpError(response, data) {
  const detail = data?.error;
  const error = new Error(detail?.message || '서버 요청 실패 (HTTP '+response.status+')');
  error.detail = detail;
  return error;
}

function renderDiagnostic(panel, detail) {
  panel.replaceChildren();
  panel.hidden = !detail?.message;
  if (panel.hidden) return;
  const stage = {download:'파일 준비', load:'모델 적재', unload:'모델 해제',
    inference:'실행', runtime:'작업자 종료'}[detail.stage];
  const reason = document.createElement('strong');
  reason.textContent = (stage ? stage+' 실패 · ' : '')+detail.message;
  panel.append(reason);
  if (detail.hint) {
    const hint = document.createElement('p');
    hint.textContent = detail.hint;
    panel.append(hint);
  }
  if (/^https:\/\/huggingface\.co\/[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(detail.url || '')) {
    const link = document.createElement('a');
    link.href = detail.url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = (detail.repo || 'Hugging Face 모델')+' 열기 ↗';
    panel.append(link);
  }
  if (detail.command) {
    const command = document.createElement('pre');
    command.textContent = detail.command;
    panel.append(command);
  }
  if (detail.log) {
    const details = document.createElement('details');
    const title = document.createElement('summary');
    title.textContent = 'NPU 서버 로그 위치';
    const path = document.createElement('pre');
    path.textContent = detail.log;
    details.append(title, path);
    panel.append(details);
  }
}
