"""投稿流程中的大模型节点及其 Pydantic 返回结构。"""

import json
from langchain_core.tools.base import BaseTool
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from playwright.sync_api import Page, Error as PlaywrightError

from tools.read_configs_login import (
    LoginSession,
    create_read_configs_login_tool,
    create_click_submit_entry_tool,
)


def create_chat_model() -> ChatOpenAI:
    """从环境变量创建 OpenAI 或 DashScope 兼容聊天模型。"""
    load_dotenv()
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    api_key = dashscope_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请在 .env 中配置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY。")

    if dashscope_key:
        return ChatOpenAI(
            model=os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
            api_key=dashscope_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7,
        )
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.7,
    )

class StepRequiredFields(BaseModel):
    """描述投稿流程当前步骤和其中的必填字段。"""

    step: str = Field(description="投稿步骤名称")
    required_fields: list[str] = Field(description="该步骤中的必填字段列表")


class SubmissionSchema(BaseModel):
    """以结构化格式返回投稿流程步骤及必填字段。"""

    steps: list[StepRequiredFields] = Field(description="投稿流程各步骤及必填字段")


class StepWithNext(BaseModel):
    """识别单步：当前步骤名、必填字段，以及进入下一步的按钮选择器。"""

    step: str = Field(description="当前步骤名称，从步骤导航栏原样选取")
    required_fields: list[str] = Field(description="该步骤必填字段的可读名称列表")
    next_button_selector: str | None = Field(
        description="进入「下一步」的按钮 CSS 选择器（Playwright 语法，如 "
        "button:has-text(\"下一步\")）；若当前已是最后一步（出现「提交/发布/确认投稿」）则为 null"
    )


class LoginSiteRequest(BaseModel):
    """从用户请求中提取站点名称。"""

    site_name: str = Field(description="站点名称，例如 A")


class LoginSelectors(BaseModel):
    """描述大模型从登录页识别出的登录控件 CSS 选择器。"""

    username_selector: str = Field(description="用户名输入框的 CSS 选择器")
    password_selector: str = Field(description="密码输入框的 CSS 选择器")
    login_button_selector: str = Field(description="登录按钮的 CSS 选择器")


class SubmitEntrySelector(BaseModel):
    """描述大模型从登录后页面识别出的「投稿」入口按钮 CSS 选择器。"""

    submit_button_selector: str = Field(
        description="「投稿 / 写文章 / 发布」等进入投稿流程的按钮或链接 CSS 选择器"
    )


def identify_site_name(user_request: str, model: BaseChatModel) -> str:
    """使用大模型从用户请求中识别需要登录的站点名称。"""
    print(repr(user_request))
    result = model.with_structured_output(LoginSiteRequest).invoke(
        [
            (
                "system",
                "从用户请求中提取站点短名称。比如“A平台”或“到A”中的站点名是“A”。"
                "只返回站点名称，不要包含‘平台’等后缀。",
            ),
            ("human", user_request),
        ]
    )
    if not isinstance(result, LoginSiteRequest) or not result.site_name.strip():
        raise ValueError("大模型未能从请求中识别站点名称。")
    return result.site_name.strip()


def identify_login_selectors(page: Page, model: BaseChatModel) -> LoginSelectors:
    """分析站点登录页并由大模型识别用户名、密码及登录按钮选择器。"""
    controls: list[dict[str, Any]] = page.locator(
        "input, button, [role='button']"
    ).evaluate_all(
        """elements => elements
            .filter(el => !el.disabled && el.getClientRects().length > 0)
            .map(el => {
                const labels = el.labels && el.labels.length
                    ? Array.from(el.labels).map(label => label.innerText.trim()).join(' ')
                    : '';
                const name = el.getAttribute('name') || '';
                const id = el.id || '';
                let suggestedSelector = '';
                if (id) {
                    suggestedSelector = '#' + CSS.escape(id);
                } else if (name) {
                    suggestedSelector = el.tagName.toLowerCase()
                        + '[name="' + CSS.escape(name) + '"]';
                }
                return {
                    tag: el.tagName.toLowerCase(),
                    type: el.type || el.getAttribute('role') || '',
                    name,
                    id,
                    label: labels || el.getAttribute('aria-label') || '',
                    placeholder: el.getAttribute('placeholder') || '',
                    text: (el.innerText || el.value || '').trim(),
                    suggested_selector: suggestedSelector
                };
            })"""
    )
    if not controls:
        raise ValueError("当前页面没有可见的登录表单控件。")

    structured_model = model.with_structured_output(LoginSelectors)
    result = structured_model.invoke(
        [
            (
                "system",
                "你是登录页控件识别助手。请从提供的可见控件中识别用户名输入框、"
                "密码输入框和登录按钮，并返回可供 Playwright locator() 使用的 CSS 选择器。"
                "优先使用控件提供的 suggested_selector；用户名优先 type=email/text，"
                "密码应为 type=password，登录按钮根据按钮文本或 submit 类型判断。"
                "不要猜测控件信息中不存在的选择器。",
            ),
            (
                "human",
                f"页面标题：{page.title()}\n页面地址：{page.url}\n"
                f"可见控件：{json.dumps(controls, ensure_ascii=False)}",
            ),
        ]
    )
    if not isinstance(result, LoginSelectors):
        raise TypeError("大模型未返回 LoginSelectors 结构。")

    selected_controls: dict[str, tuple[str, str]] = {}
    for field_name in (
        "username_selector",
        "password_selector",
        "login_button_selector",
    ):
        locator = page.locator(getattr(result, field_name))
        if locator.count() == 0 or not locator.first.is_visible():
            raise ValueError(f"大模型识别的选择器无效或不可见：{field_name}")
        selected_controls[field_name] = locator.first.evaluate(
            "el => [el.tagName.toLowerCase(), (el.type || el.getAttribute('role') || '').toLowerCase()]"
        )

    username_tag, username_type = selected_controls["username_selector"]
    if username_tag != "input" or username_type in {"password", "hidden", "submit"}:
        raise ValueError("大模型识别的用户名选择器没有指向有效的输入框。")
    password_tag, password_type = selected_controls["password_selector"]
    if password_tag != "input" or password_type != "password":
        raise ValueError("大模型识别的密码选择器没有指向密码输入框。")
    button_tag, button_type = selected_controls["login_button_selector"]
    if not (
        button_tag == "button"
        or (button_tag == "input" and button_type in {"submit", "button", "image"})
        or button_type == "button"
    ):
        raise ValueError("大模型识别的登录按钮选择器没有指向按钮控件。")
    print(f"{result.username_selector} {result.password_selector}")
    return result


def identify_submit_entry(page: Page, model: BaseChatModel) -> SubmitEntrySelector:
    """分析登录后页面并由大模型识别「投稿」按钮的选择器。

    参数：
        page: 已登录且停留在站点首页/控制台的 Playwright 页面。
        model: 用于结构化输出的 LangChain 聊天模型。

    返回：
        包含「投稿」按钮 CSS 选择器的 ``SubmitEntrySelector``。
    """
    submit_control_selector = (
        "a, button, input[type='button'], input[type='submit'], [role='button'], "
        "[onclick], li, [class*='btn'], [class*='button']"
    )

    def collect_controls() -> list[dict[str, Any]]:
        return page.locator(submit_control_selector).evaluate_all(
            """elements => elements
                .filter(el => !el.disabled
                    && el.getClientRects().length > 0
                    && (el.innerText || '').trim())
                .map(el => {
                    const id = el.id || '';
                    const name = el.getAttribute('name') || '';
                    const text = (el.innerText || el.getAttribute('aria-label') || '').trim();
                    let suggestedSelector = '';
                    if (id) {
                        suggestedSelector = '#' + CSS.escape(id);
                    } else if (name) {
                        suggestedSelector = el.tagName.toLowerCase()
                            + "[name='" + CSS.escape(name) + "']";
                    } else if (text && text.length <= 30) {
                        suggestedSelector = el.tagName.toLowerCase()
                            + ':has-text("' + text.replace(/"/g, '\\\\"') + '")';
                    }
                    return {
                        tag: el.tagName.toLowerCase(),
                        text: text,
                        href: el.href || '',
                        class: (el.className || '').toString(),
                        suggested_selector: suggestedSelector
                    };
                })"""
        )

    # 登录后页面多为 SPA，需等待渲染并做几次重试，避免页面尚未加载完就扫描
    controls: list[dict[str, Any]] = []
    for _ in range(3):
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except PlaywrightError:
            pass
        controls = collect_controls()
        if controls:
            break
        page.wait_for_timeout(1500)

    if not controls:
        print("DEBUG 诊断 -> url:", page.url)
        print("DEBUG 诊断 -> title:", page.title())
        print("DEBUG 诊断 -> iframe 数量:", len(page.frames) - 1)
        print("DEBUG 诊断 -> 页面元素总数:", page.locator("*").count())
        raise ValueError(
            "登录后页面没有可见的按钮或链接入口控件（可能仍在加载，或控件位于 iframe 中）。"
        )

    result = model.with_structured_output(SubmitEntrySelector).invoke(
        [
            (
                "system",
                "你是投稿入口识别助手。请从可见的链接/按钮中找出进入"
                "「投稿 / 写文章 / 发布」流程的入口，并返回可供 Playwright locator() "
                "使用的 CSS 选择器。优先使用控件提供的 suggested_selector；依据按钮文本"
                "或链接文字判断，例如包含「投稿」「写文章」「发布」等字样。"
                "不要猜测控件信息中不存在的选择器。",
            ),
            (
                "human",
                f"页面标题：{page.title()}\n页面地址：{page.url}\n"
                f"可见入口控件：{json.dumps(controls, ensure_ascii=False)}",
            ),
        ]
    )
    if not isinstance(result, SubmitEntrySelector):
        raise TypeError("大模型未返回 SubmitEntrySelector 结构。")

    locator = page.locator(result.submit_button_selector)
    if locator.count() == 0 or not locator.first.is_visible():
        raise ValueError("大模型识别的「投稿」入口选择器无效或不可见。")
    return result


def identify_step(page: Page, model: BaseChatModel) -> StepWithNext:
    """由大模型分析当前投稿页面（单步），提取步骤名、必填字段与「下一步」按钮选择器。

    参数：
        page: 已登录且停留在投稿表单某一步的 Playwright 页面。
        model: 用于结构化输出的 LangChain 聊天模型。

    返回：
        含 ``step``、``required_fields`` 与 ``next_button_selector`` 的单步识别结果。
    """
    form_control_selector = (
        "input, textarea, button, select, [role='checkbox'], [role='radio'], "
        "[role='combobox']"
    )

    def collect_form_controls() -> list[dict[str, Any]]:
        return page.locator(form_control_selector).evaluate_all(
            """elements => elements
                .filter(el => !el.disabled && el.getClientRects().length > 0)
                .map(el => {
                    const labels = el.labels && el.labels.length
                        ? Array.from(el.labels).map(label => label.innerText.trim()).join(' ')
                        : '';
                    const ariaLabel = el.getAttribute('aria-label') || '';
                    const placeholder = el.getAttribute('placeholder') || '';
                    // 向上查找包含标签文字的整行/分组（覆盖表格布局中的兄弟单元格）
                    const group = el.closest(
                        "tr, [class*='row'], [class*='item'], [class*='group'], "
                        + ".form-group, .control-group, label, [class*='form-item'], "
                        + "[class*='control-label'], form"
                    );
                    const groupText = group ? group.innerText.replace(/\\s+/g, ' ').trim() : '';

                    // 必填标记：HTML 属性、标签文字含 * / 必填 / 必，
                    // 或祖先的 ::before/::after 伪元素内容为红色星号（很多 SPA 这样渲染）
                    const requiredAttr = el.required
                        || el.getAttribute('aria-required') === 'true';
                    let requiredMarker = /[*★]|\u5fc5\u586b|\u5fc5/.test(
                        groupText + ' ' + labels + ' ' + ariaLabel
                    );
                    if (!requiredMarker) {
                        const anchors = [el, el.parentElement, group].filter(Boolean);
                        for (const a of anchors) {
                            for (const pseudo of ['::before', '::after']) {
                                const c = getComputedStyle(a, pseudo).content;
                                if (c && c !== 'none' && c !== 'normal' && /[*★]/.test(c)) {
                                    requiredMarker = true;
                                    break;
                                }
                            }
                            if (requiredMarker) break;
                        }
                    }

                    const label = labels || ariaLabel || placeholder || groupText || '';
                    return {
                        tag: el.tagName.toLowerCase(),
                        type: el.type || el.getAttribute('role') || '',
                        name: el.name || el.id || '',
                        label: label,
                        required: requiredAttr || requiredMarker,
                        options: el.tagName === 'SELECT'
                            ? Array.from(el.options).map(option => option.label)
                            : []
                    };
                })"""
        )

    # 点击「投稿」跳转后 SPA 需要渲染时间，等待并重试，避免过早扫描到空列表
    controls: list[dict[str, Any]] = []
    for _ in range(3):
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except PlaywrightError:
            pass
        controls = collect_form_controls()
        if controls:
            break
        page.wait_for_timeout(1500)

    if not controls:
        print("DEBUG 诊断 -> url:", page.url)
        print("DEBUG 诊断 -> title:", page.title())
        print("DEBUG 诊断 -> iframe 数量:", len(page.frames) - 1)
        print("DEBUG 诊断 -> 页面元素总数:", page.locator("*").count())
        raise ValueError("当前页面没有找到可见的表单控件。")

    print("DEBUG controls 数量:", len(controls))
    print("DEBUG controls 样本:", json.dumps(controls[:8], ensure_ascii=False))

    current_form = page.locator("form").filter(visible=True).first
    step_context = ""
    current_step = ""
    # 抓取步骤导航：优先找出「当前激活/高亮」的步骤项
    step_items = page.evaluate(
        """() => {
            const nodes = Array.from(document.querySelectorAll(
                'a.left-menu-tab, .steps a, [class*="step"] a, a[class*="step"], '
                + 'li[class*="step"], .wizard li, .breadcrumb li'
            ));
            const seen = new Set();
            const out = [];
            for (const n of nodes) {
                const text = (n.innerText || '').replace(/\\s+/g, ' ').trim();
                if (!/步骤|step/i.test(text)) continue;
                const li = n.closest('li');
                const active = n.classList.contains('active')
                    || (li && li.classList.contains('active'))
                    || /active|current|selected/.test((n.className || '').toString())
                    || n.getAttribute('aria-current') === 'step'
                    || n.getAttribute('aria-current') === 'true';
                const key = text + '|' + active;
                if (seen.has(key)) continue;
                seen.add(key);
                out.push({text, active});
            }
            return out;
        }"""
    )
    active_items = [item["text"] for item in step_items if item["active"]]
    all_items = [item["text"] for item in step_items]
    if active_items:
        # 取最后一个激活项作为「当前步骤」（避免子节点重复）
        current_step = active_items[-1]
        step_context = "\n".join(all_items)
    elif all_items:
        step_context = "\n".join(all_items)
    if not step_context:
        form_heading = current_form.locator("h1, h2, h3, legend").first
        if form_heading.count():
            step_context = form_heading.inner_text().strip()
    if not step_context:
        step_context = page.title()

    print("DEBUG step_context 内容:")
    print(step_context)
    if current_step:
        print("DEBUG 当前激活步骤:", current_step)
    structured_model = model.with_structured_output(StepWithNext)
    result = structured_model.invoke(
        [
            (
                "system",
                "你是投稿表单必填项识别助手。根据提供的当前页面信息，"
                "「当前表单步骤线索」给出页面步骤导航栏的完整条目列表，"
                "「当前激活步骤」给出导航中高亮/激活的那一项，它就是你现在正在填写的步骤；"
                "步骤名称（step 字段）必须等于「当前激活步骤」的文字，原样选取、不要改写或使用平台名。"
                "若「当前激活步骤」为空（无法判定），则步骤名称取导航里当前实际所在的那一项"
                "（检查/确认/提交页通常为导航最后一项，如「步骤 7 : 检查&提交」），"
                "不要从页面正文里其他步骤的说明文字中取值。"
                "steps 中只包含一个当前步骤的信息。"
                "当前页面通常包含多个分区（如文件上传、稿件语种、栏目、标题、摘要等），"
                "请将本页面上所有必填字段都列入 required_fields，不要只列第一个分区。"
                "required_fields 应列出必须由用户填写或选择的字段。"
                "required_fields 的每一项必须是该必填控件在页面上显示的「可读字段名称」（label），"
                "不要返回 HTML 的内部 name/id（如 docLang、typeKey 等）。"
                "对每一项做清洗：去掉开头的「*」「必填」「必」等标记；"
                "若 label 含有大段说明文字，只保留字段本名（"
                "例如「* 正文(pdf,doc,docx,xls,xlsx,zip) 数据和文字说明图……」只取「正文(pdf,doc,docx,xls,xlsx,zip)」）。"
                "正确示例：[\"正文(pdf,doc,docx,xls,xlsx,zip)\",\"稿件语种\",\"栏目\",\"标题\",\"摘要\"]。"
                "如何识别必填项：页面上红色星星标注的就是必填项；"
                "若控件 label、placeholder 或周围文字包含「*」「必填」「必」等标记，也应视为必填；"
                "优先依据 required/aria-required 标记，并结合上述视觉/文字提示判断；"
                "若仍无法判断，则将表单中用户必须填写的主要输入"
                "（如标题、摘要、栏目、语种等可见的 input/select/textarea/radio 控件）列为必填。"
                "不要把提交、下一步等按钮列为字段，也不要编造页面中不存在的字段。"
                "另外，请同时识别用于进入「下一步」的按钮（文字通常为「下一步」「继续」「保存并继续」「下一页」等），"
                "返回其 CSS 选择器（使用 Playwright 语法，例如 button:has-text(\"下一步\") 或 #nextBtn）；"
                "若当前步骤已经是投稿流程的最后一步（通常出现「提交」「发布」「确认投稿」等按钮），"
                "则 next_button_selector 返回 null。",
            ),
            (
                "human",
                f"页面标题：{page.title()}\n"
                f"当前表单步骤线索：{step_context}\n"
                f"当前激活步骤：{current_step or '（无法判定，请自行从导航中确定当前步骤）'}\n"
                f"页面控件：{json.dumps(controls, ensure_ascii=False)}",
            ),
        ]
    )
    if not isinstance(result, StepWithNext):
        raise TypeError("大模型未返回 StepWithNext 结构。")
    if not result.step:
        raise ValueError("大模型未能识别当前步骤名称。")
    return result


def run_workflow(
    user_request: str,
    *,
    model: BaseChatModel | None = None,
    headless: bool = True,
) -> tuple[bool, SubmissionSchema | None]:
    """执行“理解站点 → 识别登录选择器 → 登录工具 → 识别必填项”流程。"""
    chat_model = model or create_chat_model()
    site_name = identify_site_name(user_request, chat_model)
    with LoginSession(headless=headless) as session:
        if not session.open_login_page(site_name):
            return False, None
        print(f"已打开 {site_name} 登录页：{session.page.url}")
        print("DEBUG url :", session.page.url)
        print("DEBUG title:", session.page.title())
        print(
            "DEBUG form-control count:",
            session.page.locator(
            "input, button, [role='button']"
        ).count(),
        )
        selectors = identify_login_selectors(session.page, chat_model)
        login_tool: BaseTool = create_read_configs_login_tool(session)
        login_succeeded = login_tool.invoke(
            {"site_name": site_name, **selectors.model_dump()}
        )
        if not login_succeeded:
            return False, None

        # 登录成功后，由大模型识别并点击页面上的「投稿」按钮进入投稿流程
        print("DEBUG 登陆成功后 url  :", session.page.url)
        print("DEBUG 登陆成功后 title:", session.page.title())
        submit_entry: SubmitEntrySelector = identify_submit_entry(session.page, chat_model)
        click_submit_entry_tool = create_click_submit_entry_tool(session)
        if not click_submit_entry_tool.invoke(
            {"submit_button_selector": submit_entry.submit_button_selector}
        ):
            print("点击「投稿」按钮失败：请检查识别出的选择器。")
            return False, None
        print("DEBUG 点击投稿后 url  :", session.page.url)
        print("DEBUG 点击投稿后 title:", session.page.title())

        # 递归遍历投稿流程的每一步：识别当前步骤 → 点击「下一步」→ 识别下一步，直到最后一步
        submission = SubmissionSchema(steps=[])
        max_steps = 20
        for step_index in range(max_steps):
            try:
                step_info = identify_step(session.page, chat_model)
            except Exception as exc:  # 某步识别失败不应中断整体流程
                print(f"第 {step_index + 1} 步识别失败：{exc}")
                break

            submission.steps.append(
                StepRequiredFields(
                    step=step_info.step,
                    required_fields=step_info.required_fields,
                )
            )
            print(
                f"DEBUG 已识别第 {step_index + 1} 步：{step_info.step} "
                f"下一步按钮：{step_info.next_button_selector}"
            )

            if not step_info.next_button_selector:
                print("DEBUG 已到投稿流程最后一步，递归结束。")
                break

            next_locator = session.page.locator(step_info.next_button_selector)
            if next_locator.count() == 0 or not next_locator.first.is_visible():
                print("DEBUG 未找到「下一步」按钮，停止递归。")
                break
            next_locator.first.click()

            # 等待下一页渲染/加载完成
            try:
                session.page.wait_for_load_state("networkidle", timeout=10000)
            except PlaywrightError:
                pass
            session.page.wait_for_timeout(1500)

        return True, submission


def main() -> None:
    """从终端读取用户请求，运行节点流程并打印结构化结果。"""
    user_request = input('请输入需求（例如：投 10 篇文章到 A 平台）：').strip()
    if not user_request:
        return

    login_succeeded, required_fields = run_workflow(user_request)
    if not login_succeeded:
        print("登录失败：请检查站点名称、.env 凭据和登录选择器。")
        return
    print(required_fields.model_dump_json(indent=2))


if __name__ == "__main__":
    main()