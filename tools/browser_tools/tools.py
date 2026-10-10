"""封装常用的同步 Playwright 浏览器操作。"""

from typing import Any, Literal


def open_url(
    page: Any,
    url: str,
    *,
    wait_until: Literal["commit", "domcontentloaded", "load", "networkidle"] = "load",
) -> Any:
    """导航到指定 URL，并返回导航响应。

    参数：
        page: Playwright 同步模式的 ``Page`` 实例。
        url: 要打开的 URL。
        wait_until: 返回前等待的导航阶段。

    返回：
        导航响应；如果没有可用响应，则返回 ``None``。
    """
    return page.goto(url, wait_until=wait_until)


def enter_username(page: Any, selector: str, username: str) -> None:
    """在指定元素中输入用户名。

    参数：
        page: Playwright 同步模式的 ``Page`` 实例。
        selector: 用户名输入框的选择器。
        username: 要输入的用户名。
    """
    page.locator(selector).fill(username)


def enter_password(page: Any, selector: str, password: str) -> None:
    """在指定元素中输入密码。

    参数：
        page: Playwright 同步模式的 ``Page`` 实例。
        selector: 密码输入框的选择器。
        password: 要输入的密码。
    """
    page.locator(selector).fill(password)


def enter_credentials(
    page: Any,
    username_selector: str,
    username: str,
    password_selector: str,
    password: str,
) -> None:
    """分别在对应的输入框中填写用户名和密码。

    参数：
        page: Playwright 同步模式的 ``Page`` 实例。
        username_selector: 用户名输入框的选择器。
        username: 要输入的用户名。
        password_selector: 密码输入框的选择器。
        password: 要输入的密码。
    """
    enter_username(page, username_selector, username)
    enter_password(page, password_selector, password)


def click_radio(page: Any, selector: str) -> None:
    """选中指定的单选框。

    参数：
        page: Playwright 同步模式的 ``Page`` 实例。
        selector: 单选框的选择器。
    """
    page.locator(selector).check()


def click_button(page: Any, selector: str) -> None:
    """点击指定的按钮或其他可点击元素。

    参数：
        page: Playwright 同步模式的 ``Page`` 实例。
        selector: 要点击的按钮或元素的选择器。
    """
    page.locator(selector).click()


def upload_file(page: Any, selector: str, file_path: str) -> None:
    """通过指定的文件输入框上传文件。

    参数：
        page: Playwright 同步模式的 ``Page`` 实例。
        selector: ``input[type=file]`` 元素的选择器。
        file_path: 要上传的本地文件路径。
    """
    page.locator(selector).set_input_files(file_path)


def select_dropdown_option(
    page: Any,
    selector: str,
    *,
    value: str | None = None,
    label: str | None = None,
    index: int | None = None,
) -> list[str]:
    """在原生 HTML ``select`` 下拉框中选择选项。

    ``value``、``label`` 和 ``index`` 必须且只能指定一个。

    参数：
        page: Playwright 同步模式的 ``Page`` 实例。
        selector: ``select`` 元素的选择器。
        value: 选项的 value 属性值。
        label: 选项显示的文本。
        index: 选项从零开始的索引。

    返回：
        Playwright 所选选项的值。

    异常：
        ValueError: 未提供选择条件，或提供了多个选择条件时抛出。
    """
    selection = {"value": value, "label": label, "index": index}
    selected = {key: item for key, item in selection.items() if item is not None}
    if len(selected) != 1:
        raise ValueError("Specify exactly one of value, label, or index.")
    return page.locator(selector).select_option(**selected)


def enter_text(page: Any, selector: str, text: str) -> None:
    """在输入框或文本区域中输入文本，并替换其中现有内容。

    参数：
        page: Playwright 同步模式的 ``Page`` 实例。
        selector: 文本输入框或文本区域的选择器。
        text: 要输入的文本。
    """
    page.locator(selector).fill(text)