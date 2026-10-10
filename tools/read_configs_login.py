"""读取站点登录配置并执行浏览器登录的工具节点。"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.tools import BaseTool
from playwright.sync_api import (
    Browser,
    Error as PlaywrightError,
    Page,
    Playwright,
    sync_playwright,
)

from tools.browser_tools.tools import click_button, enter_credentials, open_url


class LoginSession:
    """持有浏览器会话，供登录配置工具和页面分析节点复用。"""

    def __init__(self, *, headless: bool = True) -> None:
        """初始化浏览器会话。

        参数：
            headless: 是否以无头模式启动浏览器。
        """
        self.headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._site_name: str | None = None

    @property
    def page(self) -> Page:
        """返回当前浏览器页面；会话尚未启动时抛出异常。"""
        if self._page is None:
            raise RuntimeError("请先使用 with LoginSession(...) 启动浏览器会话。")
        return self._page

    def __enter__(self) -> "LoginSession":
        """启动 Chromium 并创建供后续节点复用的页面。"""
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()
        return self

    def __exit__(self, *_: object) -> None:
        """关闭浏览器并释放 Playwright 资源。"""
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._page = None
        self._site_name = None

    @staticmethod
    def _read_site_config(site_name: str) -> tuple[str, str, str] | None:
        """读取站点对应的 URL、用户名和密码环境变量。"""
        if not site_name.strip():
            return None

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        site_prefix = f"SITE_{site_name.strip().upper()}"
        site_url = os.getenv(f"{site_prefix}_URL")
        username = os.getenv(f"{site_prefix}_USERNAME")
        password = os.getenv(f"{site_prefix}_PASSWORD")
        if not site_url or not username or not password:
            return None
        return site_url, username, password

    def open_login_page(self, site_name: str) -> bool:
        """读取站点 URL 并打开登录页，供大模型识别页面选择器。

        输入：
            site_name: 用户提供的站点名称，例如 ``a``。

        返回：
            页面成功打开时返回 ``True``，配置缺失或浏览器导航失败时返回 ``False``。
        """
        site_config = self._read_site_config(site_name)
        if site_config is None:
            return False

        try:
            open_url(self.page, site_config[0])
        except PlaywrightError:
            return False
        self._site_name = site_name.strip().upper()
        return True

    def read_configs_login(
        self,
        site_name: str,
        username_selector: str,
        password_selector: str,
        login_button_selector: str,
    ) -> bool:
        """读取站点凭据，使用大模型识别的选择器执行登录并验证结果。

        输入：
            site_name: 用户提供的站点名称。
            username_selector: 大模型从登录页识别出的用户名输入框 CSS 选择器。
            password_selector: 大模型从登录页识别出的密码输入框 CSS 选择器。
            login_button_selector: 大模型从登录页识别出的登录按钮 CSS 选择器。

        返回：
            登录表单提交后消失或页面 URL 变化时返回 ``True``，否则返回 ``False``。
        """
        normalized_site_name = site_name.strip().upper()
        if not normalized_site_name or normalized_site_name != self._site_name:
            return False

        site_config = self._read_site_config(site_name)
        if site_config is None:
            return False
        _, username, password = site_config

        try:
            username_field = self.page.locator(username_selector)
            password_field = self.page.locator(password_selector)
            login_button = self.page.locator(login_button_selector)
            if not all(
                locator.count() > 0 and locator.first.is_visible()
                for locator in (username_field, password_field, login_button)
            ):
                return False

            initial_url = self.page.url
            enter_credentials(
                self.page,
                username_selector,
                username,
                password_selector,
                password,
            )
            click_button(self.page, login_button_selector)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.page.url != initial_url:
                    return True
                if not any(
                    locator.count() > 0 and locator.first.is_visible()
                    for locator in (username_field, password_field, login_button)
                ):
                    return True
                self.page.wait_for_timeout(250)
            return False
        except PlaywrightError:
            return False


def create_read_configs_login_tool(session: LoginSession) -> BaseTool:
    """创建接收大模型所识别选择器的 LangChain 登录工具。

    参数：
        session: 已启动并持有浏览器页面的登录会话。

    返回：
        要求站点名和三个登录控件 CSS 选择器的 LangChain 工具。
    """

    @tool
    def read_configs_login(
        site_name: str,
        username_selector: str,
        password_selector: str,
        login_button_selector: str,
    ) -> bool:
        """读取 SITE_<站点名> 环境变量，按传入选择器登录并返回结果。"""
        return session.read_configs_login(
            site_name,
            username_selector,
            password_selector,
            login_button_selector,
        )

    return read_configs_login


def create_click_submit_entry_tool(session: LoginSession) -> BaseTool:
    """创建点击登录后页面「投稿」按钮的 LangChain 工具。

    参数：
        session: 已登录并持有浏览器页面的会话。

    返回：
        接收「投稿」按钮 CSS 选择器的 LangChain 工具，点击成功返回 ``True``。
    """

    @tool
    def click_submit_entry(submit_button_selector: str) -> bool:
        """点击登录后页面上的「投稿 / 写文章」按钮，进入投稿流程。"""
        try:
            session.page.locator(submit_button_selector).click()
            return True
        except PlaywrightError:
            return False

    return click_submit_entry
