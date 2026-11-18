#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pygments.style import Style
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String
from pygments.token import Text as TextToken
from rich import box
from rich.panel import Panel
from rich.syntax import PygmentsSyntaxTheme, Syntax

class VSCodeDarkStyle(Style):
    """
    Dark+ token colors with a darker surface to distinguish the snippet.
    """
    background_color = "#141414"
    styles = {
        TextToken: "#d4d4d4",
        Comment: "italic #6a9955",
        Keyword: "#569cd6",
        Keyword.Namespace: "#c586c0",
        Keyword.Reserved: "#c586c0",
        Name: "#9cdcfe",
        Name.Builtin: "#dcdcaa",
        Name.Class: "#4ec9b0",
        Name.Function: "#dcdcaa",
        Name.Namespace: "#4ec9b0",
        Name.Decorator: "#dcdcaa",
        Number: "#b5cea8",
        String: "#ce9178",
        String.Escape: "#d7ba7d",
        Operator: "#d4d4d4",
        Operator.Word: "#c586c0",
        Punctuation: "#d4d4d4",
    }


CODE_THEME = PygmentsSyntaxTheme(VSCodeDarkStyle)

def code_panel(
    code: str,
    language: str,
    title: str
) -> Panel:
    """
    Render a highlighted fence with the same dark surface in its padding.
    """
    return Panel(
        Syntax(
            code,
            language,
            theme=CODE_THEME,
            background_color=VSCodeDarkStyle.background_color,
            word_wrap=True,
        ),
        title=title,
        title_align="left",
        border_style="#3c3c3c",
        style=f"on {VSCodeDarkStyle.background_color}",
        box=box.ROUNDED,
        padding=(1, 2),
    )
