# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from contextvars import ContextVar
from rich.console import Console, ConsoleOptions, RenderResult
from rich.progress import Progress, ProgressColumn, SpinnerColumn, TextColumn
from rich.text import Text
from rich.traceback import Traceback
from types import MethodType

current_progress = ContextVar("current_progress", default=None)
progress_task_stack = ContextVar("progress_task_stack", default=[])

class CustomSpinnerColumn(SpinnerColumn):
    
    def __init__(
        self,
        spinner_name="dots",
        success_text="[bold green]✔[/bold green]",
        failure_text="[bright_red]✘[/bright_red]",
        style="",
    ):
        super().__init__(spinner_name=spinner_name, style=style)
        self.success_text = success_text
        self.failure_text = failure_text

    def render(self, task):
        done_text = (
            self.failure_text
            if task.fields.get("status") == "error"
            else self.success_text
        )
        return done_text if task.finished else self.spinner

class CustomTextColumn(TextColumn):
    """Custom text column that changes color based on task status"""
    
    def __init__(self, text_format="{task.description}"):
        super().__init__(text_format)
    
    def render(self, task):
        # Indent and color
        description = task.description
        indent_level = task.fields.get("indent_level", 0)
        indent = self.__get_indent(indent_level)
        task.description = f"{indent}{description}"
        if task.fields.get("status") == "error":
            task.description = f"[bright_red]{task.description}[/bright_red]"
        # Render
        text = super().render(task)
        task.description = description        
        # Return
        return text
    
    def __get_indent (self, level: int) -> str:
        if level == 0:
            return ""
        indicator = "└── "
        return " " * len(indicator) * (level - 1) + indicator

class CustomProgress(Progress):

    def __init__(
        self,
        *columns: ProgressColumn,
        console=None,
        auto_refresh=True,
        refresh_per_second = 10,
        speed_estimate_period=30,
        transient=False,
        redirect_stdout=True,
        redirect_stderr=True,
        get_time=None,
        disable=False,
        expand=False
    ):
        default_columns = list(columns) if len(columns) > 0 else [
            CustomSpinnerColumn(),
            CustomTextColumn("[progress.description]{task.description}"),
        ]
        super().__init__(
            *default_columns,
            console=console,
            auto_refresh=auto_refresh,
            refresh_per_second=refresh_per_second,
            speed_estimate_period=speed_estimate_period,
            transient=transient,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            get_time=get_time,
            disable=disable,
            expand=expand
        )
        self.default_columns = default_columns

    def __enter__(self):
        self._token = current_progress.set(self)
        self._stack_token = progress_task_stack.set([])
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_progress.reset(self._token)
        progress_task_stack.reset(self._stack_token)
        return super().__exit__(exc_type, exc_val, exc_tb)
    
    def get_renderables(self):
        for task in self.tasks:
            task_columns = task.fields.get("columns") or list()
            self.columns = self.default_columns + task_columns
            yield self.make_tasks_table([task])

class CustomProgressTask:

    def __init__(
        self,
        *,
        loading_text: str,
        done_text: str=None,
        columns: list[ProgressColumn]=None
    ):
        self.loading_text = loading_text
        self.done_text = done_text
        self.task_id = None
        self.columns = columns

    def __enter__(self):
        progress = current_progress.get()
        if progress is not None:
            self.task_id = progress.add_task(
                self.loading_text,
                total=1,
                columns=self.columns,
                indent_level=len(progress_task_stack.get())
            )
            current_stack = progress_task_stack.get()
            progress_task_stack.set(current_stack + [self.task_id])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        progress = current_progress.get()
        if progress is not None and self.task_id is not None:
            current_task = progress._tasks[self.task_id]
            final_status = "error" if exc_type is not None else current_task.fields.get("status")
            final_description = self.done_text or current_task.description
            progress.update(
                self.task_id,
                description=final_description,
                completed=current_task.total,
                status=final_status
            )
            current_stack = progress_task_stack.get()
            if current_stack:
                progress_task_stack.set(current_stack[:-1])
            # Print a finalized line for the completed task so history persists
            # in the terminal scrollback, then remove it from the live region.
            try:
                indent_level = current_task.fields.get("indent_level", 0)
                indicator = "└── "
                indent = "" if indent_level == 0 else (" " * len(indicator) * (indent_level - 1) + indicator)
                is_error = final_status == "error"
                icon = "[bright_red]✘[/bright_red]" if is_error else "[bold green]✔[/bold green]"
                message = f"{icon} {indent}{final_description}"
                if is_error:
                    message = f"[bright_red]{message}[/bright_red]"
                progress.console.print(message)
                progress.remove_task(self.task_id)
            except Exception:
                pass
        self.task_id = None
        return False

    def update(self, **kwargs):
        progress = current_progress.get()
        if progress is None or self.task_id is None:
            return
        progress.update(self.task_id, **kwargs)

    def finish (self, message: str):
        self.done_text = message
    
class TracebackMarkupConsole(Console):

    def print(
        self,
        *objects,
        sep = " ",
        end = "\n",
        style = None,
        justify = None,
        overflow = None,
        no_wrap = None,
        emoji = None,
        markup = None,
        highlight = None,
        width = None,
        height = None,
        crop = True,
        soft_wrap = None,
        new_line_start = False
    ):
        traceback = objects[0]
        if isinstance(traceback, Traceback):
            stack = traceback.trace.stacks[0]
            original_rich_console = traceback.__rich_console__
            def __rich_console__ (self: Traceback, console: Console, options: ConsoleOptions) -> RenderResult:
                for renderable in original_rich_console(console, options):
                    if (
                        isinstance(renderable, Text) and
                        any(part.startswith(f"{stack.exc_type}:") for part in renderable._text)
                    ):
                        yield Text.assemble(
                            (f"{stack.exc_type}: ", "traceback.exc_type"),
                            Text.from_markup(stack.exc_value)
                        )
                    else:
                        yield renderable
            traceback.__rich_console__ = MethodType(__rich_console__, traceback)
        return super().print(
            *objects,
            sep=sep,
            end=end,
            style=style,
            justify=justify,
            overflow=overflow,
            no_wrap=no_wrap,
            emoji=emoji,
            markup=markup,
            highlight=highlight,
            width=width,
            height=height,
            crop=crop,
            soft_wrap=soft_wrap,
            new_line_start=new_line_start
        )