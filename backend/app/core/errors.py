class AppError(Exception):
    """An error with a stable code; the frontend translates the code, never prose."""

    def __init__(self, code: str, **params: object) -> None:
        super().__init__(code)
        self.code = code
        self.params = params
