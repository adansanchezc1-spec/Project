from dataclasses import dataclass, field


@dataclass
class CleaningReport:
    missing_values_fixed: int = 0
    duplicate_rows_removed: int = 0
    formatting_issues: int = 0
    summary: str = ""
    nulls_removed: int = 0
    removed_columns: list[str] = field(default_factory=list)
    affected_rows: int = 0

    def generate_summary(self) -> str:
        if self.summary:
            return self.summary

        removed_columns = ", ".join(self.removed_columns)
        return (
            f"Nulls removed: {self.nulls_removed}; "
            f"Duplicate rows removed: {self.duplicate_rows_removed}; "
            f"Formatting issues: {self.formatting_issues}; "
            f"Removed columns: {removed_columns or 'none'}; "
            f"Affected rows: {self.affected_rows}"
        )
