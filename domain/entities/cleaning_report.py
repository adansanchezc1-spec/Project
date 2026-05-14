from dataclasses import dataclass


@dataclass
class CleaningReport:
    missing_values_fixed: int = 0
    duplicate_rows_removed: int = 0
    formatting_issues: int = 0
    summary: str = ''
