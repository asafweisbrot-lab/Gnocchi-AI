"""Build every synthetic source, in order: teams and OKRs -> CRM history -> AI sources."""
from . import ai_sources, crm_history, teams_okrs


def main() -> None:
    teams_okrs.main()
    crm_history.main()
    ai_sources.main()


if __name__ == "__main__":
    main()
