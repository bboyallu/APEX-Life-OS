"""Allow ``python -m apex`` to invoke the CLI."""

import sys

from apex.cli import main

if __name__ == "__main__":
    sys.exit(main())
