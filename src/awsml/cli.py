"""Command-line entry point.

    awsml pipeline           # ingest, train, evaluate, register
    awsml datasets           # list lake house datasets
    awsml models             # show the model registry
    awsml rag "question"     # ask the corpus
    awsml serve              # run the FastAPI inference service
"""

from __future__ import annotations

import argparse
import json

from awsml.config import Settings
from awsml.lakehouse import datasets
from awsml.pipeline import run
from awsml.rag import answer
from awsml.registry import list_models


def main() -> None:
    parser = argparse.ArgumentParser(prog="awsml", description="AWS AI/ML platform reference")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pipeline", help="run ingest, train, evaluate, register")
    sub.add_parser("datasets", help="list lake house datasets")
    sub.add_parser("models", help="show the model registry")
    sub.add_parser("serve", help="run the FastAPI inference service")

    rag_parser = sub.add_parser("rag", help="ask the corpus a question")
    rag_parser.add_argument("question")

    args = parser.parse_args()
    settings = Settings.from_env()

    if args.command == "pipeline":
        print(json.dumps(run(settings), indent=2, default=str))
    elif args.command == "datasets":
        print(json.dumps(datasets(settings), indent=2))
    elif args.command == "models":
        print(json.dumps(list_models(settings), indent=2))
    elif args.command == "rag":
        print(json.dumps(answer(args.question, settings), indent=2))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("awsml.serve:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
