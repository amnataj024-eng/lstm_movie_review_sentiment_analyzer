"""Command-line entry point.

    python main.py train      # train the LSTM and save it
    python main.py evaluate   # accuracy, precision, recall, F1, confusion matrix
    python main.py predict "This movie was fantastic!"
    python main.py interactive
    python main.py demo       # a few sample reviews
"""
import argparse

DEMO_REVIEWS = [
    "An absolutely wonderful film with brilliant acting and a touching story.",
    "Boring, predictable and far too long. I wanted my money back.",
    "The plot was weak, but the soundtrack and visuals were stunning.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM Movie Review Sentiment Analyzer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train the model")
    sub.add_parser("evaluate", help="evaluate on the test set")
    p = sub.add_parser("predict", help="predict one review")
    p.add_argument("text", help="the review text, in quotes")
    sub.add_parser("interactive", help="type reviews one by one")
    sub.add_parser("demo", help="run sample reviews")
    args = parser.parse_args()

    # Imports are inside so that `--help` is instant (TensorFlow is slow to import).
    if args.command == "train":
        from src.train import train
        train()
    elif args.command == "evaluate":
        from src.evaluate import evaluate
        evaluate()
    elif args.command == "predict":
        from src.predict import predict_sentiment
        r = predict_sentiment(args.text)
        print(f"{r['label']} (confidence {r['confidence']:.1%})")
    elif args.command == "interactive":
        from src.predict import interactive
        interactive()
    elif args.command == "demo":
        from src.predict import predict_sentiment
        for text in DEMO_REVIEWS:
            r = predict_sentiment(text)
            print(f"{r['label']:8s} {r['confidence']:.1%}  |  {text}")


if __name__ == "__main__":
    main()
