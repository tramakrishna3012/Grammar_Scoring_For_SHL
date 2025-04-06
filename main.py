import voice_app
from data_utils import load_jfleg_dataset, generate_csv
from grammar_score import grammar_score
from model_utils import train_eval_model
from inference import correct_text

def main():
    train_dataset = load_jfleg_dataset("validation[:3]")
    eval_dataset = load_jfleg_dataset("test[:3]")

    generate_csv("train.csv", train_dataset)
    generate_csv("eval.csv", eval_dataset)

    print("🚀 Training and evaluating model...")
    model, tokenizer = train_eval_model("train.csv", "eval.csv")

    # Sample inference
    print("🔮 Running sample inference:")
    samples = [
        "This sentences, has bads grammar and spelling!",
        "I am enjoys, writtings articles ons AI and I also enjoyed write articling on AI."
    ]
    for s in samples:
        print("Original:", s)
        print("Corrected:", correct_text(model, tokenizer, s))
        print("Grammar score:", grammar_score(s, correct_text(model, tokenizer, s)))

    # Optional: Run voice app
    input_text = voice_app.voice_input_to_text()
    print("Original:", input_text)
    print("Corrected:", correct_text(model, tokenizer, input_text))
    print("Grammar score:", grammar_score(input_text, correct_text(model, tokenizer, input_text)))

if __name__ == "__main__":
    main()
