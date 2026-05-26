#include "../util/ai/brain_model.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <plaintext_document.txt>\n";
        return 1;
    }

    std::string filepath = argv[1];
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open " << filepath << "\n";
        return 1;
    }

    std::cout << "Reading file and building vocabulary...\n";
    SimpleTokenizer tokenizer;
    std::vector<int> token_sequence;
    
    std::string word;
    while (file >> word) {
        for (char& c : word) {
            if ((unsigned char)c < 128) {
                c = ::tolower((unsigned char)c);
            }
        }
        token_sequence.push_back(tokenizer.get_or_add(word));
    }
    file.close();

    std::cout << "Vocabulary size: " << tokenizer.next_idx << "\n";
    std::cout << "Total tokens to process: " << token_sequence.size() << "\n";

    int vocab_size = std::max(tokenizer.next_idx, 100); 
    BrainLikeModel model(vocab_size, 64, 64, 100, 32, 5, 5);

    std::cout << "Training initiated...\n";
    
    int correct_predictions = 0;
    
    for (size_t t = 0; t < token_sequence.size() - 1; ++t) {
        int x_idx = token_sequence[t];
        int next_idx = token_sequence[t + 1];

        Eigen::VectorXf logits;
        int pred_token = model.forward(x_idx, logits);

        float reward = (pred_token == next_idx) ? 0.9f : 0.0f;
        
        // Instead of calculating heavy gradients instantly, we queue them
        model.buffer_experience(x_idx, next_idx, reward);

        if (pred_token == next_idx) correct_predictions++;

        // Emulate biological "Sleep cycles" and "context resets"
        // Prevents working memory from completely washing out over a whole book
        if (t % 1000 == 0 && t > 0) {
            std::cout << "Step " << t << " | Acc (last 1k): " 
                      << (correct_predictions / 1000.0f) * 100.0f << "%\n";
            correct_predictions = 0;
            
            // Perform the deferred heavy learning operations
            model.sleep_and_consolidate();
        }
        
        // Minor breaks in context (like ending a paragraph or train of thought)
        if (t % 50 == 0) {
            model.reset_context();
        }
    }

    // Final consolidation step
    model.sleep_and_consolidate();

    std::cout << "Training complete. Saving model...\n";
    tokenizer.save("vocab.txt");
    model.save("model.bin");
    std::cout << "Saved to vocab.txt and model.bin.\n";

    return 0;
}