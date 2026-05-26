#include "../util/ai/brain_model.hpp"
#include <iostream>
#include <string>
#include <vector>

int main() {
    SimpleTokenizer tokenizer;
    std::cout << "Loading vocabulary...\n";
    tokenizer.load("vocab.txt");
    
    if (tokenizer.next_idx == 0) {
        std::cerr << "Vocabulary is empty! Run the trainer first.\n";
        return 1;
    }

    std::cout << "Loading model...\n";
    int vocab_size = tokenizer.next_idx;
    BrainLikeModel model(vocab_size, 64, 64, 100, 32, 5, 5);
    model.load("model.bin");
    
    std::cout << "Model loaded! Commands:\n";
    std::cout << "  /quit   - Exit the chat\n";
    std::cout << "  /update - Triggers sleep cycle to solidify memories\n";

    while (true) {
        std::cout << "\nYou: ";
        std::string input;
        if (!std::getline(std::cin, input)) break;
        
        if (input == "/quit") break;
        if (input == "/update") {
            std::cout << "[System] NPC goes to sleep... Processing " << model.sleep_buffer.size() << " experiences.\n";
            model.sleep_and_consolidate();
            
            // Save the expanded vocabulary and model state so new words persist across reboots!
            tokenizer.save("vocab.txt");
            model.save("model.bin");
            
            std::cout << "[System] NPC wakes up! Memory solidified and saved.\n";
            continue;
        }

        std::stringstream ss(input);
        std::string word;
        std::vector<int> user_tokens;

        // Tokenize and build the sequence
        while (ss >> word) {
            for (char& c : word) {
                if ((unsigned char)c < 128) {
                    c = ::tolower((unsigned char)c);
                }
            }
            user_tokens.push_back(tokenizer.get_or_add(word));
        }

        if (tokenizer.next_idx > model.vocab_size) {
            std::cout << "[System] NPC learned " << (tokenizer.next_idx - model.vocab_size) << " new concept(s)!\n";
            model.expand_vocabulary(tokenizer.next_idx);
        }

        if (user_tokens.empty()) continue;

        // Process the player's full sentence to build up continuous EMA working memory
        int prev_idx = -1;
        for (int token : user_tokens) {
            if (prev_idx != -1) {
                // High reward (0.8f) to remember the player's statements 
                model.buffer_experience(prev_idx, token, 0.8f);
            }
            Eigen::VectorXf logits;
            model.forward(token, logits); 
            prev_idx = token;
        }

        std::cout << "Model: ";
        
        // Generate NPC's train of thought (10 words)
        int current_idx = prev_idx;
        for (int i = 0; i < 10; ++i) {
            Eigen::VectorXf logits;
            int next_idx = model.forward(current_idx, logits);
            
            // Standard conversational reward buffer
            model.buffer_experience(current_idx, next_idx, 0.1f);
            
            std::cout << tokenizer.get_word(next_idx) << " ";
            current_idx = next_idx;
        }
        std::cout << std::endl;
    }

    return 0;
}