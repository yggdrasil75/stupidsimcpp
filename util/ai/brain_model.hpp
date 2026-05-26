#ifndef BRAIN_MODEL_HPP
#define BRAIN_MODEL_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <sstream>

/**
 * @class LIFNeuron
 * @brief Leaky Integrate-and-Fire neuron with Habituation (Fatigue).
 */
class LIFNeuron {
public:
    float tau;         
    float threshold;   
    float base_threshold; ///< Normal resting threshold
    float threshold_decay; ///< How fast fatigue recovers
    float v_reset;     
    float dt;          
    float v;           
    int refractory;    

    LIFNeuron(float tau = 10.0f, float threshold = 1.0f, float v_reset = 0.0f, float dt = 1.0f)
        : tau(tau), threshold(threshold), base_threshold(threshold), threshold_decay(0.05f),
          v_reset(v_reset), dt(dt), v(0.0f), refractory(0) {}

    float step(float I) {
        if (refractory > 0) {
            refractory--;
            v = v_reset;
            // Recover from fatigue even while refractory
            threshold += (base_threshold - threshold) * threshold_decay;
            return 0.0f;
        }

        float dv = (-v / tau + I) * dt;
        v += dv;

        if (v >= threshold) {
            v = v_reset;
            refractory = 2;
            threshold += 0.5f; // Habituation: Neuron gets tired if it fires too much
            return 1.0f;
        }

        // Slowly relax threshold back to base level
        threshold += (base_threshold - threshold) * threshold_decay;
        return 0.0f;
    }

    void reset() {
        v = 0.0f;
        refractory = 0;
        threshold = base_threshold;
    }
};

/**
 * @class Hippocampus
 * @brief Episodic memory for instant retrieval (Vector DB / RAG simulation).
 */
class Hippocampus {
public:
    std::vector<Eigen::VectorXf> memory_keys;
    std::vector<int> memory_values;
    size_t max_memories = 500;

    void add_memory(const Eigen::VectorXf& context, int target_token) {
        if (memory_keys.size() >= max_memories) {
            memory_keys.erase(memory_keys.begin());
            memory_values.erase(memory_values.begin());
        }
        memory_keys.push_back(context);
        memory_values.push_back(target_token);
    }

    Eigen::VectorXf retrieve(const Eigen::VectorXf& current_context) {
        if(memory_keys.empty()) return Eigen::VectorXf::Zero(current_context.size());
        
        float max_sim = -1.0f;
        int best_idx = -1;
        float ctx_norm = current_context.norm();
        if (ctx_norm < 1e-6f) return Eigen::VectorXf::Zero(current_context.size());

        for(size_t i = 0; i < memory_keys.size(); i++) {
            float key_norm = memory_keys[i].norm();
            if (key_norm < 1e-6f) continue;
            
            float sim = current_context.dot(memory_keys[i]) / (ctx_norm * key_norm);
            if(sim > max_sim) { 
                max_sim = sim; 
                best_idx = i; 
            }
        }
        
        // Threshold for memory retrieval
        if (max_sim > 0.8f && best_idx != -1) {
            return memory_keys[best_idx];
        }
        return Eigen::VectorXf::Zero(current_context.size());
    }
};

class SubModel {
public:
    int id;
    int input_dim;
    int hidden_dim;
    int num_steps;
    float lr;

    Eigen::MatrixXf W_in;
    Eigen::MatrixXf W_rec;
    Eigen::MatrixXf W_out;

    std::vector<LIFNeuron> hidden_neurons;
    std::vector<Eigen::VectorXf> spike_history;

    SubModel(int id, int input_dim, int hidden_dim = 64, int num_steps = 10, float lr = 1e-3f)
        : id(id), input_dim(input_dim), hidden_dim(hidden_dim), num_steps(num_steps), lr(lr) 
    {
        float scale = 0.1f;
        W_in = Eigen::MatrixXf::Random(hidden_dim, input_dim) * scale;
        W_rec = Eigen::MatrixXf::Random(hidden_dim, hidden_dim) * (scale * 0.1f);
        W_out = Eigen::MatrixXf::Random(input_dim, hidden_dim) * scale;

        for (int i = 0; i < hidden_dim; ++i) {
            hidden_neurons.push_back(LIFNeuron(20.0f, 1.0f));
        }
    }

    Eigen::VectorXf forward(const Eigen::VectorXf& x) {
        // REMOVED: neuron.reset(). Membrane potentials now persist across words!
        spike_history.clear();

        Eigen::VectorXf spikes = Eigen::VectorXf::Zero(hidden_dim);

        for (int t = 0; t < num_steps; ++t) {
            Eigen::VectorXf I = (W_in * x) + (W_rec * spikes);
            Eigen::VectorXf new_spikes(hidden_dim);
            
            for (int i = 0; i < hidden_dim; ++i) {
                new_spikes[i] = hidden_neurons[i].step(I[i]);
            }
            spike_history.push_back(new_spikes);
            spikes = new_spikes;
        }
        return W_out * spikes;
    }

    void reset_state() {
        for (auto& neuron : hidden_neurons) neuron.reset();
        spike_history.clear();
    }

    void local_update(const Eigen::VectorXf& x, const Eigen::VectorXf& target) {
        Eigen::VectorXf pred = forward(x);
        Eigen::VectorXf error = target - pred;

        if (spike_history.empty()) return;
        Eigen::VectorXf last_spikes = spike_history.back();

        W_out += lr * (error * last_spikes.transpose());

        Eigen::VectorXf trace = Eigen::VectorXf::Zero(hidden_dim);
        for (int t = (int)spike_history.size() - 1; t >= 0; --t) {
            trace = trace * 0.9f + spike_history[t];
        }

        Eigen::VectorXf feedback = W_out.transpose() * error;

        W_in += (lr * 0.1f) * (feedback * x.transpose());
        W_rec += (lr * 0.1f) * (feedback * trace.transpose());

        W_in = W_in.cwiseMax(-1.0f).cwiseMin(1.0f);
        W_rec = W_rec.cwiseMax(-1.0f).cwiseMin(1.0f);
        W_out = W_out.cwiseMax(-1.0f).cwiseMin(1.0f);
    }
};

class GatingNetwork {
public:
    int num_submodels;
    int context_dim;
    Eigen::MatrixXf W1;
    Eigen::VectorXf b1;
    Eigen::MatrixXf W2;
    Eigen::VectorXf b2;
    Eigen::VectorXf last_probs;
    std::vector<bool> last_mask;

    GatingNetwork(int num_submodels, int context_dim, int hidden_dim = 128)
        : num_submodels(num_submodels), context_dim(context_dim) {
        float scale = 0.1f;
        W1 = Eigen::MatrixXf::Random(hidden_dim, context_dim) * scale;
        b1 = Eigen::VectorXf::Zero(hidden_dim);
        W2 = Eigen::MatrixXf::Random(num_submodels, hidden_dim) * scale;
        b2 = Eigen::VectorXf::Zero(num_submodels);
    }

    std::vector<bool> forward(const Eigen::VectorXf& context, int k) {
        Eigen::VectorXf h = (W1 * context + b1).array().tanh();
        Eigen::VectorXf logits = W2 * h + b2;

        float max_logit = logits.maxCoeff();
        Eigen::VectorXf shifted = logits.array() - max_logit;
        Eigen::VectorXf exp_logits = shifted.array().exp();
        Eigen::VectorXf probs = exp_logits / exp_logits.sum();

        std::vector<int> indices(num_submodels);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&probs](int i1, int i2) {
            return probs[i1] > probs[i2];
        });

        std::vector<bool> mask(num_submodels, false);
        for (int i = 0; i < k && i < num_submodels; ++i) {
            mask[indices[i]] = true;
        }

        last_probs = probs;
        last_mask = mask;
        return mask;
    }

    void update_with_context(const Eigen::VectorXf& context, float reward, float lr = 1e-3f) {
        if (last_mask.empty()) return;

        Eigen::VectorXf h = (W1 * context + b1).array().tanh();
        Eigen::VectorXf mask_f(num_submodels);
        for(int i=0; i<num_submodels; ++i) mask_f[i] = last_mask[i] ? 1.0f : 0.0f;

        Eigen::VectorXf grad_logits = (mask_f - last_probs) * reward;

        Eigen::MatrixXf dW2 = grad_logits * h.transpose();
        Eigen::VectorXf db2 = grad_logits;

        Eigen::VectorXf dhidden = W2.transpose() * grad_logits;
        Eigen::VectorXf dhidden_raw = dhidden.array() * (1.0f - h.array().square());

        Eigen::MatrixXf dW1 = dhidden_raw * context.transpose();
        Eigen::VectorXf db1 = dhidden_raw;

        W2 += lr * dW2;
        b2 += lr * db2;
        W1 += lr * dW1;
        b1 += lr * db1;
    }
};

/**
 * @struct Experience
 * @brief A snapshot of a memory for offline consolidation (Sleep cycle).
 */
struct Experience {
    int x_idx;
    int next_x_idx;
    Eigen::VectorXf context;
    std::vector<bool> chosen_mask;
    Eigen::VectorXf probs;
    float reward;
};

class BrainLikeModel {
public:
    int vocab_size;
    int embed_dim;
    int context_dim;
    int num_submodels;
    int top_k;

    Eigen::MatrixXf embedding;
    GatingNetwork gating;
    std::vector<SubModel> submodels;
    Eigen::MatrixXf output_W;

    Eigen::VectorXf last_context;
    std::vector<bool> last_chosen_mask;

    Hippocampus hippocampus;
    std::vector<Experience> sleep_buffer;

    float alpha_context = 0.2f; // EMA factor for continuous thought

    BrainLikeModel(int vocab_size, int embed_dim = 128, int context_dim = 128, 
                   int num_submodels = 100, int submodel_hidden = 64, 
                   int submodel_steps = 10, int top_k = 10)
        : vocab_size(vocab_size), embed_dim(embed_dim), context_dim(context_dim),
          num_submodels(num_submodels), top_k(top_k), 
          gating(num_submodels, context_dim)
    {
        embedding = Eigen::MatrixXf::Random(vocab_size, embed_dim) * 0.1f;
        output_W = Eigen::MatrixXf::Random(vocab_size, embed_dim) * 0.1f;
        last_context = Eigen::VectorXf::Zero(embed_dim);

        submodels.reserve(num_submodels);
        for (int i = 0; i < num_submodels; ++i) {
            submodels.emplace_back(i, embed_dim, submodel_hidden, submodel_steps);
        }
    }

    Eigen::VectorXf embed_input(int x_idx) {
        return embedding.row(x_idx).transpose();
    }

    int forward(int x_idx, Eigen::VectorXf& out_logits) {
        Eigen::VectorXf x = embed_input(x_idx);
        
        // 1. Accumulate continuous Working Memory
        if (last_context.norm() < 1e-6f) {
            last_context = x;
        } else {
            last_context = (1.0f - alpha_context) * last_context + alpha_context * x;
        }

        // 2. Query Episodic Memory (Hippocampus)
        Eigen::VectorXf retrieved_mem = hippocampus.retrieve(last_context);
        Eigen::VectorXf augmented_context = last_context + (retrieved_mem * 0.5f);

        // 3. Select active modules
        last_chosen_mask = gating.forward(augmented_context, top_k);

        Eigen::VectorXf combined = Eigen::VectorXf::Zero(embed_dim);
        int active_count = 0;

        for (int i = 0; i < num_submodels; ++i) {
            if (last_chosen_mask[i]) {
                combined += submodels[i].forward(x);
                active_count++;
            }
        }

        if (active_count > 0) combined /= (float)active_count;

        out_logits = output_W * combined;
        
        int predicted_token;
        out_logits.maxCoeff(&predicted_token);
        return predicted_token;
    }

    /**
     * @brief Dynamically expands the vocabulary size.
     * Keeps old weights intact, initializes new token weights with random noise.
     */
    void expand_vocabulary(int new_vocab_size) {
        if (new_vocab_size <= vocab_size) return;

        int old_size = vocab_size;

        // conservativeResize keeps the existing matrix data intact while growing it
        embedding.conservativeResize(new_vocab_size, embed_dim);
        output_W.conservativeResize(new_vocab_size, embed_dim);

        // Initialize the newly added rows with random noise
        float scale = 0.1f;
        embedding.block(old_size, 0, new_vocab_size - old_size, embed_dim) = 
            Eigen::MatrixXf::Random(new_vocab_size - old_size, embed_dim) * scale;
            
        output_W.block(old_size, 0, new_vocab_size - old_size, embed_dim) = 
            Eigen::MatrixXf::Random(new_vocab_size - old_size, embed_dim) * scale;

        vocab_size = new_vocab_size;
    }

    /**
     * @brief Defers heavy learning to a background thread/sleep cycle
     */
    void buffer_experience(int x_idx, int next_x_idx, float reward) {
        Experience exp;
        exp.x_idx = x_idx;
        exp.next_x_idx = next_x_idx;
        exp.context = last_context;
        exp.chosen_mask = last_chosen_mask;
        exp.probs = gating.last_probs;
        exp.reward = reward;
        sleep_buffer.push_back(exp);
    }

    /**
     * @brief Processes offline memories. Should be run when NPC is sleeping or off-screen.
     */
    void sleep_and_consolidate() {
        if (sleep_buffer.empty()) return;
        
        for (const auto& exp : sleep_buffer) {
            // High salience events go to long-term Episodic memory
            if (exp.reward > 0.5f) {
                hippocampus.add_memory(exp.context, exp.next_x_idx);
            }

            Eigen::VectorXf x = embed_input(exp.x_idx);
            Eigen::VectorXf target = embed_input(exp.next_x_idx);

            // Replay SNN states and run local weight updates
            for (int i = 0; i < num_submodels; ++i) {
                if (exp.chosen_mask[i]) {
                    submodels[i].reset_state();
                    submodels[i].forward(x); // Re-generate spikes for this memory
                    submodels[i].local_update(x, target);
                }
            }

            // Consolidate routing paths
            gating.last_mask = exp.chosen_mask;
            gating.last_probs = exp.probs;
            gating.update_with_context(exp.context, exp.reward);
        }
        sleep_buffer.clear();
        reset_context(); // Sleep wipes short term working memory
    }

    void reset_context() {
        last_context = Eigen::VectorXf::Zero(embed_dim);
        for(auto& sm : submodels) sm.reset_state();
    }

    // --- Helper IO Functions ---
    template<typename T> void write_matrix(std::ofstream& os, const T& m) {
        int rows = m.rows(), cols = m.cols();
        os.write((char*)&rows, sizeof(int));
        os.write((char*)&cols, sizeof(int));
        os.write((char*)m.data(), rows * cols * sizeof(float));
    }

    template<typename T> void read_matrix(std::ifstream& is, T& m) {
        int rows, cols;
        is.read((char*)&rows, sizeof(int));
        is.read((char*)&cols, sizeof(int));
        m.resize(rows, cols);
        is.read((char*)m.data(), rows * cols * sizeof(float));
    }

    void save(const std::string& filename) {
        std::ofstream os(filename, std::ios::binary);
        write_matrix(os, embedding); write_matrix(os, output_W);
        write_matrix(os, gating.W1); write_matrix(os, gating.b1);
        write_matrix(os, gating.W2); write_matrix(os, gating.b2);
        for(auto& sm : submodels) {
            write_matrix(os, sm.W_in); write_matrix(os, sm.W_rec); write_matrix(os, sm.W_out);
        }
    }

    void load(const std::string& filename) {
        std::ifstream is(filename, std::ios::binary);
        if(!is.is_open()) return;
        read_matrix(is, embedding); 
        vocab_size = embedding.rows(); 
        embed_dim = embedding.cols();
        read_matrix(is, output_W);
        read_matrix(is, gating.W1);
        read_matrix(is, gating.b1);
        read_matrix(is, gating.W2);
        read_matrix(is, gating.b2);
        for(auto& sm : submodels) {
            read_matrix(is, sm.W_in); read_matrix(is, sm.W_rec); read_matrix(is, sm.W_out);
        }
    }
};

class SimpleTokenizer {
public:
    std::unordered_map<std::string, int> word2idx;
    std::unordered_map<int, std::string> idx2word;
    int next_idx = 0;

    int get_or_add(const std::string& word) {
        if (word2idx.find(word) == word2idx.end()) {
            word2idx[word] = next_idx;
            idx2word[next_idx] = word;
            next_idx++;
        }
        return word2idx[word];
    }
    
    std::string get_word(int idx) {
        if (idx2word.find(idx) != idx2word.end()) return idx2word[idx];
        return "<UNK>";
    }

    void save(const std::string& filename) {
        std::ofstream os(filename);
        for(auto& pair : word2idx) os << pair.first << " " << pair.second << "\n";
    }

    void load(const std::string& filename) {
        std::ifstream is(filename);
        std::string word; int idx;
        while(is >> word >> idx) {
            word2idx[word] = idx;
            idx2word[idx] = word;
            if(idx >= next_idx) next_idx = idx + 1;
        }
    }
};

#endif