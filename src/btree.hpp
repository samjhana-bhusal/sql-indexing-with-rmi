#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>

class BPlusTree {
public:
    static const int B = 64; // Max keys in a node (degree)

    struct Node {
        bool is_leaf;
        std::vector<uint64_t> keys;
        Node(bool leaf) : is_leaf(leaf) {
            keys.reserve(B);
        }
        virtual ~Node() {}
    };

    struct LeafNode : public Node {
        std::vector<uint64_t> values;
        LeafNode* next;
        LeafNode() : Node(true), next(nullptr) {
            values.reserve(B);
        }
    };

    struct InternalNode : public Node {
        std::vector<Node*> children;
        InternalNode() : Node(false) {
            children.reserve(B + 1);
        }
        ~InternalNode() {
            for (auto child : children) {
                delete child;
            }
        }
    };

private:
    Node* root;
    size_t num_keys;

    void split_child(InternalNode* parent, int index, InternalNode* child) {
        InternalNode* z = new InternalNode();
        int t = B / 2;
        
        // z gets keys and children starting after index t
        z->keys.assign(child->keys.begin() + t + 1, child->keys.end());
        z->children.assign(child->children.begin() + t + 1, child->children.end());
        
        // Promoted key goes to parent
        parent->keys.insert(parent->keys.begin() + index, child->keys[t]);
        parent->children.insert(parent->children.begin() + index + 1, z);
        
        // child keeps first t keys and t+1 children
        child->keys.erase(child->keys.begin() + t, child->keys.end());
        child->children.erase(child->children.begin() + t + 1, child->children.end());
    }

    void split_leaf(InternalNode* parent, int index, LeafNode* leaf) {
        LeafNode* z = new LeafNode();
        int t = B / 2;
        
        z->keys.assign(leaf->keys.begin() + t, leaf->keys.end());
        leaf->keys.erase(leaf->keys.begin() + t, leaf->keys.end());
        
        z->values.assign(leaf->values.begin() + t, leaf->values.end());
        leaf->values.erase(leaf->values.begin() + t, leaf->values.end());
        
        z->next = leaf->next;
        leaf->next = z;
        
        parent->children.insert(parent->children.begin() + index + 1, z);
        parent->keys.insert(parent->keys.begin() + index, z->keys.front());
    }

    void insert_non_full(Node* node, uint64_t key, uint64_t value) {
        if (node->is_leaf) {
            LeafNode* leaf = static_cast<LeafNode*>(node);
            auto it = std::lower_bound(leaf->keys.begin(), leaf->keys.end(), key);
            int idx = std::distance(leaf->keys.begin(), it);
            leaf->keys.insert(it, key);
            leaf->values.insert(leaf->values.begin() + idx, value);
        } else {
            InternalNode* internal = static_cast<InternalNode*>(node);
            auto it = std::upper_bound(internal->keys.begin(), internal->keys.end(), key);
            int idx = std::distance(internal->keys.begin(), it);
            Node* child = internal->children[idx];
            
            if (child->keys.size() >= B) {
                if (child->is_leaf) {
                    split_leaf(internal, idx, static_cast<LeafNode*>(child));
                } else {
                    split_child(internal, idx, static_cast<InternalNode*>(child));
                }
                if (key >= internal->keys[idx]) {
                    idx++;
                }
            }
            insert_non_full(internal->children[idx], key, value);
        }
    }

    size_t calculate_memory(Node* node) const {
        if (!node) return 0;
        size_t total = 0;
        if (node->is_leaf) {
            LeafNode* leaf = static_cast<LeafNode*>(node);
            total += sizeof(LeafNode);
            total += leaf->keys.capacity() * sizeof(uint64_t);
            total += leaf->values.capacity() * sizeof(uint64_t);
        } else {
            InternalNode* internal = static_cast<InternalNode*>(node);
            total += sizeof(InternalNode);
            total += internal->keys.capacity() * sizeof(uint64_t);
            total += internal->children.capacity() * sizeof(Node*);
            for (auto child : internal->children) {
                total += calculate_memory(child);
            }
        }
        return total;
    }

public:
    BPlusTree() : root(nullptr), num_keys(0) {
        root = new LeafNode();
    }

    ~BPlusTree() {
        delete root;
    }

    void insert(uint64_t key, uint64_t value) {
        Node* r = root;
        if (r->keys.size() >= B) {
            InternalNode* s = new InternalNode();
            root = s;
            s->children.push_back(r);
            if (r->is_leaf) {
                split_leaf(s, 0, static_cast<LeafNode*>(r));
            } else {
                split_child(s, 0, static_cast<InternalNode*>(r));
            }
            insert_non_full(s, key, value);
        } else {
            insert_non_full(r, key, value);
        }
        num_keys++;
    }

    bool lookup(uint64_t key, uint64_t& value) const {
        Node* curr = root;
        while (!curr->is_leaf) {
            InternalNode* internal = static_cast<InternalNode*>(curr);
            auto it = std::upper_bound(internal->keys.begin(), internal->keys.end(), key);
            int idx = std::distance(internal->keys.begin(), it);
            curr = internal->children[idx];
        }
        
        LeafNode* leaf = static_cast<LeafNode*>(curr);
        auto it = std::lower_bound(leaf->keys.begin(), leaf->keys.end(), key);
        if (it != leaf->keys.end() && *it == key) {
            int idx = std::distance(leaf->keys.begin(), it);
            value = leaf->values[idx];
            return true;
        }
        return false;
    }

    size_t get_memory_size() const {
        return calculate_memory(root);
    }
    
    size_t size() const { return num_keys; }
};
