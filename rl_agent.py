import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import random

class RLAgent:
    def __init__(self, df_path="news_subset_2k.csv", emb_path="embeddings_2k.npy"):
        self.df = pd.read_csv(df_path)
        self.embeddings = np.load(emb_path)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        self.n_articles = len(self.df)
        self.embedding_dim = self.embeddings.shape[1]
        self.q_table_path = "q_table.npy"
        self.pref_path = "user_pref_vector.npy"
        self.weight_path = "user_weight.json"

        self.q_table = self.load_or_init_q_table()
        self.user_pref_vector = self.load_or_init_user_vector()
        self.total_weight = self.load_or_init_total_weight()

        self.action_map = {'like': 0, 'share': 1, 'read': 2, 'not_interested': 3}
        self.reward_map = {'like': 1.0, 'share': 1.5, 'read': 0.5, 'not_interested': -1.0}
        self.epsilon = 0.1
        self.seen_articles = set()
        self.interaction_log = []

    def load_or_init_q_table(self):
        if os.path.exists(self.q_table_path):
            return np.load(self.q_table_path)
        return np.zeros((self.n_articles, 4))

    def load_or_init_user_vector(self):
        if os.path.exists(self.pref_path):
            return np.load(self.pref_path)
        return np.zeros(self.embedding_dim)

    def load_or_init_total_weight(self):
        if os.path.exists(self.weight_path):
            with open(self.weight_path, 'r') as f:
                return json.load(f)["total_weight"]
        return 1e-5

    def update_user_pref(self, embedding, action):
        reward = self.reward_map[action]
        if reward > 0:
            self.user_pref_vector = (self.user_pref_vector * self.total_weight + reward * embedding) / (self.total_weight + reward)
            self.total_weight += reward
        elif reward < 0:
            self.user_pref_vector = (self.user_pref_vector * self.total_weight + reward * embedding) / max(self.total_weight + abs(reward), 1e-5)
            self.total_weight += abs(reward)

    def score_article(self, idx):
        sim = cosine_similarity(self.user_pref_vector.reshape(1, -1), self.embeddings[idx].reshape(1, -1))[0][0]
        q_score = np.max(self.q_table[idx])
        return 0.7 * q_score + 0.3 * sim

    def recommend_article(self):
        candidates = [i for i in range(self.n_articles) if i not in self.seen_articles]
        if not candidates:
            return None
        if random.random() < self.epsilon:
            return random.choice(candidates)
        return max(candidates, key=self.score_article)

    def update(self, article_idx, action):
        embedding = self.embeddings[article_idx]
        reward = self.reward_map[action]
        action_idx = self.action_map[action]

        # Update user preference vector
        self.update_user_pref(embedding, action)

        # Q-table update
        current_q = self.q_table[article_idx, action_idx]
        self.q_table[article_idx, action_idx] = current_q + 0.1 * (reward - current_q)

        self.seen_articles.add(article_idx)
        self.interaction_log.append((
            self.df.iloc[article_idx]['title'],
            self.df.iloc[article_idx]['category'],
            action,
            reward
        ))

        self.save_state()

    def save_state(self):
        np.save(self.q_table_path, self.q_table)
        np.save(self.pref_path, self.user_pref_vector)
        with open(self.weight_path, 'w') as f:
            json.dump({"total_weight": self.total_weight}, f)

        pd.DataFrame(self.interaction_log, columns=["Title", "Category", "Action", "Reward"]).to_csv("interaction_log.csv", index=False)
