import streamlit as st
import pandas as pd
import numpy as np
import pickle
import itertools
from collections import defaultdict

st.set_page_config(page_title="CSAO Production System", layout="wide")

st.title("🍽️ Cart Super Add-On Recommendation System (Production Demo)")

# ==========================
# LOAD MODEL
# ==========================
model = pickle.load(open("model.pkl", "rb"))





import time

start = time.time()
# prediction code
end = time.time()

print("Latency:", end - start)


# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("realistic_csao_dataset_60k.csv")

item_features = df[["item_id", "item_name", "category", "price"]].drop_duplicates()
all_items = item_features["item_id"].unique()

# ==========================
# BUILD CO-OCCURRENCE MATRIX
# ==========================
co_occurrence = defaultdict(int)

for order_id, group in df.groupby("order_id"):
    items = group["item_id"].tolist()
    for item1, item2 in itertools.permutations(items, 2):
        co_occurrence[(item1, item2)] += 1

item_counts = df["item_id"].value_counts().to_dict()

def get_cooccurrence_score(item_a, item_b):
    pair_count = co_occurrence.get((item_a, item_b), 0)
    item_count = item_counts.get(item_a, 1)
    return pair_count / item_count

# ==========================
# SIDEBAR INPUT
# ==========================
st.sidebar.header("🛒 Cart Input")

item_options = item_features.set_index("item_id")["item_name"].to_dict()

# 🔥 MULTISELECT CART (IMPORTANT CHANGE)
cart_items = st.sidebar.multiselect(
    "Select Items in Cart",
    options=list(item_options.keys()),
    format_func=lambda x: item_options[x]
)

cart_size = len(cart_items)

user_id = st.sidebar.number_input("User ID", 1, 8000, 1)
restaurant_id = st.sidebar.number_input("Restaurant ID", 1, 500, 1)

# ==========================
# GENERATE RECOMMENDATIONS
# ==========================
if st.sidebar.button("Generate Recommendations"):

    if len(cart_items) == 0:
        st.warning("Please select at least one item in cart.")
        st.stop()

    candidate_rows = []

    for item in all_items:

        # Skip items already in cart
        if item in cart_items:
            continue

        item_info = item_features[item_features["item_id"] == item].iloc[0]

        # 🔥 NEW: Score against FULL CART
        co_scores = [
            get_cooccurrence_score(cart_item, item)
            for cart_item in cart_items
        ]

        avg_co_score = np.mean(co_scores) if len(co_scores) > 0 else 0

        candidate_rows.append({
            "user_id": user_id,
            "restaurant_id": restaurant_id,
            "cart_size": cart_size,
            "candidate_item": item,
            "price": item_info["price"],
            "cooccurrence_score": avg_co_score,
            "candidate_popularity": item_counts.get(item, 0),
            "category": item_info["category"]
        })

    candidates_df = pd.DataFrame(candidate_rows)

    # One-hot encode category
    candidates_df = pd.get_dummies(candidates_df, columns=["category"])

    # Align with model features
    model_features = model.feature_name_

    for col in model_features:
        if col not in candidates_df.columns:
            candidates_df[col] = 0

    candidates_df = candidates_df[model_features]

    # Predict
    probs = model.predict_proba(candidates_df)[:, 1]
    candidates_df["probability"] = probs

    top8 = candidates_df.sort_values("probability", ascending=False).head(8)

    st.subheader("🔥 Top 8 Recommended Add-ons")

    for _, row in top8.iterrows():

        item_info = item_features[item_features["item_id"] == row["candidate_item"]].iloc[0]

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"""
            ### 🍽️ {item_info['item_name']}
            **Category:** {item_info['category']}  
            **Price:** ₹{item_info['price']}
            """)

        with col2:
            st.progress(float(row["probability"]))
            st.write(f"Score: {row['probability']:.3f}")

        st.markdown("---")