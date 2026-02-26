🍽️ Cart Super Add-On (CSAO) Recommendation System
📌 Project Overview

This project implements a production-style Cart Super Add-On (CSAO) Recommendation System inspired by large-scale food delivery platforms like Zomato.

The system dynamically recommends relevant add-on items (e.g., beverages, desserts, starters) based on the user’s current cart composition, contextual signals, and historical co-occurrence patterns.

The objective is to increase Average Order Value (AOV) while maintaining high recommendation relevance and low inference latency (<300ms).

🎯 Problem Statement

In food delivery ecosystems, customers often miss complementary items that enhance their meal (e.g., drink with pizza, dessert after biryani).

The challenge is to build a real-time recommendation engine that:

Updates dynamically as the cart changes

Suggests context-aware add-ons

Maximizes acceptance rate

Maintains low latency for high-traffic systems

Scales to millions of daily requests

🧠 Solution Approach

The system uses a ranking-based machine learning pipeline:

1️⃣ Feature Engineering

Full cart context modeling (not just last item)

Item co-occurrence probability

Candidate item popularity

Price & category features

Cart size feature

2️⃣ Ranking Model

LightGBM (Gradient Boosted Trees)

Class imbalance handling

Optimized hyperparameters

3️⃣ Real-Time Recommendation Flow
User Updates Cart
        ↓
Candidate Generation
        ↓
Feature Engineering
        ↓
LightGBM Ranking
        ↓
Top-8 Recommendations Displayed
📊 Model Performance

AUC: ~0.86+

Strong Precision@8 performance

Context-aware ranking

Sub-200ms inference latency (local testing)

📈 Business Impact Simulation

Based on attach-rate improvement simulation:

+4–6% increase in add-on acceptance

Significant projected AOV lift

Improved meal completion experience

This approach directly supports revenue growth without harming cart-to-order conversion.

🛠️ Tech Stack

Python

Pandas & NumPy

LightGBM

Scikit-learn

Streamlit

Git & GitHub

⚡ Production Considerations

Co-occurrence matrix can be precomputed and cached

Model can be deployed as REST API

Feature pipeline can be refreshed daily

Designed to meet <300ms inference SLA

🚀 Future Improvements

User embedding personalization

Two-stage retrieval + ranking architecture

Deep learning ranking models

Time-of-day contextual modeling

Cloud deployment with autoscaling

🏁 Conclusion

This project demonstrates a scalable, production-oriented approach to contextual add-on recommendation systems in high-throughput food delivery platforms.

It balances:

Model accuracy

Business impact

System latency

Scalability
