import os
from pathlib import Path
import streamlit as st
import pandas as pd
import requests

# Set page config
st.set_page_config(
    page_title="Multimodal Search Engine & Evaluation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main {
        background-color: #0c0d14;
        color: #ffffff;
    }
    .stApp {
        background-color: #0c0d14;
    }
    .user-profile-box {
        background-color: #151724;
        border: 1px solid #23273c;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: bold;
        text-transform: uppercase;
        margin-left: 10px;
        color: #ffffff;
    }
    .badge-purchase {
        background-color: #2ec4b6;
    }
    .badge-rating {
        background-color: #ff9f1c;
    }
    .badge-click {
        background-color: #e71d36;
    }
    .badge-view {
        background-color: #5865f2;
    }
    .metric-container-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 2px solid #2a2e45;
        text-align: center;
    }
    .kpi-card {
        background: linear-gradient(135deg, #171926 0%, #0f101a 100%);
        border: 1px solid #22263a;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        margin-bottom: 15px;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #00f5d4;
        margin-bottom: 5px;
    }
    .kpi-value-classical {
        color: #ff9f1c;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #9398b3;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .search-column-header {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-weight: bold;
        color: #ffffff;
    }
    .header-vector {
        background: linear-gradient(90deg, #1d4ed8 0%, #0891b2 100%);
        border: 1px solid #2563eb;
    }
    .header-classical {
        background: linear-gradient(90deg, #b45309 0%, #d97706 100%);
        border: 1px solid #d97706;
    }
    .explain-tag {
        font-size: 0.8rem;
        color: #8b92b6;
        background-color: #171926;
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid #25283d;
        display: inline-block;
        margin-right: 5px;
        margin-top: 5px;
    }
    .explain-boost {
        color: #00f5d4;
        font-weight: bold;
    }
    .history-type-badge {
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
        color: white;
    }
    .history-vector {
        background-color: #2563eb;
    }
    .history-classical {
        background-color: #d97706;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"
IMAGE_DIR = Path("data/images")


def check_backend() -> bool:
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# Title header with gradient styling
st.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h1 style="margin-bottom: 0;">🚀 Hybrid Search & Context-Aware Recommender</h1>
    <p style="color: #8b92b6; font-size: 1.1rem;">Side-by-side semantic VSD (FAISS) and classical vocabulary indexing (SPIMI)</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

backend_online = check_backend()

if not backend_online:
    st.error("⚠️ Backend API Server Offline. Please start the backend service by running `python -m src` in your terminal.")
else:
    # --- SIDEBAR: USER & HISTORY CONTEXT ---
    st.sidebar.title("👤 Active User Profile")

    # Load unique User IDs dynamically
    user_ids = ["u001"]
    if Path("data/user_interactions.csv").exists():
        interactions_df = pd.read_csv("data/user_interactions.csv")
        user_ids = sorted(list(interactions_df["user_id"].unique().astype(str)))

    selected_user = st.sidebar.selectbox(
        "Select Active User",
        user_ids,
        help="Select a user profile to personalize recommendation scores based on their history."
    )

    # Fetch User Interaction History details
    user_history_list = []
    try:
        user_history_response = requests.get(f"{API_URL}/user-history/{selected_user}")
        if user_history_response.status_code == 200:
            user_data = user_history_response.json()
            total_int = user_data.get("total_interactions", 0)
            user_history_list = user_data.get("history", [])

            st.sidebar.markdown(f"""
            <div class="user-profile-box">
                <h5 style="margin-top:0;">Profile: {selected_user}</h5>
                <p style="font-size:0.9rem; margin-bottom:5px;"><b>Logged Actions:</b> {total_int}</p>
            </div>
            """, unsafe_allow_html=True)

            breakdown = user_data.get("breakdown", {})
            if breakdown:
                st.sidebar.markdown("**Interaction Breakdown:**")
                for itype, count in breakdown.items():
                    st.sidebar.progress(min(count / max(breakdown.values()), 1.0), text=f"{itype.capitalize()}: {count}")
    except Exception:
        st.sidebar.caption("Could not load user activity summary.")

    # Sidebar User History Timeline
    if user_history_list:
        with st.sidebar.expander("🕒 User Past Activity History"):
            hist_rows = []
            for h in user_history_list[:15]:  # show top 15 recent actions
                item_id = h["item_id"]
                action = h["interaction_type"].upper()
                rating_str = f"⭐ {h['rating']}" if h.get("rating") else ""
                t_str = h["timestamp"][:10]  # Date only
                
                # Fetch title from catalog
                title = f"Product #{item_id}"
                if Path("data/mock_catalog.csv").exists():
                    cat_row = interactions_df[interactions_df["item_id"] == int(item_id)]
                    # We look up catalog title
                    cat_df = pd.read_csv("data/mock_catalog.csv")
                    matched = cat_df[cat_df["item_id"].astype(str) == str(item_id)]
                    if not matched.empty:
                        title = matched.iloc[0]["title"]
                
                hist_rows.append({"Action": action, "Product": title, "Rating": rating_str, "Date": t_str})
            
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    # --- MAIN TABS ---
    tab_search, tab_analyst = st.tabs(["🔎 Side-by-Side Search", "📊 Performance Evaluation Hub (TryRating)"])

    # --- TAB 1: SEARCH & SIDE-BY-SIDE RECOMMEND ---
    with tab_search:
        st.subheader("Multimodal Query Entrance")

        col_text, col_img = st.columns([2, 1])
        with col_text:
            text_query = st.text_input(
                "Enter text keywords:",
                placeholder="e.g. Compact Mug, Performance Earbuds, Classic Shoes...",
                help="Enter search queries to evaluate matching keywords."
            )

        with col_img:
            uploaded_image = st.file_uploader(
                "Upload image input:",
                type=["jpg", "png", "jpeg"],
                help="Upload an image to perform computer vision visual searches."
            )

        search_clicked = st.button("Run Side-by-Side Search", use_container_width=True, type="primary")

        if search_clicked:
            # 1. Fetch Vector search results
            vector_results = []
            if text_query or uploaded_image:
                search_endpoint = f"{API_URL}/search"
                files = None
                data = {
                    "user_id": selected_user,
                    "query": text_query or "",
                    "search_type": "vector"
                }
                if uploaded_image is not None:
                    files = {"image": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                
                with st.spinner("Processing deep learning embeddings & FAISS index..."):
                    try:
                        resp = requests.post(search_endpoint, data=data, files=files)
                        if resp.status_code == 200:
                            vector_results = resp.json().get("results", [])
                    except Exception as e:
                        st.error(f"Vector search request failed: {e}")

            # 2. Fetch Classical search results (text-only)
            classical_results = []
            if text_query:
                search_endpoint = f"{API_URL}/search"
                data = {
                    "user_id": selected_user,
                    "query": text_query,
                    "search_type": "classical"
                }
                try:
                    resp = requests.post(search_endpoint, data=data)
                    if resp.status_code == 200:
                        classical_results = resp.json().get("results", [])
                except Exception as e:
                    st.error(f"Classical search request failed: {e}")

            # 3. Render side-by-side columns
            col_vector_view, col_classical_view = st.columns(2)

            # Left Column: Semantic Vector Search
            with col_vector_view:
                st.markdown('<div class="search-column-header header-vector">🤖 SEMANTIC VECTOR SEARCH (FAISS)</div>', unsafe_allow_html=True)
                
                if not vector_results:
                    st.info("No matches found for vector search. Provide text or image queries.")
                else:
                    for rank, item in enumerate(vector_results, start=1):
                        item_id = str(item["item_id"])
                        title = item["title"]
                        desc = item["description"]
                        price = item["price"]
                        category = item["category"]
                        score = item["score"]
                        base_val = item.get("base_score", score)
                        boost = item.get("boost", 0.0)
                        img_fname = item["image_filename"]

                        # Recommender history badge
                        badge_html = ""
                        if Path("data/user_interactions.csv").exists():
                            user_int = interactions_df[
                                (interactions_df["user_id"] == selected_user) &
                                (interactions_df["item_id"] == int(item_id))
                            ]
                            if not user_int.empty:
                                action = user_int.iloc[0]["interaction_type"]
                                badge_html = f'<span class="badge badge-{action}">{action.upper()}</span>'

                        with st.container(border=True):
                            col_c_img, col_c_details = st.columns([1, 3])
                            with col_c_img:
                                img_path = IMAGE_DIR / img_fname
                                if img_path.exists():
                                    st.image(str(img_path), use_container_width=True)
                                else:
                                    st.caption("📷 No Image")
                            
                            with col_c_details:
                                st.markdown(f"##### #{item_id}: {title} {badge_html}", unsafe_allow_html=True)
                                st.markdown(f"<span style='font-size:0.85rem; color:#b2bccd;'>{desc}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Category:** {category} | **Price:** `${price}`")
                                
                                # Explainable scores
                                sim_pct = min(max(base_val * 100, 0), 100)
                                st.markdown(f"""
                                <div style="margin-top:5px;">
                                    <span class="explain-tag">Match Similarity: <b>{sim_pct:.1f}%</b></span>
                                    <span class="explain-tag">Recommender Boost: <b class="explain-boost">+{boost:.1f}</b></span>
                                    <span class="explain-tag" style="background-color:#1e293b;">Final Rank score: <b>{score:.3f}</b></span>
                                </div>
                                """, unsafe_allow_html=True)

                                # TryRating Grading Widget
                                st.caption("Human Relevance Grade (Star Rating):")
                                rate_key = f"rate_vec_{item_id}_{text_query}_{uploaded_image.name if uploaded_image else ''}"
                                selected_rating = st.feedback("stars", key=rate_key)
                                if selected_rating is not None:
                                    grade_data = {
                                        "user_id": selected_user,
                                        "item_id": item_id,
                                        "rating": selected_rating + 1,
                                        "query_text": text_query or "",
                                        "image_path": str(IMAGE_DIR / img_fname) if uploaded_image else ""
                                    }
                                    try:
                                        grade_resp = requests.post(f"{API_URL}/grade", data=grade_data)
                                        if grade_resp.status_code == 200:
                                            st.toast(f"Graded Vector Result #{item_id} as {selected_rating+1} stars!", icon="⭐")
                                    except Exception:
                                        pass

            # Right Column: Classical Boolean (SPIMI) Search
            with col_classical_view:
                st.markdown('<div class="search-column-header header-classical">📜 CLASSICAL SPIMI INDEX SEARCH</div>', unsafe_allow_html=True)
                
                if uploaded_image and not text_query:
                    st.warning("⚠️ Classical SPIMI Boolean search only indexes textual metadata and does not support image uploads.")
                elif not text_query:
                    st.info("Enter a text search query keywords to display classical exact matches.")
                elif not classical_results:
                    st.info("No matches found in inverted index for the given search keywords.")
                else:
                    for rank, item in enumerate(classical_results, start=1):
                        item_id = str(item["item_id"])
                        title = item["title"]
                        desc = item["description"]
                        price = item["price"]
                        category = item["category"]
                        score = item["score"]
                        base_val = item.get("base_score", score)
                        boost = item.get("boost", 0.0)
                        img_fname = item["image_filename"]

                        # Recommender history badge
                        badge_html = ""
                        if Path("data/user_interactions.csv").exists():
                            user_int = interactions_df[
                                (interactions_df["user_id"] == selected_user) &
                                (interactions_df["item_id"] == int(item_id))
                            ]
                            if not user_int.empty:
                                action = user_int.iloc[0]["interaction_type"]
                                badge_html = f'<span class="badge badge-{action}">{action.upper()}</span>'

                        with st.container(border=True):
                            col_c_img, col_c_details = st.columns([1, 3])
                            with col_c_img:
                                img_path = IMAGE_DIR / img_fname
                                if img_path.exists():
                                    st.image(str(img_path), use_container_width=True)
                                else:
                                    st.caption("📷 No Image")
                            
                            with col_c_details:
                                st.markdown(f"##### #{item_id}: {title} {badge_html}", unsafe_allow_html=True)
                                st.markdown(f"<span style='font-size:0.85rem; color:#b2bccd;'>{desc}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Category:** {category} | **Price:** `${price}`")
                                
                                # Explainable scores
                                overlap_pct = min(base_val * 100, 100)
                                st.markdown(f"""
                                <div style="margin-top:5px;">
                                    <span class="explain-tag">Token Overlap: <b>{overlap_pct:.0f}%</b></span>
                                    <span class="explain-tag">Recommender Boost: <b class="explain-boost">+{boost:.1f}</b></span>
                                    <span class="explain-tag" style="background-color:#1e293b;">Final Rank score: <b>{score:.3f}</b></span>
                                </div>
                                """, unsafe_allow_html=True)

                                # TryRating Grading Widget
                                st.caption("Human Relevance Grade (Star Rating):")
                                rate_key = f"rate_class_{item_id}_{text_query}"
                                selected_rating = st.feedback("stars", key=rate_key)
                                if selected_rating is not None:
                                    grade_data = {
                                        "user_id": selected_user,
                                        "item_id": item_id,
                                        "rating": selected_rating + 1,
                                        "query_text": text_query,
                                        "image_path": ""
                                    }
                                    try:
                                        grade_resp = requests.post(f"{API_URL}/grade", data=grade_data)
                                        if grade_resp.status_code == 200:
                                            st.toast(f"Graded Classical Result #{item_id} as {selected_rating+1} stars!", icon="⭐")
                                    except Exception:
                                        pass

    # --- TAB 2: ANALYST EVALUATION HUB (TRYRATING) ---
    with tab_analyst:
        st.subheader("Relevance Evaluation & Algorithm Benchmarks")

        # KPI Columns: Compare classical vs vector performance side-by-side
        metrics_endpoint = f"{API_URL}/metrics"
        try:
            metrics_resp = requests.get(metrics_endpoint)
            if metrics_resp.status_code == 200:
                metrics_data = metrics_resp.json()
                
                # Fetch scores
                v_ndcg = metrics_data.get("vector", {}).get("NDCG@10", 0.0)
                v_map = metrics_data.get("vector", {}).get("MAP", 0.0)
                v_count = metrics_data.get("vector", {}).get("count", 0)

                c_ndcg = metrics_data.get("classical", {}).get("NDCG@10", 0.0)
                c_map = metrics_data.get("classical", {}).get("MAP", 0.0)
                c_count = metrics_data.get("classical", {}).get("count", 0)

                col_v_metrics, col_c_metrics = st.columns(2)
                
                # Left Metrics Column: Vector Search Results
                with col_v_metrics:
                    st.markdown('<div class="metric-container-title" style="color:#00f5d4;">🤖 Semantic Vector Performance</div>', unsafe_allow_html=True)
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-value">{v_ndcg:.4f}</div>
                            <div class="kpi-label">NDCG @ 10</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-value">{v_map:.4f}</div>
                            <div class="kpi-label">MAP Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m3:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-value">{v_count}</div>
                            <div class="kpi-label">Graded Queries</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Right Metrics Column: Classical Inverted index results
                with col_c_metrics:
                    st.markdown('<div class="metric-container-title" style="color:#ff9f1c;">📜 Classical SPIMI Performance</div>', unsafe_allow_html=True)
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-value kpi-value-classical">{c_ndcg:.4f}</div>
                            <div class="kpi-label">NDCG @ 10</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-value kpi-value-classical">{c_map:.4f}</div>
                            <div class="kpi-label">MAP Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m3:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-value kpi-value-classical">{c_count}</div>
                            <div class="kpi-label">Graded Queries</div>
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Failed to fetch system metrics: {e}")

        # Metrics documentation expander
        with st.expander("ℹ️ How are NDCG@10 and MAP benchmarked?"):
            st.markdown("""
            *   **NDCG@10 (Normalized Discounted Cumulative Gain at Rank 10)**: 
                NDCG evaluates the relevance and exact ranking structure. A score of `1.0000` indicates that the engine ranked highly-graded relevant items exactly at the top. The score discounts grades (1-5 stars) logarithmically for lower ranks.
            *   **MAP (Mean Average Precision)**:
                MAP measures precision across binary relevance categories. Items with a grade of $\ge 3$ stars are treated as relevant. MAP calculates the mean average precision across all queries evaluated.
            """)

        # Historic searches
        st.markdown("### 📋 Query Logs & Audit Portal")
        try:
            history_resp = requests.get(f"{API_URL}/search-history")
            if history_resp.status_code == 200:
                history_data = history_resp.json().get("history", [])

                if not history_data:
                    st.info("No query logs found. Perform some searches to populate this section.")
                else:
                    for search in history_data[:8]:
                        sid = search["search_id"]
                        uid = search["user_id"]
                        qtxt = search["query_text"]
                        imgpath = search["image_path"]
                        stype = search["search_type"]
                        time_str = search["timestamp"]

                        query_desc = qtxt if qtxt else ""
                        if imgpath:
                            query_desc += f" [Image: {Path(imgpath).name}]"

                        # expansion expander
                        method_badge = "VECTOR" if "vector" in stype else "CLASSICAL"
                        badge_class = "history-vector" if "vector" in stype else "history-classical"

                        with st.expander(f"Search #{sid}: User {uid} | Query: '{query_desc}' | Method: {stype} ({time_str})"):
                            col_h_details, col_h_img = st.columns([3, 1])

                            with col_h_details:
                                try:
                                    results_resp = requests.get(f"{API_URL}/search-results/{sid}")
                                    if results_resp.status_code == 200:
                                        search_results = results_resp.json().get("results", [])
                                        if search_results:
                                            st.markdown("**Returned Candidates:**")
                                            for res in search_results:
                                                res_item_id = str(res["item_id"])
                                                res_title = res["title"]
                                                res_score = res["score"]
                                                res_price = res["price"]

                                                st.markdown(f"**#{res_item_id}**: {res_title} (Price: ${res_price}, Score: {res_score:.4f})")
                                                
                                                # Inline grading inside history
                                                history_rate_key = f"hist_rate_{sid}_{res_item_id}_log"
                                                selected_rating = st.feedback("stars", key=history_rate_key)
                                                if selected_rating is not None:
                                                    grade_data = {
                                                        "user_id": uid,
                                                        "item_id": res_item_id,
                                                        "rating": selected_rating + 1,
                                                        "query_text": qtxt or "",
                                                        "image_path": imgpath or ""
                                                    }
                                                    grade_resp = requests.post(f"{API_URL}/grade", data=grade_data)
                                                    if grade_resp.status_code == 200:
                                                        st.toast(f"Logged rating {selected_rating + 1} for item #{res_item_id}!", icon="✅")
                                        else:
                                            st.warning("No results recorded for this search.")
                                except Exception as e:
                                    st.error(f"Error loading historical results: {e}")

                            with col_h_img:
                                if imgpath and Path(imgpath).exists():
                                    st.image(imgpath, use_container_width=True, caption="Query Image")

        except Exception as e:
            st.error(f"Failed to load search logs: {e}")
