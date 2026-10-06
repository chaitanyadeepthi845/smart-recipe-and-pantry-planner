"""
Smart Recipe & Pantry Planner
-----------------------------
Streamlit + LangChain (Ollama) app that:
  * tracks pantry inventory (with simulated smart-shelf sensor data)
  * warns about expiring / low-stock items
  * suggests recipes from what you already have
  * generates AI recipes and weekly meal plans with a local Ollama model
  * builds a shopping list automatically

Run:  streamlit run app.py
"""

import random
from datetime import date, timedelta

import pandas as pd
import streamlit as st

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    LLM_AVAILABLE = True
except ImportError:  # app still works without the LLM libraries
    LLM_AVAILABLE = False

st.set_page_config(page_title="Smart Recipe & Pantry Planner", page_icon="🍳", layout="wide")

# --------------------------------------------------------------------------
# Sample / simulated data (stands in for a live IoT smart-pantry feed)
# --------------------------------------------------------------------------
TODAY = date.today()


def sample_pantry() -> pd.DataFrame:
    rows = [
        ("Rice", "Grains", 2.0, "kg", 300, 0.5),
        ("Dal", "Grains", 1.0, "kg", 200, 0.5),
        ("Wheat Flour", "Grains", 1.5, "kg", 120, 0.5),
        ("Pasta", "Grains", 0.4, "kg", 250, 0.5),
        ("Milk", "Dairy", 1.0, "L", 2, 1.0),
        ("Curd", "Dairy", 0.5, "kg", 4, 0.3),
        ("Paneer", "Dairy", 0.25, "kg", 3, 0.2),
        ("Eggs", "Dairy", 6, "pcs", 9, 4),
        ("Tomato", "Vegetables", 0.6, "kg", 4, 0.3),
        ("Onion", "Vegetables", 1.2, "kg", 20, 0.5),
        ("Potato", "Vegetables", 1.0, "kg", 15, 0.5),
        ("Spinach", "Vegetables", 0.2, "kg", 1, 0.2),
        ("Capsicum", "Vegetables", 0.3, "kg", 5, 0.2),
        ("Garlic", "Vegetables", 0.1, "kg", 25, 0.05),
        ("Cooking Oil", "Essentials", 0.8, "L", 150, 0.3),
        ("Salt", "Essentials", 1.0, "kg", 500, 0.2),
        ("Turmeric", "Spices", 0.1, "kg", 300, 0.05),
        ("Chilli Powder", "Spices", 0.1, "kg", 300, 0.05),
        ("Bread", "Bakery", 1, "loaf", 2, 1),
    ]
    return pd.DataFrame(
        [
            {
                "item": n,
                "category": c,
                "quantity": q,
                "unit": u,
                "expiry": TODAY + timedelta(days=d),
                "min_stock": m,
            }
            for n, c, q, u, d, m in rows
        ]
    )


RECIPES = [
    {"name": "Vegetable Pulao", "ingredients": ["rice", "onion", "tomato", "capsicum", "potato", "cooking oil", "salt", "turmeric"],
     "time": 35, "diet": "Vegetarian",
     "steps": ["Wash rice and soak for 15 min.", "Saute onion, add chopped veggies and spices.",
               "Add rice and 2x water, cook until fluffy."]},
    {"name": "Dal Tadka", "ingredients": ["dal", "onion", "tomato", "garlic", "turmeric", "chilli powder", "cooking oil", "salt"],
     "time": 30, "diet": "Vegan",
     "steps": ["Pressure cook dal with turmeric and salt.", "Prepare tadka with garlic, onion, tomato and chilli powder.",
               "Mix tadka into dal and simmer 5 min."]},
    {"name": "Paneer Bhurji", "ingredients": ["paneer", "onion", "tomato", "capsicum", "turmeric", "chilli powder", "cooking oil", "salt"],
     "time": 20, "diet": "Vegetarian",
     "steps": ["Saute onion, tomato and capsicum.", "Add spices and crumbled paneer.", "Cook 5 min and serve with bread/roti."]},
    {"name": "Egg Fried Rice", "ingredients": ["rice", "eggs", "onion", "capsicum", "garlic", "cooking oil", "salt"],
     "time": 20, "diet": "Non-Vegetarian",
     "steps": ["Scramble eggs and set aside.", "Stir-fry garlic, onion and capsicum on high heat.",
               "Add cooked rice and eggs, season and toss."]},
    {"name": "Aloo Paratha", "ingredients": ["wheat flour", "potato", "onion", "chilli powder", "salt", "cooking oil"],
     "time": 40, "diet": "Vegan",
     "steps": ["Knead dough with flour, water and salt.", "Mash boiled potato with spices for filling.",
               "Stuff, roll and cook on a tawa with oil."]},
    {"name": "Tomato Pasta", "ingredients": ["pasta", "tomato", "onion", "garlic", "cooking oil", "salt", "chilli powder"],
     "time": 25, "diet": "Vegan",
     "steps": ["Boil pasta until al dente.", "Cook garlic, onion and tomato into a sauce.", "Toss pasta in sauce."]},
    {"name": "Palak Paneer", "ingredients": ["spinach", "paneer", "onion", "tomato", "garlic", "cooking oil", "salt", "chilli powder"],
     "time": 35, "diet": "Vegetarian",
     "steps": ["Blanch and puree spinach.", "Saute onion, garlic and tomato with spices.",
               "Add puree and paneer cubes, simmer 8 min."]},
    {"name": "Masala Omelette", "ingredients": ["eggs", "onion", "tomato", "chilli powder", "salt", "cooking oil"],
     "time": 10, "diet": "Non-Vegetarian",
     "steps": ["Beat eggs with chopped onion, tomato and spices.", "Pour onto a hot oiled pan.", "Cook both sides and serve."]},
    {"name": "Curd Rice", "ingredients": ["rice", "curd", "milk", "salt"],
     "time": 15, "diet": "Vegetarian",
     "steps": ["Mash cooked rice while warm.", "Mix in curd, a little milk and salt.", "Chill and serve."]},
    {"name": "Potato Masala Sandwich", "ingredients": ["bread", "potato", "onion", "turmeric", "salt", "cooking oil"],
     "time": 20, "diet": "Vegan",
     "steps": ["Prepare spiced mashed potato masala.", "Fill between bread slices.", "Toast until golden."]},
]

DIET_OPTIONS = ["Any", "Vegetarian", "Vegan", "Non-Vegetarian"]

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def init_state():
    if "pantry" not in st.session_state:
        st.session_state.pantry = sample_pantry()
    if "ai_recipe" not in st.session_state:
        st.session_state.ai_recipe = ""
    if "meal_plan" not in st.session_state:
        st.session_state.meal_plan = ""


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add days-left, expiry status and stock status columns."""
    out = df.copy()
    out["days_left"] = (pd.to_datetime(out["expiry"]) - pd.Timestamp(TODAY)).dt.days

    def expiry_status(d):
        if d < 0:
            return "Expired"
        if d <= 3:
            return "Expiring soon"
        return "Fresh"

    out["expiry_status"] = out["days_left"].apply(expiry_status)
    out["stock_status"] = out.apply(
        lambda r: "Low" if r["quantity"] <= r["min_stock"] else "OK", axis=1
    )
    return out


def available_items(df: pd.DataFrame) -> set:
    ok = df[(df["quantity"] > 0) & (pd.to_datetime(df["expiry"]) >= pd.Timestamp(TODAY))]
    return set(ok["item"].str.lower().str.strip())


def match_recipes(df: pd.DataFrame, diet: str, max_time: int) -> list:
    have = available_items(df)
    results = []
    for r in RECIPES:
        if diet != "Any":
            # a vegan recipe is also vegetarian
            if not (r["diet"] == diet or (diet == "Vegetarian" and r["diet"] == "Vegan")):
                continue
        if r["time"] > max_time:
            continue
        missing = [i for i in r["ingredients"] if i not in have]
        score = round(100 * (len(r["ingredients"]) - len(missing)) / len(r["ingredients"]))
        results.append({**r, "missing": missing, "score": score})
    return sorted(results, key=lambda x: (-x["score"], x["time"]))


def simulate_sensor_update():
    """Simulate smart-shelf load-cell readings: items get consumed a little."""
    df = st.session_state.pantry.copy()
    for idx in df.index:
        if random.random() < 0.5:
            used = df.at[idx, "quantity"] * random.uniform(0.05, 0.25)
            df.at[idx, "quantity"] = round(max(df.at[idx, "quantity"] - used, 0), 2)
    st.session_state.pantry = df


def pantry_text(df: pd.DataFrame) -> tuple:
    e = enrich(df)
    e = e[(e["quantity"] > 0) & (e["days_left"] >= 0)]
    all_items = ", ".join(f"{r.item} ({r.quantity} {r.unit})" for r in e.itertuples())
    soon = ", ".join(e[e["days_left"] <= 3]["item"]) or "none"
    return all_items, soon


def ask_llm(model: str, template: str, **variables) -> str:
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | ChatOllama(model=model, temperature=0.7) | StrOutputParser()
    return chain.invoke(variables)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
init_state()

st.title("🍳 Smart Recipe & Pantry Planner")
st.caption("Track your pantry, cut food waste, and cook with what you have.")

with st.sidebar:
    st.header("⚙️ Settings")
    diet = st.selectbox("Diet preference", DIET_OPTIONS)
    max_time = st.slider("Max cooking time (min)", 10, 60, 45, step=5)
    servings = st.number_input("Servings", 1, 10, 2)
    st.divider()
    st.subheader("🤖 Ollama (local AI)")
    model_name = st.text_input("Model name", value="llama3.2")
    st.caption("Install Ollama, then run: `ollama pull llama3.2`")
    st.divider()
    st.subheader("📡 Smart pantry sensors")
    st.caption("Simulated data (no physical IoT device).")
    if st.button("Simulate sensor update"):
        simulate_sensor_update()
        st.success("Shelf sensors updated.")
    if st.button("Reset sample data"):
        st.session_state.pantry = sample_pantry()
        st.rerun()

pantry = enrich(st.session_state.pantry)

# top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Items in pantry", len(pantry))
c2.metric("Expiring soon", int((pantry["expiry_status"] == "Expiring soon").sum()))
c3.metric("Expired", int((pantry["expiry_status"] == "Expired").sum()))
c4.metric("Low stock", int((pantry["stock_status"] == "Low").sum()))

tab_pantry, tab_recipes, tab_ai, tab_plan, tab_shop = st.tabs(
    ["🥫 Pantry", "📖 Recipe Matches", "🤖 AI Chef", "📅 Meal Planner", "🛒 Shopping List"]
)

# ---- Pantry tab ----------------------------------------------------------
with tab_pantry:
    st.subheader("Your pantry")
    edited = st.data_editor(
        st.session_state.pantry,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "expiry": st.column_config.DateColumn("expiry"),
            "quantity": st.column_config.NumberColumn("quantity", min_value=0.0, step=0.1),
            "min_stock": st.column_config.NumberColumn("min_stock", min_value=0.0, step=0.1),
        },
        key="pantry_editor",
    )
    st.session_state.pantry = edited.dropna(subset=["item"]).reset_index(drop=True)

    with st.expander("➕ Add an item"):
        with st.form("add_item", clear_on_submit=True):
            a, b, c = st.columns(3)
            name = a.text_input("Item")
            category = b.selectbox("Category", ["Grains", "Dairy", "Vegetables", "Spices", "Essentials", "Bakery", "Other"])
            unit = c.text_input("Unit", value="kg")
            d, e_, f = st.columns(3)
            qty = d.number_input("Quantity", min_value=0.0, value=1.0, step=0.1)
            expiry = e_.date_input("Expiry date", value=TODAY + timedelta(days=7))
            min_stock = f.number_input("Min stock", min_value=0.0, value=0.5, step=0.1)
            if st.form_submit_button("Add") and name.strip():
                new = pd.DataFrame(
                    [{"item": name.strip().title(), "category": category, "quantity": qty,
                      "unit": unit, "expiry": expiry, "min_stock": min_stock}]
                )
                st.session_state.pantry = pd.concat([st.session_state.pantry, new], ignore_index=True)
                st.rerun()

    st.subheader("⚠️ Alerts")
    pantry = enrich(st.session_state.pantry)
    expired = pantry[pantry["expiry_status"] == "Expired"]
    soon = pantry[pantry["expiry_status"] == "Expiring soon"]
    low = pantry[pantry["stock_status"] == "Low"]
    if expired.empty and soon.empty and low.empty:
        st.success("All good! Nothing needs attention.")
    for r in expired.itertuples():
        st.error(f"**{r.item}** has expired — please discard.")
    for r in soon.itertuples():
        st.warning(f"**{r.item}** expires in {r.days_left} day(s) — use it soon.")
    for r in low.itertuples():
        st.info(f"**{r.item}** is running low ({r.quantity} {r.unit} left).")

# ---- Recipe matches tab -----------------------------------------------------
with tab_recipes:
    st.subheader("Recipes you can make")
    matches = match_recipes(st.session_state.pantry, diet, max_time)
    if not matches:
        st.info("No recipes match your filters. Try changing diet or cooking time.")
    for r in matches:
        with st.expander(f"{r['name']} — {r['score']}% match • {r['time']} min • {r['diet']}"):
            st.progress(r["score"] / 100)
            if r["missing"]:
                st.write("**Missing:** " + ", ".join(m.title() for m in r["missing"]))
            else:
                st.success("You have everything!")
            st.write("**Steps**")
            for i, s in enumerate(r["steps"], 1):
                st.write(f"{i}. {s}")

# ---- AI Chef tab -------------------------------------------------------------
with tab_ai:
    st.subheader("AI Chef (LangChain + Ollama)")
    extra = st.text_input("Any special request? (optional)", placeholder="e.g. spicy, high protein, kid-friendly")
    if st.button("Generate recipe"):
        if not LLM_AVAILABLE:
            st.error("LangChain / Ollama packages are not installed.")
        else:
            items, soon_items = pantry_text(st.session_state.pantry)
            template = (
                "You are a helpful home chef. Create ONE recipe for {servings} servings.\n"
                "Pantry items: {items}\n"
                "Use these first because they expire soon: {soon}\n"
                "Diet preference: {diet}. Max cooking time: {max_time} minutes.\n"
                "Special request: {extra}\n"
                "Mainly use pantry items; list any extra ingredients separately.\n"
                "Format: Title, Ingredients, Steps, Cooking time."
            )
            try:
                with st.spinner("Cooking up an idea..."):
                    st.session_state.ai_recipe = ask_llm(
                        model_name, template, servings=servings, items=items, soon=soon_items,
                        diet=diet, max_time=max_time, extra=extra or "none",
                    )
            except Exception as ex:
                st.error(f"Could not reach Ollama ({ex}). Make sure it is running and the model is pulled.")
    if st.session_state.ai_recipe:
        st.markdown(st.session_state.ai_recipe)

# ---- Meal planner tab -----------------------------------------------------------
with tab_plan:
    st.subheader("Weekly meal plan")
    days = st.slider("Number of days", 1, 7, 7)
    use_ai = st.checkbox("Use AI to create the plan", value=False)
    if st.button("Create meal plan"):
        if use_ai and LLM_AVAILABLE:
            items, soon_items = pantry_text(st.session_state.pantry)
            template = (
                "Create a {days}-day meal plan (breakfast, lunch, dinner) for {servings} people.\n"
                "Pantry items: {items}\nUse soon-to-expire items early: {soon}\n"
                "Diet: {diet}. Keep it simple and practical. Output a clean markdown table."
            )
            try:
                with st.spinner("Planning your week..."):
                    st.session_state.meal_plan = ask_llm(
                        model_name, template, days=days, servings=servings,
                        items=items, soon=soon_items, diet=diet,
                    )
            except Exception as ex:
                st.error(f"Could not reach Ollama ({ex}). Showing a rule-based plan instead.")
                use_ai = False
        if not use_ai or not LLM_AVAILABLE:
            ranked = match_recipes(st.session_state.pantry, diet, 60) or RECIPES
            lines = ["| Day | Dinner | Time |", "|---|---|---|"]
            for i in range(days):
                r = ranked[i % len(ranked)]
                lines.append(f"| Day {i + 1} | {r['name']} | {r['time']} min |")
            st.session_state.meal_plan = "\n".join(lines)
    if st.session_state.meal_plan:
        st.markdown(st.session_state.meal_plan)

# ---- Shopping list tab -------------------------------------------------------------
with tab_shop:
    st.subheader("Shopping list")
    pantry = enrich(st.session_state.pantry)
    low_items = pantry[(pantry["stock_status"] == "Low") | (pantry["expiry_status"] == "Expired")]
    shop = [{"item": r.item, "reason": "Expired" if r.expiry_status == "Expired" else "Low stock",
             "suggested_qty": round(max(r.min_stock * 2 - r.quantity, r.min_stock), 2), "unit": r.unit}
            for r in low_items.itertuples()]

    recipe_names = [r["name"] for r in RECIPES]
    picks = st.multiselect("Also shop for these recipes", recipe_names)
    have = available_items(st.session_state.pantry)
    listed = {s["item"].lower() for s in shop}
    for r in RECIPES:
        if r["name"] in picks:
            for ing in r["ingredients"]:
                if ing not in have and ing not in listed:
                    shop.append({"item": ing.title(), "reason": f"For {r['name']}", "suggested_qty": 1, "unit": ""})
                    listed.add(ing)

    if shop:
        shop_df = pd.DataFrame(shop)
        st.dataframe(shop_df, use_container_width=True, hide_index=True)
        st.download_button("Download list (CSV)", shop_df.to_csv(index=False), "shopping_list.csv", "text/csv")
    else:
        st.success("Your pantry is well stocked. Nothing to buy!")