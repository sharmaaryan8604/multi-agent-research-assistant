
import os
import streamlit as st
from dotenv import load_dotenv

# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

# Support Streamlit Cloud secrets.

try:
    for key in [
        "GROQ_API_KEY",
        "TAVILY_API_KEY",
        "GROQ_MODEL"
    ]:

        if key in st.secrets and not os.getenv(key):

            os.environ[key] = str(st.secrets[key])

except Exception:
    pass


# Import after loading environment variables.

from research_graph import run_research


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="ResearchAI",

    page_icon="🔎",

    layout="wide"

)


# =========================================================
# 3. CUSTOM UI
# =========================================================

st.title("🔎 ResearchAI")

st.markdown(
    """
    ### Multi-Agent Research Assistant

    A collaborative AI system that searches, reads,
    writes and critiques research automatically.
    """
)

st.divider()


# =========================================================
# 4. SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.markdown(
        """
        **Powered by**

        - Groq LLM
        - Tavily Search
        - LangChain LCEL
        - LangGraph
        """
    )

    st.divider()

    max_sources = st.slider(

        "Number of sources",

        min_value=2,

        max_value=5,

        value=2

    )

    st.divider()

    st.markdown(
        """
        **Research Agents**

        🔎 Search Agent

        📚 Reader Agent

        ✍️ Writer Agent

        🧪 Critic Agent
        """
    )

    st.divider()

    st.caption(
        "Research quality depends on source availability, "
        "API limits and model performance."
    )


# =========================================================
# 5. MAIN INPUT
# =========================================================

st.subheader("What would you like to research?")

topic = st.text_area(

    "Enter your research topic",

    placeholder=(
        "Example: How will Generative AI impact "
        "software engineering jobs by 2030?"
    ),

    height=130

)


# =========================================================
# 6. GENERATE BUTTON
# =========================================================

if st.button(
    "🚀 Generate Research Report",
    type="primary",
    use_container_width=True
):

    if not topic.strip():

        st.warning(
            "Please enter a research topic."
        )

    elif not os.getenv("GROQ_API_KEY"):

        st.error(
            "GROQ_API_KEY is missing. "
            "Check your .env file."
        )

    elif not os.getenv("TAVILY_API_KEY"):

        st.error(
            "TAVILY_API_KEY is missing. "
            "Check your .env file."
        )

    else:

        try:

            # =========================================
            # RUN LANGGRAPH
            # =========================================

            with st.spinner(
                "🤖 Your AI research team is working..."
            ):

                result = run_research(

                    topic=topic.strip(),

                    max_sources=max_sources

                )

            st.success(
                "Research report generated successfully!"
            )

            # =========================================
            # REPORT
            # =========================================

            st.divider()

            st.header("📄 Research Report")

            st.markdown(
                result.get(
                    "report",
                    "No report generated."
                )
            )

            # =========================================
            # DOWNLOAD REPORT
            # =========================================

            st.download_button(

                label="⬇️ Download Research Report",

                data=result.get("report", ""),

                file_name="research_report.md",

                mime="text/markdown",

                use_container_width=True

            )

            # =========================================
            # CRITIC EVALUATION
            # =========================================

            st.divider()

            st.header("🧪 Critic Evaluation")

            score = result.get("score", 0)

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Research Quality Score",
                    f"{score}/10"
                )

            with col2:

                st.metric(
                    "Revision Attempts",
                    result.get("revision_count", 0)
                )

            st.markdown("**Critic Feedback**")

            st.write(
                result.get(
                    "critique",
                    "No feedback available."
                )
            )

            # =========================================
            # RESEARCH SOURCES
            # =========================================

            st.divider()

            st.header("🔗 Research Sources")

            sources = result.get(
                "sources",
                []
            )

            for source in sources:

                with st.expander(
                    f"[{source['id']}] {source['title']}"
                ):

                    st.markdown(
                        f"[Open Source]({source['url']})"
                    )

                    st.markdown("**Extracted Evidence**")

                    st.write(
                        source.get(
                            "summary",
                            "No summary available."
                        )
                    )

            # =========================================
            # SEARCH QUERIES
            # =========================================

            with st.expander("🔎 Search Queries"):

                for query in result.get(
                    "queries",
                    []
                ):

                    st.write(f"- {query}")

        except Exception as error:

            st.error(
                "Research execution failed."
            )

            st.exception(error)


# =========================================================
# 7. FOOTER
# =========================================================

st.divider()

st.caption(
    """
    ResearchAI | Python | LangChain | LangGraph |
    Groq | Tavily | Streamlit
    """
)
