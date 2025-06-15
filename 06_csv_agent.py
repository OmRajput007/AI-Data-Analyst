import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
import io
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.title("📊 AI Data Analyst")

# Load CSV file directly
df = pd.read_csv("Fitness.csv")
st.write("Data Preview:", df.head())

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

query = st.text_input("Ask for data insight or plot")

if query:
    try:
        # Check if the query is asking for a plot
        if any(keyword in query.lower() for keyword in ["plot", "chart", "graph", "visualize", "draw", "show"]):
            # Request code specifically for plotting
            plot_query = f"Generate Python code using matplotlib to {query}. Only return executable code, no explanations."
            code_response = agent.invoke({"input": plot_query})
            code = code_response["output"]
            
            # Clean the code (remove markdown formatting if present)
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            st.subheader("Generated Code:")
            st.code(code)
            
            # Execute the code and display plot
            try:
                # Create a safe execution environment with all necessary imports
                exec_globals = {
                    '__builtins__': __builtins__,
                    'df': df,
                    'pd': pd,
                    'plt': plt,
                    'matplotlib': plt.matplotlib,
                    'numpy': pd.np if hasattr(pd, 'np') else None
                }
                
                # Add numpy import if available
                try:
                    import numpy as np
                    exec_globals['np'] = np
                    exec_globals['numpy'] = np
                except ImportError:
                    pass
                
                exec(code, exec_globals)
                
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                st.subheader("Generated Plot:")
                st.image(buf)
                plt.close()
            except Exception as e:
                st.error(f"Error executing plot code: {e}")
        else:
            # Regular data analysis query
            response = agent.invoke({"input": query})
            st.write(response["output"])
    except Exception as e:
        st.error(f"Error: {e}")