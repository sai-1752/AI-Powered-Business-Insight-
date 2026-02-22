import ollama

def generate_insights(summary):

    prompt = f"""
    You are a business analyst.

    Based on the dataset summary below,
    provide:

    1. Key trends
    2. Potential risks
    3. Business opportunities
    4. Actionable recommendations

    Dataset Summary:
    {summary}
    """

    response = ollama.chat(
        model="phi",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']