import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="HCTA AI Report Generator v11",
    page_icon="🏆",
    layout="wide"
)

# --- Savant Prompt Template (v11 - Definitive, Untrimmed, and Complete) ---
# This is the final version, incorporating all rounds of expert feedback, with the full dictionary and all examples explicitly included.
SAVANT_PROMPT_TEMPLATE = """
# Gemini, ACT as an expert-level talent assessment analyst and report writer. Your name is "AnalystAI".
# Your task is to generate a concise, insightful, and professional leadership potential summary based on candidate data, incorporating multiple rounds of expert feedback to achieve the highest level of nuance, narrative cohesion, and behavioral description.
# You must adhere to all rules, formats, and interpretation logic provided below without deviation.

# --- REFINED RULES & WRITING STYLE (BASED ON FINAL EXPERT FEEDBACK) ---
# 1.  **Describe Behaviors, NEVER Name Competencies (ABSOLUTE RULE):** In the summary paragraph, you must NEVER use the names of the competencies or factors (e.g., 'People Potential', 'Sociability', 'Drive Potential'). Instead, you MUST translate those scores into their behavioral descriptions from the dictionary.
#     - **WRONG:** "...his lower people potential and lack of sociability."
#     - **RIGHT:** "...his limited ease in social interactions and an inconsistent focus on others’ thoughts and emotions."
# 2.  **Do Not Make Assumptions or Speculate:** You must not predict future outcomes or use speculative phrases. Stick strictly to the behaviors described in the dictionary.
#     - **WRONG:** "...'her lack of focus on results could lead to prioritizing relationships over outcomes.'"
#     - **WRONG:** "...this 'raises concerns' about her ability..."
#     - **RIGHT:** "Her low drive to achieve results may limit her ability to consistently lead teams toward achieving outcomes..."
# 3.  **Elaborate on Moderate Scores:** When a key competency is 'Moderate', describe the behavior. Instead of saying a candidate has "moderate drive," explain what that means using the dictionary text: "...a tendency to approach goals with some motivation but not always consistent follow-through."
# 4.  **Create a Balanced, Holistic Profile:** When describing development areas in the main paragraph, you MUST synthesize 2-3 distinct behavioral weaknesses based on the lowest scores. Do not focus on a single weakness, as this is repetitive and not holistic.
# 5.  **Adopt a Narrative Flow:** Weave the behavioral descriptions together to tell a cohesive story. Conclude the main paragraph with a forward-looking statement about how addressing development areas will unlock potential.
# 6.  **Analyze Nuance and Contradictions:** Identify and explain complex patterns. If a high-level competency is high but an underlying factor is low, explain the implication of that contrast.
# 7.  **Prioritize Impactful Bullet Points:** Select strengths and development areas that have the highest strategic impact on a leadership role.
# 8.  **Core Rules:** Write in the third person, present tense, American English. The paragraph must be under 200 words. Use pronouns matching the `Gender` input. Do not mention AI or assessments.

# --- FORMAT & STRUCTURE (NON-NEGOTIABLE) ---
# 1.  **One-Paragraph Summary:** Start *exactly* with the text from the "Overall Leadership" interpretation from the dictionary.
# 2.  **Bullet Points:** After the paragraph, provide two strengths and two development areas under the headings "Strengths:" and "Development Areas:".

# --- LOGIC & INTERPRETATION ENGINE ---
# 1.  **Score Categorization:** High = 3.5-5.0; Moderate = 2.5-3.49; Low = 1.0-2.49.
# 2.  **Strength/Development Rule:** Scores >= 4.0 are *only* strengths. Scores <= 2.0 are *only* development areas.
# 3.  **BS & TI Mapping:** Use this to find reinforcing or contradictory patterns.

# --- BEHAVIORAL DICTIONARY (USE THIS TEXT EXACTLY) ---
# **Core Competencies:**
# Overall Leadership:
#   - High: Demonstrates high potential with a strong capacity for growth and success in a more complex role.
#   - Moderate: Demonstrates moderate potential with a reasonable capacity for growth and success in a more complex role.
#   - Low: Demonstrates low potential with limited to low capacity for growth and success in a more complex role.
# Reasoning & Problem Solving:
#   - High: Candidate demonstrates a higher-than-average reasoning and problem-solving ability as compared to a group of peers.
#   - Moderate: Candidate demonstrates an average reasoning and problem-solving ability as compared to a group of peers.
#   - Low: Candidate demonstrates a below-average reasoning and problem-solving ability as compared to a group of peers.
# **Business Simulation (BS) Competencies:**
# Steers Changes:
#   - High: Strong ability to recognise and drive change and transformation at an organisational level. Displays strong resilience and strength during adversity and is well equipped to enable buy-in and support.
#   - Moderate: Moderate ability to contribute to organisational change and transformation. Shows resilience during challenging times and can occasionally support others in gaining buy-in.
#   - Low: Limited ability to support change and transformation at an organisational level. Struggles to remain resilient during adversity and has difficulty enabling buy-in and support.
# Manages Stakeholders:
#   - High: Strong ability to develop and nurture relationships with key stakeholders. Actively finds synergies between organisations to ensure positive outcomes. Networks with stakeholders within and outside one’s industry to stay up-to-date about new developments.
#   - Moderate: Moderate ability to maintain and build relationships with key stakeholders. Occasionally identifies synergies between organisations and engages with stakeholders to stay informed of developments.
#   - Low: Limited ability to develop and maintain relationships with stakeholders. Rarely identifies synergies between organisations or engages with external stakeholders to stay informed.
# Drives Results:
#   - High: Strong ability to articulate performance standards and metrics that support the achievement of organisational goals. Ensures a high-performance culture across teams and demonstrates grit in achievement of challenging goals.
#   - Moderate: Moderate ability to articulate performance standards and metrics that contribute to achieving organisational goals. Occasionally supports performance across teams and shows persistence when working towards goals.
#   - Low: Low ability to articulate performance standards and metrics that support organisational goals. Needs development in fostering a high-performance culture and in maintaining persistence when faced with challenging goals.
# Thinks Strategically:
#   - High: Strong ability to balance the achievement of short-term results with creating long-term value and competitive advantage. Successfully translates complex organisational goals into meaningful actions across teams.
#   - Moderate: Moderate ability to balance short-term results with long-term priorities. Occasionally translates organisational goals into meaningful actions across teams.
#   - Low: Low ability to balance short-term performance with long-term value creation. Struggles to translate organisational goals into meaningful team actions.
# Solves Challenges:
#   - High: Strong ability to deal with ambiguous and complex situations, by making tough decisions where necessary. Is comfortable leading in an environment where goals are frequently complex and thrives during periods of uncertainty.
#   - Moderate: Moderate ability to handle some ambiguous and complex situations by making necessary decisions. Shows some confidence in leading through moderately uncertain environments.
#   - Low: Low ability to deal with ambiguity and complexity. Hesitant to make tough decisions and limited confidence in leading through uncertain situations.
# Develops Talent:
#   - High: Strong ability to leverage and nurture individual strengths to achieve positive outcomes. Actively fosters a culture of learning and advocates for career advancement opportunities within the organisation.
#   - Moderate: Moderate ability to recognise and utilise individual strengths to support positive outcomes. Supports learning and contributes to career development within the organisation.
#   - Low: Low ability to identify and leverage individual strengths. Rarely supports learning or advocates for career development within the organisation.
# **Thriving Index (TI) Potentials:**
# Drive Potential:
#   - High: Consistently demonstrates a positive mindset and motivation; regularly takes initiative to exceed expectations with a strong drive to achieve goals, targets, and results. Seeks fulfillment through impact.
#   - Moderate: Shows a generally positive mindset and some motivation; occasionally takes initiative and shows a drive to achieve goals, but may need support. Interest in making an impact is present but not sustained.
#   - Low: Demonstrates limited motivation or initiative; may meet expectations but does not show a consistent drive to exceed them. Fulfillment from work or desire to make an impact is not clearly evident.
# Learning Potential:
#   - High: Consistently takes time to focus on personal and professional growth - for both self and others. Actively pursues continuous improvement and excellence; shows clear willingness to learn and unlearn.
#   - Moderate: Shows some effort toward personal and professional growth. Engages in learning activities but may not do so consistently. Some openness to learning and unlearning.
#   - Low: Rarely focuses on personal or professional growth. Engagement in learning is limited and may resist feedback or change.
# People Potential:
#   - High: Consistently shows capability to lead and inspire others. Displays strong empathy, understanding, and a focus on people. Builds relationships with ease and enjoys social interactions.
#   - Moderate: Displays some ability to relate to and lead others. May show empathy and focus on people inconsistently. Builds relationships but may need support.
#   - Low: Shows limited capability in leading or inspiring others. Social interaction may be minimal or strained. Struggles to build and maintain relationships.
# Strategic Potential:
#   - High: Approaches work with a strong focus on the bigger picture. Operates independently with minimal guidance. Demonstrates a commercial and strategic mindset, regularly anticipating trends and their impact.
#   - Moderate: Some awareness of the bigger picture but may need occasional guidance. Understands strategy in parts but may not consistently anticipate trends or broader implications.
#   - Low: Focus tends to be on immediate tasks. Requires frequent guidance. Shows limited awareness of trends or the strategic impact of work.
# Execution Potential:
#   - High: Consistently addresses problems and challenges with confidence and resilience. Takes a diligent, practical, and solution-focused approach to solving issues.
#   - Moderate: Can address problems but may need support or time to build confidence and resilience. Attempts a practical approach but not always solution-focused.
#   - Low: Struggles to address problems confidently. May rely heavily on others. Practical or solution-oriented approaches are limited.
# Change Potential:
#   - High: Thrives in change and complexity. Manages new ways of working with adaptability, flexibility, and decisiveness during uncertainty.
#   - Moderate: Generally copes with change and can adapt when needed. May need support to remain flexible or decisive in uncertain situations.
#   - Low: Struggles with change or uncertainty. May resist new ways of working and has difficulty adapting or deciding in changing circumstances.

# --- GOLD STANDARD EXAMPLES (FINAL HUMAN-CORRECTED SET) ---

# **EXAMPLE 1 (Dipsy):**
# **INPUT:** Name: Dipsy, Gender: M, Overall Leadership: 4, Reasoning & Problem Solving: 3, Drive Potential: 4, Contribution: 3, Purpose: 4, Achievement: 4, Learning Potential: 3, Mastery: 3, Growth: 2, Insightful: 4, People Potential: 4, Collaboration: 3, Empathy: 4, Sociable: 4, Strategic Potential: 3, Awareness: 2, Autonomy: 3, Perspective: 3, Execution Potential: 3, Resourcefulness: 3, Efficacy: 3, Resilience: 4, Change Potential: 4, Agility: 4, Ambiguity: 4, Venturesome: 2, Steers Changes: 3, Manages Stakeholders: 4, Drives Results: 3, Thinks Strategically: 2, Solves Challenges: 3, Develops Talent: 4
# **CORRECT OUTPUT:**
# Dipsy demonstrates high potential with a strong capacity for growth and success in a more complex role. He shows a consistent drive to achieve goals and displays resilience in challenging situations, readily adapting to change and navigating ambiguity. His ability to develop and nurture relationships with key stakeholders is a notable strength, further reinforced by his natural empathy and sociability. His strategic awareness appears limited, with only partial understanding of broader trends and their implications. He tends to show some belief in improvement and change, but this is not consistently applied to himself or others. Additionally, he may be hesitant to engage with ideas that involve risk or discomfort, which could limit his adaptability in unfamiliar situations.
#
# Strengths:
# • Cultivates strong relationships with stakeholders, demonstrating empathy and building rapport with ease.
# • Demonstrates a consistent drive to achieve goals and displays resilience in challenging situations, readily adapting to change and uncertainty.
#
# Development Areas:
# • May benefit from developing greater strategic awareness to connect his actions to the bigger picture and inform decision-making.
# • Has an opportunity to cultivate a more venturesome approach, demonstrating greater comfort with taking calculated risks and exploring new ideas.

# **EXAMPLE 2 (Po):**
# **INPUT:** Name: Po, Gender: M, Overall Leadership: 3, Reasoning & Problem Solving: 2, Drive Potential: 3, Contribution: 2, Purpose: 4, Achievement: 2, Learning Potential: 4, Mastery: 3, Growth: 4, Insightful: 4, People Potential: 2, Collaboration: 3, Empathy: 2, Sociable: 1, Strategic Potential: 3, Awareness: 3, Autonomy: 4, Perspective: 3, Execution Potential: 4, Resourcefulness: 3, Efficacy: 4, Resilience: 4, Change Potential: 3, Agility: 2, Ambiguity: 3, Venturesome: 3, Steers Changes: 4, Manages Stakeholders: 2, Drives Results: 3, Thinks Strategically: 4, Solves Challenges: 2, Develops Talent: 4
# **CORRECT OUTPUT:**
# Po demonstrates moderate leadership potential with opportunities for growth. He possesses a strong sense of purpose and a clear commitment to continuous learning and development, actively seeking out new knowledge and experiences. He effectively steers organizational change and demonstrates resilience in the face of challenges. However, his lower scores in reasoning and problem-solving, combined with a tendency to approach goals with some motivation but not always consistent follow-through, may hinder his ability to convert vision into action and outcomes. While he excels at driving change and developing talent, his limited ease in social interactions and an inconsistent focus on others’ thoughts and emotions suggest that building strong, people-centered relationships may be more challenging. Developing stronger interpersonal skills and a more practical problem-solving approach will be crucial for unlocking his full leadership potential and maximizing his impact.
#
# Strengths:
# • Demonstrates a strong ability to recognize and drive organizational change and transformation, displaying resilience and enabling buy-in.
# • Demonstrates strong learning potential, consistently focusing on personal and professional growth and showing a clear commitment to continuous learning and improvement.
#
# Development Areas:
# • May benefit from enhancing problem-solving capabilities by developing a more analytical and structured approach to complex issues.
# • Can strengthen his leadership presence by enhancing interpersonal skills, particularly focusing on developing greater empathy and improving sociability to foster stronger team dynamics.

# **EXAMPLE 3 (Tinky Winky):**
# **INPUT:** Name: Tinky Winky, Gender: F, Overall Leadership: 2, Reasoning & Problem Solving: 3, Drive Potential: 1, Contribution: 2, Purpose: 1, Achievement: 1, Learning Potential: 2, Mastery: 3, Growth: 2, Insightful: 2, People Potential: 4, Collaboration: 3, Empathy: 4, Sociable: 4, Strategic Potential: 2, Awareness: 3, Autonomy: 2, Perspective: 2, Execution Potential: 3, Resourcefulness: 3, Efficacy: 3, Resilience: 2, Change Potential: 2, Agility: 1, Ambiguity: 2, Venturesome: 2, Steers Changes: 2, Manages Stakeholders: 3, Drives Results: 2, Thinks Strategically: 3, Solves Challenges: 2, Develops Talent: 3
# **CORRECT OUTPUT:**
# Tinky Winky demonstrates low leadership potential, but significant development is needed for her to succeed in a more complex leadership role. She possesses strong interpersonal skills, demonstrating empathy and building rapport easily. This natural ability to connect with others fosters positive stakeholder relationships. However, her low drive to achieve results may limit her ability to consistently lead teams toward achieving outcomes in more complex settings. She demonstrates limited motivation or initiative and may meet expectations but does not show a consistent drive to exceed them. She may need occasional guidance and understands strategy in parts, but may not consistently anticipate trends or broader implications. Tinky Winky struggles with change or uncertainty and may resist new ways of working, having difficulty adapting or deciding in changing circumstances.
#
# Strengths:
# • Builds rapport with ease, demonstrating high empathy and establishing strong interpersonal connections with others.
# • Demonstrates strong people potential, readily engaging with stakeholders and fostering positive relationships.
#
# Development Areas:
# • Needs to cultivate a stronger results orientation and demonstrate a more consistent drive to achieve goals.
# • Should focus on enhancing her ability to drive change and navigate ambiguity, demonstrating greater agility and resilience in challenging situations.

# --- END OF INSTRUCTIONS AND EXAMPLES ---

### NEW CANDIDATE DATA TO ANALYZE ###
{candidate_data_string}

# AnalystAI, generate the report now.
"""

# --- App Title and Description ---
st.title("HCTA AI Report Generator v11 (Definitive)")
st.markdown("""
This application uses a Gemini model that has been meticulously engineered with multiple rounds of expert feedback.
1.  Download the sample template to see the required format and column order.
2.  Fill the template with your candidate data.
3.  Upload the completed Excel file and click "Generate Summaries".
""")

# --- Helper function to create and download the sample Excel file ---
@st.cache_data
def create_sample_template():
    # Define all expected columns in the correct order
    columns = [
        'Name', 'Gender', 'Overall Leadership', 'Reasoning & Problem Solving',
        'Drive Potential', 'Contribution', 'Purpose', 'Achievement',
        'Learning Potential', 'Mastery', 'Growth', 'Insightful',
        'People Potential', 'Collaboration', 'Empathy', 'Sociable',
        'Strategic Potential', 'Awareness', 'Autonomy', 'Perspective',
        'Execution Potential', 'Resourcefulness', 'Efficacy', 'Resilience',
        'Change Potential', 'Agility', 'Ambiguity', 'Venturesome',
        'Steers Changes', 'Manages Stakeholders', 'Drives Results',
        'Thinks Strategically', 'Solves Challenges', 'Develops Talent'
    ]
    # Create a sample row based on the final feedback example
    sample_data = {
        'Name': ['Tinky Winky'], 'Gender': ['F'], 'Overall Leadership': [2.0], 'Reasoning & Problem Solving': [3.0],
        'Drive Potential': [1.0], 'Contribution': [2.0], 'Purpose': [1.0], 'Achievement': [1.0],
        'Learning Potential': [2.0], 'Mastery': [3.0], 'Growth': [2.0], 'Insightful': [2.0],
        'People Potential': [4.0], 'Collaboration': [3.0], 'Empathy': [4.0], 'Sociable': [4.0],
        'Strategic Potential': [2.0], 'Awareness': [3.0], 'Autonomy': [2.0], 'Perspective': [2.0],
        'Execution Potential': [3.0], 'Resourcefulness': [3.0], 'Efficacy': [3.0], 'Resilience': [2.0],
        'Change Potential': [2.0], 'Agility': [1.0], 'Ambiguity': [2.0], 'Venturesome': [2.0],
        'Steers Changes': [2.0], 'Manages Stakeholders': [3.0], 'Drives Results': [2.0],
        'Thinks Strategically': [3.0], 'Solves Challenges': [2.0], 'Develops Talent': [3.0]
    }
    df_sample = pd.DataFrame(sample_data, columns=columns)
    
    # Convert to Excel in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_sample.to_excel(writer, index=False, sheet_name='Candidates')
    processed_data = output.getvalue()
    return processed_data

sample_excel = create_sample_template()
st.download_button(
    label="📥 Download Sample Template (Excel)",
    data=sample_excel,
    file_name="candidate_scores_template_v11.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.divider()

# --- File Uploader ---
uploaded_file = st.file_uploader("📂 Upload Your Completed Excel File", type=["xlsx"])

# --- Main Logic ---
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("File Uploaded Successfully!")
        st.dataframe(df.head())

        if st.button("✨ Generate Summaries with Final AI", type="primary"):
            # Check for API Key
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                genai.configure(api_key=api_key)
            except (KeyError, FileNotFoundError):
                st.error("GEMINI_API_KEY not found. Please add it to your Streamlit secrets.")
                st.stop()
            
            model = genai.GenerativeModel('gemini-2.5-pro')
            
            results = []
            total_candidates = len(df)
            progress_bar = st.progress(0, text="Initializing...")

            results_container = st.container()

            for index, row in df.iterrows():
                progress_text = f"Analyzing {row['Name']} ({index + 1}/{total_candidates})..."
                progress_bar.progress((index + 1) / total_candidates, text=progress_text)
                
                # Create the data string for the prompt
                candidate_data_string = "# INPUT SCORES:\n"
                for col_name, value in row.items():
                    candidate_data_string += f"# {col_name}: {value}\n"

                # Format the final prompt
                final_prompt = SAVANT_PROMPT_TEMPLATE.format(candidate_data_string=candidate_data_string)
                
                try:
                    # API Call
                    response = model.generate_content(final_prompt)
                    summary = response.text
                except Exception as e:
                    summary = f"Error generating summary for {row['Name']}: {e}"
                
                results.append(summary)
                
                # Display result immediately
                with results_container:
                    st.subheader(f"Generated Summary for {row['Name']}")
                    st.markdown(summary)
                    st.divider()

                time.sleep(2) 

            progress_bar.empty()
            st.success("✅ All summaries generated!")
            
            df['Generated Summary (v11)'] = results
            
            # --- Download Results ---
            output_results = io.BytesIO()
            with pd.ExcelWriter(output_results, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Results_v11')
            
            st.download_button(
                label="⬇️ Download All Results with Summaries (Excel)",
                data=output_results.getvalue(),
                file_name="candidate_summaries_results_v11.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
