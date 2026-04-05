import gradio as gr
# from huggingface_hub import HfFolder

def add_numbers(Num1, Num2):
    return Num1 + Num2

def combine(a, b):
    return a + " " + b

# Define the interface
demo = gr.Interface(
    fn=add_numbers, 
    inputs=[gr.Number(), gr.Number()], # Create two numerical input fields where users can enter numbers
    outputs=gr.Number() # Create numerical output fields
)
demo1 = gr.Interface(
    fn=combine,
    inputs=[gr.Textbox(label="Input 1"), gr.Textbox(label="Input 2")],
    outputs=gr.Textbox(label="Output")
)

# Launch the interfaced
# demo.launch(server_name="127.0.0.1", server_port= 7860)
demo1.launch(server_name="127.0.0.1", server_port= 7861)
