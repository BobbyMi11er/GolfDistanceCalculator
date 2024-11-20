import gradio as gr

import object_det_organized

# TODO: add button for if want to be in meters or feet
# TODO: Progress bar or some other update feature so you know where in the process
# TODO: have default values or have "iPhone 12 is 120*", etc. in input text boxes
# TODO: have degrees sign and/or unit after text input
# TODO: ensure that there are values in textboxes
with gr.Blocks() as app:
    gr.Markdown("# Golf Pin Distance Finder")
    with gr.Column():
        with gr.Row():
            with gr.Column():
                pin_height_textbox = gr.Textbox(lines=1, show_label=False, placeholder="Put pin height in meters")
                camera_FOV_textbox = gr.Textbox(lines=1, show_label=False, placeholder="Put your camera's field of view in degrees")
            input_image = gr.Image(label="Input image of golf hole", width="300px")
        calculate_btn = gr.Button("Calculate Distance", variant="primary")
    with gr.Column():
        image_with_box = gr.Image(label="Image with golf pin found", width="300px")
        output_distance_feet = gr.Textbox(lines=1, show_label=False)

    calculate_btn.click(
        object_det_organized.process_golf_pin,
        inputs=[input_image,pin_height_textbox , camera_FOV_textbox],
        outputs=[image_with_box, output_distance_feet]
    )

app.launch()