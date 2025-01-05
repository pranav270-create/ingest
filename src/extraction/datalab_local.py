from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
import json
import PIL
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import fitz
import os


def polygon_to_bbox(polygon):
    """Convert polygon coordinates to bounding box format."""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    
    return {
        "left": min(x_coords),
        "top": min(y_coords),
        "width": max(x_coords) - min(x_coords),
        "height": max(y_coords) - min(y_coords)
    }

def visualize_page_results(image_path: str, parsed_elements: list, output_path: str = None) -> None:
    """
    Visualize the parsed elements by drawing bounding boxes over the original image.
    """
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Get image dimensions
    width, height = image.size
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Colors for different types of elements
    colors = {
        "Figure": "red",
        "Picture": "red",
        "FigureGroup": "red",
        "Table": "blue",
        "TableGroup": "blue",
        "Text": "green",
        "Page": "yellow",
        "PageHeader": "orange",
        "SectionHeader": "purple",
        "ListGroup": "cyan",
        "ListItem": "magenta",
        "Footnote": "pink",
        "Equation": "brown",
        "TextInlineMath": "teal"
    }
    
    for element in parsed_elements:
        # Convert polygon to bounding box
        bbox = polygon_to_bbox(element["polygon"])
            
        # Convert relative coordinates to absolute pixels
        left = bbox["left"]
        top = bbox["top"]
        right = left + bbox["width"]
        bottom = top + bbox["height"]
        
        # Get color based on block_type, default to white if not found
        color = colors.get(element["block_type"], "white")
            
        # Draw rectangle
        draw.rectangle([left, top, right, bottom], outline=color, width=2)
        
        # Add label with block type
        label = element["block_type"]
        draw.rectangle([left, max(0, top-10), left + 10, max(0, top-10) + 10], fill="white")
        draw.text((left, max(0, top-10)), label, fill=color, font=font)
    
    # Display or save the result
    if output_path:
        image.save(output_path)
    else:
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

def convert_pdf_to_images(pdf_path: str, output_dir: str) -> list[str]:
    """
    Convert a PDF file to images.
    """
    image_paths = []
    os.makedirs(output_dir, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            image_path = f"{output_dir}/page_{i+1}.png"
            pix.save(image_path)
            image_paths.append(image_path)

    return image_paths

def main(file: str):
    config = {
        "use_llm": True,
        "output_format": "json",
    }
    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    rendered = converter(file)
    
    if config["output_format"] == "markdown":
        metadata = rendered.metadata
        text = rendered.markdown
        images = rendered.images
        # Save images to a folder
        for key, image in images.items():
            image.save(f"output_datalab/image_{key}.png")
        # Save text to markdown file
        with open("output_datalab/output.md", "w") as f:
            f.write(text)
        # Save metadata to json file
        with open("output_datalab/output.json", "w") as f:
            json.dump(metadata, f, indent=4)
    elif config["output_format"] == "json":
        text = rendered.model_dump_json(exclude=["metadata"], indent=2)
        with open("output_datalab/output.json", "w") as f:
            f.write(text)
        # Convert PDF to images first
        image_paths = convert_pdf_to_images(file, "output_datalab")
        # Process each page
        data = json.load(open("output_datalab/output.json", "r"))
        for page_num, page in enumerate(data["children"]):
            # Collect all elements from the page
            elements = []
            # Add the page itself
            elements.append({
                "block_type": page["block_type"],
                "polygon": page["polygon"]
            })
            # Add all child elements
            if page.get("children"):
                for child in page["children"]:
                    elements.append({
                        "block_type": child["block_type"],
                        "polygon": child["polygon"]
                    })
            # Visualize the page
            image_path = f"output_datalab/page_{page_num + 1}.png"
            output_path = f"output_datalab/page_{page_num + 1}_annotated.png"
            visualize_page_results(image_path, elements, output_path)


def visualize(file: str):
    image_paths = convert_pdf_to_images(file, "output_datalab")
    # Process each page
    data = json.load(open("output_datalab/output.json", "r"))
    for page_num, page in enumerate(data["children"]):
        # Collect all elements from the page
        elements = []
        # Add the page itself
        elements.append({
            "block_type": page["block_type"],
            "polygon": page["polygon"]
        })
        # Add all child elements
        if page.get("children"):
            for child in page["children"]:
                elements.append({
                    "block_type": child["block_type"],
                    "polygon": child["polygon"]
                })
        # Visualize the page
        image_path = f"output_datalab/page_{page_num + 1}.png"
        output_path = f"output_datalab/page_{page_num + 1}_annotated.png"
        # get the unique block types
        visualize_page_results(image_path, elements, output_path)


if __name__ == '__main__':
    file = "/Users/pranaviyer/Apeiron/apeiron-ml/data/Other/Content/Foundation Docs/Exercise Foundation Protocol.pdf"
    main(file)
