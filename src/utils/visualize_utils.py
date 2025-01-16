import io
from PIL import Image, ImageDraw, ImageFont
from src.schemas.schemas import ExtractedFeatureType
from typing import Dict, List, Union

# Define colors for each ExtractedFeatureType
FEATURE_COLORS = {
    # Common content types
    ExtractedFeatureType.text: "green",
    ExtractedFeatureType.word: "lightgreen",
    ExtractedFeatureType.line: "darkgreen",
    ExtractedFeatureType.image: "red",
    ExtractedFeatureType.table: "blue",
    ExtractedFeatureType.figure: "red",
    ExtractedFeatureType.code: "purple",
    ExtractedFeatureType.equation: "brown",
    ExtractedFeatureType.form: "orange",
    ExtractedFeatureType.header: "darkblue",
    ExtractedFeatureType.footer: "darkblue",
    ExtractedFeatureType.section_header: "purple",
    ExtractedFeatureType.list: "cyan",
    ExtractedFeatureType.page_number: "gray",
    # Marker-specific types
    ExtractedFeatureType.span: "lightgray",
    ExtractedFeatureType.figuregroup: "darkred",
    ExtractedFeatureType.tablegroup: "darkblue",
    ExtractedFeatureType.listgroup: "darkcyan",
    ExtractedFeatureType.picturegroup: "darkred",
    ExtractedFeatureType.picture: "red",
    ExtractedFeatureType.page: "yellow",
    ExtractedFeatureType.caption: "darkgreen",
    ExtractedFeatureType.footnote: "pink",
    ExtractedFeatureType.handwriting: "violet",
    ExtractedFeatureType.textinlinemath: "brown",
    ExtractedFeatureType.tableofcontents: "teal",
    ExtractedFeatureType.document: "black",
    ExtractedFeatureType.complexregion: "olive",
    # Textract-specific types
    ExtractedFeatureType.key_value: "magenta",
    # Catch-all types
    ExtractedFeatureType.combined_text: "blue",
    ExtractedFeatureType.section_text: "darkblue",
    ExtractedFeatureType.other: "gray",
}


def group_entries_by_page(entries: list) -> Dict[int, List[dict]]:
    """
    Group entries by page number and convert to visualization format.

    Args:
        entries: List of Entry objects

    Returns:
        Dictionary mapping page numbers (0-based) to lists of element dictionaries
    """
    entries_by_page = {}

    for entry in entries:
        for location in entry.chunk_locations:
            page_num = location.index.primary - 1  # Convert to 0-based index
            if page_num not in entries_by_page:
                entries_by_page[page_num] = []

            entries_by_page[page_num].append(
                {
                    "feature_type": entry.consolidated_feature_type,
                    "bbox": {
                        "left": location.bounding_box.left,
                        "top": location.bounding_box.top,
                        "width": location.bounding_box.width,
                        "height": location.bounding_box.height,
                    },
                }
            )

    return entries_by_page


async def visualize_page_results(
    image_path: str,
    elements: list[dict],
    read=None,
) -> bytes:
    """
    Visualize extracted elements by drawing bounding boxes over the original image.

    Args:
        image_path: Path to the page image
        elements: List of dictionaries containing:
            - feature_type: ExtractedFeatureType enum value
            - bbox: Dict with keys left, top, width, height
                   or List [left, top, right, bottom]
        read: Optional async read function for cloud storage

    Returns:
        bytes: Annotated image as bytes
    """
    # Read the image using the provided read function or directly
    if read:
        img_bytes = await read(image_path, mode="rb")
        image = Image.open(io.BytesIO(img_bytes))
    else:
        image = Image.open(image_path)

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception as e:
        print(f"Error loading font: {e}")
        font = ImageFont.load_default()

    for element in elements:
        feature_type = element["feature_type"]
        bbox = element["bbox"]

        # Handle both bbox formats (left,top,right,bottom) and (left,top,width,height)
        if len(bbox) == 4:
            if isinstance(bbox, dict):
                # Handle dict format with width/height
                left = bbox["left"]
                top = bbox["top"]
                right = left + bbox["width"]
                bottom = top + bbox["height"]
            else:
                # Handle list format with right/bottom
                left, top, right, bottom = bbox
        else:
            raise ValueError("Invalid bbox format")

        # Get color from mapping, default to black if not found
        color = FEATURE_COLORS.get(feature_type, "black")

        # Draw bounding box
        draw.rectangle([left, top, right, bottom], outline=color, width=2)

        # Add label with type
        label = feature_type.value
        # Create semi-transparent background for label
        label_width = len(label) * 10
        label_height = 20
        label_bg = [left, max(0, top - label_height), left + label_width, max(0, top)]

        # Create a new RGBA image for the label background
        label_img = Image.new("RGBA", (label_width, label_height), (255, 255, 255, 128))
        image.paste(label_img, (int(label_bg[0]), int(label_bg[1])), label_img)

        # Draw text over semi-transparent background
        draw.text((left, max(0, top - label_height)), label, fill=color, font=font)

    # Save to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    image.close()
    return img_byte_arr.getvalue()
