import io
from PIL import Image, ImageDraw, ImageFont
from src.schemas.schemas import ExtractedFeatureType
from typing import Dict, List, Union

# Define colors for each ExtractedFeatureType
FEATURE_COLORS = {
    # Primary content types (distinct colors)
    ExtractedFeatureType.text: "#1f77b4",  # Blue
    ExtractedFeatureType.combined_text: "#2ca02c",  # Green
    ExtractedFeatureType.section_text: "#17becf",  # Cyan
    ExtractedFeatureType.table: "#ff7f0e",  # Orange
    ExtractedFeatureType.figure: "#d62728",  # Red
    ExtractedFeatureType.image: "#e377c2",  # Pink
    # Text hierarchy (blue family)
    ExtractedFeatureType.word: "#aec7e8",  # Light blue
    ExtractedFeatureType.line: "#7fb1e3",  # Medium blue
    ExtractedFeatureType.section_header: "#1f77b4",  # Standard blue
    ExtractedFeatureType.header: "#2d4b8e",  # Dark blue
    ExtractedFeatureType.footer: "#2d4b8e",  # Dark blue
    # Tables and forms (orange family)
    ExtractedFeatureType.tablegroup: "#ff7f0e",  # Standard orange
    ExtractedFeatureType.form: "#ffbb78",  # Light orange
    ExtractedFeatureType.key_value: "#ff9f51",  # Medium orange
    # Figures and images (red/pink family)
    ExtractedFeatureType.figuregroup: "#d62728",  # Standard red
    ExtractedFeatureType.picture: "#e377c2",  # Pink
    ExtractedFeatureType.picturegroup: "#f7b6d2",  # Light pink
    # Lists and structure (green family)
    ExtractedFeatureType.list: "#2ca02c",  # Standard green
    ExtractedFeatureType.listgroup: "#98df8a",  # Light green
    ExtractedFeatureType.page: "#a1d99b",  # Pale green
    # Special content (purple family)
    ExtractedFeatureType.equation: "#9467bd",  # Standard purple
    ExtractedFeatureType.code: "#c5b0d5",  # Light purple
    ExtractedFeatureType.textinlinemath: "#8c564b",  # Brown-purple
    # Annotations (gray family)
    ExtractedFeatureType.caption: "#7f7f7f",  # Medium gray
    ExtractedFeatureType.footnote: "#c7c7c7",  # Light gray
    ExtractedFeatureType.span: "#d9d9d9",  # Pale gray
    ExtractedFeatureType.page_number: "#bcbcbc",  # Another gray
    # Miscellaneous
    ExtractedFeatureType.handwriting: "#8c564b",  # Brown
    ExtractedFeatureType.tableofcontents: "#17becf",  # Cyan
    ExtractedFeatureType.document: "#1f77b4",  # Blue
    ExtractedFeatureType.complexregion: "#9edae5",  # Light cyan
    ExtractedFeatureType.other: "#c7c7c7",  # Light gray
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
                    "feature_type": location.extracted_feature_type,
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
