async def deprecated_process_para_blocks(
    blocks: list,
    page_num: int,
    ingestion: Ingestion,
    page_file_path: str,
    chunk_idx: int,
    counters: dict,
    page_width: float,
    page_height: float,
    extracts_dir: str,
    write=None,
) -> tuple[list[Entry], int]:
    """Recursively process para_blocks to extract entries."""
    entries = []

    for block in blocks:
        feature_type = _map_mineru_type(block["type"])

        # Initialize content
        content = ""
        extracted_path = None

        # Recursively process nested lines or spans
        if "lines" in block:
            for line in block["lines"]:
                for span in line.get("spans", []):
                    if span["type"] == "text":
                        content += f"{span['content']} "
                    elif span["type"] == "inline_equation":
                        content += f"{span.get('latex', '')} "
                    elif span["type"] == "table":
                        # Handle table content
                        content += f"{span.get('html', '')} "
                        counters["table_count"] += 1
                        # Optionally extract table image if available
                        if "image_path" in span:
                            extracted_path = (
                                f"{extracts_dir}/table_{counters['table_count']}.jpg"
                            )
                    elif span["type"] == "figure":
                        # Handle figure content
                        counters["figure_count"] += 1
                        extracted_path = (
                            f"{extracts_dir}/figure_{counters['figure_count']}.jpg"
                        )
                    else:
                        # Other types can be added here
                        continue

        elif "spans" in block:
            for span in block["spans"]:
                if span["type"] == "text":
                    content += f"{span['content']} "
                elif span["type"] == "inline_equation":
                    content += f"{span.get('latex', '')} "
                # Add other span types if necessary

        # Clean up content
        content = content.strip()

        if content:
            location = ChunkLocation(
                index=Index(
                    primary=page_num + 1,
                    secondary=chunk_idx + 1,
                ),
                extracted_feature_type=feature_type,
                extracted_file_path=extracted_path,
                page_file_path=page_file_path,
                bounding_box=_convert_bbox(
                    block.get("bbox", []), page_width, page_height
                ),
            )

            entry = Entry(
                uuid=str(uuid.uuid4()),
                ingestion=ingestion,
                string=content,
                consolidated_feature_type=feature_type,
                chunk_locations=[location],
                min_primary_index=page_num + 1,
                max_primary_index=page_num + 1,
                chunk_index=chunk_idx + 1,
            )
            entries.append(entry)
            chunk_idx += 1

        # Recursively process nested blocks if any
        if "blocks" in block:
            nested_entries, chunk_idx = await deprecated_process_para_blocks(
                block["blocks"],
                page_num,
                ingestion,
                page_file_path,
                chunk_idx,
                counters,
                page_width,
                page_height,
                extracts_dir,
                write,
            )
            entries.extend(nested_entries)

    return entries, chunk_idx