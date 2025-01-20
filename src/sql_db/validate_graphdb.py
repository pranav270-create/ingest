import sys
from pathlib import Path

import networkx as nx
from pyvis.network import Network
from sqlalchemy import select
from sqlalchemy.orm import selectinload

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.sql_db.database import get_async_db_session
from src.sql_db.etl_model import Entry


async def get_entry_relationships(pipeline_id: int) -> tuple[list[Entry], list[tuple[int, int, str]]]:
    """
    Fetch all entries and their relationships for a given pipeline ID.
    """
    async for session in get_async_db_session():
        # Get all entries for this pipeline with relationships
        entry_stmt = (
            select(Entry)
            .where(Entry.pipeline_id == pipeline_id)
            .options(
                selectinload(Entry.outgoing_relationships),
                selectinload(Entry.incoming_relationships),
            )
        )

        result = await session.execute(entry_stmt)
        entries = result.scalars().all()

        # Debug prints
        print(f"Found {len(entries)} entries for pipeline {pipeline_id}")

        # Collect all relationships
        relationships = []
        for entry in entries:
            print(f"Entry {entry.id} has:")
            print(f"  - {len(entry.outgoing_relationships)} outgoing relationships")
            print(f"  - {len(entry.incoming_relationships)} incoming relationships")

            for rel in entry.outgoing_relationships:
                relationships.append((rel.source_id, rel.target_id, rel.relationship_type))
                print(f"  - Relationship: {rel.source_id} -> {rel.target_id} ({rel.relationship_type})")

        print(f"Found {len(relationships)} total relationships")
        return entries, relationships


def create_relationship_graph(
    entries: list[Entry],
    relationships: list[tuple[int, int, str]],
    output_path: str = "entry_relationships.html"
) -> None:
    """
    Create an interactive HTML visualization of entry relationships.
    """
    # Create network
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        notebook=False,
        directed=True
    )

    # Create mapping of entry IDs to labels
    entry_map = {
        entry.id: (
            f"Entry {entry.id}\n{entry.entry_title[:30]}..."
            if entry.entry_title
            else f"Entry {entry.id}"
        )
        for entry in entries
    }

    # Add nodes
    for entry in entries:
        net.add_node(
            entry.id,
            label=entry_map[entry.id],
            title=f"Entry {entry.id}\nType: {entry.consolidated_feature_type or 'Unknown'}",
            shape='box'
        )

    # Add edges with arrows and detailed labels
    for source_id, target_id, rel_type in relationships:
        net.add_edge(
            source_id,
            target_id,
            title=f"Source: {source_id}\nTarget: {target_id}\nType: {rel_type}",
            label=f"{rel_type}\n{source_id}→{target_id}",
            arrows='to',
            physics=True
        )

    # Configure physics and styling
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "springLength": 200,
                "springConstant": 0.8
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        },
        "edges": {
            "smooth": {
                "type": "curvedCW",
                "roundness": 0.2
            },
            "font": {
                "size": 12,
                "align": "middle"
            }
        },
        "nodes": {
            "font": {
                "size": 14,
                "face": "arial"
            }
        }
    }
    """)

    # Save
    net.write_html(output_path)


async def visualize_pipeline_relationships(
    pipeline_id: int, output_path: str = "entry_relationships.html"
) -> None:
    """
    Create and save an interactive visualization of all entry relationships for a given pipeline.
    """
    entries, relationships = await get_entry_relationships(pipeline_id)
    if not entries:
        print(f"No entries found for pipeline {pipeline_id}")
        return

    print(f"Found {len(entries)} entries and {len(relationships)} relationships")
    create_relationship_graph(entries, relationships, output_path)
    print(f"Graph visualization saved to {output_path}")


if __name__ == "__main__":
    import asyncio
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize entry relationships for a pipeline')
    parser.add_argument('pipeline_id', type=int, help='Pipeline ID to visualize')
    parser.add_argument('--output', '-o', default="entry_relationships.html",
                        help='Output path for the HTML visualization (default: entry_relationships.html)')
    args = parser.parse_args()

    # Example usage
    async def main():
        await visualize_pipeline_relationships(args.pipeline_id, args.output)

    asyncio.run(main())
