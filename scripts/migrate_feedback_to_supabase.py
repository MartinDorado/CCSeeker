#!/usr/bin/env python3
"""
Migrate local .feedback_data.json to Supabase.

Usage:
    python scripts/migrate_feedback_to_supabase.py [--filepath PATH]

Requires: SUPABASE_URL and SUPABASE_SERVICE_KEY env vars (or .env file).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.analytics.feedback_tracker import JSONFeedbackStore, SupabaseFeedbackStore


def main():
    parser = argparse.ArgumentParser(description="Migrate feedback JSON to Supabase.")
    parser.add_argument("--filepath", default=None, help="Path to .feedback_data.json")
    args = parser.parse_args()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.", file=sys.stderr)
        sys.exit(1)

    source = JSONFeedbackStore(filepath=args.filepath) if args.filepath else JSONFeedbackStore()
    entries = source.load_entries()
    if not entries:
        print("No entries to migrate.")
        return

    dest = SupabaseFeedbackStore(url, key)
    success = 0
    for entry in entries:
        if dest.save_entry(entry):
            success += 1
        else:
            print(f"Failed to migrate entry: {entry.get('timestamp', '?')}", file=sys.stderr)

    print(f"Migrated {success}/{len(entries)} entries.")


if __name__ == "__main__":
    main()
