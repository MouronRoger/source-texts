#!/usr/bin/env python3
"""
Extract Greek texts from Daphnet book transcriptions using discovered CSV URLs.

This script leverages the windfall discovery of direct book transcription URLs
to extract complete chapter texts instead of individual fragments.
"""

import asyncio
import csv
import json
import logging
import os
import re
import unic