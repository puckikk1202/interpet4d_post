"""InterPet4D model package.

Models are intentionally not imported eagerly here. Several legacy models
live under ``archived/``; eager imports made every current ``models.*`` import
fail when any one legacy module was absent.
"""
