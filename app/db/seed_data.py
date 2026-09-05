"""
Seed data for the knowledge base. Loaded once at startup if the table is
empty (see app/db/seed.py). Kept as plain Python data (not a migration) so
it's trivial to edit/extend without writing a new Alembic revision.
"""

KB_SEED_ENTRIES = [
    {
        "title": "How to request a refund",
        "content": (
            "You can request a refund from your Order History page within "
            "30 days of purchase. Select the order, click 'Request refund', "
            "and choose a reason. Refunds are processed to the original "
            "payment method within 5-7 business days."
        ),
        "category": "billing",
        "tags": "refund,payment,order",
        "source": "internal_policy",
        "trust_score": 0.95,
        "verified": True,
    },
    {
        "title": "Why was I charged twice?",
        "content": (
            "Duplicate charges are usually a temporary authorization hold "
            "from your bank, not an actual double charge, and they "
            "typically disappear within 3-5 business days. If both charges "
            "are still present after 5 business days, contact support with "
            "your order number so we can issue a refund for the duplicate."
        ),
        "category": "billing",
        "tags": "charge,duplicate,payment",
        "source": "internal_policy",
        "trust_score": 0.9,
        "verified": True,
    },
    {
        "title": "Tracking your package",
        "content": (
            "Once your order ships, you'll receive a tracking link by "
            "email. You can also find it under Order History > Track "
            "package. Tracking updates can take up to 24 hours to appear "
            "after the carrier picks up the package."
        ),
        "category": "shipping",
        "tags": "tracking,delivery,shipping",
        "source": "internal_policy",
        "trust_score": 0.9,
        "verified": True,
    },
    {
        "title": "My package is marked delivered but I don't have it",
        "content": (
            "First, check with neighbors and around your delivery location, "
            "as carriers sometimes mark a package delivered slightly before "
            "it arrives. If it's still missing after 48 hours, contact "
            "support with your order number and we will open a carrier "
            "investigation or send a replacement."
        ),
        "category": "shipping",
        "tags": "lost,delivery,package",
        "source": "internal_policy",
        "trust_score": 0.85,
        "verified": True,
    },
    {
        "title": "Resetting your password",
        "content": (
            "Go to the login page and click 'Forgot password'. Enter the "
            "email on your account and you'll receive a reset link valid "
            "for 60 minutes. If you don't see the email, check your spam "
            "folder or request a new link."
        ),
        "category": "technical",
        "tags": "password,login,reset",
        "source": "internal_policy",
        "trust_score": 0.95,
        "verified": True,
    },
    {
        "title": "The website shows an error when I check out",
        "content": (
            "Checkout errors are most often caused by an expired card, an "
            "outdated browser cache, or an ad-blocker interfering with the "
            "payment form. Try clearing your browser cache, disabling "
            "extensions, or using a different browser. If the error "
            "persists, note the error code shown and contact support."
        ),
        "category": "technical",
        "tags": "error,checkout,bug",
        "source": "internal_policy",
        "trust_score": 0.7,
        "verified": True,
    },
    {
        "title": "How to update your account email",
        "content": (
            "Go to Account Settings > Profile and click 'Edit' next to "
            "your email address. You'll need to confirm the change via a "
            "verification link sent to the new address before it takes "
            "effect."
        ),
        "category": "account",
        "tags": "email,account,profile",
        "source": "internal_policy",
        "trust_score": 0.9,
        "verified": True,
    },
    {
        "title": "How to delete your account",
        "content": (
            "Account deletion is permanent and cannot be undone. Go to "
            "Account Settings > Privacy > Delete account. For safety, "
            "account deletion requests are reviewed by a support agent "
            "before being finalized, so this cannot currently be completed "
            "as a fully automated action."
        ),
        "category": "account",
        "tags": "delete,account,privacy",
        "source": "internal_policy",
        "trust_score": 0.8,
        "verified": True,
    },
    {
        "title": "Customer support hours",
        "content": (
            "Our support team is available Monday to Friday, 9am-6pm "
            "(UTC). Outside these hours you can still send a message and "
            "we'll respond as soon as we're back online."
        ),
        "category": "general",
        "tags": "hours,contact,support",
        "source": "internal_policy",
        "trust_score": 0.95,
        "verified": True,
    },
    {
        "title": "Return policy overview",
        "content": (
            "Most items can be returned within 30 days of delivery in "
            "their original condition. Personalized or perishable items "
            "are not eligible for return. Return shipping is free for "
            "defective items and paid by the customer otherwise."
        ),
        "category": "general",
        "tags": "return,policy",
        "source": "internal_policy",
        "trust_score": 0.9,
        "verified": True,
    },
]
