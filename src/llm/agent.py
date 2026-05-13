"""
LLM Agent with RAG (Retrieval-Augmented Generation) using ChromaDB and OpenAI.

This module implements the CustomerSupportAgent class that:
- Manages a ChromaDB knowledge base with 16 customer support documents
- Performs semantic search via RAG to find relevant context
- Uses OpenAI GPT-3.5-turbo to generate accurate, context-aware responses
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List

import chromadb
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Knowledge base: 16 customer support documents
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "id": "kb_001",
        "title": "Return Policy",
        "content": (
            "Our return policy allows customers to return any item within 30 days of purchase "
            "for a full refund, provided the item is in its original condition and packaging. "
            "Items must be unused and include all original accessories and documentation. "
            "To initiate a return, contact our support team with your order number. "
            "Perishable goods, digital downloads, and customised items are non-returnable. "
            "Return shipping costs are covered by us for defective or incorrectly shipped items; "
            "for change-of-mind returns the customer bears the shipping cost."
        ),
    },
    {
        "id": "kb_002",
        "title": "Shipping Information",
        "content": (
            "We offer multiple shipping options to suit your needs. Standard shipping takes 5-7 "
            "business days and is free on orders over $50. Express shipping (2-3 business days) "
            "costs $9.99. Overnight shipping is available for $24.99. Orders placed before 2 PM "
            "EST are shipped the same business day. Tracking information is emailed once the "
            "order ships. We ship to all 50 US states and over 80 countries worldwide. "
            "International delivery times vary by destination (typically 7-21 business days)."
        ),
    },
    {
        "id": "kb_003",
        "title": "Warranty Policy",
        "content": (
            "All products come with a 1-year limited manufacturer's warranty covering defects in "
            "materials and workmanship under normal use. Electronics carry an extended 2-year "
            "warranty. The warranty does not cover damage from accidents, misuse, unauthorised "
            "modifications, or normal wear and tear. To make a warranty claim, contact support "
            "with proof of purchase and a description of the defect. We will repair, replace, or "
            "issue store credit at our discretion. Warranty service is available Monday–Friday, "
            "9 AM–5 PM EST."
        ),
    },
    {
        "id": "kb_004",
        "title": "Payment Methods",
        "content": (
            "We accept a wide variety of payment methods to make checkout convenient. Credit and "
            "debit cards accepted: Visa, Mastercard, American Express, and Discover. Digital "
            "wallets: PayPal, Apple Pay, Google Pay, and Shop Pay. Buy-now-pay-later: Klarna and "
            "Afterpay (split into 4 interest-free payments). Gift cards and store credits can be "
            "applied at checkout. All transactions are secured with 256-bit SSL encryption. We "
            "do not store full card details on our servers. Cryptocurrency payments are not "
            "currently supported."
        ),
    },
    {
        "id": "kb_005",
        "title": "Order Tracking",
        "content": (
            "Once your order ships you will receive a confirmation email with a tracking number "
            "and a direct link to the carrier's tracking page. You can also track your order by "
            "logging into your account and visiting 'My Orders'. We partner with UPS, FedEx, "
            "USPS, and DHL. Live tracking updates are refreshed every 2–4 hours. If your "
            "tracking shows 'delivered' but you haven't received the package, wait 24 hours "
            "then contact support. Orders can take up to 24 hours to appear in the carrier "
            "system after the shipping confirmation email is sent."
        ),
    },
    {
        "id": "kb_006",
        "title": "Customer Support Contact",
        "content": (
            "Our customer support team is available through multiple channels. Live chat is "
            "available on our website 24/7 for immediate assistance. Email support: "
            "support@example.com — response within 24 hours. Phone support: 1-800-555-0199, "
            "available Monday-Friday 8 AM-8 PM EST and Saturday 9 AM-5 PM EST. Social media "
            "support on Twitter (@ExampleSupport) and Facebook. For urgent issues outside "
            "business hours, use live chat or email. Average response time for emails is under "
            "4 hours during business hours. Our support agents speak English and Spanish."
        ),
    },
    {
        "id": "kb_007",
        "title": "Product Exchanges",
        "content": (
            "Exchanges are accepted within 30 days of purchase for items in original, unworn "
            "condition. To exchange for a different size or colour, visit our Exchange Portal "
            "or contact support. We will send the replacement item once the original is received "
            "at our warehouse. If the replacement item has a higher price, you will be charged "
            "the difference. If it is lower, a refund is issued for the difference. One exchange "
            "per order is permitted. Items marked 'Final Sale' are not eligible for exchange. "
            "International exchanges may incur additional shipping fees."
        ),
    },
    {
        "id": "kb_008",
        "title": "Refund Process",
        "content": (
            "Approved refunds are processed within 3-5 business days of receiving the returned "
            "item. Refunds are issued to the original payment method. Credit card refunds may "
            "take an additional 5-10 business days to appear on your statement depending on "
            "your bank. PayPal refunds typically appear within 3-5 business days. Store credit "
            "refunds are instant. If you haven't received your refund after 10 business days, "
            "please contact support with your order number and return tracking number. Partial "
            "refunds may be issued for items returned with missing accessories or in non-original "
            "condition."
        ),
    },
    {
        "id": "kb_009",
        "title": "International Shipping",
        "content": (
            "We ship to over 80 countries worldwide. International shipping rates are calculated "
            "at checkout based on destination, weight, and dimensions. Duties and taxes are the "
            "customer's responsibility and are not included in our shipping fees. Delivery times "
            "range from 7 to 21 business days depending on the destination and customs clearance. "
            "We use DHL Express and FedEx International for most shipments. Some items cannot "
            "be shipped internationally due to export restrictions. International orders over "
            "$200 qualify for free standard international shipping. Tracking is provided for "
            "all international shipments."
        ),
    },
    {
        "id": "kb_010",
        "title": "Damaged Items Policy",
        "content": (
            "If you receive a damaged or defective item, please contact us within 48 hours of "
            "delivery. Take clear photos of the damaged item and the packaging and email them to "
            "support@example.com with your order number. We will arrange a free return pickup "
            "and send a replacement at no additional cost. Alternatively, you may request a full "
            "refund. For fragile items, we use extra protective packaging; however, if damage "
            "occurs during transit we will work with the carrier to file a claim. Do not discard "
            "the original packaging until the claim is resolved. Replacement processing time is "
            "2-3 business days after damage is confirmed."
        ),
    },
    {
        "id": "kb_011",
        "title": "Subscription Cancellation",
        "content": (
            "You may cancel your subscription at any time from the 'Account Settings' page under "
            "'Subscriptions'. Cancellations take effect at the end of the current billing cycle; "
            "you retain access to premium features until then. We do not offer prorated refunds "
            "for mid-cycle cancellations. Annual subscribers who cancel within 14 days of "
            "renewal receive a full refund if no premium content was downloaded during that "
            "period. After cancellation, your account reverts to the free tier. Subscription "
            "data is retained for 90 days in case you decide to reactivate. To reactivate, "
            "simply choose a new plan from the Subscriptions page."
        ),
    },
    {
        "id": "kb_012",
        "title": "Bulk Orders",
        "content": (
            "We offer special pricing for bulk orders of 50 units or more. Contact our B2B "
            "sales team at bulk@example.com or call 1-800-555-0200 to request a custom quote. "
            "Discounts range from 10% for 50-99 units to 25% for 500+ units. Bulk orders "
            "require a minimum deposit of 30% upfront. Lead time for bulk orders is 5-10 "
            "business days depending on stock availability. We can accommodate custom branding, "
            "packaging, and labelling for corporate orders. Net-30 payment terms are available "
            "for established business customers with approved credit applications. Sample kits "
            "are available for evaluation before placing a bulk order."
        ),
    },
    {
        "id": "kb_013",
        "title": "Gift Cards",
        "content": (
            "Gift cards are available in denominations of $25, $50, $100, and $200. They can be "
            "purchased on our website or in store and are delivered via email within minutes. "
            "Gift cards never expire and have no fees. They can be used for any purchase on "
            "our website or in store. Multiple gift cards can be applied to a single order. "
            "Gift cards cannot be exchanged for cash except where required by law. Lost or "
            "stolen gift cards can be replaced if you have the original purchase receipt and "
            "order number. Gift cards are redeemable for physical products, digital downloads, "
            "and subscription plans."
        ),
    },
    {
        "id": "kb_014",
        "title": "Account Management",
        "content": (
            "Managing your account is easy via the 'My Account' section of our website. You can "
            "update personal information, change your password, manage saved payment methods, "
            "view order history, and track shipments. Two-factor authentication (2FA) is "
            "available for enhanced security. If you forget your password, click 'Forgot "
            "Password' on the login page to receive a reset link via email. To delete your "
            "account, contact support — data will be permanently removed within 30 days per our "
            "privacy policy. You can manage email notification preferences from Account Settings. "
            "Creating an account is free and provides faster checkout, order tracking, and "
            "access to exclusive member discounts."
        ),
    },
    {
        "id": "kb_015",
        "title": "Privacy Policy",
        "content": (
            "We are committed to protecting your personal information. We collect only the data "
            "necessary to process orders, provide support, and improve our services (name, "
            "email, shipping address, and payment information). We never sell your personal "
            "data to third parties. Data is shared only with trusted service providers (shipping "
            "carriers, payment processors) under strict confidentiality agreements. You have the "
            "right to access, correct, or delete your data at any time by contacting "
            "privacy@example.com. We use industry-standard encryption for all data in transit "
            "and at rest. Our full privacy policy is available at example.com/privacy. We comply "
            "with GDPR, CCPA, and other applicable data protection regulations."
        ),
    },
    {
        "id": "kb_016",
        "title": "Technical Support",
        "content": (
            "Our technical support team helps with product setup, software issues, firmware "
            "updates, and troubleshooting. Support is available Monday–Friday 8 AM–8 PM EST via "
            "live chat, email (techsupport@example.com), or phone (1-800-555-0201). For common "
            "issues, visit our online Knowledge Base at support.example.com which includes "
            "step-by-step guides and video tutorials. Remote desktop assistance is available for "
            "complex software issues. Most hardware issues can be diagnosed via our online "
            "diagnostic tool. If your device cannot be repaired remotely, we will arrange a "
            "warranty replacement or repair service. Please have your product serial number "
            "ready when contacting technical support."
        ),
    },
]


class CustomerSupportAgent:
    """
    AI-powered customer support agent using Retrieval-Augmented Generation (RAG).

    Combines ChromaDB vector search with OpenAI GPT-3.5-turbo to deliver
    accurate, context-grounded answers to customer queries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the agent with optional configuration overrides.

        Args:
            config: Dictionary with keys such as 'openai_api_key',
                    'chroma_persist_dir', 'llm_model', 'llm_temperature'.
        """
        self.config: Dict[str, Any] = config or {}
        self.is_initialized: bool = False

        # ChromaDB
        self._chroma_client: Any = None
        self.collection: Any = None

        # OpenAI
        self._openai_client: Optional[AsyncOpenAI] = None

        # Settings
        self._persist_dir: str = self.config.get(
            "chroma_persist_dir",
            os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        )
        self._collection_name: str = "customer_support_kb"
        self._llm_model: str = self.config.get(
            "llm_model", os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        )
        self._llm_temperature: float = float(
            self.config.get(
                "llm_temperature", os.getenv("LLM_TEMPERATURE", "0.7")
            )
        )
        self._openai_api_key: str = self.config.get(
            "openai_api_key", os.getenv("OPENAI_API_KEY", "")
        )
        self._base_url: Optional[str] = self.config.get(
            "llm_base_url", os.getenv("LLM_BASE_URL")
        )

        # Conversation Memory: {session_id: [messages]}
        self.history: Dict[str, List[Dict[str, str]]] = {}
        self.max_history: int = int(self.config.get("max_history", 10))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Asynchronously set up ChromaDB and OpenAI clients and ingest
        knowledge base documents if not already present.
        """
        try:
            logger.info("Initialising CustomerSupportAgent...")

            # Run synchronous ChromaDB setup in a thread so we don't block
            await asyncio.get_event_loop().run_in_executor(
                None, self._setup_chromadb
            )

            # OpenAI async client
            if not self._openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. "
                    "Please add it to your .env file."
                )
            self._openai_client = AsyncOpenAI(
                api_key=self._openai_api_key,
                base_url=self._base_url
            )

            self.is_initialized = True
            logger.info(
                "CustomerSupportAgent initialised successfully. "
                "Collection '%s' contains %d documents.",
                self._collection_name,
                self.collection.count(),
            )
        except Exception as exc:
            logger.exception("Failed to initialise CustomerSupportAgent: %s", exc)
            raise

    def _setup_chromadb(self) -> None:
        """Synchronous helper: create/load ChromaDB collection and ingest docs."""
        self._chroma_client = chromadb.PersistentClient(path=self._persist_dir)

        # Get-or-create the collection
        self.collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        existing_count = self.collection.count()
        if existing_count >= len(KNOWLEDGE_BASE):
            logger.info(
                "Knowledge base already populated (%d docs). Skipping ingestion.",
                existing_count,
            )
            return

        logger.info(
            "Ingesting %d knowledge base documents into ChromaDB...",
            len(KNOWLEDGE_BASE),
        )
        self.collection.upsert(
            ids=[doc["id"] for doc in KNOWLEDGE_BASE],
            documents=[doc["content"] for doc in KNOWLEDGE_BASE],
            metadatas=[{"title": doc["title"], "id": doc["id"]} for doc in KNOWLEDGE_BASE],
        )
        logger.info("Knowledge base ingestion complete.")

    async def _rag_search(self, query: str) -> str:
        """
        Search ChromaDB for relevant documents.
        """
        try:
            # Query ChromaDB for relevant documents
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
                include=['documents', 'metadatas', 'distances']
            )

            if not results or not results['documents'][0]:
                logger.info(f"No relevant documents found for query: {query}")
                return "No relevant documents found."

            formatted_results = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                formatted_results.append(
                    f"**{meta['title']}**\n{doc}"
                )

            return "\n\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return "Error searching knowledge base."

    async def process_query(self, query: str, session_id: str = "default", language: str = "en") -> str:
        """
        Process a user query end-to-end: RAG search -> LLM generation.
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        try:
            logger.info(f"Processing query for session {session_id} (lang: {language}): {query}")

            # Step 1: Retrieve relevant knowledge base context
            context = await self._rag_search(query)

            # Step 2: Prepare messages with history
            system_prompt = (
                "You are a helpful and professional customer support agent. "
                "Use ONLY the information provided in the context below to answer "
                "the customer's question. Be concise, friendly, and accurate. "
                f"IMPORTANT: Respond in the same language as the user query (Language code: {language}). "
                "If the context does not contain enough information to answer "
                "confidently, say so politely and direct the customer to contact "
                "support directly."
            )

            # Initialize history if new session
            if session_id not in self.history:
                self.history[session_id] = []

            # Construct messages for OpenAI
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add context for the current query
            context_msg = f"Context from our knowledge base:\n{context}\n\nUser Question: {query}"
            
            # Add relevant history (last N messages)
            for hist_msg in self.history[session_id][-self.max_history:]:
                messages.append(hist_msg)
            
            # Add current user message
            messages.append({"role": "user", "content": context_msg})

            # Step 3: Call OpenAI
            response = await self._openai_client.chat.completions.create(
                model=self._llm_model,
                temperature=self._llm_temperature,
                messages=messages,
            )

            answer: str = response.choices[0].message.content.strip()
            
            # Step 4: Update history
            self.history[session_id].append({"role": "user", "content": query})
            self.history[session_id].append({"role": "assistant", "content": answer})
            
            # Trim history to avoid context window blowup
            if len(self.history[session_id]) > self.max_history * 2:
                self.history[session_id] = self.history[session_id][-(self.max_history * 2):]

            logger.info("Query processed successfully.")
            return answer

        except Exception as exc:
            logger.exception("Error processing query: %s", exc)
            return (
                "I'm sorry, I encountered an error while processing your question. "
                "Please try again or contact our support team directly."
            )
