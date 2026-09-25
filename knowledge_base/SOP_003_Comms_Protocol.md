# SOP 003: Secure Communications Handshake
## Protocol
All outbound tactical messages must use AES-256 encryption.
The daily cryptographic key is retrieved from the offline key-vault terminal at 00:00 GMT.
If the primary satellite link is lost, systems automatically failover to High Frequency (HF) band 14.5 MHz.