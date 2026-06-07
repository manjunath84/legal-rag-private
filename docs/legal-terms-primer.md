# Legal Terms Primer — Plain English

*A quick reference for the legal document types used in this project's sample corpus.
No law degree required.*

---

## NDA — Non-Disclosure Agreement

**One-line definition:** A promise to keep secrets.

Two parties agree: *"anything you share with me in confidence, I will not tell anyone else."*

### Real-world examples
- A company asks a job candidate to sign an NDA before a technical interview so the candidate doesn't leak the product roadmap after learning about it.
- Two companies considering a merger sign an NDA before sharing financials with each other.
- A startup signs an NDA with a freelance developer before showing them proprietary source code.

### Key things an NDA covers

| Clause | What it means in plain English |
|---|---|
| **Confidential Information** | What counts as a "secret" — source code, business plans, customer lists, etc. |
| **Term** | How long the agreement lasts (our sample NDA: 3 years) |
| **Obligations** | What the receiving party must (and must not) do with the secret |
| **Governing Law** | Which state's courts would handle a dispute (our sample NDA: Delaware) |
| **Termination** | How either party can end the agreement (our sample NDA: 30 days written notice) |

### Sample document in this project
`data/legal/mutual-nda.txt` — between **Northwind Analytics, Inc.** and **Cedar Grove Software, LLC**. It's "mutual" because both parties are sharing secrets with each other (as opposed to a one-way NDA where only one party shares).

---

## MSA — Master Services Agreement

**One-line definition:** The main contract when one company hires another to do ongoing work.

It sets the ground rules for the whole business relationship upfront — so the two companies don't have to renegotiate the basics every time they start a new project together.

### Real-world examples
- A hospital hires a cloud software company to host their patient records system. The MSA sets payment terms, liability limits, and data security rules for the entire relationship.
- A law firm hires a document review vendor. The MSA covers confidentiality, deliverable standards, and how disputes are handled — once, upfront, for all future work.

### Key things an MSA covers

| Clause | What it means in plain English |
|---|---|
| **Scope of Services** | What work the vendor will actually do |
| **Payment Terms** | When and how invoices get paid (our sample MSA: within 45 days) |
| **Liability Cap** | The maximum amount either party can be sued for (our sample MSA: 12 months of fees) |
| **Governing Law** | Which state's courts handle disputes (our sample MSA: New York) |
| **Termination** | How to end the relationship cleanly |

### Sample document in this project
`data/legal/services-agreement.md` — between **Atlas Cloud Services** and **Riverside Health Group**. Atlas is the vendor (cloud services); Riverside is the client (a healthcare provider).

---

## NDA vs. MSA — side by side

| | NDA | MSA |
|---|---|---|
| **Purpose** | Keep secrets before/during a relationship | Govern ongoing paid work |
| **When signed** | Before sharing sensitive information | Before starting a business engagement |
| **Money involved?** | Usually no | Yes — payments, invoices, liability caps |
| **Analogy** | A handshake before talking | The contract for actually doing business |

---

## Why these two documents are in the test corpus

They are the **most common documents** any legal or business team handles — every company deals with NDAs and MSAs regularly.

More importantly for this project, they make a *hard* retrieval test:

- Both share vocabulary: "governing law", "termination", "confidential information", "agreement"
- Both have a "Governing Law" section — one pointing to Delaware, the other to New York
- A naive keyword search for "governing law" surfaces *both*, and the wrong one can rank first

This is exactly why the chatbot needs to correctly attribute which clause belongs to which document — and why CRAG's "I don't know" behavior matters. Citing the wrong contract's terms in a legal memo is a serious professional mistake; lawyers have been sanctioned for hallucinated citations.

---

## Other legal term that appears in the corpus

**Appellate Opinion** (`data/legal/opinion-altman-v-brightwave.md`) — a written ruling from a higher court reviewing a lower court's decision. In this fictional case (*Altman v. Brightwave*), the court applies the "ABC test" to decide whether a worker is an employee or an independent contractor. Appellate opinions are important in legal research because they establish precedent — rules that lower courts must follow in similar cases.
