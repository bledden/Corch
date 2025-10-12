# Benchmark Redesign: 500 Tasks (250 Self-Contained + 250 Web-Search)

## Redundancy Analysis of Original 498 Tasks

### Tasks to Remove (248 redundant tasks)

**Category: basic_algorithms (50 → 20 kept, 30 removed)**
- REMOVE: Lines 140-188 - Generic "Implement advanced data structure variant 1-50" (50 filler tasks)
- REMOVE: Lines 198-246 - Generic "Implement algorithm optimization 1-50" (50 filler tasks)
- REMOVE: Lines 253-301 - Generic "Implement advanced algorithm 1-50" (50 filler tasks)
- REMOVE: Lines 309-407 - Generic "Implement advanced real-world task 1-100" (100 filler tasks)

**Redundancies to Remove (48 more):**
1. Sorting algorithms - Keep 3 (merge, quick, heap), remove 15 others
2. DP problems - Keep 10 unique, remove 10 similar
3. String algorithms - Keep 5, remove 5 duplicates
4. Graph algorithms - Keep 8, remove 5 variants
5. Tree operations - Keep 7, remove 5 similar
6. Linked list - Keep 3, remove 2
7. Number theory - Keep 5, remove 3
8. Compression - Keep 3, remove 3

**TOTAL REMOVED: 248 tasks**
**REMAINING: 250 high-quality self-contained tasks**

---

## NEW 250 Self-Contained Tasks (Optimized)

### Basic Algorithms (50 tasks)
1. Check if number is prime
2. Factorial using recursion
3. Reverse a string
4. Find maximum in list
5. Fibonacci sequence
6. Check palindrome
7. Remove duplicates from list
8. Count vowels
9. Find GCD
10. Leap year check
11. Power calculation (x^n)
12. Sum digits in number
13. Decimal to binary
14. Find LCM
15. Balanced parentheses
16. Second largest in array
17. Rotate array by k
18. Merge sorted arrays
19. Missing number in sequence
20. Check anagrams
21. Compound interest
22. Prime factors
23. Validate credit card (Luhn)
24. Pascal's triangle
25. Longest common prefix
26. Perfect number check
27. Manhattan distance
28. First non-repeating character
29. Roman to integer
30. Armstrong number
31. Array intersection
32. Square root (Newton's method)
33. Validate IPv4
34. Missing letter in sequence
35. Generate permutations
36. Valid sudoku check
37. Longest substring without repeats
38. Edit distance (Levenshtein)
39. Kth largest element
40. Linked list cycle detection
41. Reverse words in sentence
42. Majority element
43. Implement atoi
44. Find peak element
45. Valid parentheses combos
46. Find duplicates in array
47. Water trapping problem
48. Longest palindromic substring
49. Substring search (strStr)
50. Validate BST

### Data Structures (60 tasks)
51. Stack using list
52. Queue using two stacks
53. Min stack O(1)
54. Circular queue
55. LRU cache
56. Binary search tree insert/search
57. Balance BST
58. Tree height
59. Level order traversal
60. Serialize/deserialize tree
61. Trie for autocomplete
62. Heap with heapify
63. Median in stream
64. Priority queue
65. Graph adjacency list
66. BFS traversal
67. DFS traversal
68. Dijkstra's algorithm
69. Topological sort
70. Cycle detection in graph
71. Union-find (disjoint set)
72. Segment tree
73. Binary indexed tree (Fenwick)
74. Bloom filter
75. Skip list
76. Red-black tree insertion
77. AVL tree with rotations
78. B-tree insertion
79. KD-tree construction
80. Suffix tree
81. Sparse table RMQ
82. Fenwick tree updates
83. Doubly linked list
84. Reverse linked list
85. Linked list intersection
86. Clone graph
87. Bridge edges in graph
88. Articulation points
89. Strongly connected components
90. Kruskal's MST
91. Prim's MST
92. Floyd-Warshall
93. Bellman-Ford
94. A* pathfinding
95. Binary heap operations
96. Deque implementation
97. Circular buffer
98. Hash table with chaining
99. Consistent hashing ring
100. Cuckoo hash table
101. Treap (tree + heap hybrid)
102. Splay tree operations
103. Suffix array construction
104. Persistent data structure
105. Rope data structure
106. Interval tree
107. Range tree
108. Cartesian tree
109. Finger tree
110. Van Emde Boas tree

### Medium Algorithms (70 tasks)
111. Binary search
112. Merge sort
113. Quick sort
114. Heap sort
115. Sliding window maximum
116. Two pointer 3sum
117. Longest increasing subsequence
118. Knapsack 0/1
119. Unbounded knapsack
120. Subset sum
121. Coin change
122. Rod cutting
123. Matrix chain multiplication
124. Edit distance DP
125. Longest common subsequence
126. Longest palindromic subsequence
127. Egg dropping
128. Partition problem
129. Word break
130. Palindrome partitioning
131. Maximum subarray (Kadane)
132. Maximum product subarray
133. Stock buy/sell
134. Rain water trapping
135. Container most water
136. Jump game
137. Minimum path sum
138. Unique paths in grid
139. Climbing stairs
140. House robber
141. Decode ways
142. Word ladder
143. Regular expression matching
144. Wildcard pattern matching
145. Interleaving string
146. Scramble string validator
147. Distinct subsequences
148. Minimum window substring
149. Longest valid parentheses
150. Maximal rectangle
151. Largest rectangle histogram
152. Trapping rain water 2D
153. Dungeon game
154. Cherry pickup
155. Burst balloons
156. Remove boxes
157. Strange printer
158. Super egg drop
159. Student attendance record
160. Knight dialer
161. Number of music playlists
162. Pizza with 3n slices
163. Reduce array size to half
164. Stone game series
165. Minimum cost tree from leaf values
166. Last stone weight II
167. Tallest billboard
168. Profitable schemes
169. Number of ways to paint fence
170. Count different palindromic subsequences
171. Count unique BSTs
172. Unique BST generation
173. Restore IP addresses
174. Gray code generation
175. Subsets generation
176. Combination sum variants
177. Generate parentheses
178. Letter combinations phone
179. Palindrome permutation
180. Next permutation

### Hard Algorithms (70 tasks)
181. N-Queens solver
182. Sudoku solver
183. Knight's tour
184. Graph coloring
185. Hamiltonian path
186. Traveling salesman
187. Maximum flow (Ford-Fulkerson)
188. Min-cost max-flow
189. Bipartite matching
190. Hungarian algorithm
191. KMP string matching
192. Rabin-Karp pattern search
193. Boyer-Moore string search
194. Aho-Corasick multi-pattern
195. Suffix array construction
196. Manacher's palindrome algorithm
197. Z-algorithm pattern matching
198. Fast Fourier transform
199. Convex hull (Graham scan)
200. Closest pair of points
201. Line intersection detector
202. Voronoi diagram
203. Delaunay triangulation
204. Interval scheduling
205. Job sequencing deadlines
206. Fractional knapsack
207. Activity selection
208. Huffman coding
209. LZW compression
210. Run-length encoding
211. Burrows-Wheeler transform
212. Arithmetic coding
213. RSA encryption basics
214. Diffie-Hellman key exchange
215. Miller-Rabin primality
216. Pollard's rho factorization
217. Extended Euclidean algorithm
218. Modular exponentiation
219. Chinese remainder theorem
220. Fast modular multiplication
221. Elliptic curve operations
222. SHA-256 hash
223. Merkle tree construction
224. Count-min sketch
225. HyperLogLog cardinality
226. Locality-sensitive hashing
227. SimHash similarity
228. MinHash Jaccard similarity
229. Linear programming simplex
230. Network simplex algorithm
231. Hungarian assignment
232. Stable marriage problem
233. Gale-Shapley algorithm
234. Auction algorithm
235. Edmonds-Karp max flow
236. Dinic's max flow
237. Push-relabel max flow
238. Stoer-Wagner min cut
239. Gomory-Hu tree
240. Christofides TSP approximation
241. 2-approximation vertex cover
242. Greedy set cover
243. Dynamic programming TSP
244. Branch and bound TSP
245. Simulated annealing
246. Genetic algorithm framework
247. Particle swarm optimization
248. Ant colony optimization
249. Tabu search
250. Hill climbing optimization

---

## NEW 250 Web-Search-Requiring Tasks

### Modern Framework Integration (60 tasks)
1. Implement Next.js 15 App Router with Server Actions
2. Create React 19 component with new use() hook
3. Build Vue 3.5 Composition API with TypeScript
4. Implement Svelte 5 runes reactive system
5. Create Angular 18 standalone component
6. Build Astro 4.x island architecture component
7. Implement SolidJS reactive primitives
8. Create Qwik resumability pattern
9. Build Fresh framework Deno edge function
10. Implement Remix 2.x loader/action pattern
11. Create tRPC v11 end-to-end typesafe API
12. Build GraphQL Yoga v5 server
13. Implement Prisma 5.x schema with relations
14. Create Drizzle ORM type-safe queries
15. Build TypeORM 0.3.x migration
16. Implement Sequelize 7.x associations
17. Create Mongoose 8.x schema with validation
18. Build Zod 3.x schema validation
19. Implement Yup validation with TypeScript
20. Create Valibot schema validator
21. Build Tanstack Query v5 data fetching
22. Implement SWR 2.x data fetching hooks
23. Create Redux Toolkit 2.x slice
24. Build Zustand 4.x store with persist
25. Implement Jotai atoms pattern
26. Create Recoil selector with async
27. Build Pinia stores for Vue
28. Implement Nanostores with React
29. Create XState v5 state machine
30. Build Robot state machine
31. Implement Vitest 2.x test suite
32. Create Playwright E2E tests
33. Build Cypress 13.x component tests
34. Implement Testing Library queries
35. Create MSW 2.x API mocks
36. Build Storybook 8.x stories
37. Implement Vite 5.x plugin
38. Create Turbopack configuration
39. Build esbuild custom plugin
40. Implement Rollup 4.x config
41. Create Webpack 5 module federation
42. Build Parcel 2.x bundler config
43. Implement Biome formatter/linter
44. Create ESLint 9.x flat config
45. Build Prettier 3.x plugin
46. Implement TypeScript 5.4 satisfies operator
47. Create TypeScript 5.4 const type parameters
48. Build Bun 1.x HTTP server
49. Implement Deno 2.x Fresh middleware
50. Create Node.js 22 native fetch usage
51. Build Hono edge framework route
52. Implement Elysia Bun web framework
53. Create Fastify 5.x plugin
54. Build Express 5.x middleware
55. Implement NestJS 10.x module
56. Create tRPC Next.js App Router integration
57. Build Nuxt 3.x composables
58. Implement SvelteKit 2.x form actions
59. Create Solid Start SSR route
60. Build Qwik City middleware

### Cloud & Infrastructure (50 tasks)
61. Implement AWS Lambda Node.js 20 function
62. Create GCP Cloud Function Gen2 Python
63. Build Azure Functions v4 isolated worker
64. Implement Cloudflare Workers AI binding
65. Create Vercel Edge Functions middleware
66. Build Netlify Edge Functions
67. Implement AWS CDK v2 stack
68. Create Terraform AWS provider 5.x
69. Build Pulumi TypeScript AWS resources
70. Implement AWS S3 presigned URL v3 SDK
71. Create AWS DynamoDB single-table design
72. Build AWS EventBridge rules
73. Implement AWS Step Functions workflow
74. Create AWS AppSync GraphQL resolver
75. Build AWS Cognito authentication flow
76. Implement GCP Firestore security rules
77. Create GCP Cloud Run service
78. Build GCP Pub/Sub topic subscription
79. Implement Azure Cosmos DB queries
80. Create Azure Service Bus messaging
81. Build Docker Compose v2.x multi-stage
82. Implement Kubernetes 1.29 deployment
83. Create Helm chart v3
84. Build ArgoCD application manifest
85. Implement Istio service mesh config
86. Create Prometheus metrics exporter
87. Build Grafana dashboard JSON
88. Implement OpenTelemetry tracing
89. Create Datadog APM integration
90. Build New Relic custom instrumentation
91. Implement Sentry error tracking
92. Create LogRocket session replay
93. Build Stripe API v2024 payment intent
94. Implement PayPal Checkout v2
95. Create Plaid API bank connection
96. Build Twilio SendGrid v4 email
97. Implement SendGrid dynamic templates
98. Create Postmark transactional email
99. Build Auth0 Next.js integration
100. Implement Clerk authentication
101. Create Supabase auth with RLS
102. Build Firebase Auth with custom claims
103. Implement Okta OIDC flow
104. Create OneLogin SAML integration
105. Build AWS Cognito user pool
106. Implement Azure AD B2C custom policy
107. Create Keycloak realm configuration
108. Build FusionAuth tenant setup
109. Implement SuperTokens session management
110. Create Magic.link passwordless auth

### AI & ML Integration (50 tasks)
111. Implement OpenAI GPT-4 Turbo API
112. Create Anthropic Claude 3.5 Sonnet
113. Build Google Gemini 2.0 Flash
114. Implement Cohere Command-R Plus
115. Create Mistral Large 2 API
116. Build Meta Llama 3.2 inference
117. Implement OpenAI Whisper v3 transcription
118. Create ElevenLabs voice synthesis
119. Build Replicate model deployment
120. Implement HuggingFace Inference API
121. Create LangChain 0.3.x chain
122. Build LlamaIndex 0.11.x query engine
123. Implement Semantic Kernel plugin
124. Create AutoGen multi-agent system
125. Build CrewAI agent workflow
126. Implement Haystack 2.x pipeline
127. Create Guidance structured output
128. Build LMQL query language
129. Implement Instructor structured extraction
130. Create Outlines constrained generation
131. Build ChromaDB vector store
132. Implement Pinecone index with metadata
133. Create Weaviate schema and import
134. Build Qdrant collection with filters
135. Implement Milvus partition keys
136. Create FAISS index with IVF
137. Build Pgvector PostgreSQL extension
138. Implement LanceDB embedded vectors
139. Create Vespa query with ranking
140. Build OpenSearch vector search
141. Implement Langfuse tracing
142. Create LangSmith evaluation
143. Build Weights & Biases Weave logging
144. Implement Helicone proxy caching
145. Create LiteLLM unified API
146. Build Portkey AI gateway
147. Implement Braintrust prompt management
148. Create Promptfoo test suite
149. Build OpenLLMetry observability
150. Implement Arize Phoenix monitoring
151. Create Unstructured.io document parser
152. Build LlamaParse PDF extraction
153. Implement Firecrawl web scraping
154. Create Jina AI Embeddings v3
155. Build Cohere Rerank v3 model
156. Implement Voyage AI embeddings
157. Create Nomic Embed text model
158. Build BGE-M3 multilingual embeddings
159. Implement E5-Mistral-7B embeddings
160. Create Snowflake Arctic Embed

### Security & Compliance (40 tasks)
161. Implement OWASP Top 10 2023 fixes
162. Create CVE-2024-XXXXX vulnerability patch
163. Build OAuth 2.1 authorization server
164. Implement PKCE flow for SPAs
165. Create JWT with RS256 signing
166. Build PASETO token implementation
167. Implement WebAuthn passkey registration
168. Create FIDO2 authentication
169. Build rate limiting with Redis
170. Implement CAPTCHA v3 integration
171. Create Content Security Policy headers
172. Build CORS configuration
173. Implement Helmet.js security headers
174. Create HTTPS certificate pinning
175. Build HSTS with preload
176. Implement SRI for external resources
177. Create XSS prevention middleware
178. Build SQL injection prevention
179. Implement CSRF token validation
180. Create secure session management
181. Build bcrypt password hashing
182. Implement Argon2 password hashing
183. Create PBKDF2 key derivation
184. Build scrypt password storage
185. Implement AES-256-GCM encryption
186. Create ChaCha20-Poly1305 encryption
187. Build RSA-OAEP key encryption
188. Implement ECDH key exchange
189. Create Ed25519 signature
190. Build X.509 certificate validation
191. Implement OCSP stapling
192. Create Certificate Transparency logs
193. Build DNSSEC validation
194. Implement TLS 1.3 configuration
195. Create mTLS mutual authentication
196. Build Zero Trust network access
197. Implement GDPR compliance checks
198. Create CCPA data deletion API
199. Build HIPAA audit logging
200. Implement SOC 2 control evidence

### API & Protocol Integration (50 tasks)
201. Implement REST API with OpenAPI 3.1
202. Create GraphQL schema with Federation v2
203. Build gRPC service with protobuf
204. Implement tRPC v11 router
205. Create WebSocket server Socket.io v4
206. Build Server-Sent Events endpoint
207. Implement WebRTC peer connection
208. Create MQTT broker subscription
209. Build AMQP RabbitMQ consumer
210. Implement Apache Kafka producer
211. Create Redis Streams consumer group
212. Build Webhooks with signature verification
213. Implement Stripe webhook handler
214. Create GitHub webhook processor
215. Build Slack Events API handler
216. Implement Discord bot interactions
217. Create Twitch EventSub subscription
218. Build Shopify webhook verification
219. Implement PayPal IPN handler
220. Create Mailgun webhook parsing
221. Build Sendgrid event webhook
222. Implement Twilio webhook signature
223. Create Plaid webhook verification
224. Build Stripe Connect OAuth
225. Implement GitHub OAuth App
226. Create Google OAuth 2.0 flow
227. Build Microsoft Azure AD OAuth
228. Implement Twitter OAuth 2.0 PKCE
229. Create LinkedIn OAuth integration
230. Build Spotify OAuth with refresh
231. Implement Apple Sign In JWT
232. Create Facebook Login Graph API
233. Build Discord OAuth2 bot authorization
234. Implement Notion API integration
235. Create Airtable API with pagination
236. Build Contentful content delivery
237. Implement Sanity GROQ queries
238. Create Strapi v4 REST API
239. Build Directus GraphQL queries
240. Implement Hasura metadata actions
241. Create Supabase realtime subscriptions
242. Build Firebase Realtime Database rules
243. Implement MongoDB Change Streams
244. Create PostgreSQL LISTEN/NOTIFY
245. Build MySQL binary log replication
246. Implement Redis Pub/Sub patterns
247. Create Apache Pulsar consumer
248. Build NATS JetStream consumer
249. Implement AWS Kinesis shard reader
250. Create GCP Dataflow pipeline

---

## Complete Request Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER REQUEST ENTRY                          │
│  (CLI, API, or Direct Function Call)                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   TASK CLASSIFICATION   │
                    │  (WebSearchRouter)      │
                    │                         │
                    │  - detect_needs_web     │
                    │    _search()            │
                    │  - Returns: (bool,      │
                    │    patterns, conf)      │
                    └──────────┬──────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
     ┌──────────▼─────────┐        ┌─────────▼──────────┐
     │  SELF-CONTAINED    │        │  WEB SEARCH        │
     │  confidence < 0.5  │        │  confidence >= 0.5 │
     └──────────┬─────────┘        └─────────┬──────────┘
                │                            │
                │                   ┌────────▼────────┐
                │                   │  SEARCH ROUTING │
                │                   │  select_search  │
                │                   │  _method()      │
                │                   │                 │
                │                   │  Strategy:      │
                │                   │  - cheapest     │
                │                   │  - fastest      │
                │                   │  - quality      │
                │                   │  - balanced     │
                │                   └────────┬────────┘
                │                            │
                │                   ┌────────▼────────┐
                │                   │  EXECUTE SEARCH │
                │                   │                 │
                │                   │  Options:       │
                │                   │  - Tavily       │
                │                   │  - Perplexity   │
                │                   │  - Gemini 2.5   │
                │                   └────────┬────────┘
                │                            │
                │                   ┌────────▼────────┐
                │                   │  ENRICH CONTEXT │
                │                   │  - Add search   │
                │                   │    results      │
                │                   │  - Add metadata │
                │                   │  - Track cost   │
                │                   └────────┬────────┘
                │                            │
                └────────────┬───────────────┘
                             │
            ┌────────────────▼───────────────┐
            │   ROUTING: SEQUENTIAL VS       │
            │   BASELINE                     │
            │                                │
            │   Benchmark mode: Both paths   │
            │   Production: User choice      │
            └────────────┬───────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼─────────┐          ┌─────────▼──────────┐
│  BASELINE PATH   │          │  SEQUENTIAL PATH   │
│  (Single Model)  │          │  (5-Stage)         │
└────────┬─────────┘          └─────────┬──────────┘
         │                              │
         │                    ┌─────────▼──────────┐
         │                    │  STAGE 1: ARCHITECT│
         │                    │  - Design approach │
         │                    │  - Output: Markdown│
         │                    │  - Timeout: 270s   │
         │                    └─────────┬──────────┘
         │                              │
         │                    ┌─────────▼──────────┐
         │                    │  STAGE 2: CODER    │
         │                    │  - Implement code  │
         │                    │  - Output: Code    │
         │                    │  - Timeout: 216s   │
         │                    └─────────┬──────────┘
         │                              │
         │                    ┌─────────▼──────────┐
         │                    │  STAGE 3: REVIEWER │
         │                    │  - Validate code   │
         │                    │  - Output: JSON    │
         │                    │  - Timeout: 180s   │
         │                    └─────────┬──────────┘
         │                              │
         │                    ┌─────────▼──────────┐
         │                    │  STAGE 4: REFINER  │
         │                    │  - Fix issues      │
         │                    │  - Iterate up to 2x│
         │                    │  - Timeout: 180s   │
         │                    └─────────┬──────────┘
         │                              │
         │                    ┌─────────▼──────────┐
         │                    │  STAGE 5: DOCUMENTER│
         │                    │  - Add docs        │
         │                    │  - Output: Markdown│
         │                    │  - Timeout: 180s   │
         │                    └─────────┬──────────┘
         │                              │
         └──────────────┬───────────────┘
                        │
            ┌───────────▼───────────┐
            │  HALLUCINATION CHECK  │
            │  HallucinationDetector│
            │                       │
            │  Checks:              │
            │  - Unfounded claims   │
            │  - Contradictions     │
            │  - Fabricated details │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  QUALITY SCORING      │
            │                       │
            │  Sequential:          │
            │  - Multi-stage quality│
            │  - Pass@1: >0.7 + no  │
            │    hallucinations     │
            │                       │
            │  Baseline:            │
            │  - Code heuristics    │
            │  - Pass@1: has code + │
            │    logic + no halluc  │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  RESULT AGGREGATION   │
            │                       │
            │  Fields:              │
            │  - task_id            │
            │  - category           │
            │  - method             │
            │  - pass (binary)      │
            │  - quality_score      │
            │  - duration           │
            │  - hallucination      │
            │  - output             │
            │  - needs_external_info│
            │  - search_confidence  │
            │  - matched_patterns   │
            │  - search_method_used │
            │  - search_cost        │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  W&B WEAVE LOGGING    │
            │  @weave.op()          │
            │                       │
            │  - All function calls │
            │  - Nested traces      │
            │  - Performance metrics│
            │  - Cost tracking      │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  FINAL STATISTICS     │
            │                       │
            │  Overall:             │
            │  - Pass@1 %           │
            │  - Total successes    │
            │  - Hallucinations     │
            │  - Avg quality        │
            │  - Avg duration       │
            │                       │
            │  By Task Type:        │
            │  - Self-contained     │
            │  - Web-search         │
            │                       │
            │  By Search Method:    │
            │  - Tavily usage       │
            │  - Perplexity usage   │
            │  - Total search cost  │
            └───────────────────────┘
```

## Component Touch Points

### Every Request Touches:

1. **Entry Point** (`run_500_task_benchmark.py` or API)
2. **WebSearchRouter** (`web_search_router.py`)
   - `detect_needs_web_search()`
   - `select_search_method()` (if web search needed)
   - `route_task()` (if web search needed)
3. **LLM Client** (`agents/llm_client.py`)
   - `MultiAgentLLMOrchestrator`
   - `execute_agent_task()`
4. **Orchestrator** (Sequential OR Baseline)
   - **Sequential**: `collaborative_orchestrator.py`
     - `CollaborativeOrchestrator.collaborate()`
     - `_architect_stage()`
     - `_coder_stage()`
     - `_reviewer_stage()`
     - `_coder_refine_stage()` (iterative)
     - `_documenter_stage()`
     - `_call_llm_with_timeout()`
   - **Baseline**: `agents/llm_client.py`
     - `execute_agent_task("coder", task)`
5. **Hallucination Detection** (`agents/hallucination_detector.py`)
   - `HallucinationDetector.detect()`
6. **Quality Scoring** (`run_500_task_benchmark.py`)
   - `run_sequential()` - Pass@1 calculation
   - `run_baseline()` - Pass@1 calculation
7. **Weave Logging** (W&B - all `@weave.op()` decorated functions)
8. **Result Aggregation** (`run_500_task_benchmark.py`)
   - Metrics calculation
   - Task type breakdown
   - Search method statistics

### Files Modified/Created:
- `run_500_task_benchmark.py` - Main benchmark orchestrator
- `web_search_router.py` - Web search detection and routing
- `collaborative_orchestrator.py` - Sequential 5-stage pipeline
- `agents/llm_client.py` - LLM API wrapper
- `agents/hallucination_detector.py` - Hallucination detection
- `config.yaml` - Model and API configuration

---

## Execution Plan

### Phase 1: Smoke Tests (10 tasks)
- 5 self-contained: prime check, factorial, reverse string, palindrome, GCD
- 5 web-search: Next.js 15, React 19, Stripe 2024, OpenAI GPT-4 Turbo, OWASP 2023

### Phase 2: Full Benchmark (~500 tasks)
- 250 self-contained (sequential + baseline) = 500 runs
- 250 web-search (sequential + baseline) = 500 runs
- **TOTAL: 1000 runs** (estimated 15-20 hours)

### Phase 3: Comparison
- Original baseline (from any completed tasks in first run)
- New optimized benchmark results
- Early problem detection by comparing pass rates
