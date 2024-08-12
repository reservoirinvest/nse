# Objective
- [ ] Show the flow for click command-line-interface


```mermaid
flowchart LR
    A["get_market()"] --> B["NSE"] & C["SNP"]
    B --> F["make_nse_nakeds()"] & D["get_portfolio()"] & E["get_openords()"]
    F --> H["save"] & I["fnos"]
    C --> D & E & G["make_snp_nakeds()"]
    G --> J["save"] & K["fnos"]
    
    style A fill:#2962FF,color:#FFFFFF
    style B fill:#FF6D00
    style C fill:#00C853
    style F stroke:#FF6D00
    style G fill:#00C853
   ```