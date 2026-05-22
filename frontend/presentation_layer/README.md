# Presentation Layer

This directory contains the Vue 3 static demo UI for the NSFW detection workflow.

## Current Status
- Two-page flow: upload page and result page
- Mock-driven, but shaped around the real backend contract
- Desktop-first layout with mobile-safe stacking

## Planned Runtime
```powershell
npm install
npm run dev
```

## Notes
- Real API integration is intentionally deferred.
- The current mock data mirrors `/api/health` and `/api/detect`.
- Static UI planning details are documented in `C:\Code\net_sec\staticUI.md`.
