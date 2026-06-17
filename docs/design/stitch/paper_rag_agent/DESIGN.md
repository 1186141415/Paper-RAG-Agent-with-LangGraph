---
name: Paper RAG Agent
colors:
  surface: '#f9f9ff'
  surface-dim: '#cfdaf1'
  surface-bright: '#f9f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f0f3ff'
  surface-container: '#e7eeff'
  surface-container-high: '#dee8ff'
  surface-container-highest: '#d8e3fa'
  on-surface: '#111c2c'
  on-surface-variant: '#43474e'
  inverse-surface: '#263142'
  inverse-on-surface: '#ebf1ff'
  outline: '#74777f'
  outline-variant: '#c4c6cf'
  surface-tint: '#455f88'
  primary: '#002045'
  on-primary: '#ffffff'
  primary-container: '#1a365d'
  on-primary-container: '#86a0cd'
  inverse-primary: '#adc7f7'
  secondary: '#1960a3'
  on-secondary: '#ffffff'
  secondary-container: '#7db6ff'
  on-secondary-container: '#00477f'
  tertiary: '#321b00'
  on-tertiary: '#ffffff'
  tertiary-container: '#4f2e00'
  on-tertiary-container: '#c6955e'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d6e3ff'
  primary-fixed-dim: '#adc7f7'
  on-primary-fixed: '#001b3c'
  on-primary-fixed-variant: '#2d476f'
  secondary-fixed: '#d3e4ff'
  secondary-fixed-dim: '#a2c9ff'
  on-secondary-fixed: '#001c38'
  on-secondary-fixed-variant: '#004881'
  tertiary-fixed: '#ffddba'
  tertiary-fixed-dim: '#f2bc82'
  on-tertiary-fixed: '#2b1700'
  on-tertiary-fixed-variant: '#633f0f'
  background: '#f9f9ff'
  on-background: '#111c2c'
  surface-variant: '#d8e3fa'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  mono-code:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 48px
  container-max: 1280px
  sidebar-width: 280px
---

## Brand & Style
The design system is engineered for an **Academic-Tech** crossover, specifically tailored for researchers, data scientists, and engineers who interact with Retrieval-Augmented Generation (RAG) systems. The brand personality is authoritative yet innovative—balancing the gravity of traditional academic publishing with the speed of modern AI development.

The visual style follows a **Modern Corporate** aesthetic with a heavy emphasis on **Information Density**. It utilizes a systematic approach to whitespace and hierarchy to ensure that long-form technical content remains readable while complex agentic logs remain scannable. The UI evokes a sense of reliability, precision, and intellectual rigor.

## Colors
The palette is rooted in "Deep Academic Blue" to establish trust and authority. This is contrasted by "Innovation Blue," which is reserved strictly for interactive affordances like buttons, links, and active states. 

- **Primary (#1A365D):** Used for persistent structural elements like sidebars, headers, and primary branding.
- **Secondary (#2B6CB0):** The action color. Used for CTA buttons, selection states, and focus rings.
- **Background (#F7FAFC):** A soft, low-strain grey that mimics high-quality bond paper, reducing eye fatigue during long reading sessions.
- **Surface (#FFFFFF):** Used for content containers (cards) to provide a "lifted" feel against the paper-grey background.
- **Status Colors:** Success Green and Alert Amber are used purposefully for Agent "sufficiency" checks—indicating when the AI has found enough evidence to support a claim.

## Typography
This design system utilizes **Inter** as the primary typeface for its exceptional legibility and neutral, systematic tone. **JetBrains Mono** is employed for "Agent Trace" logs and technical data, providing a distinct visual "mode" for raw machine output versus synthesized human-readable text.

- **Scale:** High-contrast scale for headlines to aid in document navigation.
- **Body:** Body-md is optimized for the main research output, while Body-sm is used for metadata and sidebar annotations.
- **Logs:** All agent thought processes, tool calls, and API responses must be rendered in `mono-code` to distinguish "machine thinking" from "final response."

## Layout & Spacing
The layout follows a **Hybrid Grid** model. The main application shell uses a fixed left sidebar for navigation and document management, while the central workspace is a fluid container that optimizes for reading width (max-width of 800px for text-heavy content).

- **Margins:** A standard 24px (lg) margin is used for main containers.
- **Gutters:** 16px (md) gutters between cards and columns.
- **Stacking:** Elements within a card (e.g., title to body) use an 8px (sm) vertical rhythm.
- **Mobile:** On mobile, the sidebar collapses into a drawer, and horizontal margins reduce to 16px.

## Elevation & Depth
Depth is conveyed through **Tonal Layering** supplemented by **Ambient Shadows**. This approach maintains a professional, flat-ish appearance while providing necessary cues for interactivity.

- **Level 0 (Background):** Soft Paper Grey (#F7FAFC).
- **Level 1 (Cards/Surface):** Clean White (#FFFFFF) with a 1px border (#E2E8F0) and a very soft, diffused shadow: `0 1px 3px rgba(0,0,0,0.05)`.
- **Level 2 (Dropdowns/Modals):** Clean White (#FFFFFF) with a more pronounced shadow: `0 10px 15px -3px rgba(0,0,0,0.1)`.
- **Active State:** Elements being dragged or interacted with use a blue-tinted shadow to reinforce the "Innovation Blue" secondary color.

## Shapes
The design system uses a **Soft (0.25rem)** roundedness level to maintain a professional and disciplined appearance. 

- **Buttons & Inputs:** 4px (0.25rem) corner radius.
- **Cards & Large Containers:** 8px (0.5rem) corner radius.
- **Status Pills:** Fully rounded (pill-shaped) to distinguish them from functional buttons.

## Components
Consistent component behavior is critical for a tool used in data synthesis.

- **Buttons:** Primary buttons use `secondary_color_hex` (Innovation Blue) with white text. Ghost buttons use `primary_color_hex` for text with no fill, used for secondary actions like "Export" or "View Source."
- **Agent Status Indicators:** Use a pulse animation in `secondary_color_hex` when the agent is active. Use the Success/Alert colors for terminal states.
- **Source Cards:** Used for referencing PDFs or web results. These should feature a small icon (PDF/Web) in the top right and use `body-sm` for the snippet.
- **Trace Logs:** A collapsible terminal component using `mono-code` and a dark background (#1A202C) to simulate a developer environment within the academic workspace.
- **Inputs:** High-contrast borders (#CBD5E0) that thicken and change to Innovation Blue on focus.
- **Citations:** Small, superscripted chips that, when hovered, highlight the corresponding "Source Card" in the sidebar.