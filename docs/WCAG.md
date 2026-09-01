# Accessibility / WCAG 2.2 AA baseline

DataGuard targets WCAG 2.2 AA for the web interface.

Implemented baseline:

- `lang="fr-CA"` and a single main landmark;
- skip link to the main content;
- explicit labels for form controls;
- semantic table headers and caption;
- live status messaging for analysis feedback;
- visible keyboard focus indicators;
- reduced-motion support;
- dialog labelling and accessible relationships.

The repository includes an automated structural accessibility test. It is intentionally not presented as proof of WCAG conformance. Before a government production deployment, perform manual keyboard/screen-reader testing and an automated axe/WCAG scan against the deployed application, including authenticated and error states.
