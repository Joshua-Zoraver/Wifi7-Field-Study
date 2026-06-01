/**
 * Vite config for the SIGNAL predictor tool, set up to deploy as a
 * subdirectory of a GitHub Pages site.
 *
 * IMPORTANT — set `base` to match your actual deployment path.
 *
 * If your repo is `username/wifi7-study` and Pages serves it at
 *     https://username.github.io/wifi7-study/
 * and the tool lives at
 *     https://username.github.io/wifi7-study/tool/
 * then the `base` you want is `/wifi7-study/tool/`.
 *
 * If you're using a custom domain that serves the field-study site at
 * the root (e.g. https://wifi7study.com/ → field study,
 *              https://wifi7study.com/tool/ → this tool)
 * then `base` should be `/tool/`.
 *
 * Get this wrong and Vite will look for assets (predictor.js, fonts,
 * built CSS) at the wrong path and the production build will be a blank
 * page. It works in `npm run dev` regardless because the dev server
 * doesn't apply `base`. So: test the *built* output (`npm run build`
 * + `npm run preview`) before pushing to Pages.
 */
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],

  // ← CHANGE THIS to match your repo / deployment path
  base: '/REPO_NAME/tool/',
});
