// canvas-datagrid ships no type declarations. We use it through a narrow
// imperative surface (build a grid, scroll a cell into view, repaint, dispose)
// in CsvViewer.svelte, so an untyped ambient module is sufficient.
declare module "canvas-datagrid";
