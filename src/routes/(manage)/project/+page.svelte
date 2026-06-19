<script lang="ts">
  // Manage route ("/project?id=<uuid>"): a thin wrapper that renders the
  // ManagePanel for the project named in the URL. Keyed on the id so that
  // clicking another project in the sidebar (which only changes `?id=`, not the
  // route) remounts the panel with a fresh load + indexing subscription, instead
  // of relying on an effect to re-fetch in place.
  import { page } from "$app/state";
  import { goto } from "$app/navigation";
  import ManagePanel from "$lib/project/ManagePanel.svelte";

  const id = $derived(page.url.searchParams.get("id") ?? "");
</script>

{#if id}
  {#key id}
    <ManagePanel projectId={id} />
  {/key}
{:else}
  {(goto("/"), "")}
{/if}
