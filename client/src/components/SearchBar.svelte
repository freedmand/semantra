<script lang="ts">
  import { createEventDispatcher, tick } from "svelte";
  import type { Preference } from "../types";
  const dispatch = createEventDispatcher();

  export let preferences: { [key: string]: Preference };

  $: preferenceValues = Object.values(preferences).filter(
    (preference) => preference.weight !== 0
  );

  function getSearchKey(..._reactiveArgs: any[]) {
    return JSON.stringify({
      query: value,
      preferences: preferenceValues || [],
    });
  }

  let value = "";
  let lastSearchKey = getSearchKey();

  let preferenceContainer: HTMLDivElement;

  export async function scrollToBottomOfPreferences() {
    if (preferenceContainer != null) {
      // Wait for a tick so new preferences are reflected
      await tick();
      preferenceContainer.scrollTop = preferenceContainer.scrollHeight;
    }
  }

  $: searchKey = getSearchKey(value, preferenceValues);
  $: searchOutdated = searchKey !== lastSearchKey;

  function search() {
    dispatch("search", value);
    lastSearchKey = searchKey;
  }
</script>

<div class="flex flex-1 flex-col">
  <div class="flex items-center relative flex-1">
    <input
      class="bg-white py-2 px-4 pl-12 font-mono w-full rounded border-black border"
      class:bg-yellow-50={searchOutdated}
      class:border-yellow-600={searchOutdated}
      class:border-dashed={searchOutdated}
      placeholder="Search"
      bind:value
      on:keydown={(e) => {
        if (e.key === "Enter") {
          search();
        }
      }}
    />
    <button class="search-button" on:click={search}>Search</button>
  </div>

  <div
    class="max-h-24 overflow-y-auto -mb-2 mt-2"
    bind:this={preferenceContainer}
  >
    {#each preferenceValues as preference}
      <button
        class="w-64 max-sm:w-40 truncate monospace rounded px-2 inline-block mr-2 mb-2 cursor"
        class:bg-blue-200={preference.weight > 0}
        class:bg-orange-200={preference.weight < 0}
        title={`${preference.file.basename}: ${preference.searchResult.text}`}
        on:click={() => dispatch("setPreference", { ...preference, weight: 0 })}
      >
        {#if preference.weight > 0}
          <span class="text-blue-600 font-bold mr-1">+</span>
        {:else if preference.weight < 0}
          <span class="text-orange-500 font-bold mr-1">-</span>
        {/if}
        {preference.searchResult.text}
      </button>
    {/each}
  </div>
</div>

<style>
  .search-button {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='25' height='24' fill='none'%3E%3Cpath stroke='%23202020' stroke-width='3' d='M10.045 13.424A7.152 7.152 0 1 0 21.003 4.23a7.152 7.152 0 0 0-10.958 9.194Zm0 0-8.984 8.984'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    text-indent: -9999px;
    width: 25px;
    position: absolute;
    left: 10px;
  }
</style>
