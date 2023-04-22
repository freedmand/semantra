<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import {
    preferenceKey,
    type File,
    type SearchResult,
    type Preference,
  } from "../types";
  import SearchResultText from "./SearchResultText.svelte";

  export let file: File;
  export let searchResult: SearchResult;
  export let preferences: { [key: string]: Preference };
  export let showFilename = false;

  const dispatch = createEventDispatcher();

  $: prefKey = preferenceKey(file, searchResult);
  $: preference = preferences[prefKey] || {
    file,
    searchResult,
    weight: 0,
  };

  function jumpToResult(file: File, searchResult: SearchResult) {
    dispatch("navigate", { file, searchResult });
  }

  function setPreference(
    e: Event,
    file: File,
    searchResult: SearchResult,
    weight: number
  ) {
    dispatch("setPreference", {
      file,
      searchResult,
      weight,
    });
    e.stopPropagation();
  }
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<li
  on:click={() => jumpToResult(file, searchResult)}
  class="font-mono text-sm border-b last:border-0 py-4 px-4 border-black pointer"
>
  <div
    class="border-l-4 border-transparent pl-2 -ml-2"
    class:!border-blue-600={preference.weight > 0}
    class:!border-orange-500={preference.weight < 0}
  >
    {#if showFilename}
      <div class="font-bold text-base my-2">
        {file.basename}
        <span class="highlight px-1 rounded text-xs"
          >{searchResult.distance.toFixed(2)}</span
        >
      </div>
    {:else}
      <span class="text-xs highlight px-1 rounded"
        >{searchResult.distance.toFixed(2)}</span
      >
    {/if}
    <button
      class="bg-gray-300 rounded px-2"
      class:bg-blue-200={preference.weight > 0}
      class:text-blue-600={preference.weight > 0}
      class:font-bold={preference.weight > 0}
      on:click={(e) =>
        setPreference(e, file, searchResult, preference.weight > 0 ? 0 : 1)}
      >+</button
    >
    <button
      class="bg-gray-300 rounded px-2"
      class:bg-orange-200={preference.weight < 0}
      class:text-orange-500={preference.weight < 0}
      class:font-bold={preference.weight < 0}
      on:click={(e) =>
        setPreference(e, file, searchResult, preference.weight < 0 ? 0 : -1)}
      >-</button
    >
    {#key JSON.stringify( { filename: searchResult.filename, offset: searchResult.offset, queries: searchResult.queries, preferences: searchResult.preferences, text: searchResult.text } )}
      <SearchResultText {searchResult} text={searchResult.text} />
    {/key}
  </div>
</li>

<style>
  .highlight {
    background: rgb(255 222 0 / 39%);
  }
</style>
