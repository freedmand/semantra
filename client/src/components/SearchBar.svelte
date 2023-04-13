<script lang="ts">
  import { createEventDispatcher } from "svelte";
  const dispatch = createEventDispatcher();

  let value = "";

  function search() {
    dispatch("search", value);
  }
</script>

<div class="flex items-center relative flex-1">
  <input
    class="bg-white py-2 px-4 pl-12 font-mono w-full rounded border-black border"
    placeholder="enter a search query"
    bind:value
    on:keydown={(e) => {
      if (e.key === "Enter") {
        search();
      }
    }}
  />
  <button on:click={search}>Search</button>
</div>

<style>
  button {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='25' height='24' fill='none'%3E%3Cpath stroke='%23202020' stroke-width='3' d='M10.045 13.424A7.152 7.152 0 1 0 21.003 4.23a7.152 7.152 0 0 0-10.958 9.194Zm0 0-8.984 8.984'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    text-indent: -9999px;
    width: 25px;
    position: absolute;
    left: 10px;
  }
</style>
