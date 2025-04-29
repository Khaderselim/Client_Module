<!---
This template is used to display a filter for the catalog in the admin panel.
It includes a dropdown menu that allows the user to select a specific catalog to filter by.
The selected catalog will be used to filter the products displayed in the admin panel..
-->
<div class="col-lg-6">
    <div class="form-group">
        <label class="control-label col-lg-3">{l s='Filter by Catalog' mod='client'}</label>
        <div class="col-lg-9">
            <select name="catalog_filter" class="filter fixed-width-xl form-control"
                    onchange="window.location.href = '{$admin_products_url}&catalog_filter=' + this.value">
                <option value="">{l s='All Catalogs' mod='client'}</option>
                {foreach from=$catalogs item=catalog}
                    <option value="{$catalog.id_client_catalog}"
                            {if $selected_catalog == $catalog.id_client_catalog}selected="selected"{/if}>
                        {$catalog.name|escape:'html':'UTF-8'}
                    </option>
                {/foreach}
            </select>
        </div>
    </div>
</div>