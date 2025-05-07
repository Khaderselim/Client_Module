{extends file="helpers/list/list_content.tpl"}

{block name="td_content"}
    {if $key == 'image'}
        {if $list_id == 'suggestion_product'}
            {assign var=product value=SuggestionProducts::getImage($tr.id_product)}
        {else}
            {assign var=product value=ComparingProducts::getImage($tr.id_product)}
        {/if}
        {if $product}
            <div>
                <img src="{$product}" alt="{$tr.id_product}" class="img-responsive" style="max-width: 50px;" />
            </div>
        {/if}
    {elseif $key == 'name'}
        {if $list_id == 'suggestion_product'}
            {assign var=product value=SuggestionProducts::getName($tr.id_product)}
        {else}
            {assign var=product value=ComparingProducts::getName($tr.id_product)}
        {/if}
        {if $product}
            <div>
                {$product|escape:'html':'UTF-8'}
            </div>
        {/if}
    {elseif $key == 'price'}
        {assign var=currency value=Currency::getDefaultCurrency()}
        {if $list_id == 'suggestion_product'}
            {assign var=price value=SuggestionProducts::getPrice($tr.id_product)}
        {else}
            {assign var=price value=ComparingProducts::getPrice($tr.id_product)}
        {/if}
        <div>
            {number_format($price, 3, ',', ' ')|escape:'html':'UTF-8'} {$currency->iso_code}
        </div>
    {elseif $key == 'competitor_product'}
        <div class="competitors-inline-list">
            <table class="table competitors-table-{$tr.id_product}">
                <thead>
                <tr>
                    <th>Logo</th>
                    <th>Competitor</th>
                    <th class="fixed-width-sm">Brands</th>
                    <th class="fixed-width-md">Price</th>
                    <th>Similarity</th>
                </tr>
                </thead>
                <tbody>
                {assign var=competitors value=ComparingProducts::getCompetitorProduct($tr.id_product)}

                {foreach from=$competitors item=competitor}
                    <tr>
                        <td>
                            {if $competitor.logo}
                                <img src="{$competitor.logo}" alt="{$competitor.name}" class="img-responsive" style="max-width: 50px;" />
                            {/if}
                        </td>
                        <td>
                            <a href="{$competitor.url|escape:'html':'UTF-8'}" target="_blank">{$competitor.name|escape:'html':'UTF-8'}</a>
                        </td>
                        <td>
                            {if $competitor.competitor_product_brands}
                                {$competitor.competitor_product_brands|escape:'html':'UTF-8'}
                            {/if}
                        </td>
                        <td>
                            <div>{$competitor.price|escape:'html':'UTF-8'}</div>
                        </td>
                        <td>
                            {if $competitor.similarity}
                                {number_format($competitor.similarity*100, 2)}%
                            {/if}
                        </td>
                    </tr>
                {/foreach}
                </tbody>
            </table>
        </div>
    {elseif $key == 'suggestion_competitor_product'}
        <div class="competitors-inline-list">
            <table class="table competitors-table-{$tr.id_product}">
                <thead>
                <tr>
                    <th>Logo</th>
                    <th>Competitor</th>
                    <th class="fixed-width-sm">Brands</th>
                    <th class="fixed-width-md">Price</th>
                    <th>Similarity</th>
                    <th>Action</th>
                </tr>
                </thead>
                <tbody>
                {assign var=competitors value=SuggestionProducts::getCompetitorProduct($tr.id_product)}
                {foreach from=$competitors item=competitor}
                    <tr>
                        <td>
                            {if $competitor.logo}
                                <img src="{$competitor.logo}" alt="{$competitor.name}" class="img-responsive" style="max-width: 50px;" />
                            {/if}
                        </td>
                        <td>
                            <a href="{$competitor.url|escape:'html':'UTF-8'}" target="_blank">{$competitor.name|escape:'html':'UTF-8'}</a>
                        </td>
                        <td>
                            {if $competitor.competitor_product_brands}
                                {$competitor.competitor_product_brands|escape:'html':'UTF-8'}
                            {/if}
                        </td>
                        <td>
                            <div>{$competitor.price|escape:'html':'UTF-8'}</div>
                        </td>
                        <td>
                            {if $competitor.similarity}
                                {number_format($competitor.similarity*100, 2)}%
                            {/if}
                        </td>
                        <td>
                            <button type="button" class="btn btn-default add-product"
                                    data-product-id="{$tr.id_product}"
                                    data-competitor-id="{$competitor.id_competitor_product}">
                                <i class="icon-plus"></i> {l s='Add'}
                            </button>
                        </td>
                    </tr>
                {/foreach}
                </tbody>
            </table>
        </div>
        <script type="text/javascript">
            $(document).ready(function () {
                var selectedProductId = null;
                var selectedCompetitorId = null;

                // Replace the existing click handler
                $(document).off('click', '.add-product').on('click', '.add-product', function (e) {
                    e.preventDefault();
                    selectedProductId = $(this).data('product-id');
                    selectedCompetitorId = $(this).data('competitor-id');
                    $('#ClientCatalogModal').modal('show');
                });

                // Add confirm handler
                $('#confirmAdd').off('click').on('click', function () {
                    var idClientCatalog = $('#client_catalog').val();
                    var button = $(this);

                    $.ajax({
                        url: '{$link->getAdminLink('AdminCompareproducts')}&ajax=1&action=addCompare',
                        type: 'POST',
                        dataType: 'json',
                        data: {
                            id_product: selectedProductId,
                            id_competitor_product: selectedCompetitorId,
                            id_client_catalog: idClientCatalog
                        },
                        beforeSend: function() {
                            button.prop('disabled', true);
                        },
                        success: function(response) {
                            if (response.success) {
                                showSuccessMessage(response.message);
                                $('#ClientCatalogModal').modal('hide');
                                location.reload();
                            } else {
                                showErrorMessage(response.message || 'Error occurred');
                            }
                        },
                        error: function(xhr) {
                            showErrorMessage('Error: ' + xhr.statusText);
                        },
                        complete: function() {
                            button.prop('disabled', false);
                        }
                    });
                });
            });
        </script>
    {else}
        {$smarty.block.parent}
    {/if}
    <!-- Modal -->
    <div id="ClientCatalogModal" class="bootstrap modal fade" role="dialog">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <button type="button" class="close" data-dismiss="modal">&times;</button>
                    <h3 class="modal-title">{l s='Select Catalog'}</h3>
                </div>
                <div class="modal-body">
                    <div class="form-group">
                        <label class="control-label">{l s='Catalog'}</label>
                        <select name="id_client_catalog" id="client_catalog" class="form-control">
                            {foreach from=$client_catalogs item=website}
                                <option value="{$website.id_client_catalog}">{$website.name}</option>
                            {/foreach}
                        </select>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-default" data-dismiss="modal">{l s='Cancel'}</button>
                    <button type="button" class="btn btn-primary" id="confirmAdd">{l s='Add'}</button>
                </div>
            </div>
        </div>
    </div>

{/block}