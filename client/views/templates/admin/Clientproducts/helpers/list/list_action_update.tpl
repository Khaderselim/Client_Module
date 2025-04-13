<!---
This template is used to display an update button for a product in the admin panel.
-->
<a href="{$href|escape:'html':'UTF-8'}&updateproduct&id_product={$id}"
   title="{$action|escape:'html':'UTF-8'}"
   class="btn btn-default update-btn"
   onclick="return confirmUpdate({$id});"
   data-id="{$id}">
    <i class="icon-refresh"></i> {$action|escape:'html':'UTF-8'}
</a>

<script>

    function confirmUpdate(id) {
        event.preventDefault();

        if (!confirm('{l s="Are you sure you want to update this product?" js=1}')) {
            return false;
        }
        // AJAX request to update the product ( updateproduct() function in the controller )
        $.ajax({
            url: '{$href|escape:'javascript':'UTF-8'}&action=updateproduct',
            type: 'POST',
            data: {
                id_product: id,
            },
            success: function (response) {
                try {
                    var result = JSON.parse(response);
                    if (result.success) {
                        showSuccessMessage(result.message);
                        window.location.reload(); // Reload the page to see the updated product
                    } else {
                        showErrorMessage(result.errors ? result.errors.join('\n') : '{l s="Error updating product." js=1}');
                    }
                } catch (e) {
                    showErrorMessage('{l s="Error parsing response." js=1}');
                }
            },
            error: function () {
                showErrorMessage('{l s="An error occurred while updating the product." js=1}');
            }
        });

        return false;
    }
</script>