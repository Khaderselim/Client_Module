$(document).ready(function() {
    $('#update_test_url_button').off('click').on('click', function(e) {
        e.preventDefault();
        e.stopPropagation();

        $.ajax({
            url: baseAdminDir + 'index.php',
            data: {
                controller: 'AdminClientProducts',
                token: token,
                ajax: 1,
                action: 'update_test_url'
            },
            type: 'POST',
            dataType: 'json',
            success: function(response) {
                if (response.success && response.modal) {
                    if ($('#test_url_modal').length) {
                        $('#test_url_modal').remove();
                    }
                    $('body').append(response.modal);
                    $('#test_url_modal').modal('show');
                }
            }
        });
        return false;
    });
});