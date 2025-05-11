<?php
require_once _PS_MODULE_DIR_ . 'client/classes/ClientCatalog.php';

class AdminClientcatalogController extends ModuleAdminController
{
    /**
     * @throws PrestaShopException
     * This constructor initializes the AdminClientcatalogController class.
     * It sets up the necessary properties for the controller,
     * including the table name, class name, and fields for the list view.
     */
    public function __construct()
    {
        $this->bootstrap = true;
        $this->table = 'client_catalog';
        $this->className = 'ClientCatalog';
        $this->lang = false;
        $this->identifier = 'id_client_catalog';
        $this->orderBy = 'id_client_catalog';
        $this->orderWay = 'ASC';
        $this->list_no_link = true; // Disable link to edit form

        parent::__construct();

        $this->fields_list = array(
            'id_client_catalog' => array(
                'title' => $this->l('ID'),
                'align' => 'center',
                'width' => 25
            ),
            'name' => array(
                'title' => $this->l('Name'),
                'width' => 'auto'
            ),

            'active' => array(
                'title' => $this->l('Status'),
                'align' => 'center',
                'active' => 'status',
                'type' => 'bool'
            )
        );
        // Add the CRUD actions to the list
        $this->addRowAction('edit');
        $this->addRowAction('delete');

    }
    /**
     * @return void
     * This method is used to remove the default toolbar buttons.
     */
    public function initToolbar()
    {
    }

    /**
     * @return void
     * This method is used to initialize the page header toolbar.
     * It adds a button to create a new competitor.
     */
    public function initPageHeaderToolbar()
    {
        parent::initPageHeaderToolbar();
        if (Tools::getValue('addclient_catalog') === false && Tools::getValue('updateclient_catalog') === false) {
            $this->page_header_toolbar_btn['new'] = [
                'href' => self::$currentIndex . '&add' . $this->table . '&token=' . $this->token,
                'desc' => $this->trans('Add new', [], 'Admin.Actions'),
                'icon' => 'process-icon-new'
            ];
        }
    }

    /**
     * @return string
     * @throws SmartyException
     * This method is used to render the form for creating or editing a client catalog.
     */
    public function renderForm()
    {
        $this->fields_form = array(
            'legend' => array(
                'title' => $this->l('Category'),
                'icon' => 'icon-folder'
            ),
            'input' => array(
                array(
                    'type' => 'text',
                    'label' => $this->l('Name'),
                    'name' => 'name',
                    'required' => true
                ),

                array(
                    'type' => 'switch',
                    'label' => $this->l('Status'),
                    'name' => 'active',
                    'default_value' => 1,
                    'values' => array(
                        array('id' => 'active_on', 'value' => 1),
                        array('id' => 'active_off', 'value' => 0)
                    )
                )
            ),
            'submit' => array(
                'title' => $this->l('Save')
            )
        );

        return parent::renderForm();
    }
}