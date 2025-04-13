<?php
/**
 * This file is part of the Client module for PrestaShop.
 * It manages the client catalog functionality.
 */
class ClientCatalog extends ObjectModel
{
    /** @var int $id_client_catalog The unique identifier for the client catalog
     * @var string $name The name of the client catalog
     * @var bool $active Indicates if the client catalog is active
     */
    public $id_client_catalog;
    public $name;
    public $active;

    /**
     * Definition of the database table and fields for the Client's categories model.
     */
    public static $definition = array(
        'table' => 'client_catalog',
        'primary' => 'id_client_catalog',
        'fields' => array(
            'name' => array('type' => self::TYPE_STRING, 'validate' => 'isGenericName', 'required' => true),
            'active' => array('type' => self::TYPE_BOOL, 'validate' => 'isBool')
        )
    );

    /**
     * Retrieves all client catalogs.
     * @param bool $activeOnly If true, only active catalogs are retrieved.
     * @return array An array of client catalogs.
     */
    public static function getAllCatalogs($activeOnly = true)
    {
        $sql = 'SELECT id_client_catalog, name
                FROM `' . _DB_PREFIX_ . 'client_catalog`';

        if ($activeOnly) {
            $sql .= ' WHERE active = 1'; // Only active catalogs
        }

        $sql .= ' ORDER BY name ASC'; // Sort by name in ascending order

        $result = Db::getInstance()->executeS($sql);

        $catalogs = array();
        return array_merge($catalogs, $result ?: array()); // Return an empty array if no results found
    }
}
