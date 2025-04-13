<?php
/**
 * This file is part of the Client module for PrestaShop.
 * It manages the relationship between client and competitor products.
 */
class Products_relation extends ObjectModel {
    /** @var int $id The unique identifier for the product relation
     * @var int $id_product_competitor The ID of the competitor product
     * @var int $id_product_client The ID of the client product
     */
    public $id;
    public $id_product_competitor;
    public $id_product_client;

    /**
     * Definition of the database table and fields for the product relations model.
     */
    public static $definition = array(
        'table' => 'products_relation',
        'primary' => 'id',
        'fields' => array(
            'id_product_competitor' => array('type' => self::TYPE_INT, 'validate' => 'isInt', 'required' => true),
            'id_product_client' => array('type' => self::TYPE_INT, 'validate' => 'isInt', 'required' => true),
        )
    );

    /**
     * Retrieves product relations for a given client product.
     * @param int $id_product_client The ID of the client product.
     * @return array An array of product relations.
     */
    public static function getProductRelations($id_product_client) {
        $sql = 'SELECT r.*, cp.name, cp.url, cp.price, co.priority, co.logo, co.active, cp.id_product, cph.new_price, cph.old_price, cp.stock, c.price as client_price
        FROM '._DB_PREFIX_.'products_relation r
        LEFT JOIN '._DB_PREFIX_.'competitor_product cp ON r.id_product_competitor = cp.id_product
        LEFT JOIN '._DB_PREFIX_.'client_product c ON r.id_product_client = c.id_product
        LEFT JOIN '._DB_PREFIX_.'client_competitor co ON cp.id_competitor = co.id_client_competitor
        LEFT JOIN (
            SELECT ph.*
            FROM '._DB_PREFIX_.'competitor_products_history ph
            INNER JOIN (
                SELECT id_product, MAX(date_update) as max_date
                FROM '._DB_PREFIX_.'competitor_products_history
                GROUP BY id_product
            ) latest ON ph.id_product = latest.id_product
            AND ph.date_update = latest.max_date
        ) cph ON cp.id_product = cph.id_product
        WHERE r.id_product_client = '.(int)$id_product_client.'
        ORDER BY co.priority DESC';
        return Db::getInstance()->executeS($sql);
    }
}
