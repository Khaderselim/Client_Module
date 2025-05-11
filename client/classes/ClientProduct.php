<?php
/**
 * This file is part of the Client module for PrestaShop.
 * It manages the client product functionality.
 */
class ClientProduct extends ObjectModel {
    /** @var int $id_product The unique identifier for the product
     *  @var string $name The name of the product
     * @var string $img_url The image URL of the product
     * @var string $price The price of the product
     * @var string $description The description of the product
     * @var string $id_client_catalog The ID of the client category associated with this product
     * @var string $date_add The date the product was added
     * */
    public $id_client_product;
    public $id_product;
    public $name;
    public $img_url;
    public $price;
    public $description;
    public $id_client_catalog;
    public $date_add;
    /**
     * Definition of the database table and fields for the Client's products model.
     */
    public static $definition = array(
        'table' => 'client_product',
        'primary' => 'id_client_product',
        'fields' => array(
            'name' => array('type' => self::TYPE_STRING, 'validate' => 'isGenericName', 'required' => true),
            'id_product' => array('type' => self::TYPE_INT, 'validate' => 'isInt', 'required' => true),
            'img_url' => array('type' => self::TYPE_STRING, 'validate' => 'isUrl'),
            'price' => array('type' => self::TYPE_STRING,  'required' => true),
            'description' => array('type' => self::TYPE_HTML, 'validate' => 'isCleanHtml'),
            'id_client_catalog' => array('type' => self::TYPE_INT, 'validate' => 'isUnsignedId'),
            'date_add' => array('type' => self::TYPE_DATE, 'validate' => 'isDateFormat')
        )
    );
}