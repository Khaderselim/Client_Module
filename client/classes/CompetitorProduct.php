<?php
/**
 * This file is part of the Client module for PrestaShop.
 * It manages the competitor product functionality.
 */
class CompetitorProduct extends ObjectModel
{
    /** @var int $id_product The unique identifier for the competitor product
     * @var int $id_competitor The ID of the competitor associated with this product
     * @var string $name The name of the competitor product
     * @var string $url The URL of the competitor product
     * @var string $price The price of the competitor product
     * @var string $description The description of the competitor product
     * @var string $stock The stock status of the competitor product
     * @var string $date_add The date the competitor product was added
     */
    public $id_product;
    public $id_competitor;
    public $name;
    public $url;
    public $price;
    public $description;
    public $stock;
    public $date_add;

    /**
     * Definition of the database table and fields for the products of the competitors model.
     */
    public static $definition = array(
        'table' => 'competitor_product',
        'primary' => 'id_product',
        'fields' => array(
            'id_competitor' => array('type' => self::TYPE_INT, 'validate' => 'isInt', 'required' => true),
            'name' => array('type' => self::TYPE_STRING, 'validate' => 'isGenericName', 'required' => true),
            'url' => array('type' => self::TYPE_STRING, 'validate' => 'isUrl', 'required' => true),
            'price' => array('type' => self::TYPE_STRING, 'required' => true),
            'description' => array('type' => self::TYPE_HTML, 'validate' => 'isCleanHtml'),
            'stock' => array('type' => self::TYPE_STRING, 'validate' => 'isString'),
            'date_add' => array('type' => self::TYPE_DATE, 'validate' => 'isDateFormat')
        )
    );
}
