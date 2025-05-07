<?php

class ComparingProducts extends ObjectModel{

    public $id_comparing_product;
    public $id_product;
    public $id_competitor_product;
    public $product_brands;
    public $competitor_product_brands;
    public $similarity;
    public static $definition = array(
        'table' => 'client_comparing_product',
        'primary' => 'id_comparing_product',
        'fields' => array(
            'id_product' => array('type' => self::TYPE_INT, 'validate' => 'isInt', 'required' => true),
            'id_competitor_product' => array('type' => self::TYPE_INT, 'validate' => 'isInt', 'required' => true),
            'product_brands' => array('type' => self::TYPE_STRING, 'validate' => 'isString'),
            'competitor_product_brands' => array('type' => self::TYPE_STRING, 'validate' => 'isString'),
            'similarity' => array('type' => self::TYPE_FLOAT, 'validate' => 'isFloat', 'required' => true),
        )
    );

    public static function getName($id_product)
    {
        $sql = 'SELECT name FROM ' . _DB_PREFIX_ . 'product_lang WHERE id_product = ' . (int)$id_product;
        return Db::getInstance()->getValue($sql);
    }
    public static function getPrice($id_product)
    {
        $sql = 'SELECT price FROM ' . _DB_PREFIX_ . 'product WHERE id_product = ' . (int)$id_product;
        return Db::getInstance()->getValue($sql);
    }
    public static function getCompetitorProduct($id_product)
    {
        $sql = 'SELECT r.* , cp.name , cp.url, cp.price, co.logo, cp.id_product FROM '._DB_PREFIX_.'client_comparing_product r
                LEFT JOIN '._DB_PREFIX_. 'competitor_product cp ON r.id_competitor_product = cp.id_product
                LEFT JOIN '._DB_PREFIX_.'client_competitor co ON cp.id_competitor = co.id_client_competitor
                WHERE r.id_product='.(int)$id_product;
        return Db::getInstance()->executeS($sql);
    }
    public static function getImage($id_product)
    {
        $cover = Image::getCover($id_product);
        if ($cover) {
            $name = self::getName($id_product);
            return Context::getContext()->link->getImageLink($name, $cover['id_image']);
        }
        return false;
    }

}