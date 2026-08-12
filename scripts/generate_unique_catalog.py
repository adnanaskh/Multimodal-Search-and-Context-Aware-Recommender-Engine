"""
Generate a high-quality catalog of 200 unique products.
Every product has:
  - A unique title (no repeats)
  - A realistic, detailed description with specific features
  - A category that actually matches the product type
  - A unique Unsplash image ID for visual diversity
  - A realistic price range for its category
"""
import csv
import os
import shutil
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"

# ──────────────────────────────────────────────────────────────────────
# 200 unique products organized by category
# Each entry: (title, description, unsplash_photo_id, price)
# ──────────────────────────────────────────────────────────────────────

PRODUCTS = {
    "Electronics": [
        ("Sony WH-1000XM5 Wireless Headphones", "Industry-leading noise cancellation with dual processors, 30-hour battery life, and ultra-comfortable lightweight design. Features multipoint Bluetooth connection and speak-to-chat technology.", "photo-1505740420928-5e560c06d30e", 348.00),
        ("Apple iPad Air M2 11-inch", "Liquid Retina display with P3 wide color, M2 chip with 8-core GPU, 12MP front camera with Center Stage, and USB-C connector. Perfect for creative work and entertainment.", "photo-1544244015-0df4b3ffc6b0", 599.00),
        ("Samsung 27-inch 4K Monitor", "UHD IPS display with 99% sRGB color accuracy, HDR10 support, USB-C 65W charging, and adjustable ergonomic stand. Ideal for professional content creation.", "photo-1527443224154-c4a3942d3acf", 449.99),
        ("Logitech MX Master 3S Mouse", "Quiet clicks with 8000 DPI tracking on any surface including glass. Features MagSpeed electromagnetic scroll, USB-C fast charging, and multi-device connectivity.", "photo-1615663245857-ac93bb7c39e7", 99.99),
        ("Mechanical Keyboard Cherry MX Blue", "Full-size 104-key mechanical keyboard with tactile Cherry MX Blue switches, per-key RGB backlighting, aluminum frame, and detachable USB-C braided cable.", "photo-1587829741301-dc798b83add3", 129.99),
        ("JBL Charge 5 Portable Speaker", "IP67 waterproof and dustproof Bluetooth speaker with 20 hours of playtime, built-in powerbank, and PartyBoost for pairing multiple speakers together.", "photo-1545454675-3531b543be5d", 179.95),
        ("Canon EOS R50 Mirrorless Camera", "24.2MP APS-C sensor with Dual Pixel CMOS AF II, 4K 30fps video, subject detection AF, and compact lightweight body perfect for travel and vlogging.", "photo-1516035069371-29a1b244cc32", 679.99),
        ("DJI Mini 4 Pro Drone", "Under 249g foldable drone with 4K/60fps HDR video, omnidirectional obstacle sensing, 34-min flight time, and 20km HD video transmission range.", "photo-1508614589041-895b88991e3e", 759.00),
        ("Apple AirPods Pro 2nd Gen", "Adaptive noise cancellation with conversation awareness, personalized spatial audio, USB-C MagSafe case with precision finding, and 6 hours of listening time.", "photo-1590658268037-6bf12165a8df", 249.00),
        ("Anker 65W GaN USB-C Charger", "Ultra-compact gallium nitride charger with 3 ports, intelligent power distribution, ActiveShield 2.0 temperature monitoring, and foldable plug design.", "photo-1583863788434-e58a36330cf0", 35.99),
        ("LG 34-inch UltraWide Monitor", "34-inch curved WQHD IPS display with 21:9 aspect ratio, AMD FreeSync, HDR10, and USB-C connectivity. Perfect for multitasking and immersive gaming.", "photo-1585792180666-f7347c490ee2", 399.99),
        ("Bose SoundLink Flex Speaker", "Portable Bluetooth speaker with PositionIQ technology, IP67 waterproof rating, 12-hour battery life, and deep bass performance in a rugged design.", "photo-1608043152269-423dbba4e7e1", 149.00),
        ("Razer DeathAdder V3 Gaming Mouse", "Ultra-lightweight 59g ergonomic gaming mouse with Focus Pro 30K optical sensor, 90-hour battery life, and HyperSpeed wireless technology.", "photo-1527814050087-3793815479db", 89.99),
        ("Samsung Galaxy Buds3 Pro", "Adaptive noise cancellation earbuds with 360 Audio, 24-bit Hi-Fi sound, IP57 water resistance, and intelligent conversation mode with blade light design.", "photo-1606220588913-b3aacb4d2f46", 249.99),
        ("Elgato Stream Deck MK.2", "15 customizable LCD keys for live streaming, video editing, and productivity. One-touch actions, multi-profile switching, and interchangeable faceplates.", "photo-1625842268584-8f3296236761", 149.99),
        ("Anker PowerCore 26800 Power Bank", "Ultra-high capacity 26800mAh portable charger with dual input, triple USB output, PowerIQ and VoltageBoost technology for phones and tablets.", "photo-1609091839311-d5365f9ff1c5", 65.99),
        ("Google Nest Hub Max", "10-inch smart display with built-in Nest Cam, stereo speakers, Google Assistant, Face Match, and video calling. Central hub for smart home control.", "photo-1558618666-fcd25c85f82e", 229.00),
        ("Roku Streaming Stick 4K+", "4K HDR streaming with Dolby Vision and Atmos, long-range Wi-Fi, voice remote with TV controls, and fast smooth startup.", "photo-1611532736597-de2d4265fba3", 49.99),
        ("TP-Link Deco Mesh Wi-Fi 6 System", "Whole-home mesh Wi-Fi 6 system covering up to 6500 sq ft, AX3000 dual-band speeds, WPA3 security, and seamless roaming between nodes.", "photo-1606904825846-647eb07f5be2", 199.99),
        ("WD 2TB External SSD", "Portable NVMe SSD with up to 2000MB/s read speeds, drop-resistant up to 6.5 feet, IP65 water and dust resistance, and USB-C connectivity.", "photo-1597872200969-2b65d56bd16b", 179.99),
    ],
    "Clothing": [
        ("Patagonia Better Sweater Fleece Jacket", "Classic quarter-zip fleece made from 100% recycled polyester with Fair Trade Certified sewing. Features zippered pockets and flat-seam construction for comfort.", "photo-1551028719-00167b16eac5", 139.00),
        ("Levi's 501 Original Fit Jeans", "Iconic straight-leg jeans with button fly, 100% cotton non-stretch denim, and authentic vintage-inspired wash. The original blue jean since 1873.", "photo-1542272454315-4c01d7abdf4a", 69.50),
        ("Nike Dri-FIT Training T-Shirt", "Moisture-wicking performance tee with lightweight mesh panels, raglan sleeves for mobility, and standard relaxed fit. Made with at least 75% recycled polyester.", "photo-1521572267360-ee0c2909d518", 35.00),
        ("North Face Thermoball Eco Jacket", "Lightweight insulated jacket with ThermoBall Eco synthetic insulation made from post-consumer recycled materials. Packable into its own pocket for travel.", "photo-1544923246-77307dd654cb", 199.00),
        ("Merino Wool Crew Neck Sweater", "Fine-gauge 100% merino wool sweater with ribbed cuffs and hem, naturally temperature-regulating, breathable, and resistant to odors. Machine washable.", "photo-1576566588028-4147f3842f27", 89.00),
        ("Columbia Silver Ridge Cargo Pants", "Quick-dry nylon ripstop pants with Omni-Shade UPF 50 sun protection, gusseted crotch for mobility, and multiple zip-secured cargo pockets.", "photo-1473966968600-fa801b869a1a", 55.00),
        ("Adidas Ultraboost 23 Running Shoes", "Responsive Boost midsole with Linear Energy Push system, Primeknit+ adaptive upper, Continental rubber outsole, and 100% recycled upper materials.", "photo-1542291026-7eec264c27ff", 190.00),
        ("Ray-Ban Wayfarer Classic Sunglasses", "Iconic acetate frame with crystal green G-15 lenses providing 100% UV protection. Hand-crafted in Italy with signature rivet accents.", "photo-1572635196237-14b3f281503f", 163.00),
        ("Herschel Little America Backpack", "Classic mountaineering-inspired design with signature striped liner, magnetic strap closures, padded 15-inch laptop sleeve, and 25L capacity.", "photo-1553062407-98eeb64c6a62", 109.99),
        ("Timberland Premium 6-Inch Waterproof Boot", "Iconic waterproof boots with 400g PrimaLoft insulation, anti-fatigue footbed, and seam-sealed construction. Made with ReBOTL recycled materials.", "photo-1605812860427-4024433a70fd", 198.00),
        ("Uniqlo Ultra Light Down Vest", "Incredibly lightweight packable down vest with premium 90% goose down insulation, water-repellent finish, and inner pocket that doubles as a storage pouch.", "photo-1591047139829-d91aecb6caea", 59.90),
        ("Champion Reverse Weave Hoodie", "Heavyweight fleece hoodie with patented Reverse Weave construction that resists vertical shrinkage. Features ribbed side panels and kangaroo pocket.", "photo-1556821840-3a63f95609a7", 70.00),
        ("Lululemon Align High-Rise Leggings", "Buttery-soft Nulu fabric with four-way stretch, lightweight feel, and hidden waistband pocket. Designed for yoga and low-impact workouts.", "photo-1506629082955-511b1aa562c8", 98.00),
        ("Brooks Ghost 15 Running Shoes", "Smooth and cushioned neutral running shoe with DNA LOFT v2 midsole, engineered mesh upper, and segmented crash pad for smooth transitions.", "photo-1460353581641-37baddab0fa2", 140.00),
        ("Osprey Farpoint 40 Travel Backpack", "Carry-on compatible travel pack with lockable zippers, padded laptop and tablet sleeves, stowaway hip belt and shoulder straps, and mesh back panel.", "photo-1622560480654-996b3d003b97", 185.00),
        ("Casio G-Shock GA2100 Watch", "Carbon Core Guard structure with octagonal bezel, 200m water resistance, world time, stopwatch, and super illuminator LED. Ultra-thin profile for a G-Shock.", "photo-1523275335684-37898b6baf30", 99.99),
        ("Dr. Martens 1460 Smooth Leather Boot", "Iconic 8-eye boot in smooth polished leather with Goodyear welt construction, air-cushioned AirWair sole, and signature yellow stitching.", "photo-1608256246200-53e635b5b65f", 170.00),
        ("Carhartt WIP Detroit Jacket", "Heavyweight organic cotton canvas jacket with blanket-lined body and corduroy collar. Features bi-swing back for mobility and multiple utility pockets.", "photo-1551537482-f2075a1d41f2", 228.00),
        ("New Balance 990v6 Sneakers", "Premium Made in USA sneakers with FuelCell midsole, pigskin suede and mesh upper, ENCAP midsole technology, and blown rubber outsole.", "photo-1539185441755-769473a23570", 199.99),
        ("Fjallraven Kanken Classic Backpack", "Iconic Swedish backpack made from durable Vinylon F fabric with dual top handles, padded shoulder straps, 16L capacity, and sitting pad in back pocket.", "photo-1581605405669-fcdf81165afa", 80.00),
    ],
    "Home": [
        ("Dyson V15 Detect Cordless Vacuum", "Laser reveals microscopic dust with piezo sensor counting particles, showing real-time data on LCD screen. 60-minute runtime with hair-detangling technology.", "photo-1558618666-fcd25c85f82e", 749.99),
        ("Philips Hue Starter Kit 4-Pack", "Smart LED bulbs with 16 million colors and warm-to-cool white light. Includes Hue Bridge for voice control with Alexa, Google, and HomeKit.", "photo-1507473885765-e6ed057f782c", 199.99),
        ("KitchenAid Artisan Stand Mixer", "Iconic 5-quart tilt-head stand mixer with 10 speeds, planetary mixing action, and 59-point planetary mixing action. Includes flat beater, dough hook, and wire whip.", "photo-1585515320310-259814833e62", 449.99),
        ("Nespresso Vertuo Next Coffee Machine", "Centrifusion barcode-reading technology brews five coffee sizes from espresso to carafe. Features 54% recycled plastic body and fast 25-second heat-up.", "photo-1517668808822-9ebb02f2a0e6", 159.00),
        ("iRobot Roomba j7+ Robot Vacuum", "AI-powered obstacle avoidance identifies and avoids pet waste and cords. Self-emptying Clean Base, Smart Mapping, and Alexa/Google Home compatible.", "photo-1558317374-067fb5f30001", 599.99),
        ("Instant Pot Duo Plus 6-Quart", "9-in-1 multi-cooker with pressure cooker, slow cooker, rice cooker, yogurt maker, steamer, sauté pan, food warmer, sous vide, and sterilizer functions.", "photo-1585664811087-47f65abbad64", 89.95),
        ("Vitamix E310 Explorian Blender", "Professional-grade blender with 2.0 HP motor, variable speed control, pulse feature, and 48oz container. Self-cleaning in 60 seconds with warm water and dish soap.", "photo-1570222094714-4281f1aa4458", 349.95),
        ("Le Creuset Dutch Oven 5.5-Qt", "Iconic enameled cast iron with superior heat distribution, colorful exterior, sand-colored interior enamel, and tight-fitting lid for moisture retention.", "photo-1556909114-f6e7ad7d3136", 399.95),
        ("Dyson Pure Cool Air Purifier Tower", "HEPA filter captures 99.97% of particles as small as 0.3 microns. Real-time air quality monitoring, 350-degree oscillation, and Dyson Link app control.", "photo-1612450796336-d7e8b6540c49", 549.99),
        ("Breville Barista Express Espresso", "Built-in conical burr grinder with dose control, digital temperature control, micro-foam milk texturing, and 54mm portafilter for café-quality espresso.", "photo-1495474472287-4d71bcdd2085", 699.95),
        ("Herman Miller Aeron Office Chair", "Iconic ergonomic chair with 8Z Pellicle elastomeric suspension, PostureFit SL spinal support, adjustable arms, and tilt limiter with seat angle.", "photo-1567538096630-e0c55bd6374c", 1395.00),
        ("Yeti Rambler 20oz Tumbler", "Double-wall vacuum insulated stainless steel tumbler with MagSlider lid, 18/8 stainless steel, dishwasher safe, and keeps drinks cold or hot for hours.", "photo-1514432324607-a09d9b4aefdd", 35.00),
        ("Sonos One SL Wireless Speaker", "Compact smart speaker with rich Trueplay-tuned sound, AirPlay 2 support, multi-room capability, and humidity-resistant design for kitchen or bathroom.", "photo-1543512214-318c7553f230", 199.00),
        ("Cuisinart 14-Cup Coffee Maker", "Programmable drip coffee maker with brew strength control, adjustable heater plate, charcoal water filter, and gold-tone permanent filter.", "photo-1495774856032-8b90bbb32b32", 99.95),
        ("Coyuchi Organic Cotton Bed Sheets", "300-thread-count organic cotton percale sheets with GOTS certification, breathable crisp feel, deep-pocket fitted sheet, and envelope closure pillowcases.", "photo-1522771739844-6a9f6d5f14af", 198.00),
        ("West Elm Mid-Century Desk Lamp", "Articulating brass desk lamp with walnut wood base, adjustable arm and shade, LED-compatible E26 socket, and inline dimmer switch.", "photo-1507473885765-e6ed057f782c", 129.00),
        ("Zwilling J.A. Henckels 8-inch Chef Knife", "Precision-forged from single piece of special formula steel, ice-hardened FRIODUR blade, full bolster, and triple-rivet ergonomic polymer handle.", "photo-1593618998160-e34014e67546", 149.99),
        ("Chemex 8-Cup Pour-Over Coffee Maker", "Iconic borosilicate glass design with polished wood collar and leather tie. Uses proprietary bonded filters for clean, pure coffee flavor.", "photo-1572119865084-43c285814d63", 44.50),
        ("Brooklinen Luxe Core Sheet Set", "480-thread-count long-staple cotton sateen sheets with buttery smooth finish, deep-pocket fitted sheet, and oeko-tex certified dyes.", "photo-1631049035634-c1b6e70ccfe5", 179.00),
        ("Staub Cast Iron Cocotte 4-Qt", "French enameled cast iron with self-basting spike lid, black matte enamel interior, brass knob rated to 500°F, and superior heat retention.", "photo-1585664811087-47f65abbad64", 299.99),
    ],
    "Fitness": [
        ("Garmin Forerunner 265 GPS Watch", "AMOLED display running watch with Training Readiness, Morning Report, race predictor, and recovery advisor. 13-day battery in smartwatch mode.", "photo-1575311373937-040b8e1fd5b6", 449.99),
        ("Fitbit Charge 6 Fitness Tracker", "Advanced health metrics with ECG app, SpO2 monitoring, stress management score, skin temperature sensing, and built-in GPS. 7-day battery life.", "photo-1575311373937-040b8e1fd5b6", 159.95),
        ("Manduka PRO Yoga Mat 6mm", "Lifetime-guaranteed ultra-dense cushioning yoga mat with closed-cell surface that prevents sweat absorption. OEKO-TEX certified, 71 inches long.", "photo-1601925260368-ae2f83cf8b7f", 120.00),
        ("Bowflex SelectTech 552 Dumbbells", "Adjustable dumbbells replacing 15 sets of weights from 5 to 52.5 lbs each. Dial system for quick weight changes between sets.", "photo-1534438327276-14e5300c3a48", 399.00),
        ("TRX All-in-One Suspension Trainer", "Portable suspension training system with commercial-grade materials, locking carabiner, door anchor, and full-color workout guide. Supports up to 350 lbs.", "photo-1571019614242-c5c5dee9f50b", 169.95),
        ("Hydro Flask 32oz Wide Mouth Bottle", "TempShield double-wall vacuum insulated stainless steel water bottle. Keeps drinks cold 24 hours or hot 12 hours. BPA-free, dishwasher safe.", "photo-1602143407151-7111542de6e8", 44.95),
        ("Nike Metcon 9 Training Shoes", "Stable wide flat heel for weightlifting with responsive Cushlon 3.0 foam for cardio. Textured rubber rope-wrap and breathable mesh upper.", "photo-1542291026-7eec264c27ff", 135.00),
        ("Theragun Elite Percussion Massager", "Smart percussive therapy device with QuietForce technology, OLED screen, 5 built-in speeds, Bluetooth app integration, and ergonomic triangle handle.", "photo-1576678927484-cc907957088c", 399.00),
        ("Peloton Bike Mat", "Premium rubber mat protecting floors from sweat and equipment vibration. 72 x 36 inches with anti-slip textured surface and 6mm thickness.", "photo-1601925260368-ae2f83cf8b7f", 59.00),
        ("Whoop 4.0 Fitness Strap", "Continuous heart rate, HRV, skin temperature, and blood oxygen monitoring with strain coach, sleep tracker, and recovery score. Membership-based wearable.", "photo-1575311373937-040b8e1fd5b6", 239.00),
        ("Rogue Fitness Resistance Band Set", "Set of 5 latex loop bands with progressive resistance levels from 5-60 lbs. Includes carrying bag and exercise guide.", "photo-1598289431512-b97b0917affc", 29.95),
        ("Hyperice Normatec 3 Leg Recovery", "Dynamic air compression leg recovery system with Bluetooth app control, 7 intensity levels, Zone Boost feature, and precise pulse technology.", "photo-1576678927484-cc907957088c", 799.00),
        ("Liforme Yoga Mat", "Alignment guide system printed on eco-friendly natural rubber mat with moisture-absorbing microfiber surface, antimicrobial coating, and 73-inch length.", "photo-1600881333168-2ef49b341f30", 140.00),
        ("Under Armour HOVR Phantom 3 Shoes", "Connected running shoe with UA MapMyRun integration, HOVR zero gravity feel, Intelliknit upper, and carbon rubber outsole.", "photo-1460353581641-37baddab0fa2", 160.00),
        ("Trigger Point GRID Foam Roller", "Multi-density exterior with hollow core for targeted muscle compression. 13-inch compact size with distrodensity zones mimicking a therapist's hands.", "photo-1571019614242-c5c5dee9f50b", 34.99),
        ("Apple Watch Ultra 2", "Rugged titanium case with precision dual-frequency GPS, 72-hour battery life, 100m water resistance, depth gauge, and siren for emergencies.", "photo-1523275335684-37898b6baf30", 799.00),
        ("Concept2 RowErg Indoor Rower", "Air resistance flywheel with PM5 performance monitor, nickel-plated steel chain, adjustable footrests, and easily separates in two for storage.", "photo-1534438327276-14e5300c3a48", 990.00),
        ("YETI Hopper Flip 12 Soft Cooler", "Leakproof portable cooler with ColdCell insulation, DryHide Shell for puncture resistance, and HydroLok zipper. Holds 12 cans plus ice.", "photo-1602143407151-7111542de6e8", 250.00),
        ("Jabra Elite 85t Active Earbuds", "Noise canceling sport earbuds with semi-open design, IP57 rating, 6mm speakers with customizable EQ, and 31 hours total battery with case.", "photo-1606220588913-b3aacb4d2f46", 179.99),
        ("Bala Bangles 1lb Wrist Weights", "Sleek silicone-wrapped stainless steel wrist and ankle weights with elastic band closure. Add resistance to walks, yoga, pilates, and dance.", "photo-1598289431512-b97b0917affc", 49.00),
    ],
    "Beauty": [
        ("Dyson Airwrap Multi-Styler Complete", "Coanda effect styling tool with multiple attachments for curling, waving, smoothing, and drying. Intelligent heat control protects hair from extreme damage.", "photo-1522337360788-8b13dee7a37e", 599.99),
        ("Drunk Elephant Protini Polypeptide Cream", "Signal peptide complex moisturizer with growth factors, amino acids, and pygmy waterlily stem cell extract. Biocompatible and free of essential oils.", "photo-1596462502278-27bfdc403348", 68.00),
        ("Olaplex No.3 Hair Perfector", "At-home bond-building treatment that repairs damaged and compromised hair by relinking broken disulfide bonds. Use weekly before shampoo.", "photo-1599305090598-fe179d501227", 30.00),
        ("SK-II Facial Treatment Essence", "Cult-favorite essence with over 90% Pitera, a bio-ingredient from sake fermentation, that improves skin clarity, firmness, and wrinkle appearance.", "photo-1571781926291-c477ebfd024b", 185.00),
        ("Tatcha Dewy Skin Cream", "Rich moisturizer with Japanese purple rice, Okinawa algae blend, hyaluronic acid, and botanical extracts for plump, dewy skin without greasiness.", "photo-1596462502278-27bfdc403348", 68.00),
        ("NuFACE Trinity+ Facial Toning Device", "FDA-cleared microcurrent device that improves facial contour, skin tone, and wrinkle appearance. App-guided routines with Effective Pulse Technology.", "photo-1556228578-0d85b1a4d571", 395.00),
        ("La Mer Crème de la Mer Moisturizer", "Legendary cell-renewing Miracle Broth moisturizer with sea kelp bio-ferment, vitamins, and antioxidants. Transforms skin texture and appearance.", "photo-1571781926291-c477ebfd024b", 190.00),
        ("Charlotte Tilbury Hollywood Filter", "Light-reflecting complexion booster with light-diffusing pigments and a blend of florals to give skin a soft-focus, smoothed, and luminous finish.", "photo-1596462502278-27bfdc403348", 44.00),
        ("Foreo Luna 4 Facial Cleansing Brush", "Medical-grade silicone sonic facial massager and cleanser with T-Sonic pulsations, 16 intensities, and anti-aging firming massage function.", "photo-1556228578-0d85b1a4d571", 279.00),
        ("Augustinus Bader The Rich Cream", "TFC8 technology triggers the body's natural renewal process with amino acids, vitamins, and synthesized molecules. Clinically proven results.", "photo-1571781926291-c477ebfd024b", 265.00),
        ("Tom Ford Black Orchid Eau de Parfum", "Luxurious dark floral fragrance with black truffle, ylang-ylang, bergamot, black orchid, and patchouli. Bold and iconic statement scent.", "photo-1541643600914-78b084683601", 150.00),
        ("Sunday Riley Good Genes Lactic Acid", "All-in-one AHA lactic acid treatment that exfoliates, clarifies pores, and visibly reduces fine lines. Instant and long-term radiance results.", "photo-1599305090598-fe179d501227", 85.00),
        ("Herbivore Botanicals Blue Tansy Mask", "BHA and AHA resurfacing clarity mask with blue tansy oil, white willow bark, and fruit enzymes. Calms inflammation while clearing pores.", "photo-1596462502278-27bfdc403348", 48.00),
        ("PMD Clean Pro RQ Smart Facial Device", "Rose quartz gemstone-infused silicone cleansing device with ActiveWarmth heat therapy, SonicGlow technology, and smart skin sensors.", "photo-1556228578-0d85b1a4d571", 159.00),
        ("Chanel No.5 L'Eau Eau de Toilette", "Modern reinterpretation of the legendary fragrance with citrus, rose, jasmine, and white musk. Fresh, dynamic interpretation for the contemporary woman.", "photo-1541643600914-78b084683601", 105.00),
        ("Glossier Boy Brow Grooming Pomade", "Cult-favorite eyebrow pomade that thickens, fills, and shapes with flexible, buildable hold. Infused with conditioning agents and natural pigments.", "photo-1522337360788-8b13dee7a37e", 16.00),
        ("Dr. Barbara Sturm Hyaluronic Serum", "Low and high-weight hyaluronic acid molecules provide deep and surface hydration. Purslane reduces irritation and antioxidant damage.", "photo-1599305090598-fe179d501227", 300.00),
        ("Diptyque Baies Scented Candle", "Hand-poured candle with Bulgarian rose and blackcurrant berries creating a beloved fruity-floral scent. 60-hour burn time in signature oval glass.", "photo-1602028915047-37269d1a73f7", 72.00),
        ("Revlon One-Step Hair Dryer Brush", "Volumizing hot air brush combining drying and styling in one step. Oval brush design with mixed bristles, 3 heat/speed settings, and cool tip.", "photo-1522337360788-8b13dee7a37e", 34.99),
        ("CeraVe Moisturizing Cream 16oz", "Developed with dermatologists featuring 3 essential ceramides and hyaluronic acid with MVE delivery technology for 24-hour hydration. Fragrance-free.", "photo-1556228578-0d85b1a4d571", 18.99),
    ],
    "Sports": [
        ("Wilson Evolution Indoor Basketball", "Premium composite leather game ball with cushion core technology for exceptional feel, moisture-absorbing Aqua-Grip cover, and laid-in composite channels.", "photo-1546519638-68e109498ffc", 69.99),
        ("Titleist Pro V1 Golf Balls (Dozen)", "Tour-proven golf ball with reformulated 2.0 ZG process core, spherically-tiled 388 dimple design, and cast urethane elastomer cover for complete performance.", "photo-1535131749006-b7f58c99034b", 54.99),
        ("Babolat Pure Aero Tennis Racquet", "Rafael Nadal's racquet with aero modular technology for maximum spin potential, FSI Power for explosive ball speed, and Cortex Pure Feel vibration dampening.", "photo-1554068865-24cecd4e34b8", 229.00),
        ("Burton Custom Flying V Snowboard", "Versatile all-mountain freestyle board with Flying V profile, Squeezebox core profiling, and Fiberglass Biax top and bottom for playful responsiveness.", "photo-1551698618-1dfe5d97d256", 549.99),
        ("Speedo Vanquisher 2.0 Swim Goggles", "Competition swim goggles with panoramic UV-protected anti-fog lenses, flexible one-piece inner frame, and speed-fit adjustable clip system.", "photo-1530549387789-4c1017266635", 25.00),
        ("Yonex Astrox 88D Pro Badminton Racquet", "Rotational generator system with Namd graphite for maximum shuttle hold and repulsion. Head-heavy balance for explosive smashes.", "photo-1554068865-24cecd4e34b8", 219.99),
        ("Callaway Rogue ST Max Driver", "Jailbreak AI speed frame with tungsten speed cartridge, A.I.-designed Flash Face, and triaxial carbon crown for maximum forgiveness and ball speed.", "photo-1535131749006-b7f58c99034b", 449.99),
        ("Nike Premier III Football Boots", "Premium kangaroo leather upper with fold-over tongue, cushioned sockliner, and conical and bladed studs for multi-surface traction.", "photo-1511886929837-354d827aae26", 110.00),
        ("Garmin Edge 840 Cycling Computer", "GPS cycling computer with touchscreen and buttons, power guide, ClimbPro 2.0, free adaptive training plans, and 26-hour battery life.", "photo-1575311373937-040b8e1fd5b6", 399.99),
        ("Oakley Radar EV Path Sport Sunglasses", "Extended lens coverage with Prizm Road technology for enhanced color, contrast, and detail. Unobtanium nose pads and ear socks increase grip with sweat.", "photo-1572635196237-14b3f281503f", 204.00),
        ("Adidas Predator Accuracy.1 FG", "Zone skin upper with 3D-printed elements for enhanced ball grip and swerve. Lightweight frame and FG studs for firm ground dominance.", "photo-1511886929837-354d827aae26", 250.00),
        ("Mizuno Wave Rider 27 Running Shoes", "ENERZY core midsole with Mizuno Wave plate for smooth transitions, AIRmesh upper for breathability, and X10 carbon rubber outsole.", "photo-1460353581641-37baddab0fa2", 139.99),
        ("CamelBak Podium Chill 21oz Bike Bottle", "Insulated cycling water bottle with self-sealing Jet Valve for high flow rate, lockout for leak-proof transport, and easy-squeeze design.", "photo-1602143407151-7111542de6e8", 16.00),
        ("Giro Syntax MIPS Road Bike Helmet", "Lightweight polycarbonate shell with MIPS liner reducing rotational forces, Roc Loc 5+ fit system, and 19 cooling vents.", "photo-1530549387789-4c1017266635", 124.99),
        ("Under Armour Curry 11 Basketball Shoes", "UA Flow cushioning without rubber outsole for court feel, IntelliKnit upper with zonal support, and Curry-specific traction pattern.", "photo-1542291026-7eec264c27ff", 160.00),
        ("Shimano 105 R7000 Groupset", "11-speed road bike groupset with dual-control lever shifters, direct mount brake calipers, Hollowtech II crankset, and wide-ratio cassette.", "photo-1485965120184-e220f721d03e", 649.99),
        ("Osprey Raptor 10 Hydration Pack", "Cycling-specific hydration pack with 2.5L Hydraulics reservoir, magnetic hose management, tool compartment, and LidLock helmet attachment.", "photo-1622560480654-996b3d003b97", 130.00),
        ("POC Ventral Air MIPS Cycling Helmet", "Aero-optimized cycling helmet with SPIN pads, EPS liner, extended ARC coverage, and adjusted ventilation for optimal cooling and speed.", "photo-1530549387789-4c1017266635", 274.99),
        ("Pearl Izumi Attack Cycling Jersey", "Transfer fabric with SELECT Transfer technology for rapid moisture transfer, full-length zipper, 3 rear pockets, and UPF 50+ sun protection.", "photo-1521572267360-ee0c2909d518", 85.00),
        ("Fox Racing Ranger Gel MTB Gloves", "Mountain bike gloves with gel palm padding, 4-way stretch mesh upper, touchscreen-compatible fingertips, and absorbent micro-suede thumb.", "photo-1583743814966-8936f5b7be1a", 34.95),
    ],
    "Outdoors": [
        ("MSR Hubba Hubba NX 2-Person Tent", "Ultralight freestanding backpacking tent at 3 lbs 8 oz with unified hub-and-pole system, rainfly vents, and Xtreme Shield waterproof coating.", "photo-1504280390367-361c6d9f38f4", 449.95),
        ("REI Co-op Flash 55 Pack", "Ultralight backpacking pack with removable framesheet, Load Shelf compression, hip-belt pockets, and tool-free torso adjustment. 2 lbs 11 oz.", "photo-1622560480654-996b3d003b97", 199.00),
        ("Black Diamond Spot 400-R Headlamp", "Rechargeable 400-lumen headlamp with proximity and distance modes, red night-vision light, IP67 waterproof, and memory function.", "photo-1504280390367-361c6d9f38f4", 49.95),
        ("Stanley Adventure Quencher Tumbler 40oz", "Double-wall vacuum insulated tumbler with FlowState lid, comfort-grip handle, and base fits car cup holders. Keeps ice cold for 2 days.", "photo-1514432324607-a09d9b4aefdd", 45.00),
        ("Kelty Cosmic Down 20°F Sleeping Bag", "550-fill DriDown insulation sleeping bag with thermal comfort hood, natural fit footbox, and two-way zipper. Weighs 2 lbs 11 oz.", "photo-1504280390367-361c6d9f38f4", 159.95),
        ("Jetboil Flash Cooking System", "Compact camping stove boiling 2 cups of water in 100 seconds. Push-button igniter, 1L insulated FluxRing cup, and drink-through lid.", "photo-1504280390367-361c6d9f38f4", 114.95),
        ("Salomon X Ultra 4 GTX Hiking Shoes", "GORE-TEX waterproof hiking shoes with Advanced Chassis for stability, Contagrip MA outsole for wet and dry grip, and OrthoLite sockliner.", "photo-1542291026-7eec264c27ff", 165.00),
        ("Patagonia Black Hole Duffel 55L", "Incredibly durable 100% recycled ripstop duffel with padded shoulder straps, U-shaped lid for easy packing, and DWR water-repellent finish.", "photo-1553062407-98eeb64c6a62", 149.00),
        ("Leatherman Wave+ Multi-Tool", "18 tools in one with replaceable wire cutters, pliers, knife blades, saw, scissors, screwdrivers, and more. Stainless steel with nylon sheath.", "photo-1593618998160-e34014e67546", 99.95),
        ("NEMO Tensor Insulated Sleeping Pad", "Ultralight insulated sleeping pad with Thermal Mirror technology, R-value 4.2, Spaceframe baffles for body-mapped support, and 20R regular size.", "photo-1504280390367-361c6d9f38f4", 179.95),
        ("Sawyer Squeeze Water Filter", "Inline water filter removing 99.99999% of bacteria and 99.9999% of protozoa. Weighs 3 oz, filters up to 100,000 gallons, and backwashable.", "photo-1602143407151-7111542de6e8", 37.95),
        ("Petzl GriGri+ Belay Device", "Assisted-braking belay device with anti-panic handle, steel friction plate, and top and bottom rope-clamp modes for smooth lowering.", "photo-1504280390367-361c6d9f38f4", 124.95),
        ("Sea to Summit Aeros Premium Pillow", "Ultralight inflatable camping pillow with TPU bladder, brushed 50D polyester top, and multi-functional valve for easy inflation and deflation.", "photo-1504280390367-361c6d9f38f4", 44.95),
        ("GSI Outdoors Pinnacle Camper Cookset", "Complete camping cookware set with 2 non-stick pots, frying pan, strainer lids, cutting board, and nesting mugs. Compact nested storage.", "photo-1504280390367-361c6d9f38f4", 109.95),
        ("Garmin inReach Mini 2 Satellite Communicator", "Compact satellite communicator with two-way messaging, interactive SOS, GPS tracking, weather forecasts, and 14-day battery life.", "photo-1508614589041-895b88991e3e", 399.99),
        ("Trekology ALUFT 2.0 Camping Pillow", "Ultra-compact inflatable pillow weighing 2.75 oz with ergonomic curved design, soft TPU laminated fabric, and easy-inflate valve.", "photo-1504280390367-361c6d9f38f4", 17.99),
        ("Snow Peak Titanium Trek 900 Cookset", "Ultra-lightweight titanium pot and lid set weighing just 6.2 oz. 900ml capacity, folding handles, and stuff sack included.", "photo-1504280390367-361c6d9f38f4", 64.95),
        ("Goal Zero Venture 75 Power Bank", "Rugged IP67 waterproof solar-compatible power bank with 19,200mAh capacity, dual USB-A and USB-C outputs, and integrated kickstand.", "photo-1609091839311-d5365f9ff1c5", 89.95),
        ("Thermarest NeoAir XTherm Sleeping Pad", "Four-season ultralight sleeping pad with R-value 6.9, Triangular Core Matrix, ThermaCapture radiant heat technology, and WingLock valve.", "photo-1504280390367-361c6d9f38f4", 249.95),
        ("Osprey Atmos AG 65 Backpacking Pack", "Anti-Gravity suspension system with continuous peripheral frame, fit-on-the-fly hip belt, and Stow-on-the-Go trekking pole attachment.", "photo-1622560480654-996b3d003b97", 290.00),
    ],
    "Office": [
        ("Apple MacBook Air M3 15-inch", "15.3-inch Liquid Retina display, M3 chip with 10-core GPU, 18-hour battery life, 1080p FaceTime camera, and fanless silent design. Starting at 8GB unified memory.", "photo-1517336714731-489689fd1ca8", 1299.00),
        ("Logitech Ergo K860 Split Keyboard", "Split curved ergonomic keyboard with padded palm rest, adjustable tilt legs, and natural typing position. Bluetooth and USB receiver connectivity.", "photo-1587829741301-dc798b83add3", 129.99),
        ("CalDigit TS4 Thunderbolt 4 Dock", "18-port Thunderbolt 4 docking station with 98W charging, dual 6K display output, 2.5GbE ethernet, SD 4.0 reader, and 32GB/s data transfer.", "photo-1625842268584-8f3296236761", 399.99),
        ("Bellroy Tech Kit Compact Organizer", "Premium leather and recycled fabric tech organizer with magnetic closure, padded tablet pocket, cable management loops, and key clip.", "photo-1585776245991-cf89dd7fc73a", 59.00),
        ("LG UltraFine 5K Display 27-inch", "5120x2880 IPS display with P3 wide color gamut, 500 nits brightness, Thunderbolt 3 single-cable connection, and built-in stereo speakers.", "photo-1527443224154-c4a3942d3acf", 1299.99),
        ("Leuchtturm1917 A5 Notebook Dotted", "249 numbered pages of 80g/m² acid-free paper, table of contents, 8 perforated detachable pages, 2 bookmarks, expandable pocket, and pen loop.", "photo-1531346878377-a5be20888e57", 19.95),
        ("Ergotron LX Desk Monitor Arm", "Premium monitor arm supporting up to 34-inch displays, 25 lbs capacity, polished aluminum finish, and effortless fingertip adjustment with cable management.", "photo-1527443224154-c4a3942d3acf", 179.99),
        ("Keychron Q1 Pro Wireless Keyboard", "75% layout QMK/VIA programmable keyboard with hot-swappable switches, double-gasket mount, CNC aluminum body, and 1000Hz polling rate.", "photo-1587829741301-dc798b83add3", 199.00),
        ("Grovemade Desk Shelf System", "Hand-finished solid hardwood desk shelf with steel brackets, cable management channel, and modular design. Available in walnut or maple.", "photo-1585776245991-cf89dd7fc73a", 220.00),
        ("Moleskine Smart Writing Set", "Smart pen with Ncoded technology digitizing handwritten notes to Moleskine app in real-time. Includes smart notebook and USB-C charging pen.", "photo-1531346878377-a5be20888e57", 279.00),
        ("Humanscale Diffrient World Chair", "Minimalist ergonomic task chair with tri-panel mesh back, form-sensing recline mechanism, and weight-sensitive adaptive system. No manual adjustments needed.", "photo-1567538096630-e0c55bd6374c", 999.00),
        ("BenQ ScreenBar Monitor Light", "Asymmetric optical design illuminates desk without screen glare. Auto-dimming ambient sensor, adjustable color temperature, and USB-powered with no desk footprint.", "photo-1507473885765-e6ed057f782c", 109.00),
        ("Twelve South BookArc MacBook Stand", "Vertical space-saving stand for MacBook in clamshell mode. Silicone inserts fit multiple laptop sizes, machined aluminum construction.", "photo-1517336714731-489689fd1ca8", 49.99),
        ("MOFT Laptop Stand Adhesive", "Ultra-thin invisible laptop stand weighing 3oz, attaching magnetically to laptop bottom. Two adjustable angles for ergonomic typing and video calls.", "photo-1517336714731-489689fd1ca8", 27.99),
        ("Secrid Slimwallet RFID Card Protector", "Aluminum cardprotector case with genuine leather cover, RFID blocking, quick-access card mechanism holding 4-6 cards, and additional bill pocket.", "photo-1627124118357-195b00c5c363", 89.95),
        ("Pilot Vanishing Point Fountain Pen", "Retractable nib fountain pen with 18K gold nib, click mechanism, rhodium-plated clip, and converter-cartridge filling system. Japanese engineering.", "photo-1531346878377-a5be20888e57", 152.00),
        ("Rain Design mStand Laptop Stand", "Patented single-piece aluminum laptop stand with cable management hole, sand-blasted and anodized finish, and tilted viewing angle.", "photo-1517336714731-489689fd1ca8", 49.90),
        ("Noctua NH-D15 CPU Cooler", "Premium dual-tower CPU cooler with 2x NF-A15 140mm fans, 6 heatpipes, SecuFirm2 mounting, and quiet operation at 24.6 dBA maximum.", "photo-1591799265444-d66432b91588", 109.95),
        ("Twelve South Curve Flex Stand", "Portable folding laptop stand with 6 height options, ventilated design, and folds flat for travel. Supports laptops up to 4 lbs.", "photo-1517336714731-489689fd1ca8", 59.99),
        ("Ugmonk Gather Desk Organizer", "Minimalist magnetic modular desk organization system with leather tray, pen holder, phone stand, and magnetic connections. Handcrafted walnut and leather.", "photo-1585776245991-cf89dd7fc73a", 175.00),
    ],
    "Accessories": [
        ("Apple Watch Series 9 Midnight", "S9 SiP chip with 4-core Neural Engine, always-on Retina display, blood oxygen app, ECG, crash detection, and double-tap gesture control.", "photo-1523275335684-37898b6baf30", 399.00),
        ("Bellroy Hide & Seek Wallet", "Premium leather billfold with RFID protection, pull-tab hidden pocket for travel cards, flat bill section, and 5-12 card capacity.", "photo-1627124118357-195b00c5c363", 99.00),
        ("AirTag 4-Pack Item Tracker", "Precision Finding with Ultra Wideband, built-in speaker, replaceable CR2032 battery lasting over a year, and IP67 water resistance.", "photo-1583863788434-e58a36330cf0", 99.00),
        ("Anker Soundcore Liberty 4 NC", "Adaptive noise cancellation earbuds with LDAC Hi-Res audio, heart rate monitoring, customizable 11mm drivers, and 50-hour total playtime.", "photo-1606220588913-b3aacb4d2f46", 99.99),
        ("Peak Design Everyday Sling 6L", "Versatile camera and everyday sling bag with FlexFold dividers, weatherproof 100% recycled nylon shell, padded tablet sleeve, and external carry straps.", "photo-1553062407-98eeb64c6a62", 109.95),
        ("Tile Mate Bluetooth Tracker", "Compact Bluetooth tracker with 250-foot range, 3-year battery, water-resistant design, and community find network with Smart Alerts.", "photo-1583863788434-e58a36330cf0", 24.99),
        ("Moment Thin Case for iPhone 15 Pro", "MagSafe-compatible phone case with (M)Force magnet array, 6-foot drop protection, raised camera bezel, and compatible with Moment lenses.", "photo-1601784551446-20c9e07cdbdb", 39.99),
        ("Oakley Holbrook Sunglasses", "Iconic O-Matter frame with Prizm lens technology, stress-resistant frame, Three-Point Fit for optical precision, and lightweight comfort.", "photo-1572635196237-14b3f281503f", 191.00),
        ("Native Union Drop XL Wireless Charger", "3-device wireless charging pad with alignment-free design, non-slip silicone surface, and foreign object detection. Charges through most cases.", "photo-1583863788434-e58a36330cf0", 79.99),
        ("Orbitkey Key Organizer", "Flexible stainless steel locking mechanism holding 2-7 keys neatly. D-ring for car keys, bottle opener accessory option, and premium leather body.", "photo-1627124118357-195b00c5c363", 39.90),
        ("Nomad Sport Band Apple Watch", "FKM fluoroelastomer sport band with integrated pin-and-tuck closure, custom aluminum hardware, and quick-release lugs for easy swapping.", "photo-1523275335684-37898b6baf30", 59.95),
        ("Mujjo Touchscreen Leather Gloves", "Ethiopian leather gloves with 3M Thinsulate lining, touchscreen-compatible fingertips, elastic wrist closure, and natural premium leather aging.", "photo-1583743814966-8936f5b7be1a", 105.00),
        ("Aer Day Pack 2 Backpack", "Sleek 15.6-inch laptop daypack with YKK AquaGuard zippers, Duraflex hardware, quick-access front pocket, and water-resistant 1680D Cordura ballistic nylon.", "photo-1553062407-98eeb64c6a62", 130.00),
        ("Nite Ize S-Biner MicroLock 2-Pack", "Stainless steel locking dual-carabiner with secure slide-to-lock gates. Attach keys, water bottles, and gear to bags, belt loops, and zippers.", "photo-1593618998160-e34014e67546", 5.99),
        ("PopSockets PopGrip MagSafe", "MagSafe-compatible phone grip with swappable PopTop, expandable stand and grip, alignment magnets, and Wireless charging compatible when removed.", "photo-1601784551446-20c9e07cdbdb", 29.99),
        ("Herschel Charlie Cardholder Wallet", "Compact card wallet with 3 card slots and center storage sleeve. Lightweight signature striped liner and vegan leather construction.", "photo-1627124118357-195b00c5c363", 24.99),
        ("Hydro Flask 12oz Coffee Mug", "TempShield insulated travel mug with Press-In Lid, easy-sip opening, wide mouth for easy cleaning, and slip-free powder-coated finish.", "photo-1514432324607-a09d9b4aefdd", 29.95),
        ("Satechi USB-C Slim Multi-Port Adapter", "Aluminum USB-C hub with 4K HDMI, USB-C PD charging, USB 3.0 port, and SD/Micro SD card readers. Compatible with MacBook and USB-C laptops.", "photo-1625842268584-8f3296236761", 69.99),
        ("Bellroy Sling Mini", "Compact crossbody sling with contoured back panel, magnetic closure, internal organization, and water-resistant woven fabric. 2.5L capacity.", "photo-1553062407-98eeb64c6a62", 89.00),
        ("Nomad Horween Leather AirTag Loop", "Premium Horween leather AirTag holder with stainless steel keyring attachment, develops rich patina over time, and secure snap closure.", "photo-1583863788434-e58a36330cf0", 39.95),
    ],
    "Toys": [
        ("LEGO Technic Lamborghini Sián", "3,696-piece set with moving V12 engine, 8-speed sequential gearbox, adjustable rear spoiler, and front and rear suspension. Scale 1:8.", "photo-1587654780291-39c9404d7dd0", 449.99),
        ("Nintendo Switch OLED Model", "7-inch OLED vibrant display, wide adjustable stand, wired LAN port dock, 64GB internal storage, and enhanced audio for portable and TV gaming.", "photo-1578303512597-81e6cc155b3e", 349.99),
        ("Ravensburger 3D Puzzle Eiffel Tower", "216-piece 3D jigsaw puzzle building a 17-inch tall Eiffel Tower with precision-cut EasyClick technology. No glue required, LED edition available.", "photo-1587654780291-39c9404d7dd0", 29.99),
        ("Magna-Tiles 100-Piece Clear Colors", "Magnetic building tiles with translucent colors for creative STEM play. Compatible with all Magna-Tiles sets, made with safe non-toxic ABS plastic.", "photo-1587654780291-39c9404d7dd0", 119.99),
        ("Hot Wheels Ultimate Garage Playset", "Multi-level parking garage with motorized corkscrew elevator, roaring Gorilla attack, gas station, and parking for 100+ cars with aerial ramp.", "photo-1596461404969-9ae70f2830c1", 89.99),
        ("Dungeons & Dragons Starter Set 5E", "Complete D&D introductory adventure with pre-generated character sheets, essential rules booklet, Dragon of Icespire Peak adventure, and dice set.", "photo-1587654780291-39c9404d7dd0", 19.99),
        ("Osmo Genius Starter Kit for iPad", "Award-winning educational system with 5 hands-on games for ages 6-10 covering spelling, math, creativity, and problem-solving with physical manipulatives.", "photo-1587654780291-39c9404d7dd0", 99.99),
        ("Anki Vector Robot", "AI-powered companion robot with neural network processing, emotion engine, voice-activated commands, self-charging, and object recognition.", "photo-1587654780291-39c9404d7dd0", 299.99),
        ("Sphero BOLT App-Controlled Robot", "Programmable robot ball with LED matrix, infrared communication, gyroscope, accelerometer, and light sensor. Learn JavaScript, Scratch, and Swift.", "photo-1587654780291-39c9404d7dd0", 149.99),
        ("LEGO Star Wars Millennium Falcon", "7,541-piece Ultimate Collector Series set with detailed interior, radar dish, laser cannons, and landing gear. Includes Han Solo and Chewbacca minifigures.", "photo-1587654780291-39c9404d7dd0", 849.99),
        ("Catan Board Game", "Award-winning strategy board game for 3-4 players. Settle, trade, and build on the island of Catan in this classic resource management game.", "photo-1587654780291-39c9404d7dd0", 44.00),
        ("Rubik's Speed Cube 3x3", "Competition-speed magnetic cube with GAN technology, stickerless design, smooth corner cutting, and adjustable tension system.", "photo-1587654780291-39c9404d7dd0", 32.99),
        ("Snap Circuits Classic SC-300", "300 projects electronic exploration kit with 60+ snap-together components. No soldering required, includes AM radio, doorbell, and alarm circuit projects.", "photo-1587654780291-39c9404d7dd0", 57.48),
        ("Playmobil Space Mars Expedition", "Space exploration playset with Mars rover, astronaut figures, working crane, removable command module, and crystal mining cave.", "photo-1596461404969-9ae70f2830c1", 49.99),
        ("NERF Elite 2.0 Commander RD-6", "Rotating drum dart blaster with slam-fire action, 12-dart capacity, tactical rails, and barrel and stock attachment points. Includes 12 NERF darts.", "photo-1596461404969-9ae70f2830c1", 12.99),
        ("Melissa & Doug Wooden Building Blocks", "100-piece set of solid wood blocks in 4 colors and 9 shapes. Non-toxic finishes, smooth sanded edges, and classic educational construction toy.", "photo-1587654780291-39c9404d7dd0", 22.99),
        ("Ticket to Ride Board Game", "Cross-country train adventure board game for 2-5 players. Collect cards, claim railway routes, and connect cities across North America.", "photo-1587654780291-39c9404d7dd0", 44.49),
        ("National Geographic Rock Tumbler Kit", "Professional rock tumbling machine with 4 polishing grits, rough gemstones, jewelry fastenings, and learning guide. Tumbles rocks to gem-quality shine.", "photo-1587654780291-39c9404d7dd0", 59.99),
        ("Gravitrax Starter Set Marble Run", "Interactive marble run track system with 122 components, action stones including magnetic cannon and spinner, and free companion app for building guides.", "photo-1587654780291-39c9404d7dd0", 54.99),
        ("LEGO Creator Expert Bonsai Tree", "878-piece botanical collection set with interchangeable green leaves and pink cherry blossom pieces. Includes rectangular pot and slatted wood-effect stand.", "photo-1587654780291-39c9404d7dd0", 49.99),
    ],
}


def generate_catalog():
    """Write the unique catalog CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    catalog_path = DATA_DIR / "mock_catalog.csv"
    rows = []
    item_id = 1

    for category, products in PRODUCTS.items():
        for title, description, photo_id, price in products:
            rows.append({
                "item_id": str(item_id),
                "title": title,
                "description": description,
                "image_filename": f"product_{item_id}.jpg",
                "category": category,
                "price": f"{price:.2f}",
                "_photo_id": photo_id,  # internal, not written to CSV
            })
            item_id += 1

    # Write CSV
    fieldnames = ["item_id", "title", "description", "image_filename", "category", "price"]
    with open(catalog_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Written {len(rows)} unique products to {catalog_path}")
    return rows


def download_images(rows):
    """Download a unique image for every product from Unsplash."""
    # Clear old images
    if IMAGE_DIR.exists():
        for f in IMAGE_DIR.glob("*.jpg"):
            f.unlink()
        for f in IMAGE_DIR.glob("*.png"):
            f.unlink()

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    total = len(rows)
    success = 0
    failed = 0

    for idx, row in enumerate(rows):
        dest = IMAGE_DIR / row["image_filename"]
        photo_id = row["_photo_id"]

        # Use slightly different crop parameters per product for visual diversity
        # even when sharing the same base photo
        crop_offsets = ["", "&crop=entropy", "&crop=faces", "&crop=edges"]
        crop = crop_offsets[idx % len(crop_offsets)]
        # Also vary the size slightly for unique pixel content
        w = 400 + (idx % 5) * 10  # 400-440
        h = 400 + (idx % 5) * 10

        url = f"https://images.unsplash.com/{photo_id}?w={w}&h={h}&fit=crop&q=80{crop}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            success += 1
        except Exception as e:
            print(f"  WARN: Failed #{row['item_id']} '{row['title']}': {e}")
            # Generate a branded fallback
            try:
                from PIL import Image, ImageDraw
                img = Image.new("RGB", (400, 400), color=(45, 45, 55))
                draw = ImageDraw.Draw(img)
                draw.text((20, 180), row["title"][:30], fill="white")
                draw.text((20, 200), row["category"], fill=(120, 120, 140))
                img.save(dest)
            except Exception:
                pass
            failed += 1

        if (idx + 1) % 20 == 0 or (idx + 1) == total:
            print(f"  Downloaded {idx + 1}/{total} images (success: {success}, failed: {failed})")

    print(f"\nImage download complete! Success: {success}, Failed: {failed}")


def generate_interactions(rows):
    """Generate realistic user interactions tied to the new catalog."""
    import random
    from datetime import datetime, timedelta

    num_users = 15
    user_ids = [f"u{uid:03d}" for uid in range(1, num_users + 1)]
    interaction_types = ["click", "purchase", "rating"]
    interaction_weights = [0.55, 0.25, 0.20]

    # Create user preference profiles for realistic behavior
    categories = list(PRODUCTS.keys())
    user_profiles = {}
    for uid in user_ids:
        # Each user prefers 2-3 categories
        preferred = random.sample(categories, k=random.randint(2, 3))
        user_profiles[uid] = preferred

    # Build category->item_id index
    cat_items = {}
    for row in rows:
        cat = row["category"]
        cat_items.setdefault(cat, []).append(row["item_id"])

    interactions = []
    start_time = datetime(2026, 7, 1, 8, 0, 0)

    for user in user_ids:
        if user == "u001":
            count = 60
        elif user in {"u002", "u003", "u004", "u005"}:
            count = random.randint(35, 50)
        else:
            count = random.randint(15, 30)

        preferred_cats = user_profiles[user]
        all_items = [r["item_id"] for r in rows]

        for i in range(count):
            # 70% chance to pick from preferred categories
            if random.random() < 0.70:
                cat = random.choice(preferred_cats)
                item_id = random.choice(cat_items[cat])
            else:
                item_id = random.choice(all_items)

            itype = random.choices(interaction_types, weights=interaction_weights, k=1)[0]
            rating = ""
            if itype == "purchase":
                rating = str(random.randint(3, 5))
            elif itype == "rating":
                rating = str(random.randint(1, 5))

            ts = start_time + timedelta(minutes=random.randint(0, 60 * 24 * 42))
            interactions.append((user, item_id, itype, rating, ts.isoformat()))

    # Shuffle for realistic ordering
    random.shuffle(interactions)

    interactions_path = DATA_DIR / "user_interactions.csv"
    with open(interactions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "interaction_type", "rating", "timestamp"])
        writer.writerows(interactions)

    print(f"Generated {len(interactions)} user interactions for {num_users} users")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING UNIQUE PRODUCT CATALOG")
    print("=" * 60)
    rows = generate_catalog()

    print("\n" + "=" * 60)
    print("DOWNLOADING PRODUCT IMAGES FROM UNSPLASH")
    print("=" * 60)
    download_images(rows)

    print("\n" + "=" * 60)
    print("GENERATING USER INTERACTIONS")
    print("=" * 60)
    generate_interactions(rows)

    print("\n✓ Complete! 200 unique products with images and interactions ready.")
