
var startDate = '2017-02-16'; //YYYY-MM-DD
var finishDate = '2017-06-15';


var region = '[[-120.8913,47.5137], [-120.7899,47.5137], [-120.8913,47.4451], [-120.7899,47.4451]]'
var rectangle = [-120.8913,47.4451,-120.7899,47.5137];

var rectangle1 = ee.Geometry.Rectangle(rectangle);

var dataset = ee.ImageCollection("COPERNICUS/S2").filterBounds(rectangle1)
.filterDate(startDate, finishDate)
    .sort('system:time_start', true);

    var selectors = ["B2","B3","B4","B8","B12","QA60"]

    var mean_cloud = function(image){
          return(image.select("QA60").reduceRegion({
                reducer: ee.Reducer.mean(),
                  geometry: rectangle1,
                    scale: 10}).get('QA60'))
    };


//select only the useful bands
//dataset = dataset.select(selectors)
//var data = dataset.toList(dataset.size());
//print(ee.Image(data.get(0)));
//for (var i=0; i<100; i++){
//  var image = ee.Image(data.get(i));
//    print(mean_cloud(image),image.get("MGRS_TILE"));
//      image = image.select(["B2","B3","B4","B8","B12"]);
//        print(image.getDownloadURL(
//          {'region': region,
//            'scale': 10}));
//            }
