function getBathValue() {
    var uiBathrooms = document.getElementsByName("uiBathrooms");
    for(var i = 0; i < uiBathrooms.length; i++) {
      if(uiBathrooms[i].checked) {
          return parseInt(uiBathrooms[i].value);
      }
    }
    return -1; // Invalid Value
  }
  
  function getBHKValue() {
    var uiBHK = document.getElementsByName("uiBHK");
    for(var i = 0; i < uiBHK.length; i++) {
      if(uiBHK[i].checked) {
          return parseInt(uiBHK[i].value);
      }
    }
    return -1; // Invalid Value
  }
  
  function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");
    var sqft = document.getElementById("uiSqft");
    var bhk = getBHKValue();
    var bathrooms = getBathValue();
    var location = document.getElementById("uiLocations");
    var estPrice = document.getElementById("uiEstimatedPrice");
  
    // Debug logging
    console.log("Values being sent:", {
        total_sqft: parseFloat(sqft.value),
        bhk: bhk,
        bath: bathrooms,
        location: location.value
    });
  
    var url = "http://127.0.0.1:5000/predict_home_price"; //Use this if you are NOT using nginx which is first 7 tutorials
    //var url = "/api/predict_home_price"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
  
    $.ajax({
        url: url,
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            total_sqft: parseFloat(sqft.value),
            bhk: bhk,
            bath: bathrooms,
            location: location.value
        }),
                success: function(data, status) {
            console.log("Response data:", data);
            console.log("Status:", status);
            if(data.status === "success") {
                // Format the price in Indian Rupees
                const priceInRupees = data.estimated_price * 100000; // Convert lakhs to rupees
                const formattedPrice = new Intl.NumberFormat('en-IN', {
                    style: 'currency',
                    currency: 'INR',
                    maximumFractionDigits: 0
                }).format(priceInRupees);
                
                document.getElementById("priceValue").textContent = formattedPrice;
                document.getElementById("uiEstimatedPrice").style.display = "block";
            } else {
                document.getElementById("priceValue").textContent = "Error: " + (data.message || "Unknown error");
                document.getElementById("uiEstimatedPrice").style.display = "block";
            }
        },
        error: function(xhr, status, error) {
            console.error("Request failed:", error);
            console.error("Response:", xhr.responseText);
            document.getElementById("priceValue").textContent = "Error: Request failed";
            document.getElementById("uiEstimatedPrice").style.display = "block";
        }
    });
  }
  
  function onPageLoad() {
    console.log( "document loaded" );
    var url = "http://127.0.0.1:5000/get_location_names"; // Use this if you are NOT using nginx which is first 7 tutorials
    //var url = "/api/get_location_names"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
    $.get(url,function(data, status) {
        console.log("got response for get_location_names request");
        if(data) {
            var locations = data.locations;
            var uiLocations = document.getElementById("uiLocations");
            $('#uiLocations').empty();
            for(var i in locations) {
                var opt = new Option(locations[i]);
                $('#uiLocations').append(opt);
            }
        }
    });
  }
  
  window.onload = onPageLoad;