# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionAskCategories(Action):
    def name(self) -> Text:
        return "action_ask_categories"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Please mention the categories")
        return []

class ActionAskSubcategories(Action):
    def name(self) -> Text:
        return "action_ask_subcategories"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Also provide the subcategories")
        return []

class ActionAskProductDetails(Action):
    def name(self) -> Text:
        return "action_ask_product_details"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Now mention the details like brand and price")
        return []

class ActionShowProduct(Action):
    def name(self) -> Text:
        return "action_show_product"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # category = next(tracker.get_latest_entity_values("categories"))
        # subcategory = next(tracker.get_latest_entity_values("Subcategories"))
        brand = next(tracker.get_latest_entity_values("Brand"))
        price = next(tracker.get_latest_entity_values("Price"))

        response = f"Here are your search results: Brand: {brand}\nPrice: {price}\nDescription: Any"
        dispatcher.utter_message(response)
        return []
