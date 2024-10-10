import Mathlib

namespace customers_who_tipped_l3896_389692

theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) : 
  initial_customers = 29 →
  additional_customers = 20 →
  non_tipping_customers = 34 →
  initial_customers + additional_customers - non_tipping_customers = 15 :=
by sorry

end customers_who_tipped_l3896_389692


namespace product_evaluation_l3896_389637

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 6560 := by
  sorry

end product_evaluation_l3896_389637


namespace count_valid_functions_l3896_389633

def polynomial_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x * f (-x) = f (x^3)

theorem count_valid_functions :
  ∃! (valid_functions : Finset (ℝ → ℝ)),
    (∀ f ∈ valid_functions, ∃ a b c d : ℝ, 
      (∀ x : ℝ, f x = polynomial_function a b c d x) ∧
      satisfies_condition f) ∧
    (Finset.card valid_functions = 12) :=
sorry

end count_valid_functions_l3896_389633


namespace exponent_multiplication_l3896_389675

theorem exponent_multiplication (m : ℝ) : 5 * m^2 * m^3 = 5 * m^5 := by sorry

end exponent_multiplication_l3896_389675


namespace advanced_purchase_ticket_price_l3896_389613

/-- Given information about ticket sales for an art exhibition, prove the price of advanced-purchase tickets. -/
theorem advanced_purchase_ticket_price
  (total_tickets : ℕ)
  (total_revenue : ℚ)
  (door_price : ℚ)
  (advanced_tickets : ℕ)
  (h_total_tickets : total_tickets = 140)
  (h_total_revenue : total_revenue = 1720)
  (h_door_price : door_price = 14)
  (h_advanced_tickets : advanced_tickets = 100) :
  ∃ (advanced_price : ℚ),
    advanced_price * advanced_tickets + door_price * (total_tickets - advanced_tickets) = total_revenue ∧
    advanced_price = 11.60 :=
by sorry

end advanced_purchase_ticket_price_l3896_389613


namespace jack_lifetime_l3896_389663

theorem jack_lifetime :
  ∀ (L : ℝ),
  (L = (1/6)*L + (1/12)*L + (1/7)*L + 5 + (1/2)*L + 4) →
  L = 84 := by
sorry

end jack_lifetime_l3896_389663


namespace division_problem_l3896_389687

theorem division_problem (x y : ℕ+) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 
  11 * y - x = 1 := by
  sorry

end division_problem_l3896_389687


namespace range_of_even_quadratic_function_l3896_389609

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

-- Define the property of being even
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the domain of the function
def domain (b : ℝ) : Set ℝ := {x | -2*b ≤ x ∧ x ≤ 3*b - 1}

-- State the theorem
theorem range_of_even_quadratic_function (a b : ℝ) :
  (∀ x ∈ domain b, f a b x = f a b (-x)) →
  {y | ∃ x ∈ domain b, f a b x = y} = {y | 1 ≤ y ∧ y ≤ 5} :=
sorry

end range_of_even_quadratic_function_l3896_389609


namespace opposite_signs_and_larger_negative_l3896_389691

theorem opposite_signs_and_larger_negative (a b : ℝ) : 
  a + b < 0 → a * b < 0 → 
  ((a < 0 ∧ b > 0 ∧ |a| > |b|) ∨ (a > 0 ∧ b < 0 ∧ |a| < |b|)) := by
  sorry

end opposite_signs_and_larger_negative_l3896_389691


namespace hyperbola_real_axis_length_l3896_389623

/-- The length of the real axis of a hyperbola with equation x²/3 - y²/6 = 1 is 2√3 -/
theorem hyperbola_real_axis_length : 
  ∃ (f : ℝ × ℝ → ℝ), 
    (∀ x y, f (x, y) = x^2 / 3 - y^2 / 6) ∧ 
    (∃ a : ℝ, a > 0 ∧ (∀ x y, f (x, y) = 1 → x^2 / a^2 - y^2 / (2*a^2) = 1) ∧ 2*a = 2 * Real.sqrt 3) :=
by sorry

end hyperbola_real_axis_length_l3896_389623


namespace number_difference_l3896_389626

theorem number_difference (L S : ℕ) (h1 : L = 1631) (h2 : L = 6 * S + 35) : L - S = 1365 := by
  sorry

end number_difference_l3896_389626


namespace ricks_books_l3896_389668

theorem ricks_books (N : ℕ) : (N / 2 / 2 / 2 / 2 = 25) → N = 400 := by
  sorry

end ricks_books_l3896_389668


namespace not_all_linear_functions_increasing_l3896_389664

/-- A linear function from ℝ to ℝ -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

/-- A function is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- Theorem: Not all linear functions with non-zero slope are increasing on ℝ -/
theorem not_all_linear_functions_increasing :
  ¬(∀ k b : ℝ, k ≠ 0 → IsIncreasing (LinearFunction k b)) := by sorry

end not_all_linear_functions_increasing_l3896_389664


namespace divisor_calculation_l3896_389636

theorem divisor_calculation (dividend : Float) (quotient : Float) (h1 : dividend = 0.0204) (h2 : quotient = 0.0012000000000000001) :
  dividend / quotient = 17 := by
  sorry

end divisor_calculation_l3896_389636


namespace max_non_managers_l3896_389646

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 →
  (managers : ℚ) / non_managers > 5 / 24 →
  non_managers ≤ 38 :=
by
  sorry

end max_non_managers_l3896_389646


namespace sqrt_expression_equals_two_l3896_389635

theorem sqrt_expression_equals_two : 
  Real.sqrt 12 - 3 * Real.sqrt (1/3) + |2 - Real.sqrt 3| = 2 := by sorry

end sqrt_expression_equals_two_l3896_389635


namespace clothing_problem_l3896_389649

/-- Calculates the remaining clothing pieces after donations and discarding --/
def remaining_clothing (initial : ℕ) (donated1 : ℕ) (donated2_multiplier : ℕ) (discarded : ℕ) : ℕ :=
  initial - (donated1 + donated1 * donated2_multiplier) - discarded

/-- Theorem stating that given the specific values in the problem, 
    the remaining clothing pieces is 65 --/
theorem clothing_problem : 
  remaining_clothing 100 5 3 15 = 65 := by
  sorry

end clothing_problem_l3896_389649


namespace discount_calculation_l3896_389614

theorem discount_calculation (cost_price : ℝ) (profit_with_discount : ℝ) (profit_without_discount : ℝ) :
  cost_price = 100 ∧ profit_with_discount = 20 ∧ profit_without_discount = 25 →
  (cost_price + cost_price * profit_without_discount / 100) - (cost_price + cost_price * profit_with_discount / 100) = 5 := by
sorry

end discount_calculation_l3896_389614


namespace diane_allison_age_ratio_l3896_389610

/-- Proves that the ratio of Diane's age to Allison's age when Diane turns 30 is 2:1 -/
theorem diane_allison_age_ratio :
  -- Diane's current age
  ∀ (diane_current_age : ℕ),
  -- Sum of Alex's and Allison's current ages
  ∀ (alex_allison_sum : ℕ),
  -- Diane's age when she turns 30
  ∀ (diane_future_age : ℕ),
  -- Alex's age when Diane turns 30
  ∀ (alex_future_age : ℕ),
  -- Allison's age when Diane turns 30
  ∀ (allison_future_age : ℕ),
  -- Conditions
  diane_current_age = 16 →
  alex_allison_sum = 47 →
  diane_future_age = 30 →
  alex_future_age = 2 * diane_future_age →
  -- Conclusion
  (diane_future_age : ℚ) / (allison_future_age : ℚ) = 2 := by
  sorry

end diane_allison_age_ratio_l3896_389610


namespace floor_area_less_than_10_l3896_389630

/-- Represents a rectangular room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The condition that each wall requires more paint than the floor -/
def more_paint_on_walls (r : Room) : Prop :=
  r.length * r.height > r.length * r.width ∧
  r.width * r.height > r.length * r.width

/-- The floor area of the room -/
def floor_area (r : Room) : ℝ :=
  r.length * r.width

/-- Theorem stating that for a room with height 3 meters and more paint required for walls than floor,
    the floor area must be less than 10 square meters -/
theorem floor_area_less_than_10 (r : Room) 
  (h1 : r.height = 3)
  (h2 : more_paint_on_walls r) : 
  floor_area r < 10 := by
  sorry


end floor_area_less_than_10_l3896_389630


namespace one_correct_meal_servings_l3896_389624

def number_of_people : ℕ := 10
def number_of_meal_choices : ℕ := 3
def beef_orders : ℕ := 2
def chicken_orders : ℕ := 4
def fish_orders : ℕ := 4

theorem one_correct_meal_servings :
  (∃ (ways : ℕ), 
    ways = number_of_people * 
      (((beef_orders - 1) * (chicken_orders * fish_orders)) + 
       ((chicken_orders - 1) * beef_orders * fish_orders) + 
       ((fish_orders - 1) * beef_orders * chicken_orders)) ∧
    ways = 180) :=
by sorry

end one_correct_meal_servings_l3896_389624


namespace power_minus_product_equals_one_l3896_389615

theorem power_minus_product_equals_one : 3^2 - (4 * 2) = 1 := by
  sorry

end power_minus_product_equals_one_l3896_389615


namespace past_five_weeks_income_sum_l3896_389684

/-- Represents the weekly income of a salesman -/
structure WeeklyIncome where
  base : ℕ
  commission : ℕ

/-- Calculates the total income for a given number of weeks -/
def totalIncome (income : WeeklyIncome) (weeks : ℕ) : ℕ :=
  (income.base + income.commission) * weeks

/-- Represents the salesman's income data -/
structure SalesmanIncome where
  baseSalary : ℕ
  pastWeeks : ℕ
  futureWeeks : ℕ
  avgCommissionFuture : ℕ
  avgTotalIncome : ℕ

/-- Theorem: The sum of weekly incomes for the past 5 weeks is $2070 -/
theorem past_five_weeks_income_sum 
  (s : SalesmanIncome) 
  (h1 : s.baseSalary = 400)
  (h2 : s.pastWeeks = 5)
  (h3 : s.futureWeeks = 2)
  (h4 : s.avgCommissionFuture = 315)
  (h5 : s.avgTotalIncome = 500)
  (h6 : s.pastWeeks + s.futureWeeks = 7) :
  totalIncome ⟨s.baseSalary, 0⟩ s.pastWeeks + 
  (s.avgTotalIncome * (s.pastWeeks + s.futureWeeks) - 
   totalIncome ⟨s.baseSalary, s.avgCommissionFuture⟩ s.futureWeeks) = 2070 :=
by
  sorry


end past_five_weeks_income_sum_l3896_389684


namespace cafeteria_apples_l3896_389669

/-- The number of apples handed out to students -/
def apples_handed_out : ℕ := 5

/-- The number of pies made -/
def pies_made : ℕ := 9

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 5

/-- The initial number of apples in the cafeteria -/
def initial_apples : ℕ := apples_handed_out + pies_made * apples_per_pie

theorem cafeteria_apples : initial_apples = 50 := by sorry

end cafeteria_apples_l3896_389669


namespace stating_pond_population_species_c_l3896_389631

/-- Represents the number of fish initially tagged for each species -/
def initial_tagged : ℕ := 40

/-- Represents the total number of fish caught in the second catch -/
def second_catch : ℕ := 180

/-- Represents the number of tagged fish of Species C found in the second catch -/
def tagged_species_c : ℕ := 2

/-- Represents the total number of fish of Species C in the pond -/
def total_species_c : ℕ := 3600

/-- 
Theorem stating that given the conditions from the problem, 
the total number of fish for Species C in the pond is 3600 
-/
theorem pond_population_species_c : 
  initial_tagged * second_catch / tagged_species_c = total_species_c := by
  sorry

end stating_pond_population_species_c_l3896_389631


namespace sin_30_degrees_l3896_389612

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end sin_30_degrees_l3896_389612


namespace teds_age_l3896_389686

/-- Given that Ted's age is 10 years less than three times Sally's age,
    and the sum of their ages is 65, prove that Ted is 46 years old. -/
theorem teds_age (t s : ℕ) 
  (h1 : t = 3 * s - 10)
  (h2 : t + s = 65) : 
  t = 46 := by
  sorry

end teds_age_l3896_389686


namespace max_value_function_l3896_389694

theorem max_value_function (x : ℝ) (h : x < 5/4) :
  4*x - 2 + 1/(4*x - 5) ≤ 1 := by
  sorry

end max_value_function_l3896_389694


namespace water_level_rise_l3896_389641

/-- The rise in water level when a cube is immersed in a rectangular vessel --/
theorem water_level_rise (cube_edge : ℝ) (vessel_length vessel_width : ℝ) 
  (h_cube_edge : cube_edge = 10)
  (h_vessel_length : vessel_length = 20)
  (h_vessel_width : vessel_width = 15) :
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 10 ^ 3 / (20 * 15) :=
by sorry

end water_level_rise_l3896_389641


namespace max_men_with_all_items_and_married_l3896_389667

def total_men : ℕ := 500
def married_men : ℕ := 350
def men_with_tv : ℕ := 375
def men_with_radio : ℕ := 450
def men_with_car : ℕ := 325
def men_with_refrigerator : ℕ := 275
def men_with_ac : ℕ := 300

theorem max_men_with_all_items_and_married (men_with_all_items_and_married : ℕ) :
  men_with_all_items_and_married ≤ men_with_refrigerator :=
by sorry

end max_men_with_all_items_and_married_l3896_389667


namespace max_distance_from_origin_l3896_389622

/-- The maximum distance a point can be from the origin, given the constraints --/
def max_distance : ℝ := 10

/-- The coordinates of the post where the dog is tied --/
def post : ℝ × ℝ := (6, 8)

/-- The length of the rope --/
def rope_length : ℝ := 15

/-- The x-coordinate of the wall's end --/
def wall_end : ℝ := 10

/-- Theorem stating the maximum distance from the origin --/
theorem max_distance_from_origin :
  ∀ (p : ℝ × ℝ), 
    (p.1 ≤ wall_end) → -- point is not beyond the wall
    (p.2 ≥ 0) → -- point is not below the wall
    ((p.1 - post.1)^2 + (p.2 - post.2)^2 ≤ rope_length^2) → -- point is within or on the circle
    (p.1^2 + p.2^2 ≤ max_distance^2) := -- distance from origin is at most max_distance
by
  sorry


end max_distance_from_origin_l3896_389622


namespace smallest_n_congruence_l3896_389645

theorem smallest_n_congruence (n : ℕ) : ∃ (m : ℕ), m > 0 ∧ (∀ k : ℕ, 0 < k → k < m → (7^k : ℤ) % 5 ≠ (k^7 : ℤ) % 5) ∧ (7^m : ℤ) % 5 = (m^7 : ℤ) % 5 ∧ m = 3 := by
  sorry

end smallest_n_congruence_l3896_389645


namespace pentagon_area_l3896_389677

/-- Given a grid with distance m between adjacent points, prove that for a quadrilateral ABCD with area 23, the area of pentagon EFGHI is 28. -/
theorem pentagon_area (m : ℝ) (area_ABCD : ℝ) : 
  m > 0 → area_ABCD = 23 → ∃ (area_EFGHI : ℝ), area_EFGHI = 28 := by
  sorry

end pentagon_area_l3896_389677


namespace sum_of_exterior_angles_is_360_l3896_389685

/-- A polygon is a closed planar figure with straight sides -/
structure Polygon where
  sides : ℕ
  sides_positive : sides > 2

/-- An exterior angle of a polygon -/
def exterior_angle (p : Polygon) : ℝ := sorry

/-- The sum of exterior angles of a polygon -/
def sum_of_exterior_angles (p : Polygon) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of any polygon is 360° -/
theorem sum_of_exterior_angles_is_360 (p : Polygon) : 
  sum_of_exterior_angles p = 360 := by sorry

end sum_of_exterior_angles_is_360_l3896_389685


namespace distinct_numbers_count_l3896_389625

/-- Represents a two-sided card with distinct numbers on each side -/
structure Card where
  side1 : ℕ
  side2 : ℕ
  distinct : side1 ≠ side2

/-- The set of four cards as described in the problem -/
def card_set : Finset Card := sorry

/-- A function that generates all possible three-digit numbers from the card set -/
def generate_numbers (cards : Finset Card) : Finset ℕ := sorry

/-- The main theorem stating that the number of distinct three-digit numbers is 192 -/
theorem distinct_numbers_count : 
  (generate_numbers card_set).card = 192 := by sorry

end distinct_numbers_count_l3896_389625


namespace equation_solution_l3896_389653

theorem equation_solution (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) 
  (h : 3 / x + 2 / y = 1 / 3) : 
  x = 9 * y / (y - 6) := by
  sorry

end equation_solution_l3896_389653


namespace sin_3alpha_inequality_l3896_389662

theorem sin_3alpha_inequality (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 6) :
  2 * Real.sin α < Real.sin (3 * α) ∧ Real.sin (3 * α) < 3 * Real.sin α := by
  sorry

end sin_3alpha_inequality_l3896_389662


namespace banana_bunch_count_l3896_389627

theorem banana_bunch_count (x : ℕ) : 
  (6 * x + 5 * 7 = 83) → x = 8 := by
sorry

end banana_bunch_count_l3896_389627


namespace data_grouping_l3896_389644

theorem data_grouping (max min interval : ℕ) (h1 : max = 145) (h2 : min = 50) (h3 : interval = 10) :
  (max - min + interval - 1) / interval = 10 := by
  sorry

end data_grouping_l3896_389644


namespace complex_modulus_problem_l3896_389638

theorem complex_modulus_problem (x y : ℝ) (h : Complex.I * Complex.mk x y = Complex.mk 3 4) :
  Complex.abs (Complex.mk x y) = 5 := by
  sorry

end complex_modulus_problem_l3896_389638


namespace pet_supply_store_dog_food_l3896_389699

/-- Given a pet supply store with cat food and dog food, prove the number of bags of dog food. -/
theorem pet_supply_store_dog_food (cat_food : ℕ) (difference : ℕ) : 
  cat_food = 327 → difference = 273 → cat_food + difference = 600 := by
  sorry

end pet_supply_store_dog_food_l3896_389699


namespace occupation_assignment_l3896_389602

-- Define the people and professions
inductive Person : Type
  | A | B | C

inductive Profession : Type
  | Teacher | Journalist | Doctor

-- Define the age relation
def OlderThan (p1 p2 : Person) : Prop := sorry

-- Define the profession assignment
def Occupation (p : Person) (prof : Profession) : Prop := sorry

theorem occupation_assignment :
  -- C is older than the doctor
  (∀ p, Occupation p Profession.Doctor → OlderThan Person.C p) →
  -- A's age is different from the journalist
  (∀ p, Occupation p Profession.Journalist → p ≠ Person.A) →
  -- The journalist is younger than B
  (∀ p, Occupation p Profession.Journalist → OlderThan Person.B p) →
  -- Each person has exactly one profession
  (∀ p, ∃! prof, Occupation p prof) →
  -- Each profession is assigned to exactly one person
  (∀ prof, ∃! p, Occupation p prof) →
  -- The only valid assignment is:
  Occupation Person.A Profession.Doctor ∧
  Occupation Person.B Profession.Teacher ∧
  Occupation Person.C Profession.Journalist :=
by sorry

end occupation_assignment_l3896_389602


namespace average_of_numbers_sixth_and_seventh_sum_l3896_389680

def numbers : List ℝ := [54, 55, 57, 58, 59, 63, 65, 65]

theorem average_of_numbers : 
  (List.sum numbers) / (List.length numbers : ℝ) = 60 :=
by sorry

theorem sixth_and_seventh_sum : 
  List.sum (List.drop 5 (List.take 7 numbers)) = 54 :=
by sorry

#check average_of_numbers
#check sixth_and_seventh_sum

end average_of_numbers_sixth_and_seventh_sum_l3896_389680


namespace prob_not_yellow_is_seven_tenths_l3896_389654

/-- Represents the contents of a bag of jelly beans -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the probability of selecting a non-yellow jelly bean -/
def probNotYellow (bag : JellyBeanBag) : ℚ :=
  let total := bag.red + bag.green + bag.yellow + bag.blue
  let notYellow := bag.red + bag.green + bag.blue
  notYellow / total

/-- Theorem: The probability of selecting a non-yellow jelly bean from a bag
    containing 4 red, 7 green, 9 yellow, and 10 blue jelly beans is 7/10 -/
theorem prob_not_yellow_is_seven_tenths :
  probNotYellow { red := 4, green := 7, yellow := 9, blue := 10 } = 7 / 10 := by
  sorry

end prob_not_yellow_is_seven_tenths_l3896_389654


namespace ellipse_properties_l3896_389607

-- Define the ellipse and its properties
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_def : c = Real.sqrt (a^2 - b^2)
  h_e_def : e = c / a

-- Define points and line
def F₁ (E : Ellipse) : ℝ × ℝ := (-E.c, 0)
def F₂ (E : Ellipse) : ℝ × ℝ := (E.c, 0)

-- Define the properties we want to prove
def perimeter_ABF₂ (E : Ellipse) (A B : ℝ × ℝ) : ℝ := sorry

def dot_product (v w : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem ellipse_properties (E : Ellipse) (A B : ℝ × ℝ) (h_A_on_C h_B_on_C : (A.1^2 / E.a^2) + (A.2^2 / E.b^2) = 1) 
  (h_l : ∃ (t : ℝ), A = F₁ E + t • (B - F₁ E)) :
  (perimeter_ABF₂ E A B = 4 * E.a) ∧ 
  (dot_product (A - F₁ E) (A - F₂ E) = 5 * E.c^2 → E.e ≥ Real.sqrt 7 / 7) ∧
  (dot_product (A - F₁ E) (A - F₂ E) = 6 * E.c^2 → E.e ≤ Real.sqrt 7 / 7) := by
  sorry

end ellipse_properties_l3896_389607


namespace complex_number_properties_l3896_389690

theorem complex_number_properties (z : ℂ) (h : (z - 2*I) / z = 2 + I) : 
  z.im = -1 ∧ z^6 = -8*I := by sorry

end complex_number_properties_l3896_389690


namespace smallest_undefined_value_l3896_389672

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y : ℝ, y > 0 ∧ y < 1/6 → (y - 3) / (12 * y^2 - 50 * y + 12) ≠ 0) ∧ 
  ((1/6 : ℝ) - 3) / (12 * (1/6)^2 - 50 * (1/6) + 12) = 0 := by
  sorry

end smallest_undefined_value_l3896_389672


namespace quadratic_shift_sum_l3896_389659

/-- Given a quadratic function f(x) = 3x^2 + 5x - 2, prove that when it's shifted 5 units to the left,
    resulting in a new quadratic function g(x) = ax^2 + bx + c, then a + b + c = 136. -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3 * x^2 + 5 * x - 2) →
  (∀ x, g x = f (x + 5)) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 136 := by
  sorry

end quadratic_shift_sum_l3896_389659


namespace at_op_four_neg_one_l3896_389693

/-- Definition of the @ operation -/
def at_op (x y : ℤ) : ℤ := x * (y + 2) + 2 * x * y

/-- Theorem stating that 4 @ (-1) = -4 -/
theorem at_op_four_neg_one : at_op 4 (-1) = -4 := by sorry

end at_op_four_neg_one_l3896_389693


namespace total_amount_is_70000_l3896_389674

/-- The total amount of money divided -/
def total_amount : ℕ := sorry

/-- The amount given at 10% interest -/
def amount_10_percent : ℕ := 60000

/-- The amount given at 20% interest -/
def amount_20_percent : ℕ := sorry

/-- The interest rate for the first part (10%) -/
def interest_rate_10 : ℚ := 1/10

/-- The interest rate for the second part (20%) -/
def interest_rate_20 : ℚ := 1/5

/-- The total profit after one year -/
def total_profit : ℕ := 8000

/-- Theorem stating that the total amount divided is 70,000 -/
theorem total_amount_is_70000 :
  total_amount = 70000 ∧
  amount_10_percent + amount_20_percent = total_amount ∧
  amount_10_percent * interest_rate_10 + amount_20_percent * interest_rate_20 = total_profit :=
sorry

end total_amount_is_70000_l3896_389674


namespace expression_evaluation_l3896_389650

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^y + 8 * y^x - 2 * x * y = 893 := by
  sorry

end expression_evaluation_l3896_389650


namespace exists_32_chinese_l3896_389605

/-- Represents the seating arrangement of businessmen at a round table. -/
structure Seating :=
  (japanese : ℕ)
  (korean : ℕ)
  (chinese : ℕ)
  (total : ℕ)
  (total_eq : japanese + korean + chinese = total)
  (japanese_positive : japanese > 0)

/-- The condition that between any two nearest Japanese, there are exactly as many Chinese as Koreans. -/
def equal_distribution (s : Seating) : Prop :=
  ∃ k : ℕ, s.chinese = k * s.japanese ∧ s.korean = k * s.japanese

/-- The main theorem stating that it's possible to have 32 Chinese in a valid seating arrangement. -/
theorem exists_32_chinese : 
  ∃ s : Seating, s.total = 50 ∧ equal_distribution s ∧ s.chinese = 32 :=
sorry


end exists_32_chinese_l3896_389605


namespace systematic_sampling_relation_third_group_sample_l3896_389698

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  group_size : ℕ
  last_group_sample : ℕ

/-- Theorem stating the relationship between samples from different groups -/
theorem systematic_sampling_relation (s : SystematicSampling)
  (h1 : s.total_students = 180)
  (h2 : s.num_groups = 20)
  (h3 : s.group_size = s.total_students / s.num_groups)
  (h4 : s.last_group_sample = 176) :
  s.last_group_sample = (s.num_groups - 1) * s.group_size + (s.group_size + 5) :=
by sorry

/-- Corollary: The sample from the 3rd group is 23 -/
theorem third_group_sample (s : SystematicSampling)
  (h1 : s.total_students = 180)
  (h2 : s.num_groups = 20)
  (h3 : s.group_size = s.total_students / s.num_groups)
  (h4 : s.last_group_sample = 176) :
  s.group_size + 5 = 23 :=
by sorry

end systematic_sampling_relation_third_group_sample_l3896_389698


namespace triangle_side_length_l3896_389682

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = π/3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l3896_389682


namespace kaleb_books_l3896_389628

theorem kaleb_books (initial_books sold_books new_books : ℕ) : 
  initial_books = 34 → sold_books = 17 → new_books = 7 →
  initial_books - sold_books + new_books = 24 :=
by sorry

end kaleb_books_l3896_389628


namespace wristbands_per_spectator_l3896_389604

theorem wristbands_per_spectator (total_wristbands : ℕ) (total_spectators : ℕ) 
  (h1 : total_wristbands = 290) 
  (h2 : total_spectators = 145) :
  total_wristbands / total_spectators = 2 := by
  sorry

end wristbands_per_spectator_l3896_389604


namespace tangent_at_one_minimum_a_l3896_389695

noncomputable section

def f (x : ℝ) := (1/6) * x^3 + (1/2) * x - x * Real.log x

def domain : Set ℝ := {x | x > 0}

def interval : Set ℝ := {x | 1/Real.exp 1 < x ∧ x < Real.exp 1}

theorem tangent_at_one (x : ℝ) (hx : x ∈ domain) :
  (f x - f 1) = 0 * (x - 1) := by sorry

theorem minimum_a :
  ∃ a : ℝ, (∀ x ∈ interval, f x < a) ∧
  (∀ b : ℝ, (∀ x ∈ interval, f x < b) → a ≤ b) ∧
  a = (1/6) * (Real.exp 1)^3 - (1/2) * (Real.exp 1) := by sorry

end

end tangent_at_one_minimum_a_l3896_389695


namespace intersection_A_B_intersection_A_C_empty_l3896_389697

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 9 < 0}
def B : Set ℝ := {x | 2 ≤ x + 1 ∧ x + 1 ≤ 4}
def C (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem intersection_A_C_empty (m : ℝ) : 
  A ∩ C m = ∅ ↔ m ≤ -4 ∨ m ≥ 3 := by sorry

end intersection_A_B_intersection_A_C_empty_l3896_389697


namespace hyperbola_asymptote_implies_a_equals_5_l3896_389671

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 9 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = (3/5) * x

-- Theorem statement
theorem hyperbola_asymptote_implies_a_equals_5 (a : ℝ) (h1 : a > 0) :
  (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) → a = 5 :=
by sorry

end hyperbola_asymptote_implies_a_equals_5_l3896_389671


namespace tangent_line_equation_l3896_389634

/-- The function f(x) = x³ - 2x + 3 -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point of tangency -/
def P : ℝ × ℝ := (1, 2)

/-- The slope of the tangent line at P -/
def m : ℝ := f' P.1

/-- Theorem: The equation of the tangent line to y = f(x) at P(1, 2) is x - y + 1 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * (x - P.1) + P.2 ↔ x - y + 1 = 0 :=
by sorry

end tangent_line_equation_l3896_389634


namespace f_geq_g_l3896_389689

noncomputable def f (a b x : ℝ) : ℝ := x^2 * Real.exp (x - 1) + a * x^3 + b * x^2

noncomputable def g (x : ℝ) : ℝ := (2/3) * x^3 - x^2

theorem f_geq_g (a b : ℝ) :
  (∀ x : ℝ, (deriv (f a b)) x = 0 ↔ x = -2 ∨ x = 1) →
  ∀ x : ℝ, f (-1/3) (-1) x ≥ g x :=
by sorry

end f_geq_g_l3896_389689


namespace coins_equal_dollar_l3896_389616

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a half-dollar in cents -/
def half_dollar_value : ℕ := 50

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The theorem stating that the sum of the coins equals 100% of a dollar -/
theorem coins_equal_dollar :
  (nickel_value + 2 * dime_value + quarter_value + half_dollar_value) / cents_per_dollar * 100 = 100 := by
  sorry

end coins_equal_dollar_l3896_389616


namespace complex_expression_simplification_l3896_389601

theorem complex_expression_simplification (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (a + b) / ((Real.sqrt a - Real.sqrt b) ^ 2) *
  ((3 * a * b - b * Real.sqrt (a * b) + a * Real.sqrt (a * b) - 3 * b ^ 2) /
   (0.5 * Real.sqrt (0.25 * ((a / b) + (b / a)) ^ 2 - 1)) +
   (4 * a * b * Real.sqrt a + 9 * a * b * Real.sqrt b - 9 * b ^ 2 * Real.sqrt a) /
   (1.5 * Real.sqrt b - 2 * Real.sqrt a)) =
  -2 * b * (a + 3 * Real.sqrt (a * b)) := by sorry

end complex_expression_simplification_l3896_389601


namespace gcd_statements_l3896_389600

theorem gcd_statements : 
  (Nat.gcd 16 12 = 4) ∧ 
  (Nat.gcd 78 36 = 6) ∧ 
  (Nat.gcd 105 315 = 105) ∧
  (Nat.gcd 85 357 ≠ 34) := by
sorry

end gcd_statements_l3896_389600


namespace sandwich_availability_l3896_389655

theorem sandwich_availability (total : ℕ) (sold_out : ℕ) (available : ℕ) 
  (h1 : total = 50) 
  (h2 : sold_out = 33) 
  (h3 : available = total - sold_out) : 
  available = 17 := by
sorry

end sandwich_availability_l3896_389655


namespace movie_theater_open_hours_l3896_389665

/-- A movie theater with multiple screens showing movies throughout the day. -/
structure MovieTheater where
  screens : ℕ
  total_movies : ℕ
  movie_duration : ℕ

/-- Calculate the number of hours a movie theater is open. -/
def theater_open_hours (theater : MovieTheater) : ℕ :=
  (theater.total_movies * theater.movie_duration) / theater.screens

/-- Theorem: A movie theater with 6 screens showing 24 movies, each lasting 2 hours, is open for 8 hours. -/
theorem movie_theater_open_hours :
  let theater := MovieTheater.mk 6 24 2
  theater_open_hours theater = 8 := by
  sorry

end movie_theater_open_hours_l3896_389665


namespace quadratic_is_square_of_binomial_l3896_389621

theorem quadratic_is_square_of_binomial (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 4*x^2 + 14*x + a = (2*x + b)^2) → a = 49/4 := by
sorry

end quadratic_is_square_of_binomial_l3896_389621


namespace quadratic_max_value_l3896_389678

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x

-- State the theorem
theorem quadratic_max_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f m x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 2, f m x = 3) →
  m = -4 ∨ m = 2 * Real.sqrt 3 :=
by sorry

end quadratic_max_value_l3896_389678


namespace M_remainder_mod_55_l3896_389618

def M : ℕ := sorry

theorem M_remainder_mod_55 : M % 55 = 44 := by sorry

end M_remainder_mod_55_l3896_389618


namespace inequality_solution_set_l3896_389657

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem inequality_solution_set (x : ℝ) :
  (f (2*x) + f (x-1) < 0) ↔ (x < 1/3) :=
sorry

end inequality_solution_set_l3896_389657


namespace cucumbers_for_24_apples_l3896_389683

/-- The cost of a single apple -/
def apple_cost : ℝ := 1

/-- The cost of a single banana -/
def banana_cost : ℝ := 2

/-- The cost of a single cucumber -/
def cucumber_cost : ℝ := 1.5

/-- 12 apples cost the same as 6 bananas -/
axiom apple_banana_relation : 12 * apple_cost = 6 * banana_cost

/-- 3 bananas cost the same as 4 cucumbers -/
axiom banana_cucumber_relation : 3 * banana_cost = 4 * cucumber_cost

/-- The number of cucumbers that can be bought for the price of 24 apples is 16 -/
theorem cucumbers_for_24_apples : 
  (24 * apple_cost) / cucumber_cost = 16 := by sorry

end cucumbers_for_24_apples_l3896_389683


namespace higher_rate_fewer_attendees_possible_l3896_389617

/-- Represents a workshop with attendees and total capacity -/
structure Workshop where
  attendees : ℕ
  capacity : ℕ
  attendance_rate : ℚ
  attendance_rate_def : attendance_rate = attendees / capacity

/-- Theorem stating that it's possible for a workshop to have a higher attendance rate
    but fewer attendees than another workshop -/
theorem higher_rate_fewer_attendees_possible :
  ∃ (A B : Workshop), A.attendance_rate > B.attendance_rate ∧ A.attendees < B.attendees := by
  sorry


end higher_rate_fewer_attendees_possible_l3896_389617


namespace log2_derivative_l3896_389619

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) := Real.log x

-- Define the logarithm with base 2
noncomputable def log2 (x : ℝ) := ln x / ln 2

-- State the theorem
theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv log2 x = 1 / (x * ln 2) :=
sorry

end log2_derivative_l3896_389619


namespace unpainted_cubes_in_5x5x5_l3896_389640

/-- Represents a cube composed of smaller unit cubes --/
structure LargeCube where
  side_length : ℕ
  total_cubes : ℕ
  painted_surface : Bool

/-- Calculates the number of unpainted cubes in a large cube --/
def count_unpainted_cubes (c : LargeCube) : ℕ :=
  if c.painted_surface then (c.side_length - 2)^3 else c.total_cubes

/-- Theorem stating that a 5x5x5 cube with painted surface has 27 unpainted cubes --/
theorem unpainted_cubes_in_5x5x5 :
  let c : LargeCube := { side_length := 5, total_cubes := 125, painted_surface := true }
  count_unpainted_cubes c = 27 := by
  sorry

end unpainted_cubes_in_5x5x5_l3896_389640


namespace smallest_resolvable_debt_is_correct_l3896_389606

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 250

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = pig_value * p + goat_value * g

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 50

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, d > 0 ∧ d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
by sorry

end smallest_resolvable_debt_is_correct_l3896_389606


namespace original_selling_price_l3896_389661

theorem original_selling_price (P : ℝ) : 
  (P + 0.1 * P) - ((0.9 * P) + 0.3 * (0.9 * P)) = 70 → 
  P + 0.1 * P = 1100 := by
sorry

end original_selling_price_l3896_389661


namespace evaluate_expression_l3896_389642

theorem evaluate_expression : (2^(2+1) - 4*(2-1)^2)^2 = 16 := by
  sorry

end evaluate_expression_l3896_389642


namespace harkamal_fruit_purchase_cost_l3896_389632

/-- Calculates the discounted price of a fruit purchase -/
def discountedPrice (quantity : ℕ) (pricePerKg : ℚ) (discountPercent : ℚ) : ℚ :=
  quantity * pricePerKg * (1 - discountPercent / 100)

/-- Represents Harkamal's fruit purchases -/
def fruitPurchases : List (ℕ × ℚ × ℚ) := [
  (10, 70, 10),  -- grapes
  (9, 55, 0),    -- mangoes
  (12, 80, 5),   -- apples
  (7, 45, 15),   -- papayas
  (15, 30, 0),   -- oranges
  (5, 25, 0)     -- bananas
]

/-- Calculates the total cost of Harkamal's fruit purchases -/
def totalCost : ℚ :=
  fruitPurchases.foldr (fun (purchase : ℕ × ℚ × ℚ) (acc : ℚ) =>
    acc + discountedPrice purchase.1 purchase.2.1 purchase.2.2
  ) 0

/-- Theorem stating that the total cost of Harkamal's fruit purchases is $2879.75 -/
theorem harkamal_fruit_purchase_cost :
  totalCost = 2879.75 := by sorry

end harkamal_fruit_purchase_cost_l3896_389632


namespace geometric_sequence_sixth_term_l3896_389648

theorem geometric_sequence_sixth_term 
  (a : ℝ) -- first term
  (a₇ : ℝ) -- 7th term
  (h₁ : a = 1024)
  (h₂ : a₇ = 16)
  : ∃ r : ℝ, r > 0 ∧ a * r^6 = a₇ ∧ a * r^5 = 32 :=
sorry

end geometric_sequence_sixth_term_l3896_389648


namespace same_function_fifth_root_power_l3896_389629

theorem same_function_fifth_root_power (x : ℝ) : x = (x^5)^(1/5) := by
  sorry

end same_function_fifth_root_power_l3896_389629


namespace trigonometric_identity_l3896_389676

theorem trigonometric_identity (t : ℝ) : 
  1 + Real.sin (t/2) * Real.sin t - Real.cos (t/2) * (Real.sin t)^2 = 
  2 * (Real.cos (π/4 - t/2))^2 ↔ 
  ∃ k : ℤ, t = k * π := by
sorry

end trigonometric_identity_l3896_389676


namespace line_tangent_to_circle_l3896_389620

/-- The equation of a circle in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

/-- The equation of a line in polar coordinates -/
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Definition of tangency between a line and a circle -/
def is_tangent (circle line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (ρ₀ θ₀ : ℝ), circle ρ₀ θ₀ ∧ line ρ₀ θ₀ ∧
    ∀ (ρ θ : ℝ), circle ρ θ ∧ line ρ θ → (ρ = ρ₀ ∧ θ = θ₀)

theorem line_tangent_to_circle :
  is_tangent circle_equation line_equation :=
sorry

end line_tangent_to_circle_l3896_389620


namespace parking_lot_useable_percentage_l3896_389670

/-- Proves that the percentage of a parking lot useable for parking is 80%, given specific conditions. -/
theorem parking_lot_useable_percentage :
  ∀ (length width : ℝ) (area_per_car : ℝ) (num_cars : ℕ),
    length = 400 →
    width = 500 →
    area_per_car = 10 →
    num_cars = 16000 →
    (((num_cars : ℝ) * area_per_car) / (length * width)) * 100 = 80 := by
  sorry

end parking_lot_useable_percentage_l3896_389670


namespace find_y_value_l3896_389673

theorem find_y_value (x y : ℝ) 
  (h1 : (100 + 200 + 300 + x) / 4 = 250)
  (h2 : (300 + 150 + 100 + x + y) / 5 = 200) : 
  y = 50 := by
sorry

end find_y_value_l3896_389673


namespace smallest_divisible_by_18_and_45_l3896_389688

theorem smallest_divisible_by_18_and_45 : ∃ n : ℕ+, (∀ m : ℕ+, 18 ∣ m ∧ 45 ∣ m → n ≤ m) ∧ 18 ∣ n ∧ 45 ∣ n :=
  sorry

end smallest_divisible_by_18_and_45_l3896_389688


namespace average_pop_percentage_l3896_389652

/-- Calculates the percentage of popped kernels in a bag -/
def popPercentage (popped : ℕ) (total : ℕ) : ℚ :=
  (popped : ℚ) / (total : ℚ) * 100

/-- Theorem: The average percentage of popped kernels across three bags is 82% -/
theorem average_pop_percentage :
  let bag1 := popPercentage 60 75
  let bag2 := popPercentage 42 50
  let bag3 := popPercentage 82 100
  (bag1 + bag2 + bag3) / 3 = 82 := by
  sorry

end average_pop_percentage_l3896_389652


namespace football_group_size_l3896_389611

/-- The proportion of people who like football -/
def like_football_ratio : ℚ := 24 / 60

/-- The proportion of people who play football among those who like it -/
def play_football_ratio : ℚ := 1 / 2

/-- The number of people expected to play football -/
def expected_players : ℕ := 50

/-- The total number of people in the group -/
def total_people : ℕ := 250

theorem football_group_size :
  (↑expected_players : ℚ) = like_football_ratio * play_football_ratio * total_people :=
sorry

end football_group_size_l3896_389611


namespace frog_probability_l3896_389608

/-- Represents the probability of ending on a vertical side from a given position -/
def P (x y : ℝ) : ℝ := sorry

/-- The square's dimensions -/
def squareSize : ℝ := 6

theorem frog_probability :
  /- Starting position -/
  let startX : ℝ := 2
  let startY : ℝ := 2

  /- Conditions -/
  (∀ x y, 0 ≤ x ∧ x ≤ squareSize ∧ 0 ≤ y ∧ y ≤ squareSize →
    P x y = (P (x-1) y + P (x+1) y + P x (y-1) + P x (y+1)) / 4) →
  (∀ y, 0 ≤ y ∧ y ≤ squareSize → P 0 y = 1 ∧ P squareSize y = 1) →
  (∀ x, 0 ≤ x ∧ x ≤ squareSize → P x 0 = 0 ∧ P x squareSize = 0) →
  (∀ x y, P x y = P (squareSize - x) (squareSize - y)) →

  /- Conclusion -/
  P startX startY = 2/3 := by sorry

end frog_probability_l3896_389608


namespace UA_intersect_B_equals_two_three_l3896_389679

def U : Set Int := {-3, -2, -1, 0, 1, 2, 3, 4}

def A : Set Int := {x ∈ U | x * (x^2 - 1) = 0}

def B : Set Int := {x ∈ U | x ≥ 0 ∧ x^2 ≤ 9}

theorem UA_intersect_B_equals_two_three : (U \ A) ∩ B = {2, 3} := by sorry

end UA_intersect_B_equals_two_three_l3896_389679


namespace sqrt_18_times_sqrt_32_l3896_389643

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end sqrt_18_times_sqrt_32_l3896_389643


namespace multiple_of_seven_l3896_389658

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def six_digit_number (d : ℕ) : ℕ := 567800 + d * 10 + 2

theorem multiple_of_seven (d : ℕ) (h : is_single_digit d) : 
  (six_digit_number d) % 7 = 0 ↔ d = 0 ∨ d = 7 := by
sorry

end multiple_of_seven_l3896_389658


namespace polynomial_simplification_l3896_389666

theorem polynomial_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  (x - a)^3 / ((a - b) * (a - c)) + (x - b)^3 / ((b - a) * (b - c)) + (x - c)^3 / ((c - a) * (c - b)) =
  a + b + c - 3 * x := by
  sorry

end polynomial_simplification_l3896_389666


namespace largest_quotient_from_set_l3896_389647

theorem largest_quotient_from_set : ∃ (a b : ℤ), 
  a ∈ ({-32, -5, 1, 2, 12, 15} : Set ℤ) ∧ 
  b ∈ ({-32, -5, 1, 2, 12, 15} : Set ℤ) ∧ 
  b ≠ 0 ∧
  (∀ (x y : ℤ), x ∈ ({-32, -5, 1, 2, 12, 15} : Set ℤ) → 
                y ∈ ({-32, -5, 1, 2, 12, 15} : Set ℤ) → 
                y ≠ 0 → 
                (x : ℚ) / y ≤ (a : ℚ) / b) ∧
  (a : ℚ) / b = 32 := by
  sorry

end largest_quotient_from_set_l3896_389647


namespace arithmetic_series_sum_l3896_389681

variable (k : ℕ)

def first_term : ℕ → ℕ := λ k => 3 * k^2 + 2
def common_difference : ℕ := 2
def num_terms : ℕ → ℕ := λ k => 4 * k + 3

theorem arithmetic_series_sum :
  (λ k : ℕ => (num_terms k) * (2 * first_term k + (num_terms k - 1) * common_difference) / 2) =
  (λ k : ℕ => 12 * k^3 + 28 * k^2 + 28 * k + 12) :=
by sorry

end arithmetic_series_sum_l3896_389681


namespace circle_and_line_properties_l3896_389656

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 2)^2 = 25

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = (5/12) * x + 43/12

-- Theorem statement
theorem circle_and_line_properties :
  -- Circle C passes through (0, 2) and (2, -2)
  circle_C 0 2 ∧ circle_C 2 (-2) ∧
  -- Center of C lies on x - y + 1 = 0
  (∃ t : ℝ, circle_C t (t + 1)) ∧
  -- Line m passes through (1, 4)
  line_m 1 4 ∧
  -- Chord length on C is 6
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    line_m x₁ y₁ ∧ line_m x₂ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) →
  -- The standard equation of C is correct
  (∀ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 25 ↔ circle_C x y) ∧
  -- The slope-intercept equation of m is correct
  (∀ x y : ℝ, y = (5/12) * x + 43/12 ↔ line_m x y) :=
sorry

end circle_and_line_properties_l3896_389656


namespace square_root_equation_l3896_389639

theorem square_root_equation (N : ℝ) : 
  Real.sqrt (0.05 * N) * Real.sqrt 5 = 0.25 → N = 0.25 := by
sorry

end square_root_equation_l3896_389639


namespace hyperbola_foci_distance_l3896_389651

/-- The distance between the foci of a hyperbola defined by xy = 4 is 2√10. -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x * y = 4 → (x - f₁.1)^2 / f₁.1^2 - (y - f₁.2)^2 / f₁.2^2 = 1) ∧
    (∀ (x y : ℝ), x * y = 4 → (x - f₂.1)^2 / f₂.1^2 - (y - f₂.2)^2 / f₂.2^2 = 1) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 10 :=
sorry

end hyperbola_foci_distance_l3896_389651


namespace ab_max_and_reciprocal_sum_min_l3896_389603

/-- Given positive real numbers a and b satisfying a + 4b = 4,
    prove the maximum value of ab and the minimum value of 1/a + 4/b -/
theorem ab_max_and_reciprocal_sum_min (a b : ℝ) 
    (ha : a > 0) (hb : b > 0) (hab : a + 4*b = 4) : 
    (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 4*y = 4 ∧ a*b ≤ x*y) ∧ 
    (∀ (x y : ℝ), x > 0 → y > 0 → x + 4*y = 4 → a*b ≤ x*y) ∧
    (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 4*y = 4 ∧ 1/x + 4/y ≥ 25/4) ∧ 
    (∀ (x y : ℝ), x > 0 → y > 0 → x + 4*y = 4 → 1/x + 4/y ≥ 25/4) := by
  sorry

end ab_max_and_reciprocal_sum_min_l3896_389603


namespace fraction_simplification_l3896_389660

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) :
  (3*x^3 - 2*x^2 - 5*x + 1) / ((x+1)*(x-2)) - (2*x^2 - 7*x + 3) / ((x+1)*(x-2)) =
  (x-1)*(3*x^2 - x + 2) / ((x+1)*(x-2)) := by
  sorry

end fraction_simplification_l3896_389660


namespace car_trip_speed_proof_l3896_389696

/-- Proves that given a trip of 8 hours with an average speed of 34 miles per hour,
    where the first 6 hours are traveled at 30 miles per hour,
    the average speed for the remaining 2 hours is 46 miles per hour. -/
theorem car_trip_speed_proof :
  let total_time : ℝ := 8
  let first_part_time : ℝ := 6
  let first_part_speed : ℝ := 30
  let total_average_speed : ℝ := 34
  let remaining_time : ℝ := total_time - first_part_time
  let total_distance : ℝ := total_time * total_average_speed
  let first_part_distance : ℝ := first_part_time * first_part_speed
  let remaining_distance : ℝ := total_distance - first_part_distance
  let remaining_speed : ℝ := remaining_distance / remaining_time
  remaining_speed = 46 := by sorry

end car_trip_speed_proof_l3896_389696
