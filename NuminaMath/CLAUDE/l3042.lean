import Mathlib

namespace NUMINAMATH_CALUDE_sports_club_membership_l3042_304201

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h_total : total = 27)
  (h_badminton : badminton = 17)
  (h_tennis : tennis = 19)
  (h_both : both = 11) :
  total - (badminton + tennis - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_membership_l3042_304201


namespace NUMINAMATH_CALUDE_sphere_radius_from_depression_l3042_304291

theorem sphere_radius_from_depression (r : ℝ) 
  (depression_depth : ℝ) (depression_diameter : ℝ) : 
  depression_depth = 8 ∧ 
  depression_diameter = 24 ∧ 
  r^2 = (r - depression_depth)^2 + (depression_diameter / 2)^2 → 
  r = 13 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_depression_l3042_304291


namespace NUMINAMATH_CALUDE_a_value_l3042_304281

def A (a : ℝ) : Set ℝ := {1, 3, a}
def B : Set ℝ := {4, 5}

theorem a_value (a : ℝ) (h : A a ∩ B = {4}) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l3042_304281


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3042_304287

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π / 4) = 1 / 7) : 
  Real.tan α = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3042_304287


namespace NUMINAMATH_CALUDE_land_of_computation_base_l3042_304202

/-- Represents a number in base s --/
def BaseS (coeffs : List Nat) (s : Nat) : Nat :=
  coeffs.enum.foldl (fun acc (i, a) => acc + a * s^i) 0

/-- The problem statement --/
theorem land_of_computation_base (s : Nat) : 
  s > 1 → 
  BaseS [0, 5, 5] s + BaseS [0, 2, 4] s = BaseS [0, 0, 1, 1] s → 
  s = 7 := by
sorry

end NUMINAMATH_CALUDE_land_of_computation_base_l3042_304202


namespace NUMINAMATH_CALUDE_room_area_in_square_yards_l3042_304260

/-- Proves that the area of a 15 ft by 10 ft rectangular room is 16.67 square yards -/
theorem room_area_in_square_yards :
  let length : ℝ := 15
  let width : ℝ := 10
  let sq_feet_per_sq_yard : ℝ := 9
  let area_sq_feet : ℝ := length * width
  let area_sq_yards : ℝ := area_sq_feet / sq_feet_per_sq_yard
  area_sq_yards = 16.67 := by sorry

end NUMINAMATH_CALUDE_room_area_in_square_yards_l3042_304260


namespace NUMINAMATH_CALUDE_average_score_calculation_l3042_304235

theorem average_score_calculation (total_students : ℝ) (male_ratio : ℝ) 
  (male_avg_score : ℝ) (female_avg_score : ℝ) 
  (h1 : male_ratio = 0.4)
  (h2 : male_avg_score = 75)
  (h3 : female_avg_score = 80) :
  (male_ratio * male_avg_score + (1 - male_ratio) * female_avg_score) = 78 := by
  sorry

#check average_score_calculation

end NUMINAMATH_CALUDE_average_score_calculation_l3042_304235


namespace NUMINAMATH_CALUDE_tickets_distribution_l3042_304269

theorem tickets_distribution (initial_tickets best_friend_tickets schoolmate_tickets remaining_tickets : ℕ)
  (h1 : initial_tickets = 128)
  (h2 : best_friend_tickets = 7)
  (h3 : schoolmate_tickets = 4)
  (h4 : remaining_tickets = 11)
  (h5 : ∃ (best_friends schoolmates : ℕ),
    best_friends * best_friend_tickets + schoolmates * schoolmate_tickets + remaining_tickets = initial_tickets) :
  ∃ (best_friends schoolmates : ℕ),
    best_friends * best_friend_tickets + schoolmates * schoolmate_tickets + remaining_tickets = initial_tickets ∧
    best_friends + schoolmates = 20 := by
  sorry

end NUMINAMATH_CALUDE_tickets_distribution_l3042_304269


namespace NUMINAMATH_CALUDE_perimeter_of_quarter_circle_bounded_square_l3042_304259

/-- The perimeter of a region bounded by quarter-circle arcs constructed on each side of a square --/
theorem perimeter_of_quarter_circle_bounded_square (s : ℝ) (h : s = 4 / Real.pi) :
  4 * (Real.pi * s / 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_quarter_circle_bounded_square_l3042_304259


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3042_304277

theorem tan_alpha_value (α : Real) 
  (h : (Real.cos (π/4 - α)) / (Real.cos (π/4 + α)) = 1/2) : 
  Real.tan α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3042_304277


namespace NUMINAMATH_CALUDE_car_price_calculation_l3042_304233

/-- Represents the price of a car given loan terms and payments. -/
def car_price (loan_years : ℕ) (interest_rate : ℚ) (down_payment : ℚ) (monthly_payment : ℚ) : ℚ :=
  down_payment + (loan_years * 12 : ℕ) * monthly_payment

/-- Theorem stating the price of the car under given conditions. -/
theorem car_price_calculation :
  car_price 5 (4/100) 5000 250 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_car_price_calculation_l3042_304233


namespace NUMINAMATH_CALUDE_flight_distance_calculation_l3042_304273

/-- Calculates the total flight distance with headwinds and tailwinds -/
def total_flight_distance (spain_russia : ℝ) (spain_germany : ℝ) (germany_france : ℝ) (france_russia : ℝ) 
  (headwind_increase : ℝ) (tailwind_decrease : ℝ) : ℝ :=
  let france_russia_with_headwind := france_russia * (1 + headwind_increase)
  let russia_spain_via_germany := (spain_russia + spain_germany) * (1 - tailwind_decrease)
  france_russia_with_headwind + russia_spain_via_germany

/-- The total flight distance is approximately 14863.98 km -/
theorem flight_distance_calculation :
  let spain_russia : ℝ := 7019
  let spain_germany : ℝ := 1615
  let germany_france : ℝ := 956
  let france_russia : ℝ := 6180
  let headwind_increase : ℝ := 0.05
  let tailwind_decrease : ℝ := 0.03
  abs (total_flight_distance spain_russia spain_germany germany_france france_russia 
    headwind_increase tailwind_decrease - 14863.98) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_flight_distance_calculation_l3042_304273


namespace NUMINAMATH_CALUDE_glasses_in_smaller_box_l3042_304218

theorem glasses_in_smaller_box :
  ∀ x : ℕ,
  (x + 16) / 2 = 15 →
  x = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_glasses_in_smaller_box_l3042_304218


namespace NUMINAMATH_CALUDE_remainder_of_7_350_mod_43_l3042_304241

theorem remainder_of_7_350_mod_43 : 7^350 % 43 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_7_350_mod_43_l3042_304241


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3042_304271

theorem quadratic_equation_roots (x y : ℝ) : 
  x + y = 10 →
  |x - y| = 4 →
  x * y = 21 →
  x^2 - 10*x + 21 = 0 ∧ y^2 - 10*y + 21 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3042_304271


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3042_304286

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Define the point that the line must pass through
def point : ℝ × ℝ := (1, 3)

-- Define the equation of the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- State the theorem
theorem perpendicular_line_through_point :
  (perpendicular_line point.1 point.2) ∧
  (∀ (x y : ℝ), perpendicular_line x y → given_line x y → 
    (y - point.2) = -(x - point.1) * (1 / (2 : ℝ))) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3042_304286


namespace NUMINAMATH_CALUDE_louisa_travel_problem_l3042_304299

/-- Louisa's vacation travel problem -/
theorem louisa_travel_problem (speed : ℝ) (second_day_distance : ℝ) (time_difference : ℝ) :
  speed = 60 →
  second_day_distance = 420 →
  time_difference = 3 →
  ∃ (first_day_distance : ℝ),
    first_day_distance = speed * (second_day_distance / speed - time_difference) ∧
    first_day_distance = 240 :=
by sorry

end NUMINAMATH_CALUDE_louisa_travel_problem_l3042_304299


namespace NUMINAMATH_CALUDE_f_properties_l3042_304210

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem f_properties (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∀ x, f a x = -f a (-x) ↔ a = 1/2) ∧
  (a = 1/2 →
    (∀ y, -1/2 < y ∧ y < 1/2 → ∃ x, f a x = y) ∧
    (∀ x, -1/2 < f a x ∧ f a x < 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3042_304210


namespace NUMINAMATH_CALUDE_vector_problem_l3042_304229

/-- Given a vector a and a unit vector b not parallel to the x-axis such that a · b = √3, prove that b = (1/2, √3/2) -/
theorem vector_problem (a b : ℝ × ℝ) : 
  a = (Real.sqrt 3, 1) →
  ‖b‖ = 1 →
  b.1 ≠ b.2 →
  a.1 * b.1 + a.2 * b.2 = Real.sqrt 3 →
  b = (1/2, Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l3042_304229


namespace NUMINAMATH_CALUDE_carton_height_calculation_l3042_304239

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of items that can fit along one dimension -/
def maxItemsAlongDimension (containerSize itemSize : ℕ) : ℕ :=
  containerSize / itemSize

/-- Calculates the total number of items that can fit on the base of the container -/
def itemsOnBase (containerBase itemBase : Dimensions) : ℕ :=
  (maxItemsAlongDimension containerBase.length itemBase.length) *
  (maxItemsAlongDimension containerBase.width itemBase.width)

/-- Calculates the number of layers of items that can be stacked in the container -/
def numberOfLayers (maxItems itemsPerLayer : ℕ) : ℕ :=
  maxItems / itemsPerLayer

/-- Calculates the height of the container based on the number of layers and item height -/
def containerHeight (layers itemHeight : ℕ) : ℕ :=
  layers * itemHeight

theorem carton_height_calculation (cartonBase : Dimensions) (soapBox : Dimensions) (maxSoapBoxes : ℕ) :
  cartonBase.length = 25 →
  cartonBase.width = 42 →
  soapBox.length = 7 →
  soapBox.width = 12 →
  soapBox.height = 5 →
  maxSoapBoxes = 150 →
  containerHeight (numberOfLayers maxSoapBoxes (itemsOnBase cartonBase soapBox)) soapBox.height = 80 := by
  sorry

#check carton_height_calculation

end NUMINAMATH_CALUDE_carton_height_calculation_l3042_304239


namespace NUMINAMATH_CALUDE_age_ratio_nine_years_ago_l3042_304294

def henry_present_age : ℕ := 29
def jill_present_age : ℕ := 19

theorem age_ratio_nine_years_ago :
  (henry_present_age - 9) / (jill_present_age - 9) = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_nine_years_ago_l3042_304294


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_is_square_l3042_304261

theorem sum_of_fourth_powers_is_square (a b c : ℤ) (h : a + b + c = 0) :
  ∃ p : ℤ, 2 * a^4 + 2 * b^4 + 2 * c^4 = 4 * p^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_is_square_l3042_304261


namespace NUMINAMATH_CALUDE_freshman_count_l3042_304280

theorem freshman_count (f o j s : ℕ) : 
  f * 4 = o * 5 →
  o * 8 = j * 7 →
  j * 7 = s * 9 →
  f + o + j + s = 2158 →
  f = 630 :=
by sorry

end NUMINAMATH_CALUDE_freshman_count_l3042_304280


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3042_304224

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 12 * x + 1 = a * (x - h)^2 + k) → 
  a + h + k = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3042_304224


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3042_304237

/-- Rationalize the denominator of 5 / (3 * ∜7) -/
theorem rationalize_denominator :
  ∃ (P Q R : ℕ),
    (5 : ℚ) / (3 * (7 : ℚ)^(1/4)) = (P : ℚ) * (Q : ℚ)^(1/4) / R ∧
    R > 0 ∧
    ∀ (p : ℕ), Nat.Prime p → Q % p^4 ≠ 0 ∧
    P = 5 ∧
    Q = 343 ∧
    R = 21 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3042_304237


namespace NUMINAMATH_CALUDE_discount_percentage_l3042_304232

theorem discount_percentage 
  (profit_with_discount : ℝ) 
  (profit_without_discount : ℝ) 
  (h1 : profit_with_discount = 0.235) 
  (h2 : profit_without_discount = 0.30) : 
  (profit_without_discount - profit_with_discount) / (1 + profit_without_discount) = 0.05 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l3042_304232


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l3042_304207

theorem quadratic_root_condition (m : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 2 ∧ r₂ < 2 ∧ 
    r₁^2 + (2*m - 1)*r₁ + 4 - 2*m = 0 ∧
    r₂^2 + (2*m - 1)*r₂ + 4 - 2*m = 0) →
  m < -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l3042_304207


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l3042_304240

theorem tan_alpha_plus_pi_third (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3/5)
  (h2 : Real.tan (β - π/3) = 1/4) :
  Real.tan (α + π/3) = 7/23 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l3042_304240


namespace NUMINAMATH_CALUDE_combined_apples_l3042_304288

/-- The number of apples Sara ate -/
def sara_apples : ℕ := 16

/-- The ratio of apples Ali ate compared to Sara -/
def ali_ratio : ℕ := 4

/-- The total number of apples eaten by Ali and Sara -/
def total_apples : ℕ := sara_apples + ali_ratio * sara_apples

theorem combined_apples : total_apples = 80 := by
  sorry

end NUMINAMATH_CALUDE_combined_apples_l3042_304288


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l3042_304251

theorem min_value_of_fraction (x : ℝ) (h : x > 10) :
  x^2 / (x - 10) ≥ 30 ∧ ∃ y > 10, y^2 / (y - 10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l3042_304251


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3042_304265

theorem quadratic_unique_solution (a b : ℝ) : 
  (∃! x, 16 * x^2 + a * x + b = 0) → 
  a^2 = 4 * b → 
  a = 0 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3042_304265


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3042_304290

theorem inequality_and_equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^3 + b^3 + c^3 + d^3 ≥ a^2*b + b^2*c + c^2*d + d^2*a ∧
  (a^3 + b^3 + c^3 + d^3 = a^2*b + b^2*c + c^2*d + d^2*a ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3042_304290


namespace NUMINAMATH_CALUDE_platform_length_l3042_304222

/-- The length of a platform given train parameters -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 150 →
  train_speed_kmph = 75 →
  crossing_time = 24 →
  ∃ (platform_length : ℝ),
    platform_length = 350 ∧
    platform_length = (train_speed_kmph * 1000 / 3600 * crossing_time) - train_length :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3042_304222


namespace NUMINAMATH_CALUDE_piglet_banana_count_l3042_304243

/-- Represents the number of bananas eaten by each character -/
structure BananaCount where
  winnie : ℕ
  owl : ℕ
  rabbit : ℕ
  piglet : ℕ

/-- The conditions of the banana distribution problem -/
def BananaDistribution (bc : BananaCount) : Prop :=
  bc.winnie + bc.owl + bc.rabbit + bc.piglet = 70 ∧
  bc.owl + bc.rabbit = 45 ∧
  bc.winnie > bc.owl ∧
  bc.winnie > bc.rabbit ∧
  bc.winnie > bc.piglet ∧
  bc.winnie ≥ 1 ∧
  bc.owl ≥ 1 ∧
  bc.rabbit ≥ 1 ∧
  bc.piglet ≥ 1

theorem piglet_banana_count (bc : BananaCount) :
  BananaDistribution bc → bc.piglet = 1 := by
  sorry

end NUMINAMATH_CALUDE_piglet_banana_count_l3042_304243


namespace NUMINAMATH_CALUDE_bisection_sqrt2_approximation_l3042_304249

theorem bisection_sqrt2_approximation :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2| ≤ 0.1 := by
  sorry

end NUMINAMATH_CALUDE_bisection_sqrt2_approximation_l3042_304249


namespace NUMINAMATH_CALUDE_parabola_b_value_l3042_304272

/-- A parabola passing through two given points has a specific 'b' value -/
theorem parabola_b_value (b c : ℝ) : 
  ((-1)^2 + b*(-1) + c = -8) → 
  (2^2 + b*2 + c = 10) → 
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3042_304272


namespace NUMINAMATH_CALUDE_product_upper_bound_l3042_304285

theorem product_upper_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b ≤ 4) : a * b ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_product_upper_bound_l3042_304285


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l3042_304247

/-- The maximum area of a rectangle with a perimeter of 16 meters -/
theorem max_area_rectangle_with_fixed_perimeter : 
  ∀ (length width : ℝ), 
  length > 0 → width > 0 → 
  2 * (length + width) = 16 → 
  length * width ≤ 16 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l3042_304247


namespace NUMINAMATH_CALUDE_mollys_age_l3042_304255

/-- Molly's age calculation --/
theorem mollys_age (initial_candles additional_candles : ℕ) :
  initial_candles = 14 → additional_candles = 6 →
  initial_candles + additional_candles = 20 := by
  sorry

end NUMINAMATH_CALUDE_mollys_age_l3042_304255


namespace NUMINAMATH_CALUDE_num_groupings_l3042_304296

/-- The number of ways to distribute n items into 2 non-empty groups -/
def distribute (n : ℕ) : ℕ :=
  2^n - 2

/-- The number of tour guides -/
def num_guides : ℕ := 2

/-- The number of tourists -/
def num_tourists : ℕ := 6

/-- Each guide must have at least one tourist -/
axiom guides_not_empty : distribute num_tourists ≥ 1

/-- Theorem: The number of ways to distribute 6 tourists between 2 guides, 
    with each guide having at least one tourist, is 62 -/
theorem num_groupings : distribute num_tourists = 62 := by
  sorry

end NUMINAMATH_CALUDE_num_groupings_l3042_304296


namespace NUMINAMATH_CALUDE_coefficient_m3n5_in_binomial_expansion_l3042_304238

theorem coefficient_m3n5_in_binomial_expansion :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (3 : ℕ)^(8 - k) * (5 : ℕ)^k) = 56 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_m3n5_in_binomial_expansion_l3042_304238


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l3042_304223

theorem existence_of_special_integers : ∃ (a b c : ℤ), 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧   -- nonzero integers
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧   -- pairwise distinct
  a + b + c = 0 ∧           -- sum is zero
  ∃ (n : ℕ), a^13 + b^13 + c^13 = n^2  -- sum of 13th powers is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l3042_304223


namespace NUMINAMATH_CALUDE_therapy_charges_relation_l3042_304228

/-- A psychologist's charging scheme for therapy sessions. -/
structure TherapyCharges where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  first_hour_premium : firstHourCharge = additionalHourCharge + 30

/-- Calculate the total charge for a given number of therapy hours. -/
def totalCharge (charges : TherapyCharges) (hours : ℕ) : ℕ :=
  charges.firstHourCharge + (hours - 1) * charges.additionalHourCharge

/-- Theorem stating the relationship between charges for 5 hours and 3 hours of therapy. -/
theorem therapy_charges_relation (charges : TherapyCharges) :
  totalCharge charges 5 = 400 → totalCharge charges 3 = 252 := by
  sorry

#check therapy_charges_relation

end NUMINAMATH_CALUDE_therapy_charges_relation_l3042_304228


namespace NUMINAMATH_CALUDE_cube_sum_zero_or_abc_function_l3042_304219

theorem cube_sum_zero_or_abc_function (a b c : ℝ) 
  (nonzero_a : a ≠ 0) (nonzero_b : b ≠ 0) (nonzero_c : c ≠ 0)
  (sum_zero : a + b + c = 0)
  (fourth_sixth_power_eq : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  (a^3 + b^3 + c^3 = 0) ∨ (∃ f : ℝ → ℝ → ℝ → ℝ, a^3 + b^3 + c^3 = f a b c) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_zero_or_abc_function_l3042_304219


namespace NUMINAMATH_CALUDE_frustum_smaller_base_radius_l3042_304244

/-- A frustum with the given properties has a smaller base radius of 7 -/
theorem frustum_smaller_base_radius (r : ℝ) : 
  r > 0 → -- r is positive (implicit in the problem context)
  (2 * π * r) * 3 = 2 * π * (3 * r) → -- one circumference is three times the other
  3 = 3 → -- slant height is 3
  π * (r + 3 * r) * 3 = 84 * π → -- lateral surface area formula
  r = 7 := by
sorry


end NUMINAMATH_CALUDE_frustum_smaller_base_radius_l3042_304244


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3042_304257

theorem inequality_solution_set :
  {x : ℝ | 4 + 2*x > -6} = {x : ℝ | x > -5} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3042_304257


namespace NUMINAMATH_CALUDE_range_of_slopes_tangent_line_equation_l3042_304282

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

-- Theorem for the range of slopes
theorem range_of_slopes :
  ∀ x ∈ Set.Icc (-2) 1, -3 ≤ f' x ∧ f' x ≤ 9 :=
sorry

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ a b : ℝ,
    f' a = -3 ∧
    f a = b ∧
    (∀ x y : ℝ, line_l x y → (3*x + y + 6 = 0 → x = a ∧ y = b)) :=
sorry

end NUMINAMATH_CALUDE_range_of_slopes_tangent_line_equation_l3042_304282


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3042_304275

theorem rectangle_ratio (w : ℝ) (h1 : w > 0) (h2 : 2 * w + 2 * 12 = 36) :
  w / 12 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3042_304275


namespace NUMINAMATH_CALUDE_train_pass_bridge_time_l3042_304258

/-- Time for a train to pass a bridge -/
theorem train_pass_bridge_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 860)
  (h2 : train_speed_kmh = 85)
  (h3 : bridge_length = 450) :
  ∃ (t : ℝ), abs (t - 55.52) < 0.01 ∧ 
  t = (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) :=
sorry

end NUMINAMATH_CALUDE_train_pass_bridge_time_l3042_304258


namespace NUMINAMATH_CALUDE_f_2017_equals_one_l3042_304289

theorem f_2017_equals_one (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ θ, f (Real.cos θ) = Real.cos (2 * θ)) :
  f 2017 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2017_equals_one_l3042_304289


namespace NUMINAMATH_CALUDE_stratified_sampling_total_students_l3042_304256

theorem stratified_sampling_total_students 
  (total_sample : ℕ) 
  (grade_10_sample : ℕ) 
  (grade_11_sample : ℕ) 
  (grade_12_students : ℕ) 
  (h1 : total_sample = 100)
  (h2 : grade_10_sample = 24)
  (h3 : grade_11_sample = 26)
  (h4 : grade_12_students = 600)
  (h5 : grade_12_students * total_sample = 
        (total_sample - grade_10_sample - grade_11_sample) * total_students) : 
  total_students = 1200 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_total_students_l3042_304256


namespace NUMINAMATH_CALUDE_product_purely_imaginary_l3042_304234

theorem product_purely_imaginary (x : ℝ) :
  (∃ b : ℝ, (x + 2*Complex.I) * ((x + 1) + 2*Complex.I) * ((x + 2) + 2*Complex.I) * ((x + 3) + 2*Complex.I) = b * Complex.I) ↔
  x = -2 := by
sorry

end NUMINAMATH_CALUDE_product_purely_imaginary_l3042_304234


namespace NUMINAMATH_CALUDE_ben_age_l3042_304209

def Ages : List ℕ := [6, 8, 10, 12, 14]

def ParkPair (a b : ℕ) : Prop := a + b = 18 ∧ a ∈ Ages ∧ b ∈ Ages ∧ a ≠ b

def LibraryPair (a b : ℕ) : Prop := a + b < 20 ∧ a ∈ Ages ∧ b ∈ Ages ∧ a ≠ b

def RemainingAges (park1 park2 lib1 lib2 : ℕ) : List ℕ :=
  Ages.filter (λ x => x ∉ [park1, park2, lib1, lib2])

theorem ben_age :
  ∀ park1 park2 lib1 lib2 youngest_home,
    ParkPair park1 park2 →
    LibraryPair lib1 lib2 →
    youngest_home = (RemainingAges park1 park2 lib1 lib2).minimum →
    10 ∈ RemainingAges park1 park2 lib1 lib2 →
    10 ≠ youngest_home →
    10 = (RemainingAges park1 park2 lib1 lib2).maximum :=
by sorry

end NUMINAMATH_CALUDE_ben_age_l3042_304209


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_minus_sqrt_eight_equals_sqrt_two_l3042_304225

theorem sqrt_six_times_sqrt_three_minus_sqrt_eight_equals_sqrt_two :
  Real.sqrt 6 * Real.sqrt 3 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_minus_sqrt_eight_equals_sqrt_two_l3042_304225


namespace NUMINAMATH_CALUDE_equation_solution_l3042_304250

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ (1 / (x - 1) + 1 = 3 / (2 * x - 2)) ∧ x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3042_304250


namespace NUMINAMATH_CALUDE_cone_trajectory_length_l3042_304266

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cone -/
structure Cone where
  base_side_length : ℝ
  apex : Point3D
  base_center : Point3D

/-- The theorem statement -/
theorem cone_trajectory_length 
  (c : Cone) 
  (h_base_side : c.base_side_length = 2) 
  (M : Point3D) 
  (h_M_midpoint : M = Point3D.mk 0 0 ((c.apex.z - c.base_center.z) / 2)) 
  (A : Point3D) 
  (h_A_on_base : A.z = c.base_center.z ∧ (A.x - c.base_center.x)^2 + (A.y - c.base_center.y)^2 = 1) 
  (P : Point3D → Prop) 
  (h_P_on_base : ∀ p, P p → p.z = c.base_center.z ∧ (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 ≤ 1) 
  (h_AM_perp_MP : ∀ p, P p → (M.x - A.x) * (p.x - M.x) + (M.y - A.y) * (p.y - M.y) + (M.z - A.z) * (p.z - M.z) = 0) :
  (∃ l : ℝ, l = Real.sqrt 7 / 2 ∧ 
    ∀ ε > 0, ∃ δ > 0, ∀ p q, P p → P q → abs (p.x - q.x) < δ ∧ abs (p.y - q.y) < δ → 
      abs (Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) - l) < ε) :=
by sorry

end NUMINAMATH_CALUDE_cone_trajectory_length_l3042_304266


namespace NUMINAMATH_CALUDE_trig_expression_eval_l3042_304263

/-- Proves that the given trigonometric expression evaluates to -4√3 --/
theorem trig_expression_eval :
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) /
  (4 * (Real.cos (12 * π / 180))^2 * Real.sin (12 * π / 180) - 2 * Real.sin (12 * π / 180)) =
  -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_eval_l3042_304263


namespace NUMINAMATH_CALUDE_bridge_length_l3042_304214

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed = 20 →
  crossing_time = 12.099 →
  (train_speed * crossing_time) - train_length = 131.98 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3042_304214


namespace NUMINAMATH_CALUDE_seashells_given_theorem_l3042_304264

/-- The number of seashells Tim gave to Sara -/
def seashells_given_to_sara (initial_seashells final_seashells : ℕ) : ℕ :=
  initial_seashells - final_seashells

/-- Theorem stating that the number of seashells given to Sara is the difference between
    the initial and final counts of seashells Tim has -/
theorem seashells_given_theorem (initial_seashells final_seashells : ℕ) 
    (h : initial_seashells ≥ final_seashells) :
  seashells_given_to_sara initial_seashells final_seashells = initial_seashells - final_seashells :=
by
  sorry

#eval seashells_given_to_sara 679 507

end NUMINAMATH_CALUDE_seashells_given_theorem_l3042_304264


namespace NUMINAMATH_CALUDE_product_twice_prime_p_squared_minus_2q_prime_p_plus_2q_prime_l3042_304246

/-- Given that p and q are primes and x^2 - px + 2q = 0 has integral roots which are consecutive primes -/
def consecutive_prime_roots (p q : ℕ) : Prop :=
  ∃ (r s : ℕ), Prime r ∧ Prime s ∧ s = r + 1 ∧ 
  r * s = 2 * q ∧ r + s = p

/-- The product of the roots is twice a prime -/
theorem product_twice_prime {p q : ℕ} (h : consecutive_prime_roots p q) : 
  ∃ (k : ℕ), Prime k ∧ 2 * q = 2 * k :=
sorry

/-- p^2 - 2q is prime -/
theorem p_squared_minus_2q_prime {p q : ℕ} (h : consecutive_prime_roots p q) :
  Prime (p^2 - 2*q) :=
sorry

/-- p + 2q is prime -/
theorem p_plus_2q_prime {p q : ℕ} (h : consecutive_prime_roots p q) :
  Prime (p + 2*q) :=
sorry

end NUMINAMATH_CALUDE_product_twice_prime_p_squared_minus_2q_prime_p_plus_2q_prime_l3042_304246


namespace NUMINAMATH_CALUDE_larger_number_is_nine_l3042_304292

theorem larger_number_is_nine (a b : ℕ+) (h1 : a - b = 3) (h2 : a^2 + b^2 = 117) : a = 9 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_nine_l3042_304292


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3042_304278

theorem expression_simplification_and_evaluation (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4*a + 4) / (a + 1)) = (2 + a) / (2 - a) ∧
  (2 + 1) / (2 - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3042_304278


namespace NUMINAMATH_CALUDE_sin_20_cos_10_minus_cos_160_sin_10_l3042_304226

theorem sin_20_cos_10_minus_cos_160_sin_10 :
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) -
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_20_cos_10_minus_cos_160_sin_10_l3042_304226


namespace NUMINAMATH_CALUDE_power_of_power_l3042_304297

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3042_304297


namespace NUMINAMATH_CALUDE_garage_sale_items_l3042_304253

theorem garage_sale_items (prices : Finset ℕ) (radio_price : ℕ) : 
  prices.card > 0 → 
  radio_price ∈ prices → 
  (prices.filter (λ p => p > radio_price)).card = 16 → 
  (prices.filter (λ p => p < radio_price)).card = 23 → 
  prices.card = 40 := by
sorry

end NUMINAMATH_CALUDE_garage_sale_items_l3042_304253


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l3042_304217

theorem quadratic_equation_problem : 
  (∀ m : ℝ, ∃ x : ℝ, x^2 - m*x - 1 = 0) ∧ 
  (∃ x₀ : ℕ, x₀^2 - 2*x₀ - 1 ≤ 0) → 
  ¬((∀ m : ℝ, ∃ x : ℝ, x^2 - m*x - 1 = 0) ∧ 
    ¬(∃ x₀ : ℕ, x₀^2 - 2*x₀ - 1 ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l3042_304217


namespace NUMINAMATH_CALUDE_kendra_spelling_goals_l3042_304212

-- Define constants
def words_per_week : ℕ := 12
def first_goal : ℕ := 60
def second_goal : ℕ := 100
def reward_threshold : ℕ := 20
def words_learned : ℕ := 36
def weeks_to_birthday : ℕ := 3
def weeks_to_competition : ℕ := 6

-- Define the theorem
theorem kendra_spelling_goals (target : ℕ) :
  (target ≥ reward_threshold) ∧
  (target * weeks_to_birthday + words_learned ≥ first_goal) ∧
  (target * weeks_to_competition + words_learned ≥ second_goal) ↔
  target = reward_threshold :=
by sorry

end NUMINAMATH_CALUDE_kendra_spelling_goals_l3042_304212


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l3042_304227

def pebbles_on_day (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3 * (n - 1) + 2

def total_pebbles (days : ℕ) : ℕ :=
  (List.range days).map pebbles_on_day |>.sum

theorem murtha_pebble_collection :
  total_pebbles 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l3042_304227


namespace NUMINAMATH_CALUDE_zelda_success_probability_l3042_304220

theorem zelda_success_probability 
  (p_xavier : ℝ) 
  (p_yvonne : ℝ) 
  (p_xy_not_z : ℝ) 
  (h1 : p_xavier = 1/3) 
  (h2 : p_yvonne = 1/2) 
  (h3 : p_xy_not_z = 0.0625) : 
  ∃ p_zelda : ℝ, p_zelda = 0.625 ∧ p_xavier * p_yvonne * (1 - p_zelda) = p_xy_not_z :=
by sorry

end NUMINAMATH_CALUDE_zelda_success_probability_l3042_304220


namespace NUMINAMATH_CALUDE_f_and_g_are_even_and_increasing_l3042_304215

-- Define the functions
def f (x : ℝ) : ℝ := |2 * x|
def g (x : ℝ) : ℝ := 2 * x^2 + 3

-- Define evenness
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

-- Define monotonically increasing on an interval
def is_monotone_increasing_on (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → h x ≤ h y

-- Theorem statement
theorem f_and_g_are_even_and_increasing :
  (is_even f ∧ is_monotone_increasing_on f 0 1) ∧
  (is_even g ∧ is_monotone_increasing_on g 0 1) :=
sorry

end NUMINAMATH_CALUDE_f_and_g_are_even_and_increasing_l3042_304215


namespace NUMINAMATH_CALUDE_nine_times_s_on_half_l3042_304248

def s (θ : ℚ) : ℚ := 1 / (2 - θ)

theorem nine_times_s_on_half : s (s (s (s (s (s (s (s (s (1/2)))))))))  = 13/15 := by
  sorry

end NUMINAMATH_CALUDE_nine_times_s_on_half_l3042_304248


namespace NUMINAMATH_CALUDE_namjoon_rank_l3042_304268

theorem namjoon_rank (total_participants : ℕ) (better_than : ℕ) (namjoon_rank : ℕ) :
  total_participants = 13 →
  better_than = 4 →
  namjoon_rank = total_participants - better_than →
  namjoon_rank = 9 :=
by sorry

end NUMINAMATH_CALUDE_namjoon_rank_l3042_304268


namespace NUMINAMATH_CALUDE_gravel_path_cost_l3042_304295

/-- Calculate the cost of gravelling a path around a rectangular plot -/
theorem gravel_path_cost 
  (plot_length : ℝ) 
  (plot_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm : ℝ) : 
  plot_length = 110 →
  plot_width = 65 →
  path_width = 2.5 →
  cost_per_sqm = 0.7 →
  ((plot_length + 2 * path_width) * (plot_width + 2 * path_width) - plot_length * plot_width) * cost_per_sqm = 630 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l3042_304295


namespace NUMINAMATH_CALUDE_oil_mixture_volume_constant_oil_problem_solution_l3042_304270

/-- Represents the properties of an oil mixture -/
structure OilMixture where
  V_hot : ℝ  -- Volume of hot oil
  V_cold : ℝ  -- Volume of cold oil
  T_hot : ℝ  -- Temperature of hot oil
  T_cold : ℝ  -- Temperature of cold oil
  beta : ℝ  -- Coefficient of thermal expansion

/-- Calculates the final volume of an oil mixture at thermal equilibrium -/
def final_volume (mix : OilMixture) : ℝ :=
  mix.V_hot + mix.V_cold

/-- Theorem stating that the final volume of the oil mixture at thermal equilibrium
    is equal to the sum of the initial volumes -/
theorem oil_mixture_volume_constant (mix : OilMixture) :
  final_volume mix = mix.V_hot + mix.V_cold :=
by sorry

/-- Specific instance of the oil mixture problem -/
def oil_problem : OilMixture :=
  { V_hot := 2
  , V_cold := 1
  , T_hot := 100
  , T_cold := 20
  , beta := 2e-3
  }

/-- The final volume of the specific oil mixture problem is 3 liters -/
theorem oil_problem_solution :
  final_volume oil_problem = 3 :=
by sorry

end NUMINAMATH_CALUDE_oil_mixture_volume_constant_oil_problem_solution_l3042_304270


namespace NUMINAMATH_CALUDE_doctors_visit_cost_is_250_l3042_304200

/-- Calculates the cost of a doctor's visit given the following conditions:
  * Number of vaccines needed
  * Cost per vaccine
  * Insurance coverage percentage
  * Cost of the trip
  * Total amount paid by Tom
-/
def doctors_visit_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (insurance_coverage : ℚ) 
                       (trip_cost : ℚ) (total_paid : ℚ) : ℚ :=
  let total_vaccine_cost := num_vaccines * vaccine_cost
  let medical_bills := total_vaccine_cost + (total_paid - trip_cost) / (1 - insurance_coverage)
  medical_bills - total_vaccine_cost

/-- Proves that the cost of the doctor's visit is $250 given the specified conditions -/
theorem doctors_visit_cost_is_250 : 
  doctors_visit_cost 10 45 0.8 1200 1340 = 250 := by
  sorry

end NUMINAMATH_CALUDE_doctors_visit_cost_is_250_l3042_304200


namespace NUMINAMATH_CALUDE_cannot_determine_jake_peaches_l3042_304242

def steven_peaches : ℕ := 9
def steven_apples : ℕ := 8

structure Jake where
  peaches : ℕ
  apples : ℕ
  fewer_peaches : peaches < steven_peaches
  more_apples : apples = steven_apples + 3

theorem cannot_determine_jake_peaches : ∀ (jake : Jake), ∃ (jake' : Jake), jake.peaches ≠ jake'.peaches := by
  sorry

end NUMINAMATH_CALUDE_cannot_determine_jake_peaches_l3042_304242


namespace NUMINAMATH_CALUDE_f_inequality_l3042_304205

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 1/x + 2 * Real.sin x

theorem f_inequality (x : ℝ) (hx : x > 0) :
  f (1 - x) > f x ↔ 0 < x ∧ x < 1/2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l3042_304205


namespace NUMINAMATH_CALUDE_bill_face_value_l3042_304213

/-- Calculates the face value of a bill given the true discount, interest rate, and time period. -/
def calculate_face_value (true_discount : ℚ) (interest_rate : ℚ) (time_months : ℚ) : ℚ :=
  (true_discount * 100) / (interest_rate * (time_months / 12))

/-- Proves that the face value of the bill is 1575 given the specified conditions. -/
theorem bill_face_value :
  let true_discount : ℚ := 189
  let interest_rate : ℚ := 16
  let time_months : ℚ := 9
  calculate_face_value true_discount interest_rate time_months = 1575 := by
  sorry

#eval calculate_face_value 189 16 9

end NUMINAMATH_CALUDE_bill_face_value_l3042_304213


namespace NUMINAMATH_CALUDE_snack_machine_cost_l3042_304284

/-- The total cost for students buying candy bars and chips -/
def total_cost (num_students : ℕ) (candy_price chip_price : ℚ) (candy_per_student chip_per_student : ℕ) : ℚ :=
  num_students * (candy_price * candy_per_student + chip_price * chip_per_student)

/-- Theorem: The total cost for 5 students to each get 1 candy bar at $2 and 2 bags of chips at $0.50 per bag is $15 -/
theorem snack_machine_cost : total_cost 5 2 (1/2) 1 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_snack_machine_cost_l3042_304284


namespace NUMINAMATH_CALUDE_no_solution_exists_l3042_304298

theorem no_solution_exists : ¬ ∃ x : ℝ, 2 < 2 * x ∧ 2 * x < 3 ∧ 1 < 4 * x ∧ 4 * x < 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3042_304298


namespace NUMINAMATH_CALUDE_lee_earnings_theorem_l3042_304230

/-- Represents the lawn care services and their charges -/
structure LawnCareService where
  mowing : Nat
  trimming : Nat
  weedRemoval : Nat
  leafBlowing : Nat
  fertilizing : Nat

/-- Represents the number of services provided -/
structure ServicesProvided where
  mowing : Nat
  trimming : Nat
  weedRemoval : Nat
  leafBlowing : Nat
  fertilizing : Nat

/-- Represents the tips received for each service -/
structure TipsReceived where
  mowing : List Nat
  trimming : List Nat
  weedRemoval : List Nat
  leafBlowing : List Nat

/-- Calculates the total earnings from services and tips -/
def calculateTotalEarnings (charges : LawnCareService) (services : ServicesProvided) (tips : TipsReceived) : Nat :=
  let serviceEarnings := 
    charges.mowing * services.mowing +
    charges.trimming * services.trimming +
    charges.weedRemoval * services.weedRemoval +
    charges.leafBlowing * services.leafBlowing +
    charges.fertilizing * services.fertilizing
  let tipEarnings :=
    tips.mowing.sum + tips.trimming.sum + tips.weedRemoval.sum + tips.leafBlowing.sum
  serviceEarnings + tipEarnings

/-- Theorem stating that Lee's total earnings are $923 -/
theorem lee_earnings_theorem (charges : LawnCareService) (services : ServicesProvided) (tips : TipsReceived)
    (h1 : charges = { mowing := 33, trimming := 15, weedRemoval := 10, leafBlowing := 20, fertilizing := 25 })
    (h2 : services = { mowing := 16, trimming := 8, weedRemoval := 5, leafBlowing := 4, fertilizing := 3 })
    (h3 : tips = { mowing := [10, 10, 12, 15], trimming := [5, 7], weedRemoval := [5], leafBlowing := [6] }) :
    calculateTotalEarnings charges services tips = 923 := by
  sorry


end NUMINAMATH_CALUDE_lee_earnings_theorem_l3042_304230


namespace NUMINAMATH_CALUDE_four_solutions_to_simultaneous_equations_l3042_304236

theorem four_solutions_to_simultaneous_equations :
  ∃! (s : Finset (ℝ × ℝ)), (∀ (p : ℝ × ℝ), p ∈ s ↔ p.1^2 - p.2 = 2022 ∧ p.2^2 - p.1 = 2022) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_solutions_to_simultaneous_equations_l3042_304236


namespace NUMINAMATH_CALUDE_paint_cost_most_cost_effective_l3042_304203

/-- Represents the payment options for the house painting job -/
inductive PaymentOption
  | WorkerDay
  | PaintCost
  | PaintedArea
  | HourlyRate

/-- Calculates the cost of a payment option given the job parameters -/
def calculate_cost (option : PaymentOption) (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) 
  (paint_cost : ℕ) (painted_area : ℕ) : ℕ :=
  match option with
  | PaymentOption.WorkerDay => workers * days * 30
  | PaymentOption.PaintCost => (paint_cost * 30) / 100
  | PaymentOption.PaintedArea => painted_area * 12
  | PaymentOption.HourlyRate => workers * hours_per_day * days * 4

/-- Theorem stating that the PaintCost option is the most cost-effective -/
theorem paint_cost_most_cost_effective (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) 
  (paint_cost : ℕ) (painted_area : ℕ) 
  (h1 : workers = 5)
  (h2 : hours_per_day = 8)
  (h3 : days = 10)
  (h4 : paint_cost = 4800)
  (h5 : painted_area = 150) :
  ∀ option, option ≠ PaymentOption.PaintCost → 
    calculate_cost PaymentOption.PaintCost workers hours_per_day days paint_cost painted_area ≤ 
    calculate_cost option workers hours_per_day days paint_cost painted_area :=
by sorry

end NUMINAMATH_CALUDE_paint_cost_most_cost_effective_l3042_304203


namespace NUMINAMATH_CALUDE_min_dot_product_l3042_304279

/-- Ellipse C with foci at (0,-√3) and (0,√3) passing through (√3/2, 1) -/
def ellipse_C (x y : ℝ) : Prop :=
  y^2 / 4 + x^2 = 1

/-- Parabola E with vertex at (0,0) and focus at (1,0) -/
def parabola_E (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- Point on parabola E -/
def point_on_E (x y : ℝ) : Prop :=
  parabola_E x y

/-- Line through focus (1,0) with slope k -/
def line_through_focus (k x y : ℝ) : Prop :=
  y = k * (x - 1)

/-- Perpendicular line through focus (1,0) with slope -1/k -/
def perp_line_through_focus (k x y : ℝ) : Prop :=
  y = -1/k * (x - 1)

/-- Theorem: Minimum value of AG · HB is 16 -/
theorem min_dot_product :
  ∃ (min : ℝ),
    (∀ (k x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
      point_on_E x₁ y₁ ∧ point_on_E x₂ y₂ ∧ point_on_E x₃ y₃ ∧ point_on_E x₄ y₄ ∧
      line_through_focus k x₁ y₁ ∧ line_through_focus k x₂ y₂ ∧
      perp_line_through_focus k x₃ y₃ ∧ perp_line_through_focus k x₄ y₄ →
      ((x₁ - x₃) * (x₄ - x₂) + (y₁ - y₃) * (y₄ - y₂) ≥ min)) ∧
    min = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_dot_product_l3042_304279


namespace NUMINAMATH_CALUDE_smallest_vector_norm_l3042_304221

/-- Given a vector v such that ||v + (4, 2)|| = 10, 
    the smallest possible value of ||v|| is 10 - 2√5 -/
theorem smallest_vector_norm (v : ℝ × ℝ) 
    (h : ‖v + (4, 2)‖ = 10) : 
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ 
    ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖u‖ ≥ ‖w‖ := by
  sorry


end NUMINAMATH_CALUDE_smallest_vector_norm_l3042_304221


namespace NUMINAMATH_CALUDE_sequence_periodicity_l3042_304245

theorem sequence_periodicity (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 2) = |a (n + 5)| - a (n + 4)) : 
  ∃ N : ℕ, ∀ n ≥ N, a n = a (n + 9) :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l3042_304245


namespace NUMINAMATH_CALUDE_a_squared_gt_one_neither_sufficient_nor_necessary_for_one_over_a_gt_zero_l3042_304262

theorem a_squared_gt_one_neither_sufficient_nor_necessary_for_one_over_a_gt_zero :
  ¬(∀ a : ℝ, a^2 > 1 → 1/a > 0) ∧ ¬(∀ a : ℝ, 1/a > 0 → a^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_a_squared_gt_one_neither_sufficient_nor_necessary_for_one_over_a_gt_zero_l3042_304262


namespace NUMINAMATH_CALUDE_parabola_line_intersection_slope_range_l3042_304204

theorem parabola_line_intersection_slope_range :
  ∀ k : ℝ,
  (∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.2)^2 = 4 * A.1 ∧
    (B.2)^2 = 4 * B.1 ∧
    A.2 = k * (A.1 + 2) ∧
    B.2 = k * (B.1 + 2)) ↔
  (k ∈ Set.Ioo (- Real.sqrt 2 / 2) 0 ∪ Set.Ioo 0 (Real.sqrt 2 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_slope_range_l3042_304204


namespace NUMINAMATH_CALUDE_pet_store_parrot_count_l3042_304231

theorem pet_store_parrot_count (total_birds : ℕ) (num_cages : ℕ) (parakeets_per_cage : ℕ) 
  (h1 : total_birds = 48)
  (h2 : num_cages = 6)
  (h3 : parakeets_per_cage = 2) :
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 6 := by
  sorry

#check pet_store_parrot_count

end NUMINAMATH_CALUDE_pet_store_parrot_count_l3042_304231


namespace NUMINAMATH_CALUDE_simplify_expression_l3042_304254

theorem simplify_expression (a b c : ℝ) (h1 : 1 - a * b ≠ 0) (h2 : 1 + c * a ≠ 0) :
  ((a + b) / (1 - a * b) + (c - a) / (1 + c * a)) / (1 - ((a + b) / (1 - a * b) * (c - a) / (1 + c * a))) =
  (b + c) / (1 - b * c) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3042_304254


namespace NUMINAMATH_CALUDE_base9_multiplication_l3042_304216

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (n : ℕ) : ℕ :=
  let digits := n.digits 9
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * 9^i) 0

/-- Converts a base-10 number to base-9 --/
def base10ToBase9 (n : ℕ) : ℕ :=
  n.digits 9 |> List.reverse |> List.foldl (fun acc d => acc * 10 + d) 0

/-- Multiplication in base-9 --/
def multBase9 (a b : ℕ) : ℕ :=
  base10ToBase9 (base9ToBase10 a * base9ToBase10 b)

theorem base9_multiplication :
  multBase9 327 6 = 2226 := by sorry

end NUMINAMATH_CALUDE_base9_multiplication_l3042_304216


namespace NUMINAMATH_CALUDE_swimming_championship_races_swimming_championship_proof_l3042_304274

/-- Calculate the number of races needed to determine a champion in a swimming competition. -/
theorem swimming_championship_races (total_swimmers : ℕ) 
  (swimmers_per_race : ℕ) (advancing_swimmers : ℕ) : ℕ :=
  let eliminated_per_race := swimmers_per_race - advancing_swimmers
  let total_eliminations := total_swimmers - 1
  ⌈(total_eliminations : ℚ) / eliminated_per_race⌉.toNat

/-- Prove that 53 races are required for 300 swimmers with 8 per race and 2 advancing. -/
theorem swimming_championship_proof : 
  swimming_championship_races 300 8 2 = 53 := by
  sorry

end NUMINAMATH_CALUDE_swimming_championship_races_swimming_championship_proof_l3042_304274


namespace NUMINAMATH_CALUDE_triangle_identities_l3042_304211

theorem triangle_identities (a b c α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum_angles : α + β + γ = π)
  (h_law_of_sines : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ) :
  (a + b) / c = Real.cos ((α - β) / 2) / Real.sin (γ / 2) ∧
  (a - b) / c = Real.sin ((α - β) / 2) / Real.cos (γ / 2) := by
sorry

end NUMINAMATH_CALUDE_triangle_identities_l3042_304211


namespace NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l3042_304293

theorem pythagorean_triple_divisibility (x y z : ℤ) : 
  x^2 + y^2 = z^2 → ∃ k : ℤ, x * y = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l3042_304293


namespace NUMINAMATH_CALUDE_union_and_intersection_of_A_and_B_l3042_304252

variable (a : ℝ)

def A : Set ℝ := {x | (x - 3) * (x - a) = 0}
def B : Set ℝ := {x | (x - 4) * (x - 1) = 0}

theorem union_and_intersection_of_A_and_B :
  (a = 3 → (A a ∪ B = {1, 3, 4} ∧ A a ∩ B = ∅)) ∧
  (a = 1 → (A a ∪ B = {1, 3, 4} ∧ A a ∩ B = {1})) ∧
  (a = 4 → (A a ∪ B = {1, 3, 4} ∧ A a ∩ B = {4})) ∧
  (a ≠ 1 ∧ a ≠ 3 ∧ a ≠ 4 → (A a ∪ B = {1, 3, 4, a} ∧ A a ∩ B = ∅)) :=
by sorry

end NUMINAMATH_CALUDE_union_and_intersection_of_A_and_B_l3042_304252


namespace NUMINAMATH_CALUDE_total_strings_needed_johns_total_strings_l3042_304206

theorem total_strings_needed (num_basses : ℕ) (strings_per_bass : ℕ) 
  (strings_per_guitar : ℕ) (strings_per_8string_guitar : ℕ) : ℕ :=
  let num_guitars := 2 * num_basses
  let num_8string_guitars := num_guitars - 3
  let bass_strings := num_basses * strings_per_bass
  let guitar_strings := num_guitars * strings_per_guitar
  let eight_string_guitar_strings := num_8string_guitars * strings_per_8string_guitar
  bass_strings + guitar_strings + eight_string_guitar_strings

theorem johns_total_strings : 
  total_strings_needed 3 4 6 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_strings_needed_johns_total_strings_l3042_304206


namespace NUMINAMATH_CALUDE_slower_train_speed_calculation_l3042_304208

/-- The speed of the faster train in kilometers per hour -/
def faster_train_speed : ℝ := 162

/-- The length of the faster train in meters -/
def faster_train_length : ℝ := 1320

/-- The time taken by the faster train to cross a man in the slower train, in seconds -/
def crossing_time : ℝ := 33

/-- The speed of the slower train in kilometers per hour -/
def slower_train_speed : ℝ := 18

theorem slower_train_speed_calculation :
  let relative_speed := (faster_train_speed - slower_train_speed) * 1000 / 3600
  faster_train_length = relative_speed * crossing_time →
  slower_train_speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_slower_train_speed_calculation_l3042_304208


namespace NUMINAMATH_CALUDE_exchange_cookies_to_bagels_l3042_304276

/-- Represents the exchange rate between gingerbread cookies and drying rings -/
def cookie_to_ring : ℚ := 6

/-- Represents the exchange rate between drying rings and bagels -/
def ring_to_bagel : ℚ := 4/9

/-- Represents the number of gingerbread cookies we want to exchange -/
def cookies : ℚ := 3

/-- Theorem stating that 3 gingerbread cookies can be exchanged for 8 bagels -/
theorem exchange_cookies_to_bagels :
  cookies * cookie_to_ring * ring_to_bagel = 8 := by
  sorry

end NUMINAMATH_CALUDE_exchange_cookies_to_bagels_l3042_304276


namespace NUMINAMATH_CALUDE_circle_radius_l3042_304283

theorem circle_radius (x y d : Real) (h : x + y + d = 164 * Real.pi) :
  ∃ (r : Real), r = 10 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ d = 2 * r := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3042_304283


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3042_304267

theorem quadratic_factorization (b : ℤ) :
  (∃ (c d e f : ℤ), 15 * x^2 + b * x + 45 = (c * x + d) * (e * x + f)) →
  ∃ (k : ℤ), b = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3042_304267
