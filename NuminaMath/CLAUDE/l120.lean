import Mathlib

namespace NUMINAMATH_CALUDE_regular_survey_rate_l120_12031

/-- Proves that the regular rate for completing a survey is 10 given the specified conditions. -/
theorem regular_survey_rate (total_surveys : ℕ) (cellphone_surveys : ℕ) (total_earnings : ℚ) :
  total_surveys = 100 →
  cellphone_surveys = 60 →
  total_earnings = 1180 →
  ∃ (regular_rate : ℚ),
    regular_rate * (total_surveys - cellphone_surveys) +
    (regular_rate * 1.3) * cellphone_surveys = total_earnings ∧
    regular_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_survey_rate_l120_12031


namespace NUMINAMATH_CALUDE_equation_solutions_l120_12087

theorem equation_solutions :
  (∀ x : ℝ, 4 * (x - 2)^2 = 9 ↔ x = 7/2 ∨ x = 1/2) ∧
  (∀ x : ℝ, x^2 + 6*x - 1 = 0 ↔ x = -3 + Real.sqrt 10 ∨ x = -3 - Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l120_12087


namespace NUMINAMATH_CALUDE_statement_c_not_always_true_l120_12056

theorem statement_c_not_always_true : 
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 := by
sorry

end NUMINAMATH_CALUDE_statement_c_not_always_true_l120_12056


namespace NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l120_12040

theorem evaluate_sqrt_fraction (y : ℝ) (h : y < -2) :
  Real.sqrt (y / (1 - (y + 1) / (y + 2))) = -y := by
  sorry

end NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l120_12040


namespace NUMINAMATH_CALUDE_total_difference_is_122_l120_12069

/-- The total difference in the number of apples and peaches for Mia, Steven, and Jake -/
def total_difference (steven_apples steven_peaches : ℕ) : ℕ :=
  let mia_apples := 2 * steven_apples
  let jake_apples := steven_apples + 4
  let jake_peaches := steven_peaches - 3
  let mia_peaches := jake_peaches + 3
  (mia_apples + mia_peaches) + (steven_apples + steven_peaches) + (jake_apples + jake_peaches)

/-- Theorem stating the total difference in fruits for Mia, Steven, and Jake -/
theorem total_difference_is_122 :
  total_difference 19 15 = 122 :=
by sorry

end NUMINAMATH_CALUDE_total_difference_is_122_l120_12069


namespace NUMINAMATH_CALUDE_calculation_proof_l120_12091

theorem calculation_proof : 65 + 5 * 12 / (180 / 3) = 66 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l120_12091


namespace NUMINAMATH_CALUDE_exponent_rules_l120_12044

theorem exponent_rules (a b : ℝ) : 
  ((-b)^2 * (-b)^3 * (-b)^5 = b^10) ∧ ((2*a*b^2)^3 = 8*a^3*b^6) := by
  sorry

end NUMINAMATH_CALUDE_exponent_rules_l120_12044


namespace NUMINAMATH_CALUDE_circle_circumference_l120_12007

/-- Given two circles with equal areas, where half the radius of one circle is 4.5,
    prove that the circumference of the other circle is 18π. -/
theorem circle_circumference (x y : ℝ) (harea : π * x^2 = π * y^2) (hy : y / 2 = 4.5) :
  2 * π * x = 18 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_l120_12007


namespace NUMINAMATH_CALUDE_variance_best_for_stability_l120_12099

-- Define a type for math test scores
def MathScore := ℝ

-- Define a type for a set of consecutive math test scores
def ConsecutiveScores := List MathScore

-- Define a function to calculate variance
noncomputable def variance (scores : ConsecutiveScores) : ℝ := sorry

-- Define a function to calculate other statistical measures
noncomputable def otherMeasure (scores : ConsecutiveScores) : ℝ := sorry

-- Define a function to measure stability
noncomputable def stability (scores : ConsecutiveScores) : ℝ := sorry

-- Theorem stating that variance is the most appropriate measure for stability
theorem variance_best_for_stability (scores : ConsecutiveScores) :
  ∀ (other : ConsecutiveScores → ℝ), other ≠ variance →
  |stability scores - variance scores| < |stability scores - other scores| :=
sorry

end NUMINAMATH_CALUDE_variance_best_for_stability_l120_12099


namespace NUMINAMATH_CALUDE_complex_multiplication_l120_12043

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 - i) = 1 + 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l120_12043


namespace NUMINAMATH_CALUDE_factorial_ratio_l120_12001

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 10 = 132 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l120_12001


namespace NUMINAMATH_CALUDE_apple_price_difference_l120_12009

/-- The price difference between Shimla apples and Fuji apples -/
def price_difference (shimla_price fuji_price : ℝ) : ℝ :=
  shimla_price - fuji_price

/-- The condition that the sum of Shimla and Red Delicious prices is 250 more than Red Delicious and Fuji -/
def price_condition (shimla_price red_delicious_price fuji_price : ℝ) : Prop :=
  shimla_price + red_delicious_price = red_delicious_price + fuji_price + 250

theorem apple_price_difference 
  (shimla_price red_delicious_price fuji_price : ℝ) 
  (h : price_condition shimla_price red_delicious_price fuji_price) : 
  price_difference shimla_price fuji_price = 250 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_difference_l120_12009


namespace NUMINAMATH_CALUDE_postage_for_5_25_ounces_l120_12097

/-- Calculates the postage cost for a letter given its weight and postage rates. -/
def calculate_postage (weight : ℚ) (base_rate : ℕ) (additional_rate : ℕ) : ℚ :=
  let additional_weight := max (weight - 1) 0
  let additional_charges := ⌈additional_weight⌉
  (base_rate + additional_charges * additional_rate) / 100

/-- Theorem stating that the postage for a 5.25 ounce letter is $1.60 under the given rates. -/
theorem postage_for_5_25_ounces :
  calculate_postage (5.25 : ℚ) 35 25 = (1.60 : ℚ) := by
  sorry

#eval calculate_postage (5.25 : ℚ) 35 25

end NUMINAMATH_CALUDE_postage_for_5_25_ounces_l120_12097


namespace NUMINAMATH_CALUDE_inequality_solution_set_l120_12054

theorem inequality_solution_set (x : ℝ) :
  (1 / (x + 2) + 5 / (x + 4) ≤ 1) ↔ (x ≤ -4 ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l120_12054


namespace NUMINAMATH_CALUDE_work_completion_time_l120_12074

/-- The time taken to complete a work given the rates of two workers and their working schedule -/
theorem work_completion_time
  (p_completion_time q_completion_time : ℝ)
  (p_solo_time : ℝ)
  (hp : p_completion_time = 20)
  (hq : q_completion_time = 12)
  (hp_solo : p_solo_time = 4)
  : ∃ (total_time : ℝ), total_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l120_12074


namespace NUMINAMATH_CALUDE_swimmer_speed_is_five_l120_12065

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  manSpeed : ℝ  -- Speed of the man in still water (km/h)
  streamSpeed : ℝ  -- Speed of the stream (km/h)

/-- Calculates the effective speed given the swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.manSpeed + s.streamSpeed else s.manSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 5 km/h. -/
theorem swimmer_speed_is_five 
  (s : SwimmerSpeed)
  (h1 : effectiveSpeed s true = 30 / 5)  -- Downstream condition
  (h2 : effectiveSpeed s false = 20 / 5) -- Upstream condition
  : s.manSpeed = 5 := by
  sorry

#check swimmer_speed_is_five

end NUMINAMATH_CALUDE_swimmer_speed_is_five_l120_12065


namespace NUMINAMATH_CALUDE_pie_fraction_to_percentage_l120_12072

theorem pie_fraction_to_percentage : 
  let apple_fraction : ℚ := 1/5
  let cherry_fraction : ℚ := 3/4
  let total_fraction : ℚ := apple_fraction + cherry_fraction
  (total_fraction * 100 : ℚ) = 95 := by sorry

end NUMINAMATH_CALUDE_pie_fraction_to_percentage_l120_12072


namespace NUMINAMATH_CALUDE_intersection_M_N_l120_12061

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {-2, 0, 2}

theorem intersection_M_N : M ∩ N = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l120_12061


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l120_12018

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x > 5 → x > 4) ∧
  (∃ x : ℝ, x > 4 ∧ x ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l120_12018


namespace NUMINAMATH_CALUDE_fraction_subtraction_multiplication_l120_12057

theorem fraction_subtraction_multiplication :
  (5/6 - 1/3) * 3/4 = 3/8 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_multiplication_l120_12057


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l120_12055

-- Define variables
variable (a b : ℝ)
variable (m n : ℕ)

-- Define the condition that the terms are like terms
def are_like_terms : Prop :=
  ∃ (k₁ k₂ : ℝ), k₁ * a^(2*m) * b = k₂ * a^4 * b^n

-- Theorem statement
theorem like_terms_exponent_sum :
  are_like_terms a b m n → m + n = 3 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l120_12055


namespace NUMINAMATH_CALUDE_dolly_dresses_shipment_l120_12083

theorem dolly_dresses_shipment (total : ℕ) : 
  (70 : ℕ) * total = 140 * 100 → total = 200 := by
  sorry

end NUMINAMATH_CALUDE_dolly_dresses_shipment_l120_12083


namespace NUMINAMATH_CALUDE_number_of_possible_lists_l120_12085

def num_balls : ℕ := 15
def list_length : ℕ := 4

theorem number_of_possible_lists :
  (num_balls ^ list_length : ℕ) = 50625 := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_lists_l120_12085


namespace NUMINAMATH_CALUDE_total_different_groups_l120_12046

-- Define the number of marbles of each color
def red_marbles : ℕ := 1
def green_marbles : ℕ := 1
def blue_marbles : ℕ := 1
def yellow_marbles : ℕ := 3

-- Define the total number of distinct colors
def distinct_colors : ℕ := 4

-- Define the function to calculate the number of different groups
def different_groups : ℕ :=
  -- Groups with two yellow marbles
  1 +
  -- Groups with two different colors
  (distinct_colors.choose 2)

-- Theorem statement
theorem total_different_groups :
  different_groups = 7 :=
sorry

end NUMINAMATH_CALUDE_total_different_groups_l120_12046


namespace NUMINAMATH_CALUDE_angle_C_is_two_pi_third_l120_12089

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define vectors p and q
def p (t : Triangle) : ℝ × ℝ := (t.a + t.c, t.b)
def q (t : Triangle) : ℝ × ℝ := (t.b + t.a, t.c - t.a)

-- Define parallelism of vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem angle_C_is_two_pi_third (t : Triangle) 
  (h : parallel (p t) (q t)) : t.C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_two_pi_third_l120_12089


namespace NUMINAMATH_CALUDE_percentage_difference_l120_12011

theorem percentage_difference (third : ℝ) (first second : ℝ) 
  (h1 : first = 0.75 * third) 
  (h2 : second = first - 0.06 * first) : 
  (third - second) / third = 0.295 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l120_12011


namespace NUMINAMATH_CALUDE_f_properties_l120_12039

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a| + |2*x - 1/a|

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x, f 1 x ≤ 6 ↔ x ∈ Set.Icc (-7/3) (5/3)) ∧
  (∀ x, f a x ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l120_12039


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l120_12052

/-- An isosceles triangle with height 8 and perimeter 32 has an area of 48 -/
theorem isosceles_triangle_area (b : ℝ) (l : ℝ) (h : ℝ) (S : ℝ) : 
  h = 8 → -- height is 8
  2 * l + b = 32 → -- perimeter is 32
  l ^ 2 = (b / 2) ^ 2 + h ^ 2 → -- Pythagorean theorem
  S = (1 / 2) * b * h → -- area formula
  S = 48 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l120_12052


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l120_12062

-- Define propositions p and q
def p (x : ℝ) : Prop := x > 4
def q (x : ℝ) : Prop := 4 < x ∧ x < 10

-- Theorem statement
theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l120_12062


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l120_12013

theorem inverse_proposition_false : ∃ a b : ℝ, (abs a = abs b) ∧ (a ≠ b) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l120_12013


namespace NUMINAMATH_CALUDE_paint_time_theorem_l120_12049

/-- The time required to paint a square wall using a cylindrical paint roller -/
theorem paint_time_theorem (roller_length roller_diameter wall_side_length roller_speed : ℝ) :
  roller_length = 20 →
  roller_diameter = 15 →
  wall_side_length = 300 →
  roller_speed = 2 →
  (wall_side_length ^ 2) / (2 * π * (roller_diameter / 2) * roller_length * roller_speed) = 90000 / (600 * π) :=
by sorry

end NUMINAMATH_CALUDE_paint_time_theorem_l120_12049


namespace NUMINAMATH_CALUDE_max_area_rectangle_in_circle_max_area_is_8_l120_12034

theorem max_area_rectangle_in_circle (x y : ℝ) : 
  x > 0 → y > 0 → x^2 + y^2 = 16 → x * y ≤ 8 := by
  sorry

theorem max_area_is_8 : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = 16 ∧ x * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_in_circle_max_area_is_8_l120_12034


namespace NUMINAMATH_CALUDE_dads_strawberries_weight_l120_12077

/-- The weight of Marco's dad's strawberries -/
def dads_strawberries (marcos_weight total_weight : ℕ) : ℕ :=
  total_weight - marcos_weight

/-- Theorem: Marco's dad's strawberries weigh 22 pounds -/
theorem dads_strawberries_weight :
  dads_strawberries 15 37 = 22 := by
  sorry

end NUMINAMATH_CALUDE_dads_strawberries_weight_l120_12077


namespace NUMINAMATH_CALUDE_sum_invested_is_15000_l120_12029

/-- The sum invested that satisfies the given conditions -/
def find_sum (interest_rate_high : ℚ) (interest_rate_low : ℚ) (time : ℚ) (interest_difference : ℚ) : ℚ :=
  interest_difference / (time * (interest_rate_high - interest_rate_low))

/-- Theorem stating that the sum invested is 15000 given the problem conditions -/
theorem sum_invested_is_15000 :
  find_sum (15/100) (12/100) 2 900 = 15000 := by
  sorry

#eval find_sum (15/100) (12/100) 2 900

end NUMINAMATH_CALUDE_sum_invested_is_15000_l120_12029


namespace NUMINAMATH_CALUDE_tom_found_seven_seashells_l120_12012

/-- The number of seashells Tom found yesterday -/
def seashells_yesterday : ℕ := sorry

/-- The number of seashells Tom found today -/
def seashells_today : ℕ := 4

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := 11

/-- Theorem stating that Tom found 7 seashells yesterday -/
theorem tom_found_seven_seashells : seashells_yesterday = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_found_seven_seashells_l120_12012


namespace NUMINAMATH_CALUDE_wedding_attendance_percentage_l120_12063

theorem wedding_attendance_percentage 
  (total_invitations : ℕ) 
  (rsvp_rate : ℚ)
  (thank_you_cards : ℕ) 
  (no_gift_attendees : ℕ) :
  total_invitations = 200 →
  rsvp_rate = 9/10 →
  thank_you_cards = 134 →
  no_gift_attendees = 10 →
  (thank_you_cards + no_gift_attendees) / (total_invitations * rsvp_rate) = 4/5 := by
sorry

#eval (134 + 10) / (200 * (9/10)) -- This should evaluate to 4/5

end NUMINAMATH_CALUDE_wedding_attendance_percentage_l120_12063


namespace NUMINAMATH_CALUDE_solution_set_theorem_l120_12081

def inequality_system (x : ℝ) : Prop :=
  (4 * x + 1 > 2) ∧ (1 - 2 * x < 7)

theorem solution_set_theorem :
  {x : ℝ | inequality_system x} = {x : ℝ | x > 1/4} := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l120_12081


namespace NUMINAMATH_CALUDE_fifteenth_set_sum_l120_12088

def first_element (n : ℕ) : ℕ := 
  1 + (n - 1) * n / 2

def last_element (n : ℕ) : ℕ := 
  first_element n + n - 1

def set_sum (n : ℕ) : ℕ := 
  n * (first_element n + last_element n) / 2

theorem fifteenth_set_sum : set_sum 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_set_sum_l120_12088


namespace NUMINAMATH_CALUDE_law_of_sines_extended_l120_12038

theorem law_of_sines_extended 
  {a b c α β γ : ℝ} 
  (law_of_sines : a / Real.sin α = b / Real.sin β ∧ 
                  b / Real.sin β = c / Real.sin γ)
  (angle_sum : α + β + γ = Real.pi) :
  a = b * Real.cos γ + c * Real.cos β ∧
  b = c * Real.cos α + a * Real.cos γ ∧
  c = a * Real.cos β + b * Real.cos α := by
sorry

end NUMINAMATH_CALUDE_law_of_sines_extended_l120_12038


namespace NUMINAMATH_CALUDE_pizza_combinations_l120_12060

/-- The number of pizza toppings available. -/
def num_toppings : ℕ := 8

/-- The number of incompatible topping pairs. -/
def num_incompatible_pairs : ℕ := 1

/-- Calculates the number of combinations of n items taken k at a time. -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of possible one-topping and two-topping pizzas, given the number of toppings
    and the number of incompatible pairs. -/
def total_pizzas (n incompatible : ℕ) : ℕ :=
  n + combinations n 2 - incompatible

/-- Theorem stating that the total number of possible one-topping and two-topping pizzas
    is 35, given 8 toppings and 1 incompatible pair. -/
theorem pizza_combinations :
  total_pizzas num_toppings num_incompatible_pairs = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l120_12060


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l120_12010

/-- A right triangle with specific median lengths has a hypotenuse of 4√14 -/
theorem right_triangle_hypotenuse (a b : ℝ) (h_right : a^2 + b^2 = (a + b)^2 / 4) 
  (h_median1 : b^2 + (a/2)^2 = 34) (h_median2 : a^2 + (b/2)^2 = 36) : 
  (a + b) = 4 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l120_12010


namespace NUMINAMATH_CALUDE_population_theorem_l120_12020

/-- The combined population of Pirajussaraí and Tucupira three years ago -/
def combined_population_three_years_ago (pirajussarai_population : ℕ) (tucupira_population : ℕ) : ℕ :=
  pirajussarai_population + tucupira_population

/-- The current combined population of Pirajussaraí and Tucupira -/
def current_combined_population (pirajussarai_population : ℕ) (tucupira_population : ℕ) : ℕ :=
  pirajussarai_population + tucupira_population * 3 / 2

theorem population_theorem (pirajussarai_population : ℕ) (tucupira_population : ℕ) :
  current_combined_population pirajussarai_population tucupira_population = 9000 →
  combined_population_three_years_ago pirajussarai_population tucupira_population = 7200 :=
by
  sorry

#check population_theorem

end NUMINAMATH_CALUDE_population_theorem_l120_12020


namespace NUMINAMATH_CALUDE_lcm_gcd_product_equals_product_l120_12016

theorem lcm_gcd_product_equals_product (a b : ℕ) (ha : a = 12) (hb : b = 18) :
  (Nat.lcm a b) * (Nat.gcd a b) = a * b := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_equals_product_l120_12016


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_bounds_l120_12093

/-- Represents the dimensions of a rectangular solid --/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the surface area of a rectangular solid --/
def surfaceArea (d : Dimensions) : ℕ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Checks if the given dimensions use exactly 12 unit cubes --/
def usestwelveCubes (d : Dimensions) : Prop :=
  d.length * d.width * d.height = 12

theorem rectangular_solid_surface_area_bounds :
  ∃ (min max : ℕ),
    (∀ d : Dimensions, usestwelveCubes d → min ≤ surfaceArea d) ∧
    (∃ d : Dimensions, usestwelveCubes d ∧ surfaceArea d = min) ∧
    (∀ d : Dimensions, usestwelveCubes d → surfaceArea d ≤ max) ∧
    (∃ d : Dimensions, usestwelveCubes d ∧ surfaceArea d = max) ∧
    min = 32 ∧ max = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_bounds_l120_12093


namespace NUMINAMATH_CALUDE_equation_solutions_l120_12037

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => 1/((x-2)*(x-3)) + 1/((x-3)*(x-4)) + 1/((x-4)*(x-5))
  ∀ x : ℝ, f x = 1/12 ↔ x = (7 + Real.sqrt 153)/2 ∨ x = (7 - Real.sqrt 153)/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l120_12037


namespace NUMINAMATH_CALUDE_planes_intersect_l120_12084

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (are_skew : Line → Line → Prop)
variable (is_perpendicular_to_plane : Line → Plane → Prop)
variable (are_intersecting : Plane → Plane → Prop)

-- State the theorem
theorem planes_intersect (a b : Line) (α β : Plane) 
  (h1 : are_skew a b)
  (h2 : is_perpendicular_to_plane a α)
  (h3 : is_perpendicular_to_plane b β) :
  are_intersecting α β :=
sorry

end NUMINAMATH_CALUDE_planes_intersect_l120_12084


namespace NUMINAMATH_CALUDE_smallest_stamp_collection_l120_12002

theorem smallest_stamp_collection (M : ℕ) : 
  M > 2 →
  M % 5 = 2 →
  M % 7 = 2 →
  M % 9 = 2 →
  (∀ N : ℕ, N > 2 ∧ N % 5 = 2 ∧ N % 7 = 2 ∧ N % 9 = 2 → N ≥ M) →
  M = 317 :=
by sorry

end NUMINAMATH_CALUDE_smallest_stamp_collection_l120_12002


namespace NUMINAMATH_CALUDE_triangle_sine_product_l120_12071

theorem triangle_sine_product (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b + c = 3 →
  A = π / 3 →
  0 < B →
  B < π →
  0 < C →
  C < π →
  A + B + C = π →
  a = 2 * Real.sin (B / 2) * Real.sin (C / 2) →
  b = 2 * Real.sin (A / 2) * Real.sin (C / 2) →
  c = 2 * Real.sin (A / 2) * Real.sin (B / 2) →
  Real.sin B * Real.sin C = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_product_l120_12071


namespace NUMINAMATH_CALUDE_employee_count_l120_12015

theorem employee_count (avg_salary : ℕ) (salary_increase : ℕ) (manager_salary : ℕ)
  (h1 : avg_salary = 1300)
  (h2 : salary_increase = 100)
  (h3 : manager_salary = 3400) :
  ∃ n : ℕ, n * avg_salary + manager_salary = (n + 1) * (avg_salary + salary_increase) ∧ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l120_12015


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l120_12042

/-- Given a rectangle EFGH where:
  * EF is twice as long as FG
  * FG = 10 units
  * Diagonal EH = 26 units
Prove that the perimeter of EFGH is 60 units -/
theorem rectangle_perimeter (EF FG EH : ℝ) : 
  EF = 2 * FG →
  FG = 10 →
  EH = 26 →
  EH^2 = EF^2 + FG^2 →
  2 * (EF + FG) = 60 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_l120_12042


namespace NUMINAMATH_CALUDE_apple_vendor_waste_percentage_l120_12036

/-- Calculates the percentage of apples thrown away given the selling and discarding percentages -/
theorem apple_vendor_waste_percentage
  (initial_apples : ℝ)
  (day1_sell_percentage : ℝ)
  (day1_discard_percentage : ℝ)
  (day2_sell_percentage : ℝ)
  (h1 : initial_apples > 0)
  (h2 : day1_sell_percentage = 0.5)
  (h3 : day1_discard_percentage = 0.2)
  (h4 : day2_sell_percentage = 0.5)
  : (day1_discard_percentage * (1 - day1_sell_percentage) +
     (1 - day2_sell_percentage) * (1 - day1_sell_percentage) * (1 - day1_discard_percentage)) = 0.3 := by
  sorry

#check apple_vendor_waste_percentage

end NUMINAMATH_CALUDE_apple_vendor_waste_percentage_l120_12036


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l120_12000

/-- Given a hyperbola x^2 - y^2/b^2 = 1 with b > 1, the angle θ between its asymptotes is not 2arctan(b) -/
theorem hyperbola_asymptote_angle (b : ℝ) (h : b > 1) :
  let θ := Real.pi - 2 * Real.arctan b
  θ ≠ 2 * Real.arctan b :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l120_12000


namespace NUMINAMATH_CALUDE_perimeter_of_special_isosceles_triangle_l120_12073

-- Define the real numbers m and n
variable (m n : ℝ)

-- Define the condition |m-2| + √(n-4) = 0
def condition (m n : ℝ) : Prop := abs (m - 2) + Real.sqrt (n - 4) = 0

-- Define an isosceles triangle with sides m and n
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (base : ℝ)
  (is_isosceles : side1 = side2)

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.side1 + t.side2 + t.base

-- State the theorem
theorem perimeter_of_special_isosceles_triangle :
  ∀ m n : ℝ, condition m n →
  ∃ t : IsoscelesTriangle, (t.side1 = m ∨ t.side1 = n) ∧ (t.base = m ∨ t.base = n) →
  perimeter t = 10 :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_special_isosceles_triangle_l120_12073


namespace NUMINAMATH_CALUDE_nested_percentage_calculation_l120_12028

-- Define the initial amount
def initial_amount : ℝ := 3000

-- Define the percentages
def percent_1 : ℝ := 0.20
def percent_2 : ℝ := 0.35
def percent_3 : ℝ := 0.05

-- State the theorem
theorem nested_percentage_calculation :
  percent_3 * (percent_2 * (percent_1 * initial_amount)) = 10.50 := by
  sorry

end NUMINAMATH_CALUDE_nested_percentage_calculation_l120_12028


namespace NUMINAMATH_CALUDE_quadratic_inequality_l120_12017

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that f(3^x) ≤ f(2^x) for all real x, 
    given that f is monotonically increasing on (-∞, 1] -/
theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 1 → f a b c x ≤ f a b c y) →
  ∀ x : ℝ, f a b c (3^x) ≤ f a b c (2^x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l120_12017


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l120_12030

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l120_12030


namespace NUMINAMATH_CALUDE_range_of_a_l120_12008

-- Define the set of real numbers a that satisfy the condition
def A : Set ℝ := {a : ℝ | ∀ x : ℝ, a * x^2 + a * x + 1 > 0}

-- Theorem stating that A is equal to the interval [0, 4)
theorem range_of_a : A = Set.Icc 0 (4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l120_12008


namespace NUMINAMATH_CALUDE_C_power_50_l120_12079

def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem C_power_50 : C^50 = !![(-299), (-100); 800, 251] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l120_12079


namespace NUMINAMATH_CALUDE_quadratic_function_k_range_l120_12075

/-- Given a quadratic function f(x) = 4x^2 - kx - 8 with no maximum or minimum at (5, 20),
    prove that the range of k is k ≤ 40 or k ≥ 160. -/
theorem quadratic_function_k_range (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - k * x - 8
  (∀ x, f x ≠ f 5 ∨ (∃ y, y ≠ 5 ∧ f y = f 5)) →
  f 5 = 20 →
  k ≤ 40 ∨ k ≥ 160 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_k_range_l120_12075


namespace NUMINAMATH_CALUDE_only_valid_solutions_l120_12066

/-- A pair of natural numbers (m, n) is a valid solution if both n^2 + 4m and m^2 + 5n are perfect squares. -/
def is_valid_solution (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), n^2 + 4*m = a^2 ∧ m^2 + 5*n = b^2

/-- The set of all valid solutions. -/
def valid_solutions : Set (ℕ × ℕ) :=
  {p | is_valid_solution p.1 p.2}

/-- The theorem stating that the only valid solutions are (2,1), (22,9), and (9,8). -/
theorem only_valid_solutions :
  valid_solutions = {(2, 1), (22, 9), (9, 8)} :=
by sorry

end NUMINAMATH_CALUDE_only_valid_solutions_l120_12066


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l120_12078

theorem arithmetic_evaluation : 6 + (3 * 6) - 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l120_12078


namespace NUMINAMATH_CALUDE_prime_sum_equality_l120_12019

theorem prime_sum_equality (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p + q = r → p < q → q < r → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_equality_l120_12019


namespace NUMINAMATH_CALUDE_malcolm_red_lights_l120_12035

def malcolm_lights (red : ℕ) (blue : ℕ) (green : ℕ) (left_to_buy : ℕ) (total_white : ℕ) : Prop :=
  blue = 3 * red ∧
  green = 6 ∧
  left_to_buy = 5 ∧
  total_white = 59 ∧
  red + blue + green + left_to_buy = total_white

theorem malcolm_red_lights :
  ∃ (red : ℕ), malcolm_lights red (3 * red) 6 5 59 ∧ red = 12 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_red_lights_l120_12035


namespace NUMINAMATH_CALUDE_total_cats_is_thirteen_l120_12068

/-- The number of cats owned by Jamie, Gordon, and Hawkeye --/
def total_cats : ℕ :=
  let jamie_persians : ℕ := 4
  let jamie_maine_coons : ℕ := 2
  let gordon_persians : ℕ := jamie_persians / 2
  let gordon_maine_coons : ℕ := jamie_maine_coons + 1
  let hawkeye_persians : ℕ := 0
  let hawkeye_maine_coons : ℕ := gordon_maine_coons - 1
  jamie_persians + jamie_maine_coons +
  gordon_persians + gordon_maine_coons +
  hawkeye_persians + hawkeye_maine_coons

theorem total_cats_is_thirteen : total_cats = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_is_thirteen_l120_12068


namespace NUMINAMATH_CALUDE_leakage_time_to_empty_tank_l120_12058

/-- Given a pipe that takes 'a' hours to fill a tank without leakage,
    and 7a hours to fill the tank with leakage, prove that the time 'l'
    taken by the leakage alone to empty the tank is equal to 7a/6 hours. -/
theorem leakage_time_to_empty_tank (a : ℝ) (h : a > 0) :
  let l : ℝ := (7 * a) / 6
  let fill_rate : ℝ := 1 / a
  let leak_rate : ℝ := 1 / l
  fill_rate - leak_rate = 1 / (7 * a) :=
by sorry

end NUMINAMATH_CALUDE_leakage_time_to_empty_tank_l120_12058


namespace NUMINAMATH_CALUDE_special_triangle_properties_l120_12024

/-- Triangle ABC with specific conditions -/
structure SpecialTriangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side length opposite to A
  b : ℝ  -- Side length opposite to B
  c : ℝ  -- Side length opposite to C
  angle_sum : A + B + C = π
  side_condition : a + c = 3 * Real.sqrt 3 / 2
  side_b : b = Real.sqrt 3
  angle_condition : 2 * Real.cos A * Real.cos C * (Real.tan A * Real.tan C - 1) = 1

/-- Theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  t.B = π / 3 ∧ (1 / 2 * t.a * t.c * Real.sin t.B = 5 * Real.sqrt 3 / 16) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l120_12024


namespace NUMINAMATH_CALUDE_integer_root_of_polynomial_l120_12070

theorem integer_root_of_polynomial (b c : ℚ) : 
  (∃ (x : ℝ), x^3 + b*x + c = 0 ∧ x = 3 - Real.sqrt 5) → 
  (-6 : ℝ)^3 + b*(-6) + c = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_root_of_polynomial_l120_12070


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_l120_12080

theorem smallest_four_digit_congruence :
  ∃ (n : ℕ), 
    1000 ≤ n ∧ 
    n < 10000 ∧ 
    (75 * n) % 375 = 225 ∧ 
    (∀ (m : ℕ), 1000 ≤ m ∧ m < n → (75 * m) % 375 ≠ 225) ∧
    n = 1003 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_l120_12080


namespace NUMINAMATH_CALUDE_point_B_value_l120_12045

def point_A : ℝ := -1

def distance_AB : ℝ := 4

theorem point_B_value : 
  ∃ (B : ℝ), (B = 3 ∨ B = -5) ∧ |B - point_A| = distance_AB :=
by sorry

end NUMINAMATH_CALUDE_point_B_value_l120_12045


namespace NUMINAMATH_CALUDE_min_value_theorem_l120_12098

theorem min_value_theorem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  (2 * a * e)^2 + (2 * b * f)^2 + (2 * c * g)^2 + (2 * d * h)^2 ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l120_12098


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l120_12033

theorem quadratic_integer_roots (p : ℕ) : 
  Prime p ∧ 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + p*x - 512*p = 0 ∧ y^2 + p*y - 512*p = 0) ↔ 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l120_12033


namespace NUMINAMATH_CALUDE_savings_percentage_l120_12050

theorem savings_percentage (salary : ℝ) (savings_after_increase : ℝ) 
  (expense_increase_rate : ℝ) :
  salary = 5500 →
  savings_after_increase = 220 →
  expense_increase_rate = 0.2 →
  ∃ (original_savings_percentage : ℝ),
    original_savings_percentage = 20 ∧
    savings_after_increase = salary - (1 + expense_increase_rate) * 
      (salary - (original_savings_percentage / 100) * salary) :=
by sorry

end NUMINAMATH_CALUDE_savings_percentage_l120_12050


namespace NUMINAMATH_CALUDE_sqrt_a_sqrt_a_sqrt_a_l120_12092

theorem sqrt_a_sqrt_a_sqrt_a (a : ℝ) (ha : a ≥ 0) : 
  Real.sqrt (a * Real.sqrt a * Real.sqrt a) = a := by
sorry

end NUMINAMATH_CALUDE_sqrt_a_sqrt_a_sqrt_a_l120_12092


namespace NUMINAMATH_CALUDE_polyhedron_inequality_l120_12082

/-- A convex polyhedron bounded by quadrilateral faces -/
class ConvexPolyhedron where
  /-- The surface area of the polyhedron -/
  area : ℝ
  /-- The sum of the squares of the polyhedron's edges -/
  edge_sum_squares : ℝ
  /-- The polyhedron is bounded by quadrilateral faces -/
  quad_faces : Prop

/-- 
For a convex polyhedron bounded by quadrilateral faces, 
the sum of the squares of its edges is greater than or equal to twice its surface area 
-/
theorem polyhedron_inequality (p : ConvexPolyhedron) : p.edge_sum_squares ≥ 2 * p.area := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_inequality_l120_12082


namespace NUMINAMATH_CALUDE_w_sequence_properties_l120_12086

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sequence w_n
def w : ℕ → ℂ
  | 0 => 1
  | 1 => i
  | (n + 2) => 2 * w (n + 1) + 3 * w n

-- State the theorem
theorem w_sequence_properties :
  (∀ n : ℕ, w n = (1 + i) / 4 * 3^n + (3 - i) / 4 * (-1)^n) ∧
  (∀ n : ℕ, n ≥ 1 → |Complex.re (w n) - Complex.im (w n)| = 1) := by
  sorry


end NUMINAMATH_CALUDE_w_sequence_properties_l120_12086


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l120_12076

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h1 : ∀ n, a n > 0
  h2 : q ≠ 1
  h3 : ∀ n, a (n + 1) = q * a n

/-- The sum of the first and eighth terms is greater than the sum of the fourth and fifth terms -/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  seq.a 1 + seq.a 8 > seq.a 4 + seq.a 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l120_12076


namespace NUMINAMATH_CALUDE_purely_imaginary_z_equals_one_l120_12027

theorem purely_imaginary_z_equals_one (x : ℝ) :
  let z : ℂ := (x + (x^2 - 1) * Complex.I) / Complex.I
  (∃ (y : ℝ), z = Complex.I * y) → z = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_equals_one_l120_12027


namespace NUMINAMATH_CALUDE_equation_solution_l120_12032

theorem equation_solution (k : ℤ) : 
  (∃ x : ℤ, Real.sqrt (39 - 6 * Real.sqrt 12) + Real.sqrt (k * x * (k * x + Real.sqrt 12) + 3) = 2 * k) ↔ 
  (k = 3 ∨ k = 6) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l120_12032


namespace NUMINAMATH_CALUDE_unique_triple_solution_l120_12041

theorem unique_triple_solution : 
  ∃! (a b c : ℕ+), a * b + b * c = 72 ∧ a * c + b * c = 35 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l120_12041


namespace NUMINAMATH_CALUDE_fraction_not_simplifiable_l120_12094

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_simplifiable_l120_12094


namespace NUMINAMATH_CALUDE_triangle_max_area_l120_12004

theorem triangle_max_area (a b c : ℝ) (h1 : a + b = 12) (h2 : c = 8) :
  let p := (a + b + c) / 2
  let area := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  ∀ a' b' c', a' + b' = 12 → c' = 8 →
    let p' := (a' + b' + c') / 2
    let area' := Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c'))
    area ≤ area' →
  area = 8 * Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_max_area_l120_12004


namespace NUMINAMATH_CALUDE_apple_lemon_equivalence_l120_12096

/-- Represents the value of fruits in terms of a common unit -/
structure FruitValue where
  apple : ℚ
  lemon : ℚ

/-- Given that 3/4 of 14 apples are worth 9 lemons, 
    prove that 5/7 of 7 apples are worth 30/7 lemons -/
theorem apple_lemon_equivalence (v : FruitValue) 
  (h : (3/4 : ℚ) * 14 * v.apple = 9 * v.lemon) :
  (5/7 : ℚ) * 7 * v.apple = (30/7 : ℚ) * v.lemon := by
  sorry

#check apple_lemon_equivalence

end NUMINAMATH_CALUDE_apple_lemon_equivalence_l120_12096


namespace NUMINAMATH_CALUDE_disk_with_hole_moment_of_inertia_l120_12023

/-- The moment of inertia of a disk with a hole -/
theorem disk_with_hole_moment_of_inertia
  (R M : ℝ)
  (h_R : R > 0)
  (h_M : M > 0) :
  let I₀ : ℝ := (1 / 2) * M * R^2
  let m_hole : ℝ := M / 4
  let R_hole : ℝ := R / 2
  let I_center_hole : ℝ := (1 / 2) * m_hole * R_hole^2
  let d : ℝ := R / 2
  let I_hole : ℝ := I_center_hole + m_hole * d^2
  I₀ - I_hole = (13 / 32) * M * R^2 :=
sorry

end NUMINAMATH_CALUDE_disk_with_hole_moment_of_inertia_l120_12023


namespace NUMINAMATH_CALUDE_fence_perimeter_l120_12095

/-- The number of posts used to enclose the garden -/
def num_posts : ℕ := 36

/-- The width of each post in inches -/
def post_width_inches : ℕ := 3

/-- The space between adjacent posts in feet -/
def post_spacing_feet : ℕ := 4

/-- Conversion factor from inches to feet -/
def inches_to_feet : ℚ := 1 / 12

/-- The width of each post in feet -/
def post_width_feet : ℚ := post_width_inches * inches_to_feet

/-- The number of posts on each side of the square garden -/
def posts_per_side : ℕ := num_posts / 4 + 1

/-- The number of spaces between posts on each side -/
def spaces_per_side : ℕ := posts_per_side - 1

/-- The length of one side of the square garden in feet -/
def side_length : ℚ := posts_per_side * post_width_feet + spaces_per_side * post_spacing_feet

/-- The outer perimeter of the fence surrounding the square garden -/
def outer_perimeter : ℚ := 4 * side_length

/-- Theorem stating that the outer perimeter of the fence is 137 feet -/
theorem fence_perimeter : outer_perimeter = 137 := by
  sorry

end NUMINAMATH_CALUDE_fence_perimeter_l120_12095


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l120_12051

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℝ) -- Sum function for the arithmetic sequence
  (h1 : S 2 = 4) -- Given S_2 = 4
  (h2 : S 4 = 20) -- Given S_4 = 20
  : ∃ (a₁ d : ℝ), 
    (∀ n : ℕ, S n = n * (2 * a₁ + (n - 1) * d) / 2) ∧ 
    d = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l120_12051


namespace NUMINAMATH_CALUDE_car_repair_cost_l120_12005

/-- Calculates the total cost for a car repair given the mechanic's hourly rate,
    hours worked per day, number of days worked, and cost of parts. -/
theorem car_repair_cost (hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (parts_cost : ℕ) :
  hourly_rate = 60 →
  hours_per_day = 8 →
  days_worked = 14 →
  parts_cost = 2500 →
  hourly_rate * hours_per_day * days_worked + parts_cost = 9220 := by
  sorry

#check car_repair_cost

end NUMINAMATH_CALUDE_car_repair_cost_l120_12005


namespace NUMINAMATH_CALUDE_valid_purchase_has_two_notebooks_l120_12003

/-- Represents the purchase of notebooks and books -/
structure Purchase where
  notebooks : ℕ
  books : ℕ
  notebook_cost : ℕ
  book_cost : ℕ

/-- Checks if the purchase satisfies the given conditions -/
def is_valid_purchase (p : Purchase) : Prop :=
  p.books = p.notebooks + 4 ∧
  p.notebooks * p.notebook_cost = 72 ∧
  p.books * p.book_cost = 660 ∧
  p.notebooks * p.book_cost + p.books * p.notebook_cost < 444

/-- The theorem stating that the valid purchase has 2 notebooks -/
theorem valid_purchase_has_two_notebooks :
  ∃ (p : Purchase), is_valid_purchase p ∧ p.notebooks = 2 :=
sorry


end NUMINAMATH_CALUDE_valid_purchase_has_two_notebooks_l120_12003


namespace NUMINAMATH_CALUDE_number_value_l120_12059

theorem number_value (tens : ℕ) (ones : ℕ) (tenths : ℕ) (hundredths : ℕ) :
  tens = 21 →
  ones = 8 →
  tenths = 5 →
  hundredths = 34 →
  (tens * 10 : ℚ) + ones + (tenths : ℚ) / 10 + (hundredths : ℚ) / 100 = 218.84 :=
by sorry

end NUMINAMATH_CALUDE_number_value_l120_12059


namespace NUMINAMATH_CALUDE_line_canonical_form_l120_12090

theorem line_canonical_form (x y z : ℝ) :
  (2 * x - 3 * y - 3 * z - 9 = 0 ∧ x - 2 * y + z + 3 = 0) →
  ∃ (t : ℝ), x = 9 * t ∧ y = 5 * t ∧ z = t - 3 :=
by sorry

end NUMINAMATH_CALUDE_line_canonical_form_l120_12090


namespace NUMINAMATH_CALUDE_discounted_price_theorem_l120_12022

/-- Given a bag marked at $200 with a 40% discount, prove that the discounted price is $120. -/
theorem discounted_price_theorem (marked_price : ℝ) (discount_percentage : ℝ) 
  (h1 : marked_price = 200)
  (h2 : discount_percentage = 40) :
  marked_price * (1 - discount_percentage / 100) = 120 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_theorem_l120_12022


namespace NUMINAMATH_CALUDE_abab_baba_divisible_by_three_l120_12021

theorem abab_baba_divisible_by_three (A B : ℕ) :
  A ≠ B →
  A ∈ Finset.range 10 →
  B ∈ Finset.range 10 →
  A ≠ 0 →
  B ≠ 0 →
  ∃ k : ℤ, (1010 * A + 101 * B) - (101 * A + 1010 * B) = 3 * k :=
by sorry

end NUMINAMATH_CALUDE_abab_baba_divisible_by_three_l120_12021


namespace NUMINAMATH_CALUDE_ancient_chinese_problem_correct_l120_12014

/-- Represents the system of equations for the ancient Chinese math problem --/
def ancient_chinese_problem (x y : ℤ) : Prop :=
  (y = 8*x - 3) ∧ (y = 7*x + 4)

/-- Theorem stating that the system of equations correctly represents the problem --/
theorem ancient_chinese_problem_correct (x y : ℤ) :
  (ancient_chinese_problem x y) ↔
  (x ≥ 0) ∧  -- number of people is non-negative
  (y ≥ 0) ∧  -- price is non-negative
  (8*x - y = 3) ∧  -- excess of 3 coins when each contributes 8
  (y - 7*x = 4)    -- shortage of 4 coins when each contributes 7
  := by sorry

end NUMINAMATH_CALUDE_ancient_chinese_problem_correct_l120_12014


namespace NUMINAMATH_CALUDE_congruent_count_l120_12025

theorem congruent_count : Nat.card {n : ℕ | 0 < n ∧ n < 500 ∧ n % 7 = 3} = 71 := by
  sorry

end NUMINAMATH_CALUDE_congruent_count_l120_12025


namespace NUMINAMATH_CALUDE_h_equality_l120_12026

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - x + 1
def g (x : ℝ) : ℝ := -x^2 + x + 1

-- Define the polynomial h
def h (x : ℝ) : ℝ := (x - 1)^2

-- Theorem statement
theorem h_equality (x : ℝ) : h (f x) = h (g x) := by
  sorry

end NUMINAMATH_CALUDE_h_equality_l120_12026


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l120_12064

theorem largest_n_with_unique_k : ∃ (n : ℕ), n > 0 ∧ n ≤ 136 ∧
  (∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/15) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (9 : ℚ)/17 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 8/15)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l120_12064


namespace NUMINAMATH_CALUDE_complex_cube_root_unity_l120_12006

theorem complex_cube_root_unity (i : ℂ) (x : ℂ) : 
  i^2 = -1 → 
  x = (-1 + i * Real.sqrt 3) / 2 → 
  1 / (x^3 - x) = -1/2 + (i * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_unity_l120_12006


namespace NUMINAMATH_CALUDE_total_swordfish_caught_l120_12067

def fishing_trips : ℕ := 5

def shelly_catch : ℕ := 5 - 2

def sam_catch : ℕ := shelly_catch - 1

theorem total_swordfish_caught : shelly_catch * fishing_trips + sam_catch * fishing_trips = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_swordfish_caught_l120_12067


namespace NUMINAMATH_CALUDE_rectangle_area_l120_12048

theorem rectangle_area (length width : ℝ) (h1 : length = 2 * Real.sqrt 6) (h2 : width = 2 * Real.sqrt 3) :
  length * width = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l120_12048


namespace NUMINAMATH_CALUDE_product_set_sum_l120_12047

theorem product_set_sum (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Set ℚ) = {-24, -2, -3/2, -1/8, 1, 3} →
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
sorry

end NUMINAMATH_CALUDE_product_set_sum_l120_12047


namespace NUMINAMATH_CALUDE_office_count_l120_12053

theorem office_count (bureaus : ℕ) (additional : ℕ) (offices : ℕ) : 
  bureaus = 88 →
  additional = 10 →
  (bureaus + additional) % offices = 0 →
  bureaus % offices ≠ 0 →
  offices > 1 →
  offices = 7 := by
sorry

end NUMINAMATH_CALUDE_office_count_l120_12053
