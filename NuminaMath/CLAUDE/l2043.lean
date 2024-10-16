import Mathlib

namespace NUMINAMATH_CALUDE_heidi_painting_rate_l2043_204386

theorem heidi_painting_rate (total_time minutes : ℕ) (fraction : ℚ) : 
  (total_time = 30) → (minutes = 10) → (fraction = 1 / 3) →
  (fraction = (minutes : ℚ) / total_time) :=
by sorry

end NUMINAMATH_CALUDE_heidi_painting_rate_l2043_204386


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2043_204383

/-- Hyperbola C with equation x^2 - 4y^2 = 1 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 - 4*p.2^2 = 1}

/-- The asymptotes of hyperbola C -/
def asymptotes : Set (ℝ × ℝ) := {p | p.1 + 2*p.2 = 0 ∨ p.1 - 2*p.2 = 0}

/-- The imaginary axis length of hyperbola C -/
def imaginary_axis_length : ℝ := 1

/-- Theorem: The asymptotes and imaginary axis length of hyperbola C -/
theorem hyperbola_properties :
  (∀ p ∈ C, p ∈ asymptotes ↔ p.1^2 = 4*p.2^2) ∧
  imaginary_axis_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2043_204383


namespace NUMINAMATH_CALUDE_a_less_than_two_necessary_not_sufficient_l2043_204319

/-- A quadratic equation x^2 + ax + 1 = 0 with real coefficient a -/
def quadratic_equation (a : ℝ) (x : ℂ) : Prop :=
  x^2 + a*x + 1 = 0

/-- The equation has complex roots -/
def has_complex_roots (a : ℝ) : Prop :=
  ∃ x : ℂ, quadratic_equation a x ∧ x.im ≠ 0

theorem a_less_than_two_necessary_not_sufficient :
  (∀ a : ℝ, has_complex_roots a → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ ¬has_complex_roots a) :=
sorry

end NUMINAMATH_CALUDE_a_less_than_two_necessary_not_sufficient_l2043_204319


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l2043_204328

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 3 / 8)
  (h2 : material2 = 1 / 3)
  (h3 : leftover = 15 / 40) :
  material1 + material2 - leftover = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l2043_204328


namespace NUMINAMATH_CALUDE_beans_to_seitan_ratio_l2043_204350

/-- Represents the number of dishes with a specific protein combination -/
structure DishCount where
  total : ℕ
  beansAndLentils : ℕ
  beansAndSeitan : ℕ
  withLentils : ℕ
  onlyBeans : ℕ
  onlySeitan : ℕ

/-- The conditions of the problem -/
def restaurantMenu : DishCount where
  total := 10
  beansAndLentils := 2
  beansAndSeitan := 2
  withLentils := 4
  onlyBeans := 2
  onlySeitan := 2

/-- The theorem to prove -/
theorem beans_to_seitan_ratio (menu : DishCount) 
  (h1 : menu.total = 10)
  (h2 : menu.beansAndLentils = 2)
  (h3 : menu.beansAndSeitan = 2)
  (h4 : menu.withLentils = 4)
  (h5 : menu.onlyBeans + menu.onlySeitan = menu.total - menu.beansAndLentils - menu.beansAndSeitan - (menu.withLentils - menu.beansAndLentils))
  (h6 : menu.onlyBeans = menu.onlySeitan) :
  menu.onlyBeans = menu.onlySeitan := by
  sorry

end NUMINAMATH_CALUDE_beans_to_seitan_ratio_l2043_204350


namespace NUMINAMATH_CALUDE_aquarium_cost_is_63_l2043_204310

/-- The total cost of an aquarium after markdown and sales tax --/
def aquarium_total_cost (original_price : ℝ) (markdown_percent : ℝ) (tax_percent : ℝ) : ℝ :=
  let reduced_price := original_price * (1 - markdown_percent)
  let sales_tax := reduced_price * tax_percent
  reduced_price + sales_tax

/-- Theorem stating that the total cost of the aquarium is $63 --/
theorem aquarium_cost_is_63 :
  aquarium_total_cost 120 0.5 0.05 = 63 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_cost_is_63_l2043_204310


namespace NUMINAMATH_CALUDE_system_consistency_solution_values_l2043_204331

def is_consistent (a : ℝ) : Prop :=
  ∃ x : ℝ, 3 * x^2 - x - a - 10 = 0 ∧ (a + 4) * x + a + 12 = 0

theorem system_consistency :
  ∀ a : ℝ, is_consistent a ↔ (a = -10 ∨ a = -8 ∨ a = 4) :=
sorry

theorem solution_values :
  (is_consistent (-10) ∧ ∃ x : ℝ, 3 * x^2 - x - (-10) - 10 = 0 ∧ (-10 + 4) * x + (-10) + 12 = 0 ∧ x = -1/3) ∧
  (is_consistent (-8) ∧ ∃ x : ℝ, 3 * x^2 - x - (-8) - 10 = 0 ∧ (-8 + 4) * x + (-8) + 12 = 0 ∧ x = -1) ∧
  (is_consistent 4 ∧ ∃ x : ℝ, 3 * x^2 - x - 4 - 10 = 0 ∧ (4 + 4) * x + 4 + 12 = 0 ∧ x = -2) :=
sorry

end NUMINAMATH_CALUDE_system_consistency_solution_values_l2043_204331


namespace NUMINAMATH_CALUDE_white_tiger_number_count_l2043_204304

/-- A function that returns true if a number is a multiple of 6 -/
def isMultipleOf6 (n : ℕ) : Bool :=
  n % 6 = 0

/-- A function that calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- A function that returns true if a number is a "White Tiger number" -/
def isWhiteTigerNumber (n : ℕ) : Bool :=
  isMultipleOf6 n ∧ sumOfDigits n = 6

/-- The count of "White Tiger numbers" up to 2022 -/
def whiteTigerNumberCount : ℕ :=
  (List.range 2023).filter isWhiteTigerNumber |>.length

theorem white_tiger_number_count : whiteTigerNumberCount = 30 := by
  sorry

end NUMINAMATH_CALUDE_white_tiger_number_count_l2043_204304


namespace NUMINAMATH_CALUDE_king_then_ace_probability_l2043_204357

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of Kings in a standard deck -/
def numKings : ℕ := 4

/-- The number of Aces in a standard deck -/
def numAces : ℕ := 4

/-- The probability of drawing a King first and an Ace second from a standard deck -/
def probKingThenAce : ℚ := (numKings : ℚ) / standardDeckSize * numAces / (standardDeckSize - 1)

theorem king_then_ace_probability :
  probKingThenAce = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_king_then_ace_probability_l2043_204357


namespace NUMINAMATH_CALUDE_complex_magnitude_l2043_204376

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2043_204376


namespace NUMINAMATH_CALUDE_central_region_perimeter_l2043_204309

/-- The perimeter of the central region formed by four identical circles in a square formation --/
theorem central_region_perimeter (c : ℝ) (h : c = 48) : 
  let r := c / (2 * Real.pi)
  4 * (Real.pi * r / 2) = c :=
by sorry

end NUMINAMATH_CALUDE_central_region_perimeter_l2043_204309


namespace NUMINAMATH_CALUDE_polynomial_identity_l2043_204302

theorem polynomial_identity (x y : ℝ) (h : x - y = 1) :
  x^4 - x*y^3 - x^3*y - 3*x^2*y + 3*x*y^2 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2043_204302


namespace NUMINAMATH_CALUDE_alyssa_fruit_spending_l2043_204397

theorem alyssa_fruit_spending (total_spent cherries_cost : ℚ)
  (h1 : total_spent = 21.93)
  (h2 : cherries_cost = 9.85) :
  total_spent - cherries_cost = 12.08 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_fruit_spending_l2043_204397


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2043_204370

def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 2}

theorem union_of_A_and_B : A ∪ B = {x | -3 < x ∧ x < 6} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2043_204370


namespace NUMINAMATH_CALUDE_normal_level_short_gallons_needed_after_evaporation_l2043_204349

/-- Represents a water reservoir with given properties -/
structure Reservoir where
  current_level : ℝ
  normal_level : ℝ
  total_capacity : ℝ
  evaporation_rate : ℝ
  current_is_twice_normal : current_level = 2 * normal_level
  current_is_75_percent : current_level = 0.75 * total_capacity
  h_current_level : current_level = 30
  h_evaporation_rate : evaporation_rate = 0.1

/-- The normal level is 25 million gallons short of total capacity -/
theorem normal_level_short (r : Reservoir) :
  r.total_capacity - r.normal_level = 25 :=
sorry

/-- After evaporation, 13 million gallons are needed to reach total capacity -/
theorem gallons_needed_after_evaporation (r : Reservoir) :
  r.total_capacity - (r.current_level - r.evaporation_rate * r.current_level) = 13 :=
sorry

end NUMINAMATH_CALUDE_normal_level_short_gallons_needed_after_evaporation_l2043_204349


namespace NUMINAMATH_CALUDE_subtraction_and_simplification_l2043_204305

theorem subtraction_and_simplification : (8 : ℚ) / 23 - (5 : ℚ) / 46 = (11 : ℚ) / 46 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_and_simplification_l2043_204305


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2043_204392

theorem trigonometric_identities :
  (∃ (tan25 tan35 : ℝ),
    tan25 = Real.tan (25 * π / 180) ∧
    tan35 = Real.tan (35 * π / 180) ∧
    tan25 + tan35 + Real.sqrt 3 * tan25 * tan35 = Real.sqrt 3) ∧
  (Real.sin (10 * π / 180))⁻¹ - (Real.sqrt 3) * (Real.cos (10 * π / 180))⁻¹ = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2043_204392


namespace NUMINAMATH_CALUDE_area_between_parabola_and_line_l2043_204313

theorem area_between_parabola_and_line : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := x
  let area := ∫ x in (0:ℝ)..1, (g x - f x)
  area = 1/6 := by sorry

end NUMINAMATH_CALUDE_area_between_parabola_and_line_l2043_204313


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2043_204395

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 2)
  (h2 : a.1 * b.1 + a.2 * b.2 = 1)
  (h3 : angle_between a b = π / 3) :
  Real.sqrt (b.1^2 + b.2^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2043_204395


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2043_204339

/-- The complex number i -/
noncomputable def i : ℂ := Complex.I

/-- Proof that (1+i)/(1-i) = i -/
theorem complex_fraction_simplification : (1 + i) / (1 - i) = i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2043_204339


namespace NUMINAMATH_CALUDE_water_consumption_l2043_204379

theorem water_consumption (initial_water : ℚ) : 
  initial_water > 0 →
  let remaining_day1 := initial_water / 2
  let remaining_day2 := remaining_day1 * 2 / 3
  let remaining_day3 := remaining_day2 / 2
  remaining_day3 = 250 →
  initial_water = 1500 := by
sorry

end NUMINAMATH_CALUDE_water_consumption_l2043_204379


namespace NUMINAMATH_CALUDE_twins_ratios_l2043_204334

/-- Represents the family composition before and after the birth of twins -/
structure Family where
  initial_boys : ℕ
  initial_girls : ℕ
  k : ℚ
  t : ℚ

/-- The ratio of brothers to sisters for boys after the birth of twins -/
def boys_ratio (f : Family) : ℚ :=
  f.initial_boys / (f.initial_girls + 1)

/-- The ratio of brothers to sisters for girls after the birth of twins -/
def girls_ratio (f : Family) : ℚ :=
  (f.initial_boys + 1) / f.initial_girls

/-- Theorem stating the ratios after the birth of twins -/
theorem twins_ratios (f : Family) 
  (h1 : (f.initial_boys + 2) / f.initial_girls = f.k)
  (h2 : f.initial_boys / (f.initial_girls + 2) = f.t) :
  boys_ratio f = f.t ∧ girls_ratio f = f.k := by
  sorry

#check twins_ratios

end NUMINAMATH_CALUDE_twins_ratios_l2043_204334


namespace NUMINAMATH_CALUDE_valid_sets_count_l2043_204325

/-- Represents a family tree with 4 generations -/
structure FamilyTree :=
  (root : Unit)
  (gen1 : Fin 3)
  (gen2 : Fin 6)
  (gen3 : Fin 6)

/-- Represents a set of women from the family tree -/
def WomenSet := FamilyTree → Bool

/-- Checks if a set is valid (no woman and her daughter are both in the set) -/
def is_valid_set (s : WomenSet) : Bool :=
  sorry

/-- Counts the number of valid sets -/
def count_valid_sets : Nat :=
  sorry

/-- The main theorem to prove -/
theorem valid_sets_count : count_valid_sets = 793 :=
  sorry

end NUMINAMATH_CALUDE_valid_sets_count_l2043_204325


namespace NUMINAMATH_CALUDE_soup_can_price_l2043_204385

/-- Calculates the normal price of a can of soup given a "buy 1 get one free" offer -/
theorem soup_can_price (total_cans : ℕ) (total_paid : ℚ) : 
  total_cans > 0 → total_paid > 0 → (total_paid / (total_cans / 2 : ℚ) = 0.60) := by
  sorry

end NUMINAMATH_CALUDE_soup_can_price_l2043_204385


namespace NUMINAMATH_CALUDE_domain_union_sqrt_ln_l2043_204337

-- Define the domains M and N
def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x < 1}

-- State the theorem
theorem domain_union_sqrt_ln :
  M ∪ N = Set.Iio 1 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_domain_union_sqrt_ln_l2043_204337


namespace NUMINAMATH_CALUDE_average_distance_and_monthly_expense_l2043_204329

/-- Represents the daily distance traveled relative to the standard distance -/
def daily_distances : List ℤ := [-8, -11, -14, 0, -16, 41, 8]

/-- The standard distance in kilometers -/
def standard_distance : ℕ := 50

/-- Gasoline consumption in liters per 100 km -/
def gasoline_consumption : ℚ := 6 / 100

/-- Gasoline price in yuan per liter -/
def gasoline_price : ℚ := 77 / 10

/-- Number of days in a month -/
def days_in_month : ℕ := 30

theorem average_distance_and_monthly_expense :
  let avg_distance := standard_distance + (daily_distances.sum / daily_distances.length : ℚ)
  let monthly_expense := (days_in_month : ℚ) * avg_distance * gasoline_consumption * gasoline_price
  avg_distance = standard_distance ∧ monthly_expense = 693 := by
  sorry

end NUMINAMATH_CALUDE_average_distance_and_monthly_expense_l2043_204329


namespace NUMINAMATH_CALUDE_f_derivative_neg_one_l2043_204340

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Theorem statement
theorem f_derivative_neg_one (a b c : ℝ) : 
  f' a b 1 = 2 → f' a b (-1) = -2 := by
sorry

end NUMINAMATH_CALUDE_f_derivative_neg_one_l2043_204340


namespace NUMINAMATH_CALUDE_beach_ball_surface_area_l2043_204360

theorem beach_ball_surface_area 
  (circumference : ℝ) 
  (valve_diameter : ℝ) 
  (h1 : circumference = 33) 
  (h2 : valve_diameter = 1) : 
  let radius := circumference / (2 * Real.pi)
  let total_surface_area := 4 * Real.pi * radius^2
  let valve_area := Real.pi * (valve_diameter / 2)^2
  total_surface_area - valve_area = (4356 - Real.pi^2) / (4 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_beach_ball_surface_area_l2043_204360


namespace NUMINAMATH_CALUDE_gcd_2728_1575_l2043_204374

theorem gcd_2728_1575 : Nat.gcd 2728 1575 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2728_1575_l2043_204374


namespace NUMINAMATH_CALUDE_sequence_zero_at_201_l2043_204356

/-- Sequence defined by the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 134
  | 1 => 150
  | (k + 2) => a k - (k + 1) / a (k + 1)

/-- The smallest positive integer n for which a_n = 0 -/
def n : ℕ := 201

/-- Theorem stating that a_n = 0 and n is the smallest such positive integer -/
theorem sequence_zero_at_201 :
  a n = 0 ∧ ∀ m : ℕ, m > 0 ∧ m < n → a m ≠ 0 := by sorry

end NUMINAMATH_CALUDE_sequence_zero_at_201_l2043_204356


namespace NUMINAMATH_CALUDE_height_survey_is_census_l2043_204371

/-- Represents a survey method --/
inductive SurveyMethod
| HeightOfStudents
| CarCrashResistance
| TVViewership
| ShoeSoleDurability

/-- Defines the properties of a census --/
structure Census where
  collectsAllData : Bool
  isFeasible : Bool

/-- Determines if a survey method is suitable for a census --/
def isSuitableForCensus (method : SurveyMethod) : Prop :=
  ∃ (c : Census), c.collectsAllData ∧ c.isFeasible

/-- The main theorem stating that measuring the height of all students is suitable for a census --/
theorem height_survey_is_census : isSuitableForCensus SurveyMethod.HeightOfStudents :=
  sorry

end NUMINAMATH_CALUDE_height_survey_is_census_l2043_204371


namespace NUMINAMATH_CALUDE_acute_triangle_inequality_l2043_204301

/-- For any acute triangle ABC with side lengths a, b, c and angles A, B, C,
    the inequality 4abc < (a^2 + b^2 + c^2)(a cos A + b cos B + c cos C) ≤ 9/2 abc holds. -/
theorem acute_triangle_inequality (a b c : ℝ) (A B C : Real) 
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
    (h_acute : A > 0 ∧ B > 0 ∧ C > 0)
    (h_angles : A + B + C = π)
    (h_cosine_law : a^2 = b^2 + c^2 - 2*b*c*Real.cos A ∧
                    b^2 = a^2 + c^2 - 2*a*c*Real.cos B ∧
                    c^2 = a^2 + b^2 - 2*a*b*Real.cos C) :
  4*a*b*c < (a^2 + b^2 + c^2)*(a*Real.cos A + b*Real.cos B + c*Real.cos C) ∧
  (a^2 + b^2 + c^2)*(a*Real.cos A + b*Real.cos B + c*Real.cos C) ≤ 9/2*a*b*c :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_inequality_l2043_204301


namespace NUMINAMATH_CALUDE_initial_girls_count_l2043_204354

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 18) = b) →
  (4 * (b - 36) = g - 18) →
  g = 31 :=
by sorry

end NUMINAMATH_CALUDE_initial_girls_count_l2043_204354


namespace NUMINAMATH_CALUDE_fountain_distance_is_30_l2043_204348

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def fountain_distance (total_distance : ℕ) (num_trips : ℕ) : ℚ :=
  (total_distance : ℚ) / (2 * num_trips)

/-- Theorem stating that the fountain distance is 30 feet -/
theorem fountain_distance_is_30 :
  fountain_distance 120 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fountain_distance_is_30_l2043_204348


namespace NUMINAMATH_CALUDE_simplify_expression_l2043_204393

theorem simplify_expression (y : ℝ) : 3*y + 4*y^2 - 2 - (8 - 3*y - 4*y^2) = 8*y^2 + 6*y - 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2043_204393


namespace NUMINAMATH_CALUDE_distinct_digit_count_is_4032_l2043_204388

/-- The number of integers between 2000 and 9999 with four distinct digits -/
def distinct_digit_count : ℕ := sorry

/-- The range of possible first digits -/
def first_digit_range : List ℕ := List.range 8

/-- The range of possible second digits (including 0) -/
def second_digit_range : List ℕ := List.range 10

/-- The range of possible third digits -/
def third_digit_range : List ℕ := List.range 10

/-- The range of possible fourth digits -/
def fourth_digit_range : List ℕ := List.range 10

theorem distinct_digit_count_is_4032 :
  distinct_digit_count = first_digit_range.length *
                         (second_digit_range.length - 1) *
                         (third_digit_range.length - 2) *
                         (fourth_digit_range.length - 3) :=
by sorry

end NUMINAMATH_CALUDE_distinct_digit_count_is_4032_l2043_204388


namespace NUMINAMATH_CALUDE_stating_transportation_equation_correct_l2043_204389

/-- Represents the rate at which Vehicle A transports goods per day -/
def vehicle_a_rate : ℚ := 1/4

/-- Represents the time Vehicle A works alone -/
def vehicle_a_solo_time : ℚ := 1

/-- Represents the time both vehicles work together -/
def combined_work_time : ℚ := 1/2

/-- Represents the total amount of goods (100%) -/
def total_goods : ℚ := 1

/-- 
Theorem stating that the equation correctly represents the transportation situation
given the conditions of the problem
-/
theorem transportation_equation_correct (x : ℚ) : 
  vehicle_a_rate * vehicle_a_solo_time + 
  combined_work_time * (vehicle_a_rate + 1/x) = total_goods := by
  sorry

end NUMINAMATH_CALUDE_stating_transportation_equation_correct_l2043_204389


namespace NUMINAMATH_CALUDE_smallest_prime_q_l2043_204363

theorem smallest_prime_q (p : ℕ) : 
  Prime p → Prime (13 * p + 2) → (13 * p + 2) ≥ 41 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_q_l2043_204363


namespace NUMINAMATH_CALUDE_total_distance_is_thirteen_l2043_204327

/-- Represents the time in minutes to travel one mile on a given day -/
def time_per_mile (day : Nat) : Nat :=
  12 + 6 * day

/-- Represents the distance traveled in miles on a given day -/
def distance (day : Nat) : Nat :=
  60 / (time_per_mile day)

/-- The total distance traveled over five days -/
def total_distance : Nat :=
  (List.range 5).map distance |>.sum

theorem total_distance_is_thirteen :
  total_distance = 13 := by sorry

end NUMINAMATH_CALUDE_total_distance_is_thirteen_l2043_204327


namespace NUMINAMATH_CALUDE_notebooks_promotion_result_l2043_204330

/-- Calculates the maximum number of notebooks obtainable given an initial amount of money,
    the cost per notebook, and a promotion where stickers can be exchanged for free notebooks. -/
def max_notebooks (initial_money : ℕ) (cost_per_notebook : ℕ) (stickers_per_free_notebook : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given 150 rubles, with notebooks costing 4 rubles each,
    and a promotion where 5 stickers can be exchanged for an additional notebook
    (each notebook comes with a sticker), the maximum number of notebooks obtainable is 46. -/
theorem notebooks_promotion_result :
  max_notebooks 150 4 5 = 46 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_promotion_result_l2043_204330


namespace NUMINAMATH_CALUDE_max_sum_constrained_squares_l2043_204396

theorem max_sum_constrained_squares (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m^2 + n^2 = 100) :
  m + n ≤ 10 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_sum_constrained_squares_l2043_204396


namespace NUMINAMATH_CALUDE_average_of_quadratic_roots_l2043_204320

theorem average_of_quadratic_roots (p q : ℝ) (h : p ≠ 0) :
  let f : ℝ → ℝ := λ x => 3 * p * x^2 - 6 * p * x + q
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) → (x₁ + x₂) / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_of_quadratic_roots_l2043_204320


namespace NUMINAMATH_CALUDE_total_savings_together_l2043_204362

/-- Regular price of a window -/
def regular_price : ℝ := 120

/-- Calculate the number of windows to pay for given the number of windows bought -/
def windows_to_pay_for (n : ℕ) : ℕ :=
  n - n / 6

/-- Calculate the price with the special deal (free 6th window) -/
def price_with_deal (n : ℕ) : ℝ :=
  (windows_to_pay_for n : ℝ) * regular_price

/-- Apply the additional 5% discount for purchases over 10 windows -/
def apply_discount (price : ℝ) (n : ℕ) : ℝ :=
  if n > 10 then price * 0.95 else price

/-- Calculate the final price after all discounts -/
def final_price (n : ℕ) : ℝ :=
  apply_discount (price_with_deal n) n

/-- Nina's number of windows -/
def nina_windows : ℕ := 9

/-- Carl's number of windows -/
def carl_windows : ℕ := 11

/-- Theorem: The total savings when Nina and Carl buy windows together is $348 -/
theorem total_savings_together : 
  (nina_windows + carl_windows) * regular_price - final_price (nina_windows + carl_windows) = 348 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_together_l2043_204362


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2043_204343

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2043_204343


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l2043_204365

theorem smallest_number_of_eggs (total_eggs : ℕ) (containers : ℕ) : 
  total_eggs > 130 →
  total_eggs = 15 * containers - 3 →
  (∀ n : ℕ, n > 130 ∧ ∃ m : ℕ, n = 15 * m - 3 → n ≥ total_eggs) →
  total_eggs = 132 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l2043_204365


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2043_204303

theorem triangle_angle_sum (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 8) :
  let θ := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  (π - θ) * (180 / π) = 120 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2043_204303


namespace NUMINAMATH_CALUDE_unique_pie_solution_l2043_204318

/-- Represents the number of pies of each type -/
structure PieCount where
  raspberry : ℕ
  blueberry : ℕ
  strawberry : ℕ

/-- Checks if the given pie counts satisfy the problem conditions -/
def satisfiesConditions (pies : PieCount) : Prop :=
  pies.raspberry = (pies.raspberry + pies.blueberry + pies.strawberry) / 2 ∧
  pies.blueberry = pies.raspberry - 14 ∧
  pies.strawberry = (pies.raspberry + pies.blueberry) / 2

/-- Theorem stating that the given pie counts are the unique solution -/
theorem unique_pie_solution :
  ∃! (pies : PieCount), satisfiesConditions pies ∧
    pies.raspberry = 21 ∧ pies.blueberry = 7 ∧ pies.strawberry = 14 := by
  sorry

end NUMINAMATH_CALUDE_unique_pie_solution_l2043_204318


namespace NUMINAMATH_CALUDE_f_is_2x_plus_7_range_f_is_5_to_13_l2043_204308

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being a first-degree function
def is_first_degree (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

-- Define the given condition for f
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17

-- Theorem for the first part
theorem f_is_2x_plus_7 (h1 : is_first_degree f) (h2 : satisfies_condition f) :
  ∀ x, f x = 2 * x + 7 :=
sorry

-- Define the range of f for x ∈ (-1, 3]
def range_f : Set ℝ := { y | ∃ x ∈ Set.Ioc (-1) 3, f x = y }

-- Theorem for the second part
theorem range_f_is_5_to_13 (h1 : is_first_degree f) (h2 : satisfies_condition f) :
  range_f = Set.Ioc 5 13 :=
sorry

end NUMINAMATH_CALUDE_f_is_2x_plus_7_range_f_is_5_to_13_l2043_204308


namespace NUMINAMATH_CALUDE_flower_count_theorem_l2043_204359

theorem flower_count_theorem (roses tulips lilies : ℕ) : 
  roses = 58 ∧ 
  roses = tulips + 15 ∧ 
  roses = lilies - 25 → 
  roses + tulips + lilies = 184 := by
sorry

end NUMINAMATH_CALUDE_flower_count_theorem_l2043_204359


namespace NUMINAMATH_CALUDE_union_when_k_neg_one_intersection_equality_iff_k_range_l2043_204351

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

-- Theorem for part I
theorem union_when_k_neg_one :
  A ∪ B (-1) = {x : ℝ | -1 < x ∧ x < 3} := by sorry

-- Theorem for part II
theorem intersection_equality_iff_k_range :
  ∀ k : ℝ, A ∩ B k = B k ↔ k ∈ Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_union_when_k_neg_one_intersection_equality_iff_k_range_l2043_204351


namespace NUMINAMATH_CALUDE_muffin_cost_is_correct_l2043_204346

/-- The cost of a muffin given the total cost and the cost of juice -/
def muffin_cost (total_cost juice_cost : ℚ) : ℚ :=
  (total_cost - juice_cost) / 3

theorem muffin_cost_is_correct (total_cost juice_cost : ℚ) 
  (h1 : total_cost = 370/100) 
  (h2 : juice_cost = 145/100) : 
  muffin_cost total_cost juice_cost = 75/100 := by
  sorry

#eval muffin_cost (370/100) (145/100)

end NUMINAMATH_CALUDE_muffin_cost_is_correct_l2043_204346


namespace NUMINAMATH_CALUDE_parabola_point_order_l2043_204323

/-- Given a parabola y = 2x² - 4x + m and three points on it, prove their y-coordinates are in ascending order -/
theorem parabola_point_order (m : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 2 * 3^2 - 4 * 3 + m)
  (h₂ : y₂ = 2 * 4^2 - 4 * 4 + m)
  (h₃ : y₃ = 2 * 5^2 - 4 * 5 + m) :
  y₁ < y₂ ∧ y₂ < y₃ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_order_l2043_204323


namespace NUMINAMATH_CALUDE_julie_age_is_fifteen_l2043_204338

/-- Represents Julie's age and earnings during a four-month period --/
structure JulieData where
  hoursPerDay : ℕ
  hourlyRatePerAge : ℚ
  workDays : ℕ
  totalEarnings : ℚ

/-- Calculates Julie's age at the end of the four-month period --/
def calculateAge (data : JulieData) : ℕ :=
  sorry

/-- Theorem stating that Julie's age at the end of the period is 15 --/
theorem julie_age_is_fifteen (data : JulieData) 
  (h1 : data.hoursPerDay = 3)
  (h2 : data.hourlyRatePerAge = 3/4)
  (h3 : data.workDays = 60)
  (h4 : data.totalEarnings = 810) :
  calculateAge data = 15 := by
  sorry

end NUMINAMATH_CALUDE_julie_age_is_fifteen_l2043_204338


namespace NUMINAMATH_CALUDE_greg_earnings_l2043_204387

/-- Calculates the earnings for dog walking based on the given parameters -/
def dog_walking_earnings (base_charge : ℕ) (per_minute_charge : ℕ) 
  (dogs : List (ℕ × ℕ)) : ℕ :=
  dogs.foldl (fun acc (num_dogs, minutes) => 
    acc + num_dogs * base_charge + num_dogs * minutes * per_minute_charge) 0

/-- Theorem stating Greg's earnings from his dog walking business -/
theorem greg_earnings : 
  dog_walking_earnings 20 1 [(1, 10), (2, 7), (3, 9)] = 171 := by
  sorry

#eval dog_walking_earnings 20 1 [(1, 10), (2, 7), (3, 9)]

end NUMINAMATH_CALUDE_greg_earnings_l2043_204387


namespace NUMINAMATH_CALUDE_sum_interior_angles_octagon_l2043_204372

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The sum of the interior angles of an octagon is 1080 degrees -/
theorem sum_interior_angles_octagon :
  sum_interior_angles octagon_sides = 1080 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_octagon_l2043_204372


namespace NUMINAMATH_CALUDE_lateral_angle_cosine_l2043_204377

/-- A regular triangular pyramid with an inscribed sphere -/
structure RegularPyramid where
  -- The ratio of the intersection point on an edge
  intersectionRatio : ℝ
  -- Assumption that the pyramid is regular and has an inscribed sphere
  regular : Bool
  hasInscribedSphere : Bool

/-- The angle between a lateral face and the base plane of the pyramid -/
def lateralAngle (p : RegularPyramid) : ℝ := sorry

/-- Main theorem: The cosine of the lateral angle is 7/10 -/
theorem lateral_angle_cosine (p : RegularPyramid) 
  (h1 : p.intersectionRatio = 1.55)
  (h2 : p.regular = true)
  (h3 : p.hasInscribedSphere = true) : 
  Real.cos (lateralAngle p) = 7/10 := by sorry

end NUMINAMATH_CALUDE_lateral_angle_cosine_l2043_204377


namespace NUMINAMATH_CALUDE_smallest_number_l2043_204390

theorem smallest_number (a b c d e : ℝ) 
  (ha : a = 0.997) 
  (hb : b = 0.979) 
  (hc : c = 0.999) 
  (hd : d = 0.9797) 
  (he : e = 0.9709) : 
  e ≤ a ∧ e ≤ b ∧ e ≤ c ∧ e ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2043_204390


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2043_204311

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x y : ℝ, x^2 + y^2 - 1 > 0)) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2043_204311


namespace NUMINAMATH_CALUDE_hyperbola_parabola_shared_focus_l2043_204399

/-- The focus of a parabola y² = 8x is at (2, 0) -/
def parabola_focus : ℝ × ℝ := (2, 0)

/-- The equation of the hyperbola (x²/a² - y²/3 = 1) with a > 0 -/
def is_hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ x^2 / a^2 - y^2 / 3 = 1

/-- The focus of the hyperbola coincides with the focus of the parabola -/
def hyperbola_focus (a : ℝ) : ℝ × ℝ := parabola_focus

/-- Theorem: The value of 'a' for the hyperbola sharing a focus with the parabola is 1 -/
theorem hyperbola_parabola_shared_focus :
  ∃ (a : ℝ), is_hyperbola a (hyperbola_focus a).1 (hyperbola_focus a).2 ∧ a = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_shared_focus_l2043_204399


namespace NUMINAMATH_CALUDE_two_digit_division_problem_l2043_204344

theorem two_digit_division_problem :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 
  (∃ q r : ℕ, q = 9 ∧ r = 6 ∧ n = q * (n % 10) + r) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_division_problem_l2043_204344


namespace NUMINAMATH_CALUDE_parallelogram_roots_l2043_204336

theorem parallelogram_roots (b : ℝ) : 
  (∃ (z₁ z₂ z₃ z₄ : ℂ), 
    z₁^4 - 8*z₁^3 + 13*b*z₁^2 - 5*(2*b^2 + b - 2)*z₁ + 4 = 0 ∧
    z₂^4 - 8*z₂^3 + 13*b*z₂^2 - 5*(2*b^2 + b - 2)*z₂ + 4 = 0 ∧
    z₃^4 - 8*z₃^3 + 13*b*z₃^2 - 5*(2*b^2 + b - 2)*z₃ + 4 = 0 ∧
    z₄^4 - 8*z₄^3 + 13*b*z₄^2 - 5*(2*b^2 + b - 2)*z₄ + 4 = 0 ∧
    z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    (z₁ - z₂ = z₄ - z₃) ∧ (z₁ - z₃ = z₄ - z₂)) ↔ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_roots_l2043_204336


namespace NUMINAMATH_CALUDE_remainder_3012_div_97_l2043_204364

theorem remainder_3012_div_97 : 3012 % 97 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3012_div_97_l2043_204364


namespace NUMINAMATH_CALUDE_circular_pool_volume_l2043_204341

/-- The volume of a circular cylinder with diameter 80 feet and height 10 feet is approximately 50265.6 cubic feet. -/
theorem circular_pool_volume :
  let diameter : ℝ := 80
  let height : ℝ := 10
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * height
  ∃ ε > 0, |volume - 50265.6| < ε :=
by sorry

end NUMINAMATH_CALUDE_circular_pool_volume_l2043_204341


namespace NUMINAMATH_CALUDE_large_monkey_doll_cost_l2043_204315

def total_spent : ℝ := 300

def small_doll_discount : ℝ := 2

theorem large_monkey_doll_cost (large_cost : ℝ) 
  (h1 : large_cost > 0)
  (h2 : total_spent / (large_cost - small_doll_discount) = total_spent / large_cost + 25) :
  large_cost = 6 := by
sorry

end NUMINAMATH_CALUDE_large_monkey_doll_cost_l2043_204315


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_35_l2043_204369

theorem inverse_of_3_mod_35 : ∃ x : ℕ, x < 35 ∧ (3 * x) % 35 = 1 :=
by
  use 12
  sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_35_l2043_204369


namespace NUMINAMATH_CALUDE_inequality_proof_l2043_204312

theorem inequality_proof (u v w : ℝ) 
  (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h_sum : u + v + w + Real.sqrt (u * v * w) = 4) :
  Real.sqrt (u * v / w) + Real.sqrt (v * w / u) + Real.sqrt (w * u / v) ≥ u + v + w :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2043_204312


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2043_204394

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2043_204394


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l2043_204353

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : average_children = 3)
  (h3 : childless_families = 3)
  : (total_families * average_children) / (total_families - childless_families) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l2043_204353


namespace NUMINAMATH_CALUDE_factorization_equality_l2043_204378

theorem factorization_equality (x y : ℝ) : 2 * x^2 - 4 * x * y = 2 * x * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2043_204378


namespace NUMINAMATH_CALUDE_largest_n_unique_k_l2043_204384

theorem largest_n_unique_k : ∃ (n : ℕ), n > 0 ∧ n ≤ 27 ∧
  (∃! (k : ℤ), (9:ℚ)/17 < (n:ℚ)/(n+k) ∧ (n:ℚ)/(n+k) < 8/15) ∧
  (∀ (m : ℕ), m > 27 → ¬(∃! (k : ℤ), (9:ℚ)/17 < (m:ℚ)/(m+k) ∧ (m:ℚ)/(m+k) < 8/15)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_unique_k_l2043_204384


namespace NUMINAMATH_CALUDE_rational_function_value_at_one_l2043_204300

/-- A rational function with specific properties --/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_quadratic : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  asymptote_minus_three : q (-3) = 0
  asymptote_two : q 2 = 0
  passes_origin : p 0 = 0 ∧ q 0 ≠ 0
  passes_one_two : p 1 = 2 * q 1 ∧ q 1 ≠ 0

/-- The main theorem --/
theorem rational_function_value_at_one (f : RationalFunction) : f.p 1 / f.q 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_at_one_l2043_204300


namespace NUMINAMATH_CALUDE_root_ordering_implies_a_range_l2043_204358

/-- Given two quadratic equations and an ordering of their roots, 
    prove the range of the coefficient a. -/
theorem root_ordering_implies_a_range 
  (a b : ℝ) 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + 1 = 0)
  (h₂ : a * x₂^2 + b * x₂ + 1 = 0)
  (h₃ : a^2 * x₃^2 + b * x₃ + 1 = 0)
  (h₄ : a^2 * x₄^2 + b * x₄ + 1 = 0)
  (h_order : x₃ < x₁ ∧ x₁ < x₂ ∧ x₂ < x₄) : 
  0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_root_ordering_implies_a_range_l2043_204358


namespace NUMINAMATH_CALUDE_officer_selection_count_l2043_204322

/-- The number of members in the club -/
def clubSize : ℕ := 12

/-- The number of officers to be elected -/
def officerCount : ℕ := 5

/-- Calculates the number of ways to select distinct officers from club members -/
def officerSelections (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else (List.range k).foldl (fun acc i => acc * (n - i)) 1

/-- Theorem stating the number of ways to select 5 distinct officers from 12 members -/
theorem officer_selection_count :
  officerSelections clubSize officerCount = 95040 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_count_l2043_204322


namespace NUMINAMATH_CALUDE_prob_red_is_three_tenths_l2043_204381

-- Define the contents of the bags
def bag_A : Finset (Fin 10) := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def bag_B : Finset (Fin 10) := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the colors of the balls
inductive Color
| Red
| White
| Black

-- Define the color distribution in bag A
def color_A : Fin 10 → Color
| 0 | 1 | 2 => Color.Red
| 3 | 4 => Color.White
| _ => Color.Black

-- Define the color distribution in bag B
def color_B : Fin 10 → Color
| 0 | 1 | 2 => Color.Red
| 3 | 4 | 5 => Color.White
| _ => Color.Black

-- Define the probability of drawing a red ball from bag B after transfer
def prob_red_after_transfer : ℚ :=
  3 / 10

-- Theorem statement
theorem prob_red_is_three_tenths :
  prob_red_after_transfer = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_prob_red_is_three_tenths_l2043_204381


namespace NUMINAMATH_CALUDE_apple_distribution_l2043_204335

theorem apple_distribution (total_apples : ℕ) (min_apples : ℕ) (num_people : ℕ) : 
  total_apples = 30 → min_apples = 3 → num_people = 3 →
  (Nat.choose (total_apples - num_people * min_apples + num_people - 1) (num_people - 1)) = 253 := by
sorry

end NUMINAMATH_CALUDE_apple_distribution_l2043_204335


namespace NUMINAMATH_CALUDE_final_price_is_12_l2043_204333

/-- The price of a set consisting of one cup of coffee and one piece of cheesecake,
    with a 25% discount when bought together. -/
def discounted_set_price (coffee_price cheesecake_price : ℚ) (discount_rate : ℚ) : ℚ :=
  (coffee_price + cheesecake_price) * (1 - discount_rate)

/-- Theorem stating that the final price of the set is $12 -/
theorem final_price_is_12 :
  discounted_set_price 6 10 (1/4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_12_l2043_204333


namespace NUMINAMATH_CALUDE_regression_line_change_l2043_204382

/-- Represents a linear regression equation of the form y = a + bx -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- Calculates the change in y when x increases by 1 unit -/
def change_in_y (line : RegressionLine) : ℝ := -line.b

/-- Theorem: For the given regression line, when x increases by 1 unit, y decreases by 1.5 units -/
theorem regression_line_change (line : RegressionLine) 
  (h1 : line.a = 2) 
  (h2 : line.b = -1.5) : 
  change_in_y line = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_change_l2043_204382


namespace NUMINAMATH_CALUDE_division_by_fraction_fifteen_divided_by_two_thirds_result_is_twentytwo_point_five_l2043_204326

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b :=
by sorry

theorem fifteen_divided_by_two_thirds :
  15 / (2 / 3) = 45 / 2 :=
by sorry

theorem result_is_twentytwo_point_five :
  15 / (2 / 3) = 22.5 :=
by sorry

end NUMINAMATH_CALUDE_division_by_fraction_fifteen_divided_by_two_thirds_result_is_twentytwo_point_five_l2043_204326


namespace NUMINAMATH_CALUDE_amusement_park_earnings_l2043_204373

/-- Calculates the total earnings of an amusement park for a week --/
theorem amusement_park_earnings 
  (ticket_price : ℕ)
  (weekday_visitors : ℕ)
  (saturday_visitors : ℕ)
  (sunday_visitors : ℕ) :
  ticket_price = 3 →
  weekday_visitors = 100 →
  saturday_visitors = 200 →
  sunday_visitors = 300 →
  (5 * weekday_visitors + saturday_visitors + sunday_visitors) * ticket_price = 3000 := by
  sorry

#check amusement_park_earnings

end NUMINAMATH_CALUDE_amusement_park_earnings_l2043_204373


namespace NUMINAMATH_CALUDE_continuous_additive_function_is_linear_l2043_204398

theorem continuous_additive_function_is_linear 
  (f : ℝ → ℝ) 
  (hf_continuous : Continuous f) 
  (hf_additive : ∀ x y : ℝ, f (x + y) = f x + f y) : 
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_continuous_additive_function_is_linear_l2043_204398


namespace NUMINAMATH_CALUDE_remainder_3012_div_96_l2043_204366

theorem remainder_3012_div_96 : 3012 % 96 = 36 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3012_div_96_l2043_204366


namespace NUMINAMATH_CALUDE_range_of_a_l2043_204391

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 ≥ a

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  prop_p a ∧ prop_q a ↔ a ∈ Set.Iic (-2) ∪ {1} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2043_204391


namespace NUMINAMATH_CALUDE_range_of_x_for_proposition_l2043_204367

theorem range_of_x_for_proposition (x : ℝ) : 
  (∃ a : ℝ, a ∈ Set.Icc 1 3 ∧ a * x^2 + (a - 2) * x - 2 > 0) ↔ 
  x < -1 ∨ x > 2/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_for_proposition_l2043_204367


namespace NUMINAMATH_CALUDE_cubic_root_product_l2043_204375

theorem cubic_root_product (a b c : ℝ) : 
  a^3 - 15*a^2 + 22*a - 8 = 0 →
  b^3 - 15*b^2 + 22*b - 8 = 0 →
  c^3 - 15*c^2 + 22*c - 8 = 0 →
  (2+a)*(2+b)*(2+c) = 120 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_product_l2043_204375


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2043_204345

theorem rationalize_denominator :
  ∃ (a b : ℝ), a + b * Real.sqrt 3 = -Real.sqrt 3 - 2 ∧ 
  (a + b * Real.sqrt 3) * (Real.sqrt 3 - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2043_204345


namespace NUMINAMATH_CALUDE_smallest_y_cube_sum_l2043_204347

theorem smallest_y_cube_sum (v w x y : ℕ+) : 
  v.val + 1 = w.val → w.val + 1 = x.val → x.val + 1 = y.val →
  v^3 + w^3 + x^3 = y^3 →
  ∀ (z : ℕ+), z < y → ¬(∃ (a b c : ℕ+), a.val + 1 = b.val ∧ b.val + 1 = c.val ∧ c.val + 1 = z.val ∧ a^3 + b^3 + c^3 = z^3) →
  y = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_y_cube_sum_l2043_204347


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_34_l2043_204368

theorem smallest_four_digit_divisible_by_34 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 34 ∣ n → n ≥ 1020 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_34_l2043_204368


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l2043_204324

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents one draw operation -/
def draw (state : UrnState) : UrnState :=
  if state.red / (state.red + state.blue) > state.blue / (state.red + state.blue)
  then UrnState.mk (state.red + 1) state.blue
  else UrnState.mk state.red (state.blue + 1)

/-- Performs n draw operations -/
def performDraws (n : ℕ) (initial : UrnState) : UrnState :=
  match n with
  | 0 => initial
  | n + 1 => draw (performDraws n initial)

/-- Calculates the probability of selecting a red ball n times in a row -/
def redProbability (n : ℕ) (initial : UrnState) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 
    let state := performDraws n initial
    (state.red : ℚ) / (state.red + state.blue) * redProbability n initial

/-- The main theorem to prove -/
theorem urn_probability_theorem :
  let initial := UrnState.mk 2 1
  let final := performDraws 5 initial
  final.red = 8 ∧ final.blue = 4 ∧ redProbability 5 initial = 2/7 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l2043_204324


namespace NUMINAMATH_CALUDE_roberto_outfits_l2043_204316

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits :
  let trousers : ℕ := 6
  let shirts : ℕ := 7
  let jackets : ℕ := 4
  let shoes : ℕ := 2
  number_of_outfits trousers shirts jackets shoes = 336 := by
  sorry


end NUMINAMATH_CALUDE_roberto_outfits_l2043_204316


namespace NUMINAMATH_CALUDE_remainder_2013_div_85_l2043_204380

theorem remainder_2013_div_85 : 2013 % 85 = 58 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2013_div_85_l2043_204380


namespace NUMINAMATH_CALUDE_candy_mixture_price_l2043_204307

theorem candy_mixture_price (total_mixture : ℝ) (mixture_price : ℝ) 
  (first_candy_amount : ℝ) (second_candy_amount : ℝ) (first_candy_price : ℝ) :
  total_mixture = 30 ∧
  mixture_price = 3 ∧
  first_candy_amount = 20 ∧
  second_candy_amount = 10 ∧
  first_candy_price = 2.95 →
  ∃ (second_candy_price : ℝ),
    second_candy_price = 3.10 ∧
    total_mixture * mixture_price = 
      first_candy_amount * first_candy_price + second_candy_amount * second_candy_price :=
by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_price_l2043_204307


namespace NUMINAMATH_CALUDE_population_scientific_notation_l2043_204342

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem population_scientific_notation :
  let population : ℝ := 1412.60 * 1000000
  let scientific_form := toScientificNotation population
  scientific_form.coefficient = 1.4126 ∧ scientific_form.exponent = 5 := by
  sorry

end NUMINAMATH_CALUDE_population_scientific_notation_l2043_204342


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l2043_204352

/-- A quadratic function satisfying given conditions -/
def f (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem stating the minimum value of the quadratic function -/
theorem quadratic_minimum_value
  (a b c : ℝ)
  (h1 : f a b c (-7) = -9)
  (h2 : f a b c (-5) = -4)
  (h3 : f a b c (-3) = -1)
  (h4 : f a b c (-1) = 0)
  (h5 : f a b c 1 = -1) :
  ∀ x ∈ Set.Icc (-7 : ℝ) 7, f a b c x ≥ -16 ∧ ∃ x₀ ∈ Set.Icc (-7 : ℝ) 7, f a b c x₀ = -16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l2043_204352


namespace NUMINAMATH_CALUDE_correction_amount_proof_l2043_204306

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | "penny" => 1
  | _ => 0

/-- Calculates the correction amount in cents given the counting errors -/
def correction_amount (y z w : ℕ) : ℤ :=
  15 * y + 4 * z - 15 * w

theorem correction_amount_proof (y z w : ℕ) :
  correction_amount y z w = 
    y * (coin_value "quarter" - coin_value "dime") +
    z * (coin_value "nickel" - coin_value "penny") -
    w * (coin_value "quarter" - coin_value "dime") :=
  sorry

end NUMINAMATH_CALUDE_correction_amount_proof_l2043_204306


namespace NUMINAMATH_CALUDE_intersection_sum_l2043_204355

/-- Given two equations and their intersection points, prove the sum of x-coordinates is 0 and the sum of y-coordinates is 3 -/
theorem intersection_sum (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) : 
  (y₁ = x₁^3 - 3*x₁ + 2) →
  (y₂ = x₂^3 - 3*x₂ + 2) →
  (y₃ = x₃^3 - 3*x₃ + 2) →
  (2*x₁ + 3*y₁ = 3) →
  (2*x₂ + 3*y₂ = 3) →
  (2*x₃ + 3*y₃ = 3) →
  (x₁ + x₂ + x₃ = 0 ∧ y₁ + y₂ + y₃ = 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l2043_204355


namespace NUMINAMATH_CALUDE_empty_set_problem_l2043_204321

-- Define the sets
def set_A : Set ℝ := {x | x^2 - 4 = 0}
def set_B : Set ℝ := {x | x > 9 ∨ x < 3}
def set_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 0}
def set_D : Set ℝ := {x | x > 9 ∧ x < 3}

-- Theorem statement
theorem empty_set_problem :
  (set_A ≠ ∅) ∧ (set_B ≠ ∅) ∧ (set_C ≠ ∅) ∧ (set_D = ∅) :=
sorry

end NUMINAMATH_CALUDE_empty_set_problem_l2043_204321


namespace NUMINAMATH_CALUDE_common_altitude_of_triangles_l2043_204314

theorem common_altitude_of_triangles (area1 area2 base1 base2 : ℝ) 
  (h_area1 : area1 = 800)
  (h_area2 : area2 = 1200)
  (h_base1 : base1 = 40)
  (h_base2 : base2 = 60)
  (h_positive1 : area1 > 0)
  (h_positive2 : area2 > 0)
  (h_positive3 : base1 > 0)
  (h_positive4 : base2 > 0) :
  ∃ h : ℝ, h > 0 ∧ area1 = (1/2) * base1 * h ∧ area2 = (1/2) * base2 * h ∧ h = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_common_altitude_of_triangles_l2043_204314


namespace NUMINAMATH_CALUDE_expected_value_coin_flip_l2043_204332

/-- The expected value of a coin flip game -/
theorem expected_value_coin_flip :
  let p_heads : ℚ := 2/5
  let p_tails : ℚ := 3/5
  let win_amount : ℚ := 5
  let loss_amount : ℚ := 4
  let expected_value := p_heads * win_amount - p_tails * loss_amount
  expected_value = -2/5
:= by sorry

end NUMINAMATH_CALUDE_expected_value_coin_flip_l2043_204332


namespace NUMINAMATH_CALUDE_cornelias_current_age_l2043_204361

/-- Proves Cornelia's current age given the conditions of the problem -/
theorem cornelias_current_age (kilee_current_age : ℕ) (cornelia_future_age kilee_future_age : ℕ) :
  kilee_current_age = 20 →
  kilee_future_age = kilee_current_age + 10 →
  cornelia_future_age = 3 * kilee_future_age →
  cornelia_future_age - 10 = 80 :=
by sorry

end NUMINAMATH_CALUDE_cornelias_current_age_l2043_204361


namespace NUMINAMATH_CALUDE_evaluate_expression_l2043_204317

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  y * (y - 5 * x + 2) = -15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2043_204317
