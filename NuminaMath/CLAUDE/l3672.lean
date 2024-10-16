import Mathlib

namespace NUMINAMATH_CALUDE_lamp_cost_l3672_367265

theorem lamp_cost (lamp_cost bulb_cost : ℝ) : 
  (bulb_cost = lamp_cost - 4) →
  (2 * lamp_cost + 6 * bulb_cost = 32) →
  lamp_cost = 7 := by
sorry

end NUMINAMATH_CALUDE_lamp_cost_l3672_367265


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l3672_367286

theorem triangle_max_perimeter :
  ∀ (x y : ℕ),
    x > 0 →
    y > 0 →
    y = 2 * x →
    (x + y > 20 ∧ x + 20 > y ∧ y + 20 > x) →
    x + y + 20 ≤ 77 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l3672_367286


namespace NUMINAMATH_CALUDE_math_team_selection_ways_l3672_367231

/-- The number of ways to select r items from n items --/
def binomial (n r : ℕ) : ℕ := Nat.choose n r

/-- The total number of students in the math club --/
def total_students : ℕ := 14

/-- The number of students to be selected for the team --/
def team_size : ℕ := 6

/-- Theorem stating that the number of ways to select the team is 3003 --/
theorem math_team_selection_ways :
  binomial total_students team_size = 3003 := by sorry

end NUMINAMATH_CALUDE_math_team_selection_ways_l3672_367231


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3672_367209

theorem quadratic_minimum (x : ℝ) (h : x ≥ 0) : x^2 + 13*x + 4 ≥ 4 ∧ ∃ y ≥ 0, y^2 + 13*y + 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3672_367209


namespace NUMINAMATH_CALUDE_exists_non_monochromatic_coloring_l3672_367222

/-- Represents a coloring of numbers using 4 colors -/
def Coloring := Fin 2008 → Fin 4

/-- An arithmetic progression of 10 terms -/
def ArithmeticProgression := Fin 10 → Fin 2008

/-- Checks if an arithmetic progression is valid (within the range 1 to 2008) -/
def isValidAP (ap : ArithmeticProgression) : Prop :=
  ∀ i : Fin 10, ap i < 2008

/-- Checks if an arithmetic progression is monochromatic under a given coloring -/
def isMonochromatic (c : Coloring) (ap : ArithmeticProgression) : Prop :=
  ∃ color : Fin 4, ∀ i : Fin 10, c (ap i) = color

/-- The main theorem statement -/
theorem exists_non_monochromatic_coloring :
  ∃ c : Coloring, ∀ ap : ArithmeticProgression, isValidAP ap → ¬isMonochromatic c ap := by
  sorry

end NUMINAMATH_CALUDE_exists_non_monochromatic_coloring_l3672_367222


namespace NUMINAMATH_CALUDE_evaluate_expression_l3672_367214

theorem evaluate_expression : (10^8 / (2.5 * 10^5)) * 3 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3672_367214


namespace NUMINAMATH_CALUDE_f_geq_a_implies_a_leq_2_l3672_367257

/-- The function f(x) = x^2 - ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

/-- The theorem stating that if f(x) ≥ a for all x ∈ [-1, +∞), then a ≤ 2 -/
theorem f_geq_a_implies_a_leq_2 (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f a x ≥ a) → a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_f_geq_a_implies_a_leq_2_l3672_367257


namespace NUMINAMATH_CALUDE_recipe_cereal_cups_l3672_367224

/-- Given a recipe calling for 18.0 servings of cereal, where each serving is 2.0 cups,
    the total number of cups needed is 36.0. -/
theorem recipe_cereal_cups : 
  let servings : ℝ := 18.0
  let cups_per_serving : ℝ := 2.0
  servings * cups_per_serving = 36.0 := by
sorry

end NUMINAMATH_CALUDE_recipe_cereal_cups_l3672_367224


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l3672_367247

-- Define the function
def f (x : ℝ) : ℝ := 2 * |x - 3| - 8

-- Theorem statement
theorem minimum_point_of_translated_graph :
  ∀ x : ℝ, f x ≥ f 3 ∧ f 3 = -8 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l3672_367247


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l3672_367242

/-- The line y - kx - 1 = 0 always has a common point with the ellipse x²/5 + y²/m = 1 
    for all real k if and only if m ∈ [1,5) ∪ (5,+∞) -/
theorem line_ellipse_intersection (m : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y - k*x - 1 = 0 ∧ x^2/5 + y^2/m = 1) ↔ 
  (m ∈ Set.Icc 1 5 ∪ Set.Ioi 5) ∧ m ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l3672_367242


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3672_367277

theorem inequality_solution_set (x : ℝ) : 
  (1/2)^(x - x^2) < Real.log 81 / Real.log 3 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3672_367277


namespace NUMINAMATH_CALUDE_set_operations_l3672_367237

open Set

def A : Set ℝ := {x | x ≤ 5}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 8}

theorem set_operations :
  (A ∩ B = {x | -3 < x ∧ x ≤ 5}) ∧
  (A ∪ B = {x | x ≤ 8}) ∧
  (A ∪ (𝒰 \ B) = {x | x ≤ 5 ∨ x > 8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3672_367237


namespace NUMINAMATH_CALUDE_book_price_l3672_367258

theorem book_price (price : ℝ) : price = 1 + (1/3) * price → price = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_book_price_l3672_367258


namespace NUMINAMATH_CALUDE_subset_conditions_l3672_367246

/-- Given sets A and B, prove the conditions for m when A is a proper subset of B -/
theorem subset_conditions (m : ℝ) : 
  let A : Set ℝ := {3, m^2}
  let B : Set ℝ := {1, 3, 2*m-1}
  (A ⊂ B) → (m^2 ≠ 1 ∧ m^2 ≠ 2*m-1 ∧ m^2 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_subset_conditions_l3672_367246


namespace NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l3672_367218

noncomputable section

def a : ℝ := Real.log 2 / 2
def b : ℝ := Real.log 3 / 3
def c : ℝ := Real.log Real.pi / Real.pi
def d : ℝ := Real.log 2.72 / 2.72
def f : ℝ := (Real.sqrt 10 * Real.log 10) / 20

theorem order_of_logarithmic_expressions :
  a < f ∧ f < c ∧ c < b ∧ b < d := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l3672_367218


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3672_367272

universe u

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {4, 5}

theorem intersection_with_complement :
  A ∩ (U \ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3672_367272


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3672_367284

theorem sqrt_equation_solution :
  ∃! x : ℚ, Real.sqrt (3 - 4 * x) = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3672_367284


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3672_367249

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  2*x^3 - x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3672_367249


namespace NUMINAMATH_CALUDE_history_paper_pages_l3672_367215

/-- Given a paper due in 6 days with a required writing pace of 11 pages per day,
    the total number of pages in the paper is 66. -/
theorem history_paper_pages (days : ℕ) (pages_per_day : ℕ) (h1 : days = 6) (h2 : pages_per_day = 11) :
  days * pages_per_day = 66 := by
  sorry

end NUMINAMATH_CALUDE_history_paper_pages_l3672_367215


namespace NUMINAMATH_CALUDE_cricketer_average_score_l3672_367289

theorem cricketer_average_score 
  (total_matches : Nat) 
  (matches_with_known_average : Nat) 
  (known_average : ℝ) 
  (total_average : ℝ) 
  (h1 : total_matches = 5)
  (h2 : matches_with_known_average = 3)
  (h3 : known_average = 10)
  (h4 : total_average = 22) :
  let remaining_matches := total_matches - matches_with_known_average
  let remaining_average := (total_matches * total_average - matches_with_known_average * known_average) / remaining_matches
  remaining_average = 40 := by
sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l3672_367289


namespace NUMINAMATH_CALUDE_nested_inverse_expression_l3672_367207

theorem nested_inverse_expression : ((((3+2)⁻¹ - 1)⁻¹ - 1)⁻¹ - 1 : ℚ) = -13/9 := by
  sorry

end NUMINAMATH_CALUDE_nested_inverse_expression_l3672_367207


namespace NUMINAMATH_CALUDE_horner_method_properties_l3672_367244

def horner_polynomial (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

def horner_v2 (x : ℝ) : ℝ := (4 * x + 2) * x + 3.5

theorem horner_method_properties :
  let x : ℝ := 5
  (∃ (max_multiplications : ℕ), max_multiplications = 5 ∧
    ∀ (other_multiplications : ℕ),
      other_multiplications ≤ max_multiplications) ∧
  horner_v2 x = 113.5 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_properties_l3672_367244


namespace NUMINAMATH_CALUDE_remainder_5032_div_28_l3672_367230

theorem remainder_5032_div_28 : 5032 % 28 = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5032_div_28_l3672_367230


namespace NUMINAMATH_CALUDE_train_passes_jogger_train_passes_jogger_time_l3672_367202

/-- The time taken for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The time taken for the train to pass the jogger is 40 seconds -/
theorem train_passes_jogger_time : train_passes_jogger 9 45 200 200 = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_passes_jogger_train_passes_jogger_time_l3672_367202


namespace NUMINAMATH_CALUDE_apartment_complex_households_l3672_367278

/-- The maximum number of households in Jungkook's apartment complex -/
def max_households : ℕ := 2000

/-- The maximum number of buildings in the apartment complex -/
def max_buildings : ℕ := 25

/-- The maximum number of floors per building -/
def max_floors : ℕ := 10

/-- The number of households per floor -/
def households_per_floor : ℕ := 8

/-- Theorem stating that the maximum number of households in the apartment complex is 2000 -/
theorem apartment_complex_households :
  max_households = max_buildings * max_floors * households_per_floor :=
by sorry

end NUMINAMATH_CALUDE_apartment_complex_households_l3672_367278


namespace NUMINAMATH_CALUDE_distance_between_lines_l3672_367233

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- Radius of the circle -/
  radius : ℝ
  /-- Distance between adjacent parallel lines -/
  line_distance : ℝ
  /-- Length of the first chord -/
  chord1_length : ℝ
  /-- Length of the second chord -/
  chord2_length : ℝ
  /-- Length of the third chord -/
  chord3_length : ℝ
  /-- The first and second chords have equal length -/
  chord1_eq_chord2 : chord1_length = chord2_length
  /-- The first chord has length 40 -/
  chord1_is_40 : chord1_length = 40
  /-- The third chord has length 36 -/
  chord3_is_36 : chord3_length = 36

/-- Theorem stating that the distance between adjacent parallel lines is 1.5 -/
theorem distance_between_lines (c : CircleWithParallelLines) : c.line_distance = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_lines_l3672_367233


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3672_367219

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  sum_property : a 1 + a 3 = 8
  geometric_mean : a 4 ^ 2 = a 2 * a 9
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term (seq : ArithmeticSequence) : seq.a 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3672_367219


namespace NUMINAMATH_CALUDE_pencil_notebook_cost_l3672_367255

/-- The cost of pencils and notebooks -/
theorem pencil_notebook_cost : 
  ∀ (p n : ℝ), 
  3 * p + 4 * n = 60 →
  p + n = 15.512820512820513 →
  96 * p + 24 * n = 520 := by
sorry

end NUMINAMATH_CALUDE_pencil_notebook_cost_l3672_367255


namespace NUMINAMATH_CALUDE_two_distinct_prime_products_count_l3672_367282

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that counts the number of integers less than n that are the product of exactly two distinct primes -/
def countTwoDistinctPrimeProducts (n : ℕ) : ℕ := sorry

/-- Theorem stating that the count of numbers less than 1,000,000 that are the product of exactly two distinct primes is 209867 -/
theorem two_distinct_prime_products_count :
  countTwoDistinctPrimeProducts 1000000 = 209867 := by sorry

end NUMINAMATH_CALUDE_two_distinct_prime_products_count_l3672_367282


namespace NUMINAMATH_CALUDE_product_plus_one_is_square_l3672_367268

theorem product_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_is_square_l3672_367268


namespace NUMINAMATH_CALUDE_coin_combination_difference_l3672_367203

def coin_values : List Nat := [5, 10, 20, 25]

def total_amount : Nat := 45

def is_valid_combination (combination : List Nat) : Prop :=
  combination.all (λ x => x ∈ coin_values) ∧
  combination.sum = total_amount

def num_coins (combination : List Nat) : Nat :=
  combination.length

theorem coin_combination_difference :
  ∃ (min_combination max_combination : List Nat),
    is_valid_combination min_combination ∧
    is_valid_combination max_combination ∧
    (∀ c, is_valid_combination c → 
      num_coins min_combination ≤ num_coins c ∧
      num_coins c ≤ num_coins max_combination) ∧
    num_coins max_combination - num_coins min_combination = 7 :=
  sorry

end NUMINAMATH_CALUDE_coin_combination_difference_l3672_367203


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l3672_367292

theorem set_equality_implies_sum (a b : ℝ) : 
  ({0, b, b/a} : Set ℝ) = {1, a, a+b} → a + 2*b = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l3672_367292


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3672_367287

/-- Given line passing through (1,2) and perpendicular to 2x - 6y + 1 = 0 -/
def given_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

/-- Point that the perpendicular line passes through -/
def point : ℝ × ℝ := (1, 2)

/-- Equation of the perpendicular line -/
def perpendicular_line (x y : ℝ) : Prop := 3 * x + y - 5 = 0

/-- Theorem stating that the perpendicular line passing through (1,2) 
    has the equation 3x + y - 5 = 0 -/
theorem perpendicular_line_equation : 
  ∀ x y : ℝ, given_line x y → 
  (perpendicular_line x y ↔ 
   (perpendicular_line point.1 point.2 ∧ 
    (x - point.1) * (x - point.1) + (y - point.2) * (y - point.2) = 
    ((x - 1) * 2 + (y - 2) * (-6))^2 / (2^2 + (-6)^2))) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3672_367287


namespace NUMINAMATH_CALUDE_first_company_daily_rate_l3672_367298

/-- The daily rate of the first car rental company -/
def first_company_rate : ℝ := 17.99

/-- The per-mile rate of the first car rental company -/
def first_company_per_mile : ℝ := 0.18

/-- The daily rate of City Rentals -/
def city_rentals_rate : ℝ := 18.95

/-- The per-mile rate of City Rentals -/
def city_rentals_per_mile : ℝ := 0.16

/-- The number of miles driven -/
def miles_driven : ℝ := 48.0

theorem first_company_daily_rate :
  first_company_rate + first_company_per_mile * miles_driven =
  city_rentals_rate + city_rentals_per_mile * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_first_company_daily_rate_l3672_367298


namespace NUMINAMATH_CALUDE_total_produce_cost_l3672_367213

/-- Calculates the total cost of produce given specific quantities and pricing conditions -/
theorem total_produce_cost (asparagus_bundles : ℕ) (asparagus_price : ℚ)
                           (grape_boxes : ℕ) (grape_weight : ℚ) (grape_price : ℚ)
                           (apples : ℕ) (apple_price : ℚ)
                           (carrot_bags : ℕ) (carrot_orig_price : ℚ) (carrot_discount : ℚ)
                           (strawberry_pounds : ℕ) (strawberry_orig_price : ℚ) (strawberry_discount : ℚ) :
  asparagus_bundles = 60 ∧ asparagus_price = 3 ∧
  grape_boxes = 40 ∧ grape_weight = 2.2 ∧ grape_price = 2.5 ∧
  apples = 700 ∧ apple_price = 0.5 ∧
  carrot_bags = 100 ∧ carrot_orig_price = 2 ∧ carrot_discount = 0.25 ∧
  strawberry_pounds = 120 ∧ strawberry_orig_price = 3.5 ∧ strawberry_discount = 0.15 →
  (asparagus_bundles : ℚ) * asparagus_price +
  (grape_boxes : ℚ) * grape_weight * grape_price +
  ((apples / 3) * 2 : ℚ) * apple_price +
  (carrot_bags : ℚ) * carrot_orig_price * (1 - carrot_discount) +
  (strawberry_pounds : ℚ) * strawberry_orig_price * (1 - strawberry_discount) = 1140.5 := by
sorry

end NUMINAMATH_CALUDE_total_produce_cost_l3672_367213


namespace NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l3672_367204

theorem proposition_false_iff_a_in_range (a : ℝ) : 
  (¬ ∃ x : ℝ, |x - a| + |x - 1| ≤ 2) ↔ (a < -1 ∨ a > 3) := by
  sorry

end NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l3672_367204


namespace NUMINAMATH_CALUDE_jake_present_weight_l3672_367205

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 156

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 224 - jake_weight

/-- Theorem stating Jake's present weight is 156 pounds -/
theorem jake_present_weight : jake_weight = 156 := by
  have h1 : jake_weight - 20 = 2 * sister_weight := by sorry
  have h2 : jake_weight + sister_weight = 224 := by sorry
  sorry

#check jake_present_weight

end NUMINAMATH_CALUDE_jake_present_weight_l3672_367205


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_for_bounded_f_l3672_367270

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- Theorem 1
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x ≤ 1} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem range_of_a_for_bounded_f :
  ∀ a : ℝ, (∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) → (a = -1 ∨ a = 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_for_bounded_f_l3672_367270


namespace NUMINAMATH_CALUDE_multiples_of_ten_not_twenty_l3672_367206

def count_numbers (n : ℕ) : ℕ :=
  (n.div 10 + 1).div 2

theorem multiples_of_ten_not_twenty (upper_bound : ℕ) (h : upper_bound = 500) :
  count_numbers upper_bound = 25 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_ten_not_twenty_l3672_367206


namespace NUMINAMATH_CALUDE_factorization_cubic_l3672_367217

theorem factorization_cubic (a : ℝ) : a^3 - 10*a^2 + 25*a = a*(a-5)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_l3672_367217


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_two_satisfies_smallest_satisfying_integer_l3672_367234

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 900 * x % 640 = 0 → x ≥ 32 := by
  sorry

theorem thirty_two_satisfies : 900 * 32 % 640 = 0 := by
  sorry

theorem smallest_satisfying_integer : ∃! x : ℕ, x > 0 ∧ 900 * x % 640 = 0 ∧ ∀ y : ℕ, (y > 0 ∧ 900 * y % 640 = 0 → y ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_two_satisfies_smallest_satisfying_integer_l3672_367234


namespace NUMINAMATH_CALUDE_probability_x_less_than_2y_l3672_367267

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

-- Define the region where x < 2y
def region : Set (ℝ × ℝ) :=
  {p ∈ rectangle | p.1 < 2 * p.2}

-- Define the probability measure on the rectangle
noncomputable def prob : MeasureTheory.Measure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_x_less_than_2y :
  prob region / prob rectangle = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_less_than_2y_l3672_367267


namespace NUMINAMATH_CALUDE_equal_output_day_l3672_367269

def initial_output_A : ℝ := 200
def daily_output_A : ℝ := 20
def daily_output_B : ℝ := 30

def total_output_A (days : ℝ) : ℝ := initial_output_A + daily_output_A * days
def total_output_B (days : ℝ) : ℝ := daily_output_B * days

theorem equal_output_day : 
  ∃ (day : ℝ), day > 0 ∧ total_output_A day = total_output_B day ∧ day = 20 :=
sorry

end NUMINAMATH_CALUDE_equal_output_day_l3672_367269


namespace NUMINAMATH_CALUDE_barbecue_chicken_orders_l3672_367220

/-- Represents the number of pieces of chicken used in different dish types --/
structure ChickenPieces where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- Represents the number of orders for different dish types --/
structure Orders where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- The total number of chicken pieces needed for all orders --/
def totalChickenPieces (cp : ChickenPieces) (o : Orders) : ℕ :=
  cp.pasta * o.pasta + cp.barbecue * o.barbecue + cp.friedDinner * o.friedDinner

/-- The theorem to prove --/
theorem barbecue_chicken_orders
  (cp : ChickenPieces)
  (o : Orders)
  (h1 : cp.pasta = 2)
  (h2 : cp.barbecue = 3)
  (h3 : cp.friedDinner = 8)
  (h4 : o.friedDinner = 2)
  (h5 : o.pasta = 6)
  (h6 : totalChickenPieces cp o = 37) :
  o.barbecue = 3 := by
  sorry

end NUMINAMATH_CALUDE_barbecue_chicken_orders_l3672_367220


namespace NUMINAMATH_CALUDE_only_zero_is_purely_imaginary_l3672_367238

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number parameterized by m. -/
def complexNumber (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m, m^2 - 5*m + 6⟩

theorem only_zero_is_purely_imaginary :
  ∃! m : ℝ, isPurelyImaginary (complexNumber m) ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_only_zero_is_purely_imaginary_l3672_367238


namespace NUMINAMATH_CALUDE_quadratic_function_ratio_bound_l3672_367275

/-- A quadratic function f(x) = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function value at x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The derivative of the quadratic function -/
def QuadraticFunction.derivative (f : QuadraticFunction) (x : ℝ) : ℝ :=
  2 * f.a * x + f.b

theorem quadratic_function_ratio_bound (f : QuadraticFunction) 
    (h1 : f.derivative 0 > 0)
    (h2 : ∀ x : ℝ, f.eval x ≥ 0) :
    f.eval 1 / f.derivative 0 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_ratio_bound_l3672_367275


namespace NUMINAMATH_CALUDE_money_division_l3672_367226

theorem money_division (a b c : ℚ) :
  a = (1/2 : ℚ) * (b + c) →
  b = (2/3 : ℚ) * (a + c) →
  a = 122 →
  a + b + c = 366 := by
sorry

end NUMINAMATH_CALUDE_money_division_l3672_367226


namespace NUMINAMATH_CALUDE_weeklyRentIs1200_l3672_367271

/-- Calculates the weekly rent for a flower shop given the following conditions:
  * Utilities cost is 20% of rent
  * 2 employees per shift
  * Store open 16 hours a day for 5 days a week
  * Employee pay is $12.50 per hour
  * Total weekly expenses are $3440
-/
def calculateWeeklyRent (totalExpenses : ℚ) (employeePay : ℚ) (hoursPerDay : ℕ) (daysPerWeek : ℕ) (employeesPerShift : ℕ) : ℚ :=
  let totalHours : ℕ := hoursPerDay * daysPerWeek * employeesPerShift
  let weeklyWages : ℚ := employeePay * totalHours
  (totalExpenses - weeklyWages) / 1.2

/-- Proves that the weekly rent for the flower shop is $1200 -/
theorem weeklyRentIs1200 :
  calculateWeeklyRent 3440 12.5 16 5 2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_weeklyRentIs1200_l3672_367271


namespace NUMINAMATH_CALUDE_lemonade_theorem_l3672_367227

/-- Represents the number of glasses of lemonade that can be made -/
def lemonade_glasses (lemons oranges grapefruits : ℕ) : ℕ :=
  let lemon_glasses := lemons / 2
  let orange_glasses := oranges
  let citrus_glasses := min lemon_glasses orange_glasses
  let grapefruit_glasses := grapefruits
  citrus_glasses + grapefruit_glasses

/-- Theorem stating that with given ingredients, 15 glasses of lemonade can be made -/
theorem lemonade_theorem : lemonade_glasses 18 10 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_theorem_l3672_367227


namespace NUMINAMATH_CALUDE_parabola_translation_l3672_367291

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x - 3)^2 - 1

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, y = original_parabola (x - 3) - 2 ↔ y = translated_parabola x :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3672_367291


namespace NUMINAMATH_CALUDE_midpoint_property_implies_linear_l3672_367296

/-- A function satisfying the midpoint property -/
def HasMidpointProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x + y) / 2) = (f x + f y) / 2

/-- The main theorem: continuous functions with the midpoint property are linear -/
theorem midpoint_property_implies_linear
  (f : ℝ → ℝ) (hf : Continuous f) (hm : HasMidpointProperty f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_midpoint_property_implies_linear_l3672_367296


namespace NUMINAMATH_CALUDE_max_dot_product_regular_octagon_l3672_367229

/-- Regular octagon with side length 1 -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i j : Fin 8, 
    (i.val + 1) % 8 = j.val → 
    Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 1

/-- Vector between two points -/
def vector (A : RegularOctagon) (i j : Fin 8) : ℝ × ℝ :=
  ((A.vertices j).1 - (A.vertices i).1, (A.vertices j).2 - (A.vertices i).2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem max_dot_product_regular_octagon (A : RegularOctagon) :
  ∃ (i j : Fin 8), ∀ (k l : Fin 8),
    dot_product (vector A k l) (vector A 0 1) ≤ dot_product (vector A i j) (vector A 0 1) ∧
    dot_product (vector A i j) (vector A 0 1) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_dot_product_regular_octagon_l3672_367229


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3672_367223

/-- Proves the simplification of two algebraic expressions -/
theorem algebraic_simplification (a b : ℝ) :
  (3 * a - (4 * b - 2 * a + 1) = 5 * a - 4 * b - 1) ∧
  (2 * (5 * a - 3 * b) - 3 * (a^2 - 2 * b) = 10 * a - 3 * a^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3672_367223


namespace NUMINAMATH_CALUDE_cos_arcsin_tan_arccos_eq_l3672_367260

theorem cos_arcsin_tan_arccos_eq (x : ℝ) : 
  x ∈ Set.Icc (-1 : ℝ) 1 → 
  (Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_cos_arcsin_tan_arccos_eq_l3672_367260


namespace NUMINAMATH_CALUDE_james_total_matches_l3672_367256

/-- The number of boxes in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of boxes James has -/
def james_dozens : ℕ := 5

/-- The number of matches in each box -/
def matches_per_box : ℕ := 20

/-- Theorem: James has 1200 matches in total -/
theorem james_total_matches : james_dozens * dozen * matches_per_box = 1200 := by
  sorry

end NUMINAMATH_CALUDE_james_total_matches_l3672_367256


namespace NUMINAMATH_CALUDE_expand_expression_l3672_367261

theorem expand_expression (y : ℝ) : 5 * (y - 2) * (y + 7) = 5 * y^2 + 25 * y - 70 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3672_367261


namespace NUMINAMATH_CALUDE_parity_of_solutions_l3672_367274

theorem parity_of_solutions (n m p q : ℤ) : 
  (∃ k : ℤ, n = 2 * k) →  -- n is even
  (∃ k : ℤ, m = 2 * k + 1) →  -- m is odd
  p - 1988 * q = n →  -- first equation
  11 * p + 27 * q = m →  -- second equation
  (∃ k : ℤ, p = 2 * k) ∧ (∃ k : ℤ, q = 2 * k + 1) :=  -- p is even and q is odd
by sorry

end NUMINAMATH_CALUDE_parity_of_solutions_l3672_367274


namespace NUMINAMATH_CALUDE_radical_expression_equality_l3672_367276

theorem radical_expression_equality : 
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) - (Real.sqrt 3 - Real.sqrt 2)^2 = 2 * Real.sqrt 6 - 2 := by
  sorry

end NUMINAMATH_CALUDE_radical_expression_equality_l3672_367276


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l3672_367253

/-- Calculates the weighted average profit percentage for cricket bat sales -/
theorem cricket_bat_profit_percentage :
  let bat_a_quantity : ℕ := 5
  let bat_a_cost : ℚ := 900
  let bat_a_profit : ℚ := 300
  let bat_b_quantity : ℕ := 8
  let bat_b_cost : ℚ := 1200
  let bat_b_profit : ℚ := 400
  let bat_c_quantity : ℕ := 3
  let bat_c_cost : ℚ := 1500
  let bat_c_profit : ℚ := 500

  let total_cost : ℚ := bat_a_quantity * bat_a_cost + bat_b_quantity * bat_b_cost + bat_c_quantity * bat_c_cost
  let total_profit : ℚ := bat_a_quantity * bat_a_profit + bat_b_quantity * bat_b_profit + bat_c_quantity * bat_c_profit

  let weighted_avg_profit_percentage : ℚ := (total_profit / total_cost) * 100

  weighted_avg_profit_percentage = 100/3 := by sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l3672_367253


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3672_367211

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (∀ a b, (a - b) * a^2 < 0 → a < b) ∧ 
  (∃ a b, a < b ∧ (a - b) * a^2 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3672_367211


namespace NUMINAMATH_CALUDE_NaNO3_formed_l3672_367281

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String

-- Define the moles of each substance
structure Moles where
  AgNO3 : ℝ
  NaOH : ℝ
  AgOH : ℝ
  NaNO3 : ℝ

-- Define the chemical equation
def chemicalEquation : Reaction :=
  { reactant1 := "AgNO3"
  , reactant2 := "NaOH"
  , product1 := "AgOH"
  , product2 := "NaNO3" }

-- Define the initial moles
def initialMoles : Moles :=
  { AgNO3 := 1
  , NaOH := 1
  , AgOH := 0
  , NaNO3 := 0 }

-- Define the reaction completion condition
def reactionComplete (initial : Moles) (final : Moles) : Prop :=
  final.AgNO3 = 0 ∨ final.NaOH = 0

-- Define the no side reactions condition
def noSideReactions (initial : Moles) (final : Moles) : Prop :=
  initial.AgNO3 + initial.NaOH = final.AgOH + final.NaNO3

-- Theorem statement
theorem NaNO3_formed
  (reaction : Reaction)
  (initial : Moles)
  (final : Moles)
  (hReaction : reaction = chemicalEquation)
  (hInitial : initial = initialMoles)
  (hComplete : reactionComplete initial final)
  (hNoSide : noSideReactions initial final) :
  final.NaNO3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_NaNO3_formed_l3672_367281


namespace NUMINAMATH_CALUDE_triangle_transformation_l3672_367266

-- Define the points of the original triangle
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (8, 9)
def C : ℝ × ℝ := (-3, 7)

-- Define the points of the transformed triangle
def A' : ℝ × ℝ := (-2, -6)
def B' : ℝ × ℝ := (-7, -11)
def C' : ℝ × ℝ := (2, -9)

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-(x - 0.5) - 5.5, -y - 2)

-- Theorem stating that the transformation maps the original triangle to the new one
theorem triangle_transformation :
  transform A = A' ∧ transform B = B' ∧ transform C = C' := by
  sorry


end NUMINAMATH_CALUDE_triangle_transformation_l3672_367266


namespace NUMINAMATH_CALUDE_arithmetic_proof_l3672_367210

theorem arithmetic_proof : (100 + 20 / 90) * 90 = 120 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l3672_367210


namespace NUMINAMATH_CALUDE_permutation_difference_divisibility_l3672_367236

/-- For any integer n > 2 and any two permutations of {0, 1, ..., n-1},
    there exist distinct indices i and j such that n divides (aᵢ * bᵢ - aⱼ * bⱼ). -/
theorem permutation_difference_divisibility (n : ℕ) (hn : n > 2)
  (a b : Fin n → Fin n) (ha : Function.Bijective a) (hb : Function.Bijective b) :
  ∃ (i j : Fin n), i ≠ j ∧ (n : ℤ) ∣ (a i * b i - a j * b j) :=
sorry

end NUMINAMATH_CALUDE_permutation_difference_divisibility_l3672_367236


namespace NUMINAMATH_CALUDE_tan_585_degrees_l3672_367259

theorem tan_585_degrees : Real.tan (585 * Real.pi / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_585_degrees_l3672_367259


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3672_367273

-- Define the circle and square
def circle_radius : ℝ := 4
def square_side : ℝ := 2

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def A : Point := ⟨square_side, 0⟩
def B : Point := ⟨square_side, square_side⟩
def C : Point := ⟨0, square_side⟩

-- Define the extended points D and E
def D : Point := sorry
def E : Point := sorry

-- Define the shaded area
def shaded_area : ℝ := sorry

-- Theorem statement
theorem shaded_area_calculation :
  shaded_area = (16 * π / 3) - 6 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3672_367273


namespace NUMINAMATH_CALUDE_land_area_calculation_l3672_367293

/-- The total area of 9 square-shaped plots of land, each measuring 6 meters in length and width, is 324 square meters. -/
theorem land_area_calculation (num_plots : ℕ) (side_length : ℝ) : 
  num_plots = 9 → side_length = 6 → num_plots * (side_length * side_length) = 324 := by
  sorry

end NUMINAMATH_CALUDE_land_area_calculation_l3672_367293


namespace NUMINAMATH_CALUDE_expression_bounds_l3672_367280

theorem expression_bounds (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) 
    (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) : 
  4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l3672_367280


namespace NUMINAMATH_CALUDE_ship_grain_calculation_l3672_367240

/-- The amount of grain spilled from a ship, in tons -/
def grain_spilled : ℕ := 49952

/-- The amount of grain remaining on the ship, in tons -/
def grain_remaining : ℕ := 918

/-- The original amount of grain on the ship, in tons -/
def original_grain : ℕ := grain_spilled + grain_remaining

theorem ship_grain_calculation :
  original_grain = 50870 := by sorry

end NUMINAMATH_CALUDE_ship_grain_calculation_l3672_367240


namespace NUMINAMATH_CALUDE_group_size_l3672_367243

/-- The number of people in the group -/
def n : ℕ := sorry

/-- The total weight of the group before the change -/
def W : ℝ := sorry

/-- The weight increase when the new person joins -/
def weight_increase : ℝ := 2.5

/-- The weight of the person being replaced -/
def old_weight : ℝ := 55

/-- The weight of the new person -/
def new_weight : ℝ := 75

theorem group_size :
  (W + new_weight - old_weight) / n = W / n + weight_increase →
  n = 8 := by sorry

end NUMINAMATH_CALUDE_group_size_l3672_367243


namespace NUMINAMATH_CALUDE_no_rectangle_from_five_distinct_squares_l3672_367208

/-- A configuration of five squares with side lengths q₁, q₂, q₃, q₄, q₅ -/
structure FiveSquares where
  q₁ : ℝ
  q₂ : ℝ
  q₃ : ℝ
  q₄ : ℝ
  q₅ : ℝ
  h₁ : 0 < q₁
  h₂ : q₁ < q₂
  h₃ : q₂ < q₃
  h₄ : q₃ < q₄
  h₅ : q₄ < q₅

/-- Predicate to check if the five squares can form a rectangle -/
def CanFormRectangle (s : FiveSquares) : Prop :=
  ∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ w * h = s.q₁^2 + s.q₂^2 + s.q₃^2 + s.q₄^2 + s.q₅^2

/-- Theorem stating that it's impossible to form a rectangle with five squares of distinct sizes -/
theorem no_rectangle_from_five_distinct_squares :
  ¬∃ (s : FiveSquares), CanFormRectangle s := by
  sorry

end NUMINAMATH_CALUDE_no_rectangle_from_five_distinct_squares_l3672_367208


namespace NUMINAMATH_CALUDE_clean_city_people_l3672_367250

/-- The number of people working together to clean the city -/
def total_people (group_A group_B group_C group_D group_E group_F group_G group_H : ℕ) : ℕ :=
  group_A + group_B + group_C + group_D + group_E + group_F + group_G + group_H

/-- Theorem stating the total number of people cleaning the city -/
theorem clean_city_people :
  ∃ (group_A group_B group_C group_D group_E group_F group_G group_H : ℕ),
    group_A = 54 ∧
    group_B = group_A - 17 ∧
    group_C = 2 * group_B ∧
    group_D = group_A / 3 ∧
    group_E = group_C + (group_C / 4) ∧
    group_F = group_D / 2 ∧
    group_G = (group_A + group_B + group_C) - ((group_A + group_B + group_C) * 3 / 10) ∧
    group_H = group_F + group_G ∧
    total_people group_A group_B group_C group_D group_E group_F group_G group_H = 523 :=
by sorry

end NUMINAMATH_CALUDE_clean_city_people_l3672_367250


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3672_367283

theorem sum_of_two_numbers (a b : ℝ) : 
  a + b = 25 → a * b = 144 → |a - b| = 7 → a + b = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3672_367283


namespace NUMINAMATH_CALUDE_max_sum_of_integers_l3672_367285

theorem max_sum_of_integers (A B C D : ℕ) : 
  (10 ≤ A) ∧ (A < 100) ∧
  (10 ≤ B) ∧ (B < 100) ∧
  (10 ≤ C) ∧ (C < 100) ∧
  (10 ≤ D) ∧ (D < 100) ∧
  (B = 3 * C) ∧
  (D = 2 * B - C) ∧
  (A = B + D) →
  A + B + C + D ≤ 204 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_integers_l3672_367285


namespace NUMINAMATH_CALUDE_inequality_proof_l3672_367297

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : c < 1) : (a - b) * (c - 1) < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3672_367297


namespace NUMINAMATH_CALUDE_circle_ratio_l3672_367263

theorem circle_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (h_area : π * r₂^2 - π * r₁^2 = 4 * (π * r₁^2)) :
  r₁ / r₂ = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l3672_367263


namespace NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l3672_367299

/-- The number of products sampled -/
def n : ℕ := 10

/-- The event of having at least two defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- The complementary event of event_A -/
def complement_A (defective : ℕ) : Prop := defective ≤ 1

/-- Theorem stating that the complement of event_A is having at most one defective product -/
theorem complement_of_at_least_two_defective :
  ∀ defective : ℕ, defective ≤ n → (¬ event_A defective ↔ complement_A defective) :=
by sorry

end NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l3672_367299


namespace NUMINAMATH_CALUDE_fraction_relation_l3672_367212

theorem fraction_relation (x y z w : ℚ) 
  (h1 : x / y = 7)
  (h2 : z / y = 5)
  (h3 : z / w = 3 / 4) :
  w / x = 20 / 21 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l3672_367212


namespace NUMINAMATH_CALUDE_jake_sausage_cost_l3672_367241

/-- Calculates the total cost of sausages given the weight per package, number of packages, and price per pound -/
def total_cost (weight_per_package : ℕ) (num_packages : ℕ) (price_per_pound : ℕ) : ℕ :=
  weight_per_package * num_packages * price_per_pound

/-- Theorem: The total cost of Jake's sausage purchase is $24 -/
theorem jake_sausage_cost : total_cost 2 3 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_jake_sausage_cost_l3672_367241


namespace NUMINAMATH_CALUDE_smallest_product_l3672_367245

def S : Finset Int := {-10, -3, 0, 4, 6}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int) (hx : x ∈ S) (hy : y ∈ S), x * y ≤ a * b ∧ x * y = -60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l3672_367245


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l3672_367239

/-- Given two people moving in opposite directions for 1 hour, 
    where one moves at 35 km/h and they end up 60 km apart,
    prove that the speed of the other person is 25 km/h. -/
theorem opposite_direction_speed 
  (speed_person1 : ℝ) 
  (speed_person2 : ℝ) 
  (time : ℝ) 
  (total_distance : ℝ) 
  (h1 : speed_person2 = 35) 
  (h2 : time = 1) 
  (h3 : total_distance = 60) 
  (h4 : speed_person1 * time + speed_person2 * time = total_distance) : 
  speed_person1 = 25 := by
  sorry

#check opposite_direction_speed

end NUMINAMATH_CALUDE_opposite_direction_speed_l3672_367239


namespace NUMINAMATH_CALUDE_tara_yoghurt_purchase_l3672_367279

/-- The number of cartons of yoghurt Tara bought -/
def yoghurt_cartons : ℕ := sorry

/-- The number of cartons of ice cream Tara bought -/
def ice_cream_cartons : ℕ := 19

/-- The cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 7

/-- The cost of one carton of yoghurt in dollars -/
def yoghurt_cost : ℕ := 1

/-- The difference in dollars between ice cream and yoghurt spending -/
def spending_difference : ℕ := 129

theorem tara_yoghurt_purchase : 
  ice_cream_cartons * ice_cream_cost = 
  yoghurt_cartons * yoghurt_cost + spending_difference ∧ 
  yoghurt_cartons = 4 := by sorry

end NUMINAMATH_CALUDE_tara_yoghurt_purchase_l3672_367279


namespace NUMINAMATH_CALUDE_equation_solution_l3672_367248

theorem equation_solution (x : ℝ) : 
  (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) → x = -15 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3672_367248


namespace NUMINAMATH_CALUDE_solve_for_a_l3672_367235

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = -3 → a * x - y = 1) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3672_367235


namespace NUMINAMATH_CALUDE_orchids_cut_correct_l3672_367254

/-- The number of red orchids Sally cut from her garden -/
def orchids_cut (initial_red : ℕ) (final_red : ℕ) : ℕ :=
  final_red - initial_red

/-- Theorem stating that the number of orchids Sally cut is the difference between final and initial red orchids -/
theorem orchids_cut_correct (initial_red initial_white final_red : ℕ) 
  (h1 : initial_red = 9)
  (h2 : initial_white = 3)
  (h3 : final_red = 15) :
  orchids_cut initial_red final_red = 6 := by
  sorry

#eval orchids_cut 9 15

end NUMINAMATH_CALUDE_orchids_cut_correct_l3672_367254


namespace NUMINAMATH_CALUDE_milk_yogurt_quantities_l3672_367225

/-- Represents the quantities and prices of milk and yogurt --/
structure MilkYogurtData where
  milk_cost : ℝ
  yogurt_cost : ℝ
  yogurt_quantity_ratio : ℝ
  price_difference : ℝ
  milk_selling_price : ℝ
  yogurt_markup : ℝ
  yogurt_discount : ℝ
  total_profit : ℝ

/-- Theorem stating the quantities of milk and discounted yogurt --/
theorem milk_yogurt_quantities (data : MilkYogurtData) 
  (h_milk_cost : data.milk_cost = 2000)
  (h_yogurt_cost : data.yogurt_cost = 4800)
  (h_ratio : data.yogurt_quantity_ratio = 1.5)
  (h_price_diff : data.price_difference = 30)
  (h_milk_price : data.milk_selling_price = 80)
  (h_yogurt_markup : data.yogurt_markup = 0.25)
  (h_yogurt_discount : data.yogurt_discount = 0.1)
  (h_total_profit : data.total_profit = 2150) :
  ∃ (milk_quantity yogurt_discounted : ℕ),
    milk_quantity = 40 ∧ yogurt_discounted = 25 := by
  sorry

end NUMINAMATH_CALUDE_milk_yogurt_quantities_l3672_367225


namespace NUMINAMATH_CALUDE_existence_of_two_integers_l3672_367262

theorem existence_of_two_integers (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ q₁ q₂ : ℕ, q₁ ≠ q₂ ∧
    1 ≤ q₁ ∧ q₁ ≤ p - 1 ∧
    1 ≤ q₂ ∧ q₂ ≤ p - 1 ∧
    (q₁^(p-1) : ℤ) % p^2 = 1 ∧
    (q₂^(p-1) : ℤ) % p^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_two_integers_l3672_367262


namespace NUMINAMATH_CALUDE_no_solution_condition_l3672_367216

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (1 / (x - 2) + a / (2 - x) ≠ 2 * a)) ↔ (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3672_367216


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l3672_367228

-- Define the triangle AEF
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the area function
def area : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

-- Define the parallel relation
def parallel : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop := sorry

-- Define the cyclic property
def is_cyclic : Quadrilateral → Prop := sorry

-- Define the distance function
def distance : (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

-- Define the theorem
theorem area_ratio_theorem (AEF : Triangle) (ABCD : Quadrilateral) :
  distance AEF.B AEF.C = 20 →
  distance AEF.A AEF.B = 21 →
  distance AEF.A AEF.C = 21 →
  parallel ABCD.B ABCD.D AEF.B AEF.C →
  is_cyclic ABCD →
  distance ABCD.B ABCD.C = 3 →
  distance ABCD.C ABCD.D = 4 →
  (area ABCD.A ABCD.B ABCD.C + area ABCD.A ABCD.C ABCD.D) / area AEF.A AEF.B AEF.C = 49 / 400 :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l3672_367228


namespace NUMINAMATH_CALUDE_expression_evaluation_l3672_367200

theorem expression_evaluation :
  let x : ℝ := 3 + Real.sqrt 2
  (1 - 5 / (x + 2)) / ((x^2 - 6*x + 9) / (x + 2)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3672_367200


namespace NUMINAMATH_CALUDE_total_money_made_l3672_367252

/-- Represents the amount of water collected per inch of rain -/
def gallons_per_inch : ℝ := 15

/-- Represents the rainfall on Monday in inches -/
def monday_rain : ℝ := 4

/-- Represents the rainfall on Tuesday in inches -/
def tuesday_rain : ℝ := 3

/-- Represents the rainfall on Wednesday in inches -/
def wednesday_rain : ℝ := 2.5

/-- Represents the selling price per gallon on Monday -/
def monday_price : ℝ := 1.2

/-- Represents the selling price per gallon on Tuesday -/
def tuesday_price : ℝ := 1.5

/-- Represents the selling price per gallon on Wednesday -/
def wednesday_price : ℝ := 0.8

/-- Theorem stating the total money James made from selling water -/
theorem total_money_made : 
  (gallons_per_inch * monday_rain * monday_price) +
  (gallons_per_inch * tuesday_rain * tuesday_price) +
  (gallons_per_inch * wednesday_rain * wednesday_price) = 169.5 := by
  sorry

end NUMINAMATH_CALUDE_total_money_made_l3672_367252


namespace NUMINAMATH_CALUDE_xyz_value_l3672_367251

-- Define a geometric sequence of 5 terms
def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ c = b * q ∧ d = c * q ∧ e = d * q

-- State the theorem
theorem xyz_value (x y z : ℝ) 
  (h : is_geometric_sequence (-1) x y z (-4)) : x * y * z = -8 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3672_367251


namespace NUMINAMATH_CALUDE_area_two_sectors_l3672_367232

/-- The area of a figure formed by two 45° sectors of a circle with radius 15 -/
theorem area_two_sectors (r : ℝ) (θ : ℝ) (h1 : r = 15) (h2 : θ = 45 * π / 180) :
  2 * (θ / (2 * π)) * π * r^2 = 56.25 * π :=
sorry

end NUMINAMATH_CALUDE_area_two_sectors_l3672_367232


namespace NUMINAMATH_CALUDE_solve_tank_problem_l3672_367290

def tank_problem (initial_capacity : ℝ) (leak_rate1 leak_rate2 fill_rate : ℝ)
  (leak_duration1 leak_duration2 fill_duration : ℝ) (missing_amount : ℝ) : Prop :=
  let total_loss := leak_rate1 * leak_duration1 + leak_rate2 * leak_duration2
  let remaining_after_loss := initial_capacity - total_loss
  let current_amount := initial_capacity - missing_amount
  let amount_added := current_amount - remaining_after_loss
  fill_rate = amount_added / fill_duration

theorem solve_tank_problem :
  tank_problem 350000 32000 10000 40000 5 10 3 140000 := by
  sorry

end NUMINAMATH_CALUDE_solve_tank_problem_l3672_367290


namespace NUMINAMATH_CALUDE_root_product_theorem_l3672_367288

theorem root_product_theorem (a b : ℂ) : 
  a ≠ b →
  a^4 + a^3 - 1 = 0 →
  b^4 + b^3 - 1 = 0 →
  (a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3672_367288


namespace NUMINAMATH_CALUDE_coffee_shrink_theorem_l3672_367221

/-- Represents the shrink ray effect on volume --/
def shrinkEffect : ℝ := 0.5

/-- Number of coffee cups --/
def numCups : ℕ := 5

/-- Initial volume of coffee in each cup (in ounces) --/
def initialVolume : ℝ := 8

/-- Calculates the total volume of coffee after shrinking --/
def totalVolumeAfterShrink (shrinkEffect : ℝ) (numCups : ℕ) (initialVolume : ℝ) : ℝ :=
  (shrinkEffect * initialVolume) * numCups

/-- Theorem stating that the total volume of coffee after shrinking is 20 ounces --/
theorem coffee_shrink_theorem : 
  totalVolumeAfterShrink shrinkEffect numCups initialVolume = 20 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shrink_theorem_l3672_367221


namespace NUMINAMATH_CALUDE_fraction_invariance_l3672_367264

theorem fraction_invariance (x y : ℝ) :
  2 * y^2 / (x - y)^2 = 2 * (3*y)^2 / ((3*x) - (3*y))^2 :=
sorry

end NUMINAMATH_CALUDE_fraction_invariance_l3672_367264


namespace NUMINAMATH_CALUDE_jasmine_buys_six_bags_l3672_367201

/-- The number of bags of chips Jasmine buys -/
def bags_of_chips : ℕ := sorry

/-- The weight of one bag of chips in ounces -/
def chips_weight : ℕ := 20

/-- The weight of one tin of cookies in ounces -/
def cookies_weight : ℕ := 9

/-- The total weight Jasmine carries in ounces -/
def total_weight : ℕ := 21 * 16

theorem jasmine_buys_six_bags :
  bags_of_chips = 6 ∧
  chips_weight * bags_of_chips + cookies_weight * (4 * bags_of_chips) = total_weight :=
by sorry

end NUMINAMATH_CALUDE_jasmine_buys_six_bags_l3672_367201


namespace NUMINAMATH_CALUDE_cake_muffin_buyers_l3672_367294

theorem cake_muffin_buyers (cake_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ) 
  (prob_neither : ℚ) (h1 : cake_buyers = 50) (h2 : muffin_buyers = 40) 
  (h3 : both_buyers = 16) (h4 : prob_neither = 26/100) : 
  ∃ total_buyers : ℕ, 
    (total_buyers : ℚ) - ((cake_buyers : ℚ) + (muffin_buyers : ℚ) - (both_buyers : ℚ)) = 
    prob_neither * (total_buyers : ℚ) ∧ total_buyers = 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_muffin_buyers_l3672_367294


namespace NUMINAMATH_CALUDE_y_greater_than_one_l3672_367295

theorem y_greater_than_one (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 := by
  sorry

end NUMINAMATH_CALUDE_y_greater_than_one_l3672_367295
