import Mathlib

namespace NUMINAMATH_CALUDE_selection_methods_count_l3250_325089

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of male athletes. -/
def num_males : ℕ := 4

/-- The number of female athletes. -/
def num_females : ℕ := 5

/-- The total number of athletes to be selected. -/
def num_selected : ℕ := 3

theorem selection_methods_count :
  (choose num_males 2 * choose num_females 1) + (choose num_males 1 * choose num_females 2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l3250_325089


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l3250_325029

theorem inverse_proportion_ratio {x₁ x₂ y₁ y₂ : ℝ} (hx : x₁ ≠ 0 ∧ x₂ ≠ 0) (hy : y₁ ≠ 0 ∧ y₂ ≠ 0)
  (h_inv_prop : ∃ k : ℝ, k ≠ 0 ∧ x₁ * y₁ = k ∧ x₂ * y₂ = k)
  (h_x_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l3250_325029


namespace NUMINAMATH_CALUDE_even_sum_difference_l3250_325001

def sum_even_range (a b : ℕ) : ℕ := 
  (b - a + 2) / 2 * (a + b) / 2

theorem even_sum_difference : 
  sum_even_range 62 110 - sum_even_range 22 70 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_difference_l3250_325001


namespace NUMINAMATH_CALUDE_mlb_game_hits_and_misses_l3250_325037

theorem mlb_game_hits_and_misses (hits misses : ℕ) : 
  misses = 3 * hits → 
  misses = 50 → 
  hits + misses = 200 := by
  sorry

end NUMINAMATH_CALUDE_mlb_game_hits_and_misses_l3250_325037


namespace NUMINAMATH_CALUDE_min_S_value_l3250_325057

/-- Represents a 10x10 table arrangement of numbers from 1 to 100 -/
def Arrangement := Fin 10 → Fin 10 → Fin 100

/-- Checks if two positions in the table are adjacent -/
def isAdjacent (p1 p2 : Fin 10 × Fin 10) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Checks if an arrangement satisfies the adjacent sum condition -/
def satisfiesCondition (arr : Arrangement) (S : ℕ) : Prop :=
  ∀ p1 p2 : Fin 10 × Fin 10, isAdjacent p1 p2 →
    (arr p1.1 p1.2).val + (arr p2.1 p2.2).val ≤ S

/-- The main theorem stating the minimum value of S -/
theorem min_S_value :
  (∃ (arr : Arrangement), satisfiesCondition arr 106) ∧
  (∀ S : ℕ, S < 106 → ¬∃ (arr : Arrangement), satisfiesCondition arr S) :=
sorry

end NUMINAMATH_CALUDE_min_S_value_l3250_325057


namespace NUMINAMATH_CALUDE_auto_finance_fraction_l3250_325047

theorem auto_finance_fraction (total_credit : ℝ) (auto_credit_percentage : ℝ) (finance_company_credit : ℝ) :
  total_credit = 475 →
  auto_credit_percentage = 0.36 →
  finance_company_credit = 57 →
  finance_company_credit / (auto_credit_percentage * total_credit) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_auto_finance_fraction_l3250_325047


namespace NUMINAMATH_CALUDE_pythagorean_proof_depends_on_parallel_postulate_l3250_325009

-- Define Euclidean geometry
class EuclideanGeometry where
  -- Assume the existence of parallel postulate
  parallel_postulate : Prop

-- Define the concept of a direct proof of the Pythagorean theorem
class PythagoreanProof (E : EuclideanGeometry) where
  -- The proof uses similarity of triangles
  uses_triangle_similarity : Prop
  -- The proof uses equivalency of areas
  uses_area_equivalence : Prop

-- Theorem statement
theorem pythagorean_proof_depends_on_parallel_postulate 
  (E : EuclideanGeometry) 
  (P : PythagoreanProof E) : 
  E.parallel_postulate → 
  (P.uses_triangle_similarity ∨ P.uses_area_equivalence) → 
  -- The proof depends on the parallel postulate
  Prop :=
sorry

end NUMINAMATH_CALUDE_pythagorean_proof_depends_on_parallel_postulate_l3250_325009


namespace NUMINAMATH_CALUDE_tan_alpha_equals_three_implies_ratio_equals_five_l3250_325013

theorem tan_alpha_equals_three_implies_ratio_equals_five (α : Real) 
  (h : Real.tan α = 3) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_three_implies_ratio_equals_five_l3250_325013


namespace NUMINAMATH_CALUDE_sin_sum_product_l3250_325092

theorem sin_sum_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_product_l3250_325092


namespace NUMINAMATH_CALUDE_bernardo_wins_game_l3250_325008

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bernardo_wins_game :
  ∃ N : ℕ,
    N = 32 ∧
    16 * N + 1400 < 2000 ∧
    16 * N + 1500 ≥ 2000 ∧
    sum_of_digits N = 5 ∧
    ∀ m : ℕ, m < N →
      ¬(16 * m + 1400 < 2000 ∧
        16 * m + 1500 ≥ 2000 ∧
        sum_of_digits m = 5) :=
by sorry

end NUMINAMATH_CALUDE_bernardo_wins_game_l3250_325008


namespace NUMINAMATH_CALUDE_fathers_age_l3250_325069

theorem fathers_age (son_age father_age : ℕ) : 
  son_age = 10 →
  father_age = 4 * son_age →
  father_age + 20 = 2 * (son_age + 20) →
  father_age = 40 :=
by sorry

end NUMINAMATH_CALUDE_fathers_age_l3250_325069


namespace NUMINAMATH_CALUDE_workshop_workers_l3250_325076

theorem workshop_workers (total_average : ℕ) (tech_count : ℕ) (tech_average : ℕ) (non_tech_average : ℕ) :
  total_average = 8000 →
  tech_count = 7 →
  tech_average = 12000 →
  non_tech_average = 6000 →
  ∃ (total_workers : ℕ), 
    total_workers * total_average = tech_count * tech_average + (total_workers - tech_count) * non_tech_average ∧
    total_workers = 21 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l3250_325076


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3250_325049

/-- The value of r for which the line x + y = 4 is tangent to the circle (x-2)^2 + (y+1)^2 = r -/
theorem tangent_line_to_circle (x y : ℝ) :
  (x + y = 4) →
  ((x - 2)^2 + (y + 1)^2 = (9:ℝ)/2) →
  ∃ (r : ℝ), r = (9:ℝ)/2 ∧ 
    (∀ (x' y' : ℝ), (x' + y' = 4) → ((x' - 2)^2 + (y' + 1)^2 ≤ r)) ∧
    (∃ (x₀ y₀ : ℝ), (x₀ + y₀ = 4) ∧ ((x₀ - 2)^2 + (y₀ + 1)^2 = r)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3250_325049


namespace NUMINAMATH_CALUDE_discriminant_of_5x2_plus_3x_minus_8_l3250_325019

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 + 3x - 8 is 169 -/
theorem discriminant_of_5x2_plus_3x_minus_8 :
  discriminant 5 3 (-8) = 169 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_5x2_plus_3x_minus_8_l3250_325019


namespace NUMINAMATH_CALUDE_common_point_linear_functions_l3250_325075

/-- Three linear functions with a common point -/
theorem common_point_linear_functions
  (a b c d : ℝ)
  (h1 : a ≠ b)
  (h2 : ∃ (x y : ℝ), (y = a * x + a) ∧ (y = b * x + b) ∧ (y = c * x + d)) :
  c = d :=
sorry

end NUMINAMATH_CALUDE_common_point_linear_functions_l3250_325075


namespace NUMINAMATH_CALUDE_slips_theorem_l3250_325055

/-- The number of slips in the bag -/
def total_slips : ℕ := 15

/-- The expected value of a randomly drawn slip -/
def expected_value : ℚ := 46/10

/-- The value on some of the slips -/
def value1 : ℕ := 3

/-- The value on the rest of the slips -/
def value2 : ℕ := 8

/-- The number of slips with value1 -/
def slips_with_value1 : ℕ := 10

theorem slips_theorem : 
  ∃ (x : ℕ), x = slips_with_value1 ∧ 
  x ≤ total_slips ∧
  (x : ℚ) / total_slips * value1 + (total_slips - x : ℚ) / total_slips * value2 = expected_value :=
sorry

end NUMINAMATH_CALUDE_slips_theorem_l3250_325055


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_three_prime_squares_l3250_325050

def is_divisible_by_three_prime_squares (n : ℕ) : Prop :=
  ∃ p q r : ℕ, 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    n % (p^2) = 0 ∧ n % (q^2) = 0 ∧ n % (r^2) = 0

theorem smallest_number_divisible_by_three_prime_squares :
  (∀ m : ℕ, m > 0 ∧ m < 900 → ¬(is_divisible_by_three_prime_squares m)) ∧
  is_divisible_by_three_prime_squares 900 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_three_prime_squares_l3250_325050


namespace NUMINAMATH_CALUDE_conditional_probability_l3250_325010

/-- Represents the probability space for the household appliance problem -/
structure ApplianceProbability where
  /-- Probability that the appliance lasts for 3 years -/
  three_years : ℝ
  /-- Probability that the appliance lasts for 4 years -/
  four_years : ℝ
  /-- Assumption that the probability of lasting 3 years is 0.8 -/
  three_years_prob : three_years = 0.8
  /-- Assumption that the probability of lasting 4 years is 0.4 -/
  four_years_prob : four_years = 0.4
  /-- Assumption that probabilities are between 0 and 1 -/
  prob_bounds : 0 ≤ three_years ∧ three_years ≤ 1 ∧ 0 ≤ four_years ∧ four_years ≤ 1

/-- The main theorem stating the conditional probability -/
theorem conditional_probability (ap : ApplianceProbability) :
  (ap.four_years / ap.three_years) = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_conditional_probability_l3250_325010


namespace NUMINAMATH_CALUDE_line_growth_limit_l3250_325077

theorem line_growth_limit :
  let initial_length : ℝ := 2
  let growth_series (n : ℕ) : ℝ := (1 / 3^n) * (1 + Real.sqrt 3)
  (initial_length + ∑' n, growth_series n) = (6 + Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_line_growth_limit_l3250_325077


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l3250_325064

/-- Represents the number of times fruits are taken out -/
def x : ℕ := sorry

/-- The original number of apples -/
def initial_apples : ℕ := 3 * x + 1

/-- The original number of oranges -/
def initial_oranges : ℕ := 4 * x + 12

/-- The condition that the number of oranges is twice that of apples -/
axiom orange_apple_ratio : initial_oranges = 2 * initial_apples

theorem fruit_basket_problem : x = 5 := by sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l3250_325064


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3250_325074

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3250_325074


namespace NUMINAMATH_CALUDE_greatest_common_multiple_15_20_under_150_l3250_325033

def is_common_multiple (n m k : ℕ) : Prop := k % n = 0 ∧ k % m = 0

theorem greatest_common_multiple_15_20_under_150 : 
  (∀ k : ℕ, k < 150 → is_common_multiple 15 20 k → k ≤ 120) ∧ 
  is_common_multiple 15 20 120 ∧ 
  120 < 150 :=
sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_15_20_under_150_l3250_325033


namespace NUMINAMATH_CALUDE_greatest_x_value_l3250_325022

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 210000) :
  x ≤ 4 ∧ ∃ y : ℤ, y > 4 → (2.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 210000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3250_325022


namespace NUMINAMATH_CALUDE_sequence_arithmetic_progression_l3250_325099

theorem sequence_arithmetic_progression
  (s : ℕ → ℕ)
  (h_increasing : ∀ n, s n < s (n + 1))
  (h_positive : ∀ n, s n > 0)
  (h_subseq1 : ∃ a d : ℕ, ∀ n, s (s n) = a + n * d)
  (h_subseq2 : ∃ b e : ℕ, ∀ n, s (s n + 1) = b + n * e) :
  ∃ c f : ℕ, ∀ n, s n = c + n * f := by
sorry

end NUMINAMATH_CALUDE_sequence_arithmetic_progression_l3250_325099


namespace NUMINAMATH_CALUDE_golden_section_steel_l3250_325034

theorem golden_section_steel (m : ℝ) : 
  (1000 + (m - 1000) * 0.618 = 1618 ∨ m - (m - 1000) * 0.618 = 1618) → 
  (m = 2000 ∨ m = 2618) := by
sorry

end NUMINAMATH_CALUDE_golden_section_steel_l3250_325034


namespace NUMINAMATH_CALUDE_total_animals_count_l3250_325035

/-- The total number of dangerous animals pointed out by the teacher in the swamp area -/
def total_dangerous_animals : ℕ := 250

/-- The number of crocodiles observed -/
def crocodiles : ℕ := 42

/-- The number of alligators observed -/
def alligators : ℕ := 35

/-- The number of vipers observed -/
def vipers : ℕ := 10

/-- The number of water moccasins observed -/
def water_moccasins : ℕ := 28

/-- The number of cottonmouth snakes observed -/
def cottonmouth_snakes : ℕ := 15

/-- The number of piranha fish in the school -/
def piranha_fish : ℕ := 120

/-- Theorem stating that the total number of dangerous animals is the sum of all observed species -/
theorem total_animals_count :
  total_dangerous_animals = crocodiles + alligators + vipers + water_moccasins + cottonmouth_snakes + piranha_fish :=
by
  sorry

end NUMINAMATH_CALUDE_total_animals_count_l3250_325035


namespace NUMINAMATH_CALUDE_last_digit_of_35_power_last_digit_of_35_to_large_power_l3250_325006

theorem last_digit_of_35_power (n : ℕ) : 35^n ≡ 5 [MOD 10] := by sorry

theorem last_digit_of_35_to_large_power :
  35^(18 * (13^33)) ≡ 5 [MOD 10] := by sorry

end NUMINAMATH_CALUDE_last_digit_of_35_power_last_digit_of_35_to_large_power_l3250_325006


namespace NUMINAMATH_CALUDE_population_change_l3250_325012

/-- Theorem: Given an initial population that increases by 30% in the first year
    and then decreases by x% in the second year, if the initial population is 15000
    and the final population is 13650, then x = 30. -/
theorem population_change (x : ℝ) : 
  let initial_population : ℝ := 15000
  let first_year_increase : ℝ := 0.3
  let final_population : ℝ := 13650
  let population_after_first_year : ℝ := initial_population * (1 + first_year_increase)
  let population_after_second_year : ℝ := population_after_first_year * (1 - x / 100)
  population_after_second_year = final_population → x = 30 := by
sorry

end NUMINAMATH_CALUDE_population_change_l3250_325012


namespace NUMINAMATH_CALUDE_a_in_interval_l3250_325095

/-- The function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The set A = {x ∈ ℝ | f(x) ≤ 0} -/
def set_A (a b : ℝ) : Set ℝ := {x | f a b x ≤ 0}

/-- The set B = {x ∈ ℝ | f(f(x) + 1) ≤ 0} -/
def set_B (a b : ℝ) : Set ℝ := {x | f a b (f a b x + 1) ≤ 0}

/-- Theorem: If A = B ≠ ∅, then a ∈ [-2, 2] -/
theorem a_in_interval (a b : ℝ) :
  set_A a b = set_B a b ∧ (set_A a b).Nonempty → a ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_a_in_interval_l3250_325095


namespace NUMINAMATH_CALUDE_trade_value_scientific_notation_l3250_325073

-- Define the value in trillion
def trade_value : ℝ := 42.1

-- Define trillion in terms of scientific notation
def trillion : ℝ := 10^12

-- Theorem to prove
theorem trade_value_scientific_notation :
  trade_value * trillion = 4.21 * 10^13 := by
  sorry

end NUMINAMATH_CALUDE_trade_value_scientific_notation_l3250_325073


namespace NUMINAMATH_CALUDE_min_value_abs_plus_two_l3250_325045

theorem min_value_abs_plus_two (a : ℚ) : |a - 1| + 2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_plus_two_l3250_325045


namespace NUMINAMATH_CALUDE_box_problem_l3250_325083

theorem box_problem (total_boxes : ℕ) (small_box_units : ℕ) (large_box_units : ℕ)
  (h_total : total_boxes = 62)
  (h_small : small_box_units = 5)
  (h_large : large_box_units = 3)
  (h_load_large_first : ∃ (x : ℕ), x * (1 / large_box_units) + 15 * (1 / small_box_units) = (total_boxes - x) * (1 / small_box_units) + 15 * (1 / large_box_units))
  : ∃ (large_boxes : ℕ), large_boxes = 27 ∧ total_boxes = large_boxes + (total_boxes - large_boxes) :=
by
  sorry

end NUMINAMATH_CALUDE_box_problem_l3250_325083


namespace NUMINAMATH_CALUDE_product_xyz_l3250_325088

theorem product_xyz (x y z : ℕ+) 
  (h1 : x + 2*y = z) 
  (h2 : x^2 - 4*y^2 + z^2 = 310) : 
  x*y*z = 4030 ∨ x*y*z = 23870 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l3250_325088


namespace NUMINAMATH_CALUDE_sum_of_g_10_and_neg_10_l3250_325003

/-- A function g defined as a polynomial of even degree -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + 5

/-- Theorem stating that g(10) + g(-10) = 4 given g(10) = 2 -/
theorem sum_of_g_10_and_neg_10 (a b c : ℝ) (h : g a b c 10 = 2) :
  g a b c 10 + g a b c (-10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_g_10_and_neg_10_l3250_325003


namespace NUMINAMATH_CALUDE_store_paid_twenty_six_l3250_325061

/-- The price the store paid for a pair of pants, given the selling price and the difference between the selling price and the store's cost. -/
def store_paid_price (selling_price : ℕ) (price_difference : ℕ) : ℕ :=
  selling_price - price_difference

/-- Theorem stating that if the selling price is $34 and the store paid $8 less, then the store paid $26. -/
theorem store_paid_twenty_six :
  store_paid_price 34 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_store_paid_twenty_six_l3250_325061


namespace NUMINAMATH_CALUDE_minimum_requirement_proof_l3250_325005

/-- The minimum pound requirement for purchasing peanuts -/
def minimum_requirement : ℕ := 15

/-- The cost of peanuts per pound in dollars -/
def cost_per_pound : ℕ := 3

/-- The amount spent by the customer in dollars -/
def amount_spent : ℕ := 105

/-- The number of pounds purchased over the minimum requirement -/
def extra_pounds : ℕ := 20

/-- Theorem stating that the minimum requirement is correct given the conditions -/
theorem minimum_requirement_proof :
  cost_per_pound * (minimum_requirement + extra_pounds) = amount_spent :=
by sorry

end NUMINAMATH_CALUDE_minimum_requirement_proof_l3250_325005


namespace NUMINAMATH_CALUDE_haley_small_gardens_l3250_325032

def total_seeds : ℕ := 56
def big_garden_seeds : ℕ := 35
def seeds_per_small_garden : ℕ := 3

def small_gardens : ℕ := (total_seeds - big_garden_seeds) / seeds_per_small_garden

theorem haley_small_gardens : small_gardens = 7 := by
  sorry

end NUMINAMATH_CALUDE_haley_small_gardens_l3250_325032


namespace NUMINAMATH_CALUDE_product_of_reciprocals_plus_one_ge_nine_l3250_325017

theorem product_of_reciprocals_plus_one_ge_nine (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (1/a + 1) * (1/b + 1) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reciprocals_plus_one_ge_nine_l3250_325017


namespace NUMINAMATH_CALUDE_tan_value_from_sin_plus_cos_l3250_325043

theorem tan_value_from_sin_plus_cos (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = 1/5) : 
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_sin_plus_cos_l3250_325043


namespace NUMINAMATH_CALUDE_max_sum_of_factors_max_sum_of_factors_achieved_l3250_325079

theorem max_sum_of_factors (a b : ℕ) : a * b = 48 → a ≠ b → a + b ≤ 49 := by sorry

theorem max_sum_of_factors_achieved : ∃ a b : ℕ, a * b = 48 ∧ a ≠ b ∧ a + b = 49 := by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_max_sum_of_factors_achieved_l3250_325079


namespace NUMINAMATH_CALUDE_greatest_valid_number_divisible_by_11_l3250_325067

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    A < 9 ∧
    n = A * 10000 + B * 1000 + C * 100 + B * 10 + A

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem greatest_valid_number_divisible_by_11 :
  ∀ n : ℕ, is_valid_number n → is_divisible_by_11 n → n ≤ 87978 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_number_divisible_by_11_l3250_325067


namespace NUMINAMATH_CALUDE_crank_slider_motion_l3250_325021

/-- Crank-slider mechanism -/
structure CrankSlider where
  oa : ℝ
  ab : ℝ
  mb : ℝ
  ω : ℝ

/-- Point coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Velocity vector -/
structure Velocity where
  vx : ℝ
  vy : ℝ

/-- Motion equations for point M -/
def motionEquations (cs : CrankSlider) (t : ℝ) : Point :=
  sorry

/-- Trajectory equation for point M -/
def trajectoryEquation (cs : CrankSlider) (x : ℝ) (y : ℝ) : Prop :=
  sorry

/-- Velocity of point M -/
def velocityM (cs : CrankSlider) (t : ℝ) : Velocity :=
  sorry

theorem crank_slider_motion 
  (cs : CrankSlider) 
  (h1 : cs.oa = 90) 
  (h2 : cs.ab = 90) 
  (h3 : cs.mb = cs.ab / 3) 
  (h4 : cs.ω = 10) :
  ∃ (me : ℝ → Point) (te : ℝ → ℝ → Prop) (ve : ℝ → Velocity),
    me = motionEquations cs ∧
    te = trajectoryEquation cs ∧
    ve = velocityM cs :=
  sorry

end NUMINAMATH_CALUDE_crank_slider_motion_l3250_325021


namespace NUMINAMATH_CALUDE_system_solution_l3250_325093

theorem system_solution (x y : ℝ) 
  (eq1 : 2 * x + y = 4) 
  (eq2 : x + 2 * y = 5) : 
  (x - y = -1 ∧ x + y = 3) ∧ 
  (1/3 * x^2 - 1/3 * y^2) * (x^2 - 2*x*y + y^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3250_325093


namespace NUMINAMATH_CALUDE_correct_ordering_l3250_325065

theorem correct_ordering (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n := by
  sorry

end NUMINAMATH_CALUDE_correct_ordering_l3250_325065


namespace NUMINAMATH_CALUDE_log_product_range_l3250_325062

theorem log_product_range : ∃ y : ℝ,
  y = Real.log 6 / Real.log 5 * Real.log 7 / Real.log 6 * Real.log 8 / Real.log 7 * Real.log 9 / Real.log 8 * Real.log 10 / Real.log 9 ∧
  1 < y ∧ y < 2 := by
  sorry

end NUMINAMATH_CALUDE_log_product_range_l3250_325062


namespace NUMINAMATH_CALUDE_system_solution_existence_l3250_325091

/-- The system of equations has at least one solution if and only if 0.5 ≤ a ≤ 2 -/
theorem system_solution_existence (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (|x - 0.5| + |y| - a) / (Real.sqrt 3 * y - x) = 0) ↔ 
  0.5 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_system_solution_existence_l3250_325091


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3250_325094

theorem quadratic_equal_roots (m n : ℝ) : 
  (∃ x : ℝ, x^(m-1) + 4*x - n = 0 ∧ 
   ∀ y : ℝ, y^(m-1) + 4*y - n = 0 → y = x) →
  m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3250_325094


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3250_325025

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {3, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3250_325025


namespace NUMINAMATH_CALUDE_perimeter_gt_four_times_circumradius_l3250_325002

/-- Definition of an acute-angled triangle -/
def IsAcuteAngledTriangle (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- Perimeter of a triangle -/
def Perimeter (a b c : ℝ) : ℝ := a + b + c

/-- Circumradius of a triangle using the formula R = abc / (4A) where A is the area -/
noncomputable def Circumradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * area)

/-- Theorem: For any acute-angled triangle, its perimeter is greater than 4 times its circumradius -/
theorem perimeter_gt_four_times_circumradius (a b c : ℝ) 
  (h : IsAcuteAngledTriangle a b c) : 
  Perimeter a b c > 4 * Circumradius a b c := by
  sorry


end NUMINAMATH_CALUDE_perimeter_gt_four_times_circumradius_l3250_325002


namespace NUMINAMATH_CALUDE_prob_select_5_times_expectation_X_l3250_325042

-- Define the review types
inductive ReviewType
  | Good
  | Neutral
  | Bad

-- Define the age groups
inductive AgeGroup
  | Below50
  | Above50

-- Define the sample data
def sampleData : Fin 2 → Fin 3 → Nat
  | ⟨0, _⟩ => fun
    | ⟨0, _⟩ => 10000  -- Good reviews for Below50
    | ⟨1, _⟩ => 2000   -- Neutral reviews for Below50
    | ⟨2, _⟩ => 2000   -- Bad reviews for Below50
  | ⟨1, _⟩ => fun
    | ⟨0, _⟩ => 2000   -- Good reviews for Above50
    | ⟨1, _⟩ => 3000   -- Neutral reviews for Above50
    | ⟨2, _⟩ => 1000   -- Bad reviews for Above50

-- Define the total sample size
def totalSampleSize : Nat := 20000

-- Define the probability of selecting a good review
def probGoodReview : Rat :=
  (sampleData ⟨0, sorry⟩ ⟨0, sorry⟩ + sampleData ⟨1, sorry⟩ ⟨0, sorry⟩) / totalSampleSize

-- Theorem for the probability of selecting 5 times
theorem prob_select_5_times :
  (1 - probGoodReview)^5 + (1 - probGoodReview)^4 * probGoodReview = 16/625 := by sorry

-- Define the number of people giving neutral reviews in each age group
def neutralReviews : Fin 2 → Nat
  | ⟨0, _⟩ => sampleData ⟨0, sorry⟩ ⟨1, sorry⟩
  | ⟨1, _⟩ => sampleData ⟨1, sorry⟩ ⟨1, sorry⟩

-- Define the total number of neutral reviews
def totalNeutralReviews : Nat := neutralReviews ⟨0, sorry⟩ + neutralReviews ⟨1, sorry⟩

-- Define the probability distribution of X
def probX : Fin 4 → Rat
  | ⟨0, _⟩ => 1/6
  | ⟨1, _⟩ => 1/2
  | ⟨2, _⟩ => 3/10
  | ⟨3, _⟩ => 1/30

-- Theorem for the mathematical expectation of X
theorem expectation_X :
  (0 : Rat) * probX ⟨0, sorry⟩ + 1 * probX ⟨1, sorry⟩ + 2 * probX ⟨2, sorry⟩ + 3 * probX ⟨3, sorry⟩ = 6/5 := by sorry

end NUMINAMATH_CALUDE_prob_select_5_times_expectation_X_l3250_325042


namespace NUMINAMATH_CALUDE_cost_per_page_is_five_cents_l3250_325052

def manuscript_copies : ℕ := 10
def binding_cost : ℚ := 5
def pages_per_manuscript : ℕ := 400
def total_cost : ℚ := 250

theorem cost_per_page_is_five_cents :
  let total_binding_cost : ℚ := manuscript_copies * binding_cost
  let copying_cost : ℚ := total_cost - total_binding_cost
  let total_pages : ℕ := manuscript_copies * pages_per_manuscript
  let cost_per_page : ℚ := copying_cost / total_pages
  cost_per_page = 5 / 100 := by sorry

end NUMINAMATH_CALUDE_cost_per_page_is_five_cents_l3250_325052


namespace NUMINAMATH_CALUDE_train_platform_problem_l3250_325024

/-- Given a train and two platforms, calculate the length of the second platform -/
theorem train_platform_problem (train_length platform1_length : ℝ)
  (time1 time2 : ℝ) :
  train_length = 100 →
  platform1_length = 350 →
  time1 = 15 →
  time2 = 20 →
  (train_length + platform1_length) / time1 = (train_length + 500) / time2 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_problem_l3250_325024


namespace NUMINAMATH_CALUDE_polynomial_division_proof_l3250_325044

theorem polynomial_division_proof (x : ℚ) : 
  (3 * x^3 + 3 * x^2 - x - 2/3) * (3 * x + 5) + (-2/3) = 
  9 * x^4 + 18 * x^3 + 8 * x^2 - 7 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_proof_l3250_325044


namespace NUMINAMATH_CALUDE_sum_is_zero_l3250_325063

/-- Given a finite subset M of real numbers with more than 2 elements,
    if for each element the absolute value is at least as large as
    the absolute value of the sum of the other elements,
    then the sum of all elements in M is zero. -/
theorem sum_is_zero (M : Finset ℝ) (h_size : 2 < M.card) :
  (∀ a ∈ M, |a| ≥ |M.sum id - a|) → M.sum id = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_zero_l3250_325063


namespace NUMINAMATH_CALUDE_g_zero_at_neg_one_l3250_325031

-- Define the function g
def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + x^2 - 4 * x + s

-- Theorem statement
theorem g_zero_at_neg_one (s : ℝ) : g (-1) s = 0 ↔ s = -5 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_at_neg_one_l3250_325031


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3250_325071

theorem solve_exponential_equation :
  ∃ x : ℝ, 5^(3*x) = Real.sqrt 125 ∧ x = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3250_325071


namespace NUMINAMATH_CALUDE_radical_simplification_l3250_325023

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l3250_325023


namespace NUMINAMATH_CALUDE_coupon_savings_difference_coupon_savings_difference_holds_l3250_325041

theorem coupon_savings_difference : ℝ → Prop :=
  fun difference =>
    ∃ (x y : ℝ),
      x > 120 ∧ y > 120 ∧
      (∀ p : ℝ, p > 120 →
        (0.2 * p ≥ 35 ∧ 0.2 * p ≥ 0.3 * (p - 120)) →
        x ≤ p ∧ p ≤ y) ∧
      (0.2 * x ≥ 35 ∧ 0.2 * x ≥ 0.3 * (x - 120)) ∧
      (0.2 * y ≥ 35 ∧ 0.2 * y ≥ 0.3 * (y - 120)) ∧
      difference = y - x ∧
      difference = 185

theorem coupon_savings_difference_holds : coupon_savings_difference 185 := by
  sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_coupon_savings_difference_holds_l3250_325041


namespace NUMINAMATH_CALUDE_power_function_property_l3250_325098

/-- A power function is a function of the form f(x) = x^α for some real α -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

theorem power_function_property (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 2 = 4) :
  f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l3250_325098


namespace NUMINAMATH_CALUDE_beka_jackson_miles_difference_l3250_325027

/-- The difference in miles flown between Beka and Jackson -/
def miles_difference (beka_miles jackson_miles : ℕ) : ℕ :=
  beka_miles - jackson_miles

/-- Theorem stating the difference in miles flown between Beka and Jackson -/
theorem beka_jackson_miles_difference :
  miles_difference 873 563 = 310 :=
by sorry

end NUMINAMATH_CALUDE_beka_jackson_miles_difference_l3250_325027


namespace NUMINAMATH_CALUDE_construct_one_to_ten_l3250_325085

/-- A type representing the allowed operations in our constructions -/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide
  | Exponentiate

/-- A type representing a construction using threes and operations -/
inductive Construction
  | Three : Construction
  | Op : Operation → Construction → Construction → Construction

/-- Evaluate a construction to a rational number -/
def evaluate : Construction → ℚ
  | Construction.Three => 3
  | Construction.Op Operation.Add a b => evaluate a + evaluate b
  | Construction.Op Operation.Subtract a b => evaluate a - evaluate b
  | Construction.Op Operation.Multiply a b => evaluate a * evaluate b
  | Construction.Op Operation.Divide a b => evaluate a / evaluate b
  | Construction.Op Operation.Exponentiate a b => (evaluate a) ^ (evaluate b).num

/-- Count the number of threes used in a construction -/
def countThrees : Construction → ℕ
  | Construction.Three => 1
  | Construction.Op _ a b => countThrees a + countThrees b

/-- Predicate to check if a construction is valid (uses exactly five threes) -/
def isValidConstruction (c : Construction) : Prop := countThrees c = 5

/-- Theorem: We can construct all numbers from 1 to 10 using five threes and allowed operations -/
theorem construct_one_to_ten :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 →
  ∃ c : Construction, isValidConstruction c ∧ evaluate c = n := by sorry

end NUMINAMATH_CALUDE_construct_one_to_ten_l3250_325085


namespace NUMINAMATH_CALUDE_jane_calculation_l3250_325020

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 17) 
  (h2 : x - y - z = 5) : 
  x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_jane_calculation_l3250_325020


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l3250_325028

/-- Prove that given vectors a, b, u, and v with specific conditions, x = 1/2 --/
theorem vector_parallel_condition (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  let u : Fin 2 → ℝ := a + 2 • b
  let v : Fin 2 → ℝ := 2 • a - b
  (∃ (k : ℝ), u = k • v) → x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l3250_325028


namespace NUMINAMATH_CALUDE_total_games_in_league_l3250_325060

theorem total_games_in_league (n : ℕ) (h : n = 35) : 
  (n * (n - 1)) / 2 = 595 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_league_l3250_325060


namespace NUMINAMATH_CALUDE_indeterminate_roots_l3250_325081

/-- Given that the equation mx^2 - 2(m+2)x + m + 5 = 0 has no real roots,
    the number of real roots of (m-5)x^2 - 2(m+2)x + m = 0 cannot be determined
    to be exclusively 0, 1, or 2. -/
theorem indeterminate_roots (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - 2*(m+2)*x + m + 5 ≠ 0) →
  ¬(∀ x : ℝ, (m-5) * x^2 - 2*(m+2)*x + m ≠ 0) ∧
  ¬(∃! x : ℝ, (m-5) * x^2 - 2*(m+2)*x + m = 0) ∧
  ¬(∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2*(m+2)*x + m = 0 ∧ (m-5) * y^2 - 2*(m+2)*y + m = 0) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_roots_l3250_325081


namespace NUMINAMATH_CALUDE_cd_price_correct_l3250_325056

/-- The price of a CD in dollars -/
def price_cd : ℝ := 14

/-- The price of a cassette in dollars -/
def price_cassette : ℝ := 9

/-- The total amount Leanna has to spend in dollars -/
def total_money : ℝ := 37

/-- The amount left over when buying one CD and two cassettes in dollars -/
def money_left : ℝ := 5

theorem cd_price_correct : 
  (2 * price_cd + price_cassette = total_money) ∧ 
  (price_cd + 2 * price_cassette = total_money - money_left) :=
by sorry

end NUMINAMATH_CALUDE_cd_price_correct_l3250_325056


namespace NUMINAMATH_CALUDE_square_binomial_coefficient_l3250_325014

theorem square_binomial_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 24 * x + 9 = (r * x + s)^2) → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_coefficient_l3250_325014


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l3250_325039

theorem largest_n_with_unique_k : ∃ (n : ℕ), n > 0 ∧ 
  (∃! (k : ℤ), (5 : ℚ)/18 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 9/17) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (5 : ℚ)/18 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 9/17)) ∧
  n = 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l3250_325039


namespace NUMINAMATH_CALUDE_third_side_is_seven_l3250_325090

/-- A triangle with two known sides and even perimeter -/
structure EvenPerimeterTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side1_eq : side1 = 2
  side2_eq : side2 = 7
  even_perimeter : ∃ n : ℕ, side1 + side2 + side3 = 2 * n

/-- The third side of an EvenPerimeterTriangle is 7 -/
theorem third_side_is_seven (t : EvenPerimeterTriangle) : t.side3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_side_is_seven_l3250_325090


namespace NUMINAMATH_CALUDE_equation_solution_l3250_325015

theorem equation_solution : 
  ∃ x : ℚ, (17 / 60 + 7 / x = 21 / x + 1 / 15) ∧ (x = 840 / 13) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3250_325015


namespace NUMINAMATH_CALUDE_star_six_three_l3250_325097

-- Define the ⭐ operation
def star (x y : ℝ) : ℝ := 4 * x - 2 * y

-- State the theorem
theorem star_six_three : star 6 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_star_six_three_l3250_325097


namespace NUMINAMATH_CALUDE_island_marriage_fraction_l3250_325078

theorem island_marriage_fraction (N : ℚ) :
  let M := (3/2) * N  -- Total number of men
  let W := (5/3) * N  -- Total number of women
  let P := M + W      -- Total population
  (2 * N) / P = 12/19 := by
  sorry

end NUMINAMATH_CALUDE_island_marriage_fraction_l3250_325078


namespace NUMINAMATH_CALUDE_oil_mixture_price_l3250_325004

/-- Given two oils mixed together, prove the price of the first oil -/
theorem oil_mixture_price (volume1 volume2 : ℝ) (price2 mix_price : ℝ) (h1 : volume1 = 10)
    (h2 : volume2 = 5) (h3 : price2 = 66) (h4 : mix_price = 58.67) :
    ∃ (price1 : ℝ), price1 = 55.005 ∧ 
    volume1 * price1 + volume2 * price2 = (volume1 + volume2) * mix_price := by
  sorry

end NUMINAMATH_CALUDE_oil_mixture_price_l3250_325004


namespace NUMINAMATH_CALUDE_final_temperature_l3250_325051

def initial_temp : Int := -3
def temp_rise : Int := 6
def temp_drop : Int := 7

theorem final_temperature : 
  initial_temp + temp_rise - temp_drop = -4 :=
by sorry

end NUMINAMATH_CALUDE_final_temperature_l3250_325051


namespace NUMINAMATH_CALUDE_vinnie_word_count_excess_l3250_325007

def word_limit : ℕ := 1000
def friday_words : ℕ := 450
def saturday_words : ℕ := 650
def sunday_words : ℕ := 300
def friday_articles : ℕ := 25
def saturday_articles : ℕ := 40
def sunday_articles : ℕ := 15

theorem vinnie_word_count_excess :
  (friday_words + saturday_words + sunday_words) -
  (friday_articles + saturday_articles + sunday_articles) -
  word_limit = 320 := by
  sorry

end NUMINAMATH_CALUDE_vinnie_word_count_excess_l3250_325007


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3250_325054

theorem complex_number_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (i / (1 + i) : ℂ) = a + b * I :=
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3250_325054


namespace NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l3250_325011

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcf_seven_eight_factorial :
  Nat.gcd (factorial 7) (factorial 8) = factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l3250_325011


namespace NUMINAMATH_CALUDE_billy_bumper_rides_l3250_325040

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The cost of each ride in tickets -/
def ticket_cost : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := 50

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := (total_tickets - ferris_rides * ticket_cost) / ticket_cost

theorem billy_bumper_rides : bumper_rides = 3 := by
  sorry

end NUMINAMATH_CALUDE_billy_bumper_rides_l3250_325040


namespace NUMINAMATH_CALUDE_three_slice_toast_l3250_325038

/-- Represents a slice of bread with two sides -/
structure Bread :=
  (side1 : Bool)
  (side2 : Bool)

/-- Represents the state of the toaster -/
structure ToasterState :=
  (slot1 : Option Bread)
  (slot2 : Option Bread)

/-- Represents the toasting process -/
def toast (initial : List Bread) (time : Nat) : List Bread → Prop :=
  sorry

theorem three_slice_toast :
  ∀ (initial : List Bread),
    initial.length = 3 →
    ∀ (b : Bread), b ∈ initial → ¬b.side1 ∧ ¬b.side2 →
    ∃ (final : List Bread),
      toast initial 3 final ∧
      final.length = 3 ∧
      ∀ (b : Bread), b ∈ final → b.side1 ∧ b.side2 :=
by sorry

end NUMINAMATH_CALUDE_three_slice_toast_l3250_325038


namespace NUMINAMATH_CALUDE_set_operations_and_range_l3250_325030

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem set_operations_and_range :
  (∀ a : ℝ,
    (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
    (A ∪ B = {x | -1 ≤ x ∧ x < 4}) ∧
    ((Aᶜ ∩ Bᶜ) = {x | x < -1 ∨ 4 ≤ x}) ∧
    ((B ∩ C a = B) → a ≥ 4)) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l3250_325030


namespace NUMINAMATH_CALUDE_coordinate_square_area_l3250_325046

/-- A square in the coordinate plane with y-coordinates between 3 and 8 -/
structure CoordinateSquare where
  lowest_y : ℝ
  highest_y : ℝ
  is_square : lowest_y = 3 ∧ highest_y = 8

/-- The area of a CoordinateSquare is 25 -/
theorem coordinate_square_area (s : CoordinateSquare) : (s.highest_y - s.lowest_y) ^ 2 = 25 :=
sorry

end NUMINAMATH_CALUDE_coordinate_square_area_l3250_325046


namespace NUMINAMATH_CALUDE_unique_cube_difference_nineteen_l3250_325086

theorem unique_cube_difference_nineteen :
  ∃! (x y : ℕ), x^3 - y^3 = 19 ∧ x = 3 ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_cube_difference_nineteen_l3250_325086


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l3250_325080

theorem concert_ticket_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (full_price_tickets : ℕ) 
  (student_price_tickets : ℕ) 
  (full_price : ℕ) 
  (h1 : total_tickets = 150)
  (h2 : total_revenue = 2450)
  (h3 : student_price_tickets = total_tickets - full_price_tickets)
  (h4 : total_revenue = full_price_tickets * full_price + student_price_tickets * (full_price / 2))
  : full_price_tickets * full_price = 1150 := by
  sorry

#check concert_ticket_revenue

end NUMINAMATH_CALUDE_concert_ticket_revenue_l3250_325080


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3250_325072

theorem square_sum_geq_product_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + a*c :=
by sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3250_325072


namespace NUMINAMATH_CALUDE_square_area_ratio_l3250_325016

theorem square_area_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (4 * a = 4 * (4 * b)) → a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3250_325016


namespace NUMINAMATH_CALUDE_lap_distance_l3250_325087

theorem lap_distance (boys_laps girls_laps girls_miles : ℚ) : 
  boys_laps = 27 →
  girls_laps = boys_laps + 9 →
  girls_miles = 27 →
  girls_miles / girls_laps = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_lap_distance_l3250_325087


namespace NUMINAMATH_CALUDE_first_expression_value_l3250_325048

theorem first_expression_value (a : ℝ) (E : ℝ) : 
  a = 28 → (E + (3 * a - 8)) / 2 = 74 → E = 72 := by
  sorry

end NUMINAMATH_CALUDE_first_expression_value_l3250_325048


namespace NUMINAMATH_CALUDE_num_measurable_weights_l3250_325053

/-- Represents the number of weights of each type -/
def num_weights : ℕ := 3

/-- Represents the weight values -/
def weight_values : List ℕ := [1, 5, 50]

/-- Represents the maximum number of weights that can be placed on one side of the scale -/
def max_weights_per_side : ℕ := num_weights * weight_values.length

/-- Represents the set of all possible weight combinations on one side of the scale -/
def weight_combinations : Finset (List ℕ) :=
  sorry

/-- Calculates the total weight of a combination -/
def total_weight (combination : List ℕ) : ℕ :=
  sorry

/-- Represents the set of all possible positive weight differences -/
def measurable_weights : Finset ℕ :=
  sorry

/-- The main theorem stating that the number of different positive weights
    that can be measured is 63 -/
theorem num_measurable_weights : measurable_weights.card = 63 :=
  sorry

end NUMINAMATH_CALUDE_num_measurable_weights_l3250_325053


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l3250_325000

def P : ℕ → ℝ → ℝ → ℝ → ℝ
  | 0, x, y, z => 1
  | m + 1, x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem polynomial_symmetry (m : ℕ) (x y z : ℝ) :
  P m x y z = P m y x z ∧
  P m x y z = P m x z y ∧
  P m x y z = P m y z x ∧
  P m x y z = P m z x y ∧
  P m x y z = P m z y x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l3250_325000


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3250_325084

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 1
  f 1 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3250_325084


namespace NUMINAMATH_CALUDE_inequality_proof_l3250_325070

theorem inequality_proof (a b c : ℝ) (m n k : ℕ+) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≥ (a^(m:ℝ) * b^(n:ℝ) * c^(k:ℝ))^(1/(m+n+k:ℝ)) + 
              (a^(n:ℝ) * b^(k:ℝ) * c^(m:ℝ))^(1/(m+n+k:ℝ)) + 
              (a^(k:ℝ) * b^(m:ℝ) * c^(n:ℝ))^(1/(m+n+k:ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3250_325070


namespace NUMINAMATH_CALUDE_complex_inequality_implies_real_range_l3250_325018

theorem complex_inequality_implies_real_range (a : ℝ) :
  let z : ℂ := 3 + a * I
  (Complex.abs (z - 2) < 2) → (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_implies_real_range_l3250_325018


namespace NUMINAMATH_CALUDE_parallel_vectors_linear_combination_l3250_325059

/-- Given two parallel plane vectors a and b, prove their linear combination -/
theorem parallel_vectors_linear_combination (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  (2 • a + 3 • b : Fin 2 → ℝ) = ![-4, -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_linear_combination_l3250_325059


namespace NUMINAMATH_CALUDE_coin_authenticity_test_l3250_325082

/-- Represents the type of coin -/
inductive CoinType
| Real
| Fake

/-- Represents the weight difference between real and fake coins -/
def weightDifference : ℤ := 1

/-- Represents the total number of coins -/
def totalCoins (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the number of fake coins -/
def fakeCoins (k : ℕ) : ℕ := 2 * k

/-- Represents the scale reading when weighing n coins against n coins -/
def scaleReading (n k₁ k₂ : ℕ) : ℤ := (k₁ : ℤ) - (k₂ : ℤ)

/-- Main theorem: The parity of the scale reading determines the type of the chosen coin -/
theorem coin_authenticity_test (n k : ℕ) (h : k ≤ n) :
  ∀ (chosenCoin : CoinType) (k₁ k₂ : ℕ) (h₁ : k₁ + k₂ = fakeCoins k - 1),
    chosenCoin = CoinType.Fake ↔ scaleReading n k₁ k₂ % 2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_coin_authenticity_test_l3250_325082


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3250_325058

theorem total_cost_calculation (sandwich_price soda_price : ℚ) 
  (sandwich_quantity soda_quantity : ℕ) : 
  sandwich_price = 245/100 →
  soda_price = 87/100 →
  sandwich_quantity = 2 →
  soda_quantity = 4 →
  (sandwich_price * sandwich_quantity + soda_price * soda_quantity : ℚ) = 838/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3250_325058


namespace NUMINAMATH_CALUDE_num_sam_sandwiches_l3250_325068

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of restricted sandwich combinations due to roast beef and swiss cheese. -/
def roast_beef_swiss_restrictions : ℕ := num_breads

/-- Represents the number of restricted sandwich combinations due to rye bread and turkey. -/
def rye_turkey_restrictions : ℕ := num_cheeses

/-- Represents the number of restricted sandwich combinations due to roast beef and rye bread. -/
def roast_beef_rye_restrictions : ℕ := num_cheeses

/-- The total number of possible sandwich combinations without restrictions. -/
def total_combinations : ℕ := num_breads * num_meats * num_cheeses

/-- The number of restricted sandwich combinations. -/
def total_restrictions : ℕ := roast_beef_swiss_restrictions + rye_turkey_restrictions + roast_beef_rye_restrictions

/-- Theorem stating the number of sandwiches Sam can order. -/
theorem num_sam_sandwiches : total_combinations - total_restrictions = 193 := by
  sorry

end NUMINAMATH_CALUDE_num_sam_sandwiches_l3250_325068


namespace NUMINAMATH_CALUDE_regression_consistency_l3250_325096

/-- A structure representing a linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- A structure representing sample statistics -/
structure SampleStatistics where
  x_mean : ℝ
  y_mean : ℝ
  correlation : ℝ

/-- Checks if the given linear regression model is consistent with the sample statistics -/
def is_consistent_regression (stats : SampleStatistics) (model : LinearRegression) : Prop :=
  stats.correlation > 0 ∧ 
  stats.y_mean = model.slope * stats.x_mean + model.intercept

/-- The theorem stating that the given linear regression model is consistent with the sample statistics -/
theorem regression_consistency : 
  let stats : SampleStatistics := { x_mean := 3, y_mean := 3.5, correlation := 1 }
  let model : LinearRegression := { slope := 0.4, intercept := 2.3 }
  is_consistent_regression stats model := by
  sorry


end NUMINAMATH_CALUDE_regression_consistency_l3250_325096


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3250_325066

theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) * (1 - Complex.I) = 2) :
  z.im = 1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3250_325066


namespace NUMINAMATH_CALUDE_prize_interval_is_1000_l3250_325036

/-- Represents the prize structure of an international competition --/
structure PrizeStructure where
  totalPrize : ℕ
  firstPrize : ℕ
  numPositions : ℕ
  hasPrizeInterval : Bool

/-- Calculates the interval between prizes --/
def calculatePrizeInterval (ps : PrizeStructure) : ℕ :=
  sorry

/-- Theorem stating that the prize interval is 1000 given the specific conditions --/
theorem prize_interval_is_1000 (ps : PrizeStructure) 
  (h1 : ps.totalPrize = 15000)
  (h2 : ps.firstPrize = 5000)
  (h3 : ps.numPositions = 5)
  (h4 : ps.hasPrizeInterval = true) : 
  calculatePrizeInterval ps = 1000 := by
  sorry

end NUMINAMATH_CALUDE_prize_interval_is_1000_l3250_325036


namespace NUMINAMATH_CALUDE_equality_sum_l3250_325026

theorem equality_sum (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + N = 183 := by
  sorry

end NUMINAMATH_CALUDE_equality_sum_l3250_325026
