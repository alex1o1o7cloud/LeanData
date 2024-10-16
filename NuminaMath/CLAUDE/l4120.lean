import Mathlib

namespace NUMINAMATH_CALUDE_no_real_solutions_for_equation_l4120_412051

theorem no_real_solutions_for_equation :
  ∀ x : ℝ, x + Real.sqrt (2 * x - 3) ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_equation_l4120_412051


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l4120_412064

/-- Calculates the final hair length after a given time period. -/
def final_hair_length (initial_length : ℝ) (growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_length + growth_rate * months

/-- Proves that Isabella's hair will be 28 inches long after 5 months. -/
theorem isabellas_hair_growth :
  final_hair_length 18 2 5 = 28 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l4120_412064


namespace NUMINAMATH_CALUDE_max_earnings_theorem_l4120_412025

/-- Represents the exchange rates for a given day -/
structure ExchangeRates where
  gbp_to_usd : ℝ
  jpy_to_usd : ℝ
  eur_to_usd : ℝ

/-- Calculates the maximum total earnings in USD -/
def max_total_earnings (usd_hours : ℝ) (gbp_hours : ℝ) (jpy_hours : ℝ) (eur_hours : ℝ)
  (usd_rate : ℝ) (gbp_rate : ℝ) (jpy_rate : ℝ) (eur_rate : ℝ)
  (day1 : ExchangeRates) (day2 : ExchangeRates) (day3 : ExchangeRates) : ℝ :=
  sorry

/-- Theorem stating that the maximum total earnings is $32.61 -/
theorem max_earnings_theorem :
  let day1 : ExchangeRates := { gbp_to_usd := 1.35, jpy_to_usd := 0.009, eur_to_usd := 1.18 }
  let day2 : ExchangeRates := { gbp_to_usd := 1.38, jpy_to_usd := 0.0085, eur_to_usd := 1.20 }
  let day3 : ExchangeRates := { gbp_to_usd := 1.33, jpy_to_usd := 0.0095, eur_to_usd := 1.21 }
  max_total_earnings 4 0.5 1.5 1 5 3 400 4 day1 day2 day3 = 32.61 := by
  sorry

end NUMINAMATH_CALUDE_max_earnings_theorem_l4120_412025


namespace NUMINAMATH_CALUDE_piecewise_representation_of_f_l4120_412022

def f (x : ℝ) := |x - 1| + 1

theorem piecewise_representation_of_f :
  ∀ x : ℝ, f x = if x ≥ 1 then x else 2 - x := by
  sorry

end NUMINAMATH_CALUDE_piecewise_representation_of_f_l4120_412022


namespace NUMINAMATH_CALUDE_grocer_coffee_stock_theorem_l4120_412050

/-- Represents the amount of coffee in pounds and its decaffeinated percentage -/
structure CoffeeStock where
  amount : ℝ
  decaf_percent : ℝ

/-- Calculates the new coffee stock after a purchase or sale -/
def update_stock (current : CoffeeStock) (transaction : CoffeeStock) (is_sale : Bool) : CoffeeStock :=
  sorry

/-- Calculates the final percentage of decaffeinated coffee -/
def final_decaf_percentage (transactions : List (CoffeeStock × Bool)) : ℝ :=
  sorry

theorem grocer_coffee_stock_theorem (initial_stock : CoffeeStock) 
  (transactions : List (CoffeeStock × Bool)) : 
  let final_percent := final_decaf_percentage transactions
  ∃ ε > 0, |final_percent - 28.88| < ε :=
by sorry

end NUMINAMATH_CALUDE_grocer_coffee_stock_theorem_l4120_412050


namespace NUMINAMATH_CALUDE_no_solution_exists_l4120_412053

-- Function to reverse a number
def reverseNumber (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_solution_exists :
  ¬ ∃ (x : ℕ), x + 276 = 435 ∧ reverseNumber x = 731 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l4120_412053


namespace NUMINAMATH_CALUDE_line_symmetry_l4120_412008

/-- Given two lines l₁ and l₂ in the xy-plane, prove that if the angle bisector between them
    is y = x, and l₁ has the equation x + 2y + 3 = 0, then l₂ has the equation 2x + y + 3 = 0. -/
theorem line_symmetry (l₁ l₂ : Set (ℝ × ℝ)) : 
  (∀ p : ℝ × ℝ, p ∈ l₁ ↔ p.1 + 2 * p.2 + 3 = 0) →
  (∀ p : ℝ × ℝ, p ∈ l₂ ↔ 2 * p.1 + p.2 + 3 = 0) →
  (∀ p : ℝ × ℝ, p ∈ l₁ ∨ p ∈ l₂ → p.1 = p.2 → 
    ∃ q : ℝ × ℝ, (q ∈ l₁ ∧ q.1 + q.2 = p.1 + p.2) ∨ (q ∈ l₂ ∧ q.1 + q.2 = p.1 + p.2)) :=
by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l4120_412008


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l4120_412090

theorem gcd_lcm_sum : Nat.gcd 24 54 + Nat.lcm 40 10 = 46 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l4120_412090


namespace NUMINAMATH_CALUDE_bottles_left_after_second_game_l4120_412026

-- Define the given constants
def initial_cases : ℕ := 10
def bottles_per_case : ℕ := 20
def bottles_used_first_game : ℕ := 70
def bottles_used_second_game : ℕ := 110

-- Define the theorem
theorem bottles_left_after_second_game :
  initial_cases * bottles_per_case - bottles_used_first_game - bottles_used_second_game = 20 := by
  sorry

end NUMINAMATH_CALUDE_bottles_left_after_second_game_l4120_412026


namespace NUMINAMATH_CALUDE_not_monotonic_iff_a_in_range_l4120_412086

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

theorem not_monotonic_iff_a_in_range (a : ℝ) :
  (∃ x y, 2 ≤ x ∧ x < y ∧ y ≤ 4 ∧ (f a x < f a y ∧ f a y < f a x)) ↔ 3 < a ∧ a < 6 := by
  sorry

end NUMINAMATH_CALUDE_not_monotonic_iff_a_in_range_l4120_412086


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l4120_412073

def p (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l4120_412073


namespace NUMINAMATH_CALUDE_intersection_equals_three_l4120_412068

theorem intersection_equals_three :
  ∃ a : ℝ, ({1, 3, a^2 + 3*a - 4} : Set ℝ) ∩ ({0, 6, a^2 + 4*a - 2, a + 3} : Set ℝ) = {3} :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_three_l4120_412068


namespace NUMINAMATH_CALUDE_marble_statue_weight_l4120_412010

/-- The weight of a marble statue after three successive reductions -/
def final_weight (original : ℝ) : ℝ :=
  original * (1 - 0.28) * (1 - 0.18) * (1 - 0.20)

/-- Theorem stating the relationship between the original and final weights -/
theorem marble_statue_weight (original : ℝ) :
  final_weight original = 85.0176 → original = 144 := by
  sorry

#eval final_weight 144

end NUMINAMATH_CALUDE_marble_statue_weight_l4120_412010


namespace NUMINAMATH_CALUDE_jimmys_bet_l4120_412043

/-- Represents a fan with equally spaced blades -/
structure Fan where
  num_blades : ℕ
  revolutions_per_second : ℝ

/-- Represents a bullet shot -/
structure Bullet where
  shot_time : ℝ
  speed : ℝ

/-- Predicate that determines if a bullet can hit all blades of a fan -/
def can_hit_all_blades (f : Fan) (b : Bullet) : Prop :=
  ∃ t : ℝ, ∀ i : Fin f.num_blades, 
    ∃ k : ℤ, b.shot_time + (i : ℝ) * (1 / f.num_blades) = t + k / f.revolutions_per_second

/-- Theorem stating that for a fan with 4 blades rotating at 50 revolutions per second,
    there exists a bullet that can hit all blades -/
theorem jimmys_bet : 
  ∃ b : Bullet, can_hit_all_blades ⟨4, 50⟩ b :=
sorry

end NUMINAMATH_CALUDE_jimmys_bet_l4120_412043


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l4120_412093

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (31/15) * x^2 - (27/5) * x - 289/15

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-1) = 7 ∧ q 2 = -3 ∧ q 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l4120_412093


namespace NUMINAMATH_CALUDE_lizette_minerva_stamp_difference_l4120_412098

theorem lizette_minerva_stamp_difference :
  let lizette_stamps : ℕ := 813
  let minerva_stamps : ℕ := 688
  lizette_stamps > minerva_stamps →
  lizette_stamps - minerva_stamps = 125 := by
sorry

end NUMINAMATH_CALUDE_lizette_minerva_stamp_difference_l4120_412098


namespace NUMINAMATH_CALUDE_franks_age_l4120_412099

theorem franks_age (frank : ℕ) (gabriel : ℕ) : 
  gabriel = frank - 3 → 
  frank + gabriel = 17 → 
  frank = 10 := by sorry

end NUMINAMATH_CALUDE_franks_age_l4120_412099


namespace NUMINAMATH_CALUDE_quadratic_solution_square_l4120_412070

theorem quadratic_solution_square (y : ℝ) : 
  6 * y^2 + 2 = 4 * y + 12 → (12 * y - 2)^2 = 324 ∨ (12 * y - 2)^2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_square_l4120_412070


namespace NUMINAMATH_CALUDE_at_least_95_buildings_collapsed_l4120_412029

/-- Represents the number of buildings that collapsed in each earthquake --/
structure EarthquakeCollapses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Theorem stating that at least 95 buildings collapsed after five earthquakes --/
theorem at_least_95_buildings_collapsed
  (initial_buildings : ℕ)
  (collapses : EarthquakeCollapses)
  (h_initial : initial_buildings = 100)
  (h_first : collapses.first = 5)
  (h_second : collapses.second = 6)
  (h_third : collapses.third = 13)
  (h_fourth : collapses.fourth = 24)
  (h_handful : ∀ n : ℕ, n ≤ 5 → n ≤ initial_buildings - (collapses.first + collapses.second + collapses.third + collapses.fourth)) :
  95 ≤ collapses.first + collapses.second + collapses.third + collapses.fourth :=
sorry

end NUMINAMATH_CALUDE_at_least_95_buildings_collapsed_l4120_412029


namespace NUMINAMATH_CALUDE_volume_of_four_cubes_l4120_412042

theorem volume_of_four_cubes (edge_length : ℝ) (num_boxes : ℕ) : 
  edge_length = 5 → num_boxes = 4 → num_boxes * (edge_length ^ 3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_four_cubes_l4120_412042


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4120_412033

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → 
  x₂^2 - 3*x₂ - 1 = 0 → 
  x₁^2 + x₂^2 = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4120_412033


namespace NUMINAMATH_CALUDE_fibonacci_6_l4120_412094

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_6 : fibonacci 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_6_l4120_412094


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4120_412032

theorem partial_fraction_decomposition :
  ∃ (A B C : ℝ),
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 →
      (5 * x + 2) / ((x - 2) * (x - 4)^2) =
      A / (x - 2) + B / (x - 4) + C / (x - 4)^2) ∧
    A = 3 ∧ B = -3 ∧ C = 11 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4120_412032


namespace NUMINAMATH_CALUDE_expression_evaluation_l4120_412075

theorem expression_evaluation (x y z : ℚ) (hx : x = 5) (hy : y = 4) (hz : z = 3) :
  (1/y + 1/z) / (1/x) = 35/12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4120_412075


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l4120_412095

theorem isosceles_triangle_proof (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_equation : 2 * Real.sin A * Real.cos B = Real.sin C) : A = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_proof_l4120_412095


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4120_412005

theorem arithmetic_calculation : (18 / (8 - 2 * 3)) + 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4120_412005


namespace NUMINAMATH_CALUDE_orange_juice_serving_size_l4120_412000

/-- Given the conditions for preparing orange juice, prove that each serving is 6 ounces. -/
theorem orange_juice_serving_size :
  -- Conditions
  (concentrate_to_water_ratio : ℚ) →
  (concentrate_cans : ℕ) →
  (concentrate_size : ℚ) →
  (total_servings : ℕ) →
  -- Assumptions
  concentrate_to_water_ratio = 1 / 4 →
  concentrate_cans = 45 →
  concentrate_size = 12 →
  total_servings = 360 →
  -- Conclusion
  (total_volume : ℚ) →
  total_volume = concentrate_cans * concentrate_size * (1 + 1 / concentrate_to_water_ratio) →
  total_volume / total_servings = 6 :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_serving_size_l4120_412000


namespace NUMINAMATH_CALUDE_x_coordinate_of_first_point_l4120_412039

/-- Given a line with equation x = 2y + 3 and two points (m, n) and (m + 2, n + 1) on this line,
    prove that the x-coordinate of the first point, m, is equal to 2n + 3. -/
theorem x_coordinate_of_first_point
  (m n : ℝ)
  (h1 : m = 2 * n + 3)
  (h2 : m + 2 = 2 * (n + 1) + 3) :
  m = 2 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_of_first_point_l4120_412039


namespace NUMINAMATH_CALUDE_exactly_one_correct_l4120_412081

/-- The probability that exactly one of three independent events occurs, given their individual probabilities -/
theorem exactly_one_correct (pA pB pC : ℝ) 
  (hA : 0 ≤ pA ∧ pA ≤ 1) 
  (hB : 0 ≤ pB ∧ pB ≤ 1) 
  (hC : 0 ≤ pC ∧ pC ≤ 1) 
  (hpA : pA = 3/4) 
  (hpB : pB = 2/3) 
  (hpC : pC = 2/3) : 
  pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC = 7/36 := by
  sorry

#check exactly_one_correct

end NUMINAMATH_CALUDE_exactly_one_correct_l4120_412081


namespace NUMINAMATH_CALUDE_cylinder_volume_from_lateral_surface_l4120_412036

/-- The volume of a cylinder whose lateral surface is a square with side length 2 * (π^(1/3)) is 2 -/
theorem cylinder_volume_from_lateral_surface (π : ℝ) (h : π > 0) :
  let lateral_surface_side := 2 * π^(1/3)
  let cylinder_height := lateral_surface_side
  let cylinder_radius := lateral_surface_side / (2 * π)
  let cylinder_volume := π * cylinder_radius^2 * cylinder_height
  cylinder_volume = 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_lateral_surface_l4120_412036


namespace NUMINAMATH_CALUDE_eleventh_term_of_arithmetic_sequence_l4120_412007

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem eleventh_term_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_fifth : a 5 = 3 / 8)
  (h_seventeenth : a 17 = 7 / 12) :
  a 11 = 23 / 48 :=
sorry

end NUMINAMATH_CALUDE_eleventh_term_of_arithmetic_sequence_l4120_412007


namespace NUMINAMATH_CALUDE_officers_from_six_people_l4120_412078

/-- The number of ways to choose three distinct officers from a group of 6 people -/
def choose_officers (n : ℕ) : ℕ :=
  if n ≥ 3 then n * (n - 1) * (n - 2) else 0

/-- Theorem stating that choosing three distinct officers from 6 people results in 120 ways -/
theorem officers_from_six_people :
  choose_officers 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_officers_from_six_people_l4120_412078


namespace NUMINAMATH_CALUDE_division_remainder_l4120_412063

theorem division_remainder (n : ℕ) : 
  (n / 7 = 5) ∧ (n % 7 = 0) → n % 11 = 2 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_l4120_412063


namespace NUMINAMATH_CALUDE_star_equation_solution_l4120_412035

-- Define the star operation
def star (a b : ℝ) : ℝ := 2*a*b + 3*b - 2*a

-- Theorem statement
theorem star_equation_solution :
  ∀ x : ℝ, star 3 x = 60 → x = 22/3 :=
by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l4120_412035


namespace NUMINAMATH_CALUDE_thirty_minus_twelve_base5_l4120_412080

/-- Converts a natural number to its base 5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Theorem: 30 in base 10 minus 12 in base 10 equals 33 in base 5 --/
theorem thirty_minus_twelve_base5 : toBase5 (30 - 12) = [3, 3] := by
  sorry

end NUMINAMATH_CALUDE_thirty_minus_twelve_base5_l4120_412080


namespace NUMINAMATH_CALUDE_expansion_sum_theorem_l4120_412089

theorem expansion_sum_theorem (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (3*x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₁/3 + a₂/3^2 + a₃/3^3 + a₄/3^4 + a₅/3^5 + a₆/3^6 + a₇/3^7 + a₈/3^8 + a₉/3^9 = 511 := by
sorry

end NUMINAMATH_CALUDE_expansion_sum_theorem_l4120_412089


namespace NUMINAMATH_CALUDE_number_of_teams_in_league_l4120_412059

theorem number_of_teams_in_league : ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_of_teams_in_league_l4120_412059


namespace NUMINAMATH_CALUDE_probability_all_truth_l4120_412011

theorem probability_all_truth (pA pB pC pD : ℝ) 
  (hA : pA = 0.55) 
  (hB : pB = 0.60) 
  (hC : pC = 0.45) 
  (hD : pD = 0.70) : 
  pA * pB * pC * pD = 0.10395 := by
sorry

end NUMINAMATH_CALUDE_probability_all_truth_l4120_412011


namespace NUMINAMATH_CALUDE_set_with_unique_gcd_divisor_has_power_of_two_elements_l4120_412027

theorem set_with_unique_gcd_divisor_has_power_of_two_elements 
  (S : Finset ℕ+) 
  (h : ∀ (s : ℕ+) (d : ℕ+), s ∈ S → d ∣ s → ∃! (t : ℕ+), t ∈ S ∧ Nat.gcd s t = d) :
  ∃ (k : ℕ), Finset.card S = 2^k :=
sorry

end NUMINAMATH_CALUDE_set_with_unique_gcd_divisor_has_power_of_two_elements_l4120_412027


namespace NUMINAMATH_CALUDE_erased_number_proof_l4120_412076

theorem erased_number_proof (b : ℕ) (x : ℕ) : 
  3 ≤ b →
  (b - 2) * (b + 3) / 2 - x = 1015 * (b - 3) / 19 →
  x = 805 :=
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l4120_412076


namespace NUMINAMATH_CALUDE_log_equation_solution_l4120_412041

theorem log_equation_solution (p q : ℝ) (c : ℝ) (h : 0 < p ∧ 0 < q) :
  (Real.log p^2 / Real.log 10 = c - 2 * Real.log q / Real.log 10) →
  p = 10^(c/2) / q :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4120_412041


namespace NUMINAMATH_CALUDE_problems_per_page_l4120_412057

/-- Given the total number of homework problems, the number of finished problems,
    and the number of remaining pages, calculate the number of problems per page. -/
theorem problems_per_page
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : remaining_pages = 2)
  (h4 : remaining_pages > 0)
  (h5 : finished_problems ≤ total_problems) :
  (total_problems - finished_problems) / remaining_pages = 7 := by
sorry

end NUMINAMATH_CALUDE_problems_per_page_l4120_412057


namespace NUMINAMATH_CALUDE_add_squares_l4120_412054

theorem add_squares (a : ℝ) : 2 * a^2 + a^2 = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_add_squares_l4120_412054


namespace NUMINAMATH_CALUDE_gamma_value_l4120_412047

/-- Given that γ is directly proportional to the square of δ, 
    and γ = 25 when δ = 5, prove that γ = 64 when δ = 8 -/
theorem gamma_value (γ δ : ℝ) (h1 : ∃ (k : ℝ), ∀ x, γ = k * x^2) 
  (h2 : γ = 25 ∧ δ = 5) : 
  (δ = 8 → γ = 64) := by
  sorry


end NUMINAMATH_CALUDE_gamma_value_l4120_412047


namespace NUMINAMATH_CALUDE_parabola_vertex_sum_max_l4120_412012

theorem parabola_vertex_sum_max (a U : ℤ) (h_U : U ≠ 0) : 
  let parabola (x y : ℝ) := ∃ b c : ℝ, y = a * x^2 + b * x + c
  let passes_through (x y : ℝ) := parabola x y
  let N := (3 * U / 2 : ℝ) + (- 9 * a * U^2 / 4 : ℝ)
  (passes_through 0 0) ∧ 
  (passes_through (3 * U) 0) ∧ 
  (passes_through (3 * U - 1) 12) →
  N ≤ 71/4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_sum_max_l4120_412012


namespace NUMINAMATH_CALUDE_contractor_problem_l4120_412048

/-- Represents the initial number of people hired by the contractor -/
def initial_people : ℕ := 10

/-- Represents the total number of days allocated for the job -/
def total_days : ℕ := 100

/-- Represents the number of days worked before firing people -/
def days_before_firing : ℕ := 20

/-- Represents the fraction of work completed before firing people -/
def work_fraction_before_firing : ℚ := 1/4

/-- Represents the number of people fired -/
def people_fired : ℕ := 2

/-- Represents the number of days needed to complete the job after firing people -/
def days_after_firing : ℕ := 75

theorem contractor_problem :
  ∃ (p : ℕ), 
    p = initial_people ∧
    p * days_before_firing = work_fraction_before_firing * (p * total_days) ∧
    (p - people_fired) * days_after_firing = (1 - work_fraction_before_firing) * (p * total_days) :=
by sorry

end NUMINAMATH_CALUDE_contractor_problem_l4120_412048


namespace NUMINAMATH_CALUDE_pink_highlighters_count_l4120_412069

theorem pink_highlighters_count (total yellow blue : ℕ) (h1 : total = 15) (h2 : yellow = 7) (h3 : blue = 5) :
  ∃ pink : ℕ, pink + yellow + blue = total ∧ pink = 3 := by
  sorry

end NUMINAMATH_CALUDE_pink_highlighters_count_l4120_412069


namespace NUMINAMATH_CALUDE_intersection_line_equation_l4120_412061

/-- Given two lines l₁ and l₂, if a line l intersects both l₁ and l₂ such that the midpoint
    of the segment cut off by l₁ and l₂ is at the origin, then l has the equation x + 6y = 0 -/
theorem intersection_line_equation (x y : ℝ) : 
  let l₁ := {(x, y) : ℝ × ℝ | 4*x + y + 6 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | 3*x - 5*y - 6 = 0}
  let midpoint := (0, 0)
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₂ ∧
    (x₁ + x₂) / 2 = midpoint.1 ∧ (y₁ + y₂) / 2 = midpoint.2 →
  x + 6*y = 0 := by
sorry


end NUMINAMATH_CALUDE_intersection_line_equation_l4120_412061


namespace NUMINAMATH_CALUDE_consecutive_integer_fraction_minimum_l4120_412015

theorem consecutive_integer_fraction_minimum (a b : ℤ) (h1 : a = b + 1) (h2 : a > b) :
  ∀ ε > 0, ∃ a b : ℤ, a = b + 1 ∧ a > b ∧ (a + b : ℚ) / (a - b) + (a - b : ℚ) / (a + b) < 2 + ε ∧
  ∀ a' b' : ℤ, a' = b' + 1 → a' > b' → 2 ≤ (a' + b' : ℚ) / (a' - b') + (a' - b' : ℚ) / (a' + b') :=
sorry

end NUMINAMATH_CALUDE_consecutive_integer_fraction_minimum_l4120_412015


namespace NUMINAMATH_CALUDE_ophelia_age_l4120_412052

/-- Given the following conditions:
  1. In 10 years, Ophelia will be thrice as old as Lennon.
  2. In 10 years, Mike will be twice the age difference between Ophelia and Lennon.
  3. Lennon is currently 8 years old.
  4. Mike is currently 5 years older than Lennon.
Prove that Ophelia's current age is 44 years. -/
theorem ophelia_age (lennon_age : ℕ) (mike_age : ℕ) (ophelia_age : ℕ) :
  lennon_age = 8 →
  mike_age = lennon_age + 5 →
  ophelia_age + 10 = 3 * (lennon_age + 10) →
  mike_age + 10 = 2 * ((ophelia_age + 10) - (lennon_age + 10)) →
  ophelia_age = 44 := by
  sorry

end NUMINAMATH_CALUDE_ophelia_age_l4120_412052


namespace NUMINAMATH_CALUDE_area_probability_l4120_412002

/-- A square in a 2D plane -/
structure Square :=
  (A B C D : ℝ × ℝ)

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a point is inside a square -/
def isInside (s : Square) (p : Point) : Prop := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- The probability of an event occurring when a point is chosen randomly inside a square -/
def probability (s : Square) (event : Point → Prop) : ℝ := sorry

/-- The main theorem -/
theorem area_probability (s : Square) :
  probability s (fun p => 
    isInside s p ∧ 
    triangleArea s.A s.B p > triangleArea s.B s.C p ∧
    triangleArea s.A s.B p > triangleArea s.C s.D p ∧
    triangleArea s.A s.B p > triangleArea s.D s.A p) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_area_probability_l4120_412002


namespace NUMINAMATH_CALUDE_existence_of_zero_crossing_l4120_412004

open Function Set

theorem existence_of_zero_crossing (a b : ℝ) (h : a < b) :
  ∃ (f : ℝ → ℝ), ContinuousOn f (Icc a b) ∧ 
  f a * f b > 0 ∧ 
  ∃ c ∈ Ioo a b, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_zero_crossing_l4120_412004


namespace NUMINAMATH_CALUDE_break_even_components_min_profitable_components_l4120_412018

/-- The number of components produced and sold monthly -/
def components : ℕ := 150

/-- Production cost per component -/
def production_cost : ℚ := 80

/-- Shipping cost per component -/
def shipping_cost : ℚ := 5

/-- Fixed monthly costs -/
def fixed_costs : ℚ := 16500

/-- Minimum selling price per component -/
def selling_price : ℚ := 195

/-- Theorem stating that the number of components produced and sold monthly
    is the break-even point where costs equal revenues -/
theorem break_even_components :
  (selling_price * components : ℚ) = 
  fixed_costs + (production_cost + shipping_cost) * components := by
  sorry

/-- Theorem stating that the number of components is the minimum
    where revenues are not less than costs -/
theorem min_profitable_components :
  ∀ n : ℕ, n < components → 
  (selling_price * n : ℚ) < fixed_costs + (production_cost + shipping_cost) * n := by
  sorry

end NUMINAMATH_CALUDE_break_even_components_min_profitable_components_l4120_412018


namespace NUMINAMATH_CALUDE_binary_op_example_l4120_412006

def binary_op (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem binary_op_example : binary_op 4 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_op_example_l4120_412006


namespace NUMINAMATH_CALUDE_negation_of_implication_is_false_l4120_412024

theorem negation_of_implication_is_false : 
  ¬(∃ a b : ℝ, (a ≤ 1 ∨ b ≤ 1) ∧ (a + b ≤ 2)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_is_false_l4120_412024


namespace NUMINAMATH_CALUDE_min_product_of_three_l4120_412003

def S : Finset Int := {-9, -7, -1, 2, 4, 6, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x * y * z = -432 ∧ (∀ (p q r : Int), p ∈ S → q ∈ S → r ∈ S →
  p ≠ q → q ≠ r → p ≠ r → p * q * r ≥ -432) :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l4120_412003


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l4120_412097

theorem right_triangle_side_length : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  c = 13 → a = 12 →
  c^2 = a^2 + b^2 →
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l4120_412097


namespace NUMINAMATH_CALUDE_total_books_collected_l4120_412044

def books_first_week : ℕ := 9
def weeks_collecting : ℕ := 6
def multiplier : ℕ := 10

theorem total_books_collected :
  (books_first_week + (weeks_collecting - 1) * (books_first_week * multiplier)) = 459 :=
by sorry

end NUMINAMATH_CALUDE_total_books_collected_l4120_412044


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l4120_412071

theorem opposite_of_negative_three :
  ∀ x : ℤ, ((-3 : ℤ) + x = 0) → x = 3 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l4120_412071


namespace NUMINAMATH_CALUDE_machinery_expenditure_l4120_412088

theorem machinery_expenditure (total : ℝ) (raw_materials : ℝ) (cash_percentage : ℝ) :
  total = 137500 →
  raw_materials = 80000 →
  cash_percentage = 0.20 →
  ∃ machinery : ℝ,
    machinery = 30000 ∧
    raw_materials + machinery + (cash_percentage * total) = total :=
by sorry

end NUMINAMATH_CALUDE_machinery_expenditure_l4120_412088


namespace NUMINAMATH_CALUDE_greatest_k_value_l4120_412074

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = 10) →
  k ≤ 2 * Real.sqrt 33 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_value_l4120_412074


namespace NUMINAMATH_CALUDE_conditional_probability_balls_l4120_412037

/-- Represents the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of event A (drawing two balls of different colors) -/
def probA : ℚ := (choose 5 1 * choose 3 1 + choose 5 1 * choose 4 1 + choose 3 1 * choose 4 1) / choose 12 2

/-- The probability of event B (drawing one yellow and one blue ball) -/
def probB : ℚ := (choose 5 1 * choose 4 1) / choose 12 2

/-- The probability of both events A and B occurring -/
def probAB : ℚ := probB

theorem conditional_probability_balls :
  probAB / probA = 20 / 47 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_balls_l4120_412037


namespace NUMINAMATH_CALUDE_man_crossing_street_speed_l4120_412046

/-- Proves that a man crossing a 600 m street in 5 minutes has a speed of 7.2 km/h -/
theorem man_crossing_street_speed :
  let distance_m : ℝ := 600
  let time_min : ℝ := 5
  let distance_km : ℝ := distance_m / 1000
  let time_h : ℝ := time_min / 60
  let speed_km_h : ℝ := distance_km / time_h
  speed_km_h = 7.2 := by sorry

end NUMINAMATH_CALUDE_man_crossing_street_speed_l4120_412046


namespace NUMINAMATH_CALUDE_inequality_proof_l4120_412067

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ ≥ a₂ ∧ a₂ ≥ a₃ ∧ a₃ > 0)
  (hb : b₁ ≥ b₂ ∧ b₂ ≥ b₃ ∧ b₃ > 0)
  (hab : a₁ * a₂ * a₃ = b₁ * b₂ * b₃)
  (hdiff : a₁ - a₃ ≤ b₁ - b₃) :
  a₁ + a₂ + a₃ ≤ 2 * (b₁ + b₂ + b₃) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4120_412067


namespace NUMINAMATH_CALUDE_lucy_had_twenty_l4120_412082

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := sorry

/-- The amount of money Linda originally had -/
def linda_original : ℕ := 10

/-- Proposition that if Lucy gives Linda $5, they would have the same amount of money -/
def equal_after_transfer : Prop :=
  lucy_original - 5 = linda_original + 5

theorem lucy_had_twenty :
  lucy_original = 20 :=
by sorry

end NUMINAMATH_CALUDE_lucy_had_twenty_l4120_412082


namespace NUMINAMATH_CALUDE_basketball_team_min_score_l4120_412084

theorem basketball_team_min_score (n : ℕ) (min_score max_score : ℕ) 
  (h1 : n = 12) 
  (h2 : min_score = 7) 
  (h3 : max_score = 23) 
  (h4 : ∀ player_score, min_score ≤ player_score ∧ player_score ≤ max_score) : 
  n * min_score + (max_score - min_score) = 100 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_min_score_l4120_412084


namespace NUMINAMATH_CALUDE_valentines_day_theorem_l4120_412019

theorem valentines_day_theorem (boys girls : ℕ) : 
  boys * girls = boys + girls + 36 → boys * girls = 76 := by
  sorry

end NUMINAMATH_CALUDE_valentines_day_theorem_l4120_412019


namespace NUMINAMATH_CALUDE_square_sum_xy_l4120_412096

theorem square_sum_xy (x y a b : ℝ) 
  (h1 : x * y = b) 
  (h2 : 1 / (x^2) + 1 / (y^2) = a) : 
  (x + y)^2 = b * (a * b + 2) := by
sorry

end NUMINAMATH_CALUDE_square_sum_xy_l4120_412096


namespace NUMINAMATH_CALUDE_find_m_and_n_min_value_sum_equality_condition_l4120_412062

-- Define the solution set
def solution_set (x : ℝ) := 0 ≤ x ∧ x ≤ 4

-- Define the inequality
def inequality (x m n : ℝ) := |x - m| ≤ n

-- Theorem 1: Find m and n
theorem find_m_and_n (m n : ℝ) 
  (h : ∀ x, inequality x m n ↔ solution_set x) : 
  m = 2 ∧ n = 2 := by sorry

-- Theorem 2: Minimum value of a + b
theorem min_value_sum (m n a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : a + b = m / a + n / b) 
  (h4 : m = 2 ∧ n = 2) : 
  a + b ≥ 2 * Real.sqrt 2 := by sorry

-- Theorem 3: Condition for equality
theorem equality_condition (m n a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : a + b = m / a + n / b) 
  (h4 : m = 2 ∧ n = 2) 
  (h5 : a + b = 2 * Real.sqrt 2) : 
  a = Real.sqrt 2 ∧ b = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_find_m_and_n_min_value_sum_equality_condition_l4120_412062


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_decimal_l4120_412020

-- Define the repeating decimal 0.888...
def repeating_decimal : ℚ := 8 / 9

-- State the theorem
theorem eight_divided_by_repeating_decimal : 8 / repeating_decimal = 9 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_decimal_l4120_412020


namespace NUMINAMATH_CALUDE_intersection_has_one_element_l4120_412065

theorem intersection_has_one_element (a : ℝ) : 
  let A := {x : ℝ | 2^(1+x) + 2^(1-x) = a}
  let B := {y : ℝ | ∃ θ : ℝ, y = Real.sin θ}
  (∃! x : ℝ, x ∈ A ∩ B) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_has_one_element_l4120_412065


namespace NUMINAMATH_CALUDE_speed_is_48_l4120_412079

-- Define the duration of the drive in hours
def drive_duration : ℚ := 7/4

-- Define the distance driven in km
def distance_driven : ℚ := 84

-- Theorem stating that the speed is 48 km/h
theorem speed_is_48 : distance_driven / drive_duration = 48 := by
  sorry

end NUMINAMATH_CALUDE_speed_is_48_l4120_412079


namespace NUMINAMATH_CALUDE_composite_face_dots_l4120_412083

/-- Represents a single die face -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the four faces of interest in the composite figure -/
inductive CompositeFace
  | A
  | B
  | C
  | D

/-- A function that returns the number of dots on a die face -/
def dots_on_face (face : DieFace) : Nat :=
  match face with
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

/-- A function that maps a composite face to its corresponding die face -/
def composite_to_die_face (face : CompositeFace) : DieFace :=
  match face with
  | CompositeFace.A => DieFace.three
  | CompositeFace.B => DieFace.five
  | CompositeFace.C => DieFace.six
  | CompositeFace.D => DieFace.five

/-- Theorem stating the number of dots on each composite face -/
theorem composite_face_dots (face : CompositeFace) :
  dots_on_face (composite_to_die_face face) =
    match face with
    | CompositeFace.A => 3
    | CompositeFace.B => 5
    | CompositeFace.C => 6
    | CompositeFace.D => 5 := by
  sorry

end NUMINAMATH_CALUDE_composite_face_dots_l4120_412083


namespace NUMINAMATH_CALUDE_reporters_not_covering_politics_l4120_412001

theorem reporters_not_covering_politics 
  (local_politics_coverage : Real) 
  (non_local_politics_ratio : Real) 
  (h1 : local_politics_coverage = 0.12)
  (h2 : non_local_politics_ratio = 0.4) :
  1 - (local_politics_coverage / (1 - non_local_politics_ratio)) = 0.8 := by
sorry

end NUMINAMATH_CALUDE_reporters_not_covering_politics_l4120_412001


namespace NUMINAMATH_CALUDE_counterexample_exists_l4120_412087

theorem counterexample_exists : ∃ (a b c : ℝ), 0 < a ∧ a < b ∧ b < c ∧ a ≥ b * c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4120_412087


namespace NUMINAMATH_CALUDE_rectangular_box_width_l4120_412056

/-- The width of a rectangular box that fits in a wooden box -/
theorem rectangular_box_width :
  let wooden_box_length : ℝ := 8 -- in meters
  let wooden_box_width : ℝ := 7 -- in meters
  let wooden_box_height : ℝ := 6 -- in meters
  let rect_box_length : ℝ := 4 / 100 -- in meters
  let rect_box_height : ℝ := 6 / 100 -- in meters
  let max_boxes : ℕ := 2000000
  ∃ (w : ℝ),
    w > 0 ∧
    (wooden_box_length * wooden_box_width * wooden_box_height) / 
    (rect_box_length * w * rect_box_height) = max_boxes ∧
    w = 7 / 100 -- width in meters
  := by sorry


end NUMINAMATH_CALUDE_rectangular_box_width_l4120_412056


namespace NUMINAMATH_CALUDE_triangle_centroid_coordinates_l4120_412085

/-- The centroid of a triangle with vertices (2, 8), (6, 2), and (0, 4) has coordinates (8/3, 14/3). -/
theorem triangle_centroid_coordinates :
  let A : ℝ × ℝ := (2, 8)
  let B : ℝ × ℝ := (6, 2)
  let C : ℝ × ℝ := (0, 4)
  let centroid : ℝ × ℝ := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  centroid = (8/3, 14/3) := by
sorry

end NUMINAMATH_CALUDE_triangle_centroid_coordinates_l4120_412085


namespace NUMINAMATH_CALUDE_two_unit_circles_tangent_to_two_three_l4120_412092

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- A circle is externally tangent to two other circles -/
def externally_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  externally_tangent c c1 ∧ externally_tangent c c2

theorem two_unit_circles_tangent_to_two_three (c1 c2 : Circle)
  (h1 : c1.radius = 2)
  (h2 : c2.radius = 3)
  (h3 : externally_tangent c1 c2) :
  ∃! (s : Finset Circle), s.card = 2 ∧ ∀ c ∈ s, c.radius = 1 ∧ externally_tangent_to_both c c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_two_unit_circles_tangent_to_two_three_l4120_412092


namespace NUMINAMATH_CALUDE_sin_10_over_1_minus_sqrt3_tan_10_l4120_412031

theorem sin_10_over_1_minus_sqrt3_tan_10 :
  (Real.sin (10 * π / 180)) / (1 - Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_10_over_1_minus_sqrt3_tan_10_l4120_412031


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l4120_412045

/-- Given a square with perimeter 48 meters, its area is 144 square meters. -/
theorem square_area_from_perimeter :
  ∀ s : ℝ,
  s > 0 →
  4 * s = 48 →
  s * s = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l4120_412045


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4120_412013

theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_inverse : ∃ k : ℝ, ∀ x y, x^2 * y = k) 
  (h_initial : 3^2 * 8 = 9 * 8) 
  (h_final : y = 648) : x = 1/3 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4120_412013


namespace NUMINAMATH_CALUDE_coloring_books_sold_l4120_412009

theorem coloring_books_sold (initial_stock : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) : initial_stock = 87 → shelves = 9 → books_per_shelf = 6 → initial_stock - (shelves * books_per_shelf) = 33 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_sold_l4120_412009


namespace NUMINAMATH_CALUDE_problem_statement_l4120_412072

theorem problem_statement (a x y : ℝ) (h1 : a ≠ x) (h2 : a ≠ y) (h3 : x ≠ y)
  (h4 : Real.sqrt (a * (x - a)) + Real.sqrt (a * (y - a)) = Real.sqrt (x - a) - Real.sqrt (a - y)) :
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4120_412072


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4120_412066

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h5 : a 5 = 15) :
  a 3 + a 4 + a 7 + a 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4120_412066


namespace NUMINAMATH_CALUDE_people_arrangement_l4120_412014

/-- Given a total of 1600 people and columns of 85 people each, prove:
    1. The number of complete columns
    2. The number of people in the incomplete column
    3. The total number of rows
    4. The row in which the last person stands -/
theorem people_arrangement (total_people : ℕ) (people_per_column : ℕ) 
    (h1 : total_people = 1600)
    (h2 : people_per_column = 85) :
    let complete_columns := total_people / people_per_column
    let remaining_people := total_people % people_per_column
    (complete_columns = 18) ∧ 
    (remaining_people = 70) ∧
    (remaining_people = 70) ∧
    (remaining_people = 70) := by
  sorry

end NUMINAMATH_CALUDE_people_arrangement_l4120_412014


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l4120_412028

theorem negation_of_universal_quantifier (a : ℝ) :
  (¬ ∀ x > 0, Real.log x = a) ↔ (∃ x > 0, Real.log x ≠ a) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l4120_412028


namespace NUMINAMATH_CALUDE_waste_paper_collection_l4120_412038

/-- Proves that given the conditions of the waste paper collection problem,
    Vitya collected 15 kg and Vova collected 12 kg. -/
theorem waste_paper_collection :
  ∀ (v w : ℕ),
  v + w = 27 →
  5 * v + 3 * w = 111 →
  v = 15 ∧ w = 12 := by
sorry

end NUMINAMATH_CALUDE_waste_paper_collection_l4120_412038


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l4120_412016

-- Define the complex number z
def z : ℂ := sorry

-- Define the condition i · z = 1 - 2i
axiom condition : Complex.I * z = 1 - 2 * Complex.I

-- Theorem to prove
theorem z_in_third_quadrant : (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l4120_412016


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l4120_412055

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℤ => (n - 3) * (n + 5) * (n + 9) < 0)
    (Finset.Icc (-13 : ℤ) 13)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l4120_412055


namespace NUMINAMATH_CALUDE_spinner_final_direction_l4120_412030

-- Define the directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation function
def rotate (initial : Direction) (clockwise : Rat) (counterclockwise : Rat) : Direction :=
  sorry

-- Theorem statement
theorem spinner_final_direction :
  rotate Direction.South (7/2 : Rat) (7/4 : Rat) = Direction.East :=
sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l4120_412030


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sevens_l4120_412091

def set_of_sevens : List ℕ := [7, 77, 777, 7777, 77777, 777777, 7777777, 77777777, 777777777]

theorem arithmetic_mean_of_sevens :
  let sum := set_of_sevens.sum
  let count := set_of_sevens.length
  sum / count = 96308641 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sevens_l4120_412091


namespace NUMINAMATH_CALUDE_sum_of_equations_l4120_412060

theorem sum_of_equations (a b c d : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 4)
  (eq4 : d - a + b = 1) :
  2*a + 2*b + 2*c + 2*d = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_equations_l4120_412060


namespace NUMINAMATH_CALUDE_apple_basket_problem_l4120_412040

theorem apple_basket_problem (small_basket_capacity : ℕ) (small_basket_count : ℕ) 
  (large_basket_count : ℕ) (leftover_weight : ℕ) :
  small_basket_capacity = 25 →
  small_basket_count = 28 →
  large_basket_count = 10 →
  leftover_weight = 50 →
  (small_basket_capacity * small_basket_count - leftover_weight) / large_basket_count = 65 := by
  sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l4120_412040


namespace NUMINAMATH_CALUDE_expected_full_circles_l4120_412058

/-- Represents the tiling of an equilateral triangle -/
structure TriangleTiling where
  n : ℕ
  sideLength : n > 2

/-- Expected number of full circles in a triangle tiling -/
def expectedFullCircles (t : TriangleTiling) : ℚ :=
  (t.n - 2) * (t.n - 1) / 1458

/-- Theorem stating the expected number of full circles in a triangle tiling -/
theorem expected_full_circles (t : TriangleTiling) :
  expectedFullCircles t = (t.n - 2) * (t.n - 1) / 1458 :=
by sorry

end NUMINAMATH_CALUDE_expected_full_circles_l4120_412058


namespace NUMINAMATH_CALUDE_jessie_score_is_30_l4120_412077

-- Define the scoring system
def correct_points : ℚ := 2
def incorrect_points : ℚ := -0.5
def unanswered_points : ℚ := 0

-- Define Jessie's answers
def correct_answers : ℕ := 16
def incorrect_answers : ℕ := 4
def unanswered_questions : ℕ := 10

-- Define Jessie's score calculation
def jessie_score : ℚ :=
  (correct_answers : ℚ) * correct_points +
  (incorrect_answers : ℚ) * incorrect_points +
  (unanswered_questions : ℚ) * unanswered_points

-- Theorem to prove
theorem jessie_score_is_30 : jessie_score = 30 := by
  sorry

end NUMINAMATH_CALUDE_jessie_score_is_30_l4120_412077


namespace NUMINAMATH_CALUDE_basket_balls_count_l4120_412049

/-- Given a basket of balls where the ratio of white to red balls is 5:3 and there are 15 white balls, prove that there are 9 red balls. -/
theorem basket_balls_count (white_balls : ℕ) (red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 5 / 3 → white_balls = 15 → red_balls = 9 := by
  sorry

end NUMINAMATH_CALUDE_basket_balls_count_l4120_412049


namespace NUMINAMATH_CALUDE_square_root_plus_square_eq_zero_l4120_412023

theorem square_root_plus_square_eq_zero (x y : ℝ) :
  Real.sqrt (x + 2) + (x + y)^2 = 0 → x^2 - x*y = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_root_plus_square_eq_zero_l4120_412023


namespace NUMINAMATH_CALUDE_circle_is_point_l4120_412021

/-- The equation of the supposed circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 2*y + 5 = 0

/-- The center of the supposed circle -/
def center : ℝ × ℝ := (-2, 1)

theorem circle_is_point :
  ∀ (x y : ℝ), circle_equation x y ↔ (x, y) = center :=
sorry

end NUMINAMATH_CALUDE_circle_is_point_l4120_412021


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l4120_412034

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 5 * n ≡ 980 [ZMOD 33] ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬(5 * m ≡ 980 [ZMOD 33])) ↔ n = 19 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l4120_412034


namespace NUMINAMATH_CALUDE_inequality_proof_l4120_412017

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  x^2 + y^2 + z^2 + x*y + y*z + z*x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4120_412017
