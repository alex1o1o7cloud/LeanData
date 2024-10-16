import Mathlib

namespace NUMINAMATH_CALUDE_tim_bought_three_goats_l2504_250458

/-- Proves that Tim bought 3 goats given the conditions of the problem -/
theorem tim_bought_three_goats
  (goat_cost : ℕ)
  (llama_count : ℕ → ℕ)
  (llama_cost : ℕ → ℕ)
  (total_spent : ℕ)
  (h1 : goat_cost = 400)
  (h2 : ∀ g, llama_count g = 2 * g)
  (h3 : ∀ g, llama_cost g = goat_cost + goat_cost / 2)
  (h4 : total_spent = 4800)
  (h5 : ∀ g, total_spent = g * goat_cost + llama_count g * llama_cost g) :
  ∃ g : ℕ, g = 3 ∧ total_spent = g * goat_cost + llama_count g * llama_cost g :=
sorry

end NUMINAMATH_CALUDE_tim_bought_three_goats_l2504_250458


namespace NUMINAMATH_CALUDE_jenny_calculation_l2504_250467

theorem jenny_calculation (x : ℚ) : (x - 14) / 5 = 11 → (x - 5) / 7 = 64/7 := by
  sorry

end NUMINAMATH_CALUDE_jenny_calculation_l2504_250467


namespace NUMINAMATH_CALUDE_fraction_simplification_l2504_250499

theorem fraction_simplification : (5 * 8) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2504_250499


namespace NUMINAMATH_CALUDE_fudge_pan_dimension_l2504_250497

/-- Represents a rectangular pan of fudge --/
structure FudgePan where
  side1 : ℕ
  side2 : ℕ
  pieces : ℕ

/-- Theorem stating the relationship between pan dimensions and number of fudge pieces --/
theorem fudge_pan_dimension (pan : FudgePan) 
  (h1 : pan.side1 = 18)
  (h2 : pan.pieces = 522) :
  pan.side2 = 29 := by
  sorry

#check fudge_pan_dimension

end NUMINAMATH_CALUDE_fudge_pan_dimension_l2504_250497


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017_l2504_250491

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + m) - a n = m * (a 1 - a 0)

theorem arithmetic_sequence_2017 (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 2016 + a 2018 = π) : 
  a 2017 = π / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017_l2504_250491


namespace NUMINAMATH_CALUDE_total_marbles_l2504_250413

/-- The number of marbles of each color in a collection --/
structure MarbleCollection where
  red : ℝ
  blue : ℝ
  green : ℝ
  yellow : ℝ

/-- The conditions given in the problem --/
def satisfiesConditions (m : MarbleCollection) : Prop :=
  m.red = 1.3 * m.blue ∧
  m.green = 1.7 * m.red ∧
  m.yellow = m.blue + 40

/-- The theorem to be proved --/
theorem total_marbles (m : MarbleCollection) (h : satisfiesConditions m) :
  m.red + m.blue + m.green + m.yellow = 3.84615 * m.red + 40 := by
  sorry


end NUMINAMATH_CALUDE_total_marbles_l2504_250413


namespace NUMINAMATH_CALUDE_twelve_sixteen_twenty_pythagorean_triple_l2504_250449

/-- A Pythagorean triple is a set of three positive integers (a, b, c) that satisfy a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The set {12, 16, 20} is a Pythagorean triple -/
theorem twelve_sixteen_twenty_pythagorean_triple :
  isPythagoreanTriple 12 16 20 := by
  sorry

end NUMINAMATH_CALUDE_twelve_sixteen_twenty_pythagorean_triple_l2504_250449


namespace NUMINAMATH_CALUDE_coefficient_equals_nth_term_l2504_250472

def a (n : ℕ) : ℕ := 3 * n - 5

theorem coefficient_equals_nth_term :
  let coefficient : ℕ := (Nat.choose 5 4) + (Nat.choose 6 4) + (Nat.choose 7 4)
  coefficient = a 20 := by sorry

end NUMINAMATH_CALUDE_coefficient_equals_nth_term_l2504_250472


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2504_250403

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}

theorem union_of_A_and_B : A ∪ B = Ioc (-2) 4 := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2504_250403


namespace NUMINAMATH_CALUDE_find_constant_b_l2504_250400

theorem find_constant_b (b d c : ℚ) : 
  (∀ x : ℚ, (7 * x^2 - 5 * x + 11/4) * (d * x^2 + b * x + c) = 
    21 * x^4 - 26 * x^3 + 34 * x^2 - (55/4) * x + 33/4) → 
  b = -11/7 := by
sorry

end NUMINAMATH_CALUDE_find_constant_b_l2504_250400


namespace NUMINAMATH_CALUDE_farmers_wheat_estimate_l2504_250488

/-- The farmer's wheat harvest problem -/
theorem farmers_wheat_estimate (total_harvest : ℕ) (extra_bushels : ℕ) 
  (h1 : total_harvest = 48781)
  (h2 : extra_bushels = 684) :
  total_harvest - extra_bushels = 48097 := by
  sorry

end NUMINAMATH_CALUDE_farmers_wheat_estimate_l2504_250488


namespace NUMINAMATH_CALUDE_min_value_floor_sum_l2504_250494

theorem min_value_floor_sum (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∃ (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0),
    ⌊(x + y + z) / w⌋ + ⌊(y + z + w) / x⌋ + ⌊(z + w + x) / y⌋ + ⌊(w + x + y) / z⌋ = 9 ∧
    ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 →
      ⌊(a + b + c) / d⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + d + a) / b⌋ + ⌊(d + a + b) / c⌋ ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_floor_sum_l2504_250494


namespace NUMINAMATH_CALUDE_range_of_a_l2504_250484

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a > 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ∈ Set.Iic (-2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2504_250484


namespace NUMINAMATH_CALUDE_route_time_proof_l2504_250461

/-- Proves that the time to run a 5-mile route one way is 1 hour, given the round trip average speed and return speed. -/
theorem route_time_proof (route_length : ℝ) (avg_speed : ℝ) (return_speed : ℝ) 
  (h1 : route_length = 5)
  (h2 : avg_speed = 8)
  (h3 : return_speed = 20) :
  let t := (2 * route_length / avg_speed - route_length / return_speed)
  t = 1 := by sorry

end NUMINAMATH_CALUDE_route_time_proof_l2504_250461


namespace NUMINAMATH_CALUDE_square_of_difference_l2504_250456

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l2504_250456


namespace NUMINAMATH_CALUDE_fraction_addition_l2504_250402

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2504_250402


namespace NUMINAMATH_CALUDE_unique_modular_residue_l2504_250446

theorem unique_modular_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ -150 ≡ n [ZMOD 17] := by sorry

end NUMINAMATH_CALUDE_unique_modular_residue_l2504_250446


namespace NUMINAMATH_CALUDE_star_seven_three_l2504_250450

-- Define the ⋆ operation
def star (a b : ℤ) : ℤ := 4*a + 3*b - 2*a*b

-- Theorem statement
theorem star_seven_three : star 7 3 = -5 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_three_l2504_250450


namespace NUMINAMATH_CALUDE_expression_evaluation_l2504_250415

theorem expression_evaluation (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  let y := 1 / x + z
  (x - 1 / x) * (y + 1 / y) = ((x^2 - 1) * (1 + 2*x*z + x^2*z^2 + x^2)) / (x^2 * (1 + x*z)) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2504_250415


namespace NUMINAMATH_CALUDE_min_handshakes_in_gathering_l2504_250492

/-- Represents a gathering of people and their handshakes -/
structure Gathering where
  people : Nat
  min_handshakes_per_person : Nat
  total_handshakes : Nat

/-- The minimum number of handshakes in a gathering of 30 people 
    where each person shakes hands with at least 3 others -/
theorem min_handshakes_in_gathering (g : Gathering) 
  (h1 : g.people = 30)
  (h2 : g.min_handshakes_per_person ≥ 3) :
  g.total_handshakes ≥ 45 ∧ 
  ∃ (arrangement : Gathering), 
    arrangement.people = 30 ∧ 
    arrangement.min_handshakes_per_person = 3 ∧ 
    arrangement.total_handshakes = 45 := by
  sorry

end NUMINAMATH_CALUDE_min_handshakes_in_gathering_l2504_250492


namespace NUMINAMATH_CALUDE_one_neither_prime_nor_composite_l2504_250462

theorem one_neither_prime_nor_composite : 
  ¬(Nat.Prime 1) ∧ ¬(∃ a b : Nat, a > 1 ∧ b > 1 ∧ a * b = 1) := by
  sorry

end NUMINAMATH_CALUDE_one_neither_prime_nor_composite_l2504_250462


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2504_250465

theorem simplify_square_roots : Real.sqrt 49 - Real.sqrt 144 + Real.sqrt 9 = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2504_250465


namespace NUMINAMATH_CALUDE_chess_go_problem_l2504_250423

/-- Represents the prices and quantities of Chinese chess and Go sets -/
structure ChessGoSets where
  chess_price : ℝ
  go_price : ℝ
  total_sets : ℕ
  max_cost : ℝ

/-- Defines the conditions given in the problem -/
def problem_conditions (s : ChessGoSets) : Prop :=
  2 * s.chess_price + 3 * s.go_price = 140 ∧
  4 * s.chess_price + s.go_price = 130 ∧
  s.total_sets = 80 ∧
  s.max_cost = 2250

/-- Theorem stating the solution to the problem -/
theorem chess_go_problem (s : ChessGoSets) 
  (h : problem_conditions s) : 
  s.chess_price = 25 ∧ 
  s.go_price = 30 ∧ 
  (∀ m : ℕ, m * s.go_price + (s.total_sets - m) * s.chess_price ≤ s.max_cost → m ≤ 50) ∧
  (∀ a : ℝ, a > 0 → 
    (a < 10 → 0.9 * a * s.go_price < 0.7 * a * s.go_price + 60) ∧
    (a = 10 → 0.9 * a * s.go_price = 0.7 * a * s.go_price + 60) ∧
    (a > 10 → 0.9 * a * s.go_price > 0.7 * a * s.go_price + 60)) :=
by
  sorry

end NUMINAMATH_CALUDE_chess_go_problem_l2504_250423


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2504_250454

/-- For a parabola y = ax², if a point P(x₀, 2) on the parabola is at a distance of 3 
    from the focus, then the distance from P to the y-axis is 2√2. -/
theorem parabola_point_distance (a : ℝ) (x₀ : ℝ) :
  (2 = a * x₀^2) →                          -- P is on the parabola
  ((x₀ - 0)^2 + (2 - 1/(4*a))^2 = 3^2) →    -- Distance from P to focus is 3
  |x₀| = 2 * Real.sqrt 2 :=                 -- Distance from P to y-axis is 2√2
by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2504_250454


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l2504_250453

-- Equation 1
theorem solve_equation_one : 
  {x : ℝ | x^2 - 9 = 0} = {3, -3} := by sorry

-- Equation 2
theorem solve_equation_two :
  {x : ℝ | (x + 1)^3 = -8/27} = {-5/3} := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l2504_250453


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2504_250490

/-- Aaron's current age -/
def aaron_age : ℕ := sorry

/-- Beth's current age -/
def beth_age : ℕ := sorry

/-- The number of years until their age ratio is 3:2 -/
def years_until_ratio : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem age_ratio_problem :
  (aaron_age - 4 = 2 * (beth_age - 4)) ∧
  (aaron_age - 6 = 3 * (beth_age - 6)) →
  years_until_ratio = 24 ∧
  (aaron_age + years_until_ratio) * 2 = 3 * (beth_age + years_until_ratio) :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2504_250490


namespace NUMINAMATH_CALUDE_real_y_condition_l2504_250478

theorem real_y_condition (x : ℝ) :
  (∃ y : ℝ, 4 * y^2 - 2 * x * y + 2 * x + 9 = 0) ↔ (x ≤ -3 ∨ x ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_real_y_condition_l2504_250478


namespace NUMINAMATH_CALUDE_number_calculation_l2504_250421

theorem number_calculation (x : ℝ) (h : 0.3 * x = 108.0) : x = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2504_250421


namespace NUMINAMATH_CALUDE_correct_sums_l2504_250435

theorem correct_sums (total : ℕ) (wrong_ratio : ℕ) (h1 : total = 54) (h2 : wrong_ratio = 2) :
  ∃ (correct : ℕ), correct * (1 + wrong_ratio) = total ∧ correct = 18 :=
by sorry

end NUMINAMATH_CALUDE_correct_sums_l2504_250435


namespace NUMINAMATH_CALUDE_real_estate_pricing_l2504_250404

theorem real_estate_pricing (retail_price : ℝ) (retail_price_pos : retail_price > 0) :
  let z_price := retail_price * (1 - 0.3)
  let x_price := z_price * (1 - 0.15)
  let y_price := ((z_price + x_price) / 2) * (1 - 0.4)
  y_price / x_price = 0.653 := by sorry

end NUMINAMATH_CALUDE_real_estate_pricing_l2504_250404


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l2504_250412

theorem quadratic_roots_sum_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    2 * x₁^2 + k * x₁ - 2 * k + 1 = 0 ∧
    2 * x₂^2 + k * x₂ - 2 * k + 1 = 0 ∧
    x₁^2 + x₂^2 = 29/4) → 
  k = 3 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l2504_250412


namespace NUMINAMATH_CALUDE_simplify_expression_l2504_250473

theorem simplify_expression : 0.2 * 0.4 + 0.6 * 0.8 = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2504_250473


namespace NUMINAMATH_CALUDE_geometric_series_sum_specific_geometric_series_l2504_250411

theorem geometric_series_sum : ∀ (a r : ℝ), 
  a ≠ 0 → 
  |r| < 1 → 
  (∑' n, a * r^n) = a / (1 - r) :=
sorry

theorem specific_geometric_series : 
  (∑' n, (1/4) * (1/2)^n) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_specific_geometric_series_l2504_250411


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2504_250455

theorem complex_equation_solution :
  ∃ (z : ℂ), 2 - (3 + Complex.I) * z = 1 - (3 - Complex.I) * z ∧ z = Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2504_250455


namespace NUMINAMATH_CALUDE_y_intercept_two_distance_from_origin_one_l2504_250477

-- Define the general equation of line l
def line_equation (a : ℝ) (x y : ℝ) : Prop :=
  x + (a + 1) * y + 2 - a = 0

-- Theorem 1: y-intercept is 2
theorem y_intercept_two :
  ∃ a : ℝ, (∀ x y : ℝ, line_equation a x y ↔ x - 3 * y + 6 = 0) ∧
  (∃ y : ℝ, line_equation a 0 y ∧ y = 2) :=
sorry

-- Theorem 2: distance from origin is 1
theorem distance_from_origin_one :
  ∃ a : ℝ, (∀ x y : ℝ, line_equation a x y ↔ 3 * x + 4 * y + 5 = 0) ∧
  (|2 - a| / Real.sqrt (1 + (a + 1)^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_y_intercept_two_distance_from_origin_one_l2504_250477


namespace NUMINAMATH_CALUDE_alcohol_fraction_in_mixture_l2504_250469

theorem alcohol_fraction_in_mixture (water_volume : ℚ) (alcohol_water_ratio : ℚ) :
  water_volume = 4/5 →
  alcohol_water_ratio = 3/4 →
  (1 - water_volume) = 3/5 :=
by
  sorry

end NUMINAMATH_CALUDE_alcohol_fraction_in_mixture_l2504_250469


namespace NUMINAMATH_CALUDE_inequality_holds_iff_k_in_range_l2504_250476

theorem inequality_holds_iff_k_in_range :
  ∀ k : ℝ, (∀ x : ℝ, k * x^2 + k * x - 3/4 < 0) ↔ k ∈ Set.Ioc (-3) 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_k_in_range_l2504_250476


namespace NUMINAMATH_CALUDE_gas_price_increase_l2504_250486

theorem gas_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.1 * (1 - 27.27272727272727 / 100) = 1 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_gas_price_increase_l2504_250486


namespace NUMINAMATH_CALUDE_oplus_example_l2504_250436

/-- Definition of the ⊕ operation -/
def oplus (a b c : ℝ) (k : ℤ) : ℝ := b^2 - k * (a^2 * c)

/-- Theorem stating that ⊕(2, 5, 3, 3) = -11 -/
theorem oplus_example : oplus 2 5 3 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_oplus_example_l2504_250436


namespace NUMINAMATH_CALUDE_sixteen_power_divided_by_eight_l2504_250470

theorem sixteen_power_divided_by_eight (n : ℕ) : 
  n = 16^2023 → (n / 8 : ℕ) = 2^8089 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_power_divided_by_eight_l2504_250470


namespace NUMINAMATH_CALUDE_average_monthly_balance_l2504_250407

def monthly_balances : List ℝ := [100, 200, 250, 50, 300, 300]
def num_months : ℕ := 6

theorem average_monthly_balance :
  (monthly_balances.sum / num_months) = 200 := by sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l2504_250407


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l2504_250440

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 14

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 15

/-- The number of books read -/
def books_read : ℕ := 11

/-- The number of movies watched -/
def movies_watched : ℕ := 40

theorem crazy_silly_school_series :
  (num_books = num_movies + 1) ∧
  (num_books = 15) ∧
  (books_read = 11) ∧
  (movies_watched = 40) →
  num_movies = 14 := by
sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l2504_250440


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l2504_250480

theorem largest_power_of_two_dividing_difference (n : ℕ) : 
  n = 18^5 - 14^5 → ∃ k : ℕ, 2^k = 64 ∧ 2^k ∣ n ∧ ∀ m : ℕ, 2^m ∣ n → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l2504_250480


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2504_250414

theorem complex_product_theorem (z₁ z₂ : ℂ) :
  z₁ = 4 + I → z₂ = 1 - 2*I → z₁ * z₂ = 6 - 7*I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2504_250414


namespace NUMINAMATH_CALUDE_parabola_passes_through_point_l2504_250420

/-- The parabola y = (1/2)x^2 - 2 passes through the point (2, 0) -/
theorem parabola_passes_through_point :
  let f : ℝ → ℝ := fun x ↦ (1/2) * x^2 - 2
  f 2 = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_passes_through_point_l2504_250420


namespace NUMINAMATH_CALUDE_lemons_for_combined_beverages_l2504_250448

/-- The number of lemons needed for a given amount of lemonade and limeade -/
def lemons_needed (lemonade_gallons : ℚ) (limeade_gallons : ℚ) : ℚ :=
  let lemons_per_gallon_lemonade : ℚ := 36 / 48
  let lemons_per_gallon_limeade : ℚ := 2 * lemons_per_gallon_lemonade
  lemonade_gallons * lemons_per_gallon_lemonade + limeade_gallons * lemons_per_gallon_limeade

/-- Theorem stating the number of lemons needed for 18 gallons of combined lemonade and limeade -/
theorem lemons_for_combined_beverages :
  lemons_needed 9 9 = 81/4 := by
  sorry

#eval lemons_needed 9 9

end NUMINAMATH_CALUDE_lemons_for_combined_beverages_l2504_250448


namespace NUMINAMATH_CALUDE_canoe_production_sum_l2504_250410

theorem canoe_production_sum : 
  let a : ℕ := 5  -- first term
  let r : ℕ := 3  -- common ratio
  let n : ℕ := 8  -- number of terms
  a * (r^n - 1) / (r - 1) = 16400 :=
by sorry

end NUMINAMATH_CALUDE_canoe_production_sum_l2504_250410


namespace NUMINAMATH_CALUDE_trig_identity_1_trig_identity_2_l2504_250401

-- Problem 1
theorem trig_identity_1 (θ : ℝ) : (Real.sin θ - Real.cos θ) / (Real.tan θ - 1) = Real.cos θ := by
  sorry

-- Problem 2
theorem trig_identity_2 (α : ℝ) : Real.sin α ^ 4 - Real.cos α ^ 4 = 2 * Real.sin α ^ 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_1_trig_identity_2_l2504_250401


namespace NUMINAMATH_CALUDE_distance_walked_l2504_250417

/-- Proves that the distance walked is 18 miles given specific conditions on speed changes and time. -/
theorem distance_walked (speed : ℝ) (time : ℝ) : 
  speed > 0 → 
  time > 0 → 
  (speed + 1) * (3 * time / 4) = speed * time → 
  (speed - 1) * (time + 3) = speed * time → 
  speed * time = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_walked_l2504_250417


namespace NUMINAMATH_CALUDE_combined_painting_time_l2504_250442

/-- Given Shawn's and Karen's individual painting rates, calculate their combined time to paint one house -/
theorem combined_painting_time (shawn_rate karen_rate : ℝ) (h1 : shawn_rate = 1 / 18) (h2 : karen_rate = 1 / 12) :
  1 / (shawn_rate + karen_rate) = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_combined_painting_time_l2504_250442


namespace NUMINAMATH_CALUDE_racket_sales_total_l2504_250430

/-- The total amount earned from selling rackets given the average price per pair and the number of pairs sold -/
theorem racket_sales_total (avg_price : ℝ) (num_pairs : ℕ) : 
  avg_price = 9.8 → num_pairs = 55 → avg_price * (num_pairs : ℝ) = 539 := by
  sorry

end NUMINAMATH_CALUDE_racket_sales_total_l2504_250430


namespace NUMINAMATH_CALUDE_derivative_exp_sin_l2504_250433

theorem derivative_exp_sin (x : ℝ) : 
  deriv (fun x => Real.exp x * Real.sin x) x = Real.exp x * (Real.sin x + Real.cos x) := by
sorry

end NUMINAMATH_CALUDE_derivative_exp_sin_l2504_250433


namespace NUMINAMATH_CALUDE_initial_to_doubled_ratio_l2504_250475

theorem initial_to_doubled_ratio (x : ℝ) : 3 * (2 * x + 5) = 105 → x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_to_doubled_ratio_l2504_250475


namespace NUMINAMATH_CALUDE_remainder_17_63_mod_7_l2504_250471

theorem remainder_17_63_mod_7 :
  ∃ k : ℤ, 17^63 = 7 * k + 6 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_17_63_mod_7_l2504_250471


namespace NUMINAMATH_CALUDE_function_equality_implies_m_value_l2504_250409

theorem function_equality_implies_m_value :
  ∀ (m : ℚ),
  let f : ℚ → ℚ := λ x => x^2 - 3*x + m
  let g : ℚ → ℚ := λ x => x^2 - 3*x + 5*m
  3 * f 5 = 2 * g 5 →
  m = 10/7 :=
by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_value_l2504_250409


namespace NUMINAMATH_CALUDE_final_state_digits_l2504_250418

/-- Represents the state of the board as three integers -/
structure BoardState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Performs one iteration of pairwise sum replacement -/
def iterate (state : BoardState) : BoardState :=
  { a := (state.a + state.b) % 10,
    b := (state.a + state.c) % 10,
    c := (state.b + state.c) % 10 }

/-- Performs n iterations of pairwise sum replacement -/
def iterateN (n : ℕ) (state : BoardState) : BoardState :=
  match n with
  | 0 => state
  | n + 1 => iterate (iterateN n state)

/-- The main theorem to be proved -/
theorem final_state_digits (initialState : BoardState) :
  initialState.a = 1 ∧ initialState.b = 2 ∧ initialState.c = 4 →
  let finalState := iterateN 60 initialState
  (finalState.a = 6 ∧ finalState.b = 7 ∧ finalState.c = 9) ∨
  (finalState.a = 6 ∧ finalState.b = 9 ∧ finalState.c = 7) ∨
  (finalState.a = 7 ∧ finalState.b = 6 ∧ finalState.c = 9) ∨
  (finalState.a = 7 ∧ finalState.b = 9 ∧ finalState.c = 6) ∨
  (finalState.a = 9 ∧ finalState.b = 6 ∧ finalState.c = 7) ∨
  (finalState.a = 9 ∧ finalState.b = 7 ∧ finalState.c = 6) :=
by sorry

end NUMINAMATH_CALUDE_final_state_digits_l2504_250418


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2504_250406

theorem proposition_equivalence (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2504_250406


namespace NUMINAMATH_CALUDE_equality_implies_product_equality_l2504_250464

theorem equality_implies_product_equality (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end NUMINAMATH_CALUDE_equality_implies_product_equality_l2504_250464


namespace NUMINAMATH_CALUDE_special_numbers_property_l2504_250422

/-- Given a natural number, return the sum of its digits -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The list of 13 numbers that satisfy the conditions -/
def specialNumbers : List ℕ := [6, 15, 24, 33, 42, 51, 60, 105, 114, 123, 132, 141, 150]

theorem special_numbers_property :
  (∃ (nums : List ℕ),
    nums.length = 13 ∧
    nums.sum = 996 ∧
    nums.Nodup ∧
    ∀ (x y : ℕ), x ∈ nums → y ∈ nums → digitSum x = digitSum y) := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_property_l2504_250422


namespace NUMINAMATH_CALUDE_percentage_change_condition_l2504_250451

theorem percentage_change_condition (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hqlt : q < 100) (hM : M > 0) : 
  (M * (1 + p / 100) * (1 - q / 100) > M) ↔ (p > 100 * q / (100 - q)) := by
  sorry

end NUMINAMATH_CALUDE_percentage_change_condition_l2504_250451


namespace NUMINAMATH_CALUDE_sector_area_given_arc_length_l2504_250445

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4 cm,
    the area of the sector enclosed by this central angle is 4 cm². -/
theorem sector_area_given_arc_length (r : ℝ) : r * 2 = 4 → r^2 = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_given_arc_length_l2504_250445


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2504_250443

/-- A quadratic function g(x) = px^2 + qx + r -/
def g (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

/-- Theorem: If g(-2) = 0, g(3) = 0, and g(1) = 5, then q = 5/6 -/
theorem quadratic_coefficient (p q r : ℝ) :
  g p q r (-2) = 0 → g p q r 3 = 0 → g p q r 1 = 5 → q = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2504_250443


namespace NUMINAMATH_CALUDE_max_log_sum_max_log_sum_attained_l2504_250468

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 3 * y = 6) :
  (Real.log x / Real.log (3/2) + Real.log y / Real.log (3/2)) ≤ 1 :=
by sorry

theorem max_log_sum_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 3 * y = 6) :
  (Real.log x / Real.log (3/2) + Real.log y / Real.log (3/2)) = 1 ↔ x = 3/2 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_log_sum_max_log_sum_attained_l2504_250468


namespace NUMINAMATH_CALUDE_sunflower_height_difference_l2504_250452

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Represents a height in feet and inches -/
structure Height :=
  (feet : ℕ)
  (inches : ℕ)

/-- Converts a Height to total inches -/
def height_to_inches (h : Height) : ℕ := feet_to_inches h.feet + h.inches

theorem sunflower_height_difference :
  let sister_height : Height := ⟨4, 3⟩
  let sunflower_height : ℕ := feet_to_inches 6
  sunflower_height - height_to_inches sister_height = 21 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_height_difference_l2504_250452


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_257_l2504_250479

theorem modular_inverse_of_3_mod_257 : ∃ x : ℕ, x < 257 ∧ (3 * x) % 257 = 1 :=
  by
    use 86
    sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_257_l2504_250479


namespace NUMINAMATH_CALUDE_product_and_divisibility_l2504_250437

theorem product_and_divisibility (n : ℕ) : 
  n = 3 → 
  ((n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720) ∧ 
  ¬(720 % 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_and_divisibility_l2504_250437


namespace NUMINAMATH_CALUDE_trapezoid_area_sum_l2504_250408

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

/-- Calculates the area of a trapezoid using Heron's formula -/
def area (t : Trapezoid) : ℝ := sorry

/-- Checks if a number is a square-free positive integer -/
def isSquareFree (n : ℕ) : Prop := sorry

/-- Theorem: The sum of areas of all possible trapezoids with sides 4, 6, 8, and 10
    can be expressed as r₁√n₁ + r₂√n₂ + r₃, where r₁, r₂, r₃ are rational,
    n₁, n₂ are distinct square-free positive integers, and r₁ + r₂ + r₃ + n₁ + n₂ = 80 -/
theorem trapezoid_area_sum :
  ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
    let t₁ : Trapezoid := ⟨4, 6, 8, 10⟩
    let t₂ : Trapezoid := ⟨6, 10, 4, 8⟩
    isSquareFree n₁ ∧ isSquareFree n₂ ∧ n₁ ≠ n₂ ∧
    area t₁ + area t₂ = r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃ ∧
    r₁ + r₂ + r₃ + n₁ + n₂ = 80 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_sum_l2504_250408


namespace NUMINAMATH_CALUDE_no_self_composite_plus_1987_function_l2504_250474

theorem no_self_composite_plus_1987_function :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end NUMINAMATH_CALUDE_no_self_composite_plus_1987_function_l2504_250474


namespace NUMINAMATH_CALUDE_bonus_distribution_l2504_250466

theorem bonus_distribution (total_bonus : ℕ) (difference : ℕ) (junior_share : ℕ) : 
  total_bonus = 5000 →
  difference = 1200 →
  junior_share + (junior_share + difference) = total_bonus →
  junior_share = 1900 := by
sorry

end NUMINAMATH_CALUDE_bonus_distribution_l2504_250466


namespace NUMINAMATH_CALUDE_tom_bonus_percentage_l2504_250428

/-- Represents the game scoring system and Tom's performance -/
structure GameScore where
  points_per_enemy : ℕ
  bonus_threshold : ℕ
  enemies_killed : ℕ
  total_score : ℕ

/-- Calculates the percentage of the bonus in Tom's score -/
def bonus_percentage (game : GameScore) : ℚ :=
  let score_without_bonus := game.points_per_enemy * game.enemies_killed
  let bonus := game.total_score - score_without_bonus
  (bonus : ℚ) / (score_without_bonus : ℚ) * 100

/-- Theorem stating that Tom's bonus percentage is 50% -/
theorem tom_bonus_percentage :
  let game := GameScore.mk 10 100 150 2250
  bonus_percentage game = 50 := by sorry

end NUMINAMATH_CALUDE_tom_bonus_percentage_l2504_250428


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2504_250482

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (6 * x₁ = 150 / x₁) ∧ (6 * x₂ = 150 / x₂) ∧ (x₁ + x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2504_250482


namespace NUMINAMATH_CALUDE_james_tylenol_intake_l2504_250429

/-- Calculates the total milligrams of Tylenol taken per day given the dosage and frequency. -/
def tylenol_per_day (tablets_per_dose : ℕ) (mg_per_tablet : ℕ) (hours_between_doses : ℕ) (hours_per_day : ℕ) : ℕ :=
  let mg_per_dose := tablets_per_dose * mg_per_tablet
  let doses_per_day := hours_per_day / hours_between_doses
  mg_per_dose * doses_per_day

/-- Proves that James takes 3000 mg of Tylenol per day given the specified conditions. -/
theorem james_tylenol_intake : tylenol_per_day 2 375 6 24 = 3000 := by
  sorry


end NUMINAMATH_CALUDE_james_tylenol_intake_l2504_250429


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l2504_250463

theorem integer_root_of_cubic (b c : ℚ) :
  (∃ x : ℝ, x^3 + b*x + c = 0 ∧ x = 2 - Real.sqrt 5) →
  (∃ r : ℤ, r^3 + b*r + c = 0) →
  (∃ r : ℤ, r^3 + b*r + c = 0 ∧ r = -4) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l2504_250463


namespace NUMINAMATH_CALUDE_not_sufficient_for_congruence_l2504_250489

/-- Two triangles are congruent -/
def triangles_congruent (A B C D E F : Point) : Prop := sorry

/-- The measure of an angle -/
def angle_measure (A B C : Point) : ℝ := sorry

/-- The length of a line segment -/
def segment_length (A B : Point) : ℝ := sorry

/-- Theorem: Given ∠A = ∠F, ∠B = ∠E, and AC = DE, it's not sufficient to determine 
    the congruence of triangles ABC and DEF -/
theorem not_sufficient_for_congruence 
  (A B C D E F : Point) 
  (h1 : angle_measure A B C = angle_measure F E D)
  (h2 : angle_measure B A C = angle_measure E F D)
  (h3 : segment_length A C = segment_length D E) :
  ¬ (triangles_congruent A B C D E F) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_for_congruence_l2504_250489


namespace NUMINAMATH_CALUDE_jinas_mascots_l2504_250457

/-- The number of mascots Jina has -/
def total_mascots (initial_teddies : ℕ) (bunny_multiplier : ℕ) (koalas : ℕ) (additional_teddies_per_bunny : ℕ) : ℕ :=
  let bunnies := initial_teddies * bunny_multiplier
  let additional_teddies := bunnies * additional_teddies_per_bunny
  initial_teddies + bunnies + koalas + additional_teddies

/-- Theorem stating the total number of mascots Jina has -/
theorem jinas_mascots :
  total_mascots 5 3 1 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_jinas_mascots_l2504_250457


namespace NUMINAMATH_CALUDE_boxes_with_neither_crayons_nor_markers_l2504_250438

theorem boxes_with_neither_crayons_nor_markers 
  (total_boxes : ℕ) 
  (boxes_with_crayons : ℕ) 
  (boxes_with_markers : ℕ) 
  (boxes_with_both : ℕ) 
  (h1 : total_boxes = 15) 
  (h2 : boxes_with_crayons = 9) 
  (h3 : boxes_with_markers = 6) 
  (h4 : boxes_with_both = 4) : 
  total_boxes - (boxes_with_crayons + boxes_with_markers - boxes_with_both) = 4 := by
  sorry

#check boxes_with_neither_crayons_nor_markers

end NUMINAMATH_CALUDE_boxes_with_neither_crayons_nor_markers_l2504_250438


namespace NUMINAMATH_CALUDE_class_size_is_24_l2504_250434

-- Define the number of candidates
def num_candidates : Nat := 4

-- Define the number of absent students
def absent_students : Nat := 5

-- Define the function to calculate votes needed to win
def votes_to_win (x : Nat) : Nat :=
  if x % 2 = 0 then x / 2 + 1 else (x + 1) / 2

-- Define the function to calculate votes received by each candidate
def votes_received (x : Nat) (missed_by : Nat) : Nat :=
  votes_to_win x - missed_by

-- Define the theorem
theorem class_size_is_24 :
  ∃ (x : Nat),
    -- x is the number of students who voted
    x + absent_students = 24 ∧
    -- Sum of votes received by all candidates equals x
    votes_received x 3 + votes_received x 9 + votes_received x 5 + votes_received x 4 = x :=
by sorry

end NUMINAMATH_CALUDE_class_size_is_24_l2504_250434


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l2504_250405

theorem cos_seven_pi_sixths : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l2504_250405


namespace NUMINAMATH_CALUDE_factor_x6_minus_64_l2504_250444

theorem factor_x6_minus_64 (x : ℝ) : 
  x^6 - 64 = (x - 2) * (x + 2) * (x^2 + 2*x + 4) * (x^2 - 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_x6_minus_64_l2504_250444


namespace NUMINAMATH_CALUDE_complex_equality_implies_b_value_l2504_250424

theorem complex_equality_implies_b_value (b : ℝ) : 
  let z : ℂ := (1 + b * I) / (2 + I)
  z.re = z.im → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_b_value_l2504_250424


namespace NUMINAMATH_CALUDE_triangle_angle_C_l2504_250416

theorem triangle_angle_C (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  3 * Real.sin A + 4 * Real.cos B = 6 →
  4 * Real.sin B + 3 * Real.cos A = 1 →
  C = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l2504_250416


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisibility_l2504_250431

theorem product_of_three_consecutive_integers_divisibility :
  ∀ n : ℕ, n > 0 →
  ∃ k : ℕ, (n - 1) * n * (n + 1) = 6 * k ∧
  ∀ m : ℕ, m > 6 → ∃ n : ℕ, n > 0 ∧ ¬(∃ k : ℕ, (n - 1) * n * (n + 1) = m * k) :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisibility_l2504_250431


namespace NUMINAMATH_CALUDE_vegetables_problem_l2504_250425

theorem vegetables_problem (potatoes carrots onions green_beans : ℕ) :
  carrots = 6 * potatoes →
  onions = 2 * carrots →
  green_beans = onions / 3 →
  green_beans = 8 →
  potatoes = 2 := by
  sorry

end NUMINAMATH_CALUDE_vegetables_problem_l2504_250425


namespace NUMINAMATH_CALUDE_stating_tournament_orderings_l2504_250498

/-- Represents the number of players in the tournament -/
def num_players : Nat := 6

/-- Represents the number of possible outcomes for each game -/
def outcomes_per_game : Nat := 2

/-- Calculates the number of possible orderings in the tournament -/
def num_orderings : Nat := outcomes_per_game ^ (num_players - 1)

/-- 
Theorem stating that the number of possible orderings in the tournament is 32
given the specified number of players and outcomes per game.
-/
theorem tournament_orderings :
  num_orderings = 32 :=
by sorry

end NUMINAMATH_CALUDE_stating_tournament_orderings_l2504_250498


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_four_l2504_250459

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of -4 is 4 -/
theorem opposite_of_neg_four : opposite (-4 : ℝ) = 4 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_four_l2504_250459


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2504_250432

theorem boys_to_girls_ratio :
  ∀ (boys girls : ℕ),
    boys = 80 →
    girls = boys + 128 →
    (boys : ℚ) / girls = 5 / 13 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2504_250432


namespace NUMINAMATH_CALUDE_number_in_set_l2504_250495

theorem number_in_set (initial_avg : ℝ) (wrong_num : ℝ) (correct_num : ℝ) (correct_avg : ℝ) :
  initial_avg = 23 →
  wrong_num = 26 →
  correct_num = 36 →
  correct_avg = 24 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * initial_avg - wrong_num = (n : ℝ) * correct_avg - correct_num ∧
    n = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_in_set_l2504_250495


namespace NUMINAMATH_CALUDE_value_of_expression_l2504_250441

theorem value_of_expression (x : ℝ) (h : 7 * x + 6 = 3 * x - 18) : 
  3 * (2 * x + 4) = -24 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l2504_250441


namespace NUMINAMATH_CALUDE_principal_calculation_l2504_250427

/-- Proves that given specific conditions, the principal amount is 1200 --/
theorem principal_calculation (rate : ℝ) (time : ℝ) (amount : ℝ) :
  rate = 0.05 →
  time = 2 + 2 / 5 →
  amount = 1344 →
  amount = (1200 : ℝ) * (1 + rate * time) :=
by sorry

end NUMINAMATH_CALUDE_principal_calculation_l2504_250427


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l2504_250496

theorem lcm_factor_proof (A B X : ℕ) : 
  A > 0 → B > 0 →
  Nat.gcd A B = 23 →
  A = 414 →
  ∃ (Y : ℕ), Nat.lcm A B = 23 * 13 * X ∧ Nat.lcm A B = 23 * 13 * Y →
  X = 18 := by sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l2504_250496


namespace NUMINAMATH_CALUDE_shepherd_problem_l2504_250419

def checkpoint (n : ℕ) : ℕ := n / 2 + 1

def process (initial : ℕ) (checkpoints : ℕ) : ℕ :=
  match checkpoints with
  | 0 => initial
  | n + 1 => checkpoint (process initial n)

theorem shepherd_problem (initial : ℕ) (checkpoints : ℕ) :
  initial = 254 ∧ checkpoints = 6 → process initial checkpoints = 2 := by
  sorry

end NUMINAMATH_CALUDE_shepherd_problem_l2504_250419


namespace NUMINAMATH_CALUDE_min_rain_day4_overflow_l2504_250447

/-- Represents the rainstorm scenario -/
structure RainstormScenario where
  capacity : ℝ  -- capacity in feet
  drain_rate : ℝ  -- drain rate in inches per day
  day1_rain : ℝ  -- rain on day 1 in inches
  days : ℕ  -- number of days
  overflow_day : ℕ  -- day when overflow occurs

/-- Calculates the minimum amount of rain on the last day to cause overflow -/
def min_rain_to_overflow (scenario : RainstormScenario) : ℝ :=
  sorry

/-- Theorem stating the minimum amount of rain on day 4 to cause overflow -/
theorem min_rain_day4_overflow (scenario : RainstormScenario) 
  (h1 : scenario.capacity = 6)
  (h2 : scenario.drain_rate = 3)
  (h3 : scenario.day1_rain = 10)
  (h4 : scenario.days = 4)
  (h5 : scenario.overflow_day = 4) :
  min_rain_to_overflow scenario = 4 :=
  sorry

end NUMINAMATH_CALUDE_min_rain_day4_overflow_l2504_250447


namespace NUMINAMATH_CALUDE_max_min_difference_circle_l2504_250426

theorem max_min_difference_circle (x y : ℝ) (h : x^2 - 4*x + y^2 + 3 = 0) :
  let f := fun (x y : ℝ) => x^2 + y^2
  ∃ (M m : ℝ), (∀ (a b : ℝ), a^2 - 4*a + b^2 + 3 = 0 → f a b ≤ M) ∧
               (∀ (a b : ℝ), a^2 - 4*a + b^2 + 3 = 0 → m ≤ f a b) ∧
               M - m = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_circle_l2504_250426


namespace NUMINAMATH_CALUDE_chord_length_through_focus_l2504_250483

/-- Given a parabola y^2 = 8x, prove that a chord AB passing through the focus
    with endpoints A(x₁, y₁) and B(x₂, y₂) on the parabola, where x₁ + x₂ = 10,
    has length |AB| = 14. -/
theorem chord_length_through_focus (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 8*x₁ →  -- A is on the parabola
  y₂^2 = 8*x₂ →  -- B is on the parabola
  x₁ + x₂ = 10 → -- Given condition
  -- AB passes through the focus (2, 0)
  (y₂ - y₁) * 2 = (x₂ - x₁) * (y₂ + y₁) →
  -- The length of AB is 14
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 14^2 := by
sorry

end NUMINAMATH_CALUDE_chord_length_through_focus_l2504_250483


namespace NUMINAMATH_CALUDE_largest_six_digit_number_l2504_250487

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n % 10) * digit_product (n / 10)

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem largest_six_digit_number : ∀ n : ℕ, 
  100000 ≤ n ∧ n ≤ 999999 ∧ digit_product n = factorial 8 → n ≤ 987744 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_six_digit_number_l2504_250487


namespace NUMINAMATH_CALUDE_beetles_eaten_in_forest_l2504_250485

/-- The number of beetles eaten in a forest each day -/
def beetles_eaten_per_day (jaguars : ℕ) (snakes_per_jaguar : ℕ) (birds_per_snake : ℕ) (beetles_per_bird : ℕ) : ℕ :=
  jaguars * snakes_per_jaguar * birds_per_snake * beetles_per_bird

/-- Theorem stating the number of beetles eaten in a specific forest scenario -/
theorem beetles_eaten_in_forest :
  beetles_eaten_per_day 6 5 3 12 = 1080 := by
  sorry

#eval beetles_eaten_per_day 6 5 3 12

end NUMINAMATH_CALUDE_beetles_eaten_in_forest_l2504_250485


namespace NUMINAMATH_CALUDE_equilateral_triangle_cd_product_l2504_250481

/-- Given an equilateral triangle with vertices at (0,0), (c,15), and (d,47),
    the product cd equals 1216√3/9 -/
theorem equilateral_triangle_cd_product (c d : ℝ) : 
  (∀ (z : ℂ), z ^ 3 = 1 ∧ z ≠ 1 → (c + 15 * I) * z = d + 47 * I) →
  c * d = 1216 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_cd_product_l2504_250481


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2504_250493

def vector_problem (a b : ℝ × ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  let sum := (a.1 + b.1, a.2 + b.2)
  dot_product = 10 ∧ 
  magnitude sum = 5 * Real.sqrt 2 ∧ 
  a = (2, 1) →
  magnitude b = 5

theorem vector_magnitude_proof : 
  ∀ (a b : ℝ × ℝ), vector_problem a b :=
sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2504_250493


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l2504_250460

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - Real.sqrt 3

theorem ellipse_and_line_intersection :
  -- Conditions
  (∀ x y, ellipse_C x y → (x = 0 ∧ y = 0) → False) →  -- center at origin
  (∃ c > 0, ∀ x y, ellipse_C x y → x^2 / 4 + y^2 / c^2 = 1) →  -- standard form
  (ellipse_C 1 (Real.sqrt 3 / 2)) →  -- point on ellipse
  -- Conclusions
  (∀ x y, ellipse_C x y ↔ x^2 / 4 + y^2 = 1) ∧  -- equation of C
  (∃ x₁ x₂ y₁ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (8/5)^2) :=  -- length of AB
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l2504_250460


namespace NUMINAMATH_CALUDE_equation_solution_l2504_250439

theorem equation_solution :
  ∀ x : ℝ, 2 * x^2 + 9 = (4 - x)^2 ↔ x = 4 + Real.sqrt 23 ∨ x = 4 - Real.sqrt 23 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2504_250439
