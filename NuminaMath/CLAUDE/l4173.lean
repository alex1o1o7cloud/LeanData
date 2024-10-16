import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_eight_l4173_417316

theorem smallest_n_multiple_of_eight (x y : ℤ) 
  (h1 : ∃ k : ℤ, x + 2 = 8 * k) 
  (h2 : ∃ m : ℤ, y - 2 = 8 * m) : 
  (∀ n : ℕ, n > 0 → n < 4 → ¬(∃ p : ℤ, x^2 - x*y + y^2 + n = 8 * p)) ∧ 
  (∃ q : ℤ, x^2 - x*y + y^2 + 4 = 8 * q) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_eight_l4173_417316


namespace NUMINAMATH_CALUDE_divisors_of_cube_l4173_417338

/-- 
Given a natural number n with exactly two prime divisors,
if n^2 has 81 divisors, then n^3 has either 160 or 169 divisors.
-/
theorem divisors_of_cube (n : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ ∃ α β : ℕ, n = p^α * q^β) →
  (Finset.card (Nat.divisors (n^2)) = 81) →
  (Finset.card (Nat.divisors (n^3)) = 160 ∨ Finset.card (Nat.divisors (n^3)) = 169) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_cube_l4173_417338


namespace NUMINAMATH_CALUDE_lemon_bag_mass_l4173_417353

theorem lemon_bag_mass (max_load : ℕ) (num_bags : ℕ) (remaining_capacity : ℕ) 
  (h1 : max_load = 900)
  (h2 : num_bags = 100)
  (h3 : remaining_capacity = 100) :
  (max_load - remaining_capacity) / num_bags = 8 := by
  sorry

end NUMINAMATH_CALUDE_lemon_bag_mass_l4173_417353


namespace NUMINAMATH_CALUDE_circle_area_reduction_l4173_417396

theorem circle_area_reduction (r : ℝ) (h1 : π * r^2 = 36 * π) (h2 : r > 2) : 
  π * (r - 2)^2 = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_reduction_l4173_417396


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_11_l4173_417337

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
structure ArithmeticSequence (α : Type*) [AddCommGroup α] where
  a : ℕ → α
  d : α
  h : ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_11 (a : ArithmeticSequence ℝ) 
  (h : a.a 4 + a.a 8 = 16) : 
  (Finset.range 11).sum a.a = 88 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_11_l4173_417337


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4173_417395

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  1 / x + 4 / y ≥ 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4173_417395


namespace NUMINAMATH_CALUDE_no_cards_below_threshold_l4173_417382

def jungkook_card : ℚ := 0.8
def yoongi_card : ℚ := 1/2
def yoojeong_card : ℚ := 0.9
def yuna_card : ℚ := 1/3

def threshold : ℚ := 0.3

def count_below_threshold (cards : List ℚ) : ℕ :=
  (cards.filter (· < threshold)).length

theorem no_cards_below_threshold :
  count_below_threshold [jungkook_card, yoongi_card, yoojeong_card, yuna_card] = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_cards_below_threshold_l4173_417382


namespace NUMINAMATH_CALUDE_equation_has_29_solutions_l4173_417378

/-- The number of real solutions to the equation x/50 = sin x -/
def num_solutions : ℕ := 29

/-- The equation we're considering -/
def equation (x : ℝ) : Prop := x / 50 = Real.sin x

theorem equation_has_29_solutions :
  ∃! (s : Set ℝ), (∀ x ∈ s, equation x) ∧ Finite s ∧ Nat.card s = num_solutions :=
sorry

end NUMINAMATH_CALUDE_equation_has_29_solutions_l4173_417378


namespace NUMINAMATH_CALUDE_polygon_with_108_degree_interior_angles_is_pentagon_l4173_417346

theorem polygon_with_108_degree_interior_angles_is_pentagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    interior_angle = 108 →
    (n : ℝ) * (180 - interior_angle) = 360 →
    n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_108_degree_interior_angles_is_pentagon_l4173_417346


namespace NUMINAMATH_CALUDE_min_value_problem_l4173_417358

theorem min_value_problem (i j k l m n o p : ℝ) 
  (h1 : i * j * k * l = 16) 
  (h2 : m * n * o * p = 25) : 
  (i * m)^2 + (j * n)^2 + (k * o)^2 + (l * p)^2 ≥ 160 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l4173_417358


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l4173_417390

theorem rhombus_diagonal_length (d1 : ℝ) (d2 : ℝ) (square_side : ℝ) 
  (h1 : d1 = 16)
  (h2 : square_side = 8)
  (h3 : d1 * d2 / 2 = square_side ^ 2) :
  d2 = 8 := by
sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l4173_417390


namespace NUMINAMATH_CALUDE_hannah_trip_cost_l4173_417355

/-- Calculates the cost of gas for a trip given odometer readings, fuel efficiency, and gas price -/
def trip_gas_cost (initial_reading : ℕ) (final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  ((final_reading - initial_reading : ℚ) / fuel_efficiency) * gas_price

theorem hannah_trip_cost :
  let initial_reading : ℕ := 32150
  let final_reading : ℕ := 32178
  let fuel_efficiency : ℚ := 25
  let gas_price : ℚ := 375/100
  trip_gas_cost initial_reading final_reading fuel_efficiency gas_price = 420/100 := by
  sorry

end NUMINAMATH_CALUDE_hannah_trip_cost_l4173_417355


namespace NUMINAMATH_CALUDE_max_sides_convex_polygon_is_maximum_max_sides_convex_polygon_is_convex_l4173_417371

/-- The maximum number of sides for a convex polygon with interior angles
    forming an arithmetic sequence with a common difference of 1°. -/
def max_sides_convex_polygon : ℕ := 27

/-- The common difference of the arithmetic sequence formed by the interior angles. -/
def common_difference : ℝ := 1

/-- Predicate to check if a polygon is convex based on its number of sides. -/
def is_convex (n : ℕ) : Prop :=
  let α : ℝ := (n - 2) * 180 / n - (n - 1) / 2
  α > 0 ∧ α + (n - 1) * common_difference < 180

/-- Theorem stating that max_sides_convex_polygon is the maximum number of sides
    for a convex polygon with interior angles forming an arithmetic sequence
    with a common difference of 1°. -/
theorem max_sides_convex_polygon_is_maximum :
  ∀ n : ℕ, n > max_sides_convex_polygon → ¬(is_convex n) :=
sorry

/-- Theorem stating that max_sides_convex_polygon satisfies the convexity condition. -/
theorem max_sides_convex_polygon_is_convex :
  is_convex max_sides_convex_polygon :=
sorry

end NUMINAMATH_CALUDE_max_sides_convex_polygon_is_maximum_max_sides_convex_polygon_is_convex_l4173_417371


namespace NUMINAMATH_CALUDE_missing_number_is_twelve_l4173_417367

theorem missing_number_is_twelve : ∃ x : ℕ, 1234562 - x * 3 * 2 = 1234490 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_twelve_l4173_417367


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l4173_417364

theorem largest_integer_with_remainder : ∃ n : ℕ, 
  (n < 120) ∧ 
  (n % 8 = 7) ∧ 
  (∀ m : ℕ, m < 120 ∧ m % 8 = 7 → m ≤ n) ∧
  (n = 119) := by
sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l4173_417364


namespace NUMINAMATH_CALUDE_unique_periodic_modulus_l4173_417302

/-- The binomial coefficient C(n,k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence x_n = C(2n, n) -/
def x_seq (n : ℕ) : ℕ := binomial (2 * n) n

/-- A sequence is eventually periodic modulo m if there exist positive integers N and T
    such that for all n ≥ N, x_(n+T) ≡ x_n (mod m) -/
def eventually_periodic_mod (x : ℕ → ℕ) (m : ℕ) : Prop :=
  ∃ (N T : ℕ), T > 0 ∧ ∀ n ≥ N, x (n + T) % m = x n % m

/-- The main theorem: 2 is the only positive integer h > 1 such that 
    the sequence x_n = C(2n, n) is eventually periodic modulo h -/
theorem unique_periodic_modulus :
  ∀ h : ℕ, h > 1 → (eventually_periodic_mod x_seq h ↔ h = 2) := by sorry

end NUMINAMATH_CALUDE_unique_periodic_modulus_l4173_417302


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l4173_417352

theorem triangle_angle_sum 
  (A B C : ℝ) 
  (h_acute_A : 0 < A ∧ A < π/2) 
  (h_acute_B : 0 < B ∧ B < π/2)
  (h_sin_A : Real.sin A = Real.sqrt 5 / 5)
  (h_sin_B : Real.sin B = Real.sqrt 10 / 10)
  : Real.cos (A + B) = Real.sqrt 2 / 2 ∧ C = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l4173_417352


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4173_417345

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 1 + 3 * Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4173_417345


namespace NUMINAMATH_CALUDE_inequality_proof_l4173_417315

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hab : a + b ≥ 1) 
  (hbc : b + c ≥ 1) 
  (hca : c + a ≥ 1) : 
  1 ≤ (1 - a)^2 + (1 - b)^2 + (1 - c)^2 + (2 * Real.sqrt 2 * a * b * c) / Real.sqrt (a^2 + b^2 + c^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4173_417315


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4173_417365

theorem partial_fraction_decomposition (D E F : ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -2 ∧ x ≠ 6 →
    1 / (x^3 - 3*x^2 - 4*x + 12) = D / (x - 1) + E / (x + 2) + F / (x + 2)^2) →
  D = -1/15 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4173_417365


namespace NUMINAMATH_CALUDE_anthony_free_throw_improvement_l4173_417369

theorem anthony_free_throw_improvement :
  let initial_success : ℚ := 6 / 15
  let initial_attempts : ℕ := 15
  let additional_success : ℕ := 24
  let additional_attempts : ℕ := 32
  let final_success : ℚ := (6 + additional_success) / (initial_attempts + additional_attempts)
  (final_success - initial_success) * 100 = 24 := by
sorry

end NUMINAMATH_CALUDE_anthony_free_throw_improvement_l4173_417369


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l4173_417327

theorem quadratic_expression_value (x : ℝ) : 
  let a : ℝ := 2010 * x + 2010
  let b : ℝ := 2010 * x + 2011
  let c : ℝ := 2010 * x + 2012
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l4173_417327


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l4173_417398

theorem least_three_digit_multiple_of_nine : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 9 ∣ n → n ≥ 108 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l4173_417398


namespace NUMINAMATH_CALUDE_equal_intercepts_values_l4173_417368

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := a * x + y - 2 + a = 0

-- Define the condition for equal intercepts
def equal_intercepts (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ line_equation a k 0 ∧ line_equation a 0 k

-- Theorem statement
theorem equal_intercepts_values :
  ∀ a : ℝ, equal_intercepts a ↔ (a = 2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_equal_intercepts_values_l4173_417368


namespace NUMINAMATH_CALUDE_triangle_area_l4173_417343

/-- The area of a triangle with base 2t and height 3t - 1 is t(3t - 1) -/
theorem triangle_area (t : ℝ) : 
  let base : ℝ := 2 * t
  let height : ℝ := 3 * t - 1
  (1 / 2 : ℝ) * base * height = t * (3 * t - 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4173_417343


namespace NUMINAMATH_CALUDE_extreme_value_point_l4173_417335

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Theorem stating that -2 is an extreme value point of f
theorem extreme_value_point : 
  ∃ (δ : ℝ), δ > 0 ∧ 
  (∀ x₁ ∈ Set.Ioo (-2 - δ) (-2), f' x₁ < 0) ∧
  (∀ x₂ ∈ Set.Ioo (-2) (-2 + δ), f' x₂ > 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_point_l4173_417335


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4173_417319

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 2 → x^2 > 4) ∧ (∃ x, x^2 > 4 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4173_417319


namespace NUMINAMATH_CALUDE_cricket_average_increase_l4173_417357

theorem cricket_average_increase (innings : ℕ) (current_average : ℚ) (next_innings_runs : ℕ) 
  (h1 : innings = 13)
  (h2 : current_average = 22)
  (h3 : next_innings_runs = 92) : 
  let total_runs : ℚ := innings * current_average
  let new_total_runs : ℚ := total_runs + next_innings_runs
  let new_average : ℚ := new_total_runs / (innings + 1)
  new_average - current_average = 5 := by sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l4173_417357


namespace NUMINAMATH_CALUDE_K_change_implies_equilibrium_shift_l4173_417388

-- Define the equilibrium constant as a function of temperature
def K (temperature : ℝ) : ℝ := sorry

-- Define a predicate for equilibrium shift
def equilibrium_shift (initial_state final_state : ℝ) : Prop :=
  initial_state ≠ final_state

-- Define a predicate for K change
def K_change (initial_K final_K : ℝ) : Prop :=
  initial_K ≠ final_K

-- Theorem statement
theorem K_change_implies_equilibrium_shift
  (initial_temp final_temp : ℝ)
  (h_K_change : K_change (K initial_temp) (K final_temp)) :
  equilibrium_shift initial_temp final_temp :=
sorry

end NUMINAMATH_CALUDE_K_change_implies_equilibrium_shift_l4173_417388


namespace NUMINAMATH_CALUDE_constant_term_expansion_l4173_417376

/-- The constant term in the expansion of (3x + 2/x)^8 -/
def constant_term : ℕ := 90720

/-- The binomial coefficient (8 choose 4) -/
def binom_8_4 : ℕ := 70

theorem constant_term_expansion :
  constant_term = binom_8_4 * 3^4 * 2^4 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l4173_417376


namespace NUMINAMATH_CALUDE_quarter_difference_in_nickels_l4173_417309

/-- The number of nickels in a quarter -/
def nickels_per_quarter : ℕ := 5

/-- The difference in nickels between two amounts of quarters -/
def nickel_difference (alice_quarters bob_quarters : ℕ) : ℤ :=
  (alice_quarters - bob_quarters) * nickels_per_quarter

theorem quarter_difference_in_nickels (q : ℕ) :
  nickel_difference (4 * q + 3) (2 * q + 8) = 10 * q - 25 := by sorry

end NUMINAMATH_CALUDE_quarter_difference_in_nickels_l4173_417309


namespace NUMINAMATH_CALUDE_remainder_equality_l4173_417307

theorem remainder_equality (a b c : ℤ) (hc : c ≠ 0) :
  c ∣ (a - b) → a ≡ b [ZMOD c] :=
by sorry

end NUMINAMATH_CALUDE_remainder_equality_l4173_417307


namespace NUMINAMATH_CALUDE_max_value_of_f_l4173_417349

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * x^2 + 2*a*x

theorem max_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 2) 
  (h3 : ∃ x ∈ Set.Icc 1 4, ∀ y ∈ Set.Icc 1 4, f a x ≤ f a y) 
  (h4 : ∃ x ∈ Set.Icc 1 4, f a x = -16/3) :
  ∃ x ∈ Set.Icc 1 4, ∀ y ∈ Set.Icc 1 4, f a y ≤ f a x ∧ f a x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4173_417349


namespace NUMINAMATH_CALUDE_car_cost_share_l4173_417300

/-- Given a car that costs $2,100 and is used for 7 days a week, with one person using it for 4 days,
    prove that the other person's share of the cost is $900. -/
theorem car_cost_share (total_cost : ℕ) (total_days : ℕ) (days_used_by_first : ℕ) :
  total_cost = 2100 →
  total_days = 7 →
  days_used_by_first = 4 →
  (total_cost * (total_days - days_used_by_first) / total_days : ℚ) = 900 := by
  sorry

#check car_cost_share

end NUMINAMATH_CALUDE_car_cost_share_l4173_417300


namespace NUMINAMATH_CALUDE_soda_cost_l4173_417381

/-- The cost of items in cents -/
structure Cost where
  burger : ℕ
  soda : ℕ

/-- The given conditions of the problem -/
def problem_conditions (c : Cost) : Prop :=
  2 * c.burger + c.soda = 210 ∧ c.burger + 2 * c.soda = 240

/-- The theorem stating that under the given conditions, a soda costs 90 cents -/
theorem soda_cost (c : Cost) : problem_conditions c → c.soda = 90 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l4173_417381


namespace NUMINAMATH_CALUDE_right_triangle_sets_l4173_417385

/-- Checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

theorem right_triangle_sets :
  ¬(isRightTriangle 4 6 8) ∧
  (isRightTriangle 5 12 13) ∧
  (isRightTriangle 6 8 10) ∧
  (isRightTriangle 7 24 25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l4173_417385


namespace NUMINAMATH_CALUDE_debate_committee_combinations_l4173_417392

/-- The number of teams in the debate club -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the organizing team -/
def organizing_team_selection : ℕ := 4

/-- The number of members selected from each non-organizing team -/
def other_team_selection : ℕ := 3

/-- The total number of members in the debate organizing committee -/
def committee_size : ℕ := 16

/-- The number of possible debate organizing committees -/
def num_committees : ℕ := 3442073600

theorem debate_committee_combinations :
  (num_teams * Nat.choose team_size organizing_team_selection * 
   (Nat.choose team_size other_team_selection ^ (num_teams - 1))) = num_committees :=
sorry

end NUMINAMATH_CALUDE_debate_committee_combinations_l4173_417392


namespace NUMINAMATH_CALUDE_circle_center_sum_l4173_417394

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, 
    the sum of the x and y coordinates of its center is -1 -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 9 - 4*h + 6*k) ∧ h + k = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l4173_417394


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l4173_417384

theorem nested_fraction_evaluation :
  1 / (1 + 1 / (2 + 1 / (1 + 1 / 4))) = 14 / 19 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l4173_417384


namespace NUMINAMATH_CALUDE_y_is_twenty_percent_of_x_l4173_417330

/-- Given two equations involving x, y, and z, prove that y is 20% of x -/
theorem y_is_twenty_percent_of_x (x y z : ℝ) 
  (eq1 : 0.3 * (x - y) = 0.2 * (x + y))
  (eq2 : 0.4 * (x + z) = 0.1 * (y - z)) :
  y = 0.2 * x := by
  sorry

end NUMINAMATH_CALUDE_y_is_twenty_percent_of_x_l4173_417330


namespace NUMINAMATH_CALUDE_screamers_lineup_count_l4173_417362

-- Define the total number of players
def total_players : ℕ := 12

-- Define the number of players in a lineup
def lineup_size : ℕ := 5

-- Define a function to calculate combinations
def combinations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.choose n r

-- Theorem statement
theorem screamers_lineup_count : 
  combinations (total_players - 2) (lineup_size - 1) * 2 + 
  combinations (total_players - 2) lineup_size = 672 := by
  sorry


end NUMINAMATH_CALUDE_screamers_lineup_count_l4173_417362


namespace NUMINAMATH_CALUDE_expression_simplification_l4173_417351

theorem expression_simplification (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y^2) - 5 * (2 + 3 * y) = -4 * y^2 - 17 * y - 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4173_417351


namespace NUMINAMATH_CALUDE_second_car_speed_l4173_417334

/-- Given two cars starting from opposite ends of a 60-mile highway at the same time,
    with one car traveling at 13 mph and both cars meeting after 2 hours,
    prove that the speed of the second car is 17 mph. -/
theorem second_car_speed (highway_length : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  highway_length = 60 →
  time = 2 →
  speed1 = 13 →
  speed1 * time + speed2 * time = highway_length →
  speed2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l4173_417334


namespace NUMINAMATH_CALUDE_quentavious_gum_pieces_l4173_417301

/-- Represents the types of coins --/
inductive Coin
  | Nickel
  | Dime
  | Quarter

/-- Calculates the number of gum pieces for a given coin type --/
def gumPieces (c : Coin) : ℕ :=
  match c with
  | Coin.Nickel => 2
  | Coin.Dime => 3
  | Coin.Quarter => 5

/-- Represents the initial state of coins --/
structure InitialCoins where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Represents the final state of coins --/
structure FinalCoins where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the number of gum pieces received --/
def gumReceived (initial : InitialCoins) (final : FinalCoins) : ℕ :=
  let exchanged_nickels := initial.nickels - final.nickels
  let exchanged_dimes := initial.dimes - final.dimes
  let exchanged_quarters := initial.quarters - final.quarters
  if exchanged_nickels > 0 && exchanged_dimes > 0 && exchanged_quarters > 0 then
    15
  else
    exchanged_nickels * gumPieces Coin.Nickel +
    exchanged_dimes * gumPieces Coin.Dime +
    exchanged_quarters * gumPieces Coin.Quarter

theorem quentavious_gum_pieces :
  let initial := InitialCoins.mk 5 6 4
  let final := FinalCoins.mk 2 1 0
  gumReceived initial final = 15 := by
  sorry

end NUMINAMATH_CALUDE_quentavious_gum_pieces_l4173_417301


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4173_417344

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 2 * y^2 + 3 * y - 35 = (2 * y + a) * (y + b)) →
  a - b = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4173_417344


namespace NUMINAMATH_CALUDE_problem_solution_l4173_417328

theorem problem_solution (a b c : ℕ+) (h1 : 3 * a = b^3) (h2 : 5 * a = c^2) (h3 : ∃ k : ℕ, a = k * 1^6) :
  (∃ m n : ℕ, a = 3 * m ∧ a = 5 * n) ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ a → p = 3 ∨ p = 5) ∧
  a = 1125 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4173_417328


namespace NUMINAMATH_CALUDE_smallest_common_shelving_count_l4173_417314

theorem smallest_common_shelving_count : Nat.lcm 6 17 = 102 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_shelving_count_l4173_417314


namespace NUMINAMATH_CALUDE_shares_problem_l4173_417375

theorem shares_problem (total : ℕ) (a b c : ℕ) : 
  total = 1760 →
  a + b + c = total →
  3 * b = 4 * a →
  5 * a = 3 * c →
  6 * a = 8 * b →
  8 * b = 20 * c →
  c = 250 := by
sorry

end NUMINAMATH_CALUDE_shares_problem_l4173_417375


namespace NUMINAMATH_CALUDE_quadrilateral_area_inequality_l4173_417370

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  convex : Bool
  area : ℝ

-- State the theorem
theorem quadrilateral_area_inequality (q : ConvexQuadrilateral) (h : q.convex = true) :
  q.area ≤ (q.a^2 + q.b^2 + q.c^2 + q.d^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_inequality_l4173_417370


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_c_equals_five_l4173_417366

theorem infinite_solutions_iff_c_equals_five (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + c * y) = 15 * y + 15) ↔ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_c_equals_five_l4173_417366


namespace NUMINAMATH_CALUDE_oak_trees_in_park_l4173_417350

theorem oak_trees_in_park (initial_trees : ℕ) (planted_trees : ℕ) : initial_trees = 5 → planted_trees = 4 → initial_trees + planted_trees = 9 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_in_park_l4173_417350


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l4173_417332

theorem product_of_three_numbers (x y z : ℚ) : 
  x + y + z = 36 →
  x = 3 * (y + z) →
  y = 6 * z →
  x * y * z = 268 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l4173_417332


namespace NUMINAMATH_CALUDE_music_store_sales_calculation_l4173_417361

/-- Represents the sales data for a mall with two stores -/
structure MallSales where
  num_cars : ℕ
  customers_per_car : ℕ
  sports_store_sales : ℕ

/-- Calculates the number of sales made by the music store -/
def music_store_sales (mall : MallSales) : ℕ :=
  mall.num_cars * mall.customers_per_car - mall.sports_store_sales

/-- Theorem: The music store sales is equal to the total customers minus sports store sales -/
theorem music_store_sales_calculation (mall : MallSales) 
  (h1 : mall.num_cars = 10)
  (h2 : mall.customers_per_car = 5)
  (h3 : mall.sports_store_sales = 20) :
  music_store_sales mall = 30 := by
  sorry

end NUMINAMATH_CALUDE_music_store_sales_calculation_l4173_417361


namespace NUMINAMATH_CALUDE_alex_trip_distance_l4173_417380

/-- The distance from Alex's house to the harbor --/
def distance : ℝ := sorry

/-- Alex's initial speed --/
def initial_speed : ℝ := 45

/-- Alex's speed increase --/
def speed_increase : ℝ := 20

/-- Time saved by increasing speed --/
def time_saved : ℝ := 1.75

/-- The total travel time if Alex continued at the initial speed --/
def total_time_initial_speed : ℝ := sorry

theorem alex_trip_distance :
  /- Alex drives 45 miles in the first hour -/
  (initial_speed = 45) →
  /- He would be 1.5 hours late if he continues at the initial speed -/
  (total_time_initial_speed = distance / initial_speed) →
  /- He increases his speed by 20 miles per hour for the rest of the trip -/
  (∃ t : ℝ, t > 0 ∧ t < total_time_initial_speed ∧
    distance = initial_speed + (total_time_initial_speed - t) * (initial_speed + speed_increase)) →
  /- He arrives 15 minutes (0.25 hours) early -/
  (time_saved = 1.75) →
  /- The distance from Alex's house to the harbor is 613 miles -/
  distance = 613 := by sorry

end NUMINAMATH_CALUDE_alex_trip_distance_l4173_417380


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l4173_417359

/-- A rectangle with perimeter 240 feet and area equal to eight times its perimeter has its longest side equal to 80 feet. -/
theorem rectangle_longest_side (l w : ℝ) : 
  (l > 0) → 
  (w > 0) → 
  (2 * l + 2 * w = 240) → 
  (l * w = 8 * (2 * l + 2 * w)) → 
  (max l w = 80) := by
sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l4173_417359


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l4173_417348

/-- Three points are collinear if and only if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_b_value :
  ∀ b : ℚ, collinear 5 (-3) (-b + 3) 5 (3*b + 1) 4 → b = 18/31 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l4173_417348


namespace NUMINAMATH_CALUDE_students_in_neither_clubs_l4173_417354

/-- Represents the number of students in various categories in a class --/
structure ClassMembers where
  total : ℕ
  chinese : ℕ
  math : ℕ
  both : ℕ

/-- Calculates the number of students in neither the Chinese nor Math club --/
def studentsInNeither (c : ClassMembers) : ℕ :=
  c.total - (c.chinese + c.math - c.both)

/-- Theorem stating the number of students in neither club for the given scenario --/
theorem students_in_neither_clubs (c : ClassMembers) 
  (h_total : c.total = 55)
  (h_chinese : c.chinese = 32)
  (h_math : c.math = 36)
  (h_both : c.both = 18) :
  studentsInNeither c = 5 := by
  sorry

#eval studentsInNeither { total := 55, chinese := 32, math := 36, both := 18 }

end NUMINAMATH_CALUDE_students_in_neither_clubs_l4173_417354


namespace NUMINAMATH_CALUDE_union_A_B_minus_three_intersection_A_B_equals_B_iff_l4173_417313

-- Define set A
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Statement 1: A ∪ B when m = -3
theorem union_A_B_minus_three : 
  A ∪ B (-3) = {x : ℝ | -7 ≤ x ∧ x ≤ 4} := by sorry

-- Statement 2: A ∩ B = B iff m ≥ -1
theorem intersection_A_B_equals_B_iff (m : ℝ) : 
  A ∩ B m = B m ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_union_A_B_minus_three_intersection_A_B_equals_B_iff_l4173_417313


namespace NUMINAMATH_CALUDE_prime_divisor_fourth_power_l4173_417386

theorem prime_divisor_fourth_power (n : ℕ+) 
  (h : ∀ d : ℕ+, d ∣ n → ¬(n^2 ≤ d^4 ∧ d^4 ≤ n^3)) : 
  ∃ p : ℕ, p.Prime ∧ p ∣ n ∧ p^4 > n^3 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_fourth_power_l4173_417386


namespace NUMINAMATH_CALUDE_perpendicular_line_properties_l4173_417341

/-- Given a line l₁ and a point A, this theorem proves properties of the perpendicular line l₂ passing through A -/
theorem perpendicular_line_properties (x y : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 1 = 0
  let A : ℝ × ℝ := (3, 0)
  let l₂ : ℝ → ℝ → Prop := λ x y => 4 * x - 3 * y - 12 = 0
  -- l₂ passes through A
  (l₂ A.1 A.2) →
  -- l₂ is perpendicular to l₁
  (∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₁ x₂ y₂ → l₂ x₁ y₁ → l₂ x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (3) + (y₂ - y₁) * (4)) * ((x₂ - x₁) * (4) + (y₂ - y₁) * (-3)) = 0) →
  -- The equation of l₂ is correct
  (∀ x y, l₂ x y ↔ 4 * x - 3 * y - 12 = 0) ∧
  -- The area of the triangle is 6
  (let x_intercept := 3
   let y_intercept := 4
   (1 / 2 : ℝ) * x_intercept * y_intercept = 6) := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_properties_l4173_417341


namespace NUMINAMATH_CALUDE_cosine_value_proof_l4173_417399

theorem cosine_value_proof (α : Real) 
    (h : Real.sin (α - π/3) = 1/3) : 
    Real.cos (π/6 + α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_proof_l4173_417399


namespace NUMINAMATH_CALUDE_set_equality_condition_l4173_417373

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 3}

-- State the theorem
theorem set_equality_condition (a : ℝ) : 
  A ∪ B a = A ↔ a ≤ -2 ∨ a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_set_equality_condition_l4173_417373


namespace NUMINAMATH_CALUDE_inequality_implies_not_six_l4173_417321

theorem inequality_implies_not_six (m : ℝ) : m + 3 < (-m + 1) - (-13) → m ≠ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_not_six_l4173_417321


namespace NUMINAMATH_CALUDE_company_workers_problem_l4173_417322

theorem company_workers_problem (total_workers : ℕ) 
  (h1 : total_workers % 3 = 0)  -- Ensures total_workers is divisible by 3
  (h2 : total_workers ≠ 0)      -- Ensures total_workers is not zero
  (h3 : (total_workers / 3) % 5 = 0)  -- Ensures workers without plan is divisible by 5
  (h4 : (2 * total_workers / 3) % 5 = 0)  -- Ensures workers with plan is divisible by 5
  (h5 : 40 * (2 * total_workers / 3) / 100 = 128)  -- 128 male workers
  : (7 * total_workers / 15 : ℕ) = 224 := by
  sorry

end NUMINAMATH_CALUDE_company_workers_problem_l4173_417322


namespace NUMINAMATH_CALUDE_field_breadth_is_50_l4173_417383

/-- Proves that the breadth of a field is 50 meters given specific conditions -/
theorem field_breadth_is_50 (field_length : ℝ) (tank_length tank_width tank_depth : ℝ) 
  (field_rise : ℝ) (b : ℝ) : 
  field_length = 90 →
  tank_length = 25 →
  tank_width = 20 →
  tank_depth = 4 →
  field_rise = 0.5 →
  tank_length * tank_width * tank_depth = (field_length * b - tank_length * tank_width) * field_rise →
  b = 50 := by
  sorry

end NUMINAMATH_CALUDE_field_breadth_is_50_l4173_417383


namespace NUMINAMATH_CALUDE_power_function_through_point_l4173_417356

/-- A power function passing through (4, 1/2) has f(1/16) = 4 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x^a) →  -- f is a power function
  f 4 = 1/2 →             -- f passes through (4, 1/2)
  f (1/16) = 4 :=         -- prove f(1/16) = 4
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l4173_417356


namespace NUMINAMATH_CALUDE_group_division_arrangements_l4173_417311

/-- The number of teachers --/
def num_teachers : ℕ := 2

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of groups --/
def num_groups : ℕ := 2

/-- The number of teachers per group --/
def teachers_per_group : ℕ := 1

/-- The number of students per group --/
def students_per_group : ℕ := 2

/-- The total number of arrangements --/
def total_arrangements : ℕ := 12

theorem group_division_arrangements :
  (Nat.choose num_teachers teachers_per_group) *
  (Nat.choose num_students students_per_group) =
  total_arrangements :=
sorry

end NUMINAMATH_CALUDE_group_division_arrangements_l4173_417311


namespace NUMINAMATH_CALUDE_binomial_expansion_product_l4173_417308

theorem binomial_expansion_product (a a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) : 
  (a + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_product_l4173_417308


namespace NUMINAMATH_CALUDE_min_a_cubes_correct_l4173_417372

def min_a_cubes (total_cubes : ℕ) (max_funding : ℕ) (price_a : ℕ) (price_b : ℕ) : ℕ :=
  let a : ℕ := 15
  a

theorem min_a_cubes_correct (total_cubes : ℕ) (max_funding : ℕ) (price_a : ℕ) (price_b : ℕ)
  (h_total : total_cubes = 40)
  (h_max_funding : max_funding = 776)
  (h_price_a : price_a = 15)
  (h_price_b : price_b = 22) :
  let a := min_a_cubes total_cubes max_funding price_a price_b
  let b := total_cubes - a
  (b ≥ a ∧ 
   price_a * a + price_b * b ≤ max_funding ∧
   ∀ x : ℕ, x < a → 
     (let b' := total_cubes - x
      ¬(b' ≥ x ∧ price_a * x + price_b * b' ≤ max_funding))) := by
  sorry

#check min_a_cubes_correct

end NUMINAMATH_CALUDE_min_a_cubes_correct_l4173_417372


namespace NUMINAMATH_CALUDE_correct_fraction_is_five_thirds_l4173_417329

/-- The percentage error when using an incorrect fraction instead of the correct one. -/
def percentage_error : ℚ := 64.00000000000001

/-- The incorrect fraction used by the student. -/
def incorrect_fraction : ℚ := 3/5

/-- The correct fraction that should have been used. -/
def correct_fraction : ℚ := 5/3

/-- Theorem stating that given the percentage error and incorrect fraction, 
    the correct fraction is 5/3. -/
theorem correct_fraction_is_five_thirds :
  (1 - percentage_error / 100) * correct_fraction = incorrect_fraction :=
sorry

end NUMINAMATH_CALUDE_correct_fraction_is_five_thirds_l4173_417329


namespace NUMINAMATH_CALUDE_inverse_42_mod_53_l4173_417303

theorem inverse_42_mod_53 (h : (11⁻¹ : ZMod 53) = 31) : (42⁻¹ : ZMod 53) = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_42_mod_53_l4173_417303


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_l4173_417391

def is_valid_equation (a b c : ℕ) : Prop :=
  a + b = c ∧ 
  a ≥ 1000 ∧ a < 10000 ∧
  b ≥ 10 ∧ b < 100 ∧
  c ≥ 1000 ∧ c < 10000

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  (∀ i j, i ≠ j → digits.nthLe i sorry ≠ digits.nthLe j sorry) ∧
  digits.length ≤ 10

theorem smallest_four_digit_number (a b c : ℕ) :
  is_valid_equation a b c →
  has_distinct_digits a →
  has_distinct_digits b →
  has_distinct_digits c →
  (∀ x, has_distinct_digits x → x ≥ 1000 → x < c → False) →
  c = 2034 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_l4173_417391


namespace NUMINAMATH_CALUDE_paint_mixture_intensity_l4173_417326

theorem paint_mixture_intensity 
  (original_intensity : ℝ) 
  (added_intensity : ℝ) 
  (fraction_replaced : ℝ) :
  original_intensity = 0.5 →
  added_intensity = 0.2 →
  fraction_replaced = 2/3 →
  let new_intensity := (1 - fraction_replaced) * original_intensity + fraction_replaced * added_intensity
  new_intensity = 0.3 := by
sorry


end NUMINAMATH_CALUDE_paint_mixture_intensity_l4173_417326


namespace NUMINAMATH_CALUDE_parabola_translation_l4173_417312

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (-1) 0 2
  let translated := translate original 2 (-3)
  y = -(x - 2)^2 - 1 ↔ y = translated.a * x^2 + translated.b * x + translated.c :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l4173_417312


namespace NUMINAMATH_CALUDE_expression_simplification_l4173_417317

theorem expression_simplification (x : ℝ) (h : x^2 + x - 5 = 0) :
  (x - 2) / (x^2 - 4*x + 4) / (x + 2 - (x^2 + x - 4) / (x - 2)) + 1 / (x + 1) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4173_417317


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4173_417387

theorem inequality_solution_set (x : ℝ) : (3 * x + 1) / (1 - 2 * x) ≥ 0 ↔ -1/3 ≤ x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4173_417387


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l4173_417374

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = q * a n

theorem geometric_sequence_minimum_value 
  (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a) 
  (h_cond : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  (∀ q, q > 0 → 2 * a 5 + a 4 ≥ 12 * Real.sqrt 3) ∧
  (∃ q, q > 0 ∧ 2 * a 5 + a 4 = 12 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l4173_417374


namespace NUMINAMATH_CALUDE_circle_tangency_l4173_417342

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with equation x² + y² - 6x - 8y + F = 0 -/
def C₂ (F : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + F = 0}

/-- Two circles are internally tangent if they intersect at exactly one point
    and one circle is completely inside the other -/
def internally_tangent (C₁ C₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C₁ ∧ p ∈ C₂ ∧ (∀ q, q ∈ C₁ ∩ C₂ → q = p) ∧
  (∀ r, r ∈ C₁ → r ∈ C₂ ∨ r = p)

/-- Theorem: If C₁ is internally tangent to C₂, then F = -11 -/
theorem circle_tangency (F : ℝ) :
  internally_tangent C₁ (C₂ F) → F = -11 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_l4173_417342


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4173_417305

def is_increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ q > 1

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : is_increasing_geometric_sequence a q)
  (h_sum : a 1 + a 5 = 17)
  (h_prod : a 2 * a 4 = 16) :
  q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4173_417305


namespace NUMINAMATH_CALUDE_largest_b_value_l4173_417377

/-- The polynomial function representing the equation -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^4 - a*x^3 - b*x^2 - c*x - 2007

/-- Predicate to check if a number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Predicate to check if the equation has exactly three distinct integer solutions -/
def hasThreeDistinctIntegerSolutions (a b c : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    isInteger x ∧ isInteger y ∧ isInteger z ∧
    f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0 ∧
    ∀ w : ℝ, f a b c w = 0 → w = x ∨ w = y ∨ w = z

/-- The main theorem -/
theorem largest_b_value :
  ∃ b_max : ℝ, (∀ a c b : ℝ, hasThreeDistinctIntegerSolutions a b c → b ≤ b_max) ∧
    (∃ a c : ℝ, hasThreeDistinctIntegerSolutions a b_max c) ∧
    b_max = 3343 := by sorry

end NUMINAMATH_CALUDE_largest_b_value_l4173_417377


namespace NUMINAMATH_CALUDE_sales_price_calculation_l4173_417318

theorem sales_price_calculation (C S : ℝ) : 
  S - C = 1.25 * C →  -- Gross profit is 125% of the cost
  S - C = 30 →        -- Gross profit is $30
  S = 54 :=           -- Sales price is $54
by sorry

end NUMINAMATH_CALUDE_sales_price_calculation_l4173_417318


namespace NUMINAMATH_CALUDE_fourth_side_length_l4173_417347

/-- A quadrilateral inscribed in a circle with radius 150√3, where three sides are 150 units long -/
structure InscribedQuadrilateral where
  -- The radius of the circle
  r : ℝ
  -- The lengths of the four sides of the quadrilateral
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  s₄ : ℝ
  -- Conditions
  h_radius : r = 150 * Real.sqrt 3
  h_three_sides : s₁ = 150 ∧ s₂ = 150 ∧ s₃ = 150

/-- The theorem stating that the fourth side of the quadrilateral is 450 units long -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.s₄ = 450 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l4173_417347


namespace NUMINAMATH_CALUDE_usb_available_space_l4173_417320

theorem usb_available_space (total_capacity : ℝ) (occupied_percentage : ℝ) 
  (h1 : total_capacity = 128)
  (h2 : occupied_percentage = 75) :
  (1 - occupied_percentage / 100) * total_capacity = 32 := by
  sorry

end NUMINAMATH_CALUDE_usb_available_space_l4173_417320


namespace NUMINAMATH_CALUDE_second_largest_part_l4173_417306

theorem second_largest_part (total : ℚ) (a b c d : ℚ) : 
  total = 120 → 
  a + b + c + d = total →
  a / 3 = b / 2 →
  a / 3 = c / 4 →
  a / 3 = d / 5 →
  (max b (min c d)) = 240 / 7 := by
  sorry

end NUMINAMATH_CALUDE_second_largest_part_l4173_417306


namespace NUMINAMATH_CALUDE_calculation_proof_l4173_417360

theorem calculation_proof : (36 / (9 + 2 - 6)) * 4 = 28.8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4173_417360


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l4173_417333

theorem chess_tournament_participants :
  let n : ℕ := 28
  let total_games : ℕ := n * (n - 1) / 2
  let uncounted_games : ℕ := 10 * n
  total_games + uncounted_games = 672 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l4173_417333


namespace NUMINAMATH_CALUDE_complex_product_equality_complex_sum_equality_l4173_417363

-- Define the complex number i
def i : ℂ := Complex.I

-- Part 1
theorem complex_product_equality : 
  (1 : ℂ) * (1 - i) * (-1/2 + (Real.sqrt 3)/2 * i) * (1 + i) = -1 + Real.sqrt 3 * i := by sorry

-- Part 2
theorem complex_sum_equality :
  (2 + 2*i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i))^2010 = -1 := by sorry

end NUMINAMATH_CALUDE_complex_product_equality_complex_sum_equality_l4173_417363


namespace NUMINAMATH_CALUDE_double_acute_angle_range_l4173_417324

theorem double_acute_angle_range (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  0 < 2 * α ∧ 2 * α < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_double_acute_angle_range_l4173_417324


namespace NUMINAMATH_CALUDE_sin_equality_integer_solutions_l4173_417379

theorem sin_equality_integer_solutions (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.sin (750 * π / 180) →
  n = 30 ∨ n = 150 ∨ n = -30 ∨ n = -150 :=
by sorry

end NUMINAMATH_CALUDE_sin_equality_integer_solutions_l4173_417379


namespace NUMINAMATH_CALUDE_trig_equation_solution_l4173_417340

theorem trig_equation_solution (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = k * Real.pi + Real.pi / 2 ∨ 
            x = k * Real.pi / 2 + Real.pi / 4 ∨ 
            x = k * Real.pi / 3 + Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l4173_417340


namespace NUMINAMATH_CALUDE_river_throw_count_l4173_417393

/-- Represents the number of objects thrown by a person -/
structure ThrowCount where
  sticks : ℕ
  rocks : ℕ

/-- Calculates the total number of objects thrown -/
def total_objects (tc : ThrowCount) : ℕ :=
  tc.sticks + tc.rocks

theorem river_throw_count :
  let ted : ThrowCount := { sticks := 12, rocks := 18 }
  let bill : ThrowCount := { sticks := ted.sticks + 6, rocks := ted.rocks / 2 }
  let alice : ThrowCount := { sticks := ted.sticks / 2, rocks := bill.rocks * 3 }
  total_objects bill + total_objects alice = 60 := by
  sorry

end NUMINAMATH_CALUDE_river_throw_count_l4173_417393


namespace NUMINAMATH_CALUDE_dave_guitar_strings_l4173_417325

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 4

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 24

/-- The total number of guitar strings Dave needs to replace -/
def total_strings : ℕ := strings_per_night * shows_per_week * total_weeks

theorem dave_guitar_strings :
  total_strings = 576 := by sorry

end NUMINAMATH_CALUDE_dave_guitar_strings_l4173_417325


namespace NUMINAMATH_CALUDE_cubic_root_sum_l4173_417310

theorem cubic_root_sum (a b : ℝ) : 
  (∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧ r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
   (∀ x : ℝ, 4*x^3 + 7*a*x^2 + 6*b*x + 2*a = 0 ↔ (x = r ∨ x = s ∨ x = t)) ∧
   (r + s + t)^3 = 125) →
  a = -20/7 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l4173_417310


namespace NUMINAMATH_CALUDE_carters_reading_rate_l4173_417389

/-- Given reading rates for Oliver, Lucy, and Carter, prove Carter's reading rate -/
theorem carters_reading_rate 
  (oliver_rate : ℕ) 
  (lucy_rate : ℕ) 
  (carter_rate : ℕ) 
  (h1 : oliver_rate = 40)
  (h2 : lucy_rate = oliver_rate + 20)
  (h3 : carter_rate = lucy_rate / 2) : 
  carter_rate = 30 := by
sorry

end NUMINAMATH_CALUDE_carters_reading_rate_l4173_417389


namespace NUMINAMATH_CALUDE_division_increase_by_digit_swap_l4173_417323

theorem division_increase_by_digit_swap (n : Nat) (d : Nat) :
  n = 952473 →
  d = 18 →
  (954273 / d) - (n / d) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_division_increase_by_digit_swap_l4173_417323


namespace NUMINAMATH_CALUDE_median_is_106_l4173_417336

-- Define the list
def list_size (n : ℕ) : ℕ := if n ≤ 150 then n else 0

-- Define the sum of the list sizes
def total_elements : ℕ := (Finset.range 151).sum list_size

-- Define the median position
def median_position : ℕ := (total_elements + 1) / 2

-- Theorem statement
theorem median_is_106 : 
  ∃ (cumsum : ℕ → ℕ), 
    (∀ n, cumsum n = (Finset.range (n + 1)).sum list_size) ∧
    (cumsum 105 < median_position) ∧
    (median_position ≤ cumsum 106) :=
sorry

end NUMINAMATH_CALUDE_median_is_106_l4173_417336


namespace NUMINAMATH_CALUDE_function_inequality_l4173_417339

/-- Given a function f(x) = ln x - 3x defined on (0, +∞), and for all x ∈ (0, +∞),
    f(x) ≤ x(ae^x - 4) + b, prove that a + b ≥ 0. -/
theorem function_inequality (a b : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.log x - 3 * x ≤ x * (a * Real.exp x - 4) + b) → 
  a + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l4173_417339


namespace NUMINAMATH_CALUDE_garden_width_to_perimeter_ratio_l4173_417304

/-- Given a rectangular garden with length 23 feet and width 15 feet, 
    the ratio of its width to its perimeter is 15:76. -/
theorem garden_width_to_perimeter_ratio :
  let garden_length : ℕ := 23
  let garden_width : ℕ := 15
  let perimeter : ℕ := 2 * (garden_length + garden_width)
  (garden_width : ℚ) / perimeter = 15 / 76 := by
  sorry

end NUMINAMATH_CALUDE_garden_width_to_perimeter_ratio_l4173_417304


namespace NUMINAMATH_CALUDE_cosine_in_triangle_l4173_417397

/-- Given a triangle ABC with sides a and b, prove that if a = 4, b = 5, 
    and cos(B-A) = 31/32, then cos B = 9/16 -/
theorem cosine_in_triangle (A B C : ℝ) (a b c : ℝ) : 
  a = 4 → b = 5 → Real.cos (B - A) = 31/32 → Real.cos B = 9/16 := by sorry

end NUMINAMATH_CALUDE_cosine_in_triangle_l4173_417397


namespace NUMINAMATH_CALUDE_strongroom_keys_l4173_417331

theorem strongroom_keys (n : ℕ) : n > 0 → (
  (∃ (key_distribution : Fin 5 → Finset (Fin 10)),
    (∀ d : Fin 5, (key_distribution d).card = n) ∧
    (∀ majority : Finset (Fin 5), majority.card ≥ 3 →
      (majority.biUnion key_distribution).card = 10) ∧
    (∀ minority : Finset (Fin 5), minority.card ≤ 2 →
      (minority.biUnion key_distribution).card < 10))
  ↔ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_strongroom_keys_l4173_417331
