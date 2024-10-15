import Mathlib

namespace NUMINAMATH_CALUDE_roots_of_f_l1298_129843

-- Define the polynomial function f
def f (x : ℝ) : ℝ := -3 * (x + 5)^2 + 45 * (x + 5) - 108

-- State the theorem
theorem roots_of_f :
  (f 7 = 0) ∧ (f (-2) = 0) ∧
  (∀ x : ℝ, f x = 0 → x = 7 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_f_l1298_129843


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1298_129821

theorem perfect_square_trinomial (a b t : ℝ) : 
  (∃ k : ℝ, a^2 + (2*t - 1)*a*b + 4*b^2 = (k*a + 2*b)^2) → 
  (t = 5/2 ∨ t = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1298_129821


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1298_129860

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (6 * x) / (2 * y + z) + (3 * y) / (x + 2 * z) + (9 * z) / (x + y) ≥ 83 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
    (6 * x) / (2 * y + z) + (3 * y) / (x + 2 * z) + (9 * z) / (x + y) < 83 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1298_129860


namespace NUMINAMATH_CALUDE_each_angle_less_than_sum_implies_acute_l1298_129811

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the property that each angle is less than the sum of the other two
def each_angle_less_than_sum (t : Triangle) : Prop :=
  t.A < t.B + t.C ∧ t.B < t.A + t.C ∧ t.C < t.A + t.B

-- Define an acute triangle
def is_acute_triangle (t : Triangle) : Prop :=
  t.A < 90 ∧ t.B < 90 ∧ t.C < 90

-- Theorem statement
theorem each_angle_less_than_sum_implies_acute (t : Triangle) :
  each_angle_less_than_sum t → is_acute_triangle t :=
by sorry

end NUMINAMATH_CALUDE_each_angle_less_than_sum_implies_acute_l1298_129811


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_one_l1298_129828

theorem integral_sqrt_minus_one (f : ℝ → ℝ) :
  (∀ x, f x = Real.sqrt (1 - x^2) - 1) →
  (∫ x in (-1)..1, f x) = π / 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_one_l1298_129828


namespace NUMINAMATH_CALUDE_log_expression_arbitrarily_small_l1298_129803

theorem log_expression_arbitrarily_small :
  ∀ ε > 0, ∃ x > (2/3 : ℝ), Real.log (x^2 + 3) - 2 * Real.log x < ε :=
by sorry

end NUMINAMATH_CALUDE_log_expression_arbitrarily_small_l1298_129803


namespace NUMINAMATH_CALUDE_wednesday_sites_count_l1298_129891

theorem wednesday_sites_count (monday_sites tuesday_sites : ℕ)
  (monday_avg tuesday_avg wednesday_avg overall_avg : ℚ)
  (h1 : monday_sites = 5)
  (h2 : tuesday_sites = 5)
  (h3 : monday_avg = 7)
  (h4 : tuesday_avg = 5)
  (h5 : wednesday_avg = 8)
  (h6 : overall_avg = 7) :
  ∃ wednesday_sites : ℕ,
    (monday_sites * monday_avg + tuesday_sites * tuesday_avg + wednesday_sites * wednesday_avg) /
    (monday_sites + tuesday_sites + wednesday_sites : ℚ) = overall_avg ∧
    wednesday_sites = 10 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_sites_count_l1298_129891


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1298_129838

theorem arithmetic_mean_problem (p q r : ℝ) : 
  (p + q) / 2 = 10 → 
  (q + r) / 2 = 26 → 
  r - p = 32 → 
  (q + r) / 2 = 26 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1298_129838


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l1298_129832

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - 2*x*y = 0) :
  2*x + y ≥ 9/2 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ - 2*x₀*y₀ = 0 ∧ 2*x₀ + y₀ = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l1298_129832


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1298_129842

/-- The eccentricity of an ellipse with equation 16x²+4y²=1 is √3/2 -/
theorem ellipse_eccentricity : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ (x y : ℝ), 16 * x^2 + 4 * y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a^2 - b^2) / a^2 = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1298_129842


namespace NUMINAMATH_CALUDE_T_equals_five_l1298_129888

theorem T_equals_five :
  let T := 1 / (3 - Real.sqrt 8) - 1 / (Real.sqrt 8 - Real.sqrt 7) + 
           1 / (Real.sqrt 7 - Real.sqrt 6) - 1 / (Real.sqrt 6 - Real.sqrt 5) + 
           1 / (Real.sqrt 5 - 2)
  T = 5 := by sorry

end NUMINAMATH_CALUDE_T_equals_five_l1298_129888


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l1298_129817

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y ≠ 1 ∧ 3 * y^2 - m * y - 3 = 0 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l1298_129817


namespace NUMINAMATH_CALUDE_solution_y_initial_weight_l1298_129859

/-- Proves that the initial weight of solution Y is 8 kg given the problem conditions --/
theorem solution_y_initial_weight :
  ∀ (W : ℝ),
  (W > 0) →
  (0.20 * W = W * 0.20) →
  (0.25 * W = 0.20 * W + 0.4) →
  W = 8 := by
sorry

end NUMINAMATH_CALUDE_solution_y_initial_weight_l1298_129859


namespace NUMINAMATH_CALUDE_motor_pool_vehicles_l1298_129831

theorem motor_pool_vehicles (x y : ℕ) : 
  x + y < 18 →
  y < 2 * x →
  x + 4 < y →
  (x = 6 ∧ y = 11) ∨ (∀ a b : ℕ, (a + b < 18 ∧ b < 2 * a ∧ a + 4 < b) → (a ≠ x ∨ b ≠ y)) :=
by sorry

end NUMINAMATH_CALUDE_motor_pool_vehicles_l1298_129831


namespace NUMINAMATH_CALUDE_x_eq_one_iff_quadratic_eq_zero_l1298_129800

theorem x_eq_one_iff_quadratic_eq_zero : ∀ x : ℝ, x = 1 ↔ x^2 - 2*x + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_x_eq_one_iff_quadratic_eq_zero_l1298_129800


namespace NUMINAMATH_CALUDE_fraction_value_l1298_129858

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 3 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  (a * c) / (b * d) = 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1298_129858


namespace NUMINAMATH_CALUDE_least_hour_square_remainder_fifteen_satisfies_condition_fifteen_is_least_l1298_129853

theorem least_hour_square_remainder (n : ℕ) : n > 9 ∧ n % 12 = (n^2) % 12 → n ≥ 15 := by
  sorry

theorem fifteen_satisfies_condition : 15 % 12 = (15^2) % 12 := by
  sorry

theorem fifteen_is_least : ∀ m : ℕ, m > 9 ∧ m % 12 = (m^2) % 12 → m ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_least_hour_square_remainder_fifteen_satisfies_condition_fifteen_is_least_l1298_129853


namespace NUMINAMATH_CALUDE_collinear_dots_probability_l1298_129887

/-- The number of dots in each row and column of the grid -/
def gridSize : ℕ := 5

/-- The number of possible sets of four collinear dots in a 5x5 grid -/
def collinearSets : ℕ := 16

/-- The total number of ways to choose 4 dots from 25 -/
def totalChoices : ℕ := 12650

/-- The probability of selecting four collinear dots in a 5x5 grid -/
theorem collinear_dots_probability :
  (collinearSets : ℚ) / totalChoices = 8 / 6325 := by sorry

end NUMINAMATH_CALUDE_collinear_dots_probability_l1298_129887


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1298_129834

-- Problem 1
theorem problem_1 (a b : ℝ) (ha : a ≠ 0) : (2 * a^2 * b) * a * b^2 / (4 * a^3) = (1/2) * b^3 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (2*x + 5) * (x - 3) = 2*x^2 - x - 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1298_129834


namespace NUMINAMATH_CALUDE_max_games_24_l1298_129808

/-- Represents a chess tournament with 8 players -/
structure ChessTournament where
  players : Finset (Fin 8)
  games : Finset (Fin 8 × Fin 8)
  hplayers : players.card = 8
  hgames : ∀ (i j : Fin 8), (i, j) ∈ games → i ≠ j
  hunique : ∀ (i j : Fin 8), (i, j) ∈ games → (j, i) ∉ games

/-- No five players all play each other -/
def noFiveAllPlay (t : ChessTournament) : Prop :=
  ∀ (s : Finset (Fin 8)), s.card = 5 →
    ∃ (i j : Fin 8), i ∈ s ∧ j ∈ s ∧ (i, j) ∉ t.games ∧ (j, i) ∉ t.games

/-- The main theorem: maximum number of games is 24 -/
theorem max_games_24 (t : ChessTournament) (h : noFiveAllPlay t) :
  t.games.card ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_max_games_24_l1298_129808


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l1298_129822

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 5 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)) / 6)) →
  (0 ≤ n) ∧ (∀ m : ℤ, m < n → m + 5 ≥ 3 * ((m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)) / 6)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l1298_129822


namespace NUMINAMATH_CALUDE_ratio_problem_l1298_129876

theorem ratio_problem : ∃ x : ℚ, (150 : ℚ) / 1 = x / 2 ∧ x = 300 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l1298_129876


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l1298_129812

/-- Given a point A with coordinates (-2, 4), this theorem proves that the point
    symmetric to A with respect to the y-axis has coordinates (2, 4). -/
theorem symmetric_point_wrt_y_axis :
  let A : ℝ × ℝ := (-2, 4)
  let symmetric_point := (-(A.1), A.2)
  symmetric_point = (2, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l1298_129812


namespace NUMINAMATH_CALUDE_division_theorem_l1298_129856

/-- The dividend polynomial -/
def f (x : ℝ) : ℝ := 3*x^5 - 2*x^3 + 5*x - 9

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The proposed remainder polynomial -/
def r (x : ℝ) : ℝ := 92*x - 95

/-- Statement: The remainder when f(x) is divided by g(x) is r(x) -/
theorem division_theorem : ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l1298_129856


namespace NUMINAMATH_CALUDE_divisibility_implication_l1298_129815

theorem divisibility_implication (x y : ℤ) :
  ∃ k : ℤ, 14 * x + 13 * y = 11 * k → ∃ m : ℤ, 19 * x + 9 * y = 11 * m :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1298_129815


namespace NUMINAMATH_CALUDE_player1_wins_533_player1_wins_1000_l1298_129846

/-- A game where two players alternately write 1 or 2, and the player who makes the sum reach or exceed the target loses. -/
def Game (target : ℕ) := Unit

/-- A strategy for playing the game. -/
def Strategy (target : ℕ) := Unit

/-- Determines if a strategy is winning for Player 1. -/
def is_winning_strategy (target : ℕ) (s : Strategy target) : Prop := sorry

/-- Player 1 has a winning strategy for the game with target 533. -/
theorem player1_wins_533 : ∃ s : Strategy 533, is_winning_strategy 533 s := sorry

/-- Player 1 has a winning strategy for the game with target 1000. -/
theorem player1_wins_1000 : ∃ s : Strategy 1000, is_winning_strategy 1000 s := sorry

end NUMINAMATH_CALUDE_player1_wins_533_player1_wins_1000_l1298_129846


namespace NUMINAMATH_CALUDE_alternating_color_probability_value_l1298_129862

/-- Represents the number of balls of each color in the box -/
def num_balls : ℕ := 5

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := 3 * num_balls

/-- Calculates the number of ways to arrange the balls -/
def total_arrangements : ℕ := Nat.choose total_balls num_balls * Nat.choose (2 * num_balls) num_balls

/-- Calculates the number of successful sequences (alternating colors) -/
def successful_sequences : ℕ := 2 * (3 ^ (num_balls - 1))

/-- The probability of drawing balls with alternating colors -/
def alternating_color_probability : ℚ := successful_sequences / total_arrangements

theorem alternating_color_probability_value : alternating_color_probability = 162 / 1001 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_value_l1298_129862


namespace NUMINAMATH_CALUDE_rational_polynomial_has_rational_coeffs_l1298_129864

/-- A polynomial that maps rationals to rationals has rational coefficients -/
theorem rational_polynomial_has_rational_coeffs (P : Polynomial ℚ) :
  (∀ q : ℚ, ∃ r : ℚ, P.eval q = r) →
  (∀ q : ℚ, ∃ r : ℚ, (P.eval q : ℚ) = r) →
  ∀ i : ℕ, ∃ q : ℚ, P.coeff i = q :=
sorry

end NUMINAMATH_CALUDE_rational_polynomial_has_rational_coeffs_l1298_129864


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1298_129806

/-- Given a real number a and a function f(x) = x³ + ax² + (a-3)x with derivative f'(x),
    where f'(x) is an even function, prove that the equation of the tangent line to
    the curve y = f(x) at the point (2, f(2)) is 9x - y - 16 = 0. -/
theorem tangent_line_equation (a : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + (a-3)*x
  let f' : ℝ → ℝ := λ x => 3*x^2 + 2*a*x + (a-3)
  (∀ x, f' x = f' (-x)) → 
  (λ x y => 9*x - y - 16 = 0) = (λ x y => y - f 2 = f' 2 * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1298_129806


namespace NUMINAMATH_CALUDE_frog_jump_probability_l1298_129807

-- Define the probability function
noncomputable def Q (x y : ℝ) : ℝ := sorry

-- Define the boundary conditions
axiom vertical_boundary : ∀ y, 0 ≤ y ∧ y ≤ 6 → Q 0 y = 1 ∧ Q 6 y = 1
axiom horizontal_boundary : ∀ x, 0 ≤ x ∧ x ≤ 6 → Q x 0 = 0 ∧ Q x 6 = 0

-- Define the recursive relation
axiom recursive_relation : 
  Q 2 3 = (1/4) * Q 1 3 + (1/4) * Q 3 3 + (1/4) * Q 2 2 + (1/4) * Q 2 4

-- Theorem to prove
theorem frog_jump_probability : Q 2 3 = 5/8 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l1298_129807


namespace NUMINAMATH_CALUDE_circle_radius_l1298_129824

/-- The radius of the circle described by x^2 + y^2 - 4x + 6y = 0 is √13 -/
theorem circle_radius (x y : ℝ) : 
  (∀ x y, x^2 + y^2 - 4*x + 6*y = 0) → 
  ∃ r : ℝ, r = Real.sqrt 13 ∧ ∀ x y, (x - 2)^2 + (y + 3)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l1298_129824


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_shifted_sin_l1298_129857

theorem sin_cos_sum_equals_shifted_sin (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * (x + π / 12)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_shifted_sin_l1298_129857


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1298_129801

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ p = 2^n - 1 ∧ Prime p

theorem largest_mersenne_prime_under_500 :
  ∃ p : ℕ, p = 127 ∧ 
    is_mersenne_prime p ∧ 
    p < 500 ∧ 
    ∀ q : ℕ, is_mersenne_prime q → q < 500 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1298_129801


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1298_129849

theorem constant_term_expansion (x : ℝ) : 
  (∃ c : ℝ, c = -160 ∧ 
   ∃ f : ℝ → ℝ, f x = (2*x - 1/x)^6 ∧ 
   ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1298_129849


namespace NUMINAMATH_CALUDE_problem_statement_l1298_129844

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_statement (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : has_period f 2)
  (h3 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f (-1) + f (-2017) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1298_129844


namespace NUMINAMATH_CALUDE_triangle_problem_l1298_129871

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
this theorem proves that under certain conditions, the angle C is 60° and 
the sides have specific lengths.
-/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Positive side lengths
  A > 0 → B > 0 → C > 0 →  -- Positive angles
  a > b →  -- Given condition
  a * (Real.sqrt 3 * Real.tan B - 1) = 
    (b * Real.cos A / Real.cos B) + (c * Real.cos A / Real.cos C) →  -- Given equation
  a + b + c = 20 →  -- Perimeter condition
  (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3 →  -- Area condition
  C = Real.pi / 3 ∧ a = 8 ∧ b = 5 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1298_129871


namespace NUMINAMATH_CALUDE_midpoint_dot_product_sum_of_squares_l1298_129814

/-- Given vectors a and b in ℝ², if m is their midpoint [3, 7] and their dot product is 6,
    then the sum of their squared norms is 220. -/
theorem midpoint_dot_product_sum_of_squares (a b : Fin 2 → ℝ) :
  let m : Fin 2 → ℝ := ![3, 7]
  (∀ i, m i = (a i + b i) / 2) →
  a • b = 6 →
  ‖a‖^2 + ‖b‖^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_dot_product_sum_of_squares_l1298_129814


namespace NUMINAMATH_CALUDE_m_value_proof_l1298_129850

theorem m_value_proof (m : ℤ) (h : m < (Real.sqrt 11 - 1) / 2 ∧ (Real.sqrt 11 - 1) / 2 < m + 1) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_m_value_proof_l1298_129850


namespace NUMINAMATH_CALUDE_soccer_ball_purchase_l1298_129848

theorem soccer_ball_purchase (first_batch_cost second_batch_cost : ℕ) 
  (unit_price_difference : ℕ) :
  first_batch_cost = 800 →
  second_batch_cost = 1560 →
  unit_price_difference = 2 →
  ∃ (first_batch_quantity second_batch_quantity : ℕ) 
    (first_unit_price second_unit_price : ℕ),
    first_batch_quantity * first_unit_price = first_batch_cost ∧
    second_batch_quantity * second_unit_price = second_batch_cost ∧
    second_batch_quantity = 2 * first_batch_quantity ∧
    first_unit_price = second_unit_price + unit_price_difference ∧
    first_batch_quantity + second_batch_quantity = 30 :=
by sorry

end NUMINAMATH_CALUDE_soccer_ball_purchase_l1298_129848


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1298_129869

theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (passing_percentage : ℚ) (obtained_marks : ℕ) (failed_by : ℕ),
    passing_percentage = 40 / 100 →
    obtained_marks = 40 →
    failed_by = 40 →
    passing_percentage * max_marks = obtained_marks + failed_by →
    max_marks = 200 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1298_129869


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_three_even_one_odd_l1298_129879

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_three_even_one_odd (n : ℕ) : Prop :=
  let digits := n.digits 10
  3 = (digits.filter (λ d => d % 2 = 0)).length ∧
  1 = (digits.filter (λ d => d % 2 = 1)).length

theorem smallest_four_digit_divisible_by_9_three_even_one_odd :
  ∀ n : ℕ, is_four_digit n → n % 9 = 0 → has_three_even_one_odd n → 1026 ≤ n := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_three_even_one_odd_l1298_129879


namespace NUMINAMATH_CALUDE_hat_code_is_312_l1298_129878

def code_to_digit (c : Char) : Fin 6 :=
  match c with
  | 'M' => 0
  | 'A' => 1
  | 'T' => 2
  | 'H' => 3
  | 'I' => 4
  | 'S' => 5
  | _ => 0  -- Default case, should not occur in our problem

theorem hat_code_is_312 : 
  (code_to_digit 'H') * 100 + (code_to_digit 'A') * 10 + (code_to_digit 'T') = 312 := by
  sorry

end NUMINAMATH_CALUDE_hat_code_is_312_l1298_129878


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_simplification_l1298_129880

/-- Given distinct real numbers a, b, c, and d, the sum of four rational expressions
    simplifies to a linear polynomial. -/
theorem sum_of_fourth_powers_simplification 
  (a b c d : ℝ) (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) 
  (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  let f : ℝ → ℝ := λ x => 
    ((x + a)^4) / ((a - b)*(a - c)*(a - d)) + 
    ((x + b)^4) / ((b - a)*(b - c)*(b - d)) + 
    ((x + c)^4) / ((c - a)*(c - b)*(c - d)) + 
    ((x + d)^4) / ((d - a)*(d - b)*(d - c))
  ∀ x, f x = a + b + c + d + 4*x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_simplification_l1298_129880


namespace NUMINAMATH_CALUDE_eliza_age_l1298_129819

/-- Given the ages of Aunt Ellen, Dina, and Eliza, prove Eliza's age -/
theorem eliza_age (aunt_ellen_age : ℕ) (dina_age : ℕ) (eliza_age : ℕ) : 
  aunt_ellen_age = 48 →
  dina_age = aunt_ellen_age / 2 →
  eliza_age = dina_age - 6 →
  eliza_age = 18 := by
sorry

end NUMINAMATH_CALUDE_eliza_age_l1298_129819


namespace NUMINAMATH_CALUDE_binomial_minimum_sum_reciprocals_l1298_129899

/-- A discrete random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_minimum_sum_reciprocals (X : BinomialRV) (q : ℝ) 
    (h_expect : expectation X = 4)
    (h_var : variance X = q) :
    (∀ p q, p > 0 → q > 0 → 1/p + 1/q ≥ 9/4) ∧ 
    (∃ p q, p > 0 ∧ q > 0 ∧ 1/p + 1/q = 9/4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_minimum_sum_reciprocals_l1298_129899


namespace NUMINAMATH_CALUDE_line_charts_show_trend_bar_charts_dont_l1298_129841

-- Define the types of charts
inductive Chart
| BarChart
| LineChart

-- Define the capabilities of charts
def can_show_amount (c : Chart) : Prop :=
  match c with
  | Chart.BarChart => true
  | Chart.LineChart => true

def can_reflect_changes (c : Chart) : Prop :=
  match c with
  | Chart.BarChart => false
  | Chart.LineChart => true

-- Define what it means to show a trend
def can_show_trend (c : Chart) : Prop :=
  can_show_amount c ∧ can_reflect_changes c

-- Theorem statement
theorem line_charts_show_trend_bar_charts_dont :
  can_show_trend Chart.LineChart ∧ ¬can_show_trend Chart.BarChart :=
sorry

end NUMINAMATH_CALUDE_line_charts_show_trend_bar_charts_dont_l1298_129841


namespace NUMINAMATH_CALUDE_hen_price_l1298_129805

theorem hen_price (total_cost : ℕ) (pig_price : ℕ) (num_pigs : ℕ) (num_hens : ℕ) :
  total_cost = 1200 →
  pig_price = 300 →
  num_pigs = 3 →
  num_hens = 10 →
  (total_cost - num_pigs * pig_price) / num_hens = 30 :=
by sorry

end NUMINAMATH_CALUDE_hen_price_l1298_129805


namespace NUMINAMATH_CALUDE_simplify_expression_l1298_129897

theorem simplify_expression (x : ℝ) (h : x^2 ≥ 16) :
  (4 - Real.sqrt (x^2 - 16))^2 = x^2 - 8 * Real.sqrt (x^2 - 16) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1298_129897


namespace NUMINAMATH_CALUDE_irene_weekly_income_l1298_129868

/-- Calculates the total weekly income after taxes and deductions for an employee with given conditions --/
def total_weekly_income (base_salary : ℕ) (base_hours : ℕ) (overtime_rate1 : ℕ) (overtime_rate2 : ℕ) (overtime_rate3 : ℕ) (tax_rate : ℚ) (insurance_premium : ℕ) (hours_worked : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the total weekly income is $645 --/
theorem irene_weekly_income :
  let base_salary := 500
  let base_hours := 40
  let overtime_rate1 := 20
  let overtime_rate2 := 30
  let overtime_rate3 := 40
  let tax_rate := 15 / 100
  let insurance_premium := 50
  let hours_worked := 50
  total_weekly_income base_salary base_hours overtime_rate1 overtime_rate2 overtime_rate3 tax_rate insurance_premium hours_worked = 645 :=
by
  sorry

end NUMINAMATH_CALUDE_irene_weekly_income_l1298_129868


namespace NUMINAMATH_CALUDE_average_age_increase_l1298_129881

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 15 →
  student_avg_age = 10 →
  teacher_age = 26 →
  (((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1)) - student_avg_age = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l1298_129881


namespace NUMINAMATH_CALUDE_school_population_l1298_129875

theorem school_population (b g t : ℕ) : 
  b = 4 * g → 
  g = 10 * t → 
  b + g + t = (51 * b) / 40 := by
sorry

end NUMINAMATH_CALUDE_school_population_l1298_129875


namespace NUMINAMATH_CALUDE_solve_for_b_l1298_129810

theorem solve_for_b (a b : ℝ) (eq1 : 2 * a + 1 = 1) (eq2 : b + a = 3) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l1298_129810


namespace NUMINAMATH_CALUDE_polynomial_properties_l1298_129847

def f (x : ℝ) : ℝ := 8*x^7 + 5*x^6 + 3*x^4 + 2*x + 1

theorem polynomial_properties :
  (f 2 = 1397) ∧
  (f (-1) = -1) ∧
  (∃ c : ℝ, c ∈ Set.Icc (-1) 2 ∧ f c = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1298_129847


namespace NUMINAMATH_CALUDE_function_composition_result_l1298_129865

/-- Given a function f(x) = x^2 - 2x, prove that f(f(f(1))) = 3 -/
theorem function_composition_result (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2*x) : f (f (f 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_result_l1298_129865


namespace NUMINAMATH_CALUDE_polynomial_sequence_gcd_l1298_129890

/-- A sequence defined by polynomials with positive integer coefficients -/
def PolynomialSequence (p : ℕ → ℕ → ℕ) (a₀ : ℕ) : ℕ → ℕ :=
  fun n => p n a₀

/-- The theorem statement -/
theorem polynomial_sequence_gcd
  (p : ℕ → ℕ → ℕ)
  (h_p : ∀ n x, p n x > 0)
  (a₀ : ℕ)
  (a : ℕ → ℕ)
  (h_a : a = PolynomialSequence p a₀)
  (m k : ℕ) :
  Nat.gcd (a m) (a k) = a (Nat.gcd m k) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sequence_gcd_l1298_129890


namespace NUMINAMATH_CALUDE_grocer_sales_problem_l1298_129896

theorem grocer_sales_problem (sales1 sales3 sales4 sales5 : ℕ) 
  (h1 : sales1 = 5420)
  (h3 : sales3 = 6200)
  (h4 : sales4 = 6350)
  (h5 : sales5 = 6500)
  (target_average : ℕ) 
  (h_target : target_average = 6000) :
  ∃ sales2 : ℕ, 
    sales2 = 5530 ∧ 
    (sales1 + sales2 + sales3 + sales4 + sales5) / 5 = target_average :=
by
  sorry

end NUMINAMATH_CALUDE_grocer_sales_problem_l1298_129896


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1298_129886

theorem rectangular_box_volume (l w h : ℝ) (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1298_129886


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l1298_129893

/-- A function representing quadratic variation of y with respect to x -/
def quadratic_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem quadratic_symmetry (k : ℝ) :
  quadratic_variation k 5 = 25 →
  quadratic_variation k (-5) = 25 := by
  sorry

#check quadratic_symmetry

end NUMINAMATH_CALUDE_quadratic_symmetry_l1298_129893


namespace NUMINAMATH_CALUDE_largest_integer_with_conditions_l1298_129854

def digits_of (n : ℕ) : List ℕ := sorry

def sum_of_squares (l : List ℕ) : ℕ := sorry

def is_strictly_increasing (l : List ℕ) : Prop := sorry

def product_of_list (l : List ℕ) : ℕ := sorry

theorem largest_integer_with_conditions : 
  let n := 2346
  (sum_of_squares (digits_of n) = 65) ∧ 
  (is_strictly_increasing (digits_of n)) ∧
  (∀ m : ℕ, m > n → 
    (sum_of_squares (digits_of m) ≠ 65) ∨ 
    (¬ is_strictly_increasing (digits_of m))) ∧
  (product_of_list (digits_of n) = 144) := by sorry

end NUMINAMATH_CALUDE_largest_integer_with_conditions_l1298_129854


namespace NUMINAMATH_CALUDE_billy_candy_boxes_l1298_129840

/-- Given that Billy bought boxes of candy with 3 pieces per box and has a total of 21 pieces,
    prove that he bought 7 boxes. -/
theorem billy_candy_boxes : 
  ∀ (boxes : ℕ) (pieces_per_box : ℕ) (total_pieces : ℕ),
    pieces_per_box = 3 →
    total_pieces = 21 →
    boxes * pieces_per_box = total_pieces →
    boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_billy_candy_boxes_l1298_129840


namespace NUMINAMATH_CALUDE_train_crossing_time_l1298_129884

/-- Proves that a train with given length and speed takes the calculated time to cross a post -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 120 → 
  train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1298_129884


namespace NUMINAMATH_CALUDE_pattern1_unique_violation_l1298_129870

/-- Represents a square in a pattern --/
structure Square where
  color : String

/-- Represents a pattern of squares --/
structure Pattern where
  squares : List Square
  arrangement : String

/-- Checks if a pattern can be folded into a cube --/
def can_fold_to_cube (p : Pattern) : Prop :=
  p.squares.length = 6 ∧ p.arrangement ≠ "linear"

/-- Checks if a pattern violates the adjacent color rule --/
def violates_adjacent_color_rule (p : Pattern) : Prop :=
  ∃ (s1 s2 : Square), s1 ∈ p.squares ∧ s2 ∈ p.squares ∧ s1.color = s2.color

/-- The four patterns described in the problem --/
def pattern1 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }, { color := "green" }
    ],
    arrangement := "cross"
  }

def pattern2 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }
    ],
    arrangement := "T"
  }

def pattern3 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }, { color := "green" },
      { color := "red" }
    ],
    arrangement := "custom"
  }

def pattern4 : Pattern :=
  { squares := [
      { color := "blue" }, { color := "green" }, { color := "red" },
      { color := "blue" }, { color := "yellow" }, { color := "green" }
    ],
    arrangement := "linear"
  }

/-- The main theorem --/
theorem pattern1_unique_violation :
  (can_fold_to_cube pattern1 ∧ violates_adjacent_color_rule pattern1) ∧
  (¬can_fold_to_cube pattern2 ∨ ¬violates_adjacent_color_rule pattern2) ∧
  (¬can_fold_to_cube pattern3 ∨ ¬violates_adjacent_color_rule pattern3) ∧
  (¬can_fold_to_cube pattern4 ∨ ¬violates_adjacent_color_rule pattern4) :=
sorry

end NUMINAMATH_CALUDE_pattern1_unique_violation_l1298_129870


namespace NUMINAMATH_CALUDE_book_length_ratio_l1298_129889

/- Define the variables -/
def starting_age : ℕ := 6
def starting_book_length : ℕ := 8
def current_book_length : ℕ := 480

/- Define the book length at twice the starting age -/
def book_length_twice_starting_age : ℕ := starting_book_length * 5

/- Define the book length 8 years after twice the starting age -/
def book_length_8_years_after : ℕ := book_length_twice_starting_age * 3

/- Theorem: The ratio of current book length to the book length 8 years after twice the starting age is 4:1 -/
theorem book_length_ratio :
  current_book_length / book_length_8_years_after = 4 :=
by sorry

end NUMINAMATH_CALUDE_book_length_ratio_l1298_129889


namespace NUMINAMATH_CALUDE_pushup_ratio_l1298_129818

theorem pushup_ratio : 
  ∀ (monday tuesday wednesday thursday friday : ℕ),
    monday = 5 →
    tuesday = 7 →
    wednesday = 2 * tuesday →
    friday = monday + tuesday + wednesday + thursday →
    friday = 39 →
    2 * thursday = monday + tuesday + wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_pushup_ratio_l1298_129818


namespace NUMINAMATH_CALUDE_f_of_two_equals_one_l1298_129802

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2

-- Theorem statement
theorem f_of_two_equals_one : f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_one_l1298_129802


namespace NUMINAMATH_CALUDE_open_box_volume_l1298_129892

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume
  (sheet_length sheet_width cut_size : ℕ)
  (h1 : sheet_length = 40)
  (h2 : sheet_width = 30)
  (h3 : cut_size = 8)
  : (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 2688 := by
  sorry

#check open_box_volume

end NUMINAMATH_CALUDE_open_box_volume_l1298_129892


namespace NUMINAMATH_CALUDE_negation_equivalence_l1298_129833

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ + 2 < 0) ↔ (∀ x : ℝ, x^2 + x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1298_129833


namespace NUMINAMATH_CALUDE_quadratic_root_sum_cubes_equals_sum_l1298_129894

theorem quadratic_root_sum_cubes_equals_sum (k : ℚ) : 
  (∃ a b : ℚ, (4 * a^2 + 5 * a + k = 0) ∧ 
               (4 * b^2 + 5 * b + k = 0) ∧ 
               (a ≠ b) ∧
               (a^3 + b^3 = a + b)) ↔ 
  (k = 9/4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_cubes_equals_sum_l1298_129894


namespace NUMINAMATH_CALUDE_equation_roots_and_sum_l1298_129825

theorem equation_roots_and_sum : ∃ (c d : ℝ),
  (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    (∀ x : ℝ, (x + 3) * (x + c) * (x - 9) = 0 ↔ (x = r₁ ∨ x = r₂))) ∧
  (∃! (s₁ s₂ s₃ : ℝ), s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃ ∧ 
    (∀ x : ℝ, (x - c) * (x - 7) * (x + 5) = 0 ↔ (x = s₁ ∨ x = s₂ ∨ x = s₃))) ∧
  80 * c + 10 * d = 650 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_and_sum_l1298_129825


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1298_129835

theorem inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | Real.sqrt (a^2 - 2*x^2) > x + a} = {x : ℝ | (Real.sqrt 2 / 2) * a ≤ x ∧ x ≤ -(Real.sqrt 2 / 2) * a} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1298_129835


namespace NUMINAMATH_CALUDE_average_non_defective_cookies_l1298_129863

def cookie_counts : List Nat := [9, 11, 13, 16, 17, 18, 21, 22]

theorem average_non_defective_cookies :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 127 / 8 := by
  sorry

end NUMINAMATH_CALUDE_average_non_defective_cookies_l1298_129863


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1298_129874

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, -1, 1; -1, 2, -1; 1, -1, 0]

theorem matrix_equation_solution :
  ∃ (s t u : ℤ), 
    B^3 + s • B^2 + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 ∧ 
    s = -1 ∧ t = 0 ∧ u = 2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1298_129874


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1298_129895

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 
  (x - 2)^2 - (x - 3) * (x + 3) = -4 * x + 13 := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (x^2 + 2*x) / (x^2 - 1) / (x + 1 + (2*x + 1) / (x - 1)) = 1 / (x + 1) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1298_129895


namespace NUMINAMATH_CALUDE_product_sum_difference_equality_l1298_129820

theorem product_sum_difference_equality : 45 * 28 + 45 * 72 - 10 * 45 = 4050 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_difference_equality_l1298_129820


namespace NUMINAMATH_CALUDE_james_pays_40_l1298_129816

/-- The amount James pays for stickers -/
def james_payment (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ) : ℚ :=
  (packs * stickers_per_pack * cost_per_sticker) / 2

/-- Theorem: James pays $40 for the stickers -/
theorem james_pays_40 :
  james_payment 8 40 (1/4) = 40 := by
  sorry

end NUMINAMATH_CALUDE_james_pays_40_l1298_129816


namespace NUMINAMATH_CALUDE_increase_by_percentage_l1298_129882

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 80 ∧ percentage = 50 → final = initial * (1 + percentage / 100) → final = 120 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l1298_129882


namespace NUMINAMATH_CALUDE_ab_value_l1298_129804

theorem ab_value (a b : ℝ) : (a - b - 3) * (a - b + 3) = 40 → (a - b = 7 ∨ a - b = -7) := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1298_129804


namespace NUMINAMATH_CALUDE_triangles_in_circle_l1298_129826

/-- Given n points on a circle's circumference (n ≥ 6), with each pair connected by a chord
    and no three chords intersecting at a common point inside the circle,
    this function calculates the number of different triangles formed by the intersecting chords. -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6

/-- Theorem stating that the number of triangles formed by intersecting chords
    in a circle with n points (n ≥ 6) on its circumference is given by num_triangles n. -/
theorem triangles_in_circle (n : ℕ) (h : n ≥ 6) :
  (num_triangles n) =
    Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_circle_l1298_129826


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1298_129829

/-- Given a curve y = x^2 - 3x, if there exists a point where the tangent line
    has a slope of 1, then the x-coordinate of this point is 2. -/
theorem tangent_point_x_coordinate (x : ℝ) : 
  (∃ y : ℝ, y = x^2 - 3*x ∧ (deriv (fun x => x^2 - 3*x)) x = 1) → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1298_129829


namespace NUMINAMATH_CALUDE_bingley_final_bracelets_l1298_129867

/-- The number of bracelets Bingley has at the end of the exchange process. -/
def final_bracelets : ℕ :=
  let bingley_initial := 5
  let kelly_initial := 16
  let kelly_gives := kelly_initial / 4
  let kelly_sets := kelly_gives / 3
  let bingley_receives := kelly_sets
  let bingley_after_receiving := bingley_initial + bingley_receives
  let bingley_gives_away := bingley_receives / 2
  let bingley_before_sister := bingley_after_receiving - bingley_gives_away
  let sister_gets := bingley_before_sister / 3
  bingley_before_sister - sister_gets

/-- Theorem stating that Bingley ends up with 4 bracelets. -/
theorem bingley_final_bracelets : final_bracelets = 4 := by
  sorry

end NUMINAMATH_CALUDE_bingley_final_bracelets_l1298_129867


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1298_129872

theorem binomial_coefficient_equality (n : ℕ+) :
  (Nat.choose n 2 = Nat.choose (n - 1) 2 + Nat.choose (n - 1) 3) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1298_129872


namespace NUMINAMATH_CALUDE_cost_difference_is_1267_50_l1298_129845

def initial_order : ℝ := 20000

def scheme1_discount1 : ℝ := 0.25
def scheme1_discount2 : ℝ := 0.15
def scheme1_discount3 : ℝ := 0.05

def scheme2_discount1 : ℝ := 0.20
def scheme2_discount2 : ℝ := 0.10
def scheme2_discount3 : ℝ := 0.05
def scheme2_rebate : ℝ := 300

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def scheme1_final_cost : ℝ :=
  apply_discount (apply_discount (apply_discount initial_order scheme1_discount1) scheme1_discount2) scheme1_discount3

def scheme2_final_cost : ℝ :=
  apply_discount (apply_discount (apply_discount initial_order scheme2_discount1) scheme2_discount2) scheme2_discount3 - scheme2_rebate

theorem cost_difference_is_1267_50 :
  scheme1_final_cost - scheme2_final_cost = 1267.50 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_is_1267_50_l1298_129845


namespace NUMINAMATH_CALUDE_village_income_growth_and_prediction_l1298_129852

/-- Represents the annual average growth rate calculation and prediction for a village's per capita income. -/
theorem village_income_growth_and_prediction 
  (initial_income : ℝ) 
  (final_income : ℝ) 
  (years : ℕ) 
  (growth_rate : ℝ) 
  (predicted_income : ℝ)
  (h1 : initial_income = 20000)
  (h2 : final_income = 24200)
  (h3 : years = 2) :
  (final_income = initial_income * (1 + growth_rate) ^ years ∧ 
   growth_rate = 0.1 ∧
   predicted_income = final_income * (1 + growth_rate)) := by
  sorry

#check village_income_growth_and_prediction

end NUMINAMATH_CALUDE_village_income_growth_and_prediction_l1298_129852


namespace NUMINAMATH_CALUDE_new_person_weight_l1298_129836

/-- Given a group of 8 persons, if replacing one person weighing 65 kg
    with a new person increases the average weight by 2.5 kg,
    then the weight of the new person is 85 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 85 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1298_129836


namespace NUMINAMATH_CALUDE_evaluate_expression_l1298_129830

theorem evaluate_expression : 3000^3 - 2998*3000^2 - 2998^2*3000 + 2998^3 = 23992 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1298_129830


namespace NUMINAMATH_CALUDE_exist_three_numbers_with_equal_sum_l1298_129855

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Statement of the theorem
theorem exist_three_numbers_with_equal_sum :
  ∃ (m n p : ℕ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧
    m + sumOfDigits m = n + sumOfDigits n ∧
    n + sumOfDigits n = p + sumOfDigits p :=
sorry

end NUMINAMATH_CALUDE_exist_three_numbers_with_equal_sum_l1298_129855


namespace NUMINAMATH_CALUDE_expression_evaluation_l1298_129883

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1298_129883


namespace NUMINAMATH_CALUDE_equation_solution_l1298_129898

theorem equation_solution : 
  ∃ y : ℝ, (3 / y + (4 / y) / (6 / y) = 1.5) ∧ y = 3.6 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1298_129898


namespace NUMINAMATH_CALUDE_first_apartment_rent_l1298_129851

theorem first_apartment_rent (R : ℝ) : 
  R + 260 + (31 * 20 * 0.58) - (900 + 200 + (21 * 20 * 0.58)) = 76 → R = 800 := by
  sorry

end NUMINAMATH_CALUDE_first_apartment_rent_l1298_129851


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l1298_129877

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 64 ∧ 
  (∀ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^3 ≤ n * (w^6 + x^6 + y^6 + z^6)) ∧
  (∀ (m : ℕ), m < n → ∃ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^3 > m * (w^6 + x^6 + y^6 + z^6)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l1298_129877


namespace NUMINAMATH_CALUDE_initial_group_size_l1298_129823

theorem initial_group_size (initial_avg : ℝ) (new_people : ℕ) (new_avg : ℝ) (final_avg : ℝ) :
  initial_avg = 16 →
  new_people = 20 →
  new_avg = 15 →
  final_avg = 15.5 →
  ∃ x : ℕ, x = 20 ∧
    (x : ℝ) * initial_avg + (new_people : ℝ) * new_avg = (x + new_people : ℝ) * final_avg :=
by
  sorry

end NUMINAMATH_CALUDE_initial_group_size_l1298_129823


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1298_129837

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 3 < 3*x + 1 ∧ x > m + 1) ↔ x > 1) → 
  m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1298_129837


namespace NUMINAMATH_CALUDE_pencils_bought_l1298_129885

theorem pencils_bought (glue_cost pencil_cost total_paid change : ℕ) 
  (h1 : glue_cost = 270)
  (h2 : pencil_cost = 210)
  (h3 : total_paid = 1000)
  (h4 : change = 100) :
  ∃ (num_pencils : ℕ), 
    glue_cost + num_pencils * pencil_cost = total_paid - change ∧ 
    num_pencils = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_bought_l1298_129885


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1298_129861

/-- 
Given a rectangle with perimeter 60 units and one side at least half the length of the other,
the maximum possible area is 200 square units.
-/
theorem rectangle_max_area : 
  ∀ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧                 -- sides are positive
  2 * (a + b) = 60 ∧              -- perimeter is 60
  a ≥ (1/2) * b ∧ b ≥ (1/2) * a → -- one side is at least half the other
  a * b ≤ 200 :=                  -- area is at most 200
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1298_129861


namespace NUMINAMATH_CALUDE_alicia_scored_14_points_per_half_l1298_129827

/-- Alicia's points per half of the game -/
def alicia_points_per_half (total_points : ℕ) (num_players : ℕ) (other_players_average : ℕ) : ℕ :=
  (total_points - (num_players - 1) * other_players_average) / 2

/-- Proof that Alicia scored 14 points in each half of the game -/
theorem alicia_scored_14_points_per_half :
  alicia_points_per_half 63 8 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_alicia_scored_14_points_per_half_l1298_129827


namespace NUMINAMATH_CALUDE_a_to_m_eq_2023_l1298_129809

theorem a_to_m_eq_2023 (a m : ℝ) (h : m = Real.sqrt (a - 2023) - Real.sqrt (2023 - a) + 1) : 
  a ^ m = 2023 := by
sorry

end NUMINAMATH_CALUDE_a_to_m_eq_2023_l1298_129809


namespace NUMINAMATH_CALUDE_sum_leq_fourth_powers_over_product_l1298_129873

theorem sum_leq_fourth_powers_over_product (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_leq_fourth_powers_over_product_l1298_129873


namespace NUMINAMATH_CALUDE_g_monotone_decreasing_l1298_129839

/-- The function g(x) defined in terms of parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- The derivative of g(x) with respect to x -/
def g' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * (1 - a) * x - 3 * a

/-- Theorem stating the condition for g(x) to be monotonically decreasing -/
theorem g_monotone_decreasing (a : ℝ) :
  (∀ x < a / 3, g' a x ≤ 0) ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_g_monotone_decreasing_l1298_129839


namespace NUMINAMATH_CALUDE_triangle_area_in_square_l1298_129866

/-- The area of triangle ABC in a 12x12 square with specific point locations -/
theorem triangle_area_in_square : 
  let square_side : ℝ := 12
  let point_A : ℝ × ℝ := (square_side / 2, square_side)
  let point_B : ℝ × ℝ := (0, square_side / 4)
  let point_C : ℝ × ℝ := (square_side, square_side / 4)
  let triangle_area := (1 / 2) * 
    (|((point_C.1 - point_A.1) * (point_B.2 - point_A.2) - 
       (point_B.1 - point_A.1) * (point_C.2 - point_A.2))|)
  triangle_area = (27 * Real.sqrt 10) / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_in_square_l1298_129866


namespace NUMINAMATH_CALUDE_new_average_age_l1298_129813

theorem new_average_age (initial_people : ℕ) (initial_avg : ℚ) (leaving_age : ℕ) (entering_age : ℕ) :
  initial_people = 7 →
  initial_avg = 28 →
  leaving_age = 22 →
  entering_age = 30 →
  round ((initial_people * initial_avg - leaving_age + entering_age) / initial_people) = 29 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l1298_129813
