import Mathlib

namespace NUMINAMATH_CALUDE_regression_line_at_25_l246_24658

/-- The regression line equation is y = 0.5x - 0.81 -/
def regression_line (x : ℝ) : ℝ := 0.5 * x - 0.81

/-- Theorem: Given the regression line equation y = 0.5x - 0.81, when x = 25, y = 11.69 -/
theorem regression_line_at_25 : regression_line 25 = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_at_25_l246_24658


namespace NUMINAMATH_CALUDE_trig_identity_l246_24629

theorem trig_identity (x : Real) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin ((5*π)/6 - x) + (Real.cos ((π/3) - x))^2 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l246_24629


namespace NUMINAMATH_CALUDE_function_range_properties_l246_24685

open Set

/-- Given a function f with maximum M and minimum m on [a, b], prove the following statements -/
theorem function_range_properties
  (f : ℝ → ℝ) (a b M m : ℝ) (h_max : ∀ x ∈ Icc a b, f x ≤ M) (h_min : ∀ x ∈ Icc a b, m ≤ f x) :
  (∀ p, (∀ x ∈ Icc a b, p ≤ f x) → p ∈ Iic m) ∧
  (∀ p, (∃ x ∈ Icc a b, p = f x) → p ∈ Icc m M) ∧
  (∀ p, (∃ x ∈ Icc a b, p ≤ f x) → p ∈ Iic M) :=
by sorry


end NUMINAMATH_CALUDE_function_range_properties_l246_24685


namespace NUMINAMATH_CALUDE_max_score_2079_l246_24689

def score (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem max_score_2079 :
  ∀ x : ℕ, 2017 ≤ x → x ≤ 2117 → score x ≤ score 2079 :=
by sorry

end NUMINAMATH_CALUDE_max_score_2079_l246_24689


namespace NUMINAMATH_CALUDE_exists_k_undecided_tournament_l246_24634

/-- A tournament is a complete directed graph where each edge represents a match outcome. -/
def Tournament (n : ℕ) := Fin n → Fin n → Bool

/-- A tournament is k-undecided if for every set of k players, there exists a player who has defeated all of them. -/
def IsKUndecided (k : ℕ) (n : ℕ) (t : Tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k →
    ∃ (p : Fin n), p ∉ A ∧ ∀ (a : Fin n), a ∈ A → t p a = true

/-- For every positive integer k, there exists a k-undecided tournament with more than k players. -/
theorem exists_k_undecided_tournament (k : ℕ+) :
  ∃ (n : ℕ) (t : Tournament n), n > k ∧ IsKUndecided k n t :=
sorry

end NUMINAMATH_CALUDE_exists_k_undecided_tournament_l246_24634


namespace NUMINAMATH_CALUDE_exponential_growth_dominance_l246_24645

theorem exponential_growth_dominance (n : ℕ) (h : n ≥ 10) : 2^n ≥ n^3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_growth_dominance_l246_24645


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l246_24661

/-- Probability of selecting an individual in systematic sampling -/
theorem systematic_sampling_probability
  (population_size : ℕ)
  (sample_size : ℕ)
  (h1 : population_size = 1001)
  (h2 : sample_size = 50)
  (h3 : population_size > 0)
  (h4 : sample_size ≤ population_size) :
  (sample_size : ℚ) / population_size = 50 / 1001 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l246_24661


namespace NUMINAMATH_CALUDE_mollys_age_problem_l246_24628

theorem mollys_age_problem (current_age : ℕ) (years_ahead : ℕ) (multiplier : ℕ) : 
  current_age = 12 →
  years_ahead = 18 →
  multiplier = 5 →
  ∃ (years_ago : ℕ), current_age + years_ahead = multiplier * (current_age - years_ago) ∧ years_ago = 6 := by
  sorry

end NUMINAMATH_CALUDE_mollys_age_problem_l246_24628


namespace NUMINAMATH_CALUDE_inequality_multiplication_l246_24609

theorem inequality_multiplication (x y : ℝ) : x < y → 2 * x < 2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l246_24609


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l246_24620

def C : Set Nat := {33, 35, 37, 39, 41}

def has_smallest_prime_factor (n : Nat) (s : Set Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, (Nat.minFac n ≤ Nat.minFac m)

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 33 C ∧ has_smallest_prime_factor 39 C ∧
  ∀ x ∈ C, has_smallest_prime_factor x C → (x = 33 ∨ x = 39) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l246_24620


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l246_24652

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - 2 * Complex.I) : 
  z.im = -3/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l246_24652


namespace NUMINAMATH_CALUDE_vector_parallel_implies_x_value_l246_24643

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is parallel to c, then the x-coordinate of a is -11/3 -/
theorem vector_parallel_implies_x_value 
  (a b c : ℝ × ℝ) 
  (hb : b = (1, 2)) 
  (hc : c = (-1, 3)) 
  (ha : a.2 = 1) 
  (h_parallel : ∃ (k : ℝ), (a + 2 • b) = k • c) : 
  a.1 = -11/3 := by sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_x_value_l246_24643


namespace NUMINAMATH_CALUDE_unique_solution_conditions_l246_24618

/-- The system has a unique solution if and only if a = arctan(4) + πk or a = -arctan(2) + πk, where k is an integer -/
theorem unique_solution_conditions (a : ℝ) : 
  (∃! x y : ℝ, x * Real.cos a + y * Real.sin a = 5 * Real.cos a + 2 * Real.sin a ∧ 
                -3 ≤ x + 2*y ∧ x + 2*y ≤ 7 ∧ 
                -9 ≤ 3*x - 4*y ∧ 3*x - 4*y ≤ 1) ↔ 
  (∃ k : ℤ, a = Real.arctan 4 + k * Real.pi ∨ a = -Real.arctan 2 + k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_conditions_l246_24618


namespace NUMINAMATH_CALUDE_expression_simplification_l246_24642

theorem expression_simplification (x : ℝ) (h : x = 3) : 
  (x - 1 + (2 - 2*x) / (x + 1)) / ((x^2 - x) / (x + 1)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l246_24642


namespace NUMINAMATH_CALUDE_principal_is_8925_l246_24692

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  (simple_interest * 100) / (rate * time)

/-- Theorem stating that given the specific conditions, the principal amount is 8925 -/
theorem principal_is_8925 :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 9
  let time : ℕ := 5
  calculate_principal simple_interest rate time = 8925 := by
sorry

end NUMINAMATH_CALUDE_principal_is_8925_l246_24692


namespace NUMINAMATH_CALUDE_angle_terminal_side_ratio_l246_24672

theorem angle_terminal_side_ratio (a : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos a = 1 ∧ r * Real.sin a = -2) →
  (2 * Real.sin a) / Real.cos a = 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_ratio_l246_24672


namespace NUMINAMATH_CALUDE_jerry_action_figures_l246_24639

/-- Calculates the total number of action figures on Jerry's shelf -/
def total_action_figures (initial_figures : ℕ) (figures_per_set : ℕ) (added_sets : ℕ) : ℕ :=
  initial_figures + figures_per_set * added_sets

/-- Theorem stating that Jerry's shelf has 18 action figures in total -/
theorem jerry_action_figures :
  total_action_figures 8 5 2 = 18 := by
sorry

end NUMINAMATH_CALUDE_jerry_action_figures_l246_24639


namespace NUMINAMATH_CALUDE_initial_coloring_books_count_l246_24648

/-- Proves that the initial number of coloring books in stock is 40 --/
theorem initial_coloring_books_count (books_sold : ℕ) (books_per_shelf : ℕ) (shelves_used : ℕ) : 
  books_sold = 20 → books_per_shelf = 4 → shelves_used = 5 → 
  books_sold + books_per_shelf * shelves_used = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_coloring_books_count_l246_24648


namespace NUMINAMATH_CALUDE_a_4_equals_11_l246_24641

def S (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 3 * n.val

def a (n : ℕ+) : ℤ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_4_equals_11 : a 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_11_l246_24641


namespace NUMINAMATH_CALUDE_toys_storage_time_l246_24653

/-- The time required to put all toys in the box -/
def time_to_store_toys (total_toys : ℕ) (net_gain_per_interval : ℕ) (interval_seconds : ℕ) : ℚ :=
  (total_toys : ℚ) / (net_gain_per_interval : ℚ) * (interval_seconds : ℚ) / 60

/-- Theorem stating that it takes 15 minutes to store all toys -/
theorem toys_storage_time :
  time_to_store_toys 30 1 30 = 15 := by
  sorry

#eval time_to_store_toys 30 1 30

end NUMINAMATH_CALUDE_toys_storage_time_l246_24653


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l246_24659

def a (n : ℕ+) : ℕ := 2 * n.val - 1

theorem sum_of_specific_terms : 
  a 4 + a 5 + a 6 + a 7 + a 8 = 55 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l246_24659


namespace NUMINAMATH_CALUDE_two_digit_multiplication_sum_l246_24675

theorem two_digit_multiplication_sum (a b : ℕ) : 
  a ≥ 10 ∧ a < 100 ∧ b ≥ 10 ∧ b < 100 →
  a * (b + 40) = 2496 →
  a * b = 936 →
  a + b = 63 := by
sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_sum_l246_24675


namespace NUMINAMATH_CALUDE_investor_initial_investment_l246_24687

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investor's initial investment --/
theorem investor_initial_investment :
  let principal : ℝ := 7000
  let rate : ℝ := 0.10
  let time : ℕ := 2
  let final_amount : ℝ := 8470
  compound_interest principal rate time = final_amount := by
  sorry

end NUMINAMATH_CALUDE_investor_initial_investment_l246_24687


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l246_24644

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of a parallel axis ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ :=
  sorry

/-- Theorem stating the distance between foci for the given ellipse -/
theorem ellipse_foci_distance :
  ∀ (e : ParallelAxisEllipse),
    e.x_tangent = (6, 0) →
    e.y_tangent = (0, 3) →
    foci_distance e = 6 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l246_24644


namespace NUMINAMATH_CALUDE_subtracted_value_proof_l246_24612

theorem subtracted_value_proof (n : ℕ) (x : ℕ) : 
  n = 36 → 
  ((n + 10) * 2) / 2 - x = 88 / 2 ↔ 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_subtracted_value_proof_l246_24612


namespace NUMINAMATH_CALUDE_line_mb_product_l246_24635

/-- Given a line passing through points (0, -1) and (2, -6) with equation y = mx + b,
    prove that the product mb equals 5/2. -/
theorem line_mb_product (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + b) →
  (-1 : ℚ) = b →
  (-6 : ℚ) = m * 2 + b →
  m * b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_mb_product_l246_24635


namespace NUMINAMATH_CALUDE_curve_is_circle_l246_24614

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define what a circle is
def is_circle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    S = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem curve_is_circle :
  is_circle {p : ℝ × ℝ | curve p.1 p.2} :=
sorry

end NUMINAMATH_CALUDE_curve_is_circle_l246_24614


namespace NUMINAMATH_CALUDE_remainder_problem_l246_24610

theorem remainder_problem (N : ℤ) : N % 899 = 63 → N % 29 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l246_24610


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l246_24655

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat

/-- Represents the random variable ξ (number of yellow balls drawn) -/
def Xi := Nat

/-- The initial state of the box -/
def initialBox : BallCounts := { red := 1, green := 1, yellow := 2 }

/-- The probability of drawing no yellow balls before drawing the red ball -/
def probXiZero (box : BallCounts) : Real :=
  sorry

/-- The expected value of ξ -/
def expectedXi (box : BallCounts) : Real :=
  sorry

/-- The main theorem stating the probability and expectation results -/
theorem ball_drawing_theorem (box : BallCounts) 
  (h : box = initialBox) : 
  probXiZero box = 1/3 ∧ expectedXi box = 1 := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l246_24655


namespace NUMINAMATH_CALUDE_profit_increase_1995_to_1997_l246_24674

/-- Represents the financial data of a company over three years -/
structure CompanyFinances where
  R1 : ℝ  -- Revenue in 1995
  E1 : ℝ  -- Expenses in 1995
  P1 : ℝ  -- Profit in 1995
  R2 : ℝ  -- Revenue in 1996
  E2 : ℝ  -- Expenses in 1996
  P2 : ℝ  -- Profit in 1996
  R3 : ℝ  -- Revenue in 1997
  E3 : ℝ  -- Expenses in 1997
  P3 : ℝ  -- Profit in 1997

/-- The profit increase from 1995 to 1997 is 55.25% -/
theorem profit_increase_1995_to_1997 (cf : CompanyFinances)
  (h1 : cf.P1 = cf.R1 - cf.E1)
  (h2 : cf.R2 = 1.20 * cf.R1)
  (h3 : cf.E2 = 1.10 * cf.E1)
  (h4 : cf.P2 = 1.15 * cf.P1)
  (h5 : cf.R3 = 1.25 * cf.R2)
  (h6 : cf.E3 = 1.20 * cf.E2)
  (h7 : cf.P3 = 1.35 * cf.P2) :
  cf.P3 = 1.5525 * cf.P1 := by
  sorry

#check profit_increase_1995_to_1997

end NUMINAMATH_CALUDE_profit_increase_1995_to_1997_l246_24674


namespace NUMINAMATH_CALUDE_min_value_theorem_l246_24698

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100) ≥ -2500 ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + 1/x₀ = 50 ∧
    (x₀ + 1/x₀) * (x₀ + 1/x₀ - 100) + (x₀ + 1/x₀) * (x₀ + 1/x₀ - 100) = -2500) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l246_24698


namespace NUMINAMATH_CALUDE_book_donation_equation_l246_24677

/-- Proves that the equation for book donations over three years is correct -/
theorem book_donation_equation (x : ℝ) : 
  (400 : ℝ) + 400 * (1 + x) + 400 * (1 + x)^2 = 1525 → 
  (∃ (y : ℝ), y > 0 ∧ 400 * (1 + y) + 400 * (1 + y)^2 = 1125) :=
by
  sorry


end NUMINAMATH_CALUDE_book_donation_equation_l246_24677


namespace NUMINAMATH_CALUDE_fraction_modification_l246_24601

theorem fraction_modification (d : ℚ) : 
  (3 : ℚ) / d ≠ (1 : ℚ) / (3 : ℚ) →
  ((3 : ℚ) + 3) / (d + 3) = (1 : ℚ) / (3 : ℚ) →
  d = 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_modification_l246_24601


namespace NUMINAMATH_CALUDE_solution_set_f_neg_x_l246_24678

/-- Given a function f(x) = (ax-1)(x-b) where the solution set of f(x) > 0 is (-1,3),
    prove that the solution set of f(-x) < 0 is (-∞,-3)∪(1,+∞) -/
theorem solution_set_f_neg_x (a b : ℝ) : 
  (∀ x, (a * x - 1) * (x - b) > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x, (a * (-x) - 1) * (-x - b) < 0 ↔ x < -3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_f_neg_x_l246_24678


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_nine_xy_squared_l246_24663

theorem factorization_cubic_minus_nine_xy_squared (x y : ℝ) :
  x^3 - 9*x*y^2 = x*(x+3*y)*(x-3*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_nine_xy_squared_l246_24663


namespace NUMINAMATH_CALUDE_rectangle_perimeter_120_l246_24662

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculate the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.length

/-- Calculate the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

/-- Theorem: A rectangle with area 864 and width 12 less than length has perimeter 120 -/
theorem rectangle_perimeter_120 (r : Rectangle) 
  (h_area : r.area = 864)
  (h_width : r.width + 12 = r.length) :
  r.perimeter = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_120_l246_24662


namespace NUMINAMATH_CALUDE_real_roots_condition_l246_24656

theorem real_roots_condition (p q : ℝ) : 
  (∃ x : ℝ, x^4 + p*x^2 + q = 0) → 65*p^2 ≥ 4*q ∧ 
  ¬(∀ p q : ℝ, 65*p^2 ≥ 4*q → ∃ x : ℝ, x^4 + p*x^2 + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_condition_l246_24656


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l246_24669

theorem gcd_polynomial_and_multiple (a : ℤ) : 
  (∃ k : ℤ, a = 270 * k) → Int.gcd (5*a^3 + 3*a^2 + 5*a + 45) a = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l246_24669


namespace NUMINAMATH_CALUDE_point_on_line_l246_24637

/-- Given a line passing through points (0,4) and (-6,1), prove that s = 6 
    is the unique solution such that (s,7) lies on this line. -/
theorem point_on_line (s : ℝ) : 
  (∃! x : ℝ, (x - 0) / (-6 - 0) = (7 - 4) / (x - 0) ∧ x = s) → s = 6 :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l246_24637


namespace NUMINAMATH_CALUDE_mans_usual_time_l246_24682

/-- 
Given a man whose walking time increases by 24 minutes when his speed is reduced to 50% of his usual speed,
prove that his usual time to cover the distance is 24 minutes.
-/
theorem mans_usual_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0)
  (h3 : usual_speed / (0.5 * usual_speed) = (usual_time + 24) / usual_time) : 
  usual_time = 24 := by
sorry

end NUMINAMATH_CALUDE_mans_usual_time_l246_24682


namespace NUMINAMATH_CALUDE_pure_imaginary_square_l246_24693

theorem pure_imaginary_square (a : ℝ) (z : ℂ) : 
  z = a + (1 + a) * Complex.I → 
  (∃ b : ℝ, z = b * Complex.I) → 
  z^2 = -1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_l246_24693


namespace NUMINAMATH_CALUDE_fifth_term_is_1280_l246_24638

/-- A geometric sequence of positive integers with first term 5 and fourth term 320 -/
def geometric_sequence (n : ℕ) : ℕ :=
  5 * (320 / 5) ^ ((n - 1) / 3)

/-- The fifth term of the geometric sequence is 1280 -/
theorem fifth_term_is_1280 : geometric_sequence 5 = 1280 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_1280_l246_24638


namespace NUMINAMATH_CALUDE_hyperbola_sum_l246_24664

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 →
  k = 1 →
  c = Real.sqrt 50 →
  a = 4 →
  b^2 = c^2 - a^2 →
  h + k + a + b = 2 + Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l246_24664


namespace NUMINAMATH_CALUDE_no_polyhedron_with_area_2015_l246_24651

/-- Represents a polyhedron constructed from unit cubes -/
structure Polyhedron where
  num_cubes : ℕ
  num_glued_faces : ℕ

/-- Calculates the surface area of a polyhedron -/
def surface_area (p : Polyhedron) : ℕ :=
  6 * p.num_cubes - 2 * p.num_glued_faces

/-- Theorem stating the impossibility of constructing a polyhedron with surface area 2015 -/
theorem no_polyhedron_with_area_2015 :
  ∀ p : Polyhedron, surface_area p ≠ 2015 := by
  sorry


end NUMINAMATH_CALUDE_no_polyhedron_with_area_2015_l246_24651


namespace NUMINAMATH_CALUDE_book_purchase_remaining_money_l246_24650

theorem book_purchase_remaining_money (m : ℚ) (n : ℕ) (b : ℚ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : b > 0) 
  (h4 : (1/4) * m = (1/2) * n * b) : 
  m - n * b = (1/2) * m := by
sorry

end NUMINAMATH_CALUDE_book_purchase_remaining_money_l246_24650


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_2_unique_a_for_integer_solution_set_l246_24602

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2| + |x - a|

-- Theorem for part 1
theorem solution_set_when_a_eq_2 :
  ∀ x : ℝ, f x 2 ≥ 4 ↔ x ≤ 0 ∨ x ≥ 4 := by sorry

-- Theorem for part 2
theorem unique_a_for_integer_solution_set :
  (∃! a : ℝ, ∀ x : ℤ, f (x : ℝ) a < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) ∧
  (∀ a : ℝ, (∀ x : ℤ, f (x : ℝ) a < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → a = 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_2_unique_a_for_integer_solution_set_l246_24602


namespace NUMINAMATH_CALUDE_one_sixth_of_twelve_x_plus_six_l246_24636

theorem one_sixth_of_twelve_x_plus_six (x : ℝ) : (1 / 6) * (12 * x + 6) = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_one_sixth_of_twelve_x_plus_six_l246_24636


namespace NUMINAMATH_CALUDE_lucas_initial_beds_l246_24673

/-- The number of pet beds Lucas can add to his room -/
def additional_beds : ℕ := 8

/-- The number of beds required per pet -/
def beds_per_pet : ℕ := 2

/-- The total number of pets Lucas's room can accommodate -/
def total_pets : ℕ := 10

/-- The initial number of pet beds in Lucas's room -/
def initial_beds : ℕ := total_pets * beds_per_pet - additional_beds

theorem lucas_initial_beds :
  initial_beds = 12 :=
by sorry

end NUMINAMATH_CALUDE_lucas_initial_beds_l246_24673


namespace NUMINAMATH_CALUDE_complex_number_simplification_l246_24633

theorem complex_number_simplification :
  (2 - 5 * Complex.I) - (-3 + 7 * Complex.I) - 4 * (-1 + 2 * Complex.I) = 1 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l246_24633


namespace NUMINAMATH_CALUDE_probability_six_consecutive_heads_l246_24694

def coin_flips : ℕ := 8

def favorable_outcomes : ℕ := 17

def total_outcomes : ℕ := 2^coin_flips

theorem probability_six_consecutive_heads :
  (favorable_outcomes : ℚ) / total_outcomes = 17 / 256 :=
sorry

end NUMINAMATH_CALUDE_probability_six_consecutive_heads_l246_24694


namespace NUMINAMATH_CALUDE_absent_students_percentage_l246_24695

theorem absent_students_percentage
  (total_students : ℕ)
  (num_boys : ℕ)
  (num_girls : ℕ)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h1 : total_students = 120)
  (h2 : num_boys = 72)
  (h3 : num_girls = 48)
  (h4 : boys_absent_fraction = 1 / 8)
  (h5 : girls_absent_fraction = 1 / 4)
  (h6 : total_students = num_boys + num_girls) :
  (boys_absent_fraction * num_boys + girls_absent_fraction * num_girls) / total_students = 7 / 40 := by
  sorry

#eval (7 : ℚ) / 40 * 100 -- To show that 7/40 is equivalent to 17.5%

end NUMINAMATH_CALUDE_absent_students_percentage_l246_24695


namespace NUMINAMATH_CALUDE_fraction_simplification_l246_24649

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  a^2 / (a * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l246_24649


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l246_24690

theorem fourth_root_equation_solution :
  {x : ℝ | (59 - 2*x)^(1/4) + (23 + 2*x)^(1/4) = 4} = {-8, 29} := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l246_24690


namespace NUMINAMATH_CALUDE_sams_income_l246_24697

/-- Represents the income tax calculation for Sam's region -/
noncomputable def income_tax (q : ℝ) (income : ℝ) : ℝ :=
  0.01 * q * 30000 +
  0.01 * (q + 3) * (min 45000 (max 30000 income) - 30000) +
  0.01 * (q + 5) * (max 0 (income - 45000))

/-- Theorem stating Sam's annual income given the tax structure -/
theorem sams_income (q : ℝ) :
  ∃ (income : ℝ),
    income_tax q income = 0.01 * (q + 0.35) * income ∧
    income = 48376 :=
by sorry

end NUMINAMATH_CALUDE_sams_income_l246_24697


namespace NUMINAMATH_CALUDE_race_solution_l246_24683

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  half_lap_time : ℝ

/-- Represents the race configuration -/
structure RaceConfig where
  track_length : ℝ
  α : Runner
  β : Runner
  initial_distance : ℝ
  symmetry_time : ℝ
  β_to_Q_time : ℝ
  α_to_finish_time : ℝ

/-- The theorem statement -/
theorem race_solution (config : RaceConfig) 
  (h1 : config.initial_distance = 16)
  (h2 : config.β_to_Q_time = 1 + 2/15)
  (h3 : config.α_to_finish_time = 13 + 13/15)
  (h4 : config.α.speed = config.track_length / (2 * config.α.half_lap_time))
  (h5 : config.β.speed = config.track_length / (2 * config.β.half_lap_time))
  (h6 : config.α.half_lap_time + config.symmetry_time + config.β_to_Q_time + config.α_to_finish_time = 2 * config.α.half_lap_time)
  (h7 : config.β.half_lap_time = config.α.half_lap_time + config.symmetry_time + config.β_to_Q_time)
  (h8 : config.track_length / 2 = config.α.speed * config.α.half_lap_time)
  (h9 : config.track_length / 2 = config.β.speed * config.β.half_lap_time)
  (h10 : config.α.speed * (config.β_to_Q_time + config.α_to_finish_time) = config.track_length / 2) :
  config.α.speed = 8.5 ∧ config.β.speed = 7.5 ∧ config.track_length = 272 := by
  sorry


end NUMINAMATH_CALUDE_race_solution_l246_24683


namespace NUMINAMATH_CALUDE_min_a_for_monotonic_odd_function_l246_24684

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x > 0 then Real.exp x + a
  else if x < 0 then -(Real.exp (-x) + a)
  else 0

-- State the theorem
theorem min_a_for_monotonic_odd_function :
  ∀ a : ℝ, 
  (∀ x : ℝ, f a x = -f a (-x)) → -- f is odd
  (∀ x y : ℝ, x < y → f a x ≤ f a y) → -- f is monotonic
  a ≥ -1 ∧ 
  ∀ b : ℝ, (∀ x : ℝ, f b x = -f b (-x)) → 
            (∀ x y : ℝ, x < y → f b x ≤ f b y) → 
            b ≥ a :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_monotonic_odd_function_l246_24684


namespace NUMINAMATH_CALUDE_train_length_l246_24611

/-- Given a train that crosses a bridge and passes a lamp post, calculate its length. -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ)
  (h1 : bridge_length = 2500)
  (h2 : bridge_time = 120)
  (h3 : post_time = 30) :
  bridge_length * (post_time / bridge_time) / (1 - post_time / bridge_time) = 2500 * (1/4) / (1 - 1/4) :=
by sorry

end NUMINAMATH_CALUDE_train_length_l246_24611


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l246_24660

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l246_24660


namespace NUMINAMATH_CALUDE_trigonometric_sum_zero_l246_24671

theorem trigonometric_sum_zero (α : ℝ) : 
  Real.sin (2 * α - 3/2 * Real.pi) + Real.cos (2 * α - 8/3 * Real.pi) + Real.cos (2/3 * Real.pi + 2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_zero_l246_24671


namespace NUMINAMATH_CALUDE_rational_equation_solution_l246_24646

theorem rational_equation_solution (C D : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 →
    (D * x - 17) / (x^2 - 9*x + 20) = C / (x - 4) + 5 / (x - 5)) →
  C + D = 19/5 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l246_24646


namespace NUMINAMATH_CALUDE_compute_expression_l246_24632

theorem compute_expression : 3 * 3^4 - 4^55 / 4^54 = 239 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l246_24632


namespace NUMINAMATH_CALUDE_coefficient_x3y4_expansion_l246_24616

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

-- Define the expansion term
def expansionTerm (n k : ℕ) (x y : ℚ) : ℚ :=
  binomial n k * (x ^ k) * (y ^ (n - k))

-- Theorem statement
theorem coefficient_x3y4_expansion :
  let n : ℕ := 9
  let k : ℕ := 3
  let x : ℚ := 2/3
  let y : ℚ := -3/4
  expansionTerm n k x y = 441/992 := by
sorry

end NUMINAMATH_CALUDE_coefficient_x3y4_expansion_l246_24616


namespace NUMINAMATH_CALUDE_lunch_special_cost_l246_24600

theorem lunch_special_cost (total_bill : ℚ) (num_people : ℕ) (h1 : total_bill = 24) (h2 : num_people = 3) :
  total_bill / num_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_lunch_special_cost_l246_24600


namespace NUMINAMATH_CALUDE_adam_has_more_apples_l246_24647

/-- The number of apples Adam has -/
def adam_apples : ℕ := 10

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 2

/-- The difference in apples between Adam and Jackie -/
def apple_difference : ℕ := adam_apples - jackie_apples

theorem adam_has_more_apples : apple_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_adam_has_more_apples_l246_24647


namespace NUMINAMATH_CALUDE_unique_row_with_37_l246_24631

/-- Pascal's Triangle entry at row n and column k -/
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- A row of Pascal's Triangle contains 37 -/
def row_contains_37 (n : ℕ) : Prop :=
  ∃ k, 0 ≤ k ∧ k ≤ n ∧ pascal n k = 37

/-- There is exactly one row in Pascal's Triangle that contains 37 -/
theorem unique_row_with_37 : ∃! n, row_contains_37 n :=
  sorry

end NUMINAMATH_CALUDE_unique_row_with_37_l246_24631


namespace NUMINAMATH_CALUDE_sequence_general_term_l246_24667

/-- The sequence defined by a₁ = -1 and aₙ₊₁ = 3aₙ - 1 has the general term aₙ = -(3ⁿ - 1)/2 -/
theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = -1) 
    (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n - 1) : 
    ∀ n : ℕ, n ≥ 1 → a n = -(3^n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l246_24667


namespace NUMINAMATH_CALUDE_mean_home_runs_l246_24670

def home_runs : List ℕ := [5, 6, 7, 8, 9]
def players : List ℕ := [4, 5, 3, 2, 2]

theorem mean_home_runs :
  let total_hrs := (List.zip home_runs players).map (fun (hr, p) => hr * p) |>.sum
  let total_players := players.sum
  (total_hrs : ℚ) / total_players = 105 / 16 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l246_24670


namespace NUMINAMATH_CALUDE_probability_two_heads_and_three_l246_24623

def coin_outcomes : ℕ := 2
def die_outcomes : ℕ := 6

def total_outcomes : ℕ := coin_outcomes * coin_outcomes * die_outcomes

def favorable_outcome : ℕ := 1

theorem probability_two_heads_and_three : 
  (favorable_outcome : ℚ) / total_outcomes = 1 / 24 := by sorry

end NUMINAMATH_CALUDE_probability_two_heads_and_three_l246_24623


namespace NUMINAMATH_CALUDE_investment_dividend_l246_24626

/-- Calculates the total dividend received from an investment in shares -/
theorem investment_dividend (investment : ℝ) (share_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) :
  investment = 14400 →
  share_value = 100 →
  premium_rate = 0.20 →
  dividend_rate = 0.06 →
  let share_cost := share_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := share_value * dividend_rate
  dividend_per_share * num_shares = 720 := by
  sorry

end NUMINAMATH_CALUDE_investment_dividend_l246_24626


namespace NUMINAMATH_CALUDE_yard_raking_time_l246_24621

theorem yard_raking_time (your_time brother_time together_time : ℝ) 
  (h1 : brother_time = 45)
  (h2 : together_time = 18)
  (h3 : 1 / your_time + 1 / brother_time = 1 / together_time) :
  your_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_yard_raking_time_l246_24621


namespace NUMINAMATH_CALUDE_rose_flyers_count_l246_24617

def total_flyers : ℕ := 1236
def jack_flyers : ℕ := 120
def left_flyers : ℕ := 796

theorem rose_flyers_count : total_flyers - jack_flyers - left_flyers = 320 := by
  sorry

end NUMINAMATH_CALUDE_rose_flyers_count_l246_24617


namespace NUMINAMATH_CALUDE_theodore_stone_statues_l246_24606

/-- The number of stone statues Theodore crafts every month -/
def stone_statues : ℕ := sorry

/-- The number of wooden statues Theodore crafts every month -/
def wooden_statues : ℕ := 20

/-- The cost of a stone statue in dollars -/
def stone_cost : ℕ := 20

/-- The cost of a wooden statue in dollars -/
def wooden_cost : ℕ := 5

/-- The tax rate as a decimal -/
def tax_rate : ℚ := 1/10

/-- Theodore's total monthly earnings after tax in dollars -/
def total_earnings : ℕ := 270

theorem theodore_stone_statues :
  stone_statues = 10 ∧
  (stone_statues * stone_cost + wooden_statues * wooden_cost) * (1 - tax_rate) = total_earnings :=
sorry

end NUMINAMATH_CALUDE_theodore_stone_statues_l246_24606


namespace NUMINAMATH_CALUDE_paint_time_together_l246_24640

-- Define the rates of work for Harish and Ganpat
def harish_rate : ℚ := 1 / 3
def ganpat_rate : ℚ := 1 / 6

-- Define the total rate when working together
def total_rate : ℚ := harish_rate + ganpat_rate

-- Theorem to prove
theorem paint_time_together : (1 : ℚ) / total_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_paint_time_together_l246_24640


namespace NUMINAMATH_CALUDE_min_value_expression_l246_24603

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c = Real.sqrt 6) :
  ∃ (min_val : ℝ), min_val = 8 * Real.sqrt 2 - 4 ∧
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → y + z = Real.sqrt 6 →
    (x * z^2 + 2 * x) / (y * z) + 16 / (x + 2) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l246_24603


namespace NUMINAMATH_CALUDE_hcl_required_l246_24613

-- Define the chemical reaction
structure Reaction where
  hcl : ℕ  -- moles of Hydrochloric acid
  koh : ℕ  -- moles of Potassium hydroxide
  h2o : ℕ  -- moles of Water
  kcl : ℕ  -- moles of Potassium chloride

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.hcl = r.koh ∧ r.hcl = r.h2o ∧ r.hcl = r.kcl

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.koh = 2 ∧ r.h2o = 2 ∧ r.kcl = 2

-- Theorem to prove
theorem hcl_required (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : given_conditions r) : 
  r.hcl = 2 := by
  sorry

#check hcl_required

end NUMINAMATH_CALUDE_hcl_required_l246_24613


namespace NUMINAMATH_CALUDE_divisibility_implies_unit_l246_24654

theorem divisibility_implies_unit (a b c d : ℤ) 
  (h1 : (ab - cd) ∣ a) 
  (h2 : (ab - cd) ∣ b) 
  (h3 : (ab - cd) ∣ c) 
  (h4 : (ab - cd) ∣ d) : 
  ab - cd = 1 ∨ ab - cd = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_unit_l246_24654


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l246_24608

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l246_24608


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l246_24696

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence satisfying
    the given condition, 2a_7 - a_8 equals 24. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_condition : a 3 + 3 * a 6 + a 9 = 120) : 
  2 * a 7 - a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l246_24696


namespace NUMINAMATH_CALUDE_only_negative_three_l246_24605

theorem only_negative_three (a b c d : ℝ) : 
  a = |-3| ∧ b = -3 ∧ c = -(-3) ∧ d = 1/3 → 
  (b < 0 ∧ a ≥ 0 ∧ c ≥ 0 ∧ d > 0) := by sorry

end NUMINAMATH_CALUDE_only_negative_three_l246_24605


namespace NUMINAMATH_CALUDE_football_player_goals_l246_24688

/-- Proves that a football player scored 2 goals in their fifth match -/
theorem football_player_goals (total_matches : ℕ) (total_goals : ℕ) (average_increase : ℚ) : 
  total_matches = 5 → 
  total_goals = 4 → 
  average_increase = 3/10 → 
  (total_goals : ℚ) / total_matches = 
    ((total_goals : ℚ) - (total_goals - goals_in_fifth_match)) / (total_matches - 1) + average_increase →
  goals_in_fifth_match = 2 :=
by
  sorry

#check football_player_goals

end NUMINAMATH_CALUDE_football_player_goals_l246_24688


namespace NUMINAMATH_CALUDE_chess_tournament_players_l246_24699

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of players not in the lowest 15
  -- Each player plays exactly one match against every other player
  total_games : ℕ := (n + 15).choose 2
  -- Points from games between n players not in the lowest 15
  points_among_n : ℕ := n.choose 2
  -- Points earned by n players against the lowest 15
  points_n_vs_15 : ℕ := n.choose 2
  -- Points earned by the lowest 15 players among themselves
  points_among_15 : ℕ := 105
  -- Total points in the tournament
  total_points : ℕ := 2 * points_among_n + 2 * points_among_15

/-- The theorem stating that the total number of players in the tournament is 50 -/
theorem chess_tournament_players (t : ChessTournament) : t.n + 15 = 50 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l246_24699


namespace NUMINAMATH_CALUDE_students_taking_neither_math_nor_chemistry_l246_24615

theorem students_taking_neither_math_nor_chemistry :
  let total_students : ℕ := 150
  let math_students : ℕ := 80
  let chemistry_students : ℕ := 60
  let both_subjects : ℕ := 15
  let neither_subject : ℕ := total_students - (math_students + chemistry_students - both_subjects)
  neither_subject = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_neither_math_nor_chemistry_l246_24615


namespace NUMINAMATH_CALUDE_odd_power_minus_self_div_24_l246_24622

theorem odd_power_minus_self_div_24 (n : ℕ) (h : Odd n) (h' : n > 0) :
  ∃ k : ℤ, (n^n : ℤ) - n = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_odd_power_minus_self_div_24_l246_24622


namespace NUMINAMATH_CALUDE_carpet_cost_is_576_l246_24666

/-- The total cost of carpet squares needed to cover a rectangular floor -/
def total_carpet_cost (floor_length floor_width carpet_side_length carpet_cost : ℕ) : ℕ :=
  let floor_area := floor_length * floor_width
  let carpet_area := carpet_side_length * carpet_side_length
  let num_carpets := floor_area / carpet_area
  num_carpets * carpet_cost

/-- Proof that the total cost of carpet squares for the given floor is $576 -/
theorem carpet_cost_is_576 :
  total_carpet_cost 24 64 8 24 = 576 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_is_576_l246_24666


namespace NUMINAMATH_CALUDE_all_ap_lines_pass_through_point_l246_24627

/-- A line in the form ax + by = c where a, b, and c form an arithmetic progression -/
structure APLine where
  a : ℝ
  d : ℝ
  eq : ℝ × ℝ → Prop := fun (x, y) ↦ a * x + (a + d) * y = a + 2 * d

/-- The theorem stating that all APLines pass through the point (-1, 2) -/
theorem all_ap_lines_pass_through_point :
  ∀ (l : APLine), l.eq (-1, 2) :=
sorry

end NUMINAMATH_CALUDE_all_ap_lines_pass_through_point_l246_24627


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l246_24604

theorem fixed_point_on_line (a b : ℝ) (h : a + b = 1) :
  2 * a * (1/2) - b * (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l246_24604


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_l246_24668

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define the parabola
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the right focus of the ellipse
def right_focus_ellipse (x y : ℝ) : Prop := x = 2 ∧ y = 0

-- Define the focus of the parabola
def focus_parabola (p x y : ℝ) : Prop := x = p / 2 ∧ y = 0

-- Theorem statement
theorem parabola_ellipse_focus (p : ℝ) :
  (∃ x y : ℝ, right_focus_ellipse x y ∧ focus_parabola p x y) → p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_l246_24668


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l246_24607

-- Define the function f(x) = -x^3
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l246_24607


namespace NUMINAMATH_CALUDE_project_completion_time_l246_24686

/-- Proves that the total time to complete a project is 15 days given the specified conditions. -/
theorem project_completion_time 
  (a_rate : ℝ) 
  (b_rate : ℝ) 
  (a_quit_before : ℝ) 
  (h1 : a_rate = 1 / 20) 
  (h2 : b_rate = 1 / 30) 
  (h3 : a_quit_before = 5) : 
  ∃ (total_time : ℝ), total_time = 15 ∧ 
    (total_time - a_quit_before) * (a_rate + b_rate) + 
    a_quit_before * b_rate = 1 :=
sorry

end NUMINAMATH_CALUDE_project_completion_time_l246_24686


namespace NUMINAMATH_CALUDE_pen_price_calculation_l246_24630

theorem pen_price_calculation (num_pens num_pencils total_cost pencil_avg_price : ℝ) :
  num_pens = 30 →
  num_pencils = 75 →
  total_cost = 630 →
  pencil_avg_price = 2 →
  (total_cost - num_pencils * pencil_avg_price) / num_pens = 16 :=
by sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l246_24630


namespace NUMINAMATH_CALUDE_polynomial_simplification_l246_24657

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 3 * x^2 + 6 * x - 8) = x^3 + x^2 + 3 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l246_24657


namespace NUMINAMATH_CALUDE_belinda_pages_per_day_l246_24681

/-- Given that Janet reads 80 pages a day and 2100 more pages than Belinda in 6 weeks,
    prove that Belinda reads 30 pages a day. -/
theorem belinda_pages_per_day :
  let janet_pages_per_day : ℕ := 80
  let weeks : ℕ := 6
  let days_in_week : ℕ := 7
  let extra_pages : ℕ := 2100
  let belinda_pages_per_day : ℕ := 30
  janet_pages_per_day * (weeks * days_in_week) = 
    belinda_pages_per_day * (weeks * days_in_week) + extra_pages :=
by
  sorry

#check belinda_pages_per_day

end NUMINAMATH_CALUDE_belinda_pages_per_day_l246_24681


namespace NUMINAMATH_CALUDE_basketball_team_starters_l246_24665

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players -/
def total_players : ℕ := 18

/-- The number of quadruplets -/
def quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def starters : ℕ := 7

/-- The number of non-quadruplet players -/
def non_quadruplets : ℕ := total_players - quadruplets

theorem basketball_team_starters :
  choose total_players starters - choose non_quadruplets (starters - quadruplets) = 31460 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_starters_l246_24665


namespace NUMINAMATH_CALUDE_substance_mass_l246_24691

/-- Given a substance where 1 gram occupies 5 cubic centimeters, 
    the mass of 1 cubic meter of this substance is 200 kilograms. -/
theorem substance_mass (substance_density : ℝ) : 
  substance_density = 1 / 5 → -- 1 gram occupies 5 cubic centimeters
  (1 : ℝ) * substance_density * 1000000 / 1000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_substance_mass_l246_24691


namespace NUMINAMATH_CALUDE_ellipse_intersection_midpoints_line_slope_l246_24619

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of the parallel lines -/
def parallel_line (x y m : ℝ) : Prop := y = (1/4) * x + m

/-- Definition of a point being the midpoint of two other points -/
def is_midpoint (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

/-- The main theorem -/
theorem ellipse_intersection_midpoints_line_slope :
  ∀ (l : ℝ → ℝ),
  (∀ x y m x1 y1 x2 y2 : ℝ,
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    parallel_line x1 y1 m ∧ parallel_line x2 y2 m ∧
    is_midpoint x y x1 y1 x2 y2 →
    y = l x) →
  ∃ k, ∀ x, l x = -2 * x + k :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_midpoints_line_slope_l246_24619


namespace NUMINAMATH_CALUDE_scissors_count_l246_24625

theorem scissors_count (initial : Nat) (added : Nat) (total : Nat) : 
  initial = 54 → added = 22 → total = initial + added → total = 76 := by
  sorry

end NUMINAMATH_CALUDE_scissors_count_l246_24625


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l246_24676

/-- Represents the number of papers drawn from a school --/
structure SchoolSample where
  total : ℕ
  drawn : ℕ

/-- Represents the sampling data for all schools --/
structure SamplingData where
  schoolA : SchoolSample
  schoolB : SchoolSample
  schoolC : SchoolSample

/-- Calculates the total number of papers drawn using stratified sampling --/
def totalDrawn (data : SamplingData) : ℕ :=
  let ratio := data.schoolC.drawn / data.schoolC.total
  (data.schoolA.total + data.schoolB.total + data.schoolC.total) * ratio

theorem stratified_sampling_theorem (data : SamplingData) 
  (h1 : data.schoolA.total = 1260)
  (h2 : data.schoolB.total = 720)
  (h3 : data.schoolC.total = 900)
  (h4 : data.schoolC.drawn = 50) :
  totalDrawn data = 160 := by
  sorry

#eval totalDrawn { 
  schoolA := { total := 1260, drawn := 0 },
  schoolB := { total := 720, drawn := 0 },
  schoolC := { total := 900, drawn := 50 }
}

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l246_24676


namespace NUMINAMATH_CALUDE_texas_tech_sales_proof_l246_24680

/-- Calculates the money made from t-shirt sales during the Texas Tech game -/
def texas_tech_sales (profit_per_shirt : ℕ) (total_shirts : ℕ) (arkansas_shirts : ℕ) : ℕ :=
  (total_shirts - arkansas_shirts) * profit_per_shirt

/-- Proves that the money made from t-shirt sales during the Texas Tech game is $1092 -/
theorem texas_tech_sales_proof :
  texas_tech_sales 78 186 172 = 1092 := by
  sorry

end NUMINAMATH_CALUDE_texas_tech_sales_proof_l246_24680


namespace NUMINAMATH_CALUDE_paper_edge_length_l246_24624

theorem paper_edge_length (cube_edge : ℝ) (num_papers : ℕ) :
  cube_edge = 12 →
  num_papers = 54 →
  ∃ (paper_edge : ℝ),
    paper_edge^2 * num_papers = 6 * cube_edge^2 ∧
    paper_edge = 4 := by
  sorry

end NUMINAMATH_CALUDE_paper_edge_length_l246_24624


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l246_24679

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c that makes the lines y = 12x + 5 and y = (3c-1)x - 7 parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 12 * x + 5 ↔ y = (3 * c - 1) * x - 7) → c = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l246_24679
