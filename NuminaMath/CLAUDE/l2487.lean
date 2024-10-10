import Mathlib

namespace triangle_theorem_l2487_248771

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def condition (t : Triangle) : Prop :=
  t.a / (Real.sqrt 3 * Real.cos t.A) = t.c / Real.sin t.C

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) (h1 : condition t) (h2 : t.a = 6) :
  t.A = π / 3 ∧ 12 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 18 := by
  sorry


end triangle_theorem_l2487_248771


namespace f_monotonicity_g_maximum_l2487_248728

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - a * x - 1
def g (x : ℝ) : ℝ := Real.log x - x

theorem f_monotonicity :
  (a ≤ 0 → ∀ x y, x < y → f a x < f a y) ∧
  (a > 0 → (∀ x y, Real.log a < x ∧ x < y → f a x < f a y) ∧
           (∀ x y, x < y ∧ y < Real.log a → f a x > f a y)) :=
sorry

theorem g_maximum :
  ∀ x > 0, g x ≤ g 1 ∧ g 1 = -1 :=
sorry

end f_monotonicity_g_maximum_l2487_248728


namespace bonus_calculation_l2487_248777

/-- Represents the initial bonus amount -/
def initial_bonus : ℝ := sorry

/-- Represents the value of stocks after one year -/
def final_value : ℝ := 1350

theorem bonus_calculation : 
  (2 * (initial_bonus / 3) + 2 * (initial_bonus / 3) + (initial_bonus / 3) / 2) = final_value →
  initial_bonus = 900 := by
  sorry

end bonus_calculation_l2487_248777


namespace quadratic_equation_solution_l2487_248774

theorem quadratic_equation_solution (x b : ℝ) : 
  3 * x^2 - b * x + 3 = 0 → x = 1 → b = 6 := by
  sorry

end quadratic_equation_solution_l2487_248774


namespace unique_final_number_l2487_248756

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_final_number (n : ℕ) : Prop :=
  15 ≤ n ∧ n ≤ 25 ∧ sum_of_digits n % 9 = 1

theorem unique_final_number :
  ∃! n : ℕ, is_valid_final_number n ∧ n = 19 :=
sorry

end unique_final_number_l2487_248756


namespace complex_problem_l2487_248783

-- Define a complex number z
variable (z : ℂ)

-- Define the property of being purely imaginary
def isPurelyImaginary (w : ℂ) : Prop := w.re = 0 ∧ w.im ≠ 0

-- State the theorem
theorem complex_problem :
  isPurelyImaginary z →
  isPurelyImaginary ((z + 2)^2 - 8*I) →
  z = -2*I :=
by
  sorry

end complex_problem_l2487_248783


namespace product_of_numbers_l2487_248734

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 24) (sum_squares_eq : x^2 + y^2 = 400) : x * y = 88 := by
  sorry

end product_of_numbers_l2487_248734


namespace boy_walking_time_l2487_248707

/-- If a boy walks at 3/2 of his usual rate and arrives 4 minutes early, 
    his usual time to reach school is 12 minutes. -/
theorem boy_walking_time (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) (h2 : usual_time > 0) : 
    (3 / 2 : ℝ) * usual_rate * (usual_time - 4) = usual_rate * usual_time → 
    usual_time = 12 := by
  sorry

end boy_walking_time_l2487_248707


namespace min_value_x_l2487_248743

theorem min_value_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ 2 * Real.log 3 + (1/3) * Real.log x + 1) :
  x ≥ 27 * Real.exp (3/2) := by
sorry

end min_value_x_l2487_248743


namespace length_QR_is_4_l2487_248776

-- Define the points and circles
structure Point := (x : ℝ) (y : ℝ)
structure Circle := (center : Point) (radius : ℝ)

-- Define the given conditions
axiom P : Point
axiom Q : Point
axiom circle_P : Circle
axiom circle_Q : Circle
axiom R : Point

-- Radii of circles
axiom radius_P : circle_P.radius = 7
axiom radius_Q : circle_Q.radius = 4

-- Circles are externally tangent
axiom externally_tangent : 
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2) = circle_P.radius + circle_Q.radius

-- R is on the line tangent to both circles
axiom R_on_tangent_line : 
  ∃ (S T : Point),
    (Real.sqrt ((S.x - P.x)^2 + (S.y - P.y)^2) = circle_P.radius) ∧
    (Real.sqrt ((T.x - Q.x)^2 + (T.y - Q.y)^2) = circle_Q.radius) ∧
    (R.x - S.x) * (T.y - S.y) = (R.y - S.y) * (T.x - S.x)

-- R is on ray PQ
axiom R_on_ray_PQ :
  ∃ (t : ℝ), t ≥ 0 ∧ R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)

-- Vertical line from Q is tangent to circle at P
axiom vertical_tangent :
  ∃ (U : Point),
    U.x = P.x ∧ 
    Real.sqrt ((U.x - P.x)^2 + (U.y - P.y)^2) = circle_P.radius ∧
    U.x - Q.x = 0

-- Theorem to prove
theorem length_QR_is_4 :
  Real.sqrt ((R.x - Q.x)^2 + (R.y - Q.y)^2) = 4 := by
  sorry

end length_QR_is_4_l2487_248776


namespace quadratic_function_bounds_l2487_248708

/-- Given a quadratic function f(x) = a x^2 - c, prove that if -4 ≤ f(1) ≤ -1 and -1 ≤ f(2) ≤ 5, then -1 ≤ f(3) ≤ 20. -/
theorem quadratic_function_bounds (a c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = a * x^2 - c)
  (h_bound1 : -4 ≤ f 1 ∧ f 1 ≤ -1)
  (h_bound2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end quadratic_function_bounds_l2487_248708


namespace fraction_division_simplification_l2487_248720

theorem fraction_division_simplification : (10 / 21) / (4 / 9) = 15 / 14 := by
  sorry

end fraction_division_simplification_l2487_248720


namespace complement_of_M_l2487_248780

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | |x - 1| ≤ 2}

-- State the theorem
theorem complement_of_M : (U \ M) = {x | x < -1 ∨ x > 3} := by sorry

end complement_of_M_l2487_248780


namespace exponent_multiplication_l2487_248737

theorem exponent_multiplication (x : ℝ) : x^2 * x * x^4 = x^7 := by
  sorry

end exponent_multiplication_l2487_248737


namespace product_equals_square_l2487_248740

theorem product_equals_square : 50 * 39.96 * 3.996 * 500 = 3996^2 := by
  sorry

end product_equals_square_l2487_248740


namespace no_global_minimum_and_local_minimum_at_one_l2487_248795

noncomputable def f (c : ℝ) : ℝ := c^3 + (3/2)*c^2 - 6*c + 4

theorem no_global_minimum_and_local_minimum_at_one :
  (∀ m : ℝ, ∃ c : ℝ, f c < m) ∧
  (∃ δ : ℝ, δ > 0 ∧ ∀ c : ℝ, c ≠ 1 → |c - 1| < δ → f c > f 1) :=
sorry

end no_global_minimum_and_local_minimum_at_one_l2487_248795


namespace consecutive_points_length_l2487_248754

/-- Given 5 consecutive points on a straight line, prove that ae = 22 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 2 * (d - c)) →  -- bc = 2 cd
  (e - d = 8) →            -- de = 8
  (b - a = 5) →            -- ab = 5
  (c - a = 11) →           -- ac = 11
  (e - a = 22) :=          -- ae = 22
by sorry


end consecutive_points_length_l2487_248754


namespace olivia_atm_withdrawal_l2487_248791

theorem olivia_atm_withdrawal (initial_amount spent_amount final_amount : ℕ) 
  (h1 : initial_amount = 100)
  (h2 : spent_amount = 89)
  (h3 : final_amount = 159) : 
  initial_amount + (spent_amount + final_amount) - initial_amount = 148 := by
  sorry

end olivia_atm_withdrawal_l2487_248791


namespace delicate_triangle_existence_and_property_l2487_248796

/-- Definition of a delicate triangle -/
def is_delicate_triangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧ (1 / a : ℚ) = (1 / b : ℚ) + (1 / c : ℚ)

theorem delicate_triangle_existence_and_property :
  (∃ a b c : ℕ, is_delicate_triangle a b c) ∧
  (∀ a b c : ℕ, is_delicate_triangle a b c → ∃ n : ℕ, a^2 + b^2 + c^2 = n^2) :=
by sorry

end delicate_triangle_existence_and_property_l2487_248796


namespace smallest_b_in_geometric_series_l2487_248724

theorem smallest_b_in_geometric_series (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- a, b, c are positive
  (∃ r : ℝ, a = b * r ∧ c = b / r) →  -- a, b, c form a geometric series
  a * b * c = 216 →  -- product condition
  b ≥ 6 ∧ (∀ b' : ℝ, 
    (∃ a' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    (∃ r' : ℝ, a' = b' * r' ∧ c' = b' / r') ∧ 
    a' * b' * c' = 216) → 
    b' ≥ 6) :=
by sorry

end smallest_b_in_geometric_series_l2487_248724


namespace regular_polygon_sides_l2487_248768

/-- A regular polygon with perimeter 180 and side length 15 has 12 sides -/
theorem regular_polygon_sides (P : ℝ) (s : ℝ) (h1 : P = 180) (h2 : s = 15) :
  P / s = 12 := by
  sorry

#check regular_polygon_sides

end regular_polygon_sides_l2487_248768


namespace system_1_solution_system_2_solution_l2487_248758

-- System 1
theorem system_1_solution :
  ∃! (x y : ℝ), 3 * x + 2 * y = 8 ∧ y = 2 * x - 3 ∧ x = 2 ∧ y = 1 := by sorry

-- System 2
theorem system_2_solution :
  ∃! (x y : ℝ), 3 * x + 2 * y = 7 ∧ 6 * x - 2 * y = 11 ∧ x = 2 ∧ y = 1/2 := by sorry

end system_1_solution_system_2_solution_l2487_248758


namespace complex_fraction_equality_l2487_248725

theorem complex_fraction_equality : (3 + I) / (1 + I) = 2 - I := by sorry

end complex_fraction_equality_l2487_248725


namespace parabola_axis_of_symmetry_l2487_248751

/-- Given a parabola y = 2x² - bx + 3 with axis of symmetry x = 1, prove that b = 4 -/
theorem parabola_axis_of_symmetry (b : ℝ) : 
  (∀ x y : ℝ, y = 2*x^2 - b*x + 3) → 
  (1 = -b / (2*2)) → 
  b = 4 := by sorry

end parabola_axis_of_symmetry_l2487_248751


namespace exponent_division_l2487_248753

theorem exponent_division (a : ℝ) : a^10 / a^5 = a^5 := by
  sorry

end exponent_division_l2487_248753


namespace lake_circumference_difference_is_680_l2487_248739

/-- The difference between the circumferences of two lakes -/
def lake_circumference_difference : ℕ :=
  let eastern_trees : ℕ := 96
  let eastern_interval : ℕ := 10
  let western_trees : ℕ := 82
  let western_interval : ℕ := 20
  let eastern_circumference := eastern_trees * eastern_interval
  let western_circumference := western_trees * western_interval
  western_circumference - eastern_circumference

/-- Theorem stating that the difference between the circumferences of the two lakes is 680 meters -/
theorem lake_circumference_difference_is_680 : lake_circumference_difference = 680 := by
  sorry

end lake_circumference_difference_is_680_l2487_248739


namespace equation_solution_l2487_248789

theorem equation_solution (y : ℝ) : ∃ x : ℝ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 ∧ x = -1 := by
  sorry

end equation_solution_l2487_248789


namespace inequality_proof_l2487_248706

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ 
  4 * Real.sqrt ((a*b/(a+b))^2 + (a*b/(a+b))*(a*c/(a+c)) + (a*c/(a+c))^2) :=
by sorry

end inequality_proof_l2487_248706


namespace range_of_m_l2487_248790

open Real Set

theorem range_of_m (p q : Prop) (m : ℝ) : 
  (∀ x ∈ Icc 0 (π/4), tan x ≤ m) → p ∧
  (∀ x₁ ∈ Icc (-1) 3, ∃ x₂ ∈ Icc 0 2, x₁^2 + m ≥ (1/2)^x₂ - m) → q ∧
  (¬(p ∧ q)) ∧ (p ∨ q) →
  m ∈ Icc (1/8) 1 ∧ m < 1 :=
by sorry

end range_of_m_l2487_248790


namespace triangle_side_length_l2487_248726

theorem triangle_side_length (b c : ℝ) (h1 : b^2 - 7*b + 11 = 0) (h2 : c^2 - 7*c + 11 = 0) : 
  let a := Real.sqrt ((b^2 + c^2) - b*c)
  (a = 4) ∧ (b + c = 7) ∧ (b*c = 11) := by
  sorry

end triangle_side_length_l2487_248726


namespace recurrence_relation_solution_l2487_248702

def a (n : ℕ) : ℤ := 1 + 5 * 3^n - 4 * 2^n

theorem recurrence_relation_solution :
  (∀ n : ℕ, n ≥ 2 → a n = 4 * a (n - 1) - 3 * a (n - 2) + 2^n) ∧
  a 0 = 2 ∧
  a 1 = 8 := by
sorry

end recurrence_relation_solution_l2487_248702


namespace rectangles_count_l2487_248779

/-- A structure representing a square divided into rectangles -/
structure DividedSquare where
  k : ℕ  -- number of rectangles intersected by a vertical line
  l : ℕ  -- number of rectangles intersected by a horizontal line

/-- The total number of rectangles in a divided square -/
def total_rectangles (sq : DividedSquare) : ℕ := sq.k * sq.l

/-- Theorem stating that the total number of rectangles is k * l -/
theorem rectangles_count (sq : DividedSquare) : 
  total_rectangles sq = sq.k * sq.l := by sorry

end rectangles_count_l2487_248779


namespace sum_of_squares_inequality_l2487_248729

theorem sum_of_squares_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ((x + 1) / x)^2 + ((y + 1) / y)^2 ≥ 18 := by
  sorry

end sum_of_squares_inequality_l2487_248729


namespace arithmetic_sequence_a11_l2487_248788

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a11 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 = 4 → a 5 = 8 → a 11 = 12 := by
  sorry

end arithmetic_sequence_a11_l2487_248788


namespace hyperbola_condition_exclusive_or_condition_l2487_248750

def P (a : ℝ) : Prop := ∀ x, x^2 - a*x + a + 5/4 > 0

def Q (a : ℝ) : Prop := ∃ x y, x^2 / (4*a + 7) + y^2 / (a - 3) = 1

theorem hyperbola_condition (a : ℝ) : Q a ↔ a ∈ Set.Ioo (-7/4 : ℝ) 3 := by sorry

theorem exclusive_or_condition (a : ℝ) : (¬(P a ∧ Q a) ∧ (P a ∨ Q a)) ↔ 
  a ∈ Set.Ioc (-7/4 : ℝ) (-1) ∪ Set.Ico 3 5 := by sorry

end hyperbola_condition_exclusive_or_condition_l2487_248750


namespace complex_fraction_sum_l2487_248781

theorem complex_fraction_sum (a b : ℝ) :
  (Complex.I + 1) / (1 - Complex.I) = Complex.mk a b →
  a + b = 1 := by
sorry

end complex_fraction_sum_l2487_248781


namespace is_quadratic_x_squared_minus_3x_plus_2_l2487_248714

/-- Definition of a quadratic equation -/
def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ (x : ℝ), a * x^2 + b * x + c = 0

/-- The equation x² - 3x + 2 = 0 is a quadratic equation -/
theorem is_quadratic_x_squared_minus_3x_plus_2 :
  is_quadratic_equation 1 (-3) 2 := by
  sorry

end is_quadratic_x_squared_minus_3x_plus_2_l2487_248714


namespace min_value_theorem_l2487_248719

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  ∃ (m : ℝ), m = 7 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 3 * b = 5 * a * b → 3 * a + 4 * b ≥ m := by
  sorry

end min_value_theorem_l2487_248719


namespace industrial_lubricants_percentage_l2487_248711

theorem industrial_lubricants_percentage (microphotonics : ℝ) (home_electronics : ℝ) 
  (food_additives : ℝ) (genetically_modified_microorganisms : ℝ) 
  (basic_astrophysics_degrees : ℝ) :
  microphotonics = 10 →
  home_electronics = 24 →
  food_additives = 15 →
  genetically_modified_microorganisms = 29 →
  basic_astrophysics_degrees = 50.4 →
  ∃ (industrial_lubricants : ℝ),
    industrial_lubricants = 8 ∧
    microphotonics + home_electronics + food_additives + 
    genetically_modified_microorganisms + industrial_lubricants + 
    (basic_astrophysics_degrees / 360 * 100) = 100 := by
  sorry

end industrial_lubricants_percentage_l2487_248711


namespace smallest_k_sum_squares_divisible_250_l2487_248746

/-- Sum of squares formula -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Predicate to check if a number is divisible by 250 -/
def divisible_by_250 (n : ℕ) : Prop := ∃ m : ℕ, n = 250 * m

theorem smallest_k_sum_squares_divisible_250 :
  (∀ k < 375, ¬(divisible_by_250 (sum_of_squares k))) ∧
  (divisible_by_250 (sum_of_squares 375)) := by sorry

end smallest_k_sum_squares_divisible_250_l2487_248746


namespace inscribed_circle_right_triangle_l2487_248782

theorem inscribed_circle_right_triangle 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_perimeter : a + b + c = 30) 
  (h_tangency_ratio : ∃ (r : ℝ), a = 5*r/2 ∧ b = 12*r/5) : 
  (a = 5 ∧ b = 12 ∧ c = 13) := by
sorry

end inscribed_circle_right_triangle_l2487_248782


namespace friday_blood_pressure_l2487_248767

/-- Calculates the final blood pressure given an initial value and a list of daily changes. -/
def finalBloodPressure (initial : ℕ) (changes : List ℤ) : ℕ :=
  (changes.foldl (fun acc change => (acc : ℤ) + change) initial).toNat

/-- Theorem stating that given the initial blood pressure and daily changes, 
    the final blood pressure on Friday is 130 units. -/
theorem friday_blood_pressure :
  let initial : ℕ := 120
  let changes : List ℤ := [20, -30, -25, 15, 30]
  finalBloodPressure initial changes = 130 := by sorry

end friday_blood_pressure_l2487_248767


namespace gravel_path_rate_l2487_248712

/-- Calculates the rate per square meter for gravelling a path around a rectangular plot -/
theorem gravel_path_rate (length width path_width total_cost : ℝ) 
  (h1 : length = 110)
  (h2 : width = 65)
  (h3 : path_width = 2.5)
  (h4 : total_cost = 595) : 
  total_cost / ((length * width) - ((length - 2 * path_width) * (width - 2 * path_width))) = 0.7 := by
  sorry

end gravel_path_rate_l2487_248712


namespace prob_at_least_one_from_subset_l2487_248755

/-- The probability of selecting at least one song from a specified subset when randomly choosing 2 out of 4 songs -/
theorem prob_at_least_one_from_subset :
  let total_songs : ℕ := 4
  let songs_to_play : ℕ := 2
  let subset_size : ℕ := 2
  Nat.choose total_songs songs_to_play = 6 →
  (1 : ℚ) - (Nat.choose (total_songs - subset_size) songs_to_play : ℚ) / (Nat.choose total_songs songs_to_play : ℚ) = 5/6 :=
by sorry

end prob_at_least_one_from_subset_l2487_248755


namespace three_digit_number_solution_l2487_248732

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Convert a repeating decimal of the form 0.ab̅ab to a fraction -/
def repeating_decimal_to_fraction (a b : Digit) : Rat :=
  (10 * a.val + b.val : Rat) / 99

/-- Convert a repeating decimal of the form 0.abc̅abc to a fraction -/
def repeating_decimal_to_fraction_3 (a b c : Digit) : Rat :=
  (100 * a.val + 10 * b.val + c.val : Rat) / 999

/-- The main theorem -/
theorem three_digit_number_solution (c d e : Digit) :
  repeating_decimal_to_fraction c d + repeating_decimal_to_fraction_3 c d e = 44 / 99 →
  (c.val * 100 + d.val * 10 + e.val : Nat) = 400 := by
  sorry

end three_digit_number_solution_l2487_248732


namespace same_color_pen_probability_l2487_248701

theorem same_color_pen_probability (blue_pens black_pens : ℕ) 
  (h1 : blue_pens = 8) (h2 : black_pens = 5) : 
  let total_pens := blue_pens + black_pens
  (blue_pens / total_pens)^2 + (black_pens / total_pens)^2 = 89 / 169 := by
  sorry

end same_color_pen_probability_l2487_248701


namespace male_avg_is_58_l2487_248735

/-- Represents an association with male and female members selling raffle tickets -/
structure Association where
  total_avg : ℝ
  female_avg : ℝ
  male_female_ratio : ℝ

/-- The average number of tickets sold by male members -/
def male_avg (a : Association) : ℝ :=
  (3 * a.total_avg - 2 * a.female_avg)

/-- Theorem stating the average number of tickets sold by male members -/
theorem male_avg_is_58 (a : Association) 
  (h1 : a.total_avg = 66)
  (h2 : a.female_avg = 70)
  (h3 : a.male_female_ratio = 1/2) :
  male_avg a = 58 := by
  sorry


end male_avg_is_58_l2487_248735


namespace alcohol_mixture_problem_l2487_248717

-- Define the initial volume of the solution
def initial_volume : ℝ := 6

-- Define the volume of pure alcohol added
def added_alcohol : ℝ := 1.8

-- Define the final alcohol percentage
def final_percentage : ℝ := 0.5

-- Define the initial alcohol percentage (to be proven)
def initial_percentage : ℝ := 0.35

-- Theorem statement
theorem alcohol_mixture_problem :
  initial_volume * initial_percentage + added_alcohol =
  (initial_volume + added_alcohol) * final_percentage := by
  sorry

end alcohol_mixture_problem_l2487_248717


namespace geometric_sequence_special_property_l2487_248716

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_special_property 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arithmetic : 2 * ((1/2) * a 3) = a 1 + 2 * a 2) :
  q = 1 + Real.sqrt 2 :=
sorry

end geometric_sequence_special_property_l2487_248716


namespace solve_for_A_l2487_248765

/-- Given the equation 691-6A7=4 in base 10, prove that A = 8 -/
theorem solve_for_A : ∃ (A : ℕ), A < 10 ∧ 691 - (600 + A * 10 + 7) = 4 → A = 8 := by sorry

end solve_for_A_l2487_248765


namespace original_shirt_price_l2487_248742

/-- Proves that if a shirt's current price is $6 and this price is 25% of the original price, 
    then the original price was $24. -/
theorem original_shirt_price (current_price : ℝ) (original_price : ℝ) : 
  current_price = 6 → 
  current_price = 0.25 * original_price →
  original_price = 24 := by
sorry

end original_shirt_price_l2487_248742


namespace new_model_count_l2487_248738

/-- Given an initial cost per model and the ability to buy a certain number of models,
    calculate the new number of models that can be purchased after a price increase. -/
theorem new_model_count (initial_cost new_cost : ℚ) (initial_count : ℕ) : 
  initial_cost > 0 →
  new_cost > 0 →
  initial_count > 0 →
  (initial_cost * initial_count) / new_cost = 27 →
  initial_cost = 0.45 →
  new_cost = 0.50 →
  initial_count = 30 →
  ⌊(initial_cost * initial_count) / new_cost⌋ = 27 := by
sorry

#eval (0.45 * 30) / 0.50

end new_model_count_l2487_248738


namespace quadratic_and_inequality_system_l2487_248760

theorem quadratic_and_inequality_system :
  (∃ x1 x2 : ℝ, x1 = 1 ∧ x2 = 5 ∧ 
    (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x = x1 ∨ x = x2))) ∧
  (∀ x : ℝ, x + 3 > 0 ∧ 2*(x - 1) < 4 ↔ -3 < x ∧ x < 3) :=
by sorry

end quadratic_and_inequality_system_l2487_248760


namespace loss_per_metre_is_12_l2487_248762

/-- Calculates the loss per metre of cloth given the total metres sold, total selling price, and cost price per metre. -/
def loss_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  let total_cost_price := total_metres * cost_price_per_metre
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_metres

/-- Theorem stating that the loss per metre of cloth is 12 given the specified conditions. -/
theorem loss_per_metre_is_12 :
  loss_per_metre 200 12000 72 = 12 := by
  sorry

end loss_per_metre_is_12_l2487_248762


namespace train_length_calculation_l2487_248747

/-- Calculates the length of a train given its speed, bridge length, and time to pass the bridge. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (time_to_pass : Real) :
  let speed_m_s : Real := train_speed * 1000 / 3600
  let total_distance : Real := speed_m_s * time_to_pass
  let train_length : Real := total_distance - bridge_length
  train_speed = 60 ∧ bridge_length = 800 ∧ time_to_pass = 72 →
  (train_length ≥ 400.24 ∧ train_length ≤ 400.25) := by
  sorry

#check train_length_calculation

end train_length_calculation_l2487_248747


namespace angle_in_third_quadrant_l2487_248775

theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) > 0) 
  (h2 : Real.cos α < 0) : 
  π < α ∧ α < 3 * π / 2 := by
  sorry

end angle_in_third_quadrant_l2487_248775


namespace inequality_equivalence_l2487_248748

theorem inequality_equivalence (x : ℝ) : x - 2 > 1 ↔ x > 3 := by
  sorry

end inequality_equivalence_l2487_248748


namespace min_k_10_l2487_248798

-- Define a stringent function
def Stringent (h : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 ∧ y > 0 → h x + h y > 2 * y^2

-- Define the sum of k from 1 to 15
def SumK (k : ℕ → ℤ) : ℤ :=
  (List.range 15).map (fun i => k (i + 1)) |> List.sum

-- Theorem statement
theorem min_k_10 (k : ℕ → ℤ) (hk : Stringent k) 
  (hmin : ∀ j : ℕ → ℤ, Stringent j → SumK k ≤ SumK j) : 
  k 10 ≥ 120 := by
  sorry

end min_k_10_l2487_248798


namespace smallest_consecutive_sum_theorem_l2487_248773

/-- The smallest natural number that can be expressed as the sum of 9, 10, and 11 consecutive non-zero natural numbers. -/
def smallest_consecutive_sum : ℕ := 495

/-- Checks if a natural number can be expressed as the sum of n consecutive non-zero natural numbers. -/
def is_sum_of_consecutive (x n : ℕ) : Prop :=
  ∃ k : ℕ, x = (n * (2*k + n + 1)) / 2

/-- The main theorem stating that smallest_consecutive_sum is the smallest natural number
    that can be expressed as the sum of 9, 10, and 11 consecutive non-zero natural numbers. -/
theorem smallest_consecutive_sum_theorem :
  (is_sum_of_consecutive smallest_consecutive_sum 9) ∧
  (is_sum_of_consecutive smallest_consecutive_sum 10) ∧
  (is_sum_of_consecutive smallest_consecutive_sum 11) ∧
  (∀ m : ℕ, m < smallest_consecutive_sum →
    ¬(is_sum_of_consecutive m 9 ∧ is_sum_of_consecutive m 10 ∧ is_sum_of_consecutive m 11)) :=
sorry

end smallest_consecutive_sum_theorem_l2487_248773


namespace car_speed_second_hour_l2487_248705

/-- Given a car traveling for 2 hours with a speed of 20 km/h in the first hour
    and an average speed of 25 km/h, prove that the speed of the car in the second hour is 30 km/h. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (total_time : ℝ)
  (h1 : speed_first_hour = 20)
  (h2 : average_speed = 25)
  (h3 : total_time = 2) :
  let speed_second_hour := (average_speed * total_time - speed_first_hour)
  speed_second_hour = 30 := by
  sorry

end car_speed_second_hour_l2487_248705


namespace rectangle_dimensions_l2487_248778

/-- Proves that a rectangle with specific properties has dimensions 7 and 4 -/
theorem rectangle_dimensions :
  ∀ l w : ℝ,
  l = w + 3 →
  l * w = 2 * (2 * l + 2 * w) →
  l = 7 ∧ w = 4 := by
  sorry

end rectangle_dimensions_l2487_248778


namespace youngest_child_age_l2487_248772

/-- Given 5 children born at intervals of 3 years, if the sum of their ages is 70 years,
    then the age of the youngest child is 8 years. -/
theorem youngest_child_age (youngest_age : ℕ) : 
  (youngest_age + (youngest_age + 3) + (youngest_age + 6) + (youngest_age + 9) + (youngest_age + 12) = 70) →
  youngest_age = 8 := by
  sorry

end youngest_child_age_l2487_248772


namespace triangle_problem_l2487_248770

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  2 * c * Real.cos C = b * Real.cos A + a * Real.cos B ∧
  a = 6 ∧
  Real.cos A = -4/5 →
  C = π/3 ∧ c = 5 * Real.sqrt 3 := by sorry

end triangle_problem_l2487_248770


namespace vector_basis_l2487_248710

def e₁ : ℝ × ℝ := (-1, 3)
def e₂ : ℝ × ℝ := (5, -2)

theorem vector_basis : LinearIndependent ℝ ![e₁, e₂] ∧ Submodule.span ℝ {e₁, e₂} = ⊤ := by
  sorry

end vector_basis_l2487_248710


namespace first_player_wins_l2487_248718

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a rectangle piece -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents the game board -/
structure Board where
  rows : Nat
  cols : Nat

/-- Represents a game state -/
structure GameState where
  board : Board
  currentPlayer : Player

/-- Defines a valid move in the game -/
def ValidMove (rect : Rectangle) (state : GameState) : Prop :=
  match state.currentPlayer with
  | Player.First => rect.width = 1 ∧ rect.height = 2
  | Player.Second => rect.width = 2 ∧ rect.height = 1

/-- Defines the winning condition -/
def HasWinningStrategy (player : Player) (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Rectangle),
    ∀ (opponent_move : Rectangle),
      ValidMove opponent_move initialState →
      ∃ (final_state : GameState),
        (final_state.currentPlayer = player) ∧
        (¬∃ (move : Rectangle), ValidMove move final_state)

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  let initial_board : Board := { rows := 3, cols := 1000 }
  let initial_state : GameState := { board := initial_board, currentPlayer := Player.First }
  HasWinningStrategy Player.First initial_state :=
sorry

end first_player_wins_l2487_248718


namespace stating_max_segments_proof_l2487_248799

/-- 
Given an equilateral triangle with side length n, divided into n^2 smaller 
equilateral triangles with side length 1, this function returns the maximum 
number of segments that can be chosen such that no three chosen segments 
form a triangle.
-/
def max_segments (n : ℕ) : ℕ :=
  n * (n + 1)

/-- 
Theorem stating that the maximum number of segments that can be chosen 
such that no three chosen segments form a triangle is n(n+1).
-/
theorem max_segments_proof (n : ℕ) : 
  max_segments n = n * (n + 1) := by
  sorry

end stating_max_segments_proof_l2487_248799


namespace leftover_coins_value_l2487_248757

def roll_size : ℕ := 40
def quarter_value : ℚ := 0.25
def nickel_value : ℚ := 0.05

def mia_quarters : ℕ := 92
def mia_nickels : ℕ := 184
def thomas_quarters : ℕ := 138
def thomas_nickels : ℕ := 212

def total_quarters : ℕ := mia_quarters + thomas_quarters
def total_nickels : ℕ := mia_nickels + thomas_nickels

def leftover_quarters : ℕ := total_quarters % roll_size
def leftover_nickels : ℕ := total_nickels % roll_size

def leftover_value : ℚ := leftover_quarters * quarter_value + leftover_nickels * nickel_value

theorem leftover_coins_value : leftover_value = 9.30 := by
  sorry

end leftover_coins_value_l2487_248757


namespace video_games_spent_l2487_248792

def total_allowance : ℚ := 50

def books_fraction : ℚ := 1/4
def snacks_fraction : ℚ := 2/5
def apps_fraction : ℚ := 1/5

def books_spent : ℚ := total_allowance * books_fraction
def snacks_spent : ℚ := total_allowance * snacks_fraction
def apps_spent : ℚ := total_allowance * apps_fraction

theorem video_games_spent :
  total_allowance - (books_spent + snacks_spent + apps_spent) = 7.5 := by
  sorry

end video_games_spent_l2487_248792


namespace equation_solution_l2487_248744

theorem equation_solution : 
  ∀ x : ℝ, (40 / 60 : ℝ) = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end equation_solution_l2487_248744


namespace absolute_value_equation_product_l2487_248794

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|6 * x₁| + 5 = 47 ∧ |6 * x₂| + 5 = 47 ∧ x₁ ≠ x₂) ∧ x₁ * x₂ = -49) := by
  sorry

end absolute_value_equation_product_l2487_248794


namespace base8_543_to_base10_l2487_248713

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ :=
  let d₀ := n % 8
  let d₁ := (n / 8) % 8
  let d₂ := (n / 64) % 8
  d₀ + 8 * d₁ + 64 * d₂

theorem base8_543_to_base10 : base8ToBase10 543 = 355 := by sorry

end base8_543_to_base10_l2487_248713


namespace lever_force_calculation_l2487_248745

/-- Represents the force required to move an object with a lever -/
structure LeverForce where
  length : ℝ
  force : ℝ

/-- The inverse relationship between force and lever length -/
def inverse_relationship (k : ℝ) (lf : LeverForce) : Prop :=
  lf.force * lf.length = k

theorem lever_force_calculation (k : ℝ) (lf1 lf2 : LeverForce) :
  inverse_relationship k lf1 →
  inverse_relationship k lf2 →
  lf1.length = 12 →
  lf1.force = 200 →
  lf2.length = 8 →
  lf2.force = 300 :=
by
  sorry

#check lever_force_calculation

end lever_force_calculation_l2487_248745


namespace triangle_longest_side_range_l2487_248730

/-- Given a rope of length l that can exactly enclose two congruent triangles,
    prove that the longest side x of one of the triangles satisfies l/6 ≤ x < l/4 -/
theorem triangle_longest_side_range (l : ℝ) (x y z : ℝ) :
  l > 0 →
  x > 0 ∧ y > 0 ∧ z > 0 →
  x + y + z = l / 2 →
  x ≥ y ∧ x ≥ z →
  l / 6 ≤ x ∧ x < l / 4 := by
  sorry

end triangle_longest_side_range_l2487_248730


namespace sqrt_less_than_3y_minus_1_l2487_248731

theorem sqrt_less_than_3y_minus_1 (y : ℝ) :
  y > 0 → (Real.sqrt y < 3 * y - 1 ↔ y > 1) := by sorry

end sqrt_less_than_3y_minus_1_l2487_248731


namespace no_common_points_l2487_248723

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (1 + 2*t, -2 + 4*t)

-- Define the line L: 2x - y = 0
def line_L (x y : ℝ) : Prop := 2*x - y = 0

-- Theorem statement
theorem no_common_points :
  ∀ (t : ℝ), ¬(line_L (curve_C t).1 (curve_C t).2) := by
  sorry

end no_common_points_l2487_248723


namespace sin_plus_3cos_value_l2487_248703

theorem sin_plus_3cos_value (x : ℝ) (h : 2 * Real.cos x - 3 * Real.sin x = 4) :
  ∃ (y : ℝ), (Real.sin x + 3 * Real.cos x = y) ∧ (y = -2 + 6 * Real.sqrt 10 ∨ y = -2 - 6 * Real.sqrt 10) := by
  sorry

end sin_plus_3cos_value_l2487_248703


namespace housewife_purchasing_comparison_l2487_248766

theorem housewife_purchasing_comparison (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  (2 * a * b) / (a + b) < (a + b) / 2 := by
  sorry

end housewife_purchasing_comparison_l2487_248766


namespace complex_abs_one_plus_i_over_i_l2487_248733

theorem complex_abs_one_plus_i_over_i : Complex.abs ((1 + Complex.I) / Complex.I) = Real.sqrt 2 := by
  sorry

end complex_abs_one_plus_i_over_i_l2487_248733


namespace expression_evaluation_l2487_248741

theorem expression_evaluation :
  let x : ℚ := -1
  let y : ℚ := -2
  ((x + y)^2 - (3*x - y)*(3*x + y) - 2*y^2) / (-2*x) = -2 :=
by sorry

end expression_evaluation_l2487_248741


namespace absolute_value_inequality_solution_set_l2487_248749

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x + 1| > 3} = {x : ℝ | x < -2 ∨ x > 1} := by sorry

end absolute_value_inequality_solution_set_l2487_248749


namespace count_divisible_integers_l2487_248761

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (7 * n) % (n * (n + 1) / 2) = 0) ∧ 
    Finset.card S = 3 := by
  sorry

end count_divisible_integers_l2487_248761


namespace decreasing_geometric_sequence_properties_l2487_248786

/-- A decreasing geometric sequence -/
def DecreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ |a (n + 1)| < |a n|

theorem decreasing_geometric_sequence_properties
  (a : ℕ → ℝ) (h : DecreasingGeometricSequence a) :
  (∀ n : ℕ, a n > 0 → ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ m : ℕ, a (m + 1) = q * a m) ∧
  (∀ n : ℕ, a n < 0 → ∃ q : ℝ, q > 1 ∧ ∀ m : ℕ, a (m + 1) = q * a m) :=
by
  sorry

end decreasing_geometric_sequence_properties_l2487_248786


namespace stock_price_increase_percentage_l2487_248736

theorem stock_price_increase_percentage (total_stocks : ℕ) (higher_stocks : ℕ) : 
  total_stocks = 1980 →
  higher_stocks = 1080 →
  higher_stocks > (total_stocks - higher_stocks) →
  (((higher_stocks : ℝ) - (total_stocks - higher_stocks)) / (total_stocks - higher_stocks : ℝ)) * 100 = 20 :=
by sorry

end stock_price_increase_percentage_l2487_248736


namespace max_value_of_f_l2487_248764

def f (x : ℝ) : ℝ := -(x + 1)^2 + 5

theorem max_value_of_f :
  ∀ x : ℝ, f x ≤ 5 ∧ ∃ x₀ : ℝ, f x₀ = 5 :=
by
  sorry

end max_value_of_f_l2487_248764


namespace b_work_days_l2487_248784

/-- The number of days it takes for worker A to complete the work alone -/
def a_days : ℝ := 15

/-- The number of days A and B work together -/
def together_days : ℝ := 8

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.06666666666666665

/-- The number of days it takes for worker B to complete the work alone -/
def b_days : ℝ := 20

/-- Theorem stating that given the conditions, B can complete the work alone in 20 days -/
theorem b_work_days : 
  together_days * (1 / a_days + 1 / b_days) = 1 - work_left :=
sorry

end b_work_days_l2487_248784


namespace f_increasing_on_neg_two_to_zero_l2487_248700

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem f_increasing_on_neg_two_to_zero
  (a b : ℝ)
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_domain : Set.Icc (a - 1) 2 = Set.Icc (-2) 2) :
  StrictMonoOn (f a b) (Set.Icc (-2) 0) :=
sorry

end f_increasing_on_neg_two_to_zero_l2487_248700


namespace cylinder_cross_section_area_l2487_248763

/-- The area of a cross-section of a cylinder with given dimensions and cut angle -/
theorem cylinder_cross_section_area (h r : ℝ) (θ : ℝ) :
  h = 10 ∧ r = 7 ∧ θ = 150 * π / 180 →
  ∃ (A : ℝ), A = (73.5 : ℝ) * π + 70 * Real.sqrt 6 ∧
    A = r * (2 * r * Real.sin (θ / 2)) + π * (r * Real.sin (θ / 2))^2 / 2 :=
by sorry

end cylinder_cross_section_area_l2487_248763


namespace binary_101_equals_decimal_5_l2487_248721

def binary_to_decimal (b₂ b₁ b₀ : ℕ) : ℕ :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_101_equals_decimal_5 : binary_to_decimal 1 0 1 = 5 := by
  sorry

end binary_101_equals_decimal_5_l2487_248721


namespace base_nine_to_decimal_l2487_248759

/-- Given that the base-9 number 16m27₍₉₎ equals 11203 in decimal, prove that m = 3 -/
theorem base_nine_to_decimal (m : ℕ) : 
  (7 + 2 * 9^1 + m * 9^2 + 6 * 9^3 + 1 * 9^4 = 11203) → m = 3 := by
sorry

end base_nine_to_decimal_l2487_248759


namespace box_fillable_with_gamma_bricks_l2487_248793

/-- Represents a Γ-shape brick composed of three 1×1×1 cubes -/
structure GammaBrick :=
  (shape : Fin 3 → Fin 3 → Fin 3 → Bool)

/-- Represents a box with dimensions m × n × k -/
structure Box (m n k : ℕ) :=
  (filled : Fin m → Fin n → Fin k → Bool)

/-- Predicate to check if a box is completely filled with Γ-shape bricks -/
def is_filled_with_gamma_bricks (m n k : ℕ) (box : Box m n k) : Prop :=
  ∀ (i : Fin m) (j : Fin n) (l : Fin k), box.filled i j l = true

/-- Theorem stating that any box with dimensions m, n, k > 1 can be filled with Γ-shape bricks -/
theorem box_fillable_with_gamma_bricks (m n k : ℕ) (hm : m > 1) (hn : n > 1) (hk : k > 1) :
  ∃ (box : Box m n k), is_filled_with_gamma_bricks m n k box :=
sorry

end box_fillable_with_gamma_bricks_l2487_248793


namespace fraction_value_l2487_248769

theorem fraction_value (a b : ℝ) (h : a / b = 3 / 5) : (2 * a + b) / (2 * a - b) = 11 := by
  sorry

end fraction_value_l2487_248769


namespace sequence_limit_implies_first_term_l2487_248722

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = n

def limit_property (a : ℕ → ℝ) : Prop :=
  Filter.Tendsto (λ n => a n / a (n + 1)) Filter.atTop (nhds 1)

theorem sequence_limit_implies_first_term (a : ℕ → ℝ) 
    (h1 : sequence_property a) 
    (h2 : limit_property a) : 
    a 1 = Real.sqrt (2 / Real.pi) := by
  sorry

end sequence_limit_implies_first_term_l2487_248722


namespace negation_of_existence_negation_of_greater_than_1000_l2487_248752

theorem negation_of_existence (P : ℕ → Prop) :
  (¬∃ n, P n) ↔ (∀ n, ¬P n) := by sorry

theorem negation_of_greater_than_1000 :
  (¬∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end negation_of_existence_negation_of_greater_than_1000_l2487_248752


namespace fruits_picked_and_ratio_l2487_248715

/-- Represents the number of fruits picked by a person -/
structure FruitsPicked where
  pears : ℕ
  apples : ℕ

/-- Represents the orchard -/
structure Orchard where
  pear_trees : ℕ
  apple_trees : ℕ

def keith_picked : FruitsPicked := { pears := 6, apples := 4 }
def jason_picked : FruitsPicked := { pears := 9, apples := 8 }
def joan_picked : FruitsPicked := { pears := 4, apples := 12 }

def orchard : Orchard := { pear_trees := 4, apple_trees := 3 }

def total_fruits (keith jason joan : FruitsPicked) : ℕ :=
  keith.pears + keith.apples + jason.pears + jason.apples + joan.pears + joan.apples

def total_apples (keith jason joan : FruitsPicked) : ℕ :=
  keith.apples + jason.apples + joan.apples

def total_pears (keith jason joan : FruitsPicked) : ℕ :=
  keith.pears + jason.pears + joan.pears

theorem fruits_picked_and_ratio 
  (keith jason joan : FruitsPicked) 
  (o : Orchard) 
  (h_keith : keith = keith_picked)
  (h_jason : jason = jason_picked)
  (h_joan : joan = joan_picked)
  (h_orchard : o = orchard) :
  total_fruits keith jason joan = 43 ∧ 
  total_apples keith jason joan = 24 ∧
  total_pears keith jason joan = 19 := by
  sorry

end fruits_picked_and_ratio_l2487_248715


namespace relationship_abc_l2487_248785

-- Define the constants
noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.8 / Real.log 1.2
noncomputable def c : ℝ := (1.5 : ℝ) ^ (1/2)

-- State the theorem
theorem relationship_abc : c > a ∧ a > b := by sorry

end relationship_abc_l2487_248785


namespace max_product_762_l2487_248704

def digits : Finset Nat := {2, 4, 6, 7, 9}

def is_valid_pair (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit (d e : Nat) : Nat := 10 * d + e

theorem max_product_762 :
  ∀ a b c d e : Nat,
  is_valid_pair a b c d e →
  three_digit 7 6 2 * two_digit 9 4 ≥ three_digit a b c * two_digit d e :=
by sorry

end max_product_762_l2487_248704


namespace dexter_cards_total_dexter_cards_count_l2487_248797

theorem dexter_cards_total (basketball_boxes : ℕ) (basketball_cards_per_box : ℕ) 
  (football_cards_per_box : ℕ) (box_difference : ℕ) : ℕ :=
  let football_boxes := basketball_boxes - box_difference
  let total_basketball_cards := basketball_boxes * basketball_cards_per_box
  let total_football_cards := football_boxes * football_cards_per_box
  total_basketball_cards + total_football_cards

-- Main theorem
theorem dexter_cards_count : 
  dexter_cards_total 12 20 25 5 = 415 := by
  sorry

end dexter_cards_total_dexter_cards_count_l2487_248797


namespace cubic_root_form_l2487_248727

theorem cubic_root_form (x : ℝ) (hx : x > 0) (hcubic : x^3 - 4*x^2 - 2*x - Real.sqrt 3 = 0) :
  ∃ (a b : ℝ), x = a + b * Real.sqrt 3 := by
sorry

end cubic_root_form_l2487_248727


namespace no_functions_exist_for_part_a_functions_exist_for_part_b_l2487_248787

-- Part (a)
theorem no_functions_exist_for_part_a :
  ¬∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^3 := by sorry

-- Part (b)
theorem functions_exist_for_part_b :
  ∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^4 := by sorry

end no_functions_exist_for_part_a_functions_exist_for_part_b_l2487_248787


namespace two_line_relationships_l2487_248709

-- Define a type for lines in a plane
def Line : Type := sorry

-- Define a plane
def Plane : Type := sorry

-- Define what it means for two lines to be in the same plane
def inSamePlane (l1 l2 : Line) (p : Plane) : Prop := sorry

-- Define what it means for two lines to be non-overlapping
def nonOverlapping (l1 l2 : Line) : Prop := sorry

-- Define what it means for two lines to intersect
def intersecting (l1 l2 : Line) : Prop := sorry

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- The theorem to be proved
theorem two_line_relationships (l1 l2 : Line) (p : Plane) :
  inSamePlane l1 l2 p → nonOverlapping l1 l2 → intersecting l1 l2 ∨ parallel l1 l2 := by
  sorry

end two_line_relationships_l2487_248709
