import Mathlib

namespace hyperbola_and_angle_bisector_l1200_120072

/-- A hyperbola with given properties -/
structure Hyperbola where
  -- Point A lies on the hyperbola
  point_A : ℝ × ℝ
  point_A_on_hyperbola : point_A.1^2 / 4 - point_A.2^2 / 12 = 1
  -- Eccentricity is 2
  eccentricity : ℝ
  eccentricity_eq : eccentricity = 2

/-- The angle bisector of ∠F₁AF₂ -/
def angle_bisector (h : Hyperbola) : ℝ → ℝ := 
  fun x ↦ 2 * x - 2

theorem hyperbola_and_angle_bisector (h : Hyperbola) 
  (h_point_A : h.point_A = (4, 6)) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / 12 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / 12 = 1}) ∧
  (∀ x : ℝ, angle_bisector h x = 2 * x - 2) := by
  sorry

end hyperbola_and_angle_bisector_l1200_120072


namespace part_one_part_two_l1200_120023

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Part I
theorem part_one : ∃ (m n : ℝ), a = (m • b.1 + n • c.1, m • b.2 + n • c.2) := by sorry

-- Part II
theorem part_two : 
  ∃ (d : ℝ × ℝ), 
    (∃ (k : ℝ), (d.1 - c.1, d.2 - c.2) = k • (a.1 + b.1, a.2 + b.2)) ∧ 
    (d.1 - c.1)^2 + (d.2 - c.2)^2 = 5 ∧
    (d = (3, -1) ∨ d = (5, 3)) := by sorry


end part_one_part_two_l1200_120023


namespace monotone_increasing_range_a_inequality_for_m_n_l1200_120044

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem monotone_increasing_range_a :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → Monotone (f a)) → a ≤ 2 := by sorry

theorem inequality_for_m_n :
  ∀ m n : ℝ, m ≠ n → (m - n) / (Real.log m - Real.log n) < (m + n) / 2 := by sorry

end monotone_increasing_range_a_inequality_for_m_n_l1200_120044


namespace polar_bear_club_time_l1200_120097

/-- Represents the time spent in the pool by each person -/
structure PoolTime where
  jerry : ℕ
  elaine : ℕ
  george : ℕ
  kramer : ℕ

/-- Calculates the total time spent in the pool -/
def total_time (pt : PoolTime) : ℕ :=
  pt.jerry + pt.elaine + pt.george + pt.kramer

/-- Theorem stating the total time spent in the pool is 11 minutes -/
theorem polar_bear_club_time : ∃ (pt : PoolTime),
  pt.jerry = 3 ∧
  pt.elaine = 2 * pt.jerry ∧
  pt.george = pt.elaine / 3 ∧
  pt.kramer = 0 ∧
  total_time pt = 11 := by
  sorry

end polar_bear_club_time_l1200_120097


namespace max_removable_edges_50x600_l1200_120018

/-- Represents a rectangular grid -/
structure RectangularGrid where
  rows : ℕ
  cols : ℕ

/-- Calculates the number of vertices in a rectangular grid -/
def vertexCount (grid : RectangularGrid) : ℕ :=
  (grid.rows + 1) * (grid.cols + 1)

/-- Calculates the number of edges in a rectangular grid -/
def edgeCount (grid : RectangularGrid) : ℕ :=
  grid.rows * (grid.cols + 1) + grid.cols * (grid.rows + 1)

/-- Calculates the maximum number of removable edges while keeping the graph connected -/
def maxRemovableEdges (grid : RectangularGrid) : ℕ :=
  edgeCount grid - (vertexCount grid - 1)

/-- Theorem: The maximum number of removable edges in a 50 × 600 grid is 30000 -/
theorem max_removable_edges_50x600 :
  maxRemovableEdges ⟨50, 600⟩ = 30000 := by
  sorry

end max_removable_edges_50x600_l1200_120018


namespace det_special_matrix_l1200_120010

/-- The determinant of the matrix [[x+2, x, x], [x, x+2, x], [x, x, x+2]] is equal to 8x + 8 for any real number x. -/
theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![x + 2, x, x; x, x + 2, x; x, x, x + 2] = 8 * x + 8 := by
  sorry

end det_special_matrix_l1200_120010


namespace min_omega_for_even_shifted_sine_l1200_120070

/-- Given a function g and a real number ω, this theorem states that
    if g is defined as g(x) = sin(ω(x - π/3) + π/6),
    ω is positive, and g is an even function,
    then the minimum value of ω is 2. -/
theorem min_omega_for_even_shifted_sine (g : ℝ → ℝ) (ω : ℝ) :
  (∀ x, g x = Real.sin (ω * (x - Real.pi / 3) + Real.pi / 6)) →
  ω > 0 →
  (∀ x, g x = g (-x)) →
  ω ≥ 2 ∧ ∃ ω₀, ω₀ = 2 ∧ 
    (∀ x, Real.sin (ω₀ * (x - Real.pi / 3) + Real.pi / 6) = 
          Real.sin (ω₀ * ((-x) - Real.pi / 3) + Real.pi / 6)) :=
by sorry

end min_omega_for_even_shifted_sine_l1200_120070


namespace prism_pyramid_volume_ratio_l1200_120058

/-- Given a triangular prism with height m, we extend a side edge by x to form a pyramid.
    The volume ratio k of the remaining part of the prism (outside the pyramid) to the original prism
    must be less than or equal to 3/4. -/
theorem prism_pyramid_volume_ratio (m : ℝ) (x : ℝ) (k : ℝ) 
  (h1 : m > 0) (h2 : x > 0) (h3 : k > 0) : k ≤ 3/4 := by
  sorry

end prism_pyramid_volume_ratio_l1200_120058


namespace hockey_league_season_games_l1200_120026

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) * m / 2

/-- Theorem: In a hockey league with 15 teams, where each team plays every other team 10 times,
    the total number of games played in the season is 1050. -/
theorem hockey_league_season_games :
  hockey_league_games 15 10 = 1050 := by
  sorry

end hockey_league_season_games_l1200_120026


namespace point_in_fourth_quadrant_l1200_120083

theorem point_in_fourth_quadrant (A B C : Real) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A < π/2 ∧ B < π/2 ∧ C < π/2) (h_triangle : A + B + C = π) :
  let P : Real × Real := (Real.sin A - Real.cos B, Real.cos A - Real.sin C)
  P.1 > 0 ∧ P.2 < 0 :=
by sorry

end point_in_fourth_quadrant_l1200_120083


namespace max_a_for_monotonic_f_l1200_120052

/-- Given a function f(x) = x^3 - ax that is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotonic_f (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, (x^3 - a*x) ≤ (y^3 - a*y)) → a ≤ 3 :=
sorry

end max_a_for_monotonic_f_l1200_120052


namespace coefficients_of_equation_l1200_120073

-- Define the coefficients of a quadratic equation
def QuadraticCoefficients := ℝ × ℝ × ℝ

-- Function to get coefficients from a quadratic equation
def getCoefficients (a b c : ℝ) : QuadraticCoefficients := (a, b, c)

-- Theorem stating that the coefficients of 2x^2 - 6x = 9 are (2, -6, -9)
theorem coefficients_of_equation : 
  let eq := fun x : ℝ => 2 * x^2 - 6 * x - 9
  getCoefficients 2 (-6) (-9) = (2, -6, -9) := by sorry

end coefficients_of_equation_l1200_120073


namespace max_odd_numbers_in_pyramid_l1200_120063

/-- Represents a number pyramid where each number above the bottom row
    is the sum of the two numbers immediately below it. -/
structure NumberPyramid where
  rows : Nat
  cells : Nat

/-- Represents the maximum number of odd numbers that can be placed in a number pyramid. -/
def maxOddNumbers (pyramid : NumberPyramid) : Nat :=
  14

/-- Theorem stating that the maximum number of odd numbers in a number pyramid is 14. -/
theorem max_odd_numbers_in_pyramid (pyramid : NumberPyramid) :
  maxOddNumbers pyramid = 14 := by
  sorry

end max_odd_numbers_in_pyramid_l1200_120063


namespace multiple_of_nine_implies_multiple_of_three_l1200_120074

theorem multiple_of_nine_implies_multiple_of_three (n : ℤ) :
  (∀ m : ℤ, 9 ∣ m → 3 ∣ m) →
  (∃ k : ℤ, n = 9 * k ∧ n % 2 = 1) →
  3 ∣ n :=
by sorry

end multiple_of_nine_implies_multiple_of_three_l1200_120074


namespace min_value_log_sum_equality_condition_l1200_120007

theorem min_value_log_sum (x : ℝ) (h : x > 1) :
  (Real.log 9 / Real.log x) + (Real.log x / Real.log 27) ≥ 2 * Real.sqrt 6 / 3 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 1) :
  (Real.log 9 / Real.log x) + (Real.log x / Real.log 27) = 2 * Real.sqrt 6 / 3 ↔ x = 3 * Real.sqrt 6 :=
by sorry

end min_value_log_sum_equality_condition_l1200_120007


namespace tan_identity_l1200_120020

theorem tan_identity : 
  (1 + Real.tan (28 * π / 180)) * (1 + Real.tan (17 * π / 180)) = 2 := by sorry

end tan_identity_l1200_120020


namespace sin_50_sin_70_minus_cos_50_sin_20_l1200_120096

open Real

theorem sin_50_sin_70_minus_cos_50_sin_20 :
  sin (50 * π / 180) * sin (70 * π / 180) - cos (50 * π / 180) * sin (20 * π / 180) = 1/2 := by
  sorry

end sin_50_sin_70_minus_cos_50_sin_20_l1200_120096


namespace cost_per_box_l1200_120040

/-- The cost per box for packaging a fine arts collection -/
theorem cost_per_box (box_volume : ℝ) (total_volume : ℝ) (total_cost : ℝ) : 
  box_volume = 20 * 20 * 15 →
  total_volume = 3060000 →
  total_cost = 663 →
  total_cost / (total_volume / box_volume) = 1.30 := by
  sorry

#eval (663 : ℚ) / ((3060000 : ℚ) / (20 * 20 * 15 : ℚ))

end cost_per_box_l1200_120040


namespace problem_solution_l1200_120017

theorem problem_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + 2 = d + Real.sqrt (a + b + c - 2*d) →
  d = -1/8 := by
sorry

end problem_solution_l1200_120017


namespace intersection_implies_x_zero_l1200_120077

def A : Set ℝ := {0, 1, 2, 4, 5}
def B (x : ℝ) : Set ℝ := {x-2, x, x+2}

theorem intersection_implies_x_zero (x : ℝ) (h : A ∩ B x = {0, 2}) : x = 0 := by
  sorry

end intersection_implies_x_zero_l1200_120077


namespace no_such_function_exists_l1200_120025

theorem no_such_function_exists : ¬∃ (f : ℝ → ℝ), 
  (∀ x, f x ≠ 0) ∧ 
  (∀ x, 2 * f (f x) = f x) ∧ 
  (∀ x, f x ≥ 0) ∧
  Differentiable ℝ f :=
by sorry

end no_such_function_exists_l1200_120025


namespace m_range_l1200_120024

-- Define propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the range of m
def range_m (m : ℝ) : Prop := m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3

-- Theorem statement
theorem m_range : 
  ∀ m : ℝ, (¬(P m ∧ Q m) ∧ (P m ∨ Q m)) → range_m m :=
sorry

end m_range_l1200_120024


namespace imaginary_part_sum_l1200_120071

theorem imaginary_part_sum (z₁ z₂ : ℂ) : z₁ = (1 : ℂ) / (-2 + Complex.I) ∧ z₂ = (1 : ℂ) / (1 - 2*Complex.I) →
  Complex.im (z₁ + z₂) = (1 : ℝ) / 5 := by
sorry

end imaginary_part_sum_l1200_120071


namespace workshop_workers_count_l1200_120043

theorem workshop_workers_count :
  let average_salary : ℕ := 9000
  let technician_count : ℕ := 7
  let technician_salary : ℕ := 12000
  let non_technician_salary : ℕ := 6000
  ∃ (total_workers : ℕ),
    total_workers * average_salary = 
      technician_count * technician_salary + 
      (total_workers - technician_count) * non_technician_salary ∧
    total_workers = 14 :=
by
  sorry

end workshop_workers_count_l1200_120043


namespace sqrt_equals_self_implies_zero_or_one_l1200_120013

theorem sqrt_equals_self_implies_zero_or_one (x : ℝ) : Real.sqrt x = x → x = 0 ∨ x = 1 := by
  sorry

end sqrt_equals_self_implies_zero_or_one_l1200_120013


namespace root_sum_pq_l1200_120081

theorem root_sum_pq (p q : ℝ) : 
  (2 * Complex.I ^ 2 + p * Complex.I + q = 0) →
  (2 * (-3 + 2 * Complex.I) ^ 2 + p * (-3 + 2 * Complex.I) + q = 0) →
  p + q = 38 := by
sorry

end root_sum_pq_l1200_120081


namespace quadratic_factorization_l1200_120059

theorem quadratic_factorization (x : ℂ) : 
  2 * x^2 + 8 * x + 26 = 2 * (x + 2 - 3 * I) * (x + 2 + 3 * I) := by
sorry

end quadratic_factorization_l1200_120059


namespace min_sum_with_constraints_l1200_120068

theorem min_sum_with_constraints (x y z w : ℕ+) (h : (4 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val ∧ (6 : ℕ) * z.val = (7 : ℕ) * w.val) :
  x.val + y.val + z.val + w.val ≥ 319 := by
  sorry

end min_sum_with_constraints_l1200_120068


namespace sqrt_sum_simplification_l1200_120084

theorem sqrt_sum_simplification : 
  Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 4 := by
  sorry

end sqrt_sum_simplification_l1200_120084


namespace line_through_points_m_plus_b_l1200_120027

/-- Given a line passing through points (1, 3) and (3, 7) that follows the equation y = mx + b,
    prove that m + b = 3 -/
theorem line_through_points_m_plus_b (m b : ℝ) : 
  (3 : ℝ) = m * (1 : ℝ) + b ∧ 
  (7 : ℝ) = m * (3 : ℝ) + b → 
  m + b = 3 := by
  sorry

end line_through_points_m_plus_b_l1200_120027


namespace solution_set_of_inequality_l1200_120049

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 1) / (x - 3) < 0 ↔ -1 < x ∧ x < 3 :=
sorry

end solution_set_of_inequality_l1200_120049


namespace bridget_apples_proof_l1200_120011

/-- The number of apples Bridget bought -/
def total_apples : ℕ := 21

/-- The number of apples Bridget gave to Cassie and Dan -/
def apples_to_cassie_and_dan : ℕ := 7

/-- The number of apples Bridget kept for herself -/
def apples_kept : ℕ := 7

theorem bridget_apples_proof :
  total_apples = 21 ∧
  (2 * total_apples) / 3 = apples_to_cassie_and_dan + apples_kept :=
by sorry

end bridget_apples_proof_l1200_120011


namespace quadratic_inequality_properties_l1200_120030

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ := {x | f a b c x > 0}

-- State the theorem
theorem quadratic_inequality_properties (a b c : ℝ) :
  solution_set a b c = Set.Ioo (-1/2 : ℝ) 2 →
  a < 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c > 0 := by
  sorry

end quadratic_inequality_properties_l1200_120030


namespace x_range_when_ln_x_negative_l1200_120045

theorem x_range_when_ln_x_negative (x : ℝ) (h : Real.log x < 0) : 0 < x ∧ x < 1 := by
  sorry

end x_range_when_ln_x_negative_l1200_120045


namespace sum_plus_ten_is_three_times_square_l1200_120036

theorem sum_plus_ten_is_three_times_square (n : ℤ) (h : n ≠ 0) : 
  ∃ (m : ℤ), (n - 1)^4 + n^4 + (n + 1)^4 + 10 = 3 * m^2 := by
  sorry

end sum_plus_ten_is_three_times_square_l1200_120036


namespace one_third_of_one_fourth_implies_three_tenths_l1200_120086

theorem one_third_of_one_fourth_implies_three_tenths (x : ℝ) : 
  (1 / 3) * (1 / 4) * x = 18 → (3 / 10) * x = 64.8 := by
sorry

end one_third_of_one_fourth_implies_three_tenths_l1200_120086


namespace problem_solution_l1200_120091

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x) + 2, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem problem_solution (A : ℝ) (b c : ℝ) (h1 : 0 ≤ A ∧ A ≤ π/4) 
  (h2 : f A = 4) (h3 : b = 1) (h4 : 1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) :
  (∀ x ∈ Set.Icc 0 (π/4), f x ≤ 5 ∧ 4 ≤ f x) ∧ 
  c^2 = 3 :=
sorry

end problem_solution_l1200_120091


namespace willow_catkin_diameter_scientific_notation_l1200_120038

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem willow_catkin_diameter_scientific_notation :
  toScientificNotation 0.0000105 = ScientificNotation.mk 1.05 (-5) (by sorry) :=
sorry

end willow_catkin_diameter_scientific_notation_l1200_120038


namespace probability_is_one_third_l1200_120022

/-- A game board consisting of an equilateral triangle divided into six smaller triangles -/
structure GameBoard where
  /-- The total number of smaller triangles in the game board -/
  total_triangles : ℕ
  /-- The number of shaded triangles in the game board -/
  shaded_triangles : ℕ
  /-- The shaded triangles are non-adjacent -/
  non_adjacent : Bool
  /-- The total number of triangles is 6 -/
  h_total : total_triangles = 6
  /-- The number of shaded triangles is 2 -/
  h_shaded : shaded_triangles = 2
  /-- The shaded triangles are indeed non-adjacent -/
  h_non_adjacent : non_adjacent = true

/-- The probability of a spinner landing in a shaded region -/
def probability (board : GameBoard) : ℚ :=
  board.shaded_triangles / board.total_triangles

/-- Theorem stating that the probability of landing in a shaded region is 1/3 -/
theorem probability_is_one_third (board : GameBoard) : probability board = 1/3 := by
  sorry

end probability_is_one_third_l1200_120022


namespace sum_of_coordinates_X_l1200_120000

/-- Given points Y and Z, and the condition that XZ/XY = ZY/XY = 1/3,
    prove that the sum of the coordinates of point X is 10. -/
theorem sum_of_coordinates_X (Y Z X : ℝ × ℝ) : 
  Y = (2, 8) →
  Z = (0, -4) →
  (X.1 - Z.1) / (X.1 - Y.1) = 1/3 →
  (X.2 - Z.2) / (X.2 - Y.2) = 1/3 →
  (Z.1 - Y.1) / (X.1 - Y.1) = 1/3 →
  (Z.2 - Y.2) / (X.2 - Y.2) = 1/3 →
  X.1 + X.2 = 10 := by
sorry

end sum_of_coordinates_X_l1200_120000


namespace smallest_integer_cubic_inequality_l1200_120099

theorem smallest_integer_cubic_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^3 - 12*m^2 + 44*m - 48 ≤ 0 → n ≤ m) ∧ 
  (n^3 - 12*n^2 + 44*n - 48 ≤ 0) ∧ n = 2 :=
by sorry

end smallest_integer_cubic_inequality_l1200_120099


namespace total_seats_is_28_l1200_120006

/-- The number of students per bus -/
def students_per_bus : ℝ := 14.0

/-- The number of buses -/
def number_of_buses : ℝ := 2.0

/-- The total number of seats taken up by students -/
def total_seats : ℝ := students_per_bus * number_of_buses

/-- Theorem stating that the total number of seats taken up by students is 28 -/
theorem total_seats_is_28 : total_seats = 28 := by
  sorry

end total_seats_is_28_l1200_120006


namespace wire_cut_square_octagon_ratio_l1200_120066

/-- The ratio of lengths when a wire is cut to form a square and an octagon with equal areas -/
theorem wire_cut_square_octagon_ratio (a b : ℝ) (h : a > 0) (k : b > 0) : 
  (a^2 / 16 = b^2 * (1 + Real.sqrt 2) / 32) → 
  (a / b = Real.sqrt ((1 + Real.sqrt 2) / 2)) := by
  sorry


end wire_cut_square_octagon_ratio_l1200_120066


namespace range_of_m_l1200_120088

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - 4| ≤ 6 → x ≤ 1 + m) ∧ 
  ¬(x ≤ 1 + m → |x - 4| ≤ 6)) → 
  m ∈ Set.Ici 9 := by
sorry

end range_of_m_l1200_120088


namespace calculate_markup_l1200_120035

/-- Calculate the markup for an article given its purchase price, overhead percentage, and desired net profit. -/
theorem calculate_markup (purchase_price overhead_percent net_profit : ℚ) 
  (h1 : purchase_price = 48)
  (h2 : overhead_percent = 10 / 100)
  (h3 : net_profit = 12) :
  purchase_price * overhead_percent + purchase_price + net_profit - purchase_price = 168 / 10 := by
  sorry

end calculate_markup_l1200_120035


namespace consecutive_numbers_sum_l1200_120016

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 60) → n = 10 := by
  sorry

end consecutive_numbers_sum_l1200_120016


namespace prob_not_losing_l1200_120037

/-- Given a chess game between players A and B, this theorem proves
    the probability of A not losing, given the probabilities of a draw
    and A winning. -/
theorem prob_not_losing (p_draw p_win : ℝ) : 
  p_draw = 1/2 → p_win = 1/3 → p_draw + p_win = 5/6 := by
  sorry

end prob_not_losing_l1200_120037


namespace locus_of_parabola_vertices_l1200_120033

/-- The locus of vertices of parabolas -/
theorem locus_of_parabola_vertices
  (a c : ℝ) (hz : a > 0) (hc : c > 0) :
  ∀ (z : ℝ), ∃ (x_z y_z : ℝ),
    (x_z = -z / (2 * a)) ∧
    (y_z = a * x_z^2 + z * x_z + c) ∧
    (y_z = -a * x_z^2 + c) :=
by sorry

end locus_of_parabola_vertices_l1200_120033


namespace journey_time_increase_l1200_120090

theorem journey_time_increase (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  ((second_half_time - first_half_time) / first_half_time) * 100 = 200 := by
sorry

end journey_time_increase_l1200_120090


namespace condition_relationship_l1200_120076

theorem condition_relationship : 
  let A := {x : ℝ | 0 < x ∧ x < 5}
  let B := {x : ℝ | |x - 2| < 3}
  (∀ x ∈ A, x ∈ B) ∧ (∃ x ∈ B, x ∉ A) := by sorry

end condition_relationship_l1200_120076


namespace lcm_problem_l1200_120003

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 36 := by
  sorry

end lcm_problem_l1200_120003


namespace infinitely_many_solutions_l1200_120062

theorem infinitely_many_solutions (c : ℝ) : 
  (∀ x : ℝ, 3 * (5 + c * x) = 18 * x + 15) ↔ c = 6 :=
by sorry

end infinitely_many_solutions_l1200_120062


namespace mothers_biscuits_l1200_120012

/-- Represents the number of biscuits in Randy's scenario -/
structure BiscuitCount where
  initial : Nat
  fromFather : Nat
  fromMother : Nat
  eatenByBrother : Nat
  final : Nat

/-- Calculates the total number of biscuits Randy had before his brother ate some -/
def totalBeforeEating (b : BiscuitCount) : Nat :=
  b.initial + b.fromFather + b.fromMother

/-- Theorem: Randy's mother gave him 15 biscuits -/
theorem mothers_biscuits (b : BiscuitCount) 
  (h1 : b.initial = 32)
  (h2 : b.fromFather = 13)
  (h3 : b.eatenByBrother = 20)
  (h4 : b.final = 40)
  (h5 : totalBeforeEating b = b.final + b.eatenByBrother) : 
  b.fromMother = 15 := by
  sorry

#check mothers_biscuits

end mothers_biscuits_l1200_120012


namespace pencils_remaining_l1200_120055

theorem pencils_remaining (initial_pencils : ℕ) (removed_pencils : ℕ) 
  (h1 : initial_pencils = 87) 
  (h2 : removed_pencils = 4) : 
  initial_pencils - removed_pencils = 83 := by
  sorry

end pencils_remaining_l1200_120055


namespace stonewall_band_max_members_l1200_120009

theorem stonewall_band_max_members :
  ∃ (max : ℕ),
    (∀ n : ℕ, (30 * n) % 34 = 2 → 30 * n < 1500 → 30 * n ≤ max) ∧
    (∃ n : ℕ, (30 * n) % 34 = 2 ∧ 30 * n < 1500 ∧ 30 * n = max) ∧
    max = 1260 := by
  sorry

end stonewall_band_max_members_l1200_120009


namespace range_of_a_l1200_120056

-- Define the inequality as a function of x and a
def inequality (x a : ℝ) : Prop := 2 * x^2 + a * x - a^2 > 0

-- Define the theorem
theorem range_of_a : 
  (∃ a : ℝ, inequality 2 a) → 
  (∀ a : ℝ, inequality 2 a ↔ -2 < a ∧ a < 4) :=
by sorry

end range_of_a_l1200_120056


namespace expression_simplification_l1200_120092

theorem expression_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((x + 1) / (x^2 - 4) - 1 / (x + 2)) / (3 / (x - 2)) = 1 / (x + 2) :=
by sorry

end expression_simplification_l1200_120092


namespace simplify_expression_l1200_120087

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (97 * x + 30) = 100 * x + 50 := by
  sorry

end simplify_expression_l1200_120087


namespace function_bounds_l1200_120078

theorem function_bounds (x y z : ℝ) 
  (h1 : -1 ≤ 2*x + y - z ∧ 2*x + y - z ≤ 8)
  (h2 : 2 ≤ x - y + z ∧ x - y + z ≤ 9)
  (h3 : -3 ≤ x + 2*y - z ∧ x + 2*y - z ≤ 7) :
  -6 ≤ 7*x + 5*y - 2*z ∧ 7*x + 5*y - 2*z ≤ 47 := by
sorry

end function_bounds_l1200_120078


namespace largest_sum_l1200_120001

theorem largest_sum : 
  let sum1 := (1/4 : ℚ) + (1/5 : ℚ)
  let sum2 := (1/4 : ℚ) + (1/6 : ℚ)
  let sum3 := (1/4 : ℚ) + (1/3 : ℚ)
  let sum4 := (1/4 : ℚ) + (1/8 : ℚ)
  let sum5 := (1/4 : ℚ) + (1/7 : ℚ)
  sum3 > sum1 ∧ sum3 > sum2 ∧ sum3 > sum4 ∧ sum3 > sum5 := by
  sorry

end largest_sum_l1200_120001


namespace complex_power_one_minus_i_six_l1200_120089

theorem complex_power_one_minus_i_six :
  (1 - Complex.I : ℂ)^6 = 8 * Complex.I :=
by sorry

end complex_power_one_minus_i_six_l1200_120089


namespace burger_expenditure_l1200_120028

theorem burger_expenditure (total : ℝ) (movie_frac ice_cream_frac music_frac : ℚ) :
  total = 50 ∧
  movie_frac = 1/4 ∧
  ice_cream_frac = 1/6 ∧
  music_frac = 1/3 →
  total - (movie_frac + ice_cream_frac + music_frac) * total = 12.5 := by
  sorry

end burger_expenditure_l1200_120028


namespace unique_number_meeting_conditions_l1200_120079

theorem unique_number_meeting_conditions : ∃! n : ℕ+, 
  (((n < 12) ∨ (¬ 7 ∣ n) ∨ (5 * n < 70)) ∧ 
   ¬((n < 12) ∧ (¬ 7 ∣ n) ∧ (5 * n < 70))) ∧
  (((12 * n > 1000) ∨ (10 ∣ n) ∨ (n > 100)) ∧ 
   ¬((12 * n > 1000) ∧ (10 ∣ n) ∧ (n > 100))) ∧
  (((4 ∣ n) ∨ (11 * n < 1000) ∨ (9 ∣ n)) ∧ 
   ¬((4 ∣ n) ∧ (11 * n < 1000) ∧ (9 ∣ n))) ∧
  (((n < 20) ∨ Nat.Prime n ∨ (7 ∣ n)) ∧ 
   ¬((n < 20) ∧ Nat.Prime n ∧ (7 ∣ n))) ∧
  n = 89 := by
  sorry

end unique_number_meeting_conditions_l1200_120079


namespace roots_difference_squared_l1200_120093

theorem roots_difference_squared (p q r s : ℝ) : 
  (r^2 - p*r + q = 0) → (s^2 - p*s + q = 0) → (r - s)^2 = p^2 - 4*q := by
  sorry

end roots_difference_squared_l1200_120093


namespace sum_extension_terms_l1200_120065

theorem sum_extension_terms (k : ℕ) (hk : k > 1) : 
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k :=
sorry

end sum_extension_terms_l1200_120065


namespace equation_solutions_l1200_120004

def equation (x : ℝ) : Prop :=
  1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
  1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 10

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = 10 ∨ x = -3.5 := by sorry

end equation_solutions_l1200_120004


namespace perpendicular_lines_m_value_l1200_120085

/-- Given two lines l₁ and l₂, prove that if they are perpendicular, then m = 1/2 -/
theorem perpendicular_lines_m_value (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | m * x + y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | x + (m - 1) * y + 2 = 0}
  (∀ (p₁ p₂ q₁ q₂ : ℝ × ℝ), p₁ ∈ l₁ → p₂ ∈ l₁ → q₁ ∈ l₂ → q₂ ∈ l₂ → 
    p₁ ≠ p₂ → q₁ ≠ q₂ → (p₁.1 - p₂.1) * (q₁.1 - q₂.1) + (p₁.2 - p₂.2) * (q₁.2 - q₂.2) = 0) →
  m = 1 / 2 :=
sorry

end perpendicular_lines_m_value_l1200_120085


namespace prob_B_wins_third_round_correct_A_has_lower_expected_additional_time_l1200_120050

-- Define the probabilities of answering correctly for each participant
def prob_correct_A : ℚ := 3/5
def prob_correct_B : ℚ := 2/3

-- Define the number of rounds and questions per round
def num_rounds : ℕ := 3
def questions_per_round : ℕ := 3

-- Define the time penalty for an incorrect answer
def time_penalty : ℕ := 20

-- Define the time difference in recitation per round
def recitation_time_diff : ℕ := 10

-- Define the function to calculate the probability of B winning in the third round
def prob_B_wins_third_round : ℚ := 448/3375

-- Define the expected number of incorrect answers for each participant
def expected_incorrect_A : ℚ := (1 - prob_correct_A) * (num_rounds * questions_per_round : ℚ)
def expected_incorrect_B : ℚ := (1 - prob_correct_B) * (num_rounds * questions_per_round : ℚ)

-- Theorem: The probability of B winning in the third round is 448/3375
theorem prob_B_wins_third_round_correct :
  prob_B_wins_third_round = 448/3375 := by sorry

-- Theorem: A has a lower expected additional time due to incorrect answers
theorem A_has_lower_expected_additional_time :
  expected_incorrect_A * time_penalty < expected_incorrect_B * time_penalty + (num_rounds * recitation_time_diff : ℚ) := by sorry

end prob_B_wins_third_round_correct_A_has_lower_expected_additional_time_l1200_120050


namespace max_xy_value_l1200_120032

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 9*y = 12) :
  ∀ z : ℝ, z = x * y → z ≤ 4 :=
by
  sorry

end max_xy_value_l1200_120032


namespace initial_cats_is_28_l1200_120008

/-- Represents the animal shelter scenario --/
structure AnimalShelter where
  initialDogs : ℕ
  initialLizards : ℕ
  dogAdoptionRate : ℚ
  catAdoptionRate : ℚ
  lizardAdoptionRate : ℚ
  newPetsPerMonth : ℕ
  totalPetsAfterOneMonth : ℕ

/-- Calculates the initial number of cats in the shelter --/
def calculateInitialCats (shelter : AnimalShelter) : ℚ :=
  let remainingDogs : ℚ := shelter.initialDogs * (1 - shelter.dogAdoptionRate)
  let remainingLizards : ℚ := shelter.initialLizards * (1 - shelter.lizardAdoptionRate)
  let nonCatPets : ℚ := remainingDogs + remainingLizards + shelter.newPetsPerMonth
  let remainingCats : ℚ := shelter.totalPetsAfterOneMonth - nonCatPets
  remainingCats / (1 - shelter.catAdoptionRate)

/-- Theorem stating that the initial number of cats is 28 --/
theorem initial_cats_is_28 (shelter : AnimalShelter) 
  (h1 : shelter.initialDogs = 30)
  (h2 : shelter.initialLizards = 20)
  (h3 : shelter.dogAdoptionRate = 1/2)
  (h4 : shelter.catAdoptionRate = 1/4)
  (h5 : shelter.lizardAdoptionRate = 1/5)
  (h6 : shelter.newPetsPerMonth = 13)
  (h7 : shelter.totalPetsAfterOneMonth = 65) :
  calculateInitialCats shelter = 28 := by
  sorry

#eval calculateInitialCats {
  initialDogs := 30,
  initialLizards := 20,
  dogAdoptionRate := 1/2,
  catAdoptionRate := 1/4,
  lizardAdoptionRate := 1/5,
  newPetsPerMonth := 13,
  totalPetsAfterOneMonth := 65
}

end initial_cats_is_28_l1200_120008


namespace interchanged_digits_theorem_l1200_120042

theorem interchanged_digits_theorem (n m a b : ℕ) : 
  n = 10 * a + b → 
  n = m * (a + b + a) → 
  10 * b + a = (9 - m) * (a + b) := by
  sorry

end interchanged_digits_theorem_l1200_120042


namespace linda_remaining_candies_l1200_120060

/-- The number of candies Linda has left after giving some away -/
def candies_left (initial : ℝ) (given_away : ℝ) : ℝ := initial - given_away

/-- Theorem stating that Linda's remaining candies is the difference between initial and given away -/
theorem linda_remaining_candies (initial : ℝ) (given_away : ℝ) :
  candies_left initial given_away = initial - given_away :=
by sorry

end linda_remaining_candies_l1200_120060


namespace longest_non_decreasing_subsequence_12022_l1200_120051

/-- Represents a natural number as a list of its digits. -/
def digits_of (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 10) :: aux (m / 10)
  (aux n).reverse

/-- Computes the length of the longest non-decreasing subsequence in a list. -/
def longest_non_decreasing_subsequence_length (l : List ℕ) : ℕ :=
  let rec aux (prev : ℕ) (current : List ℕ) (acc : ℕ) : ℕ :=
    match current with
    | [] => acc
    | x :: xs => if x ≥ prev then aux x xs (acc + 1) else aux prev xs acc
  aux 0 l 0

/-- The theorem stating that the longest non-decreasing subsequence of digits in 12022 has length 3. -/
theorem longest_non_decreasing_subsequence_12022 :
  longest_non_decreasing_subsequence_length (digits_of 12022) = 3 := by
  sorry

end longest_non_decreasing_subsequence_12022_l1200_120051


namespace digit_concatenation_divisibility_l1200_120029

theorem digit_concatenation_divisibility (n : ℕ) (a : ℕ) (h1 : n > 1) (h2 : 10^(n-1) ≤ a) (h3 : a < 10^n) :
  let b := a * (10^n + 1)
  ∃! k : ℕ, k > 1 ∧ k ≤ 10 ∧ b = k * a^2 ∧ k = 7 :=
sorry

end digit_concatenation_divisibility_l1200_120029


namespace problem_1_problem_2_l1200_120061

-- Problem 1
theorem problem_1 : |Real.sqrt 3 - 2| + (3 - Real.pi)^0 - Real.sqrt 12 + 6 * Real.cos (30 * π / 180) = 3 := by sorry

-- Problem 2
theorem problem_2 : (1 / ((-5)^2 - 3*(-5))) / (2 / ((-5)^2 - 9)) = 1/5 := by sorry

end problem_1_problem_2_l1200_120061


namespace geometric_series_product_l1200_120002

theorem geometric_series_product (y : ℝ) : 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/y)^n → y = 9 := by
  sorry

end geometric_series_product_l1200_120002


namespace gilbert_parsley_count_l1200_120069

/-- Represents the number of herb plants Gilbert had at different stages of spring. -/
structure HerbCount where
  initial_basil : ℕ
  initial_parsley : ℕ
  initial_mint : ℕ
  final_basil : ℕ
  final_total : ℕ

/-- The conditions of Gilbert's herb garden during spring. -/
def spring_garden_conditions : HerbCount where
  initial_basil := 3
  initial_parsley := 0  -- We'll prove this is 1
  initial_mint := 2
  final_basil := 4
  final_total := 5

/-- Theorem stating that Gilbert planted 1 parsley plant initially. -/
theorem gilbert_parsley_count :
  spring_garden_conditions.initial_parsley = 1 :=
by sorry

end gilbert_parsley_count_l1200_120069


namespace larger_triangle_perimeter_l1200_120054

/-- Given an isosceles triangle with side lengths 7, 7, and 12, and a similar triangle
    with longest side 36, the perimeter of the larger triangle is 78. -/
theorem larger_triangle_perimeter (a b c : ℝ) (d : ℝ) : 
  a = 7 ∧ b = 7 ∧ c = 12 ∧ d = 36 ∧ 
  (a = b) ∧ (c ≥ a) ∧ (c ≥ b) ∧
  (d / c = 36 / 12) →
  d + (d * a / c) + (d * b / c) = 78 := by
  sorry


end larger_triangle_perimeter_l1200_120054


namespace definite_integral_2x_l1200_120082

theorem definite_integral_2x : ∫ x in (1:ℝ)..2, 2*x = 3 := by sorry

end definite_integral_2x_l1200_120082


namespace ny_striploin_cost_l1200_120039

theorem ny_striploin_cost (total_bill : ℝ) (tax_rate : ℝ) (wine_cost : ℝ) (gratuities : ℝ) :
  total_bill = 140 →
  tax_rate = 0.1 →
  wine_cost = 10 →
  gratuities = 41 →
  ∃ (striploin_cost : ℝ), abs (striploin_cost - 71.82) < 0.01 :=
by
  sorry

end ny_striploin_cost_l1200_120039


namespace quadratic_minimum_l1200_120064

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x + 5 ≥ -4 ∧ ∃ y : ℝ, y^2 + 6*y + 5 = -4 := by
  sorry

end quadratic_minimum_l1200_120064


namespace chess_group_players_l1200_120095

theorem chess_group_players (n : ℕ) : 
  (∀ i j : Fin n, i ≠ j → ∃! game : ℕ, game ≤ 36) →  -- Each player plays each other once
  (∀ game : ℕ, game ≤ 36 → ∃! i j : Fin n, i ≠ j) →  -- Each game is played by two distinct players
  (Nat.choose n 2 = 36) →                            -- Total number of games is 36
  n = 9 := by
sorry

end chess_group_players_l1200_120095


namespace units_digit_sum_base7_l1200_120041

/-- Represents a number in base 7 --/
def Base7 : Type := ℕ

/-- Converts a natural number to its representation in base 7 --/
def toBase7 (n : ℕ) : Base7 := sorry

/-- Adds two numbers in base 7 --/
def addBase7 (a b : Base7) : Base7 := sorry

/-- Gets the units digit of a number in base 7 --/
def unitsDigitBase7 (n : Base7) : ℕ := sorry

theorem units_digit_sum_base7 :
  unitsDigitBase7 (addBase7 (toBase7 65) (toBase7 34)) = 2 := by
  sorry

end units_digit_sum_base7_l1200_120041


namespace least_subtrahend_l1200_120034

def problem (n : ℕ) : Prop :=
  (2590 - n) % 9 = 6 ∧ 
  (2590 - n) % 11 = 6 ∧ 
  (2590 - n) % 13 = 6

theorem least_subtrahend : 
  problem 10 ∧ ∀ m : ℕ, m < 10 → ¬(problem m) :=
by sorry

end least_subtrahend_l1200_120034


namespace cheaper_call_rate_l1200_120098

/-- China Mobile's promotion factor -/
def china_mobile_promotion : ℚ := 130 / 100

/-- China Telecom's promotion factor -/
def china_telecom_promotion : ℚ := 100 / 40

/-- China Mobile's standard call rate (yuan per minute) -/
def china_mobile_standard_rate : ℚ := 26 / 100

/-- China Telecom's standard call rate (yuan per minute) -/
def china_telecom_standard_rate : ℚ := 30 / 100

/-- China Mobile's actual call rate (yuan per minute) -/
def china_mobile_actual_rate : ℚ := china_mobile_standard_rate / china_mobile_promotion

/-- China Telecom's actual call rate (yuan per minute) -/
def china_telecom_actual_rate : ℚ := china_telecom_standard_rate / china_telecom_promotion

theorem cheaper_call_rate :
  china_telecom_actual_rate < china_mobile_actual_rate ∧
  china_mobile_actual_rate - china_telecom_actual_rate = 8 / 100 := by
  sorry

end cheaper_call_rate_l1200_120098


namespace least_subtraction_for_divisibility_by_11_l1200_120075

theorem least_subtraction_for_divisibility_by_11 : 
  ∃ (x : ℕ), x = 7 ∧ 
  (∀ (y : ℕ), y < x → ¬(11 ∣ (427398 - y))) ∧
  (11 ∣ (427398 - x)) := by
sorry

end least_subtraction_for_divisibility_by_11_l1200_120075


namespace unit_vector_of_a_l1200_120019

/-- Given a vector a = (2, √5), prove that its unit vector is (2/3, √5/3) -/
theorem unit_vector_of_a (a : ℝ × ℝ) (h : a = (2, Real.sqrt 5)) :
  let norm_a := Real.sqrt (a.1^2 + a.2^2)
  (a.1 / norm_a, a.2 / norm_a) = (2/3, Real.sqrt 5 / 3) := by
  sorry

end unit_vector_of_a_l1200_120019


namespace selling_cheat_theorem_l1200_120021

/-- Represents a shop owner's pricing strategy -/
structure ShopOwner where
  buying_cheat_percent : ℝ
  profit_percent : ℝ

/-- Calculates the percentage by which the shop owner cheats while selling -/
def selling_cheat_percent (owner : ShopOwner) : ℝ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating the selling cheat percentage for a specific shop owner -/
theorem selling_cheat_theorem (owner : ShopOwner) 
  (h1 : owner.buying_cheat_percent = 12)
  (h2 : owner.profit_percent = 60) :
  selling_cheat_percent owner = 60 := by
  sorry

end selling_cheat_theorem_l1200_120021


namespace muffin_distribution_l1200_120048

theorem muffin_distribution (total_students : ℕ) (absent_students : ℕ) (extra_muffins : ℕ) : 
  total_students = 400 →
  absent_students = 180 →
  extra_muffins = 36 →
  (total_students * ((total_students - absent_students) * extra_muffins + total_students * (total_students - absent_students))) / 
  ((total_students - absent_students) * total_students) = 80 := by
sorry

end muffin_distribution_l1200_120048


namespace systematic_sampling_proof_l1200_120094

/-- Represents a student with an ID number -/
structure Student where
  id : Nat
  deriving Repr

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Lottery
  deriving Repr

/-- Checks if a number is divisible by 5 -/
def isDivisibleByFive (n : Nat) : Bool :=
  n % 5 == 0

/-- Selects students whose IDs are divisible by 5 -/
def selectStudents (students : List Student) : List Student :=
  students.filter (fun s => isDivisibleByFive s.id)

/-- Theorem: Selecting students with IDs divisible by 5 from a group of 60 students
    numbered 1 to 60 is an example of systematic sampling -/
theorem systematic_sampling_proof (students : List Student) 
    (h1 : students.length = 60)
    (h2 : ∀ i, 1 ≤ i ∧ i ≤ 60 → ∃ s ∈ students, s.id = i)
    (h3 : ∀ s ∈ students, 1 ≤ s.id ∧ s.id ≤ 60) :
    (selectStudents students).length = 12 ∧ 
    SamplingMethod.Systematic = 
      (match (selectStudents students) with
       | [] => SamplingMethod.SimpleRandom  -- Default case, should not occur
       | _ => SamplingMethod.Systematic) := by
  sorry

end systematic_sampling_proof_l1200_120094


namespace class_size_is_fifteen_l1200_120031

-- Define the number of students
def N : ℕ := sorry

-- Define the average age function
def averageAge (numStudents : ℕ) (totalAge : ℕ) : ℚ :=
  totalAge / numStudents

-- Theorem statement
theorem class_size_is_fifteen :
  -- Conditions
  (averageAge (N - 1) (15 * (N - 1)) = 15) →
  (averageAge 4 (14 * 4) = 14) →
  (averageAge 9 (16 * 9) = 16) →
  -- Conclusion
  N = 15 := by
  sorry

end class_size_is_fifteen_l1200_120031


namespace smallest_n_for_book_pricing_l1200_120005

theorem smallest_n_for_book_pricing : 
  ∀ n : ℕ+, (∃ x : ℕ+, (105 * x : ℕ) = 100 * n) → n ≥ 21 :=
sorry

end smallest_n_for_book_pricing_l1200_120005


namespace divisibility_implication_l1200_120067

theorem divisibility_implication (a b : ℤ) :
  (∃ k : ℤ, a^2 + 9*a*b + b^2 = 11*k) → (∃ m : ℤ, a^2 - b^2 = 11*m) := by
  sorry

end divisibility_implication_l1200_120067


namespace sequence_properties_l1200_120053

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Definition of a geometric sequence -/
def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ+, a (n + 1) = a n * q

/-- Definition of the sum of the first n terms -/
def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- Main theorem -/
theorem sequence_properties (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (is_arithmetic_sequence a ∧ is_geometric_sequence a) → a n = a (n + 1)) ∧
  (∃ α β : ℝ, ∀ n : ℕ+, S a n = α * n^2 + β * n) → is_arithmetic_sequence a ∧
  (∀ n : ℕ+, S a n = 1 - (-1)^(n : ℕ)) → is_geometric_sequence a :=
sorry

end sequence_properties_l1200_120053


namespace fourteenth_root_of_unity_l1200_120080

theorem fourteenth_root_of_unity : 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (4 * π / 14)) := by
  sorry

end fourteenth_root_of_unity_l1200_120080


namespace quadratic_equation_roots_algebraic_expression_value_l1200_120015

-- Part 1
theorem quadratic_equation_roots (x : ℝ) :
  x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

-- Part 2
theorem algebraic_expression_value (a : ℝ) :
  a^2 = 3*a + 10 → (a + 4) * (a - 4) - 3 * (a - 1) = -3 := by
  sorry

end quadratic_equation_roots_algebraic_expression_value_l1200_120015


namespace shower_water_reduction_l1200_120046

theorem shower_water_reduction 
  (original_time original_rate : ℝ) 
  (new_time : ℝ := 3/4 * original_time) 
  (new_rate : ℝ := 3/4 * original_rate) : 
  1 - (new_time * new_rate) / (original_time * original_rate) = 7/16 := by
sorry

end shower_water_reduction_l1200_120046


namespace lines_are_skew_l1200_120057

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the properties of lines
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem lines_are_skew (a b : Line) : 
  (¬ parallel a b) → (¬ intersect a b) → skew a b :=
by sorry

end lines_are_skew_l1200_120057


namespace base_conversion_subtraction_l1200_120047

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

theorem base_conversion_subtraction :
  let base_9_number := [4, 2, 3]  -- 324 in base 9 (least significant digit first)
  let base_6_number := [6, 5, 1]  -- 156 in base 6 (least significant digit first)
  (to_base_10 base_9_number 9) - (to_base_10 base_6_number 6) = 193 := by
sorry

end base_conversion_subtraction_l1200_120047


namespace proper_subset_implies_a_geq_two_l1200_120014

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

theorem proper_subset_implies_a_geq_two (a : ℝ) :
  A ⊂ B a → a ≥ 2 := by sorry

end proper_subset_implies_a_geq_two_l1200_120014
