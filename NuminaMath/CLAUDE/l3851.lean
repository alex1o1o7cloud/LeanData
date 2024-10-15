import Mathlib

namespace NUMINAMATH_CALUDE_second_group_size_l3851_385126

theorem second_group_size :
  ∀ (n : ℕ), 
  -- First group has 20 students with average height 20 cm
  (20 : ℝ) * 20 = 400 ∧
  -- Second group has n students with average height 20 cm
  (n : ℝ) * 20 = 20 * n ∧
  -- Combined group has 31 students with average height 20 cm
  (31 : ℝ) * 20 = 620 ∧
  -- Total height of combined groups equals sum of individual group heights
  400 + 20 * n = 620
  →
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_second_group_size_l3851_385126


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3851_385141

/-- The intersection point of two lines -/
def intersection_point (m1 a1 m2 a2 : ℚ) : ℚ × ℚ :=
  let x := (a2 - a1) / (m1 - m2)
  let y := m1 * x + a1
  (x, y)

/-- First line: y = 3x -/
def line1 (x : ℚ) : ℚ := 3 * x

/-- Second line: y + 6 = -9x, or y = -9x - 6 -/
def line2 (x : ℚ) : ℚ := -9 * x - 6

theorem intersection_of_lines :
  intersection_point 3 0 (-9) (-6) = (-1/2, -3/2) ∧
  line1 (-1/2) = -3/2 ∧
  line2 (-1/2) = -3/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3851_385141


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l3851_385166

/-- Given a line y = kx tangent to y = ln x and passing through the origin, k = 1/e -/
theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = 1 / x) → 
  k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l3851_385166


namespace NUMINAMATH_CALUDE_unique_symmetric_matrix_condition_l3851_385145

/-- A symmetric 2x2 matrix with real entries -/
structure SymmetricMatrix2x2 where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The trace of a symmetric 2x2 matrix -/
def trace (M : SymmetricMatrix2x2) : ℝ := M.x + M.z

/-- The determinant of a symmetric 2x2 matrix -/
def det (M : SymmetricMatrix2x2) : ℝ := M.x * M.z - M.y * M.y

/-- The main theorem -/
theorem unique_symmetric_matrix_condition (a b : ℝ) :
  (∃! M : SymmetricMatrix2x2, trace M = a ∧ det M = b) ↔ ∃ t : ℝ, a = 2 * t ∧ b = t ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_symmetric_matrix_condition_l3851_385145


namespace NUMINAMATH_CALUDE_trajectory_forms_two_rays_l3851_385105

/-- The trajectory of a point P(x, y) with a constant difference of 2 in its distances to points M(1, 0) and N(3, 0) forms two rays. -/
theorem trajectory_forms_two_rays :
  ∀ (x y : ℝ),
  |((x - 1)^2 + y^2).sqrt - ((x - 3)^2 + y^2).sqrt| = 2 →
  ∃ (a b : ℝ), y = a * x + b ∨ y = -a * x + b :=
by sorry

end NUMINAMATH_CALUDE_trajectory_forms_two_rays_l3851_385105


namespace NUMINAMATH_CALUDE_proportion_solution_l3851_385137

theorem proportion_solution (x : ℝ) : (0.60 / x = 6 / 4) → x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3851_385137


namespace NUMINAMATH_CALUDE_polygon_count_l3851_385189

theorem polygon_count (n : ℕ) (h : n = 15) : 
  2^n - (n.choose 0 + n.choose 1 + n.choose 2 + n.choose 3) = 32192 :=
by sorry

end NUMINAMATH_CALUDE_polygon_count_l3851_385189


namespace NUMINAMATH_CALUDE_biography_increase_l3851_385114

theorem biography_increase (T : ℝ) (h1 : T > 0) : 
  let initial_bio := 0.20 * T
  let final_bio := 0.32 * T
  (final_bio - initial_bio) / initial_bio = 0.60
  := by sorry

end NUMINAMATH_CALUDE_biography_increase_l3851_385114


namespace NUMINAMATH_CALUDE_inequality_minimum_a_l3851_385116

theorem inequality_minimum_a (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → x * y ≤ a * x^2 + 2 * y^2) → 
  a ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_minimum_a_l3851_385116


namespace NUMINAMATH_CALUDE_mixed_games_count_l3851_385197

/-- Represents a chess competition with men and women players -/
structure ChessCompetition where
  womenCount : ℕ
  menCount : ℕ
  womenGames : ℕ
  menGames : ℕ

/-- Calculates the number of games between a man and a woman -/
def mixedGames (c : ChessCompetition) : ℕ :=
  c.womenCount * c.menCount

/-- Theorem stating the relationship between the number of games -/
theorem mixed_games_count (c : ChessCompetition) 
  (h1 : c.womenGames = 45)
  (h2 : c.menGames = 190)
  (h3 : c.womenGames = c.womenCount * (c.womenCount - 1) / 2)
  (h4 : c.menGames = c.menCount * (c.menCount - 1) / 2) :
  mixedGames c = 200 := by
  sorry


end NUMINAMATH_CALUDE_mixed_games_count_l3851_385197


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l3851_385184

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_b (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_of_b b n + b (n + 1)

theorem arithmetic_sequence_m_value
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a3 : a 3 = 5)
  (h_a9 : a 9 = 17)
  (h_sum_b : ∀ n : ℕ, sum_of_b b n = 3^n - 1)
  (h_relation : ∃ m : ℕ, m > 0 ∧ 1 + a m = b 4) :
  ∃ m : ℕ, m > 0 ∧ 1 + a m = b 4 ∧ m = 27 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l3851_385184


namespace NUMINAMATH_CALUDE_derivative_y_l3851_385162

noncomputable def y (x : ℝ) : ℝ :=
  (1/2) * Real.tanh x + (1/(4*Real.sqrt 2)) * Real.log ((1 + Real.sqrt 2 * Real.tanh x) / (1 - Real.sqrt 2 * Real.tanh x))

theorem derivative_y (x : ℝ) :
  deriv y x = 1 / (Real.cosh x ^ 2 * (1 - Real.sinh x ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_derivative_y_l3851_385162


namespace NUMINAMATH_CALUDE_prob_at_least_one_red_l3851_385199

/-- Probability of drawing at least one red ball in three independent draws with replacement -/
theorem prob_at_least_one_red : 
  let total_balls : ℕ := 2
  let red_balls : ℕ := 1
  let num_draws : ℕ := 3
  let prob_blue : ℚ := 1 / 2
  let prob_all_blue : ℚ := prob_blue ^ num_draws
  prob_all_blue = 1 / 8 ∧ (1 : ℚ) - prob_all_blue = 7 / 8 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_red_l3851_385199


namespace NUMINAMATH_CALUDE_supplementary_angle_theorem_l3851_385110

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define addition for Angle
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let extraDegrees := totalMinutes / 60
  { degrees := a.degrees + b.degrees + extraDegrees,
    minutes := totalMinutes % 60 }

-- Define subtraction for Angle
def Angle.sub (a b : Angle) : Angle :=
  let totalMinutes := (a.degrees * 60 + a.minutes) - (b.degrees * 60 + b.minutes)
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60 }

-- Define the given complementary angle
def complementaryAngle : Angle := { degrees := 54, minutes := 38 }

-- Define 90 degrees
def rightAngle : Angle := { degrees := 90, minutes := 0 }

-- Define 180 degrees
def straightAngle : Angle := { degrees := 180, minutes := 0 }

-- Theorem statement
theorem supplementary_angle_theorem :
  let angle := Angle.sub rightAngle complementaryAngle
  Angle.sub straightAngle angle = { degrees := 144, minutes := 38 } := by sorry

end NUMINAMATH_CALUDE_supplementary_angle_theorem_l3851_385110


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l3851_385156

theorem arccos_equation_solution :
  ∃! x : ℝ, Real.arccos (2 * x) - Real.arccos x = π / 3 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l3851_385156


namespace NUMINAMATH_CALUDE_find_s_value_l3851_385135

/-- Given a relationship between R, S, and T, prove the value of S for specific R and T -/
theorem find_s_value (c : ℝ) (R S T : ℝ → ℝ) :
  (∀ x, R x = c * (S x / T x)) →  -- Relationship between R, S, and T
  R 1 = 2 →                       -- Initial condition for R
  S 1 = 1/2 →                     -- Initial condition for S
  T 1 = 4/3 →                     -- Initial condition for T
  R 2 = Real.sqrt 75 →            -- New condition for R
  T 2 = Real.sqrt 32 →            -- New condition for T
  S 2 = 45/4 :=                   -- Conclusion: value of S
by sorry

end NUMINAMATH_CALUDE_find_s_value_l3851_385135


namespace NUMINAMATH_CALUDE_ferry_time_difference_l3851_385178

/-- A ferry route between an island and the mainland -/
structure FerryRoute where
  speed : ℝ  -- Speed in km/h
  time : ℝ   -- Time in hours
  distance : ℝ -- Distance in km

/-- The problem setup for two ferry routes -/
def ferry_problem (p q : FerryRoute) : Prop :=
  p.speed = 8 ∧
  p.time = 3 ∧
  p.distance = p.speed * p.time ∧
  q.distance = 3 * p.distance ∧
  q.speed = p.speed + 1

theorem ferry_time_difference (p q : FerryRoute) 
  (h : ferry_problem p q) : q.time - p.time = 5 := by
  sorry

end NUMINAMATH_CALUDE_ferry_time_difference_l3851_385178


namespace NUMINAMATH_CALUDE_equal_profit_percentage_l3851_385154

def shopkeeper_profit (total_quantity : ℝ) (portion1_percentage : ℝ) (portion2_percentage : ℝ) (profit_percentage : ℝ) : Prop :=
  portion1_percentage + portion2_percentage = 100 ∧
  portion1_percentage > 0 ∧
  portion2_percentage > 0 ∧
  profit_percentage ≥ 0

theorem equal_profit_percentage 
  (total_quantity : ℝ) 
  (portion1_percentage : ℝ) 
  (portion2_percentage : ℝ) 
  (total_profit_percentage : ℝ) 
  (h : shopkeeper_profit total_quantity portion1_percentage portion2_percentage total_profit_percentage) :
  ∃ (individual_profit_percentage : ℝ),
    individual_profit_percentage = total_profit_percentage ∧
    individual_profit_percentage * portion1_percentage / 100 + 
    individual_profit_percentage * portion2_percentage / 100 = 
    total_profit_percentage := by
  sorry

end NUMINAMATH_CALUDE_equal_profit_percentage_l3851_385154


namespace NUMINAMATH_CALUDE_number_problem_l3851_385134

theorem number_problem : ∃ x : ℝ, 4 * x + 7 * x = 55 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3851_385134


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_geq_sqrt2_sum_l3851_385119

theorem sqrt_sum_squares_geq_sqrt2_sum (a b : ℝ) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_geq_sqrt2_sum_l3851_385119


namespace NUMINAMATH_CALUDE_sphere_surface_volume_relation_l3851_385160

theorem sphere_surface_volume_relation :
  ∀ (r R : ℝ),
  r > 0 →
  R > 0 →
  (4 * Real.pi * R^2) = (4 * (4 * Real.pi * r^2)) →
  ((4/3) * Real.pi * R^3) = (8 * ((4/3) * Real.pi * r^3)) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_volume_relation_l3851_385160


namespace NUMINAMATH_CALUDE_janet_action_figures_l3851_385175

theorem janet_action_figures (initial : ℕ) (sold : ℕ) (final_total : ℕ) :
  initial = 10 →
  sold = 6 →
  final_total = 24 →
  let remaining := initial - sold
  let brother_gift := 2 * remaining
  let before_new := remaining + brother_gift
  final_total - before_new = 12 :=
by sorry

end NUMINAMATH_CALUDE_janet_action_figures_l3851_385175


namespace NUMINAMATH_CALUDE_add_2057_minutes_to_3_15pm_l3851_385164

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem add_2057_minutes_to_3_15pm (start : Time) (result : Time) :
  start.hours = 15 ∧ start.minutes = 15 →
  result = addMinutes start 2057 →
  result.hours = 1 ∧ result.minutes = 32 := by
  sorry

end NUMINAMATH_CALUDE_add_2057_minutes_to_3_15pm_l3851_385164


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l3851_385115

theorem price_reduction_percentage (original_price reduction_amount : ℝ) : 
  original_price = 500 → 
  reduction_amount = 250 → 
  (reduction_amount / original_price) * 100 = 50 :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l3851_385115


namespace NUMINAMATH_CALUDE_no_prime_root_solution_l3851_385127

/-- A quadratic equation x^2 - 67x + k = 0 with prime roots -/
def has_prime_roots (k : ℤ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ 
  (p : ℤ) + q = 67 ∧ (p : ℤ) * q = k

/-- There are no integer values of k for which the equation x^2 - 67x + k = 0 has two prime roots -/
theorem no_prime_root_solution : ¬∃ k : ℤ, has_prime_roots k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_root_solution_l3851_385127


namespace NUMINAMATH_CALUDE_starters_with_triplet_l3851_385177

def total_players : ℕ := 12
def triplets : ℕ := 3
def starters : ℕ := 5

theorem starters_with_triplet (total_players : ℕ) (triplets : ℕ) (starters : ℕ) :
  total_players = 12 →
  triplets = 3 →
  starters = 5 →
  (Nat.choose total_players starters) - (Nat.choose (total_players - triplets) starters) = 666 :=
by sorry

end NUMINAMATH_CALUDE_starters_with_triplet_l3851_385177


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_perfect_square_l3851_385157

theorem no_integer_solutions_for_perfect_square : 
  ¬ ∃ (x : ℤ), ∃ (y : ℤ), x^4 + 4*x^3 + 10*x^2 + 4*x + 29 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_perfect_square_l3851_385157


namespace NUMINAMATH_CALUDE_alex_initial_silk_amount_l3851_385100

/-- The amount of silk Alex had in storage initially -/
def initial_silk_amount (num_friends : ℕ) (silk_per_friend : ℕ) (num_dresses : ℕ) (silk_per_dress : ℕ) : ℕ :=
  num_friends * silk_per_friend + num_dresses * silk_per_dress

/-- Theorem stating that Alex had 600 meters of silk initially -/
theorem alex_initial_silk_amount :
  initial_silk_amount 5 20 100 5 = 600 := by sorry

end NUMINAMATH_CALUDE_alex_initial_silk_amount_l3851_385100


namespace NUMINAMATH_CALUDE_sqrt_sum_rationalization_l3851_385186

theorem sqrt_sum_rationalization : ∃ (a b c : ℕ+), 
  (Real.sqrt 8 + (1 / Real.sqrt 8) + Real.sqrt 9 + (1 / Real.sqrt 9) = (a * Real.sqrt 8 + b * Real.sqrt 9) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    (Real.sqrt 8 + (1 / Real.sqrt 8) + Real.sqrt 9 + (1 / Real.sqrt 9) = (a' * Real.sqrt 8 + b' * Real.sqrt 9) / c') →
    c ≤ c') ∧
  (a + b + c = 31) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_rationalization_l3851_385186


namespace NUMINAMATH_CALUDE_dessert_combinations_eq_twelve_l3851_385129

/-- The number of dessert options available -/
def num_desserts : ℕ := 4

/-- The number of courses in the meal -/
def num_courses : ℕ := 2

/-- Function to calculate the number of ways to order the dessert -/
def dessert_combinations : ℕ := num_desserts * (num_desserts - 1)

/-- Theorem stating that the number of ways to order the dessert is 12 -/
theorem dessert_combinations_eq_twelve : dessert_combinations = 12 := by
  sorry

end NUMINAMATH_CALUDE_dessert_combinations_eq_twelve_l3851_385129


namespace NUMINAMATH_CALUDE_pascals_triangle_56th_row_second_element_l3851_385140

theorem pascals_triangle_56th_row_second_element :
  let n : ℕ := 56  -- The row number (0-indexed) with 57 elements
  let k : ℕ := 1   -- The position of the second element (0-indexed)
  Nat.choose n k = 56 := by
sorry

end NUMINAMATH_CALUDE_pascals_triangle_56th_row_second_element_l3851_385140


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l3851_385122

/-- Given a geometric sequence {a_n} where a_1+1, a_3+4, a_5+7 form an arithmetic sequence,
    the common difference of this arithmetic sequence is 3. -/
theorem geometric_arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d) :
  ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l3851_385122


namespace NUMINAMATH_CALUDE_spurs_basketball_count_l3851_385198

theorem spurs_basketball_count :
  let num_players : ℕ := 22
  let balls_per_player : ℕ := 11
  num_players * balls_per_player = 242 :=
by sorry

end NUMINAMATH_CALUDE_spurs_basketball_count_l3851_385198


namespace NUMINAMATH_CALUDE_fraction_is_integer_l3851_385149

theorem fraction_is_integer (b t : ℤ) (h : b ≠ 1) :
  ∃ k : ℤ, (t^5 - 5*b + 4) / (b^2 - 2*b + 1) = k :=
sorry

end NUMINAMATH_CALUDE_fraction_is_integer_l3851_385149


namespace NUMINAMATH_CALUDE_row_sum_equals_2013_squared_l3851_385168

theorem row_sum_equals_2013_squared :
  let n : ℕ := 1007
  let row_sum (k : ℕ) : ℕ := k * (2 * k - 1)
  row_sum n = 2013^2 := by
  sorry

end NUMINAMATH_CALUDE_row_sum_equals_2013_squared_l3851_385168


namespace NUMINAMATH_CALUDE_number_sum_proof_l3851_385173

theorem number_sum_proof (x : ℤ) : x + 14 = 68 → x + (x + 41) = 149 := by
  sorry

end NUMINAMATH_CALUDE_number_sum_proof_l3851_385173


namespace NUMINAMATH_CALUDE_dividend_divisor_product_l3851_385138

theorem dividend_divisor_product (d : ℤ) (D : ℤ) : 
  D = d + 78 → D = 6 * d + 3 → D * d = 1395 := by
sorry

end NUMINAMATH_CALUDE_dividend_divisor_product_l3851_385138


namespace NUMINAMATH_CALUDE_specific_ellipse_equation_l3851_385118

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  /-- The focal length of the ellipse -/
  focal_length : ℝ
  /-- The x-coordinate of one directrix -/
  directrix_x : ℝ

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 4 = 1

/-- Theorem stating the equation of the specific ellipse -/
theorem specific_ellipse_equation (e : Ellipse) 
  (h1 : e.focal_length = 4)
  (h2 : e.directrix_x = -4) :
  ∀ x y : ℝ, ellipse_equation e x y := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_equation_l3851_385118


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3851_385172

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  a₁ ≠ 0 →
  q ≠ 1 →
  let S₂ := a₁ * (1 - q^2) / (1 - q)
  let S₃ := a₁ * (1 - q^3) / (1 - q)
  S₃ + 3 * S₂ = 0 →
  q = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3851_385172


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_f_extrema_sum_max_l3851_385167

noncomputable section

def f (a x : ℝ) := x^2 / 2 - 4*a*x + a * Real.log x + 3*a^2 + 2*a

def f_deriv (a x : ℝ) := x - 4*a + a/x

theorem f_monotonicity_and_extrema (a : ℝ) (ha : a > 0) :
  (∀ x > 0, f_deriv a x ≥ 0) ∨
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0) :=
sorry

theorem f_extrema_sum_max (a : ℝ) (ha : a > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 →
  ∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → f_deriv a y₁ = 0 → f_deriv a y₂ = 0 →
  f a x₁ + f a x₂ ≥ f a y₁ + f a y₂ ∧
  f a x₁ + f a x₂ ≤ 1 :=
sorry

end

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_f_extrema_sum_max_l3851_385167


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3851_385163

/-- Calculates the average speed of a round trip given the outbound speed and the fact that the return journey takes twice as long. -/
theorem round_trip_average_speed (outbound_speed : ℝ) 
  (h : outbound_speed = 45) : 
  let return_time := 2 * (1 / outbound_speed)
  let total_time := 1 / outbound_speed + return_time
  let total_distance := 2
  (total_distance / total_time) = 30 := by sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l3851_385163


namespace NUMINAMATH_CALUDE_gold_cube_comparison_l3851_385179

/-- Represents the properties of a cube of gold -/
structure GoldCube where
  side_length : ℝ
  weight : ℝ
  value : ℝ

/-- Theorem stating the relationship between two gold cubes of different sizes -/
theorem gold_cube_comparison (small_cube large_cube : GoldCube) :
  small_cube.side_length = 4 →
  small_cube.weight = 5 →
  small_cube.value = 1200 →
  large_cube.side_length = 6 →
  (large_cube.weight = 16.875 ∧ large_cube.value = 4050) :=
by
  sorry

#check gold_cube_comparison

end NUMINAMATH_CALUDE_gold_cube_comparison_l3851_385179


namespace NUMINAMATH_CALUDE_hexagon_toothpicks_l3851_385170

/-- Represents a hexagonal pattern of small equilateral triangles -/
structure HexagonalPattern :=
  (max_row_triangles : ℕ)

/-- Calculates the total number of small triangles in the hexagonal pattern -/
def total_triangles (h : HexagonalPattern) : ℕ :=
  let half_triangles := (h.max_row_triangles * (h.max_row_triangles + 1)) / 2
  2 * half_triangles + h.max_row_triangles

/-- Calculates the number of boundary toothpicks in the hexagonal pattern -/
def boundary_toothpicks (h : HexagonalPattern) : ℕ :=
  6 * h.max_row_triangles

/-- Calculates the total number of toothpicks required to construct the hexagonal pattern -/
def total_toothpicks (h : HexagonalPattern) : ℕ :=
  3 * total_triangles h - boundary_toothpicks h

/-- Theorem stating that a hexagonal pattern with 1001 triangles in its largest row requires 3006003 toothpicks -/
theorem hexagon_toothpicks :
  let h : HexagonalPattern := ⟨1001⟩
  total_toothpicks h = 3006003 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_toothpicks_l3851_385170


namespace NUMINAMATH_CALUDE_cyclic_number_property_l3851_385111

def digit_set (n : ℕ) : Set ℕ :=
  {d | ∃ k, n = d + 10 * k ∨ k = d + 10 * n}

def has_same_digits (a b : ℕ) : Prop :=
  digit_set a = digit_set b

theorem cyclic_number_property (n : ℕ) (h : n = 142857) :
  ∀ k : ℕ, k ≥ 1 → k ≤ 6 → has_same_digits n (n * k) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_number_property_l3851_385111


namespace NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l3851_385196

-- Define the parallel lines
def line1 (x y : ℝ) : Prop := x + 3 * y - 5 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y - 3 = 0

-- Define the line containing the center of the circle
def centerLine (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the circle equation
def circleEquation (x y : ℝ) : Prop := (x + 13/5)^2 + (y - 11/5)^2 = 1/10

-- Theorem stating the circle equation given the conditions
theorem circle_tangent_to_parallel_lines :
  ∀ (C : Set (ℝ × ℝ)),
  (∃ (x₁ y₁ : ℝ), (x₁, y₁) ∈ C ∧ line1 x₁ y₁) ∧
  (∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ C ∧ line2 x₂ y₂) ∧
  (∃ (x₀ y₀ : ℝ), (x₀, y₀) ∈ C ∧ centerLine x₀ y₀) →
  ∀ (x y : ℝ), (x, y) ∈ C ↔ circleEquation x y :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l3851_385196


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_given_subset_l3851_385180

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |2*x + 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ 1} = {x : ℝ | -1 ≤ x ∧ x ≤ -1/3} :=
sorry

-- Part 2
def P (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ -2*x + 1}

theorem range_of_a_given_subset :
  (∀ a : ℝ, Set.Icc (-1 : ℝ) (-1/4) ⊆ P a) →
  {a : ℝ | ∀ x ∈ Set.Icc (-1 : ℝ) (-1/4), f a x ≤ -2*x + 1} = Set.Icc (-3/4 : ℝ) (5/4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_given_subset_l3851_385180


namespace NUMINAMATH_CALUDE_t_shirts_per_package_l3851_385121

theorem t_shirts_per_package (packages : ℕ) (total_shirts : ℕ) 
  (h1 : packages = 71) (h2 : total_shirts = 426) : 
  total_shirts / packages = 6 := by
sorry

end NUMINAMATH_CALUDE_t_shirts_per_package_l3851_385121


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3851_385112

theorem container_volume_ratio : 
  ∀ (v1 v2 v3 : ℝ), 
    v1 > 0 → v2 > 0 → v3 > 0 →
    (2/3 : ℝ) * v1 = (1/2 : ℝ) * v2 →
    (1/2 : ℝ) * v2 = (3/5 : ℝ) * v3 →
    v1 / v3 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3851_385112


namespace NUMINAMATH_CALUDE_modulus_of_x_is_sqrt_10_l3851_385171

-- Define the complex number x
def x : ℂ := sorry

-- State the theorem
theorem modulus_of_x_is_sqrt_10 :
  x + Complex.I = (2 - Complex.I) / Complex.I →
  Complex.abs x = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_x_is_sqrt_10_l3851_385171


namespace NUMINAMATH_CALUDE_total_lateness_l3851_385109

/-- Given a student who is 20 minutes late and four other students who are each 10 minutes later than the first student, 
    the total time of lateness for all five students is 140 minutes. -/
theorem total_lateness (charlize_lateness : ℕ) (classmates_count : ℕ) (additional_lateness : ℕ) : 
  charlize_lateness = 20 →
  classmates_count = 4 →
  additional_lateness = 10 →
  charlize_lateness + classmates_count * (charlize_lateness + additional_lateness) = 140 :=
by sorry

end NUMINAMATH_CALUDE_total_lateness_l3851_385109


namespace NUMINAMATH_CALUDE_min_yacht_capacity_l3851_385113

/-- Represents the number of sheikhs --/
def num_sheikhs : ℕ := 10

/-- Represents the number of wives per sheikh --/
def wives_per_sheikh : ℕ := 100

/-- Represents the total number of wives --/
def total_wives : ℕ := num_sheikhs * wives_per_sheikh

/-- Represents the law: a woman must not be with a man other than her husband unless her husband is present --/
def law_compliant (n : ℕ) : Prop :=
  ∀ (women_on_bank : ℕ) (men_on_bank : ℕ),
    women_on_bank ≤ total_wives ∧ men_on_bank ≤ num_sheikhs →
    (women_on_bank ≤ n ∨ men_on_bank = num_sheikhs ∨ women_on_bank = 0)

/-- Theorem stating that the smallest yacht capacity that allows all sheikhs and wives to cross the river while complying with the law is 10 --/
theorem min_yacht_capacity :
  ∃ (n : ℕ), n = 10 ∧ law_compliant n ∧ ∀ (m : ℕ), m < n → ¬law_compliant m :=
sorry

end NUMINAMATH_CALUDE_min_yacht_capacity_l3851_385113


namespace NUMINAMATH_CALUDE_unique_solution_for_B_l3851_385191

theorem unique_solution_for_B : ∃! B : ℕ, 
  B < 10 ∧ (∃ A : ℕ, A < 10 ∧ 500 + 10 * A + 8 - (100 * B + 14) = 364) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_B_l3851_385191


namespace NUMINAMATH_CALUDE_serena_age_proof_l3851_385169

/-- Serena's current age -/
def serena_age : ℕ := 9

/-- Serena's mother's current age -/
def mother_age : ℕ := 39

/-- Years into the future when the age comparison is made -/
def years_later : ℕ := 6

theorem serena_age_proof :
  serena_age = 9 ∧
  mother_age = 39 ∧
  mother_age + years_later = 3 * (serena_age + years_later) :=
by sorry

end NUMINAMATH_CALUDE_serena_age_proof_l3851_385169


namespace NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l3851_385144

/-- Represents the four quadrants in a coordinate system -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines the quadrant of an angle given in degrees -/
def angle_quadrant (α : ℝ) : Quadrant :=
  sorry

/-- Theorem: For any integer k, the angle α = k·180° + 45° lies in either the first or third quadrant -/
theorem angle_in_first_or_third_quadrant (k : ℤ) :
  let α := k * 180 + 45
  (angle_quadrant α = Quadrant.first) ∨ (angle_quadrant α = Quadrant.third) :=
sorry

end NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l3851_385144


namespace NUMINAMATH_CALUDE_cosine_value_in_special_triangle_l3851_385159

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem cosine_value_in_special_triangle (t : Triangle) 
  (h1 : t.c = 2 * t.a)  -- Given condition: c = 2a
  (h2 : Real.sin t.B ^ 2 = Real.sin t.A * Real.sin t.C)  -- Given condition: sin²B = sin A * sin C
  : Real.cos t.B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_in_special_triangle_l3851_385159


namespace NUMINAMATH_CALUDE_line_l_equation_and_symmetric_points_l3851_385158

/-- Parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Line l intersecting the parabola -/
def Line_l : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (m b : ℝ), p.2 = m * p.1 + b}

/-- Point P that bisects segment AB -/
def P : ℝ × ℝ := (2, 2)

/-- A and B are points where line l intersects the parabola -/
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

theorem line_l_equation_and_symmetric_points :
  (∀ p ∈ Line_l, 2 * p.1 - p.2 - 2 = 0) ∧
  (∃ (C D : ℝ × ℝ), C ∈ Parabola ∧ D ∈ Parabola ∧
    (∀ p ∈ Line_l, (C.1 + D.1) * p.2 = (C.2 + D.2) * p.1 + C.1 * D.2 - C.2 * D.1) ∧
    (∀ p ∈ {p : ℝ × ℝ | p.1 + 2 * p.2 - 19 = 0}, p = C ∨ p = D)) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_and_symmetric_points_l3851_385158


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3851_385147

theorem fraction_to_decimal : (2 : ℚ) / 25 = 0.08 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3851_385147


namespace NUMINAMATH_CALUDE_ron_pick_frequency_l3851_385117

/-- Represents a book club with a given number of members -/
structure BookClub where
  members : ℕ

/-- Calculates how many times a member gets to pick a book in a year -/
def pickFrequency (club : BookClub) (weeksInYear : ℕ) : ℕ :=
  weeksInYear / club.members

theorem ron_pick_frequency :
  let couples := 3
  let singlePeople := 5
  let ronAndWife := 2
  let weeksInYear := 52
  let club := BookClub.mk (couples * 2 + singlePeople + ronAndWife)
  pickFrequency club weeksInYear = 4 := by
  sorry

end NUMINAMATH_CALUDE_ron_pick_frequency_l3851_385117


namespace NUMINAMATH_CALUDE_constant_value_proof_l3851_385133

theorem constant_value_proof (x y : ℝ) (a : ℝ) 
  (h1 : (a * x + 8 * y) / (x - 2 * y) = 29)
  (h2 : x / (2 * y) = 3 / 2) : 
  a = 7 := by sorry

end NUMINAMATH_CALUDE_constant_value_proof_l3851_385133


namespace NUMINAMATH_CALUDE_committee_formation_ways_l3851_385152

theorem committee_formation_ways (n m : ℕ) (hn : n = 10) (hm : m = 4) : 
  Nat.choose n m = 210 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_ways_l3851_385152


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3851_385108

theorem square_sum_geq_product_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a ∧
  (a^2 + b^2 + c^2 = a*b + b*c + c*a ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3851_385108


namespace NUMINAMATH_CALUDE_fourth_sphere_radius_l3851_385136

/-- Given four spheres on a table, where each sphere touches the table and the other three spheres,
    and three of the spheres have radius R, the radius of the fourth sphere is 4R/3. -/
theorem fourth_sphere_radius (R : ℝ) (R_pos : R > 0) : ∃ r : ℝ,
  (∀ (i j : Fin 4), i ≠ j → ∃ (x y z : ℝ),
    (i.val < 3 → norm (⟨x, y, z⟩ : ℝ × ℝ × ℝ) = R) ∧
    (i.val = 3 → norm (⟨x, y, z⟩ : ℝ × ℝ × ℝ) = r) ∧
    (j.val < 3 → ∃ (x' y' z' : ℝ), norm (⟨x - x', y - y', z - z'⟩ : ℝ × ℝ × ℝ) = R + R) ∧
    (j.val = 3 → ∃ (x' y' z' : ℝ), norm (⟨x - x', y - y', z - z'⟩ : ℝ × ℝ × ℝ) = R + r) ∧
    z ≥ R ∧ z' ≥ R) ∧
  r = 4 * R / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_sphere_radius_l3851_385136


namespace NUMINAMATH_CALUDE_bankers_discount_example_l3851_385194

/-- Calculates the banker's discount given the face value and true discount of a bill -/
def bankers_discount (face_value : ℚ) (true_discount : ℚ) : ℚ :=
  let present_value := face_value - true_discount
  (true_discount * face_value) / present_value

/-- Theorem: Given a bill with face value 2460 and true discount 360, the banker's discount is 422 -/
theorem bankers_discount_example : bankers_discount 2460 360 = 422 := by
  sorry

#eval bankers_discount 2460 360

end NUMINAMATH_CALUDE_bankers_discount_example_l3851_385194


namespace NUMINAMATH_CALUDE_sheep_count_l3851_385192

theorem sheep_count : ∀ (num_sheep : ℕ), 
  (∀ (sheep : ℕ), sheep ≤ num_sheep → (sheep * 1 = sheep)) →  -- One sheep eats one bag in 40 days
  (num_sheep * 1 = 40) →  -- Total bags eaten by all sheep is 40
  num_sheep = 40 := by
sorry

end NUMINAMATH_CALUDE_sheep_count_l3851_385192


namespace NUMINAMATH_CALUDE_chili_paste_can_size_l3851_385106

/-- Proves that the size of smaller chili paste cans is 15 ounces -/
theorem chili_paste_can_size 
  (larger_can_size : ℕ) 
  (larger_can_count : ℕ) 
  (extra_smaller_cans : ℕ) 
  (smaller_can_size : ℕ) : 
  larger_can_size = 25 → 
  larger_can_count = 45 → 
  extra_smaller_cans = 30 → 
  (larger_can_count + extra_smaller_cans) * smaller_can_size = larger_can_count * larger_can_size → 
  smaller_can_size = 15 := by
sorry

end NUMINAMATH_CALUDE_chili_paste_can_size_l3851_385106


namespace NUMINAMATH_CALUDE_hotdogs_day1_proof_l3851_385190

/-- Represents the price of a hamburger in dollars -/
def hamburger_price : ℚ := 2

/-- Represents the price of a hot dog in dollars -/
def hotdog_price : ℚ := 1

/-- Represents the number of hamburgers bought on day 1 -/
def hamburgers_day1 : ℕ := 3

/-- Represents the number of hamburgers bought on day 2 -/
def hamburgers_day2 : ℕ := 2

/-- Represents the number of hot dogs bought on day 2 -/
def hotdogs_day2 : ℕ := 3

/-- Represents the total cost of purchases on day 1 in dollars -/
def total_cost_day1 : ℚ := 10

/-- Represents the total cost of purchases on day 2 in dollars -/
def total_cost_day2 : ℚ := 7

/-- Calculates the number of hot dogs bought on day 1 -/
def hotdogs_day1 : ℕ := 4

theorem hotdogs_day1_proof : 
  hamburgers_day1 * hamburger_price + hotdogs_day1 * hotdog_price = total_cost_day1 ∧
  hamburgers_day2 * hamburger_price + hotdogs_day2 * hotdog_price = total_cost_day2 :=
by sorry

end NUMINAMATH_CALUDE_hotdogs_day1_proof_l3851_385190


namespace NUMINAMATH_CALUDE_tourist_walking_speed_l3851_385142

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hMinutesValid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Represents the problem scenario -/
structure TouristProblem where
  scheduledArrival : Time
  actualArrival : Time
  busSpeed : ℝ
  earlyArrival : ℕ

/-- Calculates the tourists' walking speed -/
noncomputable def touristSpeed (problem : TouristProblem) : ℝ :=
  let walkingTime := timeDifference problem.actualArrival problem.scheduledArrival - problem.earlyArrival
  let distance := problem.busSpeed * (problem.earlyArrival / 2) / 60
  distance / (walkingTime / 60)

/-- The main theorem to prove -/
theorem tourist_walking_speed (problem : TouristProblem) 
  (hScheduledArrival : problem.scheduledArrival = ⟨5, 0, by norm_num⟩)
  (hActualArrival : problem.actualArrival = ⟨3, 10, by norm_num⟩)
  (hBusSpeed : problem.busSpeed = 60)
  (hEarlyArrival : problem.earlyArrival = 20) :
  touristSpeed problem = 6 := by
  sorry

end NUMINAMATH_CALUDE_tourist_walking_speed_l3851_385142


namespace NUMINAMATH_CALUDE_expression_equals_one_l3851_385130

theorem expression_equals_one (a b c k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0)
  (sum_zero : a + b + c = 0) (a_squared : a^2 = k * b^2) :
  (a^2 * b^2) / ((a^2 - b*c) * (b^2 - a*c)) +
  (a^2 * c^2) / ((a^2 - b*c) * (c^2 - a*b)) +
  (b^2 * c^2) / ((b^2 - a*c) * (c^2 - a*b)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3851_385130


namespace NUMINAMATH_CALUDE_plates_added_before_fall_l3851_385181

theorem plates_added_before_fall (initial_plates : Nat) (second_addition : Nat) (total_plates : Nat)
  (h1 : initial_plates = 27)
  (h2 : second_addition = 37)
  (h3 : total_plates = 83) :
  total_plates - (initial_plates + second_addition) = 19 := by
  sorry

end NUMINAMATH_CALUDE_plates_added_before_fall_l3851_385181


namespace NUMINAMATH_CALUDE_special_triangle_ratio_l3851_385150

/-- A scalene triangle with two medians equal to two altitudes -/
structure SpecialTriangle where
  -- The triangle is scalene
  is_scalene : Bool
  -- Two medians are equal to two altitudes
  two_medians_equal_altitudes : Bool

/-- The ratio of the third median to the third altitude -/
def third_median_altitude_ratio (t : SpecialTriangle) : ℚ :=
  7 / 2

/-- Theorem stating the ratio of the third median to the third altitude -/
theorem special_triangle_ratio (t : SpecialTriangle) 
  (h1 : t.is_scalene = true) 
  (h2 : t.two_medians_equal_altitudes = true) : 
  third_median_altitude_ratio t = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_ratio_l3851_385150


namespace NUMINAMATH_CALUDE_equal_probabilities_l3851_385123

/-- Represents a box containing colored balls -/
structure Box where
  red_balls : ℕ
  green_balls : ℕ

/-- Represents the state of both boxes -/
structure BoxState where
  red_box : Box
  green_box : Box

/-- Initial state of the boxes -/
def initial_state : BoxState :=
  { red_box := { red_balls := 100, green_balls := 0 },
    green_box := { red_balls := 0, green_balls := 100 } }

/-- State after first transfer (8 red balls from red to green box) -/
def first_transfer (state : BoxState) : BoxState :=
  { red_box := { red_balls := state.red_box.red_balls - 8, green_balls := state.red_box.green_balls },
    green_box := { red_balls := state.green_box.red_balls + 8, green_balls := state.green_box.green_balls } }

/-- Probability of drawing a specific color from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => box.red_balls / (box.red_balls + box.green_balls)
  | "green" => box.green_balls / (box.red_balls + box.green_balls)
  | _ => 0

/-- Theorem stating the equality of probabilities after transfers and mixing -/
theorem equal_probabilities (final_state : BoxState) 
    (h1 : final_state.red_box.green_balls + final_state.green_box.green_balls = 100) 
    (h2 : final_state.red_box.red_balls + final_state.green_box.red_balls = 100) :
    prob_draw final_state.red_box "green" = prob_draw final_state.green_box "red" :=
  sorry


end NUMINAMATH_CALUDE_equal_probabilities_l3851_385123


namespace NUMINAMATH_CALUDE_prob_non_white_ball_l3851_385195

/-- The probability of drawing a non-white ball from a bag -/
theorem prob_non_white_ball (white yellow red : ℕ) (h : white = 6 ∧ yellow = 5 ∧ red = 4) :
  (yellow + red) / (white + yellow + red) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_white_ball_l3851_385195


namespace NUMINAMATH_CALUDE_function_non_negative_range_l3851_385102

theorem function_non_negative_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = x^2 - 4*x + a) →
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) →
  a ∈ Set.Ici 3 := by
sorry

end NUMINAMATH_CALUDE_function_non_negative_range_l3851_385102


namespace NUMINAMATH_CALUDE_absolute_difference_in_terms_of_sum_and_product_l3851_385107

theorem absolute_difference_in_terms_of_sum_and_product (x₁ x₂ a b : ℝ) 
  (h_sum : x₁ + x₂ = a) (h_product : x₁ * x₂ = b) : 
  |x₁ - x₂| = Real.sqrt (a^2 - 4*b) := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_in_terms_of_sum_and_product_l3851_385107


namespace NUMINAMATH_CALUDE_correct_day_is_thursday_l3851_385131

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define the statements made by each person
def statement_A (today : DayOfWeek) : Prop := today = DayOfWeek.Friday
def statement_B (today : DayOfWeek) : Prop := today = DayOfWeek.Wednesday
def statement_C (today : DayOfWeek) : Prop := ¬(statement_A today ∨ statement_B today)
def statement_D (today : DayOfWeek) : Prop := today ≠ DayOfWeek.Thursday

-- Define the condition that only one statement is correct
def only_one_correct (today : DayOfWeek) : Prop :=
  (statement_A today ∧ ¬statement_B today ∧ ¬statement_C today ∧ ¬statement_D today) ∨
  (¬statement_A today ∧ statement_B today ∧ ¬statement_C today ∧ ¬statement_D today) ∨
  (¬statement_A today ∧ ¬statement_B today ∧ statement_C today ∧ ¬statement_D today) ∨
  (¬statement_A today ∧ ¬statement_B today ∧ ¬statement_C today ∧ statement_D today)

-- Theorem stating that Thursday is the only day satisfying all conditions
theorem correct_day_is_thursday :
  ∃! today : DayOfWeek, only_one_correct today ∧ today = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_correct_day_is_thursday_l3851_385131


namespace NUMINAMATH_CALUDE_power_of_three_even_tens_digit_l3851_385187

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem power_of_three_even_tens_digit (n : ℕ) (h : n ≥ 3) :
  Even (tens_digit (3^n)) := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_even_tens_digit_l3851_385187


namespace NUMINAMATH_CALUDE_herd_division_l3851_385153

theorem herd_division (herd : ℚ) : 
  (1/3 : ℚ) + (1/6 : ℚ) + (1/9 : ℚ) + (8 : ℚ) / herd = 1 → 
  herd = 144/7 := by
  sorry

end NUMINAMATH_CALUDE_herd_division_l3851_385153


namespace NUMINAMATH_CALUDE_smallest_p_value_l3851_385132

theorem smallest_p_value (p q : ℕ+) (h1 : (5 : ℚ) / 8 < p / q) (h2 : p / q < (7 : ℚ) / 8) (h3 : p + q = 2005) :
  p.val ≥ 772 ∧ (∀ (p' : ℕ+), p'.val ≥ 772 → (5 : ℚ) / 8 < p' / (2005 - p') → p' / (2005 - p') < (7 : ℚ) / 8 → p'.val ≤ p.val) :=
sorry

end NUMINAMATH_CALUDE_smallest_p_value_l3851_385132


namespace NUMINAMATH_CALUDE_money_distribution_l3851_385146

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 350)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 350) :
  c = 200 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3851_385146


namespace NUMINAMATH_CALUDE_cow_fraction_sold_l3851_385128

/-- Represents the number of animals on a petting farm. -/
structure PettingFarm where
  cows : ℕ
  dogs : ℕ

/-- Represents the state of the petting farm before and after selling animals. -/
structure FarmState where
  initial : PettingFarm
  final : PettingFarm

theorem cow_fraction_sold (farm : FarmState) : 
  farm.initial.cows = 184 →
  farm.initial.cows = 2 * farm.initial.dogs →
  farm.final.dogs = farm.initial.dogs / 4 →
  farm.final.cows + farm.final.dogs = 161 →
  (farm.initial.cows - farm.final.cows) / farm.initial.cows = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cow_fraction_sold_l3851_385128


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3851_385176

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) :
  (2 * r) ^ 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3851_385176


namespace NUMINAMATH_CALUDE_hoseok_payment_l3851_385101

/-- The price of item (a) bought by Hoseok at the mart -/
def item_price : ℕ := 7450

/-- The number of 1000 won bills used -/
def bills_1000 : ℕ := 7

/-- The number of 100 won coins used -/
def coins_100 : ℕ := 4

/-- The number of 10 won coins used -/
def coins_10 : ℕ := 5

/-- The denomination of the bills used -/
def bill_value : ℕ := 1000

/-- The denomination of the first type of coins used -/
def coin_value_100 : ℕ := 100

/-- The denomination of the second type of coins used -/
def coin_value_10 : ℕ := 10

theorem hoseok_payment :
  item_price = bills_1000 * bill_value + coins_100 * coin_value_100 + coins_10 * coin_value_10 :=
by sorry

end NUMINAMATH_CALUDE_hoseok_payment_l3851_385101


namespace NUMINAMATH_CALUDE_amy_school_year_hours_l3851_385183

/-- Calculates the required weekly hours for Amy's school year work --/
def school_year_weekly_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) : ℕ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_hours_needed := school_year_target / hourly_wage
  total_hours_needed / school_year_weeks

/-- Theorem stating that Amy needs to work 15 hours per week during the school year --/
theorem amy_school_year_hours : 
  school_year_weekly_hours 8 40 3200 32 4800 = 15 := by
  sorry

end NUMINAMATH_CALUDE_amy_school_year_hours_l3851_385183


namespace NUMINAMATH_CALUDE_cos_two_alpha_plus_two_beta_l3851_385188

theorem cos_two_alpha_plus_two_beta (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3)
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_plus_two_beta_l3851_385188


namespace NUMINAMATH_CALUDE_pure_imaginary_real_part_zero_l3851_385148

/-- A complex number z is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_real_part_zero (a : ℝ) :
  isPureImaginary (Complex.mk a 1) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_real_part_zero_l3851_385148


namespace NUMINAMATH_CALUDE_triangles_from_points_on_circle_l3851_385104

def points_on_circle : ℕ := 10
def vertices_per_triangle : ℕ := 3

theorem triangles_from_points_on_circle :
  Nat.choose points_on_circle vertices_per_triangle = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangles_from_points_on_circle_l3851_385104


namespace NUMINAMATH_CALUDE_min_mn_tangent_line_circle_l3851_385182

/-- Given positive real numbers m and n, if the line (m+1)x + (n+1)y - 2 = 0 is tangent to the circle (x-1)^2 + (y-1)^2 = 1, then the minimum value of mn is 3 + 2√2. -/
theorem min_mn_tangent_line_circle (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_tangent : ∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 ≥ 1) :
  ∃ (min_mn : ℝ), min_mn = 3 + 2 * Real.sqrt 2 ∧ m * n ≥ min_mn := by
  sorry

end NUMINAMATH_CALUDE_min_mn_tangent_line_circle_l3851_385182


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_satisfying_conditions_l3851_385151

theorem infinitely_many_pairs_satisfying_conditions :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ,
    (Nat.gcd (a n) (a (n + 1)) = 1) ∧
    ((a n) ∣ ((a (n + 1))^2 - 5)) ∧
    ((a (n + 1)) ∣ ((a n)^2 - 5))) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_satisfying_conditions_l3851_385151


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3851_385103

/-- Proves that in an arithmetic sequence with a₁ = 2 and a₃ = 8, the common difference is 3 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 1 = 2)  -- First term is 2
  (h3 : a 3 = 8)  -- Third term is 8
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 2 - a 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3851_385103


namespace NUMINAMATH_CALUDE_field_trip_attendance_l3851_385124

/-- The number of people who went on the field trip -/
def total_people (num_vans : ℕ) (num_buses : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ) : ℕ :=
  num_vans * people_per_van + num_buses * people_per_bus

/-- Theorem stating that the total number of people on the field trip is 342 -/
theorem field_trip_attendance : total_people 9 10 8 27 = 342 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_attendance_l3851_385124


namespace NUMINAMATH_CALUDE_correct_observation_value_l3851_385139

theorem correct_observation_value
  (n : ℕ)
  (initial_mean : ℝ)
  (wrong_value : ℝ)
  (corrected_mean : ℝ)
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : wrong_value = 21)
  (h4 : corrected_mean = 36.54)
  : ∃ (correct_value : ℝ),
    n * corrected_mean = n * initial_mean - wrong_value + correct_value ∧
    correct_value = 48 :=
by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l3851_385139


namespace NUMINAMATH_CALUDE_cubic_sum_l3851_385155

theorem cubic_sum (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 12) : x^3 + y^3 = 224 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_l3851_385155


namespace NUMINAMATH_CALUDE_N_mod_five_l3851_385165

def base_nine_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

def N : Nat :=
  base_nine_to_decimal [2, 5, 0, 0, 0, 0, 0, 6, 0, 0, 7, 2]

theorem N_mod_five : N % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_N_mod_five_l3851_385165


namespace NUMINAMATH_CALUDE_family_pizza_order_correct_l3851_385125

def family_pizza_order (adults : Nat) (children : Nat) (adult_slices : Nat) (child_slices : Nat) (slices_per_pizza : Nat) : Nat :=
  let total_slices := adults * adult_slices + children * child_slices
  (total_slices + slices_per_pizza - 1) / slices_per_pizza

theorem family_pizza_order_correct :
  family_pizza_order 2 12 5 2 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_family_pizza_order_correct_l3851_385125


namespace NUMINAMATH_CALUDE_difference_of_squares_l3851_385143

theorem difference_of_squares (a b : ℝ) : (a + b) * (b - a) = b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3851_385143


namespace NUMINAMATH_CALUDE_jungkook_balls_count_l3851_385161

/-- The number of boxes Jungkook has -/
def num_boxes : ℕ := 3

/-- The number of balls in each box -/
def balls_per_box : ℕ := 2

/-- The total number of balls Jungkook has -/
def total_balls : ℕ := num_boxes * balls_per_box

theorem jungkook_balls_count : total_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_balls_count_l3851_385161


namespace NUMINAMATH_CALUDE_positive_cubes_inequality_l3851_385185

theorem positive_cubes_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_cubes_inequality_l3851_385185


namespace NUMINAMATH_CALUDE_marble_jar_problem_l3851_385193

theorem marble_jar_problem (total_marbles : ℕ) : 
  (∃ (marbles_per_person : ℕ), 
    total_marbles = 18 * marbles_per_person ∧ 
    total_marbles = 20 * (marbles_per_person - 1)) → 
  total_marbles = 180 := by
sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l3851_385193


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3851_385174

theorem right_triangle_hypotenuse (a b : ℝ) (ha : a = 24) (hb : b = 32) :
  Real.sqrt (a^2 + b^2) = 40 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3851_385174


namespace NUMINAMATH_CALUDE_judy_pencil_cost_l3851_385120

/-- Calculates the cost of pencils for a given number of days based on weekly usage and pack price -/
def pencil_cost (weekly_usage : ℕ) (days_per_week : ℕ) (pencils_per_pack : ℕ) (pack_price : ℕ) (total_days : ℕ) : ℕ :=
  let daily_usage := weekly_usage / days_per_week
  let total_pencils := daily_usage * total_days
  let packs_needed := (total_pencils + pencils_per_pack - 1) / pencils_per_pack
  packs_needed * pack_price

theorem judy_pencil_cost : pencil_cost 10 5 30 4 45 = 12 := by
  sorry

#eval pencil_cost 10 5 30 4 45

end NUMINAMATH_CALUDE_judy_pencil_cost_l3851_385120
