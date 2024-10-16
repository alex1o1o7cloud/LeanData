import Mathlib

namespace NUMINAMATH_CALUDE_sequence_sum_equals_n_squared_l1928_192817

def sequence_sum (n : ℕ) : ℕ :=
  (List.range (n + 1)).sum + (List.range n).sum

theorem sequence_sum_equals_n_squared (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_n_squared_l1928_192817


namespace NUMINAMATH_CALUDE_positive_sum_l1928_192894

theorem positive_sum (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 1 < z ∧ z < 2) : 
  0 < y + z := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_l1928_192894


namespace NUMINAMATH_CALUDE_solve_equation_l1928_192871

theorem solve_equation (x : ℝ) (h : (8 / x) + 6 = 8) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1928_192871


namespace NUMINAMATH_CALUDE_special_hexagon_area_l1928_192884

/-- An equilateral hexagon with specific interior angles -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side_length : ℝ
  -- Interior angles of the hexagon in radians
  angles : Fin 6 → ℝ
  -- The hexagon is equilateral
  equilateral : side_length = 1
  -- The interior angles are as specified
  angle_values : angles = ![π/2, 2*π/3, 5*π/6, π/2, 2*π/3, 5*π/6]

/-- The area of the special hexagon -/
def area (h : SpecialHexagon) : ℝ := sorry

/-- Theorem stating the area of the special hexagon -/
theorem special_hexagon_area (h : SpecialHexagon) : area h = (3 + Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_special_hexagon_area_l1928_192884


namespace NUMINAMATH_CALUDE_digit_sum_l1928_192841

/-- Given digits c and d, if 5c * d4 = 1200, then c + d = 2 -/
theorem digit_sum (c d : ℕ) : 
  c < 10 → d < 10 → (50 + c) * (10 * d + 4) = 1200 → c + d = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_l1928_192841


namespace NUMINAMATH_CALUDE_tangent_slope_squared_l1928_192865

/-- A line with slope m passing through the point (0, 2) -/
def line (m : ℝ) (x : ℝ) : ℝ := m * x + 2

/-- An ellipse centered at the origin with semi-major axis 3 and semi-minor axis 1 -/
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

/-- The condition for the line to be tangent to the ellipse -/
def is_tangent (m : ℝ) : Prop :=
  ∃! x, ellipse x (line m x)

theorem tangent_slope_squared (m : ℝ) :
  is_tangent m → m^2 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_squared_l1928_192865


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l1928_192887

/-- A regular nonagon is a 9-sided polygon with all sides and angles equal. -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices. -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- The set of all diagonals in a regular nonagon. -/
def AllDiagonals (n : RegularNonagon) : Set (Diagonal n) := sorry

/-- Two diagonals intersect if they cross each other inside the nonagon. -/
def Intersect (n : RegularNonagon) (d1 d2 : Diagonal n) : Prop := sorry

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes. -/
def Probability (n : RegularNonagon) (event : Set (Diagonal n × Diagonal n)) : ℚ := sorry

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  Probability n {p : Diagonal n × Diagonal n | Intersect n p.1 p.2} = 14/39 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l1928_192887


namespace NUMINAMATH_CALUDE_exists_fixed_point_with_iteration_l1928_192813

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℕ → ℕ) : Prop :=
  (∀ x, 1 ≤ f x - x ∧ f x - x ≤ 2019) ∧
  (∀ x, f (f x) % 2019 = x % 2019)

/-- The main theorem -/
theorem exists_fixed_point_with_iteration (f : ℕ → ℕ) (h : SatisfyingFunction f) :
  ∃ x, ∀ k, f^[k] x = x + 2019 * k :=
sorry

end NUMINAMATH_CALUDE_exists_fixed_point_with_iteration_l1928_192813


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1928_192896

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  3 * X^4 - 8 * X^3 + 20 * X^2 - 7 * X + 13 = 
  (X^2 + 5 * X - 3) * q + (168 * X^2 + 44 * X + 85) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1928_192896


namespace NUMINAMATH_CALUDE_product_of_repeating_third_and_nine_l1928_192872

/-- The repeating decimal 0.333... -/
def repeating_third : ℚ := 1/3

theorem product_of_repeating_third_and_nine :
  repeating_third * 9 = 3 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_third_and_nine_l1928_192872


namespace NUMINAMATH_CALUDE_first_number_proof_l1928_192850

theorem first_number_proof (x : ℝ) (h : x / 14.5 = 175) : x = 2537.5 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l1928_192850


namespace NUMINAMATH_CALUDE_complement_union_problem_l1928_192838

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_problem : (U \ A) ∪ B = {0, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l1928_192838


namespace NUMINAMATH_CALUDE_matrix_power_vector_l1928_192836

theorem matrix_power_vector (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B.mulVec (![3, -1]) = ![12, -4] →
  (B^4).mulVec (![3, -1]) = ![768, -256] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_vector_l1928_192836


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1928_192803

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1928_192803


namespace NUMINAMATH_CALUDE_intersection_problem_complement_problem_l1928_192868

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}
def U : Set ℝ := Set.univ

theorem intersection_problem (a : ℝ) :
  (A ∩ B a = {2}) → (a = -1 ∨ a = -3) :=
by sorry

theorem complement_problem (a : ℝ) :
  (A ∩ (U \ B a) = A) → 
  (a < -3 ∨ (-3 < a ∧ a < -1-Real.sqrt 3) ∨ 
   (-1-Real.sqrt 3 < a ∧ a < -1) ∨ 
   (-1 < a ∧ a < -1+Real.sqrt 3) ∨ 
   a > -1+Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_problem_complement_problem_l1928_192868


namespace NUMINAMATH_CALUDE_T_greater_than_N_l1928_192824

/-- Represents an 8x8 chessboard -/
def Board := Fin 8 → Fin 8 → Bool

/-- Represents a domino placement on the board -/
def DominoPlacement := List (Fin 8 × Fin 8 × Bool)

/-- Returns true if the given domino placement is valid on the board -/
def isValidPlacement (board : Board) (placement : DominoPlacement) : Prop :=
  sorry

/-- Counts the number of valid domino placements for a given number of dominoes -/
def countPlacements (n : Nat) : Nat :=
  sorry

/-- The number of ways to place 32 dominoes -/
def N : Nat := countPlacements 32

/-- The number of ways to place 24 dominoes -/
def T : Nat := countPlacements 24

/-- Theorem stating that T is greater than N -/
theorem T_greater_than_N : T > N := by
  sorry

end NUMINAMATH_CALUDE_T_greater_than_N_l1928_192824


namespace NUMINAMATH_CALUDE_prob_win_at_least_once_l1928_192858

-- Define the probability of winning a single game
def prob_win_single : ℚ := 1 / 9

-- Define the probability of losing a single game
def prob_lose_single : ℚ := 1 - prob_win_single

-- Define the number of games played
def num_games : ℕ := 3

-- Theorem statement
theorem prob_win_at_least_once :
  1 - prob_lose_single ^ num_games = 217 / 729 := by
  sorry

end NUMINAMATH_CALUDE_prob_win_at_least_once_l1928_192858


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequences_l1928_192878

/-- Geometric sequence with a₁ = 2 and a₄ = 16 -/
def geometric_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2 * (2 ^ (n - 1))

/-- Arithmetic sequence with b₃ = a₃ and b₅ = a₅ -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  12 * n - 28

/-- Sum of first n terms of the arithmetic sequence -/
def arithmetic_sum (n : ℕ) : ℝ :=
  6 * n^2 - 22 * n

theorem geometric_arithmetic_sequences :
  (∀ n, geometric_sequence n = 2^n) ∧
  (∀ n, arithmetic_sequence n = 12 * n - 28) ∧
  (∀ n, arithmetic_sum n = 6 * n^2 - 22 * n) ∧
  geometric_sequence 1 = 2 ∧
  geometric_sequence 4 = 16 ∧
  arithmetic_sequence 3 = geometric_sequence 3 ∧
  arithmetic_sequence 5 = geometric_sequence 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequences_l1928_192878


namespace NUMINAMATH_CALUDE_inequality_implies_sum_nonpositive_l1928_192891

theorem inequality_implies_sum_nonpositive 
  {a b x y : ℝ} 
  (h1 : 1 < a) 
  (h2 : a < b) 
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) : 
  x + y ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_sum_nonpositive_l1928_192891


namespace NUMINAMATH_CALUDE_largest_zero_S_l1928_192808

def S : ℕ → ℤ
  | 0 => 0
  | n + 1 => S n + (n + 1) * (if S n < n + 1 then 1 else -1)

theorem largest_zero_S : ∃ k : ℕ, k ≤ 2010 ∧ S k = 0 ∧ ∀ m : ℕ, m ≤ 2010 ∧ m > k → S m ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_zero_S_l1928_192808


namespace NUMINAMATH_CALUDE_juniper_has_six_bones_l1928_192888

/-- Calculates the number of bones Juniper has remaining after her master doubles 
    her initial number of bones and the neighbor's dog steals two bones. -/
def junipersBones (initialBones : ℕ) : ℕ :=
  2 * initialBones - 2

/-- Theorem stating that Juniper has 6 bones remaining after the events. -/
theorem juniper_has_six_bones : junipersBones 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_juniper_has_six_bones_l1928_192888


namespace NUMINAMATH_CALUDE_batsman_85_run_innings_l1928_192843

/-- Represents a batsman's scoring record -/
structure Batsman where
  totalRuns : ℕ
  totalInnings : ℕ

/-- Calculate the average score of a batsman -/
def average (b : Batsman) : ℚ :=
  (b.totalRuns : ℚ) / b.totalInnings

/-- The innings in which the batsman scored 85 -/
def targetInnings (b : Batsman) : ℕ := b.totalInnings

theorem batsman_85_run_innings (b : Batsman) 
  (h1 : average b = 37)
  (h2 : average { totalRuns := b.totalRuns - 85, totalInnings := b.totalInnings - 1 } = 34) :
  targetInnings b = 17 := by
  sorry

end NUMINAMATH_CALUDE_batsman_85_run_innings_l1928_192843


namespace NUMINAMATH_CALUDE_line_l_equation_l1928_192828

/-- The fixed point A through which the line mx - y - m + 2 = 0 always passes -/
def A : ℝ × ℝ := (1, 2)

/-- The slope of the line 2x + y - 2 = 0 -/
def k : ℝ := -2

/-- The equation of the line l passing through A and parallel to 2x + y - 2 = 0 -/
def line_l (x y : ℝ) : Prop := 2 * x + y - 4 = 0

theorem line_l_equation : ∀ m : ℝ, 
  (m * A.1 - A.2 - m + 2 = 0) → 
  (∀ x y : ℝ, line_l x y ↔ y - A.2 = k * (x - A.1)) :=
by sorry

end NUMINAMATH_CALUDE_line_l_equation_l1928_192828


namespace NUMINAMATH_CALUDE_point_translation_l1928_192839

def translate_point (x y dx dy : ℝ) : ℝ × ℝ :=
  (x + dx, y - dy)

theorem point_translation :
  let initial_point : ℝ × ℝ := (1, 2)
  let right_translation : ℝ := 1
  let down_translation : ℝ := 3
  translate_point initial_point.1 initial_point.2 right_translation down_translation = (2, -1) := by
sorry

end NUMINAMATH_CALUDE_point_translation_l1928_192839


namespace NUMINAMATH_CALUDE_bus_average_speed_with_stoppages_l1928_192801

/-- Proves that given a bus with an average speed of 60 km/hr excluding stoppages
    and stopping for 45 minutes per hour, the average speed including stoppages is 15 km/hr. -/
theorem bus_average_speed_with_stoppages
  (speed_without_stoppages : ℝ)
  (stopping_time : ℝ)
  (h1 : speed_without_stoppages = 60)
  (h2 : stopping_time = 45) :
  let moving_time : ℝ := 60 - stopping_time
  let distance_covered : ℝ := speed_without_stoppages * (moving_time / 60)
  let speed_with_stoppages : ℝ := distance_covered
  speed_with_stoppages = 15 := by
  sorry

end NUMINAMATH_CALUDE_bus_average_speed_with_stoppages_l1928_192801


namespace NUMINAMATH_CALUDE_rectangle_dimension_relationship_l1928_192877

/-- Given a rectangle with perimeter 20m, prove that the relationship between its length y and width x is y = -x + 10 -/
theorem rectangle_dimension_relationship (x y : ℝ) : 
  (2 * (x + y) = 20) → (y = -x + 10) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_relationship_l1928_192877


namespace NUMINAMATH_CALUDE_inequality_problem_l1928_192805

theorem inequality_problem :
  (∀ x : ℝ, |x + 7| + |x - 1| ≥ 8) ∧
  (¬ ∃ m : ℝ, m > 8 ∧ ∀ x : ℝ, |x + 7| + |x - 1| ≥ m) ∧
  (∀ x : ℝ, |x - 3| - 2*x ≤ 4 ↔ x ≥ -1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l1928_192805


namespace NUMINAMATH_CALUDE_min_distance_complex_l1928_192848

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 1 - 2*I) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_complex_l1928_192848


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1928_192897

theorem shortest_altitude_right_triangle :
  ∀ (a b c h : ℝ),
    a = 9 ∧ b = 12 ∧ c = 15 →
    a^2 + b^2 = c^2 →
    (1/2) * a * b = (1/2) * c * h →
    h = 7.2 :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1928_192897


namespace NUMINAMATH_CALUDE_three_std_dev_below_mean_l1928_192853

/-- Given a distribution with mean 16.2 and standard deviation 2.3,
    the value 3 standard deviations below the mean is 9.3 -/
theorem three_std_dev_below_mean (μ : ℝ) (σ : ℝ) 
  (h_mean : μ = 16.2) (h_std_dev : σ = 2.3) :
  μ - 3 * σ = 9.3 := by
  sorry

end NUMINAMATH_CALUDE_three_std_dev_below_mean_l1928_192853


namespace NUMINAMATH_CALUDE_max_b_value_l1928_192842

theorem max_b_value (x b : ℤ) : 
  x^2 + b*x = -20 → 
  b > 0 → 
  ∃ (y : ℤ), x^2 + y*x = -20 ∧ y > 0 → 
  y ≤ 21 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l1928_192842


namespace NUMINAMATH_CALUDE_max_value_theorem_l1928_192849

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hsum : a + b + c = 3) :
  (a * b / (a + b)) + (b * c / (b + c)) + (c * a / (c + a)) ≤ 3 / 2 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 3 ∧
    (a' * b' / (a' + b')) + (b' * c' / (b' + c')) + (c' * a' / (c' + a')) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1928_192849


namespace NUMINAMATH_CALUDE_triangle_area_equivalence_l1928_192809

/-- Given a triangle with sides a, b, c, semi-perimeter s, and opposite angles α, β, γ,
    prove that the area formula using sines of half-angles is equivalent to Heron's formula. -/
theorem triangle_area_equivalence (a b c s : ℝ) (α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_semi_perimeter : s = (a + b + c) / 2)
  (h_angles : α + β + γ = Real.pi) :
  Real.sqrt (a * b * c * s * Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2)) = 
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) := by
sorry

end NUMINAMATH_CALUDE_triangle_area_equivalence_l1928_192809


namespace NUMINAMATH_CALUDE_train_speed_theorem_l1928_192823

-- Define the given constants
def train_length : ℝ := 110
def bridge_length : ℝ := 170
def crossing_time : ℝ := 16.7986561075114

-- Define the theorem
theorem train_speed_theorem :
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * 3.6
  speed_kmh = 60 := by sorry

end NUMINAMATH_CALUDE_train_speed_theorem_l1928_192823


namespace NUMINAMATH_CALUDE_fill_675_cans_l1928_192861

/-- A machine that fills paint cans at a specific rate -/
structure PaintMachine where
  cans_per_batch : ℕ
  minutes_per_batch : ℕ

/-- Calculate the time needed to fill a given number of cans -/
def time_to_fill (machine : PaintMachine) (total_cans : ℕ) : ℕ := 
  (total_cans * machine.minutes_per_batch + machine.cans_per_batch - 1) / machine.cans_per_batch

/-- Theorem: The given machine takes 36 minutes to fill 675 cans -/
theorem fill_675_cans (machine : PaintMachine) 
  (h1 : machine.cans_per_batch = 150) 
  (h2 : machine.minutes_per_batch = 8) : 
  time_to_fill machine 675 = 36 := by sorry

end NUMINAMATH_CALUDE_fill_675_cans_l1928_192861


namespace NUMINAMATH_CALUDE_probability_above_parabola_l1928_192834

def is_single_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def above_parabola (a b : ℕ) : Prop := ∀ x : ℚ, b > a * x^2 + b * x

def count_valid_pairs : ℕ := 69

def total_pairs : ℕ := 81

theorem probability_above_parabola :
  (count_valid_pairs : ℚ) / (total_pairs : ℚ) = 23 / 27 := by sorry

end NUMINAMATH_CALUDE_probability_above_parabola_l1928_192834


namespace NUMINAMATH_CALUDE_faulty_balance_measurement_l1928_192882

/-- Represents a faulty balance with unequal arm lengths -/
structure FaultyBalance where
  m : ℝ  -- Length of one arm
  n : ℝ  -- Length of the other arm
  hm : m > 0
  hn : n > 0
  hneq : m ≠ n

/-- Theorem: For a faulty balance, the arithmetic mean of measurements is greater than the true weight -/
theorem faulty_balance_measurement (balance : FaultyBalance) (a b G : ℝ)
  (ha : a > 0) (hb : b > 0) (hG : G > 0)
  (heq1 : balance.m * a = balance.n * G)
  (heq2 : balance.n * b = balance.m * G) :
  (a + b) / 2 > G := by
  sorry

end NUMINAMATH_CALUDE_faulty_balance_measurement_l1928_192882


namespace NUMINAMATH_CALUDE_product_change_l1928_192818

theorem product_change (a b : ℝ) (h : a * b = 1620) :
  (4 * a) * (b / 2) = 3240 := by
  sorry

end NUMINAMATH_CALUDE_product_change_l1928_192818


namespace NUMINAMATH_CALUDE_train_length_calculation_l1928_192880

/-- Two trains of equal length passing each other -/
def train_passing_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) : Prop :=
  faster_speed > slower_speed ∧
  faster_speed = 75 ∧
  slower_speed = 60 ∧
  passing_time = 45 ∧
  ∃ (train_length : ℝ),
    train_length = (faster_speed - slower_speed) * passing_time * (5 / 18) / 2

theorem train_length_calculation (faster_speed slower_speed passing_time : ℝ) :
  train_passing_problem faster_speed slower_speed passing_time →
  ∃ (train_length : ℝ), train_length = 93.75 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1928_192880


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1928_192847

theorem quadratic_inequality_solution_sets
  (a b : ℝ)
  (h : Set.Ioo (-1 : ℝ) (1/2) = {x | a * x^2 + b * x + 3 > 0}) :
  Set.Ioo (-1 : ℝ) 2 = {x | 3 * x^2 + b * x + a < 0} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1928_192847


namespace NUMINAMATH_CALUDE_carrot_stick_calories_prove_carrot_stick_calories_l1928_192819

theorem carrot_stick_calories : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_calories burger_calories cookie_count cookie_calories carrot_stick_count carrot_stick_calories =>
    (total_calories = burger_calories + cookie_count * cookie_calories + carrot_stick_count * carrot_stick_calories) →
    (total_calories = 750) →
    (burger_calories = 400) →
    (cookie_count = 5) →
    (cookie_calories = 50) →
    (carrot_stick_count = 5) →
    (carrot_stick_calories = 20)

theorem prove_carrot_stick_calories : carrot_stick_calories 750 400 5 50 5 20 :=
by sorry

end NUMINAMATH_CALUDE_carrot_stick_calories_prove_carrot_stick_calories_l1928_192819


namespace NUMINAMATH_CALUDE_money_division_l1928_192815

theorem money_division (a b c : ℝ) : 
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a = b + 30 →
  a + b + c = 280 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l1928_192815


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1928_192883

/-- The length of the major axis of an ellipse defined by the equation 16x^2 + 9y^2 = 144 is 8 -/
theorem ellipse_major_axis_length : 
  let ellipse_eq := fun (x y : ℝ) => 16 * x^2 + 9 * y^2 = 144
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
    (∀ x y, ellipse_eq x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    2 * a = 8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1928_192883


namespace NUMINAMATH_CALUDE_power_equality_l1928_192875

theorem power_equality (m : ℕ) : 9^4 = 3^m → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1928_192875


namespace NUMINAMATH_CALUDE_correct_operation_l1928_192822

theorem correct_operation (a : ℝ) : 3 * a^3 * 2 * a^2 = 6 * a^5 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1928_192822


namespace NUMINAMATH_CALUDE_total_students_in_high_school_l1928_192857

-- Define the number of students in each grade
def freshman_students : ℕ := sorry
def sophomore_students : ℕ := sorry
def senior_students : ℕ := 1200

-- Define the sample sizes
def freshman_sample : ℕ := 75
def sophomore_sample : ℕ := 60
def senior_sample : ℕ := 50

-- Define the total sample size
def total_sample : ℕ := 185

-- Theorem statement
theorem total_students_in_high_school :
  freshman_students + sophomore_students + senior_students = 4440 :=
by
  -- Assuming the stratified sampling method ensures equal ratios
  have h1 : (freshman_sample : ℚ) / freshman_students = (senior_sample : ℚ) / senior_students := sorry
  have h2 : (sophomore_sample : ℚ) / sophomore_students = (senior_sample : ℚ) / senior_students := sorry
  
  -- The total sample size is the sum of individual sample sizes
  have h3 : freshman_sample + sophomore_sample + senior_sample = total_sample := sorry

  sorry -- Complete the proof

end NUMINAMATH_CALUDE_total_students_in_high_school_l1928_192857


namespace NUMINAMATH_CALUDE_min_max_quadratic_form_l1928_192856

theorem min_max_quadratic_form (x y : ℝ) (h : 2 * x^2 + 3 * x * y + y^2 = 2) :
  (∀ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 → 4 * a^2 + 4 * a * b + 3 * b^2 ≥ 4) ∧
  (∀ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 → 4 * a^2 + 4 * a * b + 3 * b^2 ≤ 6) ∧
  (∃ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 ∧ 4 * a^2 + 4 * a * b + 3 * b^2 = 4) ∧
  (∃ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 ∧ 4 * a^2 + 4 * a * b + 3 * b^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_max_quadratic_form_l1928_192856


namespace NUMINAMATH_CALUDE_perpendicular_segments_sum_maximum_l1928_192899

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point within the circle's disk
structure PointInDisk (c : Circle) where
  point : ℝ × ℝ
  in_disk : Real.sqrt ((point.1 - c.center.1)^2 + (point.2 - c.center.2)^2) ≤ c.radius

-- Define two perpendicular line segments from a point to the circle's boundary
structure PerpendicularSegments (c : Circle) (p : PointInDisk c) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ
  on_circle1 : Real.sqrt ((endpoint1.1 - c.center.1)^2 + (endpoint1.2 - c.center.2)^2) = c.radius
  on_circle2 : Real.sqrt ((endpoint2.1 - c.center.1)^2 + (endpoint2.2 - c.center.2)^2) = c.radius
  perpendicular : (endpoint1.1 - p.point.1) * (endpoint2.1 - p.point.1) + 
                  (endpoint1.2 - p.point.2) * (endpoint2.2 - p.point.2) = 0

-- Theorem statement
theorem perpendicular_segments_sum_maximum (c : Circle) (p : PointInDisk c) 
  (segments : PerpendicularSegments c p) :
  ∃ (max_segments : PerpendicularSegments c p),
    (Real.sqrt ((max_segments.endpoint1.1 - p.point.1)^2 + (max_segments.endpoint1.2 - p.point.2)^2) +
     Real.sqrt ((max_segments.endpoint2.1 - p.point.1)^2 + (max_segments.endpoint2.2 - p.point.2)^2)) ≥
    (Real.sqrt ((segments.endpoint1.1 - p.point.1)^2 + (segments.endpoint1.2 - p.point.2)^2) +
     Real.sqrt ((segments.endpoint2.1 - p.point.1)^2 + (segments.endpoint2.2 - p.point.2)^2)) ∧
    (Real.sqrt ((max_segments.endpoint1.1 - p.point.1)^2 + (max_segments.endpoint1.2 - p.point.2)^2) =
     Real.sqrt ((max_segments.endpoint2.1 - p.point.1)^2 + (max_segments.endpoint2.2 - p.point.2)^2)) ∧
    (Real.sqrt ((max_segments.endpoint1.1 - max_segments.endpoint2.1)^2 + 
                (max_segments.endpoint1.2 - max_segments.endpoint2.2)^2) = 2 * c.radius) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_segments_sum_maximum_l1928_192899


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l1928_192840

theorem hamburgers_left_over (hamburgers_made : ℕ) (hamburgers_served : ℕ) : 
  hamburgers_made = 15 → hamburgers_served = 8 → hamburgers_made - hamburgers_served = 7 := by
  sorry

#check hamburgers_left_over

end NUMINAMATH_CALUDE_hamburgers_left_over_l1928_192840


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l1928_192846

theorem simplify_nested_expression : -(-(-|(-1)|^2)^3)^4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l1928_192846


namespace NUMINAMATH_CALUDE_accessories_cost_l1928_192835

theorem accessories_cost (computer_cost : ℝ) (playstation_worth : ℝ) (pocket_payment : ℝ)
  (h1 : computer_cost = 700)
  (h2 : playstation_worth = 400)
  (h3 : pocket_payment = 580) :
  let playstation_sold := playstation_worth * 0.8
  let total_available := playstation_sold + pocket_payment
  let accessories_cost := total_available - computer_cost
  accessories_cost = 200 := by sorry

end NUMINAMATH_CALUDE_accessories_cost_l1928_192835


namespace NUMINAMATH_CALUDE_roberto_salary_raise_l1928_192867

def starting_salary : ℝ := 80000
def previous_salary : ℝ := starting_salary * 1.4
def current_salary : ℝ := 134400

theorem roberto_salary_raise :
  (current_salary - previous_salary) / previous_salary * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_roberto_salary_raise_l1928_192867


namespace NUMINAMATH_CALUDE_tub_capacity_l1928_192873

/-- Calculates the capacity of a tub given specific filling conditions -/
theorem tub_capacity 
  (flow_rate : ℕ) 
  (escape_rate : ℕ) 
  (cycle_time : ℕ) 
  (total_time : ℕ) 
  (h1 : flow_rate = 12)
  (h2 : escape_rate = 1)
  (h3 : cycle_time = 2)
  (h4 : total_time = 24) :
  (total_time / cycle_time) * (flow_rate - escape_rate - escape_rate) = 120 :=
by sorry

end NUMINAMATH_CALUDE_tub_capacity_l1928_192873


namespace NUMINAMATH_CALUDE_sum_of_unknown_numbers_l1928_192860

def known_numbers : List ℕ := [690, 744, 745, 747, 748, 749, 752, 752, 753, 755, 760, 769]

theorem sum_of_unknown_numbers 
  (total_count : ℕ) 
  (average : ℕ) 
  (h1 : total_count = 15) 
  (h2 : average = 750) 
  (h3 : known_numbers.length = 12) : 
  (total_count * average) - known_numbers.sum = 2336 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_unknown_numbers_l1928_192860


namespace NUMINAMATH_CALUDE_trig_expression_equals_neg_sqrt_three_l1928_192827

theorem trig_expression_equals_neg_sqrt_three : 
  (2 * Real.sin (10 * π / 180) - Real.cos (20 * π / 180)) / Real.cos (70 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_neg_sqrt_three_l1928_192827


namespace NUMINAMATH_CALUDE_shirt_tie_belt_combinations_l1928_192811

/-- Given a number of shirts, ties, and belts, calculates the total number of
    shirt-and-tie or shirt-and-belt combinations -/
def total_combinations (shirts : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * ties + shirts * belts

/-- Theorem stating that with 7 shirts, 6 ties, and 4 belts, 
    the total number of combinations is 70 -/
theorem shirt_tie_belt_combinations :
  total_combinations 7 6 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_belt_combinations_l1928_192811


namespace NUMINAMATH_CALUDE_mississippi_permutations_l1928_192812

theorem mississippi_permutations :
  let total_letters : ℕ := 11
  let m_count : ℕ := 1
  let i_count : ℕ := 4
  let s_count : ℕ := 4
  let p_count : ℕ := 2
  (Nat.factorial total_letters) / 
  (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count) = 34650 := by
  sorry

end NUMINAMATH_CALUDE_mississippi_permutations_l1928_192812


namespace NUMINAMATH_CALUDE_number_of_possible_scores_l1928_192876

-- Define the scoring system
def problem_scores : List Nat := [1, 2, 3, 4]
def time_bonuses : List Nat := [1, 2, 3, 4]
def all_correct_bonus : Nat := 20

-- Function to calculate all possible scores
def calculate_scores : List Nat :=
  let base_scores := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let multiplied_scores := 
    List.join (base_scores.map (λ s => time_bonuses.map (λ t => s * t)))
  let all_correct_scores := 
    time_bonuses.map (λ t => 10 * t + all_correct_bonus)
  List.eraseDups (multiplied_scores ++ all_correct_scores)

-- Theorem statement
theorem number_of_possible_scores : 
  calculate_scores.length = 25 := by sorry

end NUMINAMATH_CALUDE_number_of_possible_scores_l1928_192876


namespace NUMINAMATH_CALUDE_cos_100_in_terms_of_sin_80_l1928_192855

theorem cos_100_in_terms_of_sin_80 (a : ℝ) (h : Real.sin (80 * π / 180) = a) :
  Real.cos (100 * π / 180) = -Real.sqrt (1 - a^2) := by
  sorry

end NUMINAMATH_CALUDE_cos_100_in_terms_of_sin_80_l1928_192855


namespace NUMINAMATH_CALUDE_equation_positive_root_l1928_192829

-- Define the equation
def equation (x a : ℝ) : Prop :=
  (x - a) / (x - 1) - 3 / x = 1

-- State the theorem
theorem equation_positive_root (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ equation x a) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_positive_root_l1928_192829


namespace NUMINAMATH_CALUDE_shortest_distance_to_parabola_l1928_192863

/-- The parabola defined by x = 2y^2 -/
def parabola (y : ℝ) : ℝ := 2 * y^2

/-- The point from which we measure the distance -/
def point : ℝ × ℝ := (8, 14)

/-- The shortest distance between the point and the parabola -/
def shortest_distance : ℝ := 26

/-- Theorem stating that the shortest distance between the point (8,14) and the parabola x = 2y^2 is 26 -/
theorem shortest_distance_to_parabola :
  ∃ (y : ℝ), 
    shortest_distance = 
      Real.sqrt ((parabola y - point.1)^2 + (y - point.2)^2) ∧
    ∀ (z : ℝ), 
      Real.sqrt ((parabola z - point.1)^2 + (z - point.2)^2) ≥ shortest_distance :=
by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_to_parabola_l1928_192863


namespace NUMINAMATH_CALUDE_three_digit_number_difference_l1928_192890

/-- Represents a three-digit number with digits h, t, u from left to right -/
structure ThreeDigitNumber where
  h : ℕ
  t : ℕ
  u : ℕ
  h_lt_10 : h < 10
  t_lt_10 : t < 10
  u_lt_10 : u < 10
  h_gt_u : h > u

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : ℕ :=
  100 * n.h + 10 * n.t + n.u

/-- The reversed value of a three-digit number -/
def ThreeDigitNumber.reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.u + 10 * n.t + n.h

theorem three_digit_number_difference (n : ThreeDigitNumber) :
  n.value - n.reversed_value = 4 → n.h = 9 ∧ n.u = 5 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_difference_l1928_192890


namespace NUMINAMATH_CALUDE_roller_alignment_l1928_192804

/-- The number of rotations needed for alignment of two rollers -/
def alignmentRotations (r1 r2 : ℕ) : ℕ :=
  (Nat.lcm r1 r2) / r1

/-- Theorem: The number of rotations for alignment of rollers with radii 105 and 90 is 6 -/
theorem roller_alignment :
  alignmentRotations 105 90 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roller_alignment_l1928_192804


namespace NUMINAMATH_CALUDE_safe_password_l1928_192879

def digits : List Nat := [6, 2, 5]

def largest_number (digits : List Nat) : Nat :=
  100 * (digits.maximum?.getD 0) + 
  10 * (digits.filter (· ≠ digits.maximum?.getD 0)).maximum?.getD 0 + 
  (digits.filter (· ∉ [digits.maximum?.getD 0, (digits.filter (· ≠ digits.maximum?.getD 0)).maximum?.getD 0])).sum

def smallest_number (digits : List Nat) : Nat :=
  100 * (digits.minimum?.getD 0) + 
  10 * (digits.filter (· ≠ digits.minimum?.getD 0)).minimum?.getD 0 + 
  (digits.filter (· ∉ [digits.minimum?.getD 0, (digits.filter (· ≠ digits.minimum?.getD 0)).minimum?.getD 0])).sum

theorem safe_password : 
  largest_number digits + smallest_number digits = 908 := by
  sorry

end NUMINAMATH_CALUDE_safe_password_l1928_192879


namespace NUMINAMATH_CALUDE_lizzy_money_problem_l1928_192869

/-- The amount of cents Lizzy's father gave her -/
def father_gave : ℕ := 40

/-- The amount of cents Lizzy spent on candy -/
def spent_on_candy : ℕ := 50

/-- The amount of cents Lizzy's uncle gave her -/
def uncle_gave : ℕ := 70

/-- The amount of cents Lizzy has now -/
def current_amount : ℕ := 140

/-- The amount of cents Lizzy's mother gave her -/
def mother_gave : ℕ := 80

theorem lizzy_money_problem :
  mother_gave = current_amount + spent_on_candy - (father_gave + uncle_gave) :=
by sorry

end NUMINAMATH_CALUDE_lizzy_money_problem_l1928_192869


namespace NUMINAMATH_CALUDE_identity_condition_l1928_192851

theorem identity_condition (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) ↔ 
  ((a = 0 ∧ b = 0 ∧ c = 1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = -1) ∨ 
   (a = 0 ∧ b = 1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
   (a = 1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = -1 ∧ b = 0 ∧ c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_identity_condition_l1928_192851


namespace NUMINAMATH_CALUDE_cubic_polynomials_relation_l1928_192866

/-- Two monic cubic polynomials with specific roots and a relation between them -/
theorem cubic_polynomials_relation (k : ℝ) 
  (f g : ℝ → ℝ)
  (hf_monic : ∀ x, f x = x^3 + a * x^2 + b * x + c)
  (hg_monic : ∀ x, g x = x^3 + d * x^2 + e * x + i)
  (hf_roots : (k + 2) * (k + 6) * (f (k + 2)) = 0 ∧ (k + 2) * (k + 6) * (f (k + 6)) = 0)
  (hg_roots : (k + 4) * (k + 8) * (g (k + 4)) = 0 ∧ (k + 4) * (k + 8) * (g (k + 8)) = 0)
  (h_diff : ∀ x, f x - g x = x + k) : 
  k = 7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_relation_l1928_192866


namespace NUMINAMATH_CALUDE_solution_exists_l1928_192859

-- Define the vector type
def Vec2 := Fin 2 → ℝ

-- Define the constants a and b
variable (a b : ℝ)

-- Define the vectors
def v1 : Vec2 := ![1, 4]
def v2 : Vec2 := ![3, -2]
def result : Vec2 := ![5, 6]

-- Define vector addition and scalar multiplication
def add (u v : Vec2) : Vec2 := λ i => u i + v i
def smul (c : ℝ) (v : Vec2) : Vec2 := λ i => c * v i

-- State the theorem
theorem solution_exists :
  ∃ a b : ℝ, add (smul a v1) (smul b v2) = result ∧ a = 2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l1928_192859


namespace NUMINAMATH_CALUDE_no_quadratic_with_discriminant_2019_l1928_192830

theorem no_quadratic_with_discriminant_2019 : ¬ ∃ (a b c : ℤ), b^2 - 4*a*c = 2019 := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_with_discriminant_2019_l1928_192830


namespace NUMINAMATH_CALUDE_seeds_sowed_l1928_192874

/-- Proves that the number of buckets of seeds sowed is 2.75 -/
theorem seeds_sowed (initial : ℝ) (final : ℝ) (h1 : initial = 8.75) (h2 : final = 6) :
  initial - final = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_seeds_sowed_l1928_192874


namespace NUMINAMATH_CALUDE_triangle_side_length_l1928_192892

theorem triangle_side_length 
  (A B C : Real) 
  (AB BC AC : Real) :
  A = π / 3 →
  Real.tan B = 1 / 2 →
  AB = 2 * Real.sqrt 3 + 1 →
  BC = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1928_192892


namespace NUMINAMATH_CALUDE_four_letter_word_count_l1928_192898

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The length of the word -/
def word_length : ℕ := 4

theorem four_letter_word_count : 
  alphabet_size * vowel_count * alphabet_size = 3380 := by
  sorry

#check four_letter_word_count

end NUMINAMATH_CALUDE_four_letter_word_count_l1928_192898


namespace NUMINAMATH_CALUDE_product_of_sums_l1928_192886

theorem product_of_sums : 
  (8 - Real.sqrt 500 + 8 + Real.sqrt 500) * (12 - Real.sqrt 72 + 12 + Real.sqrt 72) = 384 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_l1928_192886


namespace NUMINAMATH_CALUDE_min_people_to_ask_for_hat_color_l1928_192816

/-- Represents the minimum number of people to ask to ensure a majority of truthful answers -/
def min_people_to_ask (knights : ℕ) (civilians : ℕ) : ℕ :=
  civilians + (civilians + 1)

/-- Theorem stating the minimum number of people to ask in the given scenario -/
theorem min_people_to_ask_for_hat_color (knights : ℕ) (civilians : ℕ) 
  (h1 : knights = 50) (h2 : civilians = 15) :
  min_people_to_ask knights civilians = 31 := by
  sorry

#eval min_people_to_ask 50 15

end NUMINAMATH_CALUDE_min_people_to_ask_for_hat_color_l1928_192816


namespace NUMINAMATH_CALUDE_problem_1_l1928_192885

theorem problem_1 : (-1) * (-4) + 2^2 / (7 - 5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1928_192885


namespace NUMINAMATH_CALUDE_remainder_squared_multiply_l1928_192807

theorem remainder_squared_multiply (n a b : ℤ) : 
  n > 0 → b = 3 → a * b ≡ 1 [ZMOD n] → a^2 * b ≡ a [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_remainder_squared_multiply_l1928_192807


namespace NUMINAMATH_CALUDE_lyka_savings_plan_l1928_192895

/-- Calculates the number of weeks needed to save for a smartphone. -/
def weeks_to_save (smartphone_cost : ℕ) (current_savings : ℕ) (weekly_savings : ℕ) : ℕ :=
  (smartphone_cost - current_savings) / weekly_savings

/-- Proves that Lyka needs to save for 8 weeks to buy the smartphone. -/
theorem lyka_savings_plan :
  weeks_to_save 160 40 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_lyka_savings_plan_l1928_192895


namespace NUMINAMATH_CALUDE_modulus_of_Z_l1928_192881

/-- The modulus of the complex number Z = 1 / (i - 1) is equal to √2/2 -/
theorem modulus_of_Z : Complex.abs (1 / (Complex.I - 1)) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_Z_l1928_192881


namespace NUMINAMATH_CALUDE_largest_number_l1928_192826

theorem largest_number : 
  (1 : ℝ) ≥ Real.sqrt 29 - Real.sqrt 21 ∧ 
  (1 : ℝ) ≥ Real.pi / 3.142 ∧ 
  (1 : ℝ) ≥ 5.1 * Real.sqrt 0.0361 ∧ 
  (1 : ℝ) ≥ 6 / (Real.sqrt 13 + Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1928_192826


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1928_192889

def M : Set ℝ := {x | 4 ≤ x ∧ x ≤ 7}
def N : Set ℝ := {3, 5, 8}

theorem intersection_of_M_and_N : M ∩ N = {5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1928_192889


namespace NUMINAMATH_CALUDE_total_amount_spent_l1928_192844

def num_pens : ℕ := 30
def num_pencils : ℕ := 75
def avg_price_pencil : ℚ := 2
def avg_price_pen : ℚ := 16

theorem total_amount_spent : 
  num_pens * avg_price_pen + num_pencils * avg_price_pencil = 630 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_spent_l1928_192844


namespace NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l1928_192852

theorem power_of_eight_sum_equals_power_of_two : ∃ x : ℕ, 8^4 + 8^4 + 8^4 = 2^x ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l1928_192852


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_circle_center_l1928_192810

/-- A circle with center (1, 3) tangent to the line 3x - 4y - 6 = 0 -/
def TangentCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 9}

/-- The line 3x - 4y - 6 = 0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 - 6 = 0}

theorem circle_tangent_to_line :
  (∃ (p : ℝ × ℝ), p ∈ TangentCircle ∧ p ∈ TangentLine) ∧
  (∀ (p : ℝ × ℝ), p ∈ TangentCircle → p ∈ TangentLine → 
    ∀ (q : ℝ × ℝ), q ∈ TangentCircle → q = p) :=
by sorry

theorem circle_center :
  ∀ (p : ℝ × ℝ), p ∈ TangentCircle → (p.1 - 1)^2 + (p.2 - 3)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_circle_center_l1928_192810


namespace NUMINAMATH_CALUDE_green_ball_packs_l1928_192831

/-- Given the number of packs of red and yellow balls, the number of balls per pack,
    and the total number of balls, calculate the number of packs of green balls. -/
theorem green_ball_packs (red_packs yellow_packs : ℕ) (balls_per_pack : ℕ) (total_balls : ℕ) : 
  red_packs = 3 → 
  yellow_packs = 10 → 
  balls_per_pack = 19 → 
  total_balls = 399 → 
  (total_balls - (red_packs + yellow_packs) * balls_per_pack) / balls_per_pack = 8 :=
by sorry

end NUMINAMATH_CALUDE_green_ball_packs_l1928_192831


namespace NUMINAMATH_CALUDE_susans_chairs_l1928_192854

theorem susans_chairs (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : yellow = 4 * red)
  (h2 : blue = yellow - 2)
  (h3 : red + yellow + blue = 43) :
  red = 5 := by
  sorry

end NUMINAMATH_CALUDE_susans_chairs_l1928_192854


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1928_192825

theorem arithmetic_progression_first_term
  (d : ℝ)
  (a₁₂ : ℝ)
  (h₁ : d = 8)
  (h₂ : a₁₂ = 90)
  : ∃ (a₁ : ℝ), a₁₂ = a₁ + (12 - 1) * d ∧ a₁ = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1928_192825


namespace NUMINAMATH_CALUDE_smallest_reducible_fraction_l1928_192821

/-- The smallest positive integer n such that (n-17)/(7n+8) is a non-zero reducible fraction -/
def smallest_reducible_n : ℕ := 144

/-- A fraction a/b is reducible if and only if gcd(a,b) > 1 -/
def is_reducible (a b : ℤ) : Prop := Nat.gcd a.natAbs b.natAbs > 1

/-- Main theorem: 144 is the smallest positive integer n such that (n-17)/(7n+8) is a non-zero reducible fraction -/
theorem smallest_reducible_fraction :
  (∀ n : ℕ, 0 < n → n < smallest_reducible_n → ¬(is_reducible (n - 17) (7 * n + 8))) ∧
  (is_reducible (smallest_reducible_n - 17) (7 * smallest_reducible_n + 8)) :=
sorry

end NUMINAMATH_CALUDE_smallest_reducible_fraction_l1928_192821


namespace NUMINAMATH_CALUDE_prob_red_fifth_black_tenth_correct_l1928_192833

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the color of a card -/
inductive Color
| Red
| Black

/-- Function to determine the color of a card based on its number -/
def card_color (n : Fin 52) : Color :=
  if n.val < 26 then Color.Red else Color.Black

/-- Probability of drawing a red card as the fifth and a black card as the tenth from a shuffled deck -/
def prob_red_fifth_black_tenth (d : Deck) : ℚ :=
  13 / 51

/-- Theorem stating the probability of drawing a red card as the fifth and a black card as the tenth from a shuffled deck -/
theorem prob_red_fifth_black_tenth_correct (d : Deck) :
  prob_red_fifth_black_tenth d = 13 / 51 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_fifth_black_tenth_correct_l1928_192833


namespace NUMINAMATH_CALUDE_male_alligators_mating_season_l1928_192802

/-- Represents the alligator population on Lagoon Island -/
structure AlligatorPopulation where
  males : ℕ
  adult_females : ℕ
  juvenile_females : ℕ

/-- The ratio of males to adult females to juvenile females -/
def population_ratio : AlligatorPopulation := ⟨2, 3, 5⟩

/-- The number of adult females during non-mating season -/
def non_mating_adult_females : ℕ := 15

/-- Theorem stating the number of male alligators during mating season -/
theorem male_alligators_mating_season :
  ∃ (pop : AlligatorPopulation),
    pop.adult_females = non_mating_adult_females ∧
    pop.males * population_ratio.adult_females = population_ratio.males * pop.adult_females ∧
    pop.males = 10 :=
by sorry

end NUMINAMATH_CALUDE_male_alligators_mating_season_l1928_192802


namespace NUMINAMATH_CALUDE_expand_triple_product_l1928_192800

theorem expand_triple_product (x y z : ℝ) :
  (x - 5) * (3 * y + 6) * (z + 4) =
  3 * x * y * z + 6 * x * z - 15 * y * z - 30 * z + 12 * x * y + 24 * x - 60 * y - 120 :=
by sorry

end NUMINAMATH_CALUDE_expand_triple_product_l1928_192800


namespace NUMINAMATH_CALUDE_parabola_ellipse_coincident_foci_l1928_192820

/-- Given a parabola and an ellipse, proves that if their foci coincide, then the parameter of the parabola is 4. -/
theorem parabola_ellipse_coincident_foci (p : ℝ) : 
  (∀ x y, y^2 = 2*p*x → x^2/6 + y^2/2 = 1 → x = p/2 ∧ x = 2) → p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_coincident_foci_l1928_192820


namespace NUMINAMATH_CALUDE_money_division_l1928_192806

theorem money_division (a b c : ℝ) (h1 : a = (1/2) * b) (h2 : b = (1/2) * c) (h3 : c = 208) :
  a + b + c = 364 := by sorry

end NUMINAMATH_CALUDE_money_division_l1928_192806


namespace NUMINAMATH_CALUDE_acute_angle_alpha_l1928_192862

theorem acute_angle_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α = 1 - Real.sqrt 3 * Real.tan (π / 18) * Real.sin α) : 
  α = π / 3.6 := by sorry

end NUMINAMATH_CALUDE_acute_angle_alpha_l1928_192862


namespace NUMINAMATH_CALUDE_xy_equals_zero_l1928_192864

theorem xy_equals_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_zero_l1928_192864


namespace NUMINAMATH_CALUDE_probability_multiple_4_5_7_l1928_192870

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (max : ℕ) (divisor : ℕ) : ℕ :=
  (max / divisor : ℕ)

theorem probability_multiple_4_5_7 (max : ℕ) (h : max = 150) :
  (count_multiples max 4 + count_multiples max 5 + count_multiples max 7
   - count_multiples max 20 - count_multiples max 28 - count_multiples max 35
   + count_multiples max 140) / max = 73 / 150 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_4_5_7_l1928_192870


namespace NUMINAMATH_CALUDE_triangle_acute_angled_l1928_192893

theorem triangle_acute_angled (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (eq : a^3 + b^3 = c^3) :
  c^2 < a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_acute_angled_l1928_192893


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1928_192845

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (9 * a ^ 3 + 14 * a ^ 2 + 2047 * a + 3024 = 0) →
  (9 * b ^ 3 + 14 * b ^ 2 + 2047 * b + 3024 = 0) →
  (9 * c ^ 3 + 14 * c ^ 2 + 2047 * c + 3024 = 0) →
  (a + b) ^ 3 + (b + c) ^ 3 + (c + a) ^ 3 = -58198 / 729 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1928_192845


namespace NUMINAMATH_CALUDE_cyclic_product_sum_theorem_l1928_192814

/-- A permutation of (1, 2, 3, 4, 5, 6) -/
def Permutation := Fin 6 → Fin 6

/-- The cyclic product sum for a given permutation -/
def cyclicProductSum (p : Permutation) : ℕ :=
  (p 0) * (p 1) + (p 1) * (p 2) + (p 2) * (p 3) + (p 3) * (p 4) + (p 4) * (p 5) + (p 5) * (p 0)

/-- Predicate to check if a function is a valid permutation of (1, 2, 3, 4, 5, 6) -/
def isValidPermutation (p : Permutation) : Prop :=
  Function.Injective p ∧ Function.Surjective p

/-- The maximum value of the cyclic product sum -/
def M : ℕ := 79

/-- The number of permutations that achieve the maximum value -/
def N : ℕ := 12

theorem cyclic_product_sum_theorem :
  (∀ p : Permutation, isValidPermutation p → cyclicProductSum p ≤ M) ∧
  (∃! (s : Finset Permutation), s.card = N ∧ 
    ∀ p ∈ s, isValidPermutation p ∧ cyclicProductSum p = M) :=
sorry

end NUMINAMATH_CALUDE_cyclic_product_sum_theorem_l1928_192814


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1928_192832

theorem simplify_and_rationalize (x : ℝ) (h : x = 1 / (1 + 1 / (Real.sqrt 2 + 2))) :
  x = (4 + Real.sqrt 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1928_192832


namespace NUMINAMATH_CALUDE_last_digit_base_9_of_221122211111_base_3_l1928_192837

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def last_digit_base_9 (n : Nat) : Nat :=
  n % 9

theorem last_digit_base_9_of_221122211111_base_3 :
  let y : Nat := base_3_to_10 [1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2]
  last_digit_base_9 y = 6 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_base_9_of_221122211111_base_3_l1928_192837
