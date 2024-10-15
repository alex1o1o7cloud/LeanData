import Mathlib

namespace NUMINAMATH_CALUDE_f_at_two_l1313_131309

/-- Horner's method representation of the polynomial 2x^4 + 3x^3 + 5x - 4 --/
def f (x : ℝ) : ℝ := ((2 * x + 3) * x + 0) * x + 5 * x - 4

/-- Theorem stating that f(2) = 62 --/
theorem f_at_two : f 2 = 62 := by sorry

end NUMINAMATH_CALUDE_f_at_two_l1313_131309


namespace NUMINAMATH_CALUDE_symmetric_series_sum_sqrt_l1313_131357

def symmetric_series (n : ℕ) : ℕ := 
  2 * (n * (n + 1) / 2) + (n + 1)

theorem symmetric_series_sum_sqrt (n : ℕ) : 
  Real.sqrt (symmetric_series n) = (n : ℝ) + 0.5 :=
sorry

end NUMINAMATH_CALUDE_symmetric_series_sum_sqrt_l1313_131357


namespace NUMINAMATH_CALUDE_class_size_l1313_131349

theorem class_size (total : ℕ) (girls_ratio : ℚ) (boys : ℕ) : 
  girls_ratio = 5 / 8 → boys = 60 → total = 160 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1313_131349


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l1313_131315

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_n_value
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a4 : a 4 = 7)
  (h_a3_a6 : a 3 + a 6 = 16)
  (h_an : ∃ n : ℕ, a n = 31) :
  ∃ n : ℕ, a n = 31 ∧ n = 16 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l1313_131315


namespace NUMINAMATH_CALUDE_area_ratio_preserved_under_affine_transformation_l1313_131382

-- Define a polygon type
def Polygon := Set (ℝ × ℝ)

-- Define an affine transformation type
def AffineTransformation := (ℝ × ℝ) → (ℝ × ℝ)

-- Define an area function for polygons
noncomputable def area (P : Polygon) : ℝ := sorry

-- State the theorem
theorem area_ratio_preserved_under_affine_transformation
  (M N : Polygon) (f : AffineTransformation) :
  let M' := f '' M
  let N' := f '' N
  area M / area N = area M' / area N' := by sorry

end NUMINAMATH_CALUDE_area_ratio_preserved_under_affine_transformation_l1313_131382


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l1313_131370

theorem min_value_2x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 1) :
  2 * x + y ≥ 2 * Real.sqrt 2 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x * y = 1 ∧ 2 * x + y = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l1313_131370


namespace NUMINAMATH_CALUDE_frequency_converges_to_probability_l1313_131386

/-- A random event in an experiment. -/
structure Event where
  -- Add necessary fields here
  mk :: -- Constructor

/-- The frequency of an event after a given number of trials. -/
def frequency (e : Event) (n : ℕ) : ℝ :=
  sorry

/-- The probability of an event. -/
def probability (e : Event) : ℝ :=
  sorry

/-- Statement: As the number of trials increases, the frequency of an event
    converges to its probability. -/
theorem frequency_converges_to_probability (e : Event) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |frequency e n - probability e| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_converges_to_probability_l1313_131386


namespace NUMINAMATH_CALUDE_angle_expression_equals_half_l1313_131393

theorem angle_expression_equals_half (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin θ + Real.cos (π - θ)) / (Real.sin (π / 2 - θ) - Real.sin (π + θ)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_equals_half_l1313_131393


namespace NUMINAMATH_CALUDE_largest_integer_in_fraction_inequality_five_satisfies_inequality_largest_integer_is_five_l1313_131391

theorem largest_integer_in_fraction_inequality :
  ∀ x : ℤ, (2 : ℚ) / 5 < (x : ℚ) / 7 ∧ (x : ℚ) / 7 < 8 / 11 → x ≤ 5 :=
by sorry

theorem five_satisfies_inequality :
  (2 : ℚ) / 5 < (5 : ℚ) / 7 ∧ (5 : ℚ) / 7 < 8 / 11 :=
by sorry

theorem largest_integer_is_five :
  ∃! x : ℤ, x = 5 ∧
    ((2 : ℚ) / 5 < (x : ℚ) / 7 ∧ (x : ℚ) / 7 < 8 / 11) ∧
    (∀ y : ℤ, (2 : ℚ) / 5 < (y : ℚ) / 7 ∧ (y : ℚ) / 7 < 8 / 11 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_fraction_inequality_five_satisfies_inequality_largest_integer_is_five_l1313_131391


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_one_max_value_implies_a_l1313_131387

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

-- Part I
theorem solution_set_when_a_eq_one :
  {x : ℝ | f 1 x < 1} = {x : ℝ | x < (1/2 : ℝ)} := by sorry

-- Part II
theorem max_value_implies_a :
  (∃ (x : ℝ), f a x = 6) ∧ (∀ (x : ℝ), f a x ≤ 6) → a = 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_one_max_value_implies_a_l1313_131387


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l1313_131379

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * x - 7 * y = 35

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ := (0, -5)

/-- Theorem: The point (0, -5) is the intersection of the line 5x - 7y = 35 with the y-axis -/
theorem line_y_axis_intersection :
  line_equation intersection_point.1 intersection_point.2 ∧
  y_axis intersection_point.1 := by
  sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l1313_131379


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l1313_131306

theorem arithmetic_sequence_solution :
  let a₁ : ℚ := 2/3
  let a₂ := y - 2
  let a₃ := 4*y - 1
  (a₂ - a₁ = a₃ - a₂) → y = 11/6 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l1313_131306


namespace NUMINAMATH_CALUDE_max_value_of_f_l1313_131352

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 18 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1313_131352


namespace NUMINAMATH_CALUDE_jane_crayon_count_l1313_131383

/-- The number of crayons Jane ends up with after various events -/
def final_crayon_count (initial_count : ℕ) (eaten : ℕ) (packs_bought : ℕ) (crayons_per_pack : ℕ) (broken : ℕ) : ℕ :=
  initial_count - eaten + packs_bought * crayons_per_pack - broken

/-- Theorem stating that Jane ends up with 127 crayons given the conditions -/
theorem jane_crayon_count :
  final_crayon_count 87 7 5 10 3 = 127 := by
  sorry

end NUMINAMATH_CALUDE_jane_crayon_count_l1313_131383


namespace NUMINAMATH_CALUDE_divisor_quotient_remainder_equality_l1313_131353

theorem divisor_quotient_remainder_equality (n : ℕ) (h : n > 1) :
  let divisors := {d : ℕ | d ∣ (n + 1)}
  let quotients := {q : ℕ | ∃ d ∈ divisors, q = n / d}
  let remainders := {r : ℕ | ∃ d ∈ divisors, r = n % d}
  quotients = remainders :=
by sorry

end NUMINAMATH_CALUDE_divisor_quotient_remainder_equality_l1313_131353


namespace NUMINAMATH_CALUDE_games_played_l1313_131377

/-- Given that Andrew spent $9.00 for each game and $45 in total,
    prove that the number of games played is 5. -/
theorem games_played (cost_per_game : ℝ) (total_spent : ℝ) : 
  cost_per_game = 9 → total_spent = 45 → (total_spent / cost_per_game : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_games_played_l1313_131377


namespace NUMINAMATH_CALUDE_special_number_l1313_131334

def is_consecutive (a b c d e : ℕ) : Prop :=
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ (d + 1 = e)

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    is_consecutive a b c d e ∧
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (a * 10 + b) * c = d * 10 + e

theorem special_number :
  satisfies_condition 13452 :=
sorry

end NUMINAMATH_CALUDE_special_number_l1313_131334


namespace NUMINAMATH_CALUDE_paint_for_similar_statues_l1313_131314

/-- The amount of paint needed for similar statues -/
theorem paint_for_similar_statues
  (original_height : ℝ)
  (original_paint : ℝ)
  (new_height : ℝ)
  (num_statues : ℕ)
  (h1 : original_height = 8)
  (h2 : original_paint = 1)
  (h3 : new_height = 2)
  (h4 : num_statues = 320) :
  (num_statues : ℝ) * original_paint * (new_height / original_height) ^ 2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_paint_for_similar_statues_l1313_131314


namespace NUMINAMATH_CALUDE_basketball_court_width_l1313_131371

theorem basketball_court_width (perimeter : ℝ) (length_diff : ℝ) : perimeter = 96 ∧ length_diff = 14 → 
  ∃ width : ℝ, width = 17 ∧ 2 * (width + length_diff) + 2 * width = perimeter := by
  sorry

end NUMINAMATH_CALUDE_basketball_court_width_l1313_131371


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l1313_131300

theorem arctan_equation_solution (x : ℝ) :
  3 * Real.arctan (1/4) + Real.arctan (1/7) + Real.arctan (1/x) = π/4 →
  x = 34/13 := by
sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l1313_131300


namespace NUMINAMATH_CALUDE_janet_return_time_l1313_131354

/-- Represents the walking pattern of Janet in a grid system -/
structure WalkingPattern where
  north : ℕ
  west : ℕ
  south : ℕ
  east : ℕ

/-- Calculates the time taken to return home given a walking pattern and speed -/
def timeToReturnHome (pattern : WalkingPattern) (speed : ℕ) : ℕ :=
  let net_north := pattern.south - pattern.north
  let net_west := pattern.west - pattern.east
  let total_blocks := net_north + net_west
  total_blocks / speed

/-- Janet's specific walking pattern -/
def janetsPattern : WalkingPattern :=
  { north := 3
  , west := 7 * 3
  , south := 8
  , east := 2 * 8 }

/-- Janet's walking speed in blocks per minute -/
def janetsSpeed : ℕ := 2

/-- Theorem stating that it takes Janet 5 minutes to return home -/
theorem janet_return_time :
  timeToReturnHome janetsPattern janetsSpeed = 5 := by
  sorry

end NUMINAMATH_CALUDE_janet_return_time_l1313_131354


namespace NUMINAMATH_CALUDE_parabola_vertex_y_zero_l1313_131367

/-- The parabola y = x^2 - 10x + d has its vertex at y = 0 when d = 25 -/
theorem parabola_vertex_y_zero (x y d : ℝ) : 
  y = x^2 - 10*x + d → 
  (∃ x₀, ∀ x, x^2 - 10*x + d ≥ x₀^2 - 10*x₀ + d) → 
  d = 25 ↔ x^2 - 10*x + d ≥ 0 ∧ ∃ x₁, x₁^2 - 10*x₁ + d = 0 := by
sorry


end NUMINAMATH_CALUDE_parabola_vertex_y_zero_l1313_131367


namespace NUMINAMATH_CALUDE_triangle_circumcircle_point_length_l1313_131304

/-- Triangle PQR with sides PQ = 39, QR = 52, PR = 25 -/
structure Triangle :=
  (PQ QR PR : ℝ)
  (PQ_pos : PQ > 0)
  (QR_pos : QR > 0)
  (PR_pos : PR > 0)

/-- S is a point on the circumcircle of triangle PQR -/
structure CircumcirclePoint (t : Triangle) :=
  (S : ℝ × ℝ)

/-- S is on the perpendicular bisector of PR, not on the same side as Q -/
def onPerpendicularBisector (t : Triangle) (p : CircumcirclePoint t) : Prop := sorry

/-- The length of PS can be expressed as a√b where a and b are positive integers -/
def PSLength (t : Triangle) (p : CircumcirclePoint t) : ℕ × ℕ := sorry

/-- b is not divisible by the square of any prime -/
def notDivisibleBySquare : ℕ → Prop := sorry

theorem triangle_circumcircle_point_length 
  (t : Triangle) 
  (h1 : t.PQ = 39 ∧ t.QR = 52 ∧ t.PR = 25) 
  (p : CircumcirclePoint t) 
  (h2 : onPerpendicularBisector t p) 
  (h3 : let (a, b) := PSLength t p; notDivisibleBySquare b) : 
  let (a, b) := PSLength t p
  (a : ℕ) + Real.sqrt b = 54 := by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_point_length_l1313_131304


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1313_131373

-- Define the repeating decimals
def repeating_246 : ℚ := 246 / 999
def repeating_135 : ℚ := 135 / 999
def repeating_579 : ℚ := 579 / 999

-- State the theorem
theorem repeating_decimal_subtraction :
  repeating_246 - repeating_135 - repeating_579 = -156 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1313_131373


namespace NUMINAMATH_CALUDE_cos_theta_value_l1313_131338

theorem cos_theta_value (θ : Real) 
  (h1 : 0 ≤ θ ∧ θ ≤ π/2) 
  (h2 : Real.sin (θ - π/6) = 1/3) : 
  Real.cos θ = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_value_l1313_131338


namespace NUMINAMATH_CALUDE_quadratic_point_relationship_l1313_131358

/-- A quadratic function of the form y = -(x-1)² + c -/
def quadratic_function (x c : ℝ) : ℝ := -(x - 1)^2 + c

/-- Three points on the quadratic function -/
structure Points where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ

/-- The theorem stating the relationship between y₁, y₂, and y₃ -/
theorem quadratic_point_relationship (c : ℝ) (p : Points) :
  p.y₁ = quadratic_function (-3) c →
  p.y₂ = quadratic_function (-1) c →
  p.y₃ = quadratic_function 5 c →
  p.y₂ > p.y₁ ∧ p.y₁ = p.y₃ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_point_relationship_l1313_131358


namespace NUMINAMATH_CALUDE_simplified_expression_value_l1313_131325

theorem simplified_expression_value (a b : ℝ) (h : (b - 1)^2 + |a + 3| = 0) :
  -a^2*b + (3*a*b^2 - a^2*b) - 2*(2*a*b^2 - a^2*b) = 3 := by sorry

end NUMINAMATH_CALUDE_simplified_expression_value_l1313_131325


namespace NUMINAMATH_CALUDE_tangent_parallel_at_minus_one_l1313_131324

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_parallel_at_minus_one :
  ∃ (P : ℝ × ℝ), P.1 = -1 ∧ P.2 = f P.1 ∧ f' P.1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_at_minus_one_l1313_131324


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1313_131331

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (b / a = Real.sqrt 3) → 
  (∃ c : ℝ, c = 4 ∧ c^2 = a^2 + b^2) → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 / 12 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1313_131331


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1313_131310

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1313_131310


namespace NUMINAMATH_CALUDE_unique_perfect_square_l1313_131366

theorem unique_perfect_square (x : ℕ) (y : ℕ) : ∃! x, y = x^4 + 2*x^3 + 2*x^2 + 2*x + 1 ∧ ∃ z, y = z^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_l1313_131366


namespace NUMINAMATH_CALUDE_abs_x_plus_one_gt_three_l1313_131369

theorem abs_x_plus_one_gt_three (x : ℝ) :
  |x + 1| > 3 ↔ x < -4 ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_abs_x_plus_one_gt_three_l1313_131369


namespace NUMINAMATH_CALUDE_cube_path_exists_l1313_131347

/-- Represents a cell on the chessboard --/
structure Cell :=
  (x : Fin 8)
  (y : Fin 8)

/-- Represents a face of the cube --/
inductive Face
  | Top
  | Bottom
  | North
  | South
  | East
  | West

/-- Represents the state of the cube on the board --/
structure CubeState :=
  (position : Cell)
  (topFace : Face)

/-- Represents a move of the cube --/
inductive Move
  | North
  | South
  | East
  | West

/-- Function to apply a move to a cube state --/
def applyMove (state : CubeState) (move : Move) : CubeState :=
  sorry

/-- Predicate to check if a cell has been visited --/
def hasVisited (cell : Cell) (path : List CubeState) : Prop :=
  sorry

/-- Theorem: There exists a path for the cube that visits all cells while keeping one face never touching the board --/
theorem cube_path_exists : 
  ∃ (initialState : CubeState) (path : List Move),
    (∀ cell : Cell, hasVisited cell (initialState :: (List.scanl applyMove initialState path))) ∧
    (∃ face : Face, ∀ state ∈ (initialState :: (List.scanl applyMove initialState path)), state.topFace ≠ face) :=
  sorry

end NUMINAMATH_CALUDE_cube_path_exists_l1313_131347


namespace NUMINAMATH_CALUDE_coeff_x_cubed_eq_60_l1313_131396

/-- The coefficient of x^3 in the expansion of x(1+2x)^6 -/
def coeff_x_cubed : ℕ :=
  (Finset.range 7).sum (fun k => k.choose 6 * 2^k * if k = 2 then 1 else 0)

/-- Theorem stating that the coefficient of x^3 in x(1+2x)^6 is 60 -/
theorem coeff_x_cubed_eq_60 : coeff_x_cubed = 60 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_cubed_eq_60_l1313_131396


namespace NUMINAMATH_CALUDE_cubic_monotonically_increasing_iff_l1313_131318

/-- A cubic function f(x) = ax³ + bx² + cx + d -/
def cubic_function (a b c d : ℝ) : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d

/-- A function is monotonically increasing -/
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem: For a cubic function f(x) = ax³ + bx² + cx + d with a > 0,
    f(x) is monotonically increasing on ℝ if and only if b² - 3ac ≤ 0 -/
theorem cubic_monotonically_increasing_iff {a b c d : ℝ} (ha : a > 0) :
  monotonically_increasing (cubic_function a b c d) ↔ b^2 - 3*a*c ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_monotonically_increasing_iff_l1313_131318


namespace NUMINAMATH_CALUDE_max_k_value_l1313_131374

theorem max_k_value (x y k : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_k : k > 0)
  (h_eq : 4 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 17) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_k_value_l1313_131374


namespace NUMINAMATH_CALUDE_rancher_loss_calculation_l1313_131375

/-- Represents the rancher's cattle situation and calculates the loss --/
def rancher_loss (initial_cattle : ℕ) (initial_total_price : ℕ) (dead_cattle : ℕ) (price_reduction : ℕ) : ℕ :=
  let original_price_per_head := initial_total_price / initial_cattle
  let new_price_per_head := original_price_per_head - price_reduction
  let remaining_cattle := initial_cattle - dead_cattle
  let new_total_price := new_price_per_head * remaining_cattle
  initial_total_price - new_total_price

/-- Theorem stating the rancher's loss given the problem conditions --/
theorem rancher_loss_calculation :
  rancher_loss 340 204000 172 150 = 128400 := by
  sorry

end NUMINAMATH_CALUDE_rancher_loss_calculation_l1313_131375


namespace NUMINAMATH_CALUDE_room_width_calculation_l1313_131302

/-- Given a rectangular room with the following properties:
  * length: 5.5 meters
  * total paving cost: 16500 Rs
  * paving rate: 800 Rs per square meter
  This theorem proves that the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate : ℝ) :
  length = 5.5 →
  total_cost = 16500 →
  rate = 800 →
  (total_cost / rate) / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1313_131302


namespace NUMINAMATH_CALUDE_car_distance_calculation_l1313_131364

/-- Given a car's speed and how a speed increase affects travel time, calculate the distance traveled. -/
theorem car_distance_calculation (V : ℝ) (D : ℝ) (h1 : V = 40) 
  (h2 : D / V - D / (V + 20) = 0.5) : D = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_calculation_l1313_131364


namespace NUMINAMATH_CALUDE_arrangements_theorem_l1313_131339

/-- The number of arrangements of 5 people in a row with exactly 1 person between A and B -/
def arrangements_count : ℕ := 36

/-- The number of people in the arrangement -/
def total_people : ℕ := 5

/-- The number of people between A and B -/
def people_between : ℕ := 1

theorem arrangements_theorem :
  ∀ (n : ℕ) (k : ℕ),
  n = total_people →
  k = people_between →
  arrangements_count = 36 :=
sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l1313_131339


namespace NUMINAMATH_CALUDE_count_positive_3_to_1400_l1313_131335

/-- Represents the operation of flipping signs in three cells -/
def flip_operation (strip : List Bool) (i j k : Nat) : List Bool :=
  sorry

/-- Checks if a number N is positive according to the problem definition -/
def is_positive (N : Nat) : Bool :=
  sorry

/-- Counts the number of positive integers in the range [3, 1400] -/
def count_positive : Nat :=
  (List.range 1398).filter (fun n => is_positive (n + 3)) |>.length

/-- The main theorem stating the count of positive numbers -/
theorem count_positive_3_to_1400 : count_positive = 1396 :=
  sorry

end NUMINAMATH_CALUDE_count_positive_3_to_1400_l1313_131335


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1313_131399

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) ↔
  (x^2 / 25 + y^2 / 21 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1313_131399


namespace NUMINAMATH_CALUDE_f_geq_2_solution_set_f_minus_abs_geq_0_t_range_l1313_131359

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 2*|x + 3|

-- Theorem for the first part of the problem
theorem f_geq_2_solution_set :
  {x : ℝ | f x ≥ 2} = {x : ℝ | -4 ≤ x ∧ x ≤ -8/3} := by sorry

-- Theorem for the second part of the problem
theorem f_minus_abs_geq_0_t_range :
  {t : ℝ | ∃ x, f x - |3*t - 2| ≥ 0} = {t : ℝ | -1/3 ≤ t ∧ t ≤ 5/3} := by sorry

end NUMINAMATH_CALUDE_f_geq_2_solution_set_f_minus_abs_geq_0_t_range_l1313_131359


namespace NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l1313_131345

theorem m_fourth_plus_n_fourth (m n : ℝ) 
  (h1 : m - n = -5)
  (h2 : m^2 + n^2 = 13) :
  m^4 + n^4 = 97 := by
sorry

end NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l1313_131345


namespace NUMINAMATH_CALUDE_min_diameter_bounds_l1313_131368

/-- The minimum diameter of n points on a plane where the distance between any two points is at least 1 -/
def min_diameter (n : ℕ) : ℝ :=
  sorry

/-- The distance between any two points is at least 1 -/
axiom min_distance (n : ℕ) (i j : Fin n) (points : Fin n → ℝ × ℝ) :
  i ≠ j → dist (points i) (points j) ≥ 1

theorem min_diameter_bounds :
  (∀ n : ℕ, n = 2 ∨ n = 3 → min_diameter n ≥ 1) ∧
  (min_diameter 4 ≥ Real.sqrt 2) ∧
  (min_diameter 5 ≥ (1 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_min_diameter_bounds_l1313_131368


namespace NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_9_and_tens_greater_l1313_131380

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that returns the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ :=
  n / 10

/-- A function that returns the units digit of a two-digit number -/
def units_digit (n : ℕ) : ℕ :=
  n % 10

/-- The main theorem -/
theorem no_two_digit_primes_with_digit_sum_9_and_tens_greater : 
  ∀ n : ℕ, 10 ≤ n → n < 100 → 
    (tens_digit n + units_digit n = 9 ∧ tens_digit n > units_digit n) → 
    ¬(is_prime n) :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_9_and_tens_greater_l1313_131380


namespace NUMINAMATH_CALUDE_simplify_expression_l1313_131361

theorem simplify_expression :
  (8 * 10^12) / (4 * 10^4) + 2 * 10^3 = 200002000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1313_131361


namespace NUMINAMATH_CALUDE_parabola_intersects_line_segment_range_l1313_131342

/-- Parabola equation -/
def parabola (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - 1

/-- Line segment AB -/
def line_segment_AB (x : ℝ) : ℝ := -x + 3

/-- Point A -/
def point_A : ℝ × ℝ := (3, 0)

/-- Point B -/
def point_B : ℝ × ℝ := (0, 3)

/-- Theorem stating the range of m for which the parabola intersects line segment AB at two distinct points -/
theorem parabola_intersects_line_segment_range :
  ∃ (m_min m_max : ℝ), m_min = 3 ∧ m_max = 10/3 ∧
  ∀ (m : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
              0 ≤ x₁ ∧ x₁ ≤ 3 ∧ 0 ≤ x₂ ∧ x₂ ≤ 3 ∧
              parabola m x₁ = line_segment_AB x₁ ∧
              parabola m x₂ = line_segment_AB x₂) ↔
             (m_min ≤ m ∧ m ≤ m_max) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersects_line_segment_range_l1313_131342


namespace NUMINAMATH_CALUDE_same_color_isosceles_count_independent_l1313_131365

/-- Represents a coloring of vertices in a regular polygon -/
structure Coloring (n : ℕ) where
  red : Finset (Fin (6*n+1))
  blue : Finset (Fin (6*n+1))
  partition : red ∪ blue = Finset.univ
  disjoint : red ∩ blue = ∅

/-- Counts the number of isosceles triangles with vertices of the same color -/
def count_same_color_isosceles_triangles (n : ℕ) (c : Coloring n) : ℕ := sorry

/-- Theorem stating that the count of same-color isosceles triangles is independent of coloring -/
theorem same_color_isosceles_count_independent (n : ℕ) :
  ∀ c₁ c₂ : Coloring n, count_same_color_isosceles_triangles n c₁ = count_same_color_isosceles_triangles n c₂ :=
sorry

end NUMINAMATH_CALUDE_same_color_isosceles_count_independent_l1313_131365


namespace NUMINAMATH_CALUDE_equation_roots_relation_l1313_131351

theorem equation_roots_relation (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 3 * x₁ - 4 = a ∧ (x₂ + a) / 3 = 1 ∧ x₁ = 2 * x₂) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_relation_l1313_131351


namespace NUMINAMATH_CALUDE_cube_weight_doubling_l1313_131372

/-- Given a cube of metal weighing 7 pounds, prove that another cube of the same metal
    with sides twice as long will weigh 56 pounds. -/
theorem cube_weight_doubling (ρ : ℝ) (s : ℝ) (h1 : s > 0) (h2 : ρ * s^3 = 7) :
  ρ * (2*s)^3 = 56 := by
sorry

end NUMINAMATH_CALUDE_cube_weight_doubling_l1313_131372


namespace NUMINAMATH_CALUDE_consecutive_numbers_with_special_properties_l1313_131389

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem consecutive_numbers_with_special_properties :
  ∃ (n : ℕ), sum_of_digits n = 8 ∧ (n + 1) % 8 = 0 :=
by
  use 71
  sorry

#eval sum_of_digits 71  -- Should output 8
#eval (72 % 8)  -- Should output 0

end NUMINAMATH_CALUDE_consecutive_numbers_with_special_properties_l1313_131389


namespace NUMINAMATH_CALUDE_unclaimed_candy_l1313_131344

/-- Represents the order of arrival of the winners -/
inductive Winner : Type
  | Al | Bert | Carl | Dana

/-- The ratio of candy each winner should receive -/
def candy_ratio (w : Winner) : ℚ :=
  match w with
  | Winner.Al => 4 / 10
  | Winner.Bert => 3 / 10
  | Winner.Carl => 2 / 10
  | Winner.Dana => 1 / 10

/-- The amount of candy each winner actually takes -/
def candy_taken (w : Winner) : ℚ :=
  match w with
  | Winner.Al => 4 / 10
  | Winner.Bert => 9 / 50
  | Winner.Carl => 21 / 250
  | Winner.Dana => 19 / 250

theorem unclaimed_candy :
  1 - (candy_taken Winner.Al + candy_taken Winner.Bert + candy_taken Winner.Carl + candy_taken Winner.Dana) = 46 / 125 := by
  sorry

end NUMINAMATH_CALUDE_unclaimed_candy_l1313_131344


namespace NUMINAMATH_CALUDE_solve_movie_problem_l1313_131378

def movie_problem (regular_price child_discount adults_count money_given change : ℕ) : Prop :=
  let child_price := regular_price - child_discount
  let total_spent := money_given - change
  let adults_cost := adults_count * regular_price
  let children_cost := total_spent - adults_cost
  ∃ (children_count : ℕ), children_count * child_price = children_cost

theorem solve_movie_problem :
  movie_problem 9 2 2 40 1 = ∃ (children_count : ℕ), children_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_movie_problem_l1313_131378


namespace NUMINAMATH_CALUDE_tj_race_second_half_time_l1313_131313

/-- Represents a race with given parameters -/
structure Race where
  totalDistance : ℝ
  firstHalfTime : ℝ
  averagePace : ℝ

/-- Calculates the time for the second half of the race -/
def secondHalfTime (race : Race) : ℝ :=
  race.averagePace * race.totalDistance - race.firstHalfTime

/-- Theorem stating that for a 10K race with given conditions, 
    the second half time is 30 minutes -/
theorem tj_race_second_half_time :
  let race : Race := {
    totalDistance := 10,
    firstHalfTime := 20,
    averagePace := 5
  }
  secondHalfTime race = 30 := by
  sorry


end NUMINAMATH_CALUDE_tj_race_second_half_time_l1313_131313


namespace NUMINAMATH_CALUDE_square_perimeter_l1313_131384

/-- Given a square cut into four equal rectangles, where each rectangle's length is four times
    its width, and these rectangles are arranged to form a shape with perimeter 56,
    prove that the perimeter of the original square is 32. -/
theorem square_perimeter (x : ℝ) : 
  x > 0 →  -- width of each rectangle is positive
  (4 * x) > 0 →  -- length of each rectangle is positive
  (28 * x) = 56 →  -- perimeter of the P shape
  (4 * (4 * x)) = 32  -- perimeter of the original square
  := by sorry

end NUMINAMATH_CALUDE_square_perimeter_l1313_131384


namespace NUMINAMATH_CALUDE_apples_in_box_l1313_131305

/-- The number of boxes containing apples -/
def num_boxes : ℕ := 5

/-- The number of apples removed from each box -/
def apples_removed : ℕ := 60

/-- The number of apples initially in each box -/
def apples_per_box : ℕ := 100

theorem apples_in_box : 
  (num_boxes * apples_per_box) - (num_boxes * apples_removed) = 2 * apples_per_box := by
  sorry

end NUMINAMATH_CALUDE_apples_in_box_l1313_131305


namespace NUMINAMATH_CALUDE_similar_triangles_height_l1313_131328

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 5 →
  area_ratio = 25 / 9 →
  ∃ h_large : ℝ, h_large = h_small * (area_ratio.sqrt) ∧ h_large = 25 / 3 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l1313_131328


namespace NUMINAMATH_CALUDE_tank_capacity_is_900_l1313_131376

/-- Represents the capacity of a tank and its filling/draining rates. -/
structure TankSystem where
  capacity : ℕ
  fill_rate_A : ℕ
  fill_rate_B : ℕ
  drain_rate_C : ℕ

/-- Calculates the net amount of water added to the tank in one cycle. -/
def net_fill_per_cycle (t : TankSystem) : ℕ :=
  t.fill_rate_A + t.fill_rate_B - t.drain_rate_C

/-- Theorem stating that under given conditions, the tank capacity is 900 liters. -/
theorem tank_capacity_is_900 (t : TankSystem) 
  (h1 : t.fill_rate_A = 40)
  (h2 : t.fill_rate_B = 30)
  (h3 : t.drain_rate_C = 20)
  (h4 : (54 : ℕ) * (net_fill_per_cycle t) / 3 = t.capacity) :
  t.capacity = 900 := by
  sorry

#check tank_capacity_is_900

end NUMINAMATH_CALUDE_tank_capacity_is_900_l1313_131376


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l1313_131388

theorem sum_of_cubes_divisibility (k n : ℤ) : 
  (∃ m : ℤ, k + n = 3 * m) → (∃ l : ℤ, k^3 + n^3 = 9 * l) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l1313_131388


namespace NUMINAMATH_CALUDE_problem_two_l1313_131348

theorem problem_two : -2.5 / (5/16) * (-1/8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_two_l1313_131348


namespace NUMINAMATH_CALUDE_inequalities_proof_l1313_131395

theorem inequalities_proof :
  (((12 : ℝ) / 11) ^ 11 > ((11 : ℝ) / 10) ^ 10) ∧
  (((12 : ℝ) / 11) ^ 12 < ((11 : ℝ) / 10) ^ 11) ∧
  (((12 : ℝ) / 11) ^ 10 > ((11 : ℝ) / 10) ^ 9) ∧
  (((11 : ℝ) / 10) ^ 12 > ((12 : ℝ) / 11) ^ 13) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1313_131395


namespace NUMINAMATH_CALUDE_factorization_equality_l1313_131326

theorem factorization_equality (x : ℝ) : 3 * x^2 - 9 * x = 3 * x * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1313_131326


namespace NUMINAMATH_CALUDE_heart_diamond_club_probability_l1313_131340

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of cards of each suit
def cards_per_suit : ℕ := 13

-- Define the probability of drawing a specific sequence of cards
def draw_probability (deck_size : ℕ) (hearts diamonds clubs : ℕ) : ℚ :=
  (hearts : ℚ) / deck_size *
  (diamonds : ℚ) / (deck_size - 1) *
  (clubs : ℚ) / (deck_size - 2)

-- Theorem statement
theorem heart_diamond_club_probability :
  draw_probability standard_deck cards_per_suit cards_per_suit cards_per_suit = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_heart_diamond_club_probability_l1313_131340


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l1313_131312

/-- A line that is a perpendicular bisector of a line segment passes through its midpoint -/
axiom perpendicular_bisector_passes_through_midpoint {x₁ y₁ x₂ y₂ : ℝ} (b : ℝ) :
  (∀ x y, x + y = b → (x - (x₁ + x₂) / 2)^2 + (y - (y₁ + y₂) / 2)^2 = ((x₂ - x₁)^2 + (y₂ - y₁)^2) / 4) →
  b = (x₁ + x₂) / 2 + (y₁ + y₂) / 2

/-- The value of b for the perpendicular bisector of the line segment from (2,1) to (8,7) -/
theorem perpendicular_bisector_value : 
  (∀ x y, x + y = b → (x - 5)^2 + (y - 4)^2 = 25) → b = 9 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_value_l1313_131312


namespace NUMINAMATH_CALUDE_trig_identity_l1313_131303

theorem trig_identity : 4 * Real.cos (15 * π / 180) * Real.cos (75 * π / 180) - Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1313_131303


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l1313_131363

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {1, 2, 3, 4}

theorem complement_intersection_equals_set : 
  (U \ (A ∩ B)) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l1313_131363


namespace NUMINAMATH_CALUDE_complex_product_real_implies_a_equals_negative_one_l1313_131330

theorem complex_product_real_implies_a_equals_negative_one (a : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑(1 + a * Complex.I) * ↑(1 + Complex.I) : ℂ).im = 0 →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_implies_a_equals_negative_one_l1313_131330


namespace NUMINAMATH_CALUDE_bags_sold_is_30_l1313_131321

-- Define the variables
def cost_price : ℕ := 4
def selling_price : ℕ := 8
def total_profit : ℕ := 120

-- Define the profit per bag
def profit_per_bag : ℕ := selling_price - cost_price

-- Theorem to prove
theorem bags_sold_is_30 : total_profit / profit_per_bag = 30 := by
  sorry

end NUMINAMATH_CALUDE_bags_sold_is_30_l1313_131321


namespace NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l1313_131320

theorem quadratic_polynomial_negative_root
  (f : ℝ → ℝ)
  (h1 : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0)
  (h2 : ∀ (a b : ℝ), f (a^2 + b^2) ≥ f (2*a*b)) :
  ∃ (x : ℝ), x < 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l1313_131320


namespace NUMINAMATH_CALUDE_opposite_solutions_k_value_l1313_131337

theorem opposite_solutions_k_value (x y k : ℝ) : 
  (2 * x + 5 * y = k) → 
  (x - 4 * y = 15) → 
  (x + y = 0) → 
  k = -9 := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_k_value_l1313_131337


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1313_131308

theorem sum_of_fractions_equals_one (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1313_131308


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1313_131346

theorem complex_fraction_equality : (1 - I) / (2 - I) = 3/5 - (1/5) * I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1313_131346


namespace NUMINAMATH_CALUDE_train_stop_time_l1313_131394

/-- Proves that a train with given speeds stops for 18 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 30)
  (h2 : speed_with_stops = 21) : ℝ :=
by
  -- Define the stop time in minutes
  let stop_time : ℝ := 18
  
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_train_stop_time_l1313_131394


namespace NUMINAMATH_CALUDE_roots_cubed_equation_reciprocal_squares_equation_sum_and_reciprocal_equation_quotient_roots_equation_l1313_131322

-- Define the quadratic equation and its roots
variable (p q : ℝ)
variable (x₁ x₂ : ℝ)

-- Define the original equation
def original_eq (x : ℝ) : Prop := x^2 + p*x + q = 0

-- Define that x₁ and x₂ are roots of the original equation
axiom root_x₁ : original_eq p q x₁
axiom root_x₂ : original_eq p q x₂

-- Part a
theorem roots_cubed_equation :
  ∀ y, y^2 + (p^3 - 3*p*q)*y + q^3 = 0 ↔ (y = x₁^3 ∨ y = x₂^3) :=
sorry

-- Part b
theorem reciprocal_squares_equation :
  ∀ y, q^2*y^2 + (2*q - p^2)*y + 1 = 0 ↔ (y = 1/x₁^2 ∨ y = 1/x₂^2) :=
sorry

-- Part c
theorem sum_and_reciprocal_equation :
  ∀ y, q*y^2 + p*(q + 1)*y + (q + 1)^2 = 0 ↔ (y = x₁ + 1/x₂ ∨ y = x₂ + 1/x₁) :=
sorry

-- Part d
theorem quotient_roots_equation :
  ∀ y, q*y^2 + (2*q - p^2)*y + q = 0 ↔ (y = x₂/x₁ ∨ y = x₁/x₂) :=
sorry

end NUMINAMATH_CALUDE_roots_cubed_equation_reciprocal_squares_equation_sum_and_reciprocal_equation_quotient_roots_equation_l1313_131322


namespace NUMINAMATH_CALUDE_crayons_per_unit_is_six_l1313_131319

/-- Given the total number of units, cost per crayon, and total cost,
    calculate the number of crayons in each unit. -/
def crayons_per_unit (total_units : ℕ) (cost_per_crayon : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost / cost_per_crayon) / total_units

/-- Theorem stating that under the given conditions, there are 6 crayons in each unit. -/
theorem crayons_per_unit_is_six :
  crayons_per_unit 4 2 48 = 6 := by
  sorry

#eval crayons_per_unit 4 2 48

end NUMINAMATH_CALUDE_crayons_per_unit_is_six_l1313_131319


namespace NUMINAMATH_CALUDE_circle_area_diameter_increase_l1313_131355

theorem circle_area_diameter_increase : 
  ∀ (A D A' D' : ℝ), 
  A > 0 → D > 0 →
  A = π * (D / 2)^2 →
  A' = 4 * A →
  A' = π * (D' / 2)^2 →
  D' / D - 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_area_diameter_increase_l1313_131355


namespace NUMINAMATH_CALUDE_union_of_sets_l1313_131362

theorem union_of_sets : 
  let A : Set ℤ := {0, 1, 2}
  let B : Set ℤ := {-1, 0}
  A ∪ B = {-1, 0, 1, 2} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1313_131362


namespace NUMINAMATH_CALUDE_eugene_model_house_l1313_131397

theorem eugene_model_house (cards_per_deck : ℕ) (unused_cards : ℕ) (toothpicks_per_card : ℕ) (toothpicks_per_box : ℕ) :
  cards_per_deck = 52 →
  unused_cards = 16 →
  toothpicks_per_card = 75 →
  toothpicks_per_box = 450 →
  toothpicks_per_box = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_eugene_model_house_l1313_131397


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l1313_131360

theorem divisibility_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l1313_131360


namespace NUMINAMATH_CALUDE_curve_C_extrema_l1313_131316

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 6 = 0

-- Define the function we want to maximize/minimize
def f (x y : ℝ) : ℝ := x + 2*y

-- State the theorem
theorem curve_C_extrema :
  (∀ x y : ℝ, C x y → 10 - Real.sqrt 6 ≤ f x y) ∧
  (∀ x y : ℝ, C x y → f x y ≤ 10 + Real.sqrt 6) ∧
  (∃ x₁ y₁ : ℝ, C x₁ y₁ ∧ f x₁ y₁ = 10 - Real.sqrt 6) ∧
  (∃ x₂ y₂ : ℝ, C x₂ y₂ ∧ f x₂ y₂ = 10 + Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_extrema_l1313_131316


namespace NUMINAMATH_CALUDE_sperners_lemma_l1313_131333

theorem sperners_lemma (n : ℕ) (A : Finset (Finset ℕ)) :
  (∀ (i j : Finset ℕ), i ∈ A → j ∈ A → i ≠ j → (¬ i ⊆ j ∧ ¬ j ⊆ i)) →
  (∀ i ∈ A, i ⊆ Finset.range n) →
  A.card ≤ Nat.choose n (n / 2) := by
  sorry

end NUMINAMATH_CALUDE_sperners_lemma_l1313_131333


namespace NUMINAMATH_CALUDE_three_digit_number_eleven_times_sum_of_digits_l1313_131341

theorem three_digit_number_eleven_times_sum_of_digits :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = 11 * (n / 100 + (n / 10) % 10 + n % 10) :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_eleven_times_sum_of_digits_l1313_131341


namespace NUMINAMATH_CALUDE_infinite_triangular_pairs_l1313_131323

theorem infinite_triangular_pairs :
  ∃ (S : Set Nat), Set.Infinite S ∧ 
    (∀ p ∈ S, Prime p ∧ Odd p ∧
      (∀ t : Nat, t > 0 →
        (∃ n : Nat, t = n * (n + 1) / 2) ↔
        (∃ m : Nat, p^2 * t + (p^2 - 1) / 8 = m * (m + 1) / 2))) := by
  sorry

end NUMINAMATH_CALUDE_infinite_triangular_pairs_l1313_131323


namespace NUMINAMATH_CALUDE_largest_trifecta_sum_l1313_131392

/-- A trifecta is an ordered triple of positive integers (a, b, c) with a < b < c
    such that a divides b, b divides c, and c divides ab. --/
def is_trifecta (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ b % a = 0 ∧ c % b = 0 ∧ (a * b) % c = 0

/-- The sum of a trifecta (a, b, c) --/
def trifecta_sum (a b c : ℕ) : ℕ := a + b + c

/-- The largest possible sum of a trifecta of three-digit integers is 700 --/
theorem largest_trifecta_sum :
  (∃ a b c : ℕ, 100 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 999 ∧ is_trifecta a b c ∧
    trifecta_sum a b c = 700) ∧
  (∀ a b c : ℕ, 100 ≤ a → a < b → b < c → c ≤ 999 → is_trifecta a b c →
    trifecta_sum a b c ≤ 700) :=
by sorry

end NUMINAMATH_CALUDE_largest_trifecta_sum_l1313_131392


namespace NUMINAMATH_CALUDE_function_properties_l1313_131329

-- Define the function f
def f (m x : ℝ) : ℝ := (m^2 - 1) * x + m^2 - 3*m + 2

-- State the theorem
theorem function_properties :
  ∀ m : ℝ,
  (∀ x y : ℝ, x < y → f m x > f m y) →  -- f is decreasing
  f m 1 = 0 →                          -- f(1) = 0
  (m = 1/2 ∧                           -- m = 1/2
   ∀ x : ℝ, f m (x+1) ≥ x^2 ↔ -3/4 ≤ x ∧ x ≤ 0)  -- range of x
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l1313_131329


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l1313_131327

def purchase_price : ℚ := 14000
def transportation_charges : ℚ := 1000
def selling_price : ℚ := 30000
def profit_percentage : ℚ := 50

theorem repair_cost_calculation (repair_cost : ℚ) :
  (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage / 100) = selling_price →
  repair_cost = 5000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l1313_131327


namespace NUMINAMATH_CALUDE_share_face_value_l1313_131332

/-- Given a share with a 9% dividend rate and a market value of Rs. 42,
    prove that the face value is Rs. 56 if an investor wants a 12% return. -/
theorem share_face_value (dividend_rate : ℝ) (market_value : ℝ) (desired_return : ℝ) :
  dividend_rate = 0.09 →
  market_value = 42 →
  desired_return = 0.12 →
  ∃ (face_value : ℝ), face_value = 56 ∧ dividend_rate * face_value = desired_return * market_value :=
by
  sorry

#check share_face_value

end NUMINAMATH_CALUDE_share_face_value_l1313_131332


namespace NUMINAMATH_CALUDE_red_bead_cost_l1313_131385

/-- The cost of a box of red beads -/
def red_cost : ℝ := 2.30

/-- The cost of a box of yellow beads -/
def yellow_cost : ℝ := 2.00

/-- The number of boxes of each color used -/
def boxes_per_color : ℕ := 4

/-- The total number of mixed boxes -/
def total_boxes : ℕ := 10

/-- The cost per box of mixed beads -/
def mixed_cost : ℝ := 1.72

theorem red_bead_cost :
  red_cost * boxes_per_color + yellow_cost * boxes_per_color = mixed_cost * total_boxes := by
  sorry

#check red_bead_cost

end NUMINAMATH_CALUDE_red_bead_cost_l1313_131385


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l1313_131398

theorem pure_imaginary_product (x : ℝ) : 
  (∃ k : ℝ, (x + 2*Complex.I) * ((x + 2) + 2*Complex.I) * ((x + 4) + 2*Complex.I) = k * Complex.I) ↔ 
  (x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l1313_131398


namespace NUMINAMATH_CALUDE_ratio_equality_l1313_131350

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 49)
  (sum_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1313_131350


namespace NUMINAMATH_CALUDE_max_product_of_three_l1313_131390

def S : Finset Int := {-9, -5, -3, 1, 4, 6, 8}

theorem max_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a * b * c ≤ 360 ∧ ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 360 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_three_l1313_131390


namespace NUMINAMATH_CALUDE_cookies_with_new_ingredients_l1313_131343

/-- Represents the number of cookies that can be made with given amounts of flour and sugar. -/
def cookies_made (flour : ℚ) (sugar : ℚ) : ℚ :=
  18 * (flour / 2) -- or equivalently, 18 * (sugar / 1)

/-- Theorem stating that 27 cookies can be made with 3 cups of flour and 1.5 cups of sugar,
    given the initial ratio of ingredients to cookies. -/
theorem cookies_with_new_ingredients :
  cookies_made 3 1.5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cookies_with_new_ingredients_l1313_131343


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1313_131356

/-- An arithmetic sequence starting at -58 with common difference 7 -/
def arithmeticSequence (n : ℕ) : ℤ := -58 + (n - 1) * 7

/-- The property that the sequence ends at or before 44 -/
def sequenceEndsBeforeOrAt44 (n : ℕ) : Prop := arithmeticSequence n ≤ 44

theorem arithmetic_sequence_length :
  ∃ (n : ℕ), n = 15 ∧ sequenceEndsBeforeOrAt44 n ∧ ¬sequenceEndsBeforeOrAt44 (n + 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1313_131356


namespace NUMINAMATH_CALUDE_prob_exactly_two_prob_at_least_one_l1313_131301

/-- The probability of exactly two out of three independent events occurring, 
    given their individual probabilities -/
theorem prob_exactly_two (p1 p2 p3 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3 = 
    0.398 ↔ p1 = 0.8 ∧ p2 = 0.7 ∧ p3 = 0.9 :=
sorry

/-- The probability of at least one out of three independent events occurring, 
    given their individual probabilities -/
theorem prob_at_least_one (p1 p2 p3 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 
    0.994 ↔ p1 = 0.8 ∧ p2 = 0.7 ∧ p3 = 0.9 :=
sorry

end NUMINAMATH_CALUDE_prob_exactly_two_prob_at_least_one_l1313_131301


namespace NUMINAMATH_CALUDE_inequality_transformation_l1313_131307

theorem inequality_transformation (a b : ℝ) (h : a < b) : -a/3 > -b/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l1313_131307


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1313_131381

/-- The distance from a point on the parabola y^2 = 4x to its focus -/
def distance_to_focus (x : ℝ) : ℝ :=
  x + 1

theorem parabola_focus_distance :
  let x : ℝ := 2
  let y : ℝ := 2 * Real.sqrt 2
  y^2 = 4*x → distance_to_focus x = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1313_131381


namespace NUMINAMATH_CALUDE_perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l1313_131317

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- Define perpendicularity condition for two lines
def perpendicular (m : ℝ) : Prop := 1 * (m - 2) + m * 3 = 0

-- Define parallelism condition for two lines
def parallel (m : ℝ) : Prop := 1 / (m - 2) = m / 3 ∧ m ≠ 3

-- Theorem 1: If l₁ is perpendicular to l₂, then m = 1/2
theorem perpendicular_implies_m_eq_half :
  ∀ m : ℝ, perpendicular m → m = 1/2 :=
by sorry

-- Theorem 2: If l₁ is parallel to l₂, then m = -1
theorem parallel_implies_m_eq_neg_one :
  ∀ m : ℝ, parallel m → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l1313_131317


namespace NUMINAMATH_CALUDE_tan_seven_pi_sixths_l1313_131336

theorem tan_seven_pi_sixths : Real.tan (7 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_sixths_l1313_131336


namespace NUMINAMATH_CALUDE_square_of_one_plus_sqrt_two_l1313_131311

theorem square_of_one_plus_sqrt_two : (1 + Real.sqrt 2) ^ 2 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_one_plus_sqrt_two_l1313_131311
