import Mathlib

namespace NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l3765_376588

theorem at_least_one_less_than_or_equal_to_one
  (x y z : ℝ)
  (pos_x : 0 < x)
  (pos_y : 0 < y)
  (pos_z : 0 < z)
  (sum_eq_three : x + y + z = 3) :
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l3765_376588


namespace NUMINAMATH_CALUDE_initial_concentration_proof_l3765_376533

theorem initial_concentration_proof (volume_replaced : ℝ) 
  (replacement_concentration : ℝ) (final_concentration : ℝ) :
  volume_replaced = 0.7142857142857143 →
  replacement_concentration = 0.25 →
  final_concentration = 0.35 →
  ∃ initial_concentration : ℝ,
    initial_concentration = 0.6 ∧
    (1 - volume_replaced) * initial_concentration + 
      volume_replaced * replacement_concentration = final_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_initial_concentration_proof_l3765_376533


namespace NUMINAMATH_CALUDE_prob_comparison_l3765_376531

/-- The probability of drawing two balls of the same color from two bags -/
def prob_same_color (m n : ℕ) : ℚ :=
  2 * m * n / ((m + n) * (m + n))

/-- The probability of drawing two balls of different colors from two bags -/
def prob_diff_color (m n : ℕ) : ℚ :=
  (m * m + n * n) / ((m + n) * (m + n))

theorem prob_comparison (m n : ℕ) :
  prob_same_color m n ≤ prob_diff_color m n ∧
  (prob_same_color m n = prob_diff_color m n ↔ m = n) :=
sorry

end NUMINAMATH_CALUDE_prob_comparison_l3765_376531


namespace NUMINAMATH_CALUDE_min_dimension_sum_for_2310_volume_l3765_376522

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- The volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- The sum of the dimensions of a box -/
def dimensionSum (d : BoxDimensions) : ℕ := d.length + d.width + d.height

/-- Theorem stating that the minimum sum of dimensions for a box with volume 2310 is 42 -/
theorem min_dimension_sum_for_2310_volume :
  (∃ (d : BoxDimensions), boxVolume d = 2310) →
  (∀ (d : BoxDimensions), boxVolume d = 2310 → dimensionSum d ≥ 42) ∧
  (∃ (d : BoxDimensions), boxVolume d = 2310 ∧ dimensionSum d = 42) :=
by sorry

end NUMINAMATH_CALUDE_min_dimension_sum_for_2310_volume_l3765_376522


namespace NUMINAMATH_CALUDE_dark_tile_fraction_is_one_fourth_l3765_376517

/-- Represents a 4x4 tile pattern -/
structure TilePattern :=
  (darkTilesInRow : Fin 4 → Nat)
  (h_valid : ∀ i, darkTilesInRow i ≤ 4)

/-- The specific tile pattern described in the problem -/
def problemPattern : TilePattern :=
  { darkTilesInRow := λ i => if i.val < 2 then 2 else 0,
    h_valid := by sorry }

/-- The fraction of dark tiles in a given pattern -/
def darkTileFraction (pattern : TilePattern) : Rat :=
  (pattern.darkTilesInRow 0 + pattern.darkTilesInRow 1 + 
   pattern.darkTilesInRow 2 + pattern.darkTilesInRow 3) / 16

theorem dark_tile_fraction_is_one_fourth :
  darkTileFraction problemPattern = 1/4 := by sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_is_one_fourth_l3765_376517


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_min_value_points_l3765_376501

theorem min_value_quadratic (x y : ℝ) : 2*x^2 + 2*y^2 - 8*x + 6*y + 28 ≥ 10.5 :=
by sorry

theorem min_value_achieved : ∃ (x y : ℝ), 2*x^2 + 2*y^2 - 8*x + 6*y + 28 = 10.5 :=
by sorry

theorem min_value_points : 2*2^2 + 2*(-3/2)^2 - 8*2 + 6*(-3/2) + 28 = 10.5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_min_value_points_l3765_376501


namespace NUMINAMATH_CALUDE_students_in_one_language_class_l3765_376572

theorem students_in_one_language_class 
  (french_class : ℕ) 
  (spanish_class : ℕ) 
  (both_classes : ℕ) 
  (h1 : french_class = 21) 
  (h2 : spanish_class = 21) 
  (h3 : both_classes = 6) :
  french_class + spanish_class - 2 * both_classes = 36 := by
  sorry

end NUMINAMATH_CALUDE_students_in_one_language_class_l3765_376572


namespace NUMINAMATH_CALUDE_vector_collinearity_implies_x_value_l3765_376515

theorem vector_collinearity_implies_x_value :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), a = k • (a - b)) →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_implies_x_value_l3765_376515


namespace NUMINAMATH_CALUDE_sum_odd_integers_11_to_39_l3765_376577

/-- The sum of odd integers from 11 to 39 (inclusive) is 375 -/
theorem sum_odd_integers_11_to_39 : 
  (Finset.range 15).sum (fun i => 2 * i + 11) = 375 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_11_to_39_l3765_376577


namespace NUMINAMATH_CALUDE_son_age_l3765_376539

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_son_age_l3765_376539


namespace NUMINAMATH_CALUDE_a_range_l3765_376576

theorem a_range (a : ℝ) : 
  (a + 1)^(-1/4 : ℝ) < (3 - 2*a)^(-1/4 : ℝ) → 2/3 < a ∧ a < 3/2 := by
sorry

end NUMINAMATH_CALUDE_a_range_l3765_376576


namespace NUMINAMATH_CALUDE_train_speed_l3765_376574

/-- The speed of a train traveling between two points, given the conditions of the problem -/
theorem train_speed (distance : ℝ) (return_speed : ℝ) (time_difference : ℝ) :
  distance = 480 ∧ 
  return_speed = 120 ∧ 
  time_difference = 1 →
  ∃ speed : ℝ, 
    speed = 160 ∧ 
    distance / speed + time_difference = distance / return_speed :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3765_376574


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3765_376512

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 30*x^2 + 105*x - 114 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 30*s^2 + 105*s - 114) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1/A + 1/B + 1/C = 300 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3765_376512


namespace NUMINAMATH_CALUDE_checkerboard_squares_l3765_376583

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  row : Nat
  col : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 10

/-- Checks if a square is valid on the board -/
def isValidSquare (s : Square) : Bool :=
  s.size > 0 && s.size <= boardSize && s.row + s.size <= boardSize && s.col + s.size <= boardSize

/-- Counts the number of black squares in a given square -/
def countBlackSquares (s : Square) : Nat :=
  sorry

/-- Counts the number of valid squares with at least 6 black squares -/
def countValidSquares : Nat :=
  sorry

theorem checkerboard_squares : countValidSquares = 155 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_squares_l3765_376583


namespace NUMINAMATH_CALUDE_problem_solution_l3765_376596

noncomputable def f (x : ℝ) := x + 1 + abs (3 - x)

theorem problem_solution :
  (∀ x ≥ -1, f x ≤ 6 ↔ -1 ≤ x ∧ x ≤ 4) ∧
  (∀ x ≥ -1, f x ≥ 4) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 8 * a * b = a + 2 * b → 2 * a + b ≥ 9/8) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3765_376596


namespace NUMINAMATH_CALUDE_julie_initial_savings_l3765_376529

/-- The amount of money Julie saved initially before doing jobs to buy a mountain bike. -/
def initial_savings : ℕ := sorry

/-- The cost of the mountain bike Julie wants to buy. -/
def bike_cost : ℕ := 2345

/-- The number of lawns Julie plans to mow. -/
def lawns_to_mow : ℕ := 20

/-- The payment Julie receives for mowing each lawn. -/
def payment_per_lawn : ℕ := 20

/-- The number of newspapers Julie plans to deliver. -/
def newspapers_to_deliver : ℕ := 600

/-- The payment Julie receives for delivering each newspaper (in cents). -/
def payment_per_newspaper : ℕ := 40

/-- The number of dogs Julie plans to walk. -/
def dogs_to_walk : ℕ := 24

/-- The payment Julie receives for walking each dog. -/
def payment_per_dog : ℕ := 15

/-- The amount of money Julie has left after purchasing the bike. -/
def money_left : ℕ := 155

/-- Theorem stating that Julie's initial savings were $1190. -/
theorem julie_initial_savings :
  initial_savings = 1190 :=
by sorry

end NUMINAMATH_CALUDE_julie_initial_savings_l3765_376529


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3765_376587

theorem opposite_of_2023 : 
  ∃ x : ℤ, (x + 2023 = 0) ∧ (x = -2023) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3765_376587


namespace NUMINAMATH_CALUDE_jose_investment_is_225000_l3765_376552

/-- Calculates Jose's investment given the problem conditions -/
def calculate_jose_investment (tom_investment : ℕ) (tom_duration : ℕ) (jose_duration : ℕ) (total_profit : ℕ) (jose_profit : ℕ) : ℕ :=
  (tom_investment * tom_duration * (total_profit - jose_profit)) / (jose_profit * jose_duration)

/-- Proves that Jose's investment is 225000 given the problem conditions -/
theorem jose_investment_is_225000 :
  calculate_jose_investment 30000 12 10 27000 15000 = 225000 := by
  sorry

end NUMINAMATH_CALUDE_jose_investment_is_225000_l3765_376552


namespace NUMINAMATH_CALUDE_fraction_simplification_l3765_376568

theorem fraction_simplification (a b c d : ℝ) 
  (ha : a = Real.sqrt 125)
  (hb : b = 3 * Real.sqrt 45)
  (hc : c = 4 * Real.sqrt 20)
  (hd : d = Real.sqrt 75) :
  5 / (a + b + c + d) = Real.sqrt 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3765_376568


namespace NUMINAMATH_CALUDE_smallest_M_bound_l3765_376534

theorem smallest_M_bound : ∃ (M : ℕ),
  (∀ (a b c : ℝ), (∀ (x : ℝ), |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) →
    (∀ (x : ℝ), |x| ≤ 1 → |2*a*x + b| ≤ M)) ∧
  (∀ (N : ℕ), N < M →
    ∃ (a b c : ℝ), (∀ (x : ℝ), |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) ∧
      (∃ (x : ℝ), |x| ≤ 1 ∧ |2*a*x + b| > N)) ∧
  M = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_M_bound_l3765_376534


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l3765_376520

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_t_value :
  let m : ℝ × ℝ := (2, 8)
  let n : ℝ → ℝ × ℝ := fun t ↦ (-4, t)
  ∀ t, parallel m (n t) → t = -16 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l3765_376520


namespace NUMINAMATH_CALUDE_max_m_over_n_l3765_376509

open Real

noncomputable def f (m n x : ℝ) : ℝ := Real.exp (-x) + (n * x) / (m * x + n)

theorem max_m_over_n (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, x ≥ 0 → f m n x ≥ 1) ∧ f m n 0 = 1 →
  m / n ≤ (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_over_n_l3765_376509


namespace NUMINAMATH_CALUDE_betty_herb_garden_total_l3765_376502

/-- The number of basil plants in Betty's herb garden. -/
def basil_plants : ℕ := 5

/-- The number of oregano plants in Betty's herb garden. -/
def oregano_plants : ℕ := 2 * basil_plants + 2

/-- The total number of plants in Betty's herb garden. -/
def total_plants : ℕ := basil_plants + oregano_plants

/-- Theorem stating that the total number of plants in Betty's herb garden is 17. -/
theorem betty_herb_garden_total : total_plants = 17 := by
  sorry

end NUMINAMATH_CALUDE_betty_herb_garden_total_l3765_376502


namespace NUMINAMATH_CALUDE_point_coordinates_l3765_376585

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem stating the coordinates of point P given the conditions -/
theorem point_coordinates (p : Point) 
  (h1 : isInSecondQuadrant p)
  (h2 : distanceToXAxis p = 4)
  (h3 : distanceToYAxis p = 5) :
  p = Point.mk (-5) 4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3765_376585


namespace NUMINAMATH_CALUDE_tangency_condition_min_area_triangle_l3765_376530

/-- The curve C: x^2 + y^2 - 2x - 2y + 1 = 0 -/
def curve (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The line l: bx + ay = ab -/
def line (a b x y : ℝ) : Prop :=
  b*x + a*y = a*b

/-- The line l is tangent to the curve C -/
def is_tangent (a b : ℝ) : Prop :=
  ∃ x y, curve x y ∧ line a b x y

theorem tangency_condition (a b : ℝ) (ha : a > 2) (hb : b > 2) (h_tangent : is_tangent a b) :
  (a - 2) * (b - 2) = 2 :=
sorry

theorem min_area_triangle (a b : ℝ) (ha : a > 2) (hb : b > 2) (h_tangent : is_tangent a b) :
  ∃ area : ℝ, area = 3 + 2 * Real.sqrt 2 ∧ 
  ∀ a' b', a' > 2 → b' > 2 → is_tangent a' b' → (1/2 * a' * b' ≥ area) :=
sorry

end NUMINAMATH_CALUDE_tangency_condition_min_area_triangle_l3765_376530


namespace NUMINAMATH_CALUDE_trig_problem_l3765_376548

theorem trig_problem (α β : ℝ) 
  (h1 : Real.sin α - Real.sin β = -1/3)
  (h2 : Real.cos α - Real.cos β = 1/2)
  (h3 : Real.tan (α + β) = 2/5)
  (h4 : Real.tan (β - π/4) = 1/4) :
  Real.cos (α - β) = 59/72 ∧ Real.tan (α + π/4) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l3765_376548


namespace NUMINAMATH_CALUDE_compressor_stations_theorem_l3765_376565

/-- Represents the configuration of three compressor stations -/
structure CompressorStations where
  x : ℝ  -- Distance between first and second stations
  y : ℝ  -- Distance between second and third stations
  z : ℝ  -- Distance between first and third stations
  a : ℝ  -- Additional parameter

/-- The conditions for the compressor stations configuration -/
def validConfiguration (s : CompressorStations) : Prop :=
  s.x > 0 ∧ s.y > 0 ∧ s.z > 0 ∧  -- Positive distances
  s.x + s.y = 2 * s.z ∧          -- Condition 1
  s.x + s.z = s.y + s.a ∧        -- Condition 2
  s.x + s.z = 75                 -- Condition 3

/-- The theorem stating the properties of the compressor stations configuration -/
theorem compressor_stations_theorem (s : CompressorStations) 
  (h : validConfiguration s) : 
  0 < s.a ∧ s.a < 100 ∧ 
  (s.a = 15 → s.x = 42 ∧ s.y = 24 ∧ s.z = 33) := by
  sorry

#check compressor_stations_theorem

end NUMINAMATH_CALUDE_compressor_stations_theorem_l3765_376565


namespace NUMINAMATH_CALUDE_theodore_sturgeon_collection_hardcovers_l3765_376579

/-- Given a collection of books with two price options and a total cost,
    calculate the number of books purchased at the higher price. -/
def hardcover_count (total_volumes : ℕ) (paperback_price hardcover_price : ℕ) (total_cost : ℕ) : ℕ :=
  let h := (2 * total_cost - paperback_price * total_volumes) / (2 * (hardcover_price - paperback_price))
  h

/-- Theorem stating that given the specific conditions of the problem,
    the number of hardcover books purchased is 6. -/
theorem theodore_sturgeon_collection_hardcovers :
  hardcover_count 12 15 30 270 = 6 := by
  sorry

end NUMINAMATH_CALUDE_theodore_sturgeon_collection_hardcovers_l3765_376579


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_fifth_powers_l3765_376532

theorem highest_power_of_two_dividing_difference_of_fifth_powers :
  ∃ k : ℕ, k = 5 ∧ 2^k ∣ (17^5 - 15^5) ∧ ∀ m : ℕ, 2^m ∣ (17^5 - 15^5) → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_fifth_powers_l3765_376532


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_l3765_376560

theorem tangent_line_to_ln (k : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ k * x₀ = Real.log x₀ ∧ k = 1 / x₀) → k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_l3765_376560


namespace NUMINAMATH_CALUDE_simplify_expression_l3765_376592

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^10 = 1342177280 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3765_376592


namespace NUMINAMATH_CALUDE_perpendicular_lines_and_circle_l3765_376511

-- Define the lines and circle
def l₁ (a x y : ℝ) : Prop := a * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def C (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 8*y + 21 = 0

-- State the theorem
theorem perpendicular_lines_and_circle 
  (a : ℝ) -- Coefficient of x in l₁
  (h_perp : a * 2 + 4 = 0) -- Perpendicularity condition
  : 
  -- Part 1: Intersection point
  (∃ x y : ℝ, l₁ a x y ∧ l₂ x y ∧ x = -1 ∧ y = 0) ∧ 
  -- Part 2: No common points between l₁ and C
  (∀ x y : ℝ, ¬(l₁ a x y ∧ C x y)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_and_circle_l3765_376511


namespace NUMINAMATH_CALUDE_unique_functional_equation_l3765_376558

/-- Given g: ℂ → ℂ, w ∈ ℂ, a ∈ ℂ, where w³ = 1 and w ≠ 1, 
    prove that the unique function f: ℂ → ℂ satisfying 
    f(z) + f(wz + a) = g(z) for all z ∈ ℂ 
    is given by f(z) = (g(z) + g(w²z + wa + a) - g(wz + a)) / 2 -/
theorem unique_functional_equation (g : ℂ → ℂ) (w a : ℂ) 
    (hw : w^3 = 1) (hw_neq : w ≠ 1) :
    ∃! f : ℂ → ℂ, ∀ z : ℂ, f z + f (w * z + a) = g z ∧
    f = fun z ↦ (g z + g (w^2 * z + w * a + a) - g (w * z + a)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_l3765_376558


namespace NUMINAMATH_CALUDE_camera_pictures_l3765_376549

def picture_problem (total_albums : ℕ) (pics_per_album : ℕ) (pics_from_phone : ℕ) : Prop :=
  let total_pics := total_albums * pics_per_album
  total_pics - pics_from_phone = 13

theorem camera_pictures :
  picture_problem 5 4 7 := by
  sorry

end NUMINAMATH_CALUDE_camera_pictures_l3765_376549


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_of_squares_l3765_376516

theorem polynomial_coefficient_sum_of_squares 
  (a b c d e f : ℤ) 
  (h : ∀ x : ℝ, 8 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) : 
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 356 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_of_squares_l3765_376516


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l3765_376589

theorem no_solution_quadratic_inequality :
  ∀ x : ℝ, ¬(3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l3765_376589


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3765_376584

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  prop1 : a 5 * a 8 = 6
  prop2 : a 3 + a 10 = 5

/-- The ratio of a_20 to a_13 in the geometric sequence is either 3/2 or 2/3 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 20 / seq.a 13 = 3/2 ∨ seq.a 20 / seq.a 13 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3765_376584


namespace NUMINAMATH_CALUDE_min_value_f_min_m_value_l3765_376569

noncomputable section

def f (a b x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x + b

theorem min_value_f (a b : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a b x ≥ 1/2 + b) ∧
  (1 < a → a < 4 → ∀ x ∈ Set.Icc 1 2, f a b x ≥ a/2 - a * Real.log (Real.sqrt a) + b) ∧
  (4 ≤ a → ∀ x ∈ Set.Icc 1 2, f a b x ≥ 2 - a * Real.log 2 + b) :=
sorry

theorem min_m_value (a b : ℝ) (h : -2 ≤ a ∧ a < 0) :
  ∃ m : ℝ, m = 12 ∧ 
  ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo 0 2 → x₂ ∈ Set.Ioo 0 2 →
  |f a b x₁ - f a b x₂| ≤ m * |1/x₁ - 1/x₂| :=
sorry

end NUMINAMATH_CALUDE_min_value_f_min_m_value_l3765_376569


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l3765_376581

theorem reciprocal_inequality (a b : ℝ) :
  (∀ a b, b < a ∧ a < 0 → 1/b > 1/a) ∧
  (∃ a b, 1/b > 1/a ∧ ¬(b < a ∧ a < 0)) :=
sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l3765_376581


namespace NUMINAMATH_CALUDE_cubic_equation_root_squared_l3765_376551

theorem cubic_equation_root_squared (r : ℝ) : 
  r^3 - r + 3 = 0 → (r^2)^3 - 2*(r^2)^2 + r^2 - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_squared_l3765_376551


namespace NUMINAMATH_CALUDE_yarn_length_proof_l3765_376541

theorem yarn_length_proof (green_length red_length total_length : ℕ) : 
  green_length = 156 ∧ 
  red_length = 3 * green_length + 8 →
  total_length = green_length + red_length →
  total_length = 632 := by
sorry

end NUMINAMATH_CALUDE_yarn_length_proof_l3765_376541


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l3765_376590

theorem polynomial_root_problem (a b : ℝ) : 
  (∀ x : ℝ, a*x^4 + (a + b)*x^3 + (b - 2*a)*x^2 + 5*b*x + (12 - a) = 0 ↔ 
    x = 1 ∨ x = -3 ∨ x = 4 ∨ x = -92/297) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l3765_376590


namespace NUMINAMATH_CALUDE_non_degenerate_ellipse_condition_l3765_376556

/-- The equation of an ellipse in general form -/
def ellipse_equation (x y k : ℝ) : Prop :=
  2 * x^2 + 9 * y^2 - 12 * x - 27 * y = k

/-- Condition for the equation to represent a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -135/4

/-- Theorem stating the condition for a non-degenerate ellipse -/
theorem non_degenerate_ellipse_condition :
  ∀ k, (∃ x y, ellipse_equation x y k) ∧ is_non_degenerate_ellipse k ↔
    (∀ x y, ellipse_equation x y k → is_non_degenerate_ellipse k) :=
by sorry

end NUMINAMATH_CALUDE_non_degenerate_ellipse_condition_l3765_376556


namespace NUMINAMATH_CALUDE_books_together_l3765_376504

-- Define the number of books Tim and Mike have
def tim_books : ℕ := 22
def mike_books : ℕ := 20

-- Define the total number of books
def total_books : ℕ := tim_books + mike_books

-- Theorem to prove
theorem books_together : total_books = 42 := by
  sorry

end NUMINAMATH_CALUDE_books_together_l3765_376504


namespace NUMINAMATH_CALUDE_system_solution_l3765_376527

theorem system_solution :
  ∃ (a b c d : ℝ),
    (a + c = -1) ∧
    (a * c + b + d = -1) ∧
    (a * d + b * c = -5) ∧
    (b * d = 6) ∧
    ((a = -3 ∧ b = 2 ∧ c = 2 ∧ d = 3) ∨
     (a = 2 ∧ b = 3 ∧ c = -3 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3765_376527


namespace NUMINAMATH_CALUDE_babylonian_conversion_l3765_376535

/-- Converts a Babylonian sexagesimal number to its decimal representation -/
def babylonian_to_decimal (a b : ℕ) : ℕ :=
  60^a + 10 * 60^b

/-- The Babylonian number 60^8 + 10 * 60^7 in decimal form -/
theorem babylonian_conversion :
  babylonian_to_decimal 8 7 = 195955200000000 := by
  sorry

#eval babylonian_to_decimal 8 7

end NUMINAMATH_CALUDE_babylonian_conversion_l3765_376535


namespace NUMINAMATH_CALUDE_largest_quantity_l3765_376571

def D : ℚ := 2007 / 2006 + 2007 / 2008
def E : ℚ := 2008 / 2007 + 2010 / 2007
def F : ℚ := 2009 / 2008 + 2009 / 2010

theorem largest_quantity : E > D ∧ E > F := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l3765_376571


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3765_376547

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3765_376547


namespace NUMINAMATH_CALUDE_henrys_deductions_l3765_376510

/-- Henry's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- State tax rate as a decimal -/
def tax_rate : ℚ := 21 / 1000

/-- Fixed community fee in dollars per hour -/
def community_fee : ℚ := 1 / 2

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℚ := 100

/-- Calculate the total deductions in cents -/
def total_deductions : ℚ :=
  hourly_wage * tax_rate * dollars_to_cents + community_fee * dollars_to_cents

theorem henrys_deductions :
  total_deductions = 102.5 := by sorry

end NUMINAMATH_CALUDE_henrys_deductions_l3765_376510


namespace NUMINAMATH_CALUDE_sum_of_products_inequality_l3765_376540

theorem sum_of_products_inequality (a b c d : ℝ) (h : a + b + c + d = 1) :
  a * b + b * c + c * d + d * a ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_inequality_l3765_376540


namespace NUMINAMATH_CALUDE_blue_parrots_count_l3765_376537

theorem blue_parrots_count (total : ℕ) (green_fraction : ℚ) (blue_parrots : ℕ) : 
  total = 160 →
  green_fraction = 5/8 →
  blue_parrots = total - (green_fraction * total).num →
  blue_parrots = 60 := by
sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l3765_376537


namespace NUMINAMATH_CALUDE_initial_term_range_l3765_376521

/-- A strictly increasing sequence satisfying the given recursive formula -/
def StrictlyIncreasingSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧ 
  (∀ n, a (n + 1) = (4 * a n - 2) / (a n + 1))

/-- The theorem stating that the initial term of the sequence must be in (1, 2) -/
theorem initial_term_range (a : ℕ → ℝ) :
  StrictlyIncreasingSequence a → 1 < a 1 ∧ a 1 < 2 := by
  sorry


end NUMINAMATH_CALUDE_initial_term_range_l3765_376521


namespace NUMINAMATH_CALUDE_inequality_theorem_l3765_376578

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : y₁ > 0) (hy₂ : y₂ > 0)
  (hz₁ : x₁ * y₁ - z₁^2 > 0) (hz₂ : x₂ * y₂ - z₂^2 > 0) :
  ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2)⁻¹ ≤ (x₁ * y₁ - z₁^2)⁻¹ + (x₂ * y₂ - z₂^2)⁻¹ ∧
  (((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2)⁻¹ = (x₁ * y₁ - z₁^2)⁻¹ + (x₂ * y₂ - z₂^2)⁻¹ ↔ 
   x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3765_376578


namespace NUMINAMATH_CALUDE_brothers_age_equation_l3765_376593

theorem brothers_age_equation (x : ℝ) (h1 : x > 0) : 
  (x - 6) + (2*x - 6) = 15 :=
by
  sorry

#check brothers_age_equation

end NUMINAMATH_CALUDE_brothers_age_equation_l3765_376593


namespace NUMINAMATH_CALUDE_smallest_c_value_l3765_376545

theorem smallest_c_value (c d : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x : ℝ, x^3 - c*x^2 + d*x - 2550 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  r₁ * r₂ * r₃ = 2550 →
  c = r₁ + r₂ + r₃ →
  c ≥ 42 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l3765_376545


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3765_376526

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 2 →
    (48 * x^2 + 26 * x - 35) / (x^2 - 3 * x + 2) = M₁ / (x - 1) + M₂ / (x - 2)) →
  M₁ * M₂ = -1056 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3765_376526


namespace NUMINAMATH_CALUDE_perfect_square_fraction_solutions_l3765_376557

theorem perfect_square_fraction_solutions :
  ∀ m n p : ℕ+,
  p.val.Prime →
  (∃ k : ℕ+, ((5^(m.val) + 2^(n.val) * p.val) : ℚ) / (5^(m.val) - 2^(n.val) * p.val) = (k.val : ℚ)^2) →
  ((m = 1 ∧ n = 1 ∧ p = 2) ∨ (m = 2 ∧ n = 3 ∧ p = 3) ∨ (m = 2 ∧ n = 2 ∧ p = 5)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_fraction_solutions_l3765_376557


namespace NUMINAMATH_CALUDE_total_food_consumption_l3765_376544

/-- The amount of food needed per soldier per day on the first side -/
def food_per_soldier_first : ℕ := 10

/-- The amount of food needed per soldier per day on the second side -/
def food_per_soldier_second : ℕ := food_per_soldier_first - 2

/-- The number of soldiers on the first side -/
def soldiers_first : ℕ := 4000

/-- The number of soldiers on the second side -/
def soldiers_second : ℕ := soldiers_first - 500

/-- The total amount of food consumed by both sides per day -/
def total_food : ℕ := soldiers_first * food_per_soldier_first + soldiers_second * food_per_soldier_second

theorem total_food_consumption :
  total_food = 68000 := by sorry

end NUMINAMATH_CALUDE_total_food_consumption_l3765_376544


namespace NUMINAMATH_CALUDE_missing_number_in_mean_l3765_376550

theorem missing_number_in_mean (known_numbers : List ℤ) (mean : ℚ) : 
  known_numbers = [22, 23, 24, 25, 26, 27, 2] ∧ 
  mean = 20 ∧ 
  (List.sum known_numbers + (missing_number : ℤ)) / 7 = mean →
  missing_number = -9 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_number_in_mean_l3765_376550


namespace NUMINAMATH_CALUDE_smallest_earring_collection_l3765_376595

theorem smallest_earring_collection (M : ℕ) : 
  M > 2 ∧ 
  M % 7 = 2 ∧ 
  M % 11 = 2 ∧ 
  M % 13 = 2 → 
  M ≥ 1003 :=
by sorry

end NUMINAMATH_CALUDE_smallest_earring_collection_l3765_376595


namespace NUMINAMATH_CALUDE_min_value_theorem_l3765_376536

theorem min_value_theorem (x A B C : ℝ) (hx : x > 0) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hxA : x^2 + 1/x^2 = A)
  (hxB : x - 1/x = B)
  (hxC : x^3 - 1/x^3 = C) :
  ∃ (m : ℝ), m = 6.4 ∧ ∀ (A' B' C' x' : ℝ), 
    x' > 0 → A' > 0 → B' > 0 → C' > 0 →
    x'^2 + 1/x'^2 = A' →
    x' - 1/x' = B' →
    x'^3 - 1/x'^3 = C' →
    A'^3 / C' ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3765_376536


namespace NUMINAMATH_CALUDE_five_two_difference_in_book_pages_l3765_376582

/-- Count occurrences of a digit in a range of numbers -/
def countDigit (d : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The difference between occurrences of 5 and 2 in page numbers -/
def diffFiveTwo (totalPages : Nat) : Int :=
  (countDigit 5 1 totalPages : Int) - (countDigit 2 1 totalPages : Int)

/-- Theorem stating the difference between 5's and 2's in a 625-page book -/
theorem five_two_difference_in_book_pages : diffFiveTwo 625 = 20 := by
  sorry

end NUMINAMATH_CALUDE_five_two_difference_in_book_pages_l3765_376582


namespace NUMINAMATH_CALUDE_nearest_integer_to_x_minus_y_l3765_376543

theorem nearest_integer_to_x_minus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + y = 5) (h2 : |x| * y - x^3 = 0) : x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_x_minus_y_l3765_376543


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_merry_go_round_specific_case_l3765_376506

/-- Given two circular paths with different radii, prove that the number of revolutions
    needed to cover the same distance is inversely proportional to their radii. -/
theorem merry_go_round_revolutions (r1 r2 n1 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hn1 : n1 > 0) :
  let n2 := (r1 * n1) / r2
  2 * Real.pi * r1 * n1 = 2 * Real.pi * r2 * n2 := by sorry

/-- Prove that for the specific case of r1 = 30, r2 = 10, and n1 = 36, 
    the number of revolutions n2 for the second path is 108. -/
theorem merry_go_round_specific_case :
  let r1 : ℝ := 30
  let r2 : ℝ := 10
  let n1 : ℝ := 36
  let n2 := (r1 * n1) / r2
  n2 = 108 := by sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_merry_go_round_specific_case_l3765_376506


namespace NUMINAMATH_CALUDE_angle_system_solution_l3765_376563

theorem angle_system_solution (k : ℤ) :
  let x : ℝ := π/3 + k*π
  let y : ℝ := k*π
  (x - y = π/3) ∧ (Real.tan x - Real.tan y = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_angle_system_solution_l3765_376563


namespace NUMINAMATH_CALUDE_cat_mouse_positions_after_360_moves_l3765_376564

/-- Represents the number of squares in the cat's path -/
def cat_squares : ℕ := 5

/-- Represents the number of segments in the mouse's path -/
def mouse_segments : ℕ := 10

/-- Represents the number of segments the mouse moves per turn -/
def mouse_move_rate : ℕ := 2

/-- Represents the total number of moves -/
def total_moves : ℕ := 360

/-- Calculates the cat's position after a given number of moves -/
def cat_position (moves : ℕ) : ℕ :=
  moves % cat_squares + 1

/-- Calculates the mouse's effective moves after accounting for skipped segments -/
def mouse_effective_moves (moves : ℕ) : ℕ :=
  (moves / mouse_segments) * (mouse_segments - 1) + (moves % mouse_segments)

/-- Calculates the mouse's position after a given number of effective moves -/
def mouse_position (effective_moves : ℕ) : ℕ :=
  (effective_moves * mouse_move_rate) % mouse_segments + 1

theorem cat_mouse_positions_after_360_moves :
  cat_position total_moves = 1 ∧ 
  mouse_position (mouse_effective_moves total_moves) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cat_mouse_positions_after_360_moves_l3765_376564


namespace NUMINAMATH_CALUDE_contacts_in_sphere_tetrahedron_l3765_376538

/-- The number of contacts in a tetrahedral stack of spheres -/
def tetrahedron_contacts (n : ℕ) : ℕ := n^3 - n

/-- 
Theorem: In a tetrahedron formed by stacking identical spheres, 
where each edge has n spheres, the total number of points of 
tangency between the spheres is n³ - n.
-/
theorem contacts_in_sphere_tetrahedron (n : ℕ) : 
  tetrahedron_contacts n = n^3 - n := by
  sorry

end NUMINAMATH_CALUDE_contacts_in_sphere_tetrahedron_l3765_376538


namespace NUMINAMATH_CALUDE_bead_problem_l3765_376507

theorem bead_problem (blue_beads : Nat) (yellow_beads : Nat) : 
  blue_beads = 23 → 
  yellow_beads = 16 → 
  let total_beads := blue_beads + yellow_beads
  let parts := 3
  let beads_per_part := total_beads / parts
  let removed_beads := 10
  let remaining_beads := beads_per_part - removed_beads
  let final_beads := remaining_beads * 2
  final_beads = 6 := by sorry

end NUMINAMATH_CALUDE_bead_problem_l3765_376507


namespace NUMINAMATH_CALUDE_temperature_at_speed_0_4_l3765_376567

/-- The temperature in degrees Celsius given the speed of sound in meters per second -/
def temperature (v : ℝ) : ℝ := 15 * v^2

/-- Theorem: When the speed of sound is 0.4 m/s, the temperature is 2.4°C -/
theorem temperature_at_speed_0_4 : temperature 0.4 = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_temperature_at_speed_0_4_l3765_376567


namespace NUMINAMATH_CALUDE_school_qualification_percentage_l3765_376570

theorem school_qualification_percentage 
  (school_a_qualification_rate : ℝ)
  (school_b_appearance_increase : ℝ)
  (school_b_qualification_increase : ℝ)
  (h1 : school_a_qualification_rate = 0.7)
  (h2 : school_b_appearance_increase = 0.2)
  (h3 : school_b_qualification_increase = 0.5)
  : (school_a_qualification_rate * (1 + school_b_qualification_increase)) / 
    (1 + school_b_appearance_increase) = 0.875 := by
  sorry

#check school_qualification_percentage

end NUMINAMATH_CALUDE_school_qualification_percentage_l3765_376570


namespace NUMINAMATH_CALUDE_star_op_value_l3765_376597

-- Define the * operation for non-zero integers
def star_op (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

-- Theorem statement
theorem star_op_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a + b = 15 → a * b = 56 → star_op a b = 15 / 56 := by
  sorry

end NUMINAMATH_CALUDE_star_op_value_l3765_376597


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_squared_l3765_376528

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ

/-- The conditions of the trapezoid as described in the problem -/
def trapezoid_conditions (t : Trapezoid) : Prop :=
  -- The longer base is 150 units longer than the shorter base
  ∃ (longer_base : ℝ), longer_base = t.shorter_base + 150
  -- The midline divides the trapezoid into regions with area ratio 3:4
  ∧ (t.shorter_base + t.shorter_base + 75) / (t.shorter_base + 75 + t.shorter_base + 150) = 3 / 4
  -- t.equal_area_segment divides the trapezoid into two equal-area regions
  ∧ ∃ (h₁ : ℝ), 2 * (1/2 * h₁ * (t.shorter_base + t.equal_area_segment)) = 
                 1/2 * t.height * (t.shorter_base + t.shorter_base + 150)

/-- The theorem to be proved -/
theorem trapezoid_segment_length_squared (t : Trapezoid) 
  (h : trapezoid_conditions t) : 
  ⌊t.equal_area_segment^2 / 150⌋ = 300 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_squared_l3765_376528


namespace NUMINAMATH_CALUDE_billy_crayons_l3765_376555

/-- The number of crayons left after a monkey and hippopotamus eat some crayons -/
def crayons_left (total : ℕ) (monkey_ate : ℕ) : ℕ :=
  total - (monkey_ate + 2 * monkey_ate)

/-- Theorem stating that given 200 total crayons, if a monkey eats 64 crayons,
    then 8 crayons are left -/
theorem billy_crayons : crayons_left 200 64 = 8 := by
  sorry

end NUMINAMATH_CALUDE_billy_crayons_l3765_376555


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l3765_376500

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) (h_arith : ArithmeticSequence a)
    (h_a4 : a 4 = 4) (h_sum : a 3 + a 8 = 5) : a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l3765_376500


namespace NUMINAMATH_CALUDE_people_owning_only_dogs_l3765_376553

theorem people_owning_only_dogs :
  let total_pet_owners : ℕ := 79
  let only_cats : ℕ := 10
  let cats_and_dogs : ℕ := 5
  let cats_dogs_snakes : ℕ := 3
  let total_snakes : ℕ := 49
  let only_dogs : ℕ := total_pet_owners - only_cats - cats_and_dogs - cats_dogs_snakes - (total_snakes - cats_dogs_snakes)
  only_dogs = 15 :=
by sorry

end NUMINAMATH_CALUDE_people_owning_only_dogs_l3765_376553


namespace NUMINAMATH_CALUDE_fraction_repeating_block_length_l3765_376503

/-- The number of digits in the smallest repeating block of the decimal expansion of 3/11 -/
def smallest_repeating_block_length : ℕ := 2

/-- The fraction we're considering -/
def fraction : ℚ := 3 / 11

theorem fraction_repeating_block_length :
  smallest_repeating_block_length = 2 ∧ 
  ∃ (a b : ℕ), fraction = (a : ℚ) / (10^smallest_repeating_block_length - 1 : ℚ) + (b : ℚ) / (10^smallest_repeating_block_length : ℚ) :=
sorry

end NUMINAMATH_CALUDE_fraction_repeating_block_length_l3765_376503


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l3765_376505

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 9)) / x ≤ 3 - Real.sqrt 6 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, x > 0 ∧ (x^2 + 3 - Real.sqrt (x^4 + 9)) / x = 3 - Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l3765_376505


namespace NUMINAMATH_CALUDE_problem_solution_l3765_376594

theorem problem_solution (p q r s : ℕ+) 
  (h1 : p^3 = q^2) 
  (h2 : r^4 = s^3) 
  (h3 : r - p = 17) : 
  s - q = 73 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3765_376594


namespace NUMINAMATH_CALUDE_kendra_shirts_l3765_376599

/-- Represents the number of shirts Kendra needs for a two-week period --/
def shirts_needed : ℕ :=
  let weekday_shirts := 5
  let club_shirts := 3
  let saturday_shirt := 1
  let sunday_shirts := 2
  let weekly_shirts := weekday_shirts + club_shirts + saturday_shirt + sunday_shirts
  2 * weekly_shirts

/-- Theorem stating that Kendra needs 22 shirts for a two-week period --/
theorem kendra_shirts : shirts_needed = 22 := by
  sorry

end NUMINAMATH_CALUDE_kendra_shirts_l3765_376599


namespace NUMINAMATH_CALUDE_complex_number_location_l3765_376562

theorem complex_number_location (Z : ℂ) : Z = Complex.I :=
  by
  -- Define Z
  have h1 : Z = (Real.sqrt 2 - Complex.I ^ 3) / (1 - Real.sqrt 2 * Complex.I) := by sorry
  
  -- Define properties of complex numbers
  have h2 : Complex.I ^ 2 = -1 := by sorry
  have h3 : Complex.I ^ 3 = -Complex.I := by sorry
  
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3765_376562


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l3765_376514

theorem quadratic_root_implies_k (k : ℝ) : 
  ((-1 : ℝ)^2 - 2*k*(-1) + k^2 = 0) → k = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l3765_376514


namespace NUMINAMATH_CALUDE_geometry_test_passing_l3765_376546

theorem geometry_test_passing (total_problems : Nat) (passing_percentage : Rat) 
  (hp : total_problems = 50)
  (hq : passing_percentage = 85 / 100) : 
  (max_missed_problems : Nat) → 
  (max_missed_problems = total_problems - Int.ceil (passing_percentage * total_problems)) ∧
  max_missed_problems = 7 := by
  sorry

end NUMINAMATH_CALUDE_geometry_test_passing_l3765_376546


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l3765_376523

-- Define the hexagon
structure Hexagon :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DE : ℝ)
  (EF : ℝ)
  (AC : ℝ)
  (AD : ℝ)
  (AE : ℝ)
  (AF : ℝ)

-- Define the theorem
theorem hexagon_perimeter (h : Hexagon) :
  h.AB = 1 →
  h.BC = 2 →
  h.CD = 2 →
  h.DE = 2 →
  h.EF = 3 →
  h.AC^2 = h.AB^2 + h.BC^2 →
  h.AD^2 = h.AC^2 + h.CD^2 →
  h.AE^2 = h.AD^2 + h.DE^2 →
  h.AF^2 = h.AE^2 + h.EF^2 →
  h.AB + h.BC + h.CD + h.DE + h.EF + h.AF = 10 + Real.sqrt 22 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l3765_376523


namespace NUMINAMATH_CALUDE_min_throws_correct_l3765_376554

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def minSum : ℕ := numDice

/-- The maximum possible sum when rolling the dice -/
def maxSum : ℕ := numDice * numFaces

/-- The number of possible unique sums -/
def numUniqueSums : ℕ := maxSum - minSum + 1

/-- The minimum number of throws required to ensure the same sum is rolled twice -/
def minThrows : ℕ := numUniqueSums + 1

/-- Theorem stating that minThrows is the minimum number of throws required -/
theorem min_throws_correct :
  minThrows = 22 ∧
  ∀ n : ℕ, n < minThrows → ∃ outcome : Fin n → Fin (maxSum - minSum + 1),
    Function.Injective outcome :=
by sorry

end NUMINAMATH_CALUDE_min_throws_correct_l3765_376554


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3765_376542

def m : Fin 2 → ℝ := ![(-1), 2]
def n (b : ℝ) : Fin 2 → ℝ := ![2, b]

theorem vector_difference_magnitude (b : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ m = k • n b) →
  ‖m - n b‖ = 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3765_376542


namespace NUMINAMATH_CALUDE_sum_a3_a5_equals_5_l3765_376508

/-- A positive arithmetic-geometric sequence -/
def PositiveArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r q : ℝ, ∀ k, a (k + 1) = r * a k + q

/-- The theorem statement -/
theorem sum_a3_a5_equals_5 (a : ℕ → ℝ) 
  (h_seq : PositiveArithmeticGeometricSequence a)
  (h_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) :
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_a3_a5_equals_5_l3765_376508


namespace NUMINAMATH_CALUDE_basketball_team_selection_l3765_376580

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def players_to_select : ℕ := 7
def quadruplets_to_select : ℕ := 3

theorem basketball_team_selection :
  (Nat.choose quadruplets quadruplets_to_select) *
  (Nat.choose (total_players - quadruplets) (players_to_select - quadruplets_to_select)) = 1980 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l3765_376580


namespace NUMINAMATH_CALUDE_replaced_person_weight_l3765_376598

/-- Given a group of 10 persons, if replacing one person with a new person
    weighing 100 kg increases the average weight by 3.5 kg,
    then the weight of the replaced person is 65 kg. -/
theorem replaced_person_weight
  (n : ℕ) (initial_average : ℝ) (new_person_weight : ℝ) (average_increase : ℝ) :
  n = 10 →
  new_person_weight = 100 →
  average_increase = 3.5 →
  initial_average + average_increase = (n * initial_average - replaced_weight + new_person_weight) / n →
  replaced_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l3765_376598


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l3765_376566

theorem sin_thirteen_pi_sixths : Real.sin (13 * Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l3765_376566


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3765_376586

theorem unique_four_digit_number : ∃! n : ℕ,
  (1000 ≤ n) ∧ (n < 10000) ∧  -- 4-digit number
  (∃ a : ℕ, n = a^2) ∧  -- perfect square
  (∃ b : ℕ, n % 1000 = b^3) ∧  -- removing first digit results in a perfect cube
  (∃ c : ℕ, n % 100 = c^4) ∧  -- removing first two digits results in a fourth power
  n = 9216 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3765_376586


namespace NUMINAMATH_CALUDE_cos_sin_pi_12_product_l3765_376561

theorem cos_sin_pi_12_product (π : Real) : 
  (Real.cos (π / 12) - Real.sin (π / 12)) * (Real.cos (π / 12) + Real.sin (π / 12)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_pi_12_product_l3765_376561


namespace NUMINAMATH_CALUDE_spirangle_length_is_10301_l3765_376575

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def spirangle_length (a₁ : ℕ) (d : ℕ) (last_seq : ℕ) (final_seg : ℕ) : ℕ :=
  let n := (last_seq - a₁) / d + 1
  arithmetic_sequence_sum a₁ d n + final_seg

theorem spirangle_length_is_10301 :
  spirangle_length 2 2 200 201 = 10301 :=
by sorry

end NUMINAMATH_CALUDE_spirangle_length_is_10301_l3765_376575


namespace NUMINAMATH_CALUDE_masha_creates_more_words_l3765_376525

/-- Represents a word as a list of characters -/
def Word := List Char

/-- Counts the number of distinct words formed by removing exactly two letters from a given word -/
def countDistinctWordsRemovingTwo (w : Word) : Nat :=
  sorry

/-- The word "ИНТЕГРИРОВАНИЕ" -/
def integrirovanie : Word :=
  ['И', 'Н', 'Т', 'Е', 'Г', 'Р', 'И', 'Р', 'О', 'В', 'А', 'Н', 'И', 'Е']

/-- The word "СУПЕРКОМПЬЮТЕР" -/
def superkomputer : Word :=
  ['С', 'У', 'П', 'Е', 'Р', 'К', 'О', 'М', 'П', 'Ь', 'Ю', 'Т', 'Е', 'Р']

theorem masha_creates_more_words :
  countDistinctWordsRemovingTwo superkomputer > countDistinctWordsRemovingTwo integrirovanie :=
sorry

end NUMINAMATH_CALUDE_masha_creates_more_words_l3765_376525


namespace NUMINAMATH_CALUDE_common_divisor_nineteen_l3765_376518

theorem common_divisor_nineteen (a : ℤ) : Int.gcd (35 * a + 57) (45 * a + 76) = 19 := by
  sorry

end NUMINAMATH_CALUDE_common_divisor_nineteen_l3765_376518


namespace NUMINAMATH_CALUDE_percentage_relation_l3765_376573

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.06 * x) (h2 : b = 0.18 * x) :
  a / b * 100 = 100 / 3 :=
by sorry

end NUMINAMATH_CALUDE_percentage_relation_l3765_376573


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3765_376559

theorem unique_integer_solution (m : ℤ) : 
  (∃! (x : ℤ), |2*x - m| ≤ 1) ∧ 
  (∀ (x : ℤ), |2*x - m| ≤ 1 → x = 2) → 
  m = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3765_376559


namespace NUMINAMATH_CALUDE_comparison_and_inequality_l3765_376519

theorem comparison_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^2 + b^2 ≥ 2*(2*a - b) - 5 ∧ 
  a^a * b^b ≥ (a*b)^((a+b)/2) ∧
  (a^a * b^b = (a*b)^((a+b)/2) ↔ a = b) := by sorry

end NUMINAMATH_CALUDE_comparison_and_inequality_l3765_376519


namespace NUMINAMATH_CALUDE_base_eight_digits_of_1728_l3765_376513

theorem base_eight_digits_of_1728 : ∃ n : ℕ, n > 0 ∧ 8^(n-1) ≤ 1728 ∧ 1728 < 8^n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_digits_of_1728_l3765_376513


namespace NUMINAMATH_CALUDE_relationship_abc_l3765_376591

theorem relationship_abc (a b c : ℝ) : 
  a = (2/5)^(2/5) → 
  b = (3/5)^(2/5) → 
  c = Real.log (2/5) / Real.log (3/5) → 
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3765_376591


namespace NUMINAMATH_CALUDE_card_count_l3765_376524

theorem card_count (black red spades diamonds hearts clubs : ℕ) : 
  black = 7 →
  red = 6 →
  diamonds = 2 * spades →
  hearts = 2 * diamonds →
  clubs = 6 →
  black = spades + clubs →
  red = diamonds + hearts →
  spades + diamonds + hearts + clubs = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_card_count_l3765_376524
