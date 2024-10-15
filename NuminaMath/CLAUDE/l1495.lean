import Mathlib

namespace NUMINAMATH_CALUDE_remainder_problem_l1495_149502

theorem remainder_problem : 123456789012 % 240 = 132 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1495_149502


namespace NUMINAMATH_CALUDE_square_rolling_octagon_l1495_149530

/-- Represents the faces of a square -/
inductive SquareFace
  | Left
  | Right
  | Top
  | Bottom

/-- Represents the rotation of a square -/
def squareRotation (n : ℕ) : ℕ := n * 135

/-- The final position of an object on a square face after rolling around an octagon -/
def finalPosition (initialFace : SquareFace) : SquareFace :=
  match (squareRotation 4) % 360 with
  | 180 => match initialFace with
    | SquareFace.Left => SquareFace.Right
    | SquareFace.Right => SquareFace.Left
    | SquareFace.Top => SquareFace.Bottom
    | SquareFace.Bottom => SquareFace.Top
  | _ => initialFace

theorem square_rolling_octagon :
  finalPosition SquareFace.Left = SquareFace.Right :=
by sorry

end NUMINAMATH_CALUDE_square_rolling_octagon_l1495_149530


namespace NUMINAMATH_CALUDE_pasture_feeding_duration_l1495_149512

/-- Represents the daily grass consumption of a single cow -/
def daily_consumption_per_cow : ℝ := sorry

/-- Represents the initial amount of grass in the pasture -/
def initial_grass : ℝ := sorry

/-- Represents the daily growth rate of grass -/
def daily_growth_rate : ℝ := sorry

/-- The grass consumed by a number of cows over a period of days
    equals the initial grass plus the grass grown during that period -/
def grass_consumption (cows : ℝ) (days : ℝ) : Prop :=
  cows * daily_consumption_per_cow * days = initial_grass + daily_growth_rate * days

theorem pasture_feeding_duration :
  grass_consumption 20 40 ∧ grass_consumption 35 10 →
  grass_consumption 25 20 := by sorry

end NUMINAMATH_CALUDE_pasture_feeding_duration_l1495_149512


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1495_149581

theorem cistern_fill_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) : 
  empty_rate = 1 / 9 →
  combined_fill_time = 7 / 3 →
  1 / fill_time - empty_rate = 1 / combined_fill_time →
  fill_time = 63 / 34 := by
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1495_149581


namespace NUMINAMATH_CALUDE_detergent_amount_in_altered_solution_l1495_149594

/-- The ratio of bleach to detergent to water in a solution -/
structure SolutionRatio :=
  (bleach : ℚ)
  (detergent : ℚ)
  (water : ℚ)

/-- The amount of each component in a solution -/
structure SolutionAmount :=
  (bleach : ℚ)
  (detergent : ℚ)
  (water : ℚ)

def original_ratio : SolutionRatio :=
  { bleach := 2, detergent := 25, water := 100 }

def altered_ratio (r : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * r.bleach,
    detergent := r.detergent,
    water := 2 * r.water }

def water_amount : ℚ := 300

theorem detergent_amount_in_altered_solution :
  ∀ (r : SolutionRatio) (w : ℚ),
  r = original_ratio →
  w = water_amount →
  ∃ (a : SolutionAmount),
    a.water = w ∧
    a.detergent = 37.5 ∧
    a.bleach / a.detergent = (altered_ratio r).bleach / (altered_ratio r).detergent ∧
    a.detergent / a.water = (altered_ratio r).detergent / (altered_ratio r).water :=
by sorry

end NUMINAMATH_CALUDE_detergent_amount_in_altered_solution_l1495_149594


namespace NUMINAMATH_CALUDE_campers_rowing_count_l1495_149567

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 15

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 17

/-- The total number of campers who went rowing that day -/
def total_campers : ℕ := morning_campers + afternoon_campers

theorem campers_rowing_count : total_campers = 32 := by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_count_l1495_149567


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l1495_149515

/-- Proves that for a rectangular plot with given conditions, the length is 20 metres more than the breadth -/
theorem rectangular_plot_length_difference (length width : ℝ) : 
  length = 60 ∧ 
  (2 * length + 2 * width) * 26.5 = 5300 →
  length - width = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l1495_149515


namespace NUMINAMATH_CALUDE_max_value_difference_l1495_149541

noncomputable def f (x : ℝ) := x^3 - 3*x^2 - x + 1

theorem max_value_difference (x₀ m : ℝ) : 
  (∀ x, f x ≤ f x₀) →  -- f attains maximum at x₀
  m ≠ x₀ →             -- m is not equal to x₀
  f x₀ = f m →         -- f(x₀) = f(m)
  |m - x₀| = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_difference_l1495_149541


namespace NUMINAMATH_CALUDE_distance_to_point_l1495_149536

theorem distance_to_point : Real.sqrt ((8 - 0)^2 + (-15 - 0)^2) = 17 := by sorry

end NUMINAMATH_CALUDE_distance_to_point_l1495_149536


namespace NUMINAMATH_CALUDE_binomial_square_proof_l1495_149521

theorem binomial_square_proof :
  ∃ (r s : ℚ), (r * x + s)^2 = (81/16 : ℚ) * x^2 + 18 * x + 16 :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_square_proof_l1495_149521


namespace NUMINAMATH_CALUDE_smallest_perimeter_triangle_l1495_149590

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle formed by two rays from a vertex -/
structure Angle where
  vertex : Point
  ray1 : Point
  ray2 : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line passing through a point -/
structure Line where
  point : Point
  direction : ℝ

/-- Checks if a point is inside an angle -/
def isPointInsideAngle (p : Point) (a : Angle) : Prop := sorry

/-- Finds the larger inscribed circle passing through a point in an angle -/
def largerInscribedCircle (p : Point) (a : Angle) : Circle := sorry

/-- Checks if a line is tangent to a circle at a point -/
def isTangentLine (l : Line) (c : Circle) (p : Point) : Prop := sorry

/-- Calculates the perimeter of a triangle formed by a line intersecting an angle -/
def trianglePerimeter (l : Line) (a : Angle) : ℝ := sorry

/-- The main theorem -/
theorem smallest_perimeter_triangle 
  (M : Point) (KAL : Angle) 
  (h_inside : isPointInsideAngle M KAL) :
  let S := largerInscribedCircle M KAL
  let tangent_line := Line.mk M (sorry : ℝ)  -- Direction that makes it tangent
  ∀ (l : Line), 
    l.point = M → 
    isTangentLine tangent_line S M → 
    trianglePerimeter l KAL ≥ trianglePerimeter tangent_line KAL :=
by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_triangle_l1495_149590


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1495_149553

theorem simplify_trig_expression :
  1 / Real.sin (15 * π / 180) - 1 / Real.cos (15 * π / 180) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1495_149553


namespace NUMINAMATH_CALUDE_line_intersection_l1495_149550

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 7*x + 12

-- Define the linear function
def g (m b x : ℝ) : ℝ := m*x + b

-- Define the distance between two points on the same vertical line
def distance (k m b : ℝ) : ℝ := |f k - g m b k|

theorem line_intersection (m b : ℝ) : 
  (∃ k, distance k m b = 8) ∧ 
  g m b 2 = 7 ∧ 
  b ≠ 0 →
  (m = 1 ∧ b = 5) ∨ (m = 5 ∧ b = -3) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_l1495_149550


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l1495_149544

/-- Given two circles where one encloses the other, this theorem proves
    the radius of the smaller circle given specific conditions. -/
theorem smaller_circle_radius
  (R : ℝ) -- Radius of the larger circle
  (r : ℝ) -- Radius of the smaller circle
  (A₁ : ℝ) -- Area of the smaller circle
  (A₂ : ℝ) -- Area difference between the two circles
  (h1 : R = 5) -- The larger circle has a radius of 5 units
  (h2 : A₁ = π * r^2) -- Area formula for the smaller circle
  (h3 : A₂ = π * R^2 - A₁) -- Area difference
  (h4 : ∃ (d : ℝ), A₁ + d = A₂ ∧ A₂ + d = A₁ + A₂) -- Arithmetic progression condition
  : r = 5 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l1495_149544


namespace NUMINAMATH_CALUDE_books_to_pens_ratio_l1495_149522

def total_stationery : ℕ := 400
def num_books : ℕ := 280

theorem books_to_pens_ratio :
  let num_pens := total_stationery - num_books
  (num_books / (Nat.gcd num_books num_pens)) = 7 ∧
  (num_pens / (Nat.gcd num_books num_pens)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_to_pens_ratio_l1495_149522


namespace NUMINAMATH_CALUDE_divisor_pairing_l1495_149529

theorem divisor_pairing (n : ℕ+) (h : ¬ ∃ m : ℕ, n = m ^ 2) :
  ∃ f : {d : ℕ // d ∣ n} → {d : ℕ // d ∣ n},
    ∀ d : {d : ℕ // d ∣ n}, 
      (f (f d) = d) ∧ 
      ((d.val ∣ (f d).val) ∨ ((f d).val ∣ d.val)) :=
sorry

end NUMINAMATH_CALUDE_divisor_pairing_l1495_149529


namespace NUMINAMATH_CALUDE_initial_fund_calculation_l1495_149557

theorem initial_fund_calculation (initial_per_employee final_per_employee undistributed : ℕ) : 
  initial_per_employee = 50 →
  final_per_employee = 45 →
  undistributed = 95 →
  (initial_per_employee - final_per_employee) * (undistributed / (initial_per_employee - final_per_employee)) = 950 := by
  sorry

end NUMINAMATH_CALUDE_initial_fund_calculation_l1495_149557


namespace NUMINAMATH_CALUDE_max_coins_distribution_l1495_149596

theorem max_coins_distribution (n : ℕ) : 
  n < 150 ∧ 
  ∃ k : ℕ, n = 13 * k + 3 →
  n ≤ 146 :=
by sorry

end NUMINAMATH_CALUDE_max_coins_distribution_l1495_149596


namespace NUMINAMATH_CALUDE_triangle_area_l1495_149532

/-- Given a triangle with perimeter 60 cm and inradius 2.5 cm, its area is 75 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 60 → inradius = 2.5 → area = perimeter / 2 * inradius → area = 75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1495_149532


namespace NUMINAMATH_CALUDE_parallel_subset_parallel_perpendicular_planes_parallel_l1495_149573

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (lineparallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Theorem 1
theorem parallel_subset_parallel 
  (a : Line) (α β : Plane) :
  parallel α β → subset a α → lineparallel a β := by sorry

-- Theorem 2
theorem perpendicular_planes_parallel 
  (a : Line) (α β : Plane) :
  perpendicular a α → perpendicular a β → parallel α β := by sorry

end NUMINAMATH_CALUDE_parallel_subset_parallel_perpendicular_planes_parallel_l1495_149573


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1495_149564

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -1/2
  let r : ℚ := a₂ / a₁
  geometric_sequence a₁ r 7 = 1/15552 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1495_149564


namespace NUMINAMATH_CALUDE_expression_factorization_l1495_149543

theorem expression_factorization (x : ℝ) : 
  (16 * x^7 + 81 * x^4 - 9) - (4 * x^7 - 18 * x^4 + 3) = 3 * (4 * x^7 + 33 * x^4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1495_149543


namespace NUMINAMATH_CALUDE_fruit_punch_ratio_l1495_149548

theorem fruit_punch_ratio (orange_punch apple_juice cherry_punch total_punch : ℝ) : 
  orange_punch = 4.5 →
  apple_juice = cherry_punch - 1.5 →
  total_punch = orange_punch + cherry_punch + apple_juice →
  total_punch = 21 →
  cherry_punch / orange_punch = 2 := by
sorry

end NUMINAMATH_CALUDE_fruit_punch_ratio_l1495_149548


namespace NUMINAMATH_CALUDE_stratified_sampling_most_suitable_l1495_149539

-- Define the land types
inductive LandType
  | Mountainous
  | Hilly
  | Flat
  | LowLying

-- Define the village structure
structure Village where
  landAreas : LandType → ℕ
  totalArea : ℕ
  sampleSize : ℕ

-- Define the sampling methods
inductive SamplingMethod
  | Drawing
  | RandomNumberTable
  | Systematic
  | Stratified

-- Define the suitability of a sampling method
def isSuitable (v : Village) (m : SamplingMethod) : Prop :=
  m = SamplingMethod.Stratified

-- Theorem statement
theorem stratified_sampling_most_suitable (v : Village) 
  (h1 : v.landAreas LandType.Mountainous = 8000)
  (h2 : v.landAreas LandType.Hilly = 12000)
  (h3 : v.landAreas LandType.Flat = 24000)
  (h4 : v.landAreas LandType.LowLying = 4000)
  (h5 : v.totalArea = 48000)
  (h6 : v.sampleSize = 480) :
  isSuitable v SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_suitable_l1495_149539


namespace NUMINAMATH_CALUDE_problem_solution_l1495_149595

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem problem_solution (a : ℝ) (m : ℝ) : 
  (∀ x : ℝ, f a x ≤ 3 ↔ x ∈ Set.Icc (-6) 0) →
  (a = -3 ∧ 
   (∀ x : ℝ, f a x + f a (x + 5) ≥ 2 * m) → m ≤ 5/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1495_149595


namespace NUMINAMATH_CALUDE_number_problem_l1495_149540

theorem number_problem : ∃ x : ℝ, (x / 6) * 12 = 13 ∧ x = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1495_149540


namespace NUMINAMATH_CALUDE_profit_calculation_l1495_149506

theorem profit_calculation :
  let selling_price : ℝ := 84
  let profit_percentage : ℝ := 0.4
  let loss_percentage : ℝ := 0.2
  let cost_price_profit_item : ℝ := selling_price / (1 + profit_percentage)
  let cost_price_loss_item : ℝ := selling_price / (1 - loss_percentage)
  let total_cost : ℝ := cost_price_profit_item + cost_price_loss_item
  let total_revenue : ℝ := 2 * selling_price
  total_revenue - total_cost = 3 := by
sorry

end NUMINAMATH_CALUDE_profit_calculation_l1495_149506


namespace NUMINAMATH_CALUDE_sum_a_d_equals_two_l1495_149508

theorem sum_a_d_equals_two 
  (a b c d : ℤ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_two_l1495_149508


namespace NUMINAMATH_CALUDE_parallel_segments_length_l1495_149500

/-- Given a quadrilateral ABYZ where AB is parallel to YZ, this theorem proves
    that if AZ = 54, BQ = 18, and QY = 36, then QZ = 36. -/
theorem parallel_segments_length (A B Y Z Q : ℝ × ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ B - A = k • (Z - Y)) →  -- AB parallel to YZ
  dist A Z = 54 →
  dist B Q = 18 →
  dist Q Y = 36 →
  dist Q Z = 36 := by
sorry


end NUMINAMATH_CALUDE_parallel_segments_length_l1495_149500


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l1495_149576

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, 
    f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * f y + 4 * y^2 :=
by
  -- The unique function is f(x) = x^2
  use fun x => x^2
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l1495_149576


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1495_149558

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x - y + b₁ = 0 ↔ m₂ * x - y + b₂ = 0) ↔ m₁ = m₂

/-- The value of a for which ax-2y+2=0 is parallel to x+(a-3)y+1=0 -/
theorem parallel_lines_a_value :
  ∃ a : ℝ, (∀ x y : ℝ, a * x - 2 * y + 2 = 0 ↔ x + (a - 3) * y + 1 = 0) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_a_value_l1495_149558


namespace NUMINAMATH_CALUDE_area_ratio_ABJ_ADE_l1495_149547

/-- Represents a regular octagon with vertices ABCDEFGH -/
structure RegularOctagon where
  vertices : Fin 8 → Point

/-- Represents a triangle within the regular octagon -/
structure OctagonTriangle where
  vertices : Fin 3 → Point

/-- The area of a triangle -/
def area (t : OctagonTriangle) : ℝ := sorry

/-- The regular octagon ABCDEFGH -/
def octagon : RegularOctagon := sorry

/-- Triangle ABJ formed by two smaller equilateral triangles -/
def triangle_ABJ : OctagonTriangle := sorry

/-- Triangle ADE formed by connecting every third vertex of the octagon -/
def triangle_ADE : OctagonTriangle := sorry

theorem area_ratio_ABJ_ADE :
  area triangle_ABJ / area triangle_ADE = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_area_ratio_ABJ_ADE_l1495_149547


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1495_149507

theorem tangent_line_to_parabola (x y : ℝ) :
  (y = x^2) →  -- The curve equation
  (∃ (m b : ℝ), (y = m*x + b) ∧  -- The tangent line equation
                (-1 = m*1 + b) ∧  -- The line passes through (1, -1)
                (∃ (a : ℝ), y = (2*a)*x - a^2 - a)) →  -- Tangent line touches the curve
  ((y = (2 + 2*Real.sqrt 2)*x - (3 + 2*Real.sqrt 2)) ∨
   (y = (2 - 2*Real.sqrt 2)*x - (3 - 2*Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1495_149507


namespace NUMINAMATH_CALUDE_translated_linear_function_range_l1495_149501

theorem translated_linear_function_range (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x + 2
  f x > 0 → x > -2 := by
sorry

end NUMINAMATH_CALUDE_translated_linear_function_range_l1495_149501


namespace NUMINAMATH_CALUDE_bob_corn_harvest_l1495_149503

/-- Calculates the number of bushels of corn harvested given the number of rows,
    corn stalks per row, and corn stalks per bushel. -/
def corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) : ℕ :=
  (rows * stalks_per_row) / stalks_per_bushel

/-- Proves that Bob will harvest 50 bushels of corn given the specified conditions. -/
theorem bob_corn_harvest :
  corn_harvest 5 80 8 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bob_corn_harvest_l1495_149503


namespace NUMINAMATH_CALUDE_heroes_on_back_l1495_149580

/-- The number of heroes Will drew on the front of the paper -/
def heroes_on_front : ℕ := 2

/-- The total number of heroes Will drew -/
def total_heroes : ℕ := 9

/-- Theorem: The number of heroes Will drew on the back of the paper is 7 -/
theorem heroes_on_back : total_heroes - heroes_on_front = 7 := by
  sorry

end NUMINAMATH_CALUDE_heroes_on_back_l1495_149580


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l1495_149546

def f (x : ℝ) := x^3 - 12*x

theorem max_value_of_f_on_interval :
  ∃ (M : ℝ), M = 16 ∧ ∀ x ∈ Set.Icc (-3) 3, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l1495_149546


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_12_l1495_149589

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_12 :
  ∀ n : ℕ, is_four_digit n → (sum_of_first_n n % 12 = 0) → n ≥ 1001 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_12_l1495_149589


namespace NUMINAMATH_CALUDE_area_of_inner_rectangle_l1495_149555

theorem area_of_inner_rectangle (s : ℝ) (h : s > 0) : 
  let larger_square_area := s^2
  let half_larger_square_area := larger_square_area / 2
  let inner_rectangle_side := s / 2
  let inner_rectangle_area := inner_rectangle_side^2
  half_larger_square_area = 80 → inner_rectangle_area = 40 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inner_rectangle_l1495_149555


namespace NUMINAMATH_CALUDE_puzzle_solution_l1495_149568

def puzzle_problem (total_pieces : ℕ) (num_sons : ℕ) (reyn_pieces : ℕ) : ℕ :=
  let pieces_per_son := total_pieces / num_sons
  let rhys_pieces := 2 * reyn_pieces
  let rory_pieces := 3 * reyn_pieces
  let placed_pieces := reyn_pieces + rhys_pieces + rory_pieces
  total_pieces - placed_pieces

theorem puzzle_solution :
  puzzle_problem 300 3 25 = 150 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1495_149568


namespace NUMINAMATH_CALUDE_point_not_in_region_l1495_149578

def plane_region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_region : ¬ plane_region 2 0 := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_region_l1495_149578


namespace NUMINAMATH_CALUDE_problem_statement_l1495_149585

theorem problem_statement (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a^2 + b^2 = a*b + 1) (hcd : c*d > 1) :
  (a + b ≤ 2) ∧ (Real.sqrt (a*c) + Real.sqrt (b*d) < c + d) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1495_149585


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l1495_149520

theorem complex_equation_solutions :
  ∃! (s : Finset ℂ), s.card = 4 ∧
  (∀ c ∈ s, ∃ u v w : ℂ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    ∀ z : ℂ, (z - u) * (z - v) * (z - w) = (z - c*u) * (z - c*v) * (z - c*w)) ∧
  (∀ c : ℂ, c ∉ s →
    ¬∃ u v w : ℂ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    ∀ z : ℂ, (z - u) * (z - v) * (z - w) = (z - c*u) * (z - c*v) * (z - c*w)) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l1495_149520


namespace NUMINAMATH_CALUDE_parabola_equation_l1495_149549

/-- A parabola with vertex at the origin, coordinate axes as axes of symmetry, 
    and passing through (-2, 3) has equation x^2 = (4/3)y or y^2 = -(9/2)x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 3)) →  -- vertex at origin and passes through (-2, 3)
  (∀ x y, p (x, y) ↔ p (-x, y)) →  -- symmetry about y-axis
  (∀ x y, p (x, y) ↔ p (x, -y)) →  -- symmetry about x-axis
  (∃ a b : ℝ, (∀ x y, p (x, y) ↔ x^2 = a*y) ∨ (∀ x y, p (x, y) ↔ y^2 = b*x)) →
  (∀ x y, p (x, y) ↔ x^2 = (4/3)*y ∨ y^2 = -(9/2)*x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1495_149549


namespace NUMINAMATH_CALUDE_barber_total_loss_l1495_149571

/-- Represents the barber's financial transactions and losses --/
def barber_loss : ℕ → Prop :=
  fun loss =>
    ∃ (haircut_cost change_given flower_shop_exchange bakery_exchange counterfeit_50 counterfeit_10 replacement_50 replacement_10 : ℕ),
      haircut_cost = 25 ∧
      change_given = 25 ∧
      flower_shop_exchange = 50 ∧
      bakery_exchange = 10 ∧
      counterfeit_50 = 50 ∧
      counterfeit_10 = 10 ∧
      replacement_50 = 50 ∧
      replacement_10 = 10 ∧
      loss = haircut_cost + change_given + counterfeit_50 + counterfeit_10 + replacement_50 + replacement_10 - flower_shop_exchange

theorem barber_total_loss :
  barber_loss 120 :=
sorry

end NUMINAMATH_CALUDE_barber_total_loss_l1495_149571


namespace NUMINAMATH_CALUDE_smallest_integer_y_l1495_149597

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y < 20) ∧ (∀ z : ℤ, z < y → ¬(7 - 3 * z < 20)) → y = -4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l1495_149597


namespace NUMINAMATH_CALUDE_no_common_terms_except_one_l1495_149559

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem no_common_terms_except_one : ∀ n : ℕ, x n = y n → x n = 1 := by sorry

end NUMINAMATH_CALUDE_no_common_terms_except_one_l1495_149559


namespace NUMINAMATH_CALUDE_original_people_count_l1495_149563

theorem original_people_count (x : ℚ) : 
  (2 * x / 3 + 6 - x / 6 = 15) → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_original_people_count_l1495_149563


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1495_149511

/-- Given a line passing through points (1,3) and (3,11), 
    prove that the sum of its slope and y-intercept equals 3. -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
  (3 : ℝ) = m * (1 : ℝ) + b → 
  (11 : ℝ) = m * (3 : ℝ) + b → 
  m + b = 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1495_149511


namespace NUMINAMATH_CALUDE_bleacher_exercise_calories_l1495_149504

/-- Given the number of round trips, stairs one way, and total calories burned,
    calculate the number of calories burned per stair. -/
def calories_per_stair (round_trips : ℕ) (stairs_one_way : ℕ) (total_calories : ℕ) : ℚ :=
  total_calories / (2 * round_trips * stairs_one_way)

/-- Theorem stating that under the given conditions, each stair burns 2 calories. -/
theorem bleacher_exercise_calories :
  calories_per_stair 40 32 5120 = 2 := by
  sorry

end NUMINAMATH_CALUDE_bleacher_exercise_calories_l1495_149504


namespace NUMINAMATH_CALUDE_servant_worked_months_l1495_149586

def yearly_salary : ℚ := 90
def turban_value : ℚ := 50
def received_cash : ℚ := 55

def total_yearly_salary : ℚ := yearly_salary + turban_value
def monthly_salary : ℚ := total_yearly_salary / 12
def total_received : ℚ := received_cash + turban_value

theorem servant_worked_months : 
  ∃ (months : ℚ), months * monthly_salary = total_received ∧ months = 9 := by
  sorry

end NUMINAMATH_CALUDE_servant_worked_months_l1495_149586


namespace NUMINAMATH_CALUDE_remainder_8347_div_9_l1495_149528

theorem remainder_8347_div_9 : 8347 % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8347_div_9_l1495_149528


namespace NUMINAMATH_CALUDE_unique_a_with_integer_solutions_l1495_149582

theorem unique_a_with_integer_solutions : 
  ∃! a : ℕ+, (a : ℝ) ≤ 100 ∧ 
  ∃ x y : ℤ, x ≠ y ∧ 
  (x : ℝ)^2 + (2 * (a : ℝ) - 3) * (x : ℝ) + ((a : ℝ) - 1)^2 = 0 ∧
  (y : ℝ)^2 + (2 * (a : ℝ) - 3) * (y : ℝ) + ((a : ℝ) - 1)^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_a_with_integer_solutions_l1495_149582


namespace NUMINAMATH_CALUDE_man_downstream_speed_l1495_149591

/-- Calculates the downstream speed of a man given his upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem: Given a man's upstream speed of 26 km/h and still water speed of 28 km/h, 
    his downstream speed is 30 km/h -/
theorem man_downstream_speed :
  downstream_speed 26 28 = 30 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l1495_149591


namespace NUMINAMATH_CALUDE_online_price_is_6_l1495_149518

/-- The price of an item online -/
def online_price : ℝ := 6

/-- The price of an item in the regular store -/
def regular_price : ℝ := online_price + 2

/-- The total amount spent in the regular store -/
def regular_total : ℝ := 96

/-- The total amount spent online -/
def online_total : ℝ := 90

/-- The number of additional items bought online compared to the regular store -/
def additional_items : ℕ := 3

theorem online_price_is_6 :
  (online_total / online_price) = (regular_total / regular_price) + additional_items ∧
  online_price = 6 := by sorry

end NUMINAMATH_CALUDE_online_price_is_6_l1495_149518


namespace NUMINAMATH_CALUDE_notebooks_distribution_l1495_149599

/-- 
Given a class where:
- The total number of notebooks distributed is 512
- Each child initially receives a number of notebooks equal to 1/8 of the total number of children
Prove that if the number of children is halved, each child would receive 16 notebooks.
-/
theorem notebooks_distribution (C : ℕ) (h1 : C > 0) : 
  (C * (C / 8) = 512) → ((512 / (C / 2)) = 16) :=
by sorry

end NUMINAMATH_CALUDE_notebooks_distribution_l1495_149599


namespace NUMINAMATH_CALUDE_pharmacy_loss_l1495_149569

theorem pharmacy_loss (a b : ℝ) (h : a < b) : 
  100 * ((a + b) / 2) - (41 * a + 59 * b) < 0 := by
  sorry

#check pharmacy_loss

end NUMINAMATH_CALUDE_pharmacy_loss_l1495_149569


namespace NUMINAMATH_CALUDE_water_depth_is_60_feet_l1495_149562

def ron_height : ℝ := 12

def water_depth : ℝ := 5 * ron_height

theorem water_depth_is_60_feet : water_depth = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_is_60_feet_l1495_149562


namespace NUMINAMATH_CALUDE_solution_relationship_l1495_149513

theorem solution_relationship (x y : ℝ) : 
  2 * x + y = 7 → x - y = 5 → x + 2 * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_relationship_l1495_149513


namespace NUMINAMATH_CALUDE_probability_is_seventy_percent_l1495_149587

/-- Represents a frequency interval with its lower bound, upper bound, and frequency count -/
structure FrequencyInterval where
  lower : ℝ
  upper : ℝ
  frequency : ℕ

/-- The sample data -/
def sample : List FrequencyInterval := [
  ⟨10, 20, 2⟩,
  ⟨20, 30, 3⟩,
  ⟨30, 40, 4⟩,
  ⟨40, 50, 5⟩,
  ⟨50, 60, 4⟩,
  ⟨60, 70, 2⟩
]

/-- The total sample size -/
def sampleSize : ℕ := 20

/-- The upper bound of the interval in question -/
def intervalUpperBound : ℝ := 50

/-- Calculates the probability of the sample data falling within (-∞, intervalUpperBound) -/
def probabilityWithinInterval (sample : List FrequencyInterval) (sampleSize : ℕ) (intervalUpperBound : ℝ) : ℚ :=
  sorry

/-- Theorem stating that the probability of the sample data falling within (-∞, 50) is 70% -/
theorem probability_is_seventy_percent :
  probabilityWithinInterval sample sampleSize intervalUpperBound = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_seventy_percent_l1495_149587


namespace NUMINAMATH_CALUDE_original_bottle_caps_l1495_149584

theorem original_bottle_caps (removed : ℕ) (left : ℕ) (original : ℕ) : 
  removed = 47 → left = 40 → original = removed + left → original = 87 := by
  sorry

end NUMINAMATH_CALUDE_original_bottle_caps_l1495_149584


namespace NUMINAMATH_CALUDE_chocolates_distribution_l1495_149524

theorem chocolates_distribution (total_children boys girls : ℕ) 
  (boys_chocolates girls_chocolates : ℕ) :
  total_children = 120 →
  boys = 60 →
  girls = 60 →
  boys + girls = total_children →
  boys_chocolates = 2 →
  girls_chocolates = 3 →
  boys * boys_chocolates + girls * girls_chocolates = 300 :=
by sorry

end NUMINAMATH_CALUDE_chocolates_distribution_l1495_149524


namespace NUMINAMATH_CALUDE_monomial_existence_l1495_149583

/-- A monomial in variables a and b -/
structure Monomial where
  coeff : ℤ
  a_power : ℕ
  b_power : ℕ

/-- Multiplication of monomials -/
def mul_monomial (x y : Monomial) : Monomial :=
  { coeff := x.coeff * y.coeff,
    a_power := x.a_power + y.a_power,
    b_power := x.b_power + y.b_power }

/-- Addition of monomials -/
def add_monomial (x y : Monomial) : Option Monomial :=
  if x.a_power = y.a_power ∧ x.b_power = y.b_power then
    some { coeff := x.coeff + y.coeff,
           a_power := x.a_power,
           b_power := x.b_power }
  else
    none

theorem monomial_existence : ∃ (x y : Monomial),
  (mul_monomial x y = { coeff := -12, a_power := 4, b_power := 2 }) ∧
  (∃ (z : Monomial), add_monomial x y = some z ∧ z.coeff = 1) :=
sorry

end NUMINAMATH_CALUDE_monomial_existence_l1495_149583


namespace NUMINAMATH_CALUDE_log_product_equals_one_l1495_149527

theorem log_product_equals_one : 
  Real.log 3 / Real.log 2 * (Real.log 4 / Real.log 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_one_l1495_149527


namespace NUMINAMATH_CALUDE_ngon_triangle_partition_l1495_149593

/-- A function that checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A structure representing an n-gon -/
structure Ngon (n : ℕ) :=
  (vertices : Fin n → ℝ × ℝ)
  (is_convex : sorry)  -- Additional property to ensure the n-gon is convex

/-- 
Given an n-gon and three vertex indices, return the lengths of the three parts 
of the boundary divided by these vertices
-/
def boundary_parts (n : ℕ) (poly : Ngon n) (i j k : Fin n) : ℝ × ℝ × ℝ := sorry

/-- The main theorem -/
theorem ngon_triangle_partition (n : ℕ) (h : n ≥ 3 ∧ n ≠ 4) :
  ∀ (poly : Ngon n), ∃ (i j k : Fin n),
    let (a, b, c) := boundary_parts n poly i j k
    can_form_triangle a b c :=
  sorry

end NUMINAMATH_CALUDE_ngon_triangle_partition_l1495_149593


namespace NUMINAMATH_CALUDE_vision_survey_is_sampling_l1495_149525

/-- Represents a survey method -/
inductive SurveyMethod
| Sampling
| Census
| Other

/-- Represents a school with a given population of eighth-grade students -/
structure School where
  population : ℕ

/-- Represents a vision survey conducted in a school -/
structure VisionSurvey where
  school : School
  sample_size : ℕ
  selection_method : String

/-- Determines the survey method based on the vision survey parameters -/
def determine_survey_method (survey : VisionSurvey) : SurveyMethod :=
  if survey.sample_size < survey.school.population ∧ survey.selection_method = "Random" then
    SurveyMethod.Sampling
  else if survey.sample_size = survey.school.population then
    SurveyMethod.Census
  else
    SurveyMethod.Other

/-- Theorem stating that the given vision survey uses a sampling survey method -/
theorem vision_survey_is_sampling (school : School) (survey : VisionSurvey) :
  school.population = 400 →
  survey.school = school →
  survey.sample_size = 80 →
  survey.selection_method = "Random" →
  determine_survey_method survey = SurveyMethod.Sampling :=
by
  sorry

#check vision_survey_is_sampling

end NUMINAMATH_CALUDE_vision_survey_is_sampling_l1495_149525


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1495_149537

/-- Represents a sampling method -/
inductive SamplingMethod
  | Systematic
  | SimpleRandom
  | Stratified

/-- Represents a sampling scenario -/
structure SamplingScenario where
  method : SamplingMethod
  description : String

/-- The milk production line sampling scenario -/
def milkProductionScenario : SamplingScenario :=
  { method := SamplingMethod.Systematic,
    description := "Sampling a bag every 30 minutes on a milk production line" }

/-- The math enthusiasts sampling scenario -/
def mathEnthusiastsScenario : SamplingScenario :=
  { method := SamplingMethod.SimpleRandom,
    description := "Selecting 3 individuals from 30 math enthusiasts in a middle school" }

/-- Theorem stating that the sampling methods are correctly identified -/
theorem correct_sampling_methods :
  (milkProductionScenario.method = SamplingMethod.Systematic) ∧
  (mathEnthusiastsScenario.method = SamplingMethod.SimpleRandom) :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1495_149537


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1495_149519

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 40)
  (area2 : w * h = 10)
  (area3 : l * h = 8) :
  l * w * h = 40 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1495_149519


namespace NUMINAMATH_CALUDE_T_is_integer_at_smallest_n_l1495_149523

/-- Sum of reciprocals of non-zero digits from 1 to 5^n -/
def T (n : ℕ) : ℚ :=
  sorry

/-- The smallest positive integer n for which T n is an integer -/
def smallest_n : ℕ := 504

theorem T_is_integer_at_smallest_n :
  (T smallest_n).isInt ∧ ∀ m : ℕ, m > 0 ∧ m < smallest_n → ¬(T m).isInt :=
sorry

end NUMINAMATH_CALUDE_T_is_integer_at_smallest_n_l1495_149523


namespace NUMINAMATH_CALUDE_sequence_not_in_interval_l1495_149572

/-- Given an infinite sequence of real numbers {aₙ} where aₙ₊₁ = √(aₙ² + aₙ - 1) for all n ≥ 1,
    prove that a₁ ∉ (-2, 1). -/
theorem sequence_not_in_interval (a : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = Real.sqrt ((a n)^2 + a n - 1)) : 
    a 1 ∉ Set.Ioo (-2 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_not_in_interval_l1495_149572


namespace NUMINAMATH_CALUDE_intersection_line_slope_l1495_149574

/-- The slope of the line passing through the intersection points of two circles -/
theorem intersection_line_slope (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 4*y - 8 = 0) ∧ 
  (x^2 + y^2 - 10*x + 18*y + 40 = 0) →
  (∃ m : ℚ, m = 2/7 ∧ 
   ∀ (x₁ y₁ x₂ y₂ : ℝ), 
   (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 8 = 0) ∧ 
   (x₁^2 + y₁^2 - 10*x₁ + 18*y₁ + 40 = 0) ∧
   (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 8 = 0) ∧ 
   (x₂^2 + y₂^2 - 10*x₂ + 18*y₂ + 40 = 0) ∧
   x₁ ≠ x₂ →
   m = (y₂ - y₁) / (x₂ - x₁)) :=
sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l1495_149574


namespace NUMINAMATH_CALUDE_cost_difference_l1495_149505

/-- Represents the pricing policy of the store -/
def pencil_cost (quantity : ℕ) : ℚ :=
  if quantity < 40 then 4 else (7/2)

/-- Calculate the total cost for a given quantity of pencils -/
def total_cost (quantity : ℕ) : ℚ :=
  (pencil_cost quantity) * quantity

/-- The number of pencils Joy bought -/
def joy_pencils : ℕ := 30

/-- The number of pencils Colleen bought -/
def colleen_pencils : ℕ := 50

/-- Theorem stating the difference in cost between Colleen's and Joy's purchases -/
theorem cost_difference : 
  total_cost colleen_pencils - total_cost joy_pencils = 55 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l1495_149505


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l1495_149531

/-- Given two concentric circles with areas A1 and A1 + A2, where the larger circle
    has radius 5 and A1, A2, A1 + A2 form an arithmetic progression,
    prove that the radius of the smaller circle is 5√2/2 -/
theorem smaller_circle_radius
  (A1 A2 : ℝ)
  (h1 : A1 > 0)
  (h2 : A2 > 0)
  (h3 : (A1 + A2) = π * 5^2)
  (h4 : A2 = (A1 + (A1 + A2)) / 2)
  : ∃ (r : ℝ), r > 0 ∧ A1 = π * r^2 ∧ r = 5 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l1495_149531


namespace NUMINAMATH_CALUDE_monitor_height_l1495_149551

theorem monitor_height 
  (width : ℝ) 
  (pixel_density : ℝ) 
  (total_pixels : ℝ) 
  (h1 : width = 21)
  (h2 : pixel_density = 100)
  (h3 : total_pixels = 2520000) :
  (total_pixels / (width * pixel_density)) / pixel_density = 12 := by
  sorry

end NUMINAMATH_CALUDE_monitor_height_l1495_149551


namespace NUMINAMATH_CALUDE_soda_quarters_needed_l1495_149598

theorem soda_quarters_needed (total_amount : ℚ) (quarters_per_soda : ℕ) : 
  total_amount = 213.75 ∧ quarters_per_soda = 7 →
  (⌊(total_amount / 0.25) / quarters_per_soda⌋ + 1) * quarters_per_soda - 
  (total_amount / 0.25).floor = 6 := by
  sorry

end NUMINAMATH_CALUDE_soda_quarters_needed_l1495_149598


namespace NUMINAMATH_CALUDE_sum_f_equals_1326_l1495_149535

/-- The number of integer points on the line segment from (0,0) to (n, n+3), excluding endpoints -/
def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

/-- The sum of f(n) for n from 1 to 1990 -/
def sum_f : ℕ := (Finset.range 1990).sum f

theorem sum_f_equals_1326 : sum_f = 1326 := by sorry

end NUMINAMATH_CALUDE_sum_f_equals_1326_l1495_149535


namespace NUMINAMATH_CALUDE_integer_solution_x4_y4_eq_3x3y_l1495_149509

theorem integer_solution_x4_y4_eq_3x3y :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y ↔ x = 0 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_solution_x4_y4_eq_3x3y_l1495_149509


namespace NUMINAMATH_CALUDE_min_value_z_l1495_149533

theorem min_value_z (x y : ℝ) : 3 * x^2 + 5 * y^2 + 12 * x - 10 * y + 40 ≥ 23 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l1495_149533


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l1495_149542

theorem factorization_of_quadratic (a : ℝ) : a^2 + 3*a = a*(a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l1495_149542


namespace NUMINAMATH_CALUDE_living_room_set_cost_l1495_149561

theorem living_room_set_cost (coach_cost sectional_cost paid_amount : ℚ)
  (h1 : coach_cost = 2500)
  (h2 : sectional_cost = 3500)
  (h3 : paid_amount = 7200)
  (discount_rate : ℚ)
  (h4 : discount_rate = 0.1) :
  ∃ (additional_cost : ℚ),
    paid_amount = (1 - discount_rate) * (coach_cost + sectional_cost + additional_cost) ∧
    additional_cost = 2000 := by
sorry

end NUMINAMATH_CALUDE_living_room_set_cost_l1495_149561


namespace NUMINAMATH_CALUDE_candy_boxes_problem_l1495_149575

theorem candy_boxes_problem (a b c : ℕ) : 
  a = b + c - 8 → 
  b = a + c - 12 → 
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_boxes_problem_l1495_149575


namespace NUMINAMATH_CALUDE_smallest_fixed_point_of_R_l1495_149526

/-- The transformation R that reflects a line first on l₁: y = √3x and then on l₂: y = -√3x -/
def R (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The n-th iteration of R -/
def R_iter (n : ℕ) (l : ℝ → ℝ) : ℝ → ℝ :=
  match n with
  | 0 => l
  | n + 1 => R (R_iter n l)

/-- Any line can be represented as y = kx for some k -/
def line (k : ℝ) : ℝ → ℝ := λ x => k * x

theorem smallest_fixed_point_of_R :
  ∀ k : ℝ, ∃ m : ℕ, m > 0 ∧ R_iter m (line k) = line k ∧
  ∀ n : ℕ, 0 < n → n < m → R_iter n (line k) ≠ line k :=
by sorry

end NUMINAMATH_CALUDE_smallest_fixed_point_of_R_l1495_149526


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1495_149565

theorem complex_magnitude_product : 
  Complex.abs ((Real.sqrt 8 - Complex.I * 2) * (Real.sqrt 3 * 2 + Complex.I * 6)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1495_149565


namespace NUMINAMATH_CALUDE_power_of_three_expression_l1495_149545

theorem power_of_three_expression : 3^(1+2+3) - (3^1 + 3^2 + 3^4) = 636 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_expression_l1495_149545


namespace NUMINAMATH_CALUDE_problem_statement_l1495_149514

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a / 2 * x^2

def l (k : ℤ) (x : ℝ) : ℝ := (k - 2 : ℝ) * x - k + 1

theorem problem_statement :
  (∀ a : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ f a x₀ > 0) →
    a < 2 / Real.exp 1) ∧
  (∃ k : ℤ, k = 4 ∧
    ∀ k' : ℤ, (∀ x : ℝ, x > 1 → f 0 x > l k' x) → k' ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1495_149514


namespace NUMINAMATH_CALUDE_smallest_solution_congruences_l1495_149579

theorem smallest_solution_congruences :
  ∃ x : ℕ, x > 0 ∧
    x % 2 = 1 ∧
    x % 3 = 2 ∧
    x % 4 = 3 ∧
    x % 5 = 4 ∧
    (∀ y : ℕ, y > 0 →
      y % 2 = 1 →
      y % 3 = 2 →
      y % 4 = 3 →
      y % 5 = 4 →
      y ≥ x) ∧
  x = 59 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruences_l1495_149579


namespace NUMINAMATH_CALUDE_min_value_expression_l1495_149592

theorem min_value_expression (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_one : x + y + z + w = 1) (x_eq_y : x = y) :
  ∀ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d = 1 ∧ a = b →
  (a + b + c) / (a * b * c * d) ≥ (x + y + z) / (x * y * z * w) ∧
  (x + y + z) / (x * y * z * w) ≥ 1024 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1495_149592


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l1495_149538

theorem smallest_solution_floor_equation :
  ∀ x : ℝ, (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 19 → x ≥ Real.sqrt 119 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l1495_149538


namespace NUMINAMATH_CALUDE_dean_books_count_l1495_149534

theorem dean_books_count (tony_books breanna_books total_different_books : ℕ)
  (tony_dean_shared all_shared : ℕ) :
  tony_books = 23 →
  breanna_books = 17 →
  tony_dean_shared = 3 →
  all_shared = 1 →
  total_different_books = 47 →
  ∃ dean_books : ℕ,
    dean_books = 16 ∧
    total_different_books =
      (tony_books - tony_dean_shared - all_shared) +
      (dean_books - tony_dean_shared - all_shared) +
      (breanna_books - all_shared) :=
by sorry

end NUMINAMATH_CALUDE_dean_books_count_l1495_149534


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l1495_149517

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4702 [ZMOD 8] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l1495_149517


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1495_149577

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h_prop : InverselyProportional x₁ y₁)
  (h_init : x₁ = 40 ∧ y₁ = 8)
  (h_final : y₂ = 10) :
  x₂ = 32 ∧ InverselyProportional x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1495_149577


namespace NUMINAMATH_CALUDE_ratio_sum_squares_implies_sum_l1495_149556

theorem ratio_sum_squares_implies_sum (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 3 * a →
  a^2 + b^2 + c^2 = 2016 →
  a + b + c = 72 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_implies_sum_l1495_149556


namespace NUMINAMATH_CALUDE_congruence_solution_l1495_149588

theorem congruence_solution (n : ℤ) : 19 * n ≡ 13 [ZMOD 47] → n ≡ 25 [ZMOD 47] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1495_149588


namespace NUMINAMATH_CALUDE_boat_upstream_downstream_ratio_l1495_149554

/-- Given a boat with speed in still water and a stream with its own speed,
    prove that the ratio of time taken to row upstream to downstream is 2:1 -/
theorem boat_upstream_downstream_ratio
  (boat_speed : ℝ) (stream_speed : ℝ)
  (h1 : boat_speed = 54)
  (h2 : stream_speed = 18) :
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

#check boat_upstream_downstream_ratio

end NUMINAMATH_CALUDE_boat_upstream_downstream_ratio_l1495_149554


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l1495_149510

theorem cubic_sum_over_product (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a + b + c + d = 0) : 
  (a^3 + b^3 + c^3 + d^3) / (a * b * c * d) = -3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l1495_149510


namespace NUMINAMATH_CALUDE_tiffany_bags_l1495_149560

/-- The number of bags found the next day -/
def bags_found (initial_bags total_bags : ℕ) : ℕ := total_bags - initial_bags

/-- Proof that Tiffany found 8 bags the next day -/
theorem tiffany_bags : bags_found 4 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_l1495_149560


namespace NUMINAMATH_CALUDE_total_gas_consumption_is_18_gallons_l1495_149516

/-- Represents the number of cuts for a lawn in a given month. -/
structure MonthlyCuts where
  regular : Nat  -- Number of cuts in regular months
  peak : Nat     -- Number of cuts in peak months

/-- Represents the gas consumption pattern for a lawn. -/
structure GasConsumption where
  gallons : Nat  -- Number of gallons consumed
  frequency : Nat  -- Frequency of consumption (every nth cut)

/-- Calculates the total number of cuts for a lawn over the season. -/
def totalCuts (cuts : MonthlyCuts) : Nat :=
  4 * cuts.regular + 4 * cuts.peak

/-- Calculates the gas consumed for a lawn over the season. -/
def gasConsumed (cuts : Nat) (consumption : GasConsumption) : Nat :=
  (cuts / consumption.frequency) * consumption.gallons

/-- Theorem stating that the total gas consumption is 18 gallons. -/
theorem total_gas_consumption_is_18_gallons 
  (large_lawn_cuts : MonthlyCuts)
  (small_lawn_cuts : MonthlyCuts)
  (large_lawn_gas : GasConsumption)
  (small_lawn_gas : GasConsumption)
  (h1 : large_lawn_cuts = { regular := 1, peak := 3 })
  (h2 : small_lawn_cuts = { regular := 2, peak := 2 })
  (h3 : large_lawn_gas = { gallons := 2, frequency := 3 })
  (h4 : small_lawn_gas = { gallons := 1, frequency := 2 })
  : gasConsumed (totalCuts large_lawn_cuts) large_lawn_gas + 
    gasConsumed (totalCuts small_lawn_cuts) small_lawn_gas = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_gas_consumption_is_18_gallons_l1495_149516


namespace NUMINAMATH_CALUDE_circle_area_difference_l1495_149566

/-- The difference in areas between a circle with radius 25 inches and a circle with diameter 15 inches is 568.75π square inches. -/
theorem circle_area_difference : 
  let r1 : ℝ := 25
  let d2 : ℝ := 15
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 568.75 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1495_149566


namespace NUMINAMATH_CALUDE_resulting_surface_area_l1495_149570

/-- Represents the dimensions of the large cube -/
def large_cube_dim : ℕ := 12

/-- Represents the dimensions of the small cubes -/
def small_cube_dim : ℕ := 3

/-- The number of small cubes in the original large cube -/
def total_small_cubes : ℕ := 64

/-- The number of small cubes removed -/
def removed_cubes : ℕ := 8

/-- The number of remaining small cubes after removal -/
def remaining_cubes : ℕ := total_small_cubes - removed_cubes

/-- The surface area of a single small cube before modification -/
def small_cube_surface : ℕ := 6 * small_cube_dim ^ 2

/-- The number of new surfaces exposed per small cube after modification -/
def new_surfaces_per_cube : ℕ := 12

/-- The number of edge-shared internal faces -/
def edge_shared_faces : ℕ := 12

/-- The area of each edge-shared face -/
def edge_shared_face_area : ℕ := small_cube_dim ^ 2

/-- Theorem stating the surface area of the resulting structure -/
theorem resulting_surface_area :
  (remaining_cubes * (small_cube_surface + new_surfaces_per_cube)) -
  (4 * 3 * edge_shared_faces * edge_shared_face_area) = 3408 := by
  sorry

end NUMINAMATH_CALUDE_resulting_surface_area_l1495_149570


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1495_149552

theorem fraction_evaluation : (15 : ℚ) / 45 - 2 / 9 + 1 / 4 * 8 / 3 = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1495_149552
