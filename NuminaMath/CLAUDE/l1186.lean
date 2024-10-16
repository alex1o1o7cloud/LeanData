import Mathlib

namespace NUMINAMATH_CALUDE_solution_x_percent_l1186_118632

/-- Represents a chemical solution with a certain percentage of chemical A -/
structure Solution where
  percentA : ℝ
  percentB : ℝ
  sum_to_one : percentA + percentB = 1

/-- Represents a mixture of two solutions -/
structure Mixture where
  solution1 : Solution
  solution2 : Solution
  ratio1 : ℝ
  ratio2 : ℝ
  sum_to_one : ratio1 + ratio2 = 1
  percentA : ℝ

/-- The main theorem to be proved -/
theorem solution_x_percent (solution2 : Solution) (mixture : Mixture) :
  solution2.percentA = 0.4 →
  mixture.percentA = 0.32 →
  mixture.ratio1 = 0.8 →
  mixture.ratio2 = 0.2 →
  mixture.solution2 = solution2 →
  mixture.solution1.percentA = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_percent_l1186_118632


namespace NUMINAMATH_CALUDE_johann_oranges_l1186_118629

theorem johann_oranges (initial : ℕ) (eaten : ℕ) (returned : ℕ) (final : ℕ) : 
  initial = 60 →
  returned = 5 →
  final = 30 →
  (initial - eaten) / 2 + returned = final →
  eaten = 10 := by
sorry

end NUMINAMATH_CALUDE_johann_oranges_l1186_118629


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1186_118664

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 1 → (1 / a < 1)) ∧ ¬((1 / a < 1) → (a > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1186_118664


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1186_118682

/-- The distance between the foci of an ellipse with semi-major axis 9 and semi-minor axis 3 -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 9) (hb : b = 3) :
  2 * Real.sqrt (a^2 - b^2) = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1186_118682


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l1186_118688

/-- The number of full books Robert can read in a given time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours : ℕ) : ℕ :=
  (pages_per_hour * hours) / pages_per_book

/-- Theorem: Robert can read 2 full 360-page books in 8 hours at 120 pages per hour -/
theorem robert_reading_capacity : books_read 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l1186_118688


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1186_118699

theorem regular_polygon_sides (n : ℕ) : 
  n > 2 → 
  (180 * (n - 2) : ℝ) / n = 160 → 
  n = 18 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1186_118699


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1186_118635

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 4 + a 7 = 2) →
  (a 5 * a 8 = -8) →
  (a 1 + a 10 = -7) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1186_118635


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l1186_118668

def a : Fin 3 → ℝ := ![3, 5, 4]
def b : Fin 3 → ℝ := ![5, 9, 7]
def c₁ : Fin 3 → ℝ := fun i => -2 * a i + b i
def c₂ : Fin 3 → ℝ := fun i => 3 * a i - 2 * b i

theorem vectors_not_collinear : ¬ ∃ (k : ℝ), c₁ = fun i => k * c₂ i := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_collinear_l1186_118668


namespace NUMINAMATH_CALUDE_envelope_weight_l1186_118679

/-- Given 800 envelopes with a total weight of 6.8 kg, prove that one envelope weighs 8.5 grams. -/
theorem envelope_weight (num_envelopes : ℕ) (total_weight_kg : ℝ) :
  num_envelopes = 800 →
  total_weight_kg = 6.8 →
  (total_weight_kg * 1000) / num_envelopes = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_envelope_weight_l1186_118679


namespace NUMINAMATH_CALUDE_product_congruence_l1186_118651

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def product_of_list (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

theorem product_congruence :
  let seq := arithmetic_sequence 3 5 21
  (product_of_list seq) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_congruence_l1186_118651


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l1186_118667

/-- The number of Popsicles Megan consumes in a given time period -/
def popsicles_consumed (minutes_per_popsicle : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / minutes_per_popsicle

theorem megan_popsicle_consumption :
  popsicles_consumed 18 (5 * 60 + 36) = 18 := by
  sorry

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l1186_118667


namespace NUMINAMATH_CALUDE_max_area_triangle_l1186_118610

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3)

theorem max_area_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi / 2 →
  0 < C ∧ C < Real.pi / 2 →
  A + B + C = Real.pi →
  a > 0 ∧ b > 0 ∧ c > 0 →
  g (A / 2) = 1 / 2 →
  a = 1 →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  (1 / 2 * b * c * Real.sin A) ≤ (2 + Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_l1186_118610


namespace NUMINAMATH_CALUDE_second_triangle_weight_l1186_118608

/-- Represents an equilateral triangle with given side length and weight -/
structure EquilateralTriangle where
  side_length : ℝ
  weight : ℝ

/-- Calculate the weight of a second equilateral triangle given the properties of a first triangle -/
def calculate_second_triangle_weight (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side_length = 2 ∧ 
  t1.weight = 20 ∧ 
  t2.side_length = 4 ∧ 
  t2.weight = 80

theorem second_triangle_weight (t1 t2 : EquilateralTriangle) : 
  calculate_second_triangle_weight t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_second_triangle_weight_l1186_118608


namespace NUMINAMATH_CALUDE_min_value_implies_a_solution_set_inequality_l1186_118693

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4|

-- Theorem for part 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f (2*x + a) + f (2*x - a) ≥ 4) ∧
  (∃ x, f (2*x + a) + f (2*x - a) = 4) →
  a = 2 ∨ a = -2 :=
sorry

-- Theorem for part 2
theorem solution_set_inequality :
  {x : ℝ | f x > 1 - (1/2)*x} = {x : ℝ | x > -2 ∨ x < -10} :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_solution_set_inequality_l1186_118693


namespace NUMINAMATH_CALUDE_max_perimeter_of_divided_isosceles_triangle_l1186_118603

/-- The maximum perimeter of a piece when an isosceles triangle is divided into four equal areas -/
theorem max_perimeter_of_divided_isosceles_triangle :
  let base : ℝ := 12
  let height : ℝ := 15
  let segment_length : ℝ := base / 4
  let perimeter (k : ℝ) : ℝ := segment_length + Real.sqrt (height^2 + k^2) + Real.sqrt (height^2 + (k + 1)^2)
  let max_perimeter : ℝ := perimeter 2
  max_perimeter = 3 + Real.sqrt 229 + Real.sqrt 234 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_of_divided_isosceles_triangle_l1186_118603


namespace NUMINAMATH_CALUDE_tree_planting_variance_l1186_118672

def tree_planting_data : List (Nat × Nat) := [(5, 3), (6, 4), (7, 3)]

def total_groups : Nat := tree_planting_data.map (·.2) |>.sum

theorem tree_planting_variance (h : total_groups = 10) :
  let mean := (tree_planting_data.map (fun (x, y) => x * y) |>.sum) / total_groups
  let variance := (1 : ℝ) / total_groups *
    (tree_planting_data.map (fun (x, y) => y * ((x : ℝ) - mean)^2) |>.sum)
  variance = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_variance_l1186_118672


namespace NUMINAMATH_CALUDE_exam_score_problem_l1186_118695

theorem exam_score_problem (correct_score : ℕ) (wrong_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) :
  correct_score = 4 →
  wrong_score = 1 →
  total_score = 160 →
  correct_answers = 44 →
  ∃ (total_questions : ℕ),
    total_questions = correct_answers + (total_score - correct_score * correct_answers) / wrong_score ∧
    total_questions = 60 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l1186_118695


namespace NUMINAMATH_CALUDE_problem_1_l1186_118605

theorem problem_1 : (-20) + 3 - (-5) - 7 = -19 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1186_118605


namespace NUMINAMATH_CALUDE_second_group_frequency_l1186_118666

theorem second_group_frequency (total : ℕ) (group1 group2 group3 group4 group5 : ℕ) 
  (h1 : total = 50)
  (h2 : group1 = 2)
  (h3 : group3 = 8)
  (h4 : group4 = 10)
  (h5 : group5 = 20)
  (h6 : total = group1 + group2 + group3 + group4 + group5) :
  group2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_group_frequency_l1186_118666


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_real_solutions_l1186_118650

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a^2| + |x + 2*a - 5|

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | |x + 1| + |x - 3| < 5} = Set.Ioo (-3/2) (7/2) := by sorry

-- Theorem for part (2)
theorem range_of_a_for_real_solutions :
  {a : ℝ | ∃ x, f x a < 5} = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_real_solutions_l1186_118650


namespace NUMINAMATH_CALUDE_locus_of_circumscribed_rectangles_centers_l1186_118600

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  center : Point2D
  width : ℝ
  height : ℝ

/-- Represents a curvilinear triangle formed by semicircles -/
structure CurvilinearTriangle where
  midpoints : Triangle  -- Represents the triangle formed by midpoints of the original triangle

/-- Checks if a triangle is acute-angled -/
def isAcuteTriangle (t : Triangle) : Prop :=
  sorry  -- Definition of acute triangle

/-- Checks if a rectangle is circumscribed around a triangle -/
def isCircumscribed (r : Rectangle) (t : Triangle) : Prop :=
  sorry  -- Definition of circumscribed rectangle

/-- Computes the midpoints of a triangle -/
def midpoints (t : Triangle) : Triangle :=
  sorry  -- Computation of midpoints

/-- Checks if a point is on the locus (curvilinear triangle) -/
def isOnLocus (p : Point2D) (ct : CurvilinearTriangle) : Prop :=
  sorry  -- Definition of being on the locus

/-- The main theorem -/
theorem locus_of_circumscribed_rectangles_centers 
  (t : Triangle) (h : isAcuteTriangle t) :
  ∀ (r : Rectangle), isCircumscribed r t → 
    isOnLocus r.center (CurvilinearTriangle.mk (midpoints t)) :=
  sorry

end NUMINAMATH_CALUDE_locus_of_circumscribed_rectangles_centers_l1186_118600


namespace NUMINAMATH_CALUDE_tan_difference_angle_sum_l1186_118670

-- Problem 1
theorem tan_difference (A B : Real) (h : 2 * Real.tan A = 3 * Real.tan B) :
  Real.tan (A - B) = Real.sin (2 * B) / (5 - Real.cos (2 * B)) := by sorry

-- Problem 2
theorem angle_sum (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.tan α = 1/7) 
  (h4 : Real.sin β = Real.sqrt 10 / 10) :
  α + 2*β = π/4 := by sorry

end NUMINAMATH_CALUDE_tan_difference_angle_sum_l1186_118670


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l1186_118622

/-- The quadratic equation x^2 - 6kx + 9k has exactly one real root if and only if k = 1, where k is positive. -/
theorem unique_root_quadratic (k : ℝ) (h : k > 0) :
  (∃! x : ℝ, x^2 - 6*k*x + 9*k = 0) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l1186_118622


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1186_118644

/-- Given an ellipse C with equation x^2/4 + y^2/m^2 = 1 and focal length 4,
    prove that the length of its major axis is 4√2. -/
theorem ellipse_major_axis_length (m : ℝ) :
  let C := {(x, y) : ℝ × ℝ | x^2/4 + y^2/m^2 = 1}
  let focal_length : ℝ := 4
  ∃ (major_axis_length : ℝ), major_axis_length = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1186_118644


namespace NUMINAMATH_CALUDE_no_root_in_interval_l1186_118698

-- Define the function f(x) = x^5 - 3x - 1
def f (x : ℝ) : ℝ := x^5 - 3*x - 1

-- State the theorem
theorem no_root_in_interval :
  (∀ x ∈ Set.Ioo 2 3, f x ≠ 0) ∧ Continuous f := by sorry

end NUMINAMATH_CALUDE_no_root_in_interval_l1186_118698


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l1186_118639

theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  -- ABCDEF is a convex hexagon (sum of angles is 720°)
  A + B + C + D + E + F = 720 →
  -- Angles A, B, and C are congruent
  A = B ∧ B = C →
  -- Angles D, E, and F are congruent
  D = E ∧ E = F →
  -- Measure of angle A is 20 degrees less than measure of angle D
  A + 20 = D →
  -- Prove that the measure of angle D is 130 degrees
  D = 130 := by
sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l1186_118639


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l1186_118626

/-- A function that checks if a quadratic equation with rational coefficients has rational solutions -/
def has_rational_solutions (a b c : ℚ) : Prop :=
  ∃ x : ℚ, a * x^2 + b * x + c = 0

/-- The set of positive integer values of d for which 3x^2 + 7x + d = 0 has rational solutions -/
def D : Set ℕ+ :=
  {d : ℕ+ | has_rational_solutions 3 7 d.val}

theorem quadratic_rational_solutions :
  (∃ d1 d2 : ℕ+, d1 ≠ d2 ∧ D = {d1, d2}) ∧
  (∀ d1 d2 : ℕ+, d1 ∈ D → d2 ∈ D → d1.val * d2.val = 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l1186_118626


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l1186_118680

/-- Convert a binary number (represented as a list of 0s and 1s) to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.foldr (fun bit acc => 2 * acc + bit) 0

/-- Convert a decimal number to its binary representation -/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 2) ((m % 2) :: acc)
  aux n []

theorem binary_arithmetic_equality :
  let a := [1, 0, 0, 1, 1, 0]  -- 100110₂
  let b := [1, 0, 0, 1]        -- 1001₂
  let c := [1, 1, 0]           -- 110₂
  let d := [1, 1]              -- 11₂
  let result := [1, 0, 1, 1, 1, 1, 0]  -- 1011110₂
  (binary_to_decimal a + binary_to_decimal b) * binary_to_decimal c / binary_to_decimal d =
  binary_to_decimal result := by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l1186_118680


namespace NUMINAMATH_CALUDE_parabola_directrix_l1186_118681

/-- Represents a parabola in the form y = ax^2 -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ := fun x => a * x^2

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  fun y => ∃ k, y = -k ∧ p.a = 1 / (4 * k)

theorem parabola_directrix (p : Parabola) (h : p.a = 1/4) :
  directrix p = fun y => y = -1 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1186_118681


namespace NUMINAMATH_CALUDE_min_value_of_function_l1186_118614

theorem min_value_of_function (x : ℝ) (h : x > 5/4) : 
  ∀ y : ℝ, y = 4*x + 1/(4*x - 5) → y ≥ 7 := by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1186_118614


namespace NUMINAMATH_CALUDE_alphabet_value_problem_l1186_118645

theorem alphabet_value_problem (H M A T E : ℤ) : 
  H = 8 →
  M + A + T + H = 31 →
  T + E + A + M = 40 →
  M + E + E + T = 44 →
  M + A + T + E = 39 →
  A = 12 := by
sorry

end NUMINAMATH_CALUDE_alphabet_value_problem_l1186_118645


namespace NUMINAMATH_CALUDE_xyz_inequality_l1186_118641

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z ≥ x * y + y * z + z * x) : x * y * z ≥ 3 * (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l1186_118641


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1186_118660

theorem sum_of_squares_and_products (a b c : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → 
  a^2 + b^2 + c^2 = 48 → 
  a*b + b*c + c*a = 24 → 
  a + b + c = 4 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1186_118660


namespace NUMINAMATH_CALUDE_fraction_sum_equal_decimal_l1186_118637

theorem fraction_sum_equal_decimal : 
  (2 / 20 : ℝ) + (8 / 200 : ℝ) + (3 / 300 : ℝ) + 2 * (5 / 40000 : ℝ) = 0.15025 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equal_decimal_l1186_118637


namespace NUMINAMATH_CALUDE_james_earnings_ratio_l1186_118661

theorem james_earnings_ratio :
  ∀ (february_earnings : ℕ),
    4000 + february_earnings + (february_earnings - 2000) = 18000 →
    february_earnings / 4000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_earnings_ratio_l1186_118661


namespace NUMINAMATH_CALUDE_puzzle_completion_percentage_l1186_118687

theorem puzzle_completion_percentage (total_pieces : ℕ) 
  (day1_percentage day2_percentage : ℚ) (pieces_left : ℕ) : 
  total_pieces = 1000 →
  day1_percentage = 1/10 →
  day2_percentage = 1/5 →
  pieces_left = 504 →
  let pieces_after_day1 := total_pieces - (total_pieces * day1_percentage).num
  let pieces_after_day2 := pieces_after_day1 - (pieces_after_day1 * day2_percentage).num
  let pieces_completed_day3 := pieces_after_day2 - pieces_left
  (pieces_completed_day3 : ℚ) / pieces_after_day2 = 3/10 := by sorry

end NUMINAMATH_CALUDE_puzzle_completion_percentage_l1186_118687


namespace NUMINAMATH_CALUDE_subway_length_l1186_118604

/-- Calculates the length of a subway given its speed, distance between stations, and time to pass a station. -/
theorem subway_length
  (speed : ℝ)                  -- Speed of the subway in km/min
  (station_distance : ℝ)       -- Distance between stations in km
  (passing_time : ℝ)           -- Time to pass the station in minutes
  (h1 : speed = 1.6)           -- Given speed
  (h2 : station_distance = 4.85) -- Given distance between stations
  (h3 : passing_time = 3.25)   -- Given time to pass the station
  : (speed * passing_time - station_distance) * 1000 = 350 :=
by sorry

end NUMINAMATH_CALUDE_subway_length_l1186_118604


namespace NUMINAMATH_CALUDE_fourth_day_distance_l1186_118655

def distance_on_day (initial_distance : ℕ) (day : ℕ) : ℕ :=
  initial_distance * 2^(day - 1)

theorem fourth_day_distance (initial_distance : ℕ) :
  initial_distance = 18 → distance_on_day initial_distance 4 = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_day_distance_l1186_118655


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l1186_118619

theorem polynomial_product_sum (g h k : ℤ) : 
  (∀ d : ℤ, (5*d^2 + 4*d + g) * (4*d^2 + h*d - 5) = 20*d^4 + 11*d^3 - 9*d^2 + k*d - 20) →
  g + h + k = -16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l1186_118619


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_147_l1186_118656

theorem greatest_prime_factor_of_147 : ∃ p : ℕ, p = 7 ∧ Nat.Prime p ∧ p ∣ 147 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 147 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_147_l1186_118656


namespace NUMINAMATH_CALUDE_union_of_sets_l1186_118678

theorem union_of_sets : 
  let M : Set Nat := {1, 2, 5}
  let N : Set Nat := {1, 3, 5, 7}
  M ∪ N = {1, 2, 3, 5, 7} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1186_118678


namespace NUMINAMATH_CALUDE_thanksgiving_to_christmas_l1186_118674

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents months of the year -/
inductive Month
  | November
  | December

/-- Represents a date in a month -/
structure Date where
  month : Month
  day : Nat

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after n days -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (nextDay d) n

/-- Theorem: If Thanksgiving is on Thursday, November 28, then December 25 falls on a Wednesday -/
theorem thanksgiving_to_christmas 
  (thanksgiving : Date)
  (thanksgiving_day : DayOfWeek)
  (h1 : thanksgiving.month = Month.November)
  (h2 : thanksgiving.day = 28)
  (h3 : thanksgiving_day = DayOfWeek.Thursday) :
  dayAfter thanksgiving_day 27 = DayOfWeek.Wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_thanksgiving_to_christmas_l1186_118674


namespace NUMINAMATH_CALUDE_triangle_side_length_l1186_118618

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  2 * b = a + c →  -- Arithmetic sequence condition
  B = Real.pi / 3 →  -- 60 degrees in radians
  (1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 3 →  -- Area condition
  b = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1186_118618


namespace NUMINAMATH_CALUDE_pears_left_l1186_118624

/-- 
Given that Jason picked 46 pears, Keith picked 47 pears, and Mike ate 12 pears,
prove that the number of pears left is 81.
-/
theorem pears_left (jason_pears keith_pears : ℕ) (mike_ate : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_ate = 12) :
  jason_pears + keith_pears - mike_ate = 81 := by
  sorry

end NUMINAMATH_CALUDE_pears_left_l1186_118624


namespace NUMINAMATH_CALUDE_divisible_by_10101010101_has_at_least_6_nonzero_digits_l1186_118643

/-- The number of non-zero digits in the decimal representation of a natural number -/
def num_nonzero_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Any natural number divisible by 10101010101 has at least 6 non-zero digits -/
theorem divisible_by_10101010101_has_at_least_6_nonzero_digits (k : ℕ) :
  k % 10101010101 = 0 → num_nonzero_digits k ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_10101010101_has_at_least_6_nonzero_digits_l1186_118643


namespace NUMINAMATH_CALUDE_wedge_volume_l1186_118683

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (α : ℝ) (V : ℝ) : 
  d = 10 → -- diameter of the log
  α = 60 → -- angle between the two cuts in degrees
  V = (125/18) * Real.pi → -- volume of the wedge
  ∃ (r h : ℝ),
    r = d/2 ∧ -- radius of the log
    h = r ∧ -- height of the cone (equal to radius due to 60° angle)
    V = (1/6) * ((1/3) * Real.pi * r^2 * h) -- volume formula
  :=
by sorry

end NUMINAMATH_CALUDE_wedge_volume_l1186_118683


namespace NUMINAMATH_CALUDE_equation_solution_l1186_118634

theorem equation_solution :
  ∀ x : ℝ, x^3 - 4*x + 80 ≥ 0 →
  ((x / Real.sqrt 2 + 3 * Real.sqrt 2) * Real.sqrt (x^3 - 4*x + 80) = x^2 + 10*x + 24) ↔
  (x = 4 ∨ x = -1 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1186_118634


namespace NUMINAMATH_CALUDE_square_form_existence_l1186_118601

theorem square_form_existence (a b : ℕ+) (h : a.val^3 + 4 * a.val = b.val^2) :
  ∃ t : ℕ+, a.val = 2 * t.val^2 := by
sorry

end NUMINAMATH_CALUDE_square_form_existence_l1186_118601


namespace NUMINAMATH_CALUDE_overlap_area_and_perimeter_l1186_118642

/-- Given two strips of widths 1 and 2 overlapping at an angle of π/4 radians,
    the area of the overlap region is √2 and the perimeter is 4√3. -/
theorem overlap_area_and_perimeter :
  ∀ (strip1_width strip2_width overlap_angle : ℝ),
    strip1_width = 1 →
    strip2_width = 2 →
    overlap_angle = π / 4 →
    ∃ (area perimeter : ℝ),
      area = Real.sqrt 2 ∧
      perimeter = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_overlap_area_and_perimeter_l1186_118642


namespace NUMINAMATH_CALUDE_fourth_month_sale_l1186_118673

def average_sale : ℝ := 2500
def month1_sale : ℝ := 2435
def month2_sale : ℝ := 2920
def month3_sale : ℝ := 2855
def month5_sale : ℝ := 2560
def month6_sale : ℝ := 1000

theorem fourth_month_sale (x : ℝ) : 
  (month1_sale + month2_sale + month3_sale + x + month5_sale + month6_sale) / 6 = average_sale →
  x = 3230 := by
sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l1186_118673


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l1186_118640

theorem sqrt_sum_equals_seven (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l1186_118640


namespace NUMINAMATH_CALUDE_sum_nine_terms_is_99_l1186_118652

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 1 + a 4 + a 7 = 35) ∧
  (a 3 + a 6 + a 9 = 27)

/-- The sum of the first n terms of an arithmetic sequence -/
def SumArithmeticSequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- Theorem: The sum of the first 9 terms of the specified arithmetic sequence is 99 -/
theorem sum_nine_terms_is_99 (a : ℕ → ℚ) (h : ArithmeticSequence a) :
  SumArithmeticSequence a 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_nine_terms_is_99_l1186_118652


namespace NUMINAMATH_CALUDE_square_minus_four_l1186_118694

theorem square_minus_four (y : ℤ) (h : y^2 = 2209) : (y + 2) * (y - 2) = 2205 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_four_l1186_118694


namespace NUMINAMATH_CALUDE_cube_properties_l1186_118691

/-- Given a cube with volume 343 cubic centimeters, prove its surface area and internal space diagonal --/
theorem cube_properties (V : ℝ) (h : V = 343) : 
  ∃ (s : ℝ), s > 0 ∧ s^3 = V ∧ 6 * s^2 = 294 ∧ s * Real.sqrt 3 = 7 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_properties_l1186_118691


namespace NUMINAMATH_CALUDE_enhanced_mindmaster_codes_l1186_118663

/-- The number of colors available in the enhanced Mindmaster game -/
def num_colors : ℕ := 7

/-- The number of slots in a secret code -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in the enhanced Mindmaster game -/
def num_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the number of possible secret codes is 16807 -/
theorem enhanced_mindmaster_codes :
  num_codes = 16807 := by sorry

end NUMINAMATH_CALUDE_enhanced_mindmaster_codes_l1186_118663


namespace NUMINAMATH_CALUDE_max_segment_sum_l1186_118606

/-- A rhombus constructed from two equal equilateral triangles, divided into 2n^2 smaller triangles --/
structure Rhombus (n : ℕ) where
  triangles : Fin (2 * n^2) → ℕ
  triangle_values : ∀ i, 1 ≤ triangles i ∧ triangles i ≤ 2 * n^2
  distinct_values : ∀ i j, i ≠ j → triangles i ≠ triangles j

/-- The sum of positive differences on common segments of the rhombus --/
def segmentSum (n : ℕ) (r : Rhombus n) : ℕ :=
  sorry

/-- Theorem: The maximum sum of positive differences on common segments is 3n^4 - 4n^2 + 4n - 2 --/
theorem max_segment_sum (n : ℕ) : 
  (∀ r : Rhombus n, segmentSum n r ≤ 3 * n^4 - 4 * n^2 + 4 * n - 2) ∧
  (∃ r : Rhombus n, segmentSum n r = 3 * n^4 - 4 * n^2 + 4 * n - 2) :=
sorry

end NUMINAMATH_CALUDE_max_segment_sum_l1186_118606


namespace NUMINAMATH_CALUDE_smallest_multiple_five_satisfies_five_is_smallest_l1186_118659

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 625 ∣ (x * 500) → x ≥ 5 := by
  sorry

theorem five_satisfies : 625 ∣ (5 * 500) := by
  sorry

theorem five_is_smallest : ∀ (x : ℕ), x > 0 ∧ 625 ∣ (x * 500) → x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_five_satisfies_five_is_smallest_l1186_118659


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l1186_118675

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l1186_118675


namespace NUMINAMATH_CALUDE_square_area_increase_l1186_118631

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.35 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.8225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l1186_118631


namespace NUMINAMATH_CALUDE_no_valid_digit_c_l1186_118633

theorem no_valid_digit_c : ¬∃ (C : ℕ), C < 10 ∧ (200 + 10 * C + 7) % 2 = 0 ∧ (200 + 10 * C + 7) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_digit_c_l1186_118633


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1186_118647

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1186_118647


namespace NUMINAMATH_CALUDE_lottery_jackpot_probability_l1186_118676

def num_megaballs : ℕ := 30
def num_winnerballs : ℕ := 49
def num_chosen_winnerballs : ℕ := 6
def lower_sum_bound : ℕ := 100
def upper_sum_bound : ℕ := 150

def N : ℕ := sorry -- Number of ways to choose 6 numbers from 49 that sum to [100, 150]

theorem lottery_jackpot_probability :
  ∃ (p : ℚ), p = (1 : ℚ) / num_megaballs * (N : ℚ) / (Nat.choose num_winnerballs num_chosen_winnerballs) :=
by sorry

end NUMINAMATH_CALUDE_lottery_jackpot_probability_l1186_118676


namespace NUMINAMATH_CALUDE_polygon_sides_l1186_118625

theorem polygon_sides (n : ℕ) : n > 2 →
  ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ (n - 2) * 180 + x = 1350 →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l1186_118625


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1186_118697

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 24 →
  b = 23 →
  max a (max b c) = b + 4 →
  min a (min b c) = 22 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1186_118697


namespace NUMINAMATH_CALUDE_product_121_54_l1186_118617

theorem product_121_54 : 121 * 54 = 6534 := by
  sorry

end NUMINAMATH_CALUDE_product_121_54_l1186_118617


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1186_118690

theorem circle_area_ratio (R_A R_B : ℝ) (h : R_A > 0 ∧ R_B > 0) :
  (60 : ℝ) / 360 * (2 * Real.pi * R_A) = (40 : ℝ) / 360 * (2 * Real.pi * R_B) →
  (R_A^2 * Real.pi) / (R_B^2 * Real.pi) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1186_118690


namespace NUMINAMATH_CALUDE_solution_range_l1186_118671

theorem solution_range (k : ℝ) : 
  (∃ x : ℝ, x + 2*k = 4*(x + k) + 1 ∧ x < 0) → k > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l1186_118671


namespace NUMINAMATH_CALUDE_specific_cube_surface_area_l1186_118658

/-- Represents a cube with circular holes -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_diameter : ℝ

/-- Calculates the total surface area of a cube with circular holes -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  sorry

/-- Theorem stating the total surface area of a specific cube with holes -/
theorem specific_cube_surface_area :
  let cube : CubeWithHoles := { edge_length := 4, hole_diameter := 2 }
  total_surface_area cube = 96 + 42 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_specific_cube_surface_area_l1186_118658


namespace NUMINAMATH_CALUDE_room_tiles_theorem_l1186_118607

/-- Given a room with length and width in centimeters, 
    calculate the least number of square tiles required to cover the floor. -/
def leastNumberOfTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  (length / tileSize) * (width / tileSize)

/-- Theorem stating that for a room with length 720 cm and width 432 cm,
    the least number of square tiles required is 15. -/
theorem room_tiles_theorem :
  leastNumberOfTiles 720 432 = 15 := by
  sorry

#eval leastNumberOfTiles 720 432

end NUMINAMATH_CALUDE_room_tiles_theorem_l1186_118607


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_l1186_118620

-- Define the parabola
def parabola (x m : ℝ) : ℝ := x^2 + 2*x + m - 1

-- Theorem statement
theorem parabola_intersects_x_axis (m : ℝ) :
  (∃ x : ℝ, parabola x m = 0) ↔ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_l1186_118620


namespace NUMINAMATH_CALUDE_final_salt_concentration_l1186_118615

/-- Represents the volume of salt solution in arbitrary units -/
def initialVolume : ℝ := 30

/-- Represents the initial concentration of salt in the solution -/
def initialConcentration : ℝ := 0.15

/-- Represents the volume ratio of the large ball -/
def largeBallRatio : ℝ := 10

/-- Represents the volume ratio of the medium ball -/
def mediumBallRatio : ℝ := 5

/-- Represents the volume ratio of the small ball -/
def smallBallRatio : ℝ := 3

/-- Represents the overflow percentage caused by the small ball -/
def overflowPercentage : ℝ := 0.1

/-- Theorem stating that the final salt concentration is 10% -/
theorem final_salt_concentration :
  let totalOverflow := smallBallRatio + mediumBallRatio + largeBallRatio
  let remainingVolume := initialVolume - totalOverflow
  let initialSaltAmount := initialVolume * initialConcentration
  (initialSaltAmount / initialVolume) * 100 = 10 := by
  sorry


end NUMINAMATH_CALUDE_final_salt_concentration_l1186_118615


namespace NUMINAMATH_CALUDE_f_increasing_sufficient_not_necessary_l1186_118685

/-- The function f(x) defined as |x-a| + |x| --/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x|

/-- f is increasing on [0, +∞) --/
def is_increasing_on_nonneg (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f a x ≤ f a y

theorem f_increasing_sufficient_not_necessary :
  (∀ a : ℝ, a < 0 → is_increasing_on_nonneg a) ∧
  (∃ a : ℝ, a ≥ 0 ∧ is_increasing_on_nonneg a) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_sufficient_not_necessary_l1186_118685


namespace NUMINAMATH_CALUDE_evaluate_expression_l1186_118611

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  (3 + x * (3 + x) - 3^2 + x) / (x - 3 + x^2 - x) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1186_118611


namespace NUMINAMATH_CALUDE_notebook_cost_l1186_118653

theorem notebook_cost (total_students : ℕ) 
  (buyers : ℕ) 
  (notebooks_per_buyer : ℕ) 
  (notebook_cost : ℕ) 
  (total_cost : ℕ) :
  total_students = 40 →
  buyers > total_students / 2 →
  notebooks_per_buyer > 2 →
  notebook_cost > 2 * notebooks_per_buyer →
  buyers * notebooks_per_buyer * notebook_cost = total_cost →
  total_cost = 4515 →
  notebook_cost = 35 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l1186_118653


namespace NUMINAMATH_CALUDE_article_original_price_l1186_118628

/-- Calculates the original price of an article given its selling price and loss percentage. -/
def originalPrice (sellingPrice : ℚ) (lossPercent : ℚ) : ℚ :=
  sellingPrice / (1 - lossPercent / 100)

/-- Theorem stating that an article sold for 450 with a 25% loss had an original price of 600. -/
theorem article_original_price :
  originalPrice 450 25 = 600 := by
  sorry

end NUMINAMATH_CALUDE_article_original_price_l1186_118628


namespace NUMINAMATH_CALUDE_largest_prime_factor_and_difference_l1186_118669

def n : ℕ := 3136

theorem largest_prime_factor_and_difference (p : ℕ → Prop) :
  (∀ q, q > 1 ∧ n % q = 0 → p q) →
  (∃ q, q > 1 ∧ n % q = 0 ∧ p q) →
  (∀ q, q > 1 ∧ n % q = 0 ∧ p q → q ≤ 7) →
  n % 7 = 0 →
  (∃ q, q > 1 ∧ n % q = 0 ∧ p q ∧ ∀ r, r > 1 ∧ n % r = 0 ∧ p r → r ≤ q) →
  (∃ q, q > 1 ∧ n % q = 0 ∧ p q ∧ ∀ r, r > 1 ∧ n % r = 0 ∧ p r → q ≤ r) →
  7^2 - (minp : ℕ) = 47 →
  minp > 1 ∧ n % minp = 0 ∧ p minp ∧ ∀ q, q > 1 ∧ n % q = 0 ∧ p q → minp ≤ q :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_and_difference_l1186_118669


namespace NUMINAMATH_CALUDE_grocery_store_salary_l1186_118689

/-- Calculates the total daily salary of employees in a grocery store. -/
def total_daily_salary (manager_salary : ℕ) (clerk_salary : ℕ) (num_managers : ℕ) (num_clerks : ℕ) : ℕ :=
  manager_salary * num_managers + clerk_salary * num_clerks

/-- Proves that the total daily salary of all employees in the grocery store is $16. -/
theorem grocery_store_salary : total_daily_salary 5 2 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_salary_l1186_118689


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1186_118662

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1186_118662


namespace NUMINAMATH_CALUDE_shortest_chord_line_l1186_118602

/-- The circle C in the 2D plane -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 2)^2 = 5}

/-- The line l passing through (1,1) -/
def l : Set (ℝ × ℝ) := {p | p.1 - p.2 = 0}

/-- The point (1,1) -/
def A : ℝ × ℝ := (1, 1)

/-- Theorem: The line l intersects the circle C with the shortest chord length -/
theorem shortest_chord_line :
  A ∈ l ∧
  (∃ p q : ℝ × ℝ, p ∈ l ∧ q ∈ l ∧ p ∈ C ∧ q ∈ C ∧ p ≠ q) ∧
  (∀ m : Set (ℝ × ℝ), A ∈ m →
    (∃ p q : ℝ × ℝ, p ∈ m ∧ q ∈ m ∧ p ∈ C ∧ q ∈ C ∧ p ≠ q) →
    ∃ p q : ℝ × ℝ, p ∈ l ∧ q ∈ l ∧ p ∈ C ∧ q ∈ C ∧
    ∀ r s : ℝ × ℝ, r ∈ m ∧ s ∈ m ∧ r ∈ C ∧ s ∈ C →
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((r.1 - s.1)^2 + (r.2 - s.2)^2)) :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_line_l1186_118602


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_a_value_when_p_intersect_q_equals_q_l1186_118648

-- Define the solution sets P and Q
def P (a : ℝ) : Set ℝ := {x : ℝ | (x - a) / (x + 1) < 0}
def Q : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

-- Theorem 1: When a = 3, P = {x | -1 < x < 3}
theorem solution_set_when_a_is_3 : 
  P 3 = {x : ℝ | -1 < x ∧ x < 3} := by sorry

-- Theorem 2: When P ∩ Q = Q, a = 2
theorem a_value_when_p_intersect_q_equals_q : 
  (∃ a : ℝ, a > 0 ∧ P a ∩ Q = Q) → (∃ a : ℝ, a = 2 ∧ P a ∩ Q = Q) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_a_value_when_p_intersect_q_equals_q_l1186_118648


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1186_118686

theorem product_sum_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 6)
  (eq3 : a + c + d = 15)
  (eq4 : b + c + d = 10) :
  a * b + c * d = 408 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1186_118686


namespace NUMINAMATH_CALUDE_brody_calculator_theorem_l1186_118646

def calculator_problem (total_battery : ℝ) (used_fraction : ℝ) (exam_duration : ℝ) : Prop :=
  let remaining_before_exam := total_battery * (1 - used_fraction)
  let remaining_after_exam := remaining_before_exam - exam_duration
  remaining_after_exam = 13

theorem brody_calculator_theorem :
  calculator_problem 60 (3/4) 2 := by
  sorry

end NUMINAMATH_CALUDE_brody_calculator_theorem_l1186_118646


namespace NUMINAMATH_CALUDE_roses_in_vase_l1186_118696

/-- The number of roses in a vase after adding new roses -/
def total_roses (initial_roses new_roses : ℕ) : ℕ :=
  initial_roses + new_roses

/-- Theorem: There are 23 roses in the vase after Jessica adds her newly cut roses -/
theorem roses_in_vase : total_roses 7 16 = 23 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l1186_118696


namespace NUMINAMATH_CALUDE_art_fair_sales_l1186_118609

/-- The total number of paintings sold at Tracy's art fair booth -/
def total_paintings_sold (first_group : Nat) (second_group : Nat) (third_group : Nat)
  (first_group_purchase : Nat) (second_group_purchase : Nat) (third_group_purchase : Nat) : Nat :=
  first_group * first_group_purchase +
  second_group * second_group_purchase +
  third_group * third_group_purchase

/-- Theorem stating the total number of paintings sold at Tracy's art fair booth -/
theorem art_fair_sales :
  total_paintings_sold 4 12 4 2 1 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_art_fair_sales_l1186_118609


namespace NUMINAMATH_CALUDE_melanie_grew_more_turnips_l1186_118638

/-- The number of turnips Melanie grew -/
def melanie_turnips : ℕ := 139

/-- The number of turnips Benny grew -/
def benny_turnips : ℕ := 113

/-- The difference in turnips grown between Melanie and Benny -/
def turnip_difference : ℕ := melanie_turnips - benny_turnips

theorem melanie_grew_more_turnips : turnip_difference = 26 := by
  sorry

end NUMINAMATH_CALUDE_melanie_grew_more_turnips_l1186_118638


namespace NUMINAMATH_CALUDE_log_xyz_value_l1186_118665

-- Define the variables
variable (x y z : ℝ)
variable (log : ℝ → ℝ)

-- State the theorem
theorem log_xyz_value (h1 : log (x * y^3 * z) = 2) (h2 : log (x^2 * y * z^2) = 3) :
  log (x * y * z) = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_log_xyz_value_l1186_118665


namespace NUMINAMATH_CALUDE_trajectory_equation_l1186_118684

theorem trajectory_equation (x y : ℝ) (h1 : x > 0) :
  (((x - 1/2)^2 + y^2)^(1/2) = x + 1/2) → y^2 = 2*x := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1186_118684


namespace NUMINAMATH_CALUDE_line_intercepts_l1186_118621

/-- The equation of the line -/
def line_equation (x y : ℚ) : Prop := 4 * x + 7 * y = 28

/-- Definition of x-intercept -/
def is_x_intercept (x : ℚ) : Prop := line_equation x 0

/-- Definition of y-intercept -/
def is_y_intercept (y : ℚ) : Prop := line_equation 0 y

/-- Theorem: The x-intercept of the line 4x + 7y = 28 is (7, 0), and the y-intercept is (0, 4) -/
theorem line_intercepts : is_x_intercept 7 ∧ is_y_intercept 4 := by sorry

end NUMINAMATH_CALUDE_line_intercepts_l1186_118621


namespace NUMINAMATH_CALUDE_original_average_proof_l1186_118623

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℝ) : 
  n = 30 → 
  new_avg = 90 → 
  new_avg = 2 * original_avg → 
  original_avg = 45 := by
sorry

end NUMINAMATH_CALUDE_original_average_proof_l1186_118623


namespace NUMINAMATH_CALUDE_exactly_one_zero_in_interval_l1186_118627

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

theorem exactly_one_zero_in_interval (a : ℝ) (h : a > 3) :
  ∃! x, x ∈ (Set.Ioo 0 2) ∧ f a x = 0 := by
sorry

end NUMINAMATH_CALUDE_exactly_one_zero_in_interval_l1186_118627


namespace NUMINAMATH_CALUDE_complete_square_sum_l1186_118613

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → 
  b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1186_118613


namespace NUMINAMATH_CALUDE_no_solution_exists_l1186_118677

theorem no_solution_exists : ¬ ∃ (a b : ℤ), a^2 = b^15 + 1004 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1186_118677


namespace NUMINAMATH_CALUDE_richards_third_day_distance_l1186_118654

/-- Represents Richard's journey from Cincinnati to New York City -/
structure Journey where
  total_distance : ℝ
  day1_distance : ℝ
  day2_distance : ℝ
  day3_distance : ℝ
  remaining_distance : ℝ

/-- Theorem stating the distance Richard walked on the third day -/
theorem richards_third_day_distance (j : Journey)
  (h1 : j.total_distance = 70)
  (h2 : j.day1_distance = 20)
  (h3 : j.day2_distance = j.day1_distance / 2 - 6)
  (h4 : j.remaining_distance = 36)
  (h5 : j.day1_distance + j.day2_distance + j.day3_distance + j.remaining_distance = j.total_distance) :
  j.day3_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_richards_third_day_distance_l1186_118654


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l1186_118692

/-- A regular tetrahedron with unit edge length -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- A sphere touching three faces of a regular tetrahedron and three sides of its fourth face -/
structure InscribedSphere (t : RegularTetrahedron) where
  radius : ℝ
  touches_three_faces : True  -- Placeholder for the condition
  touches_three_sides_of_fourth_face : True  -- Placeholder for the condition

/-- The radius of the inscribed sphere is √6/8 -/
theorem inscribed_sphere_radius (t : RegularTetrahedron) (s : InscribedSphere t) :
  s.radius = Real.sqrt 6 / 8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l1186_118692


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_l1186_118649

theorem least_sum_of_exponents (h : ℕ+) (a b c d e : ℕ) 
  (h_div_225 : 225 ∣ h) (h_div_216 : 216 ∣ h) (h_div_847 : 847 ∣ h)
  (h_factorization : h = 2^a * 3^b * 5^c * 7^d * 11^e) :
  ∃ (a' b' c' d' e' : ℕ), 
    h = 2^a' * 3^b' * 5^c' * 7^d' * 11^e' ∧
    a' + b' + c' + d' + e' ≤ a + b + c + d + e ∧
    a' + b' + c' + d' + e' = 10 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_l1186_118649


namespace NUMINAMATH_CALUDE_max_elevation_l1186_118630

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 200 * t - 20 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (t' : ℝ), s t' ≤ s t ∧ s t = 500 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_l1186_118630


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l1186_118616

theorem line_slope_intercept_product (m b : ℚ) : 
  m > 0 → b < 0 → m = 3/4 → b = -2/3 → -1 < m * b ∧ m * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l1186_118616


namespace NUMINAMATH_CALUDE_total_rainfall_l1186_118612

def rainfall_problem (first_week : ℝ) (second_week : ℝ) : Prop :=
  (second_week = 1.5 * first_week) ∧
  (second_week = 12) ∧
  (first_week + second_week = 20)

theorem total_rainfall : ∃ (first_week second_week : ℝ), 
  rainfall_problem first_week second_week :=
by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_l1186_118612


namespace NUMINAMATH_CALUDE_base12_2413_mod_9_l1186_118657

-- Define a function to convert base-12 to decimal
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

-- Define the base-12 number 2413
def base12_2413 : List Nat := [2, 4, 1, 3]

-- Theorem statement
theorem base12_2413_mod_9 :
  (base12ToDecimal base12_2413) % 9 = 8 := by
  sorry


end NUMINAMATH_CALUDE_base12_2413_mod_9_l1186_118657


namespace NUMINAMATH_CALUDE_inequality_solutions_count_l1186_118636

theorem inequality_solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => 
    (p.1 : ℚ) / 76 + (p.2 : ℚ) / 71 < 1) 
    (Finset.product (Finset.range 76) (Finset.range 71))).card = 2625 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_count_l1186_118636
