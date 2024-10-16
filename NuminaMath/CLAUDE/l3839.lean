import Mathlib

namespace NUMINAMATH_CALUDE_circle_T_six_three_l3839_383950

-- Define the operation ⊤
def circle_T (a b : ℤ) : ℤ := 4 * a - 7 * b

-- Theorem statement
theorem circle_T_six_three : circle_T 6 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_T_six_three_l3839_383950


namespace NUMINAMATH_CALUDE_star_negative_two_five_l3839_383934

-- Define the ⋆ operation
def star (a b : ℤ) : ℤ := a * b^3 - b^2 + 2

-- Theorem statement
theorem star_negative_two_five : star (-2) 5 = -273 := by
  sorry

end NUMINAMATH_CALUDE_star_negative_two_five_l3839_383934


namespace NUMINAMATH_CALUDE_coin_order_l3839_383989

/-- Represents a coin in the arrangement -/
inductive Coin
| A | B | C | D | E | F

/-- Represents the relative position of two coins -/
inductive Position
| Above | Below

/-- Defines the relationship between two coins -/
def relation (c1 c2 : Coin) : Position := sorry

/-- Theorem stating the correct order of coins from top to bottom -/
theorem coin_order :
  (relation Coin.F Coin.E = Position.Above) ∧
  (relation Coin.F Coin.C = Position.Above) ∧
  (relation Coin.F Coin.D = Position.Above) ∧
  (relation Coin.F Coin.A = Position.Above) ∧
  (relation Coin.F Coin.B = Position.Above) ∧
  (relation Coin.E Coin.C = Position.Above) ∧
  (relation Coin.E Coin.D = Position.Above) ∧
  (relation Coin.E Coin.A = Position.Above) ∧
  (relation Coin.E Coin.B = Position.Above) ∧
  (relation Coin.C Coin.A = Position.Above) ∧
  (relation Coin.C Coin.B = Position.Above) ∧
  (relation Coin.D Coin.A = Position.Above) ∧
  (relation Coin.D Coin.B = Position.Above) ∧
  (relation Coin.A Coin.B = Position.Above) →
  ∀ (c : Coin), c ≠ Coin.F →
    (relation Coin.F c = Position.Above) ∧
    (∀ (d : Coin), d ≠ Coin.B →
      (relation c Coin.B = Position.Above)) :=
by sorry

end NUMINAMATH_CALUDE_coin_order_l3839_383989


namespace NUMINAMATH_CALUDE_triangle_construction_from_feet_l3839_383966

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The foot of an altitude in a triangle -/
def altitude_foot (T : Triangle) (v : Point) : Point :=
  sorry

/-- The foot of an angle bisector in a triangle -/
def angle_bisector_foot (T : Triangle) (v : Point) : Point :=
  sorry

/-- Theorem: A unique triangle exists given the feet of two altitudes and one angle bisector -/
theorem triangle_construction_from_feet 
  (A₁ B₁ B₂ : Point) : 
  ∃! (T : Triangle), 
    altitude_foot T T.A = A₁ ∧ 
    altitude_foot T T.B = B₁ ∧ 
    angle_bisector_foot T T.B = B₂ :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_from_feet_l3839_383966


namespace NUMINAMATH_CALUDE_factorization_equality_l3839_383968

theorem factorization_equality (x y : ℝ) : 3 * x^2 * y - 3 * y^3 = 3 * y * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3839_383968


namespace NUMINAMATH_CALUDE_modulus_of_z_l3839_383931

def i : ℂ := Complex.I

def z : ℂ := (1 + i) * (1 + 2*i)

theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3839_383931


namespace NUMINAMATH_CALUDE_ilwoong_drive_files_l3839_383976

theorem ilwoong_drive_files (num_folders : ℕ) (subfolders_per_folder : ℕ) (files_per_subfolder : ℕ) :
  num_folders = 25 →
  subfolders_per_folder = 10 →
  files_per_subfolder = 8 →
  num_folders * subfolders_per_folder * files_per_subfolder = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ilwoong_drive_files_l3839_383976


namespace NUMINAMATH_CALUDE_percentage_sum_l3839_383909

theorem percentage_sum (A B C : ℝ) : 
  (0.45 * A = 270) → 
  (0.35 * B = 210) → 
  (0.25 * C = 150) → 
  (0.75 * A + 0.65 * B + 0.45 * C = 1110) := by
sorry

end NUMINAMATH_CALUDE_percentage_sum_l3839_383909


namespace NUMINAMATH_CALUDE_guard_skipped_circles_l3839_383991

def warehouse_length : ℕ := 600
def warehouse_width : ℕ := 400
def intended_circles : ℕ := 10
def actual_distance : ℕ := 16000

def perimeter : ℕ := 2 * (warehouse_length + warehouse_width)
def intended_distance : ℕ := intended_circles * perimeter
def skipped_distance : ℕ := intended_distance - actual_distance
def times_skipped : ℕ := skipped_distance / perimeter

theorem guard_skipped_circles :
  times_skipped = 2 := by sorry

end NUMINAMATH_CALUDE_guard_skipped_circles_l3839_383991


namespace NUMINAMATH_CALUDE_point_relationship_on_line_l3839_383920

/-- Proves that for two points on a line with positive slope and non-negative y-intercept,
    if the x-coordinate of the first point is greater than the x-coordinate of the second point,
    then the y-coordinate of the first point is greater than the y-coordinate of the second point. -/
theorem point_relationship_on_line (k b x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : y₁ = k * x₁ + b)
  (h2 : y₂ = k * x₂ + b)
  (h3 : k > 0)
  (h4 : b ≥ 0)
  (h5 : x₁ > x₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_point_relationship_on_line_l3839_383920


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l3839_383961

theorem polynomial_value_theorem (x : ℝ) (h : x^2 - 8*x - 3 = 0) :
  (x - 1) * (x - 3) * (x - 5) * (x - 7) = 180 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l3839_383961


namespace NUMINAMATH_CALUDE_certain_number_exists_l3839_383937

theorem certain_number_exists : ∃ x : ℝ, (1.78 * x) / 5.96 = 377.8020134228188 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l3839_383937


namespace NUMINAMATH_CALUDE_cube_dot_path_length_l3839_383967

theorem cube_dot_path_length (cube_edge : ℝ) (dot_path : ℝ) : 
  cube_edge = 2 →
  dot_path = Real.sqrt 5 * Real.pi →
  ∃ (rotation_radius : ℝ),
    rotation_radius = Real.sqrt (1^2 + 2^2) ∧
    dot_path = 4 * (1/4 * 2 * Real.pi * rotation_radius) :=
by sorry

end NUMINAMATH_CALUDE_cube_dot_path_length_l3839_383967


namespace NUMINAMATH_CALUDE_rectangle_perimeter_is_128_l3839_383978

/-- Represents a rectangle with an inscribed ellipse -/
structure RectangleWithEllipse where
  rect_area : ℝ
  ellipse_area : ℝ
  major_axis : ℝ

/-- The perimeter of the rectangle given the specified conditions -/
def rectangle_perimeter (r : RectangleWithEllipse) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the rectangle under given conditions -/
theorem rectangle_perimeter_is_128 (r : RectangleWithEllipse) 
  (h1 : r.rect_area = 4032)
  (h2 : r.ellipse_area = 4032 * Real.pi)
  (h3 : r.major_axis = 2 * rectangle_perimeter r / 2) : 
  rectangle_perimeter r = 128 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_is_128_l3839_383978


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l3839_383903

def reading_speed : ℝ := 120
def pages_per_book : ℕ := 360
def available_time : ℝ := 10

def books_read : ℕ := 3

theorem robert_reading_capacity : 
  (reading_speed * available_time) / pages_per_book ≥ books_read ∧ 
  (reading_speed * available_time) / pages_per_book < books_read + 1 :=
sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l3839_383903


namespace NUMINAMATH_CALUDE_pebble_collection_l3839_383954

theorem pebble_collection (n : ℕ) (h : n = 20) : (n * (n + 1)) / 2 = 210 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_l3839_383954


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3839_383953

/-- Given a rectangle where the sum of its length and width is 24 centimeters,
    prove that its perimeter is 48 centimeters. -/
theorem rectangle_perimeter (length width : ℝ) (h : length + width = 24) :
  2 * (length + width) = 48 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3839_383953


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l3839_383984

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 1997 is the smallest prime whose digits sum to 23 -/
theorem smallest_prime_digit_sum_23 :
  (is_prime 1997) ∧ 
  (digit_sum 1997 = 23) ∧ 
  (∀ n : ℕ, n < 1997 → (is_prime n ∧ digit_sum n = 23) → False) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l3839_383984


namespace NUMINAMATH_CALUDE_complex_root_quadratic_l3839_383907

theorem complex_root_quadratic (b c : ℝ) : 
  (Complex.I * Real.sqrt 2 + 1) ^ 2 + b * (Complex.I * Real.sqrt 2 + 1) + c = 0 → 
  b = -2 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_quadratic_l3839_383907


namespace NUMINAMATH_CALUDE_solutions_difference_squared_l3839_383947

theorem solutions_difference_squared (α β : ℝ) : 
  α ≠ β ∧ α^2 = 2*α + 1 ∧ β^2 = 2*β + 1 → (α - β)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_solutions_difference_squared_l3839_383947


namespace NUMINAMATH_CALUDE_metaphase_mitosis_observable_l3839_383910

/-- Represents the types of cell division that can occur in testis --/
inductive CellDivisionType
| Mitosis
| Meiosis

/-- Represents the phases of mitosis --/
inductive MitosisPhase
| Prophase
| Metaphase
| Anaphase
| Telophase

/-- Represents a cell in a testis slice --/
structure TestisCell where
  divisionType : CellDivisionType
  phase : Option MitosisPhase

/-- Represents a locust testis slice --/
structure LocustTestisSlice where
  cells : List TestisCell

/-- Condition: Both meiosis and mitosis can occur in the testis --/
def testisCanUndergoMitosisAndMeiosis (slice : LocustTestisSlice) : Prop :=
  ∃ (c1 c2 : TestisCell), c1 ∈ slice.cells ∧ c2 ∈ slice.cells ∧
    c1.divisionType = CellDivisionType.Mitosis ∧
    c2.divisionType = CellDivisionType.Meiosis

/-- Theorem: In locust testis slices, cells in the metaphase of mitosis can be observed --/
theorem metaphase_mitosis_observable (slice : LocustTestisSlice) 
  (h : testisCanUndergoMitosisAndMeiosis slice) :
  ∃ (c : TestisCell), c ∈ slice.cells ∧ 
    c.divisionType = CellDivisionType.Mitosis ∧
    c.phase = some MitosisPhase.Metaphase :=
  sorry

end NUMINAMATH_CALUDE_metaphase_mitosis_observable_l3839_383910


namespace NUMINAMATH_CALUDE_total_goats_l3839_383990

def washington_herd : ℕ := 5000
def paddington_difference : ℕ := 220

theorem total_goats : washington_herd + (washington_herd + paddington_difference) = 10220 := by
  sorry

end NUMINAMATH_CALUDE_total_goats_l3839_383990


namespace NUMINAMATH_CALUDE_count_valid_programs_l3839_383994

/-- Represents the available courses --/
inductive Course
| English
| Algebra
| Geometry
| History
| Art
| Latin
| Science

/-- Checks if a course is a mathematics course --/
def isMathCourse (c : Course) : Bool :=
  match c with
  | Course.Algebra => true
  | Course.Geometry => true
  | _ => false

/-- Checks if a course is a science course --/
def isScienceCourse (c : Course) : Bool :=
  match c with
  | Course.Science => true
  | _ => false

/-- Represents a program of 4 courses --/
structure Program :=
  (courses : Finset Course)
  (size_eq : courses.card = 4)
  (has_english : Course.English ∈ courses)
  (has_math : ∃ c ∈ courses, isMathCourse c)
  (has_science : ∃ c ∈ courses, isScienceCourse c)

/-- The set of all valid programs --/
def validPrograms : Finset Program := sorry

theorem count_valid_programs :
  validPrograms.card = 19 := by sorry

end NUMINAMATH_CALUDE_count_valid_programs_l3839_383994


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l3839_383940

theorem roots_sum_of_squares (p q : ℝ) : 
  (p^2 - 5*p + 6 = 0) → (q^2 - 5*q + 6 = 0) → p ≠ q → p^2 + q^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l3839_383940


namespace NUMINAMATH_CALUDE_cab_driver_income_l3839_383995

def average_income : ℝ := 440
def num_days : ℕ := 5
def known_incomes : List ℝ := [250, 650, 400, 500]

theorem cab_driver_income :
  let total_income := average_income * num_days
  let known_total := known_incomes.sum
  total_income - known_total = 400 := by sorry

end NUMINAMATH_CALUDE_cab_driver_income_l3839_383995


namespace NUMINAMATH_CALUDE_parabola_vertex_l3839_383973

/-- The parabola defined by the equation y = 2(x-3)^2 - 7 -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 - 7

/-- The vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem: The vertex of the parabola y = 2(x-3)^2 - 7 is at the point (3, -7) -/
theorem parabola_vertex : 
  ∃ (v : Vertex), v.x = 3 ∧ v.y = -7 ∧ 
  (∀ (x y : ℝ), parabola x y → (x - v.x)^2 ≤ (y - v.y) / 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3839_383973


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3839_383955

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3839_383955


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3839_383914

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of the tangent line
def has_tangent_line (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ (x : ℝ), f x = m * x + b

-- State the theorem
theorem tangent_line_sum (h : has_tangent_line f) :
  f 1 + (deriv f) 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3839_383914


namespace NUMINAMATH_CALUDE_probability_divisor_of_twelve_l3839_383969

def divisors_of_twelve : Finset ℕ := {1, 2, 3, 4, 6, 12}

theorem probability_divisor_of_twelve (die : Finset ℕ) 
  (h1 : die = Finset.range 12) 
  (h2 : die.card = 12) : 
  (divisors_of_twelve.card : ℚ) / (die.card : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisor_of_twelve_l3839_383969


namespace NUMINAMATH_CALUDE_simplification_proofs_l3839_383979

theorem simplification_proofs :
  (∀ x : ℝ, x ≥ 0 → Real.sqrt (x^2) = x) ∧
  ((5 * Real.sqrt 5)^2 = 125) ∧
  (Real.sqrt ((-1/7)^2) = 1/7) ∧
  ((-Real.sqrt (2/3))^2 = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_simplification_proofs_l3839_383979


namespace NUMINAMATH_CALUDE_time_to_empty_tank_l3839_383992

/-- Time to empty a tank given its volume and pipe rates -/
theorem time_to_empty_tank 
  (tank_volume : ℝ) 
  (inlet_rate : ℝ) 
  (outlet_rate1 : ℝ) 
  (outlet_rate2 : ℝ) 
  (h1 : tank_volume = 30) 
  (h2 : inlet_rate = 3) 
  (h3 : outlet_rate1 = 12) 
  (h4 : outlet_rate2 = 6) : 
  (tank_volume * 1728) / (outlet_rate1 + outlet_rate2 - inlet_rate) = 3456 := by
  sorry


end NUMINAMATH_CALUDE_time_to_empty_tank_l3839_383992


namespace NUMINAMATH_CALUDE_apple_bag_weight_l3839_383946

/-- Given a bag of apples costing 3.50 dollars, and knowing that 7 pounds of apples
    at the same rate would cost 4.9 dollars, prove that the bag contains 5 pounds of apples. -/
theorem apple_bag_weight (bag_cost : ℝ) (rate_pounds : ℝ) (rate_cost : ℝ) :
  bag_cost = 3.50 →
  rate_pounds = 7 →
  rate_cost = 4.9 →
  (rate_cost / rate_pounds) * (bag_cost / (rate_cost / rate_pounds)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_apple_bag_weight_l3839_383946


namespace NUMINAMATH_CALUDE_remaining_milk_calculation_l3839_383983

/-- The amount of milk arranged by the shop owner in liters -/
def total_milk : ℝ := 21.52

/-- The amount of milk sold in liters -/
def sold_milk : ℝ := 12.64

/-- The amount of remaining milk in liters -/
def remaining_milk : ℝ := total_milk - sold_milk

theorem remaining_milk_calculation : remaining_milk = 8.88 := by
  sorry

end NUMINAMATH_CALUDE_remaining_milk_calculation_l3839_383983


namespace NUMINAMATH_CALUDE_small_ring_rotation_l3839_383944

theorem small_ring_rotation (r₁ r₂ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 4) :
  (2 * r₂ * Real.pi - 2 * r₁ * Real.pi) / (2 * r₁ * Real.pi) = 3 := by
  sorry

end NUMINAMATH_CALUDE_small_ring_rotation_l3839_383944


namespace NUMINAMATH_CALUDE_multiply_213_by_16_l3839_383929

theorem multiply_213_by_16 (h : 213 * 1.6 = 340.8) : 213 * 16 = 3408 := by
  sorry

end NUMINAMATH_CALUDE_multiply_213_by_16_l3839_383929


namespace NUMINAMATH_CALUDE_pen_price_proof_l3839_383987

theorem pen_price_proof (total_cost : ℝ) (notebook_ratio : ℝ) :
  total_cost = 36.45 →
  notebook_ratio = 15 / 4 →
  ∃ (pen_price : ℝ),
    pen_price + 3 * (notebook_ratio * pen_price) = total_cost ∧
    pen_price = 5.4 :=
by sorry

end NUMINAMATH_CALUDE_pen_price_proof_l3839_383987


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3839_383986

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3839_383986


namespace NUMINAMATH_CALUDE_train_speed_l3839_383965

/-- The speed of a train crossing a platform -/
theorem train_speed (train_length platform_length : Real) (crossing_time : Real) 
  (h1 : train_length = 120)
  (h2 : platform_length = 130.02)
  (h3 : crossing_time = 15) : 
  ∃ (speed : Real), abs (speed - 60.0048) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3839_383965


namespace NUMINAMATH_CALUDE_constant_z_is_plane_l3839_383930

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying z = c in cylindrical coordinates -/
def ConstantZSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.z = c}

/-- Definition of a plane in cylindrical coordinates -/
def IsPlane (S : Set CylindricalPoint) : Prop :=
  ∃ (a b c d : ℝ), c ≠ 0 ∧ ∀ p ∈ S, a * p.r * (Real.cos p.θ) + b * p.r * (Real.sin p.θ) + c * p.z = d

theorem constant_z_is_plane (c : ℝ) : IsPlane (ConstantZSet c) := by
  sorry

end NUMINAMATH_CALUDE_constant_z_is_plane_l3839_383930


namespace NUMINAMATH_CALUDE_ln_graph_rotation_l3839_383906

open Real

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := log x

-- Define the rotation angle
variable (θ : ℝ)

-- State the theorem
theorem ln_graph_rotation (h : ∃ x > 0, f x * cos θ + x * sin θ = 0) :
  sin θ = ℯ * cos θ :=
sorry

end NUMINAMATH_CALUDE_ln_graph_rotation_l3839_383906


namespace NUMINAMATH_CALUDE_cards_found_l3839_383918

theorem cards_found (initial_cards final_cards : ℕ) : 
  initial_cards = 7 → final_cards = 54 → final_cards - initial_cards = 47 := by
  sorry

end NUMINAMATH_CALUDE_cards_found_l3839_383918


namespace NUMINAMATH_CALUDE_units_digit_sum_base8_l3839_383936

-- Define a function to get the units digit in base 8
def units_digit_base8 (n : Nat) : Nat :=
  n % 8

-- Define the addition operation in base 8
def add_base8 (a b : Nat) : Nat :=
  (a + b) % 8

-- Theorem statement
theorem units_digit_sum_base8 :
  units_digit_base8 (add_base8 65 75) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_base8_l3839_383936


namespace NUMINAMATH_CALUDE_triangle_problem_l3839_383927

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < π/2 ∧
  0 < t.B ∧ t.B < π/2 ∧
  0 < t.C ∧ t.C < π/2

def satisfiesSineLaw (t : Triangle) : Prop :=
  t.a / sin t.A = t.b / sin t.B ∧
  t.b / sin t.B = t.c / sin t.C

def satisfiesGivenCondition (t : Triangle) : Prop :=
  2 * t.a * sin t.B = t.b

-- The main theorem
theorem triangle_problem (t : Triangle)
  (h_acute : isAcute t)
  (h_sine_law : satisfiesSineLaw t)
  (h_condition : satisfiesGivenCondition t) :
  t.A = π/6 ∧
  (t.a = 6 ∧ t.b + t.c = 8 →
    1/2 * t.b * t.c * sin t.A = 14 - 7 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3839_383927


namespace NUMINAMATH_CALUDE_sqrt_4_plus_2_inv_l3839_383998

theorem sqrt_4_plus_2_inv : Real.sqrt 4 + 2⁻¹ = 5/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_4_plus_2_inv_l3839_383998


namespace NUMINAMATH_CALUDE_annual_rent_per_square_foot_l3839_383997

/-- Calculates the annual rent per square foot of a shop given its dimensions and monthly rent. -/
theorem annual_rent_per_square_foot 
  (length width : ℝ) 
  (monthly_rent : ℝ) 
  (h1 : length = 18) 
  (h2 : width = 20) 
  (h3 : monthly_rent = 3600) : 
  (monthly_rent * 12) / (length * width) = 120 := by
  sorry

end NUMINAMATH_CALUDE_annual_rent_per_square_foot_l3839_383997


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l3839_383982

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_expression_simplification (a b : V) :
  (1 / 2 : ℝ) • ((2 : ℝ) • a - (4 : ℝ) • b) + (2 : ℝ) • b = a := by sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l3839_383982


namespace NUMINAMATH_CALUDE_sequence_properties_l3839_383948

def S (n : ℕ+) : ℤ := -n^2 + 24*n

def a (n : ℕ+) : ℤ := -2*n + 25

theorem sequence_properties :
  (∀ n : ℕ+, S n - S (n-1) = a n) ∧
  (∀ n : ℕ+, n ≤ 12 → S n ≤ S 12) ∧
  (S 12 = 144) := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3839_383948


namespace NUMINAMATH_CALUDE_f_max_value_l3839_383943

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.cos x

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l3839_383943


namespace NUMINAMATH_CALUDE_inequality_proof_l3839_383913

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 2015) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ (Real.sqrt a + Real.sqrt b + Real.sqrt c) / Real.sqrt 2015 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3839_383913


namespace NUMINAMATH_CALUDE_greendale_points_greendale_points_equals_130_l3839_383959

/-- Calculates the total points for Greendale High School in a basketball tournament --/
theorem greendale_points (roosevelt_first_game : ℕ) (bonus : ℕ) (difference : ℕ) : ℕ :=
  let roosevelt_second_game := roosevelt_first_game / 2
  let roosevelt_third_game := roosevelt_second_game * 3
  let roosevelt_total := roosevelt_first_game + roosevelt_second_game + roosevelt_third_game + bonus
  roosevelt_total - difference

/-- Proves that Greendale High School's total points equal 130 --/
theorem greendale_points_equals_130 : greendale_points 30 50 10 = 130 := by
  sorry

end NUMINAMATH_CALUDE_greendale_points_greendale_points_equals_130_l3839_383959


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_base_ten_is_perfect_square_ten_is_smallest_base_l3839_383952

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 4 → (2 * b + 5 = n ^ 2 → n ≥ 5) → b ≥ 10 :=
by
  sorry

theorem base_ten_is_perfect_square : 
  ∃ n : ℕ, 2 * 10 + 5 = n ^ 2 :=
by
  sorry

theorem ten_is_smallest_base :
  (∀ b : ℕ, b > 4 ∧ b < 10 → ¬∃ n : ℕ, 2 * b + 5 = n ^ 2) ∧
  (∃ n : ℕ, 2 * 10 + 5 = n ^ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_base_ten_is_perfect_square_ten_is_smallest_base_l3839_383952


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l3839_383999

/-- Given a rectangle ABCD and a square EFGH, if the rectangle shares 60% of its area with the square,
    and the square shares 30% of its area with the rectangle, then the ratio of the rectangle's length
    to its width is 8. -/
theorem rectangle_square_overlap_ratio :
  ∀ (rect_area square_area overlap_area : ℝ) (rect_length rect_width : ℝ),
    rect_area > 0 →
    square_area > 0 →
    overlap_area > 0 →
    rect_length > 0 →
    rect_width > 0 →
    rect_area = rect_length * rect_width →
    overlap_area = 0.6 * rect_area →
    overlap_area = 0.3 * square_area →
    rect_length / rect_width = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l3839_383999


namespace NUMINAMATH_CALUDE_collinear_points_imply_a_equals_four_l3839_383996

-- Define the points
def A : ℝ × ℝ := (2, 2)
def B : ℝ → ℝ × ℝ := λ a => (a, 0)
def C : ℝ × ℝ := (0, 4)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, q.1 - p.1 = t * (r.1 - p.1) ∧ q.2 - p.2 = t * (r.2 - p.2)

-- Theorem statement
theorem collinear_points_imply_a_equals_four :
  ∀ a : ℝ, collinear A (B a) C → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_imply_a_equals_four_l3839_383996


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l3839_383911

/-- 
Given a cone with base radius √2 whose lateral surface can be unfolded into a semicircle,
prove that the length of its generatrix is 2√2.
-/
theorem cone_generatrix_length 
  (base_radius : ℝ) 
  (h_base_radius : base_radius = Real.sqrt 2) 
  (lateral_surface_is_semicircle : Bool) 
  (h_lateral_surface : lateral_surface_is_semicircle = true) : 
  ∃ (generatrix_length : ℝ), generatrix_length = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l3839_383911


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3839_383956

/-- A quadratic function with a symmetry axis at x = 2 -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The symmetry axis of f is at x = 2 -/
def symmetry_axis (b c : ℝ) : Prop := ∀ x : ℝ, f b c (2 - x) = f b c (2 + x)

theorem quadratic_inequality (b c : ℝ) (h : symmetry_axis b c) : 
  f b c 2 > f b c 1 ∧ f b c 1 > f b c 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3839_383956


namespace NUMINAMATH_CALUDE_w_squared_equals_one_fourth_l3839_383945

theorem w_squared_equals_one_fourth (w : ℝ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_equals_one_fourth_l3839_383945


namespace NUMINAMATH_CALUDE_empty_can_weight_is_two_l3839_383960

/-- Calculates the weight of each empty can given the total weight, number of soda cans, soda weight per can, and number of empty cans. -/
def empty_can_weight (total_weight : ℕ) (soda_cans : ℕ) (soda_weight_per_can : ℕ) (empty_cans : ℕ) : ℕ :=
  (total_weight - soda_cans * soda_weight_per_can) / (soda_cans + empty_cans)

/-- Proves that each empty can weighs 2 ounces given the problem conditions. -/
theorem empty_can_weight_is_two :
  empty_can_weight 88 6 12 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_empty_can_weight_is_two_l3839_383960


namespace NUMINAMATH_CALUDE_partition_6_4_l3839_383905

/-- The number of ways to partition n indistinguishable objects into at most k indistinguishable parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 9 ways to partition 6 indistinguishable objects into at most 4 indistinguishable parts -/
theorem partition_6_4 : partition_count 6 4 = 9 := by sorry

end NUMINAMATH_CALUDE_partition_6_4_l3839_383905


namespace NUMINAMATH_CALUDE_rhombus_area_from_square_circumference_l3839_383938

/-- The area of a rhombus formed by connecting the midpoints of a square's sides,
    given the square's circumference. -/
theorem rhombus_area_from_square_circumference (circumference : ℝ) :
  circumference = 96 →
  let square_side := circumference / 4
  let rhombus_area := square_side^2 / 2
  rhombus_area = 288 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_from_square_circumference_l3839_383938


namespace NUMINAMATH_CALUDE_kevin_repaired_phones_l3839_383915

/-- The number of phones Kevin repaired by the afternoon -/
def phones_repaired : ℕ := 3

/-- The initial number of phones Kevin had to repair -/
def initial_phones : ℕ := 15

/-- The number of phones dropped off by a client -/
def new_phones : ℕ := 6

/-- The number of phones each person (Kevin and his coworker) needs to repair -/
def phones_per_person : ℕ := 9

theorem kevin_repaired_phones :
  phones_repaired = 3 ∧
  initial_phones - phones_repaired + new_phones = 2 * phones_per_person :=
sorry

end NUMINAMATH_CALUDE_kevin_repaired_phones_l3839_383915


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3839_383932

theorem sum_of_x_and_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x^2 + y^2 = 1) (h4 : (3*x - 4*x^3) * (3*y - 4*y^3) = -1/2) : 
  x + y = Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3839_383932


namespace NUMINAMATH_CALUDE_first_term_of_constant_ratio_l3839_383949

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℕ) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (h : d = 5) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → arithmetic_sum a d (4 * n) / arithmetic_sum a d n = c) →
  a = 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_first_term_of_constant_ratio_l3839_383949


namespace NUMINAMATH_CALUDE_jordan_rectangle_length_l3839_383912

theorem jordan_rectangle_length :
  ∀ (carol_length carol_width jordan_width jordan_length : ℝ),
    carol_length = 5 →
    carol_width = 24 →
    jordan_width = 30 →
    carol_length * carol_width = jordan_width * jordan_length →
    jordan_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_jordan_rectangle_length_l3839_383912


namespace NUMINAMATH_CALUDE_net_gain_proof_l3839_383921

def initial_value : ℝ := 15000

def first_sale (value : ℝ) : ℝ := value * 1.2
def second_sale (value : ℝ) : ℝ := value * 0.85
def third_sale (value : ℝ) : ℝ := value * 1.1
def fourth_sale (value : ℝ) : ℝ := value * 0.95

def total_expense (initial : ℝ) : ℝ :=
  second_sale (first_sale initial) + fourth_sale (third_sale (second_sale (first_sale initial)))

def total_income (initial : ℝ) : ℝ :=
  first_sale initial + third_sale (second_sale (first_sale initial))

theorem net_gain_proof :
  total_income initial_value - total_expense initial_value = 3541.50 := by
  sorry

end NUMINAMATH_CALUDE_net_gain_proof_l3839_383921


namespace NUMINAMATH_CALUDE_candy_mixture_price_l3839_383981

theorem candy_mixture_price (price1 price2 : ℝ) (h1 : price1 = 10) (h2 : price2 = 15) : 
  let weight_ratio := 3
  let total_weight := weight_ratio + 1
  let total_cost := price1 * weight_ratio + price2
  total_cost / total_weight = 11.25 := by
sorry

end NUMINAMATH_CALUDE_candy_mixture_price_l3839_383981


namespace NUMINAMATH_CALUDE_power_difference_equals_seven_l3839_383962

theorem power_difference_equals_seven : 2^5 - 5^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equals_seven_l3839_383962


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3839_383970

theorem logarithm_expression_equality : 
  (Real.log (27^(1/2)) + Real.log 8 - 3 * Real.log (10^(1/2))) / Real.log 1.2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3839_383970


namespace NUMINAMATH_CALUDE_factor_implies_absolute_value_l3839_383985

theorem factor_implies_absolute_value (m n : ℤ) :
  (∀ x : ℝ, (x - 3) * (x + 4) ∣ (3 * x^3 - m * x + n)) →
  |3 * m - 2 * n| = 45 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_absolute_value_l3839_383985


namespace NUMINAMATH_CALUDE_pizza_topping_cost_l3839_383926

/-- The cost per pizza in dollars -/
def cost_per_pizza : ℚ := 10

/-- The number of pizzas ordered -/
def num_pizzas : ℕ := 3

/-- The total number of toppings across all pizzas -/
def total_toppings : ℕ := 4

/-- The tip amount in dollars -/
def tip : ℚ := 5

/-- The total cost of the order including tip in dollars -/
def total_cost : ℚ := 39

/-- The cost per topping in dollars -/
def cost_per_topping : ℚ := 1

theorem pizza_topping_cost :
  cost_per_pizza * num_pizzas + cost_per_topping * total_toppings + tip = total_cost :=
sorry

end NUMINAMATH_CALUDE_pizza_topping_cost_l3839_383926


namespace NUMINAMATH_CALUDE_field_length_calculation_l3839_383972

theorem field_length_calculation (w : ℝ) (l : ℝ) : 
  l = 2 * w →  -- length is double the width
  25 = (1/8) * (l * w) →  -- pond area (5^2) is 1/8 of field area
  l = 20 := by
  sorry

end NUMINAMATH_CALUDE_field_length_calculation_l3839_383972


namespace NUMINAMATH_CALUDE_train_speed_increase_time_l3839_383958

/-- The speed equation for a subway train -/
def speed_equation (s : ℝ) : ℝ := s^2 + 2*s

/-- Theorem: The time when the train's speed increases by 39 km/h from its speed at 4 seconds is 7 seconds -/
theorem train_speed_increase_time : 
  ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 7 ∧ 
  speed_equation s = speed_equation 4 + 39 ∧
  s = 7 := by
  sorry

#check train_speed_increase_time

end NUMINAMATH_CALUDE_train_speed_increase_time_l3839_383958


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3839_383925

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3839_383925


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l3839_383993

theorem cubic_sum_theorem (p q r : ℝ) 
  (sum_eq : p + q + r = 4)
  (sum_prod_eq : p * q + p * r + q * r = 6)
  (prod_eq : p * q * r = -8) : 
  p^3 + q^3 + r^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l3839_383993


namespace NUMINAMATH_CALUDE_angle_CDB_is_15_l3839_383901

/-- A triangle that shares a side with a rectangle -/
structure TriangleWithRectangle where
  /-- The length of the shared side -/
  side : ℝ
  /-- The triangle is equilateral -/
  equilateral : True
  /-- The adjacent side of the rectangle is perpendicular to the shared side -/
  perpendicular : True
  /-- The adjacent side of the rectangle is twice the length of the shared side -/
  adjacent_side : ℝ := 2 * side

/-- The measure of angle CDB in degrees -/
def angle_CDB (t : TriangleWithRectangle) : ℝ := 15

/-- Theorem: The measure of angle CDB is 15 degrees -/
theorem angle_CDB_is_15 (t : TriangleWithRectangle) : angle_CDB t = 15 := by sorry

end NUMINAMATH_CALUDE_angle_CDB_is_15_l3839_383901


namespace NUMINAMATH_CALUDE_rectangle_rotation_forms_cylinder_l3839_383933

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  width : ℝ
  height : ℝ
  width_positive : width > 0
  height_positive : height > 0

/-- Represents the solid formed by rotating a rectangle -/
inductive RotatedSolid
  | Cylinder
  | Other

/-- Function that determines the shape of the solid formed by rotating a rectangle -/
def rotate_rectangle (rect : Rectangle) : RotatedSolid := sorry

/-- Theorem stating that rotating a rectangle forms a cylinder -/
theorem rectangle_rotation_forms_cylinder (rect : Rectangle) :
  rotate_rectangle rect = RotatedSolid.Cylinder := by sorry

end NUMINAMATH_CALUDE_rectangle_rotation_forms_cylinder_l3839_383933


namespace NUMINAMATH_CALUDE_equation_system_solution_l3839_383902

theorem equation_system_solution (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - 2 * z = 0)
  (eq2 : x + 2 * y - 7 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 - 2*x*y) / (y^2 + 4*z^2) = -0.252 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3839_383902


namespace NUMINAMATH_CALUDE_magnitude_of_vector_2_neg1_l3839_383928

/-- The magnitude of a 2D vector (2, -1) is √5 -/
theorem magnitude_of_vector_2_neg1 :
  let a : Fin 2 → ℝ := ![2, -1]
  Real.sqrt ((a 0) ^ 2 + (a 1) ^ 2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_2_neg1_l3839_383928


namespace NUMINAMATH_CALUDE_used_computer_lifespan_l3839_383977

/-- Proves the lifespan of used computers given certain conditions -/
theorem used_computer_lifespan 
  (new_computer_cost : ℕ)
  (new_computer_lifespan : ℕ)
  (used_computer_cost : ℕ)
  (num_used_computers : ℕ)
  (savings : ℕ)
  (h1 : new_computer_cost = 600)
  (h2 : new_computer_lifespan = 6)
  (h3 : used_computer_cost = 200)
  (h4 : num_used_computers = 2)
  (h5 : savings = 200)
  (h6 : new_computer_cost - savings = num_used_computers * used_computer_cost) :
  ∃ (used_computer_lifespan : ℕ), 
    used_computer_lifespan * num_used_computers = new_computer_lifespan ∧ 
    used_computer_lifespan = 3 := by
  sorry


end NUMINAMATH_CALUDE_used_computer_lifespan_l3839_383977


namespace NUMINAMATH_CALUDE_cubic_sum_ratio_l3839_383904

theorem cubic_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 15)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 18 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_ratio_l3839_383904


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l3839_383917

/-- Given an initial angle of 50 degrees that is rotated 580 degrees clockwise,
    the resulting acute angle is 90 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℕ) : 
  initial_angle = 50 → 
  rotation = 580 → 
  (initial_angle + rotation) % 360 = 270 → 
  360 - ((initial_angle + rotation) % 360) = 90 :=
by sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l3839_383917


namespace NUMINAMATH_CALUDE_probability_theorem_l3839_383900

def total_balls : ℕ := 22
def red_balls : ℕ := 5
def blue_balls : ℕ := 6
def green_balls : ℕ := 7
def yellow_balls : ℕ := 4
def balls_picked : ℕ := 3

def probability_at_least_two_red_not_blue : ℚ :=
  (Nat.choose red_balls 2 * (green_balls + yellow_balls) +
   Nat.choose red_balls 3) /
  Nat.choose total_balls balls_picked

theorem probability_theorem :
  probability_at_least_two_red_not_blue = 12 / 154 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l3839_383900


namespace NUMINAMATH_CALUDE_parabola_line_intersection_theorem_l3839_383980

/-- Represents a parabola with focus on the x-axis and vertex at the origin -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := λ x y => y^2 = 2 * p * x

/-- Represents a line in the form x = my + b -/
structure Line where
  m : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := λ x y => x = m * y + b

/-- Theorem stating the existence of a specific line intersecting the parabola -/
theorem parabola_line_intersection_theorem (C : Parabola) (h1 : C.equation 2 1) :
  ∃ (l : Line), l.b = 2 ∧
    (∃ (A B : ℝ × ℝ),
      C.equation A.1 A.2 ∧
      C.equation B.1 B.2 ∧
      l.equation A.1 A.2 ∧
      l.equation B.1 B.2 ∧
      let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
      let N := (M.1, Real.sqrt (2 * C.p * M.1))
      (N.1 - A.1) * (N.1 - B.1) + (N.2 - A.2) * (N.2 - B.2) = 0) ∧
    (l.m = 2 ∨ l.m = -2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_theorem_l3839_383980


namespace NUMINAMATH_CALUDE_intersection_count_l3839_383942

-- Define the two curves
def curve1 (x y : ℝ) : Prop := (x + 2*y - 3) * (4*x - y + 1) = 0
def curve2 (x y : ℝ) : Prop := (2*x - y - 5) * (3*x + 4*y - 8) = 0

-- Define what it means for a point to be on both curves
def intersection_point (x y : ℝ) : Prop := curve1 x y ∧ curve2 x y

-- State the theorem
theorem intersection_count : 
  ∃ (points : Finset (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ points ↔ intersection_point p.1 p.2) ∧ 
    points.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l3839_383942


namespace NUMINAMATH_CALUDE_smallest_with_16_divisors_l3839_383988

def divisor_count (n : ℕ+) : ℕ := (Nat.divisors n.val).card

def has_16_divisors (n : ℕ+) : Prop := divisor_count n = 16

theorem smallest_with_16_divisors : 
  ∃ (n : ℕ+), has_16_divisors n ∧ ∀ (m : ℕ+), has_16_divisors m → n ≤ m :=
by
  use 216
  sorry

end NUMINAMATH_CALUDE_smallest_with_16_divisors_l3839_383988


namespace NUMINAMATH_CALUDE_increasing_function_integral_inequality_l3839_383964

open Set
open MeasureTheory
open Interval

theorem increasing_function_integral_inequality 
  (f : ℝ → ℝ) (hf : Continuous f) :
  (∀ (a b c : ℝ), a < b → b < c → 
    (c - b) * ∫ x in Icc a b, f x ≤ (b - a) * ∫ x in Icc b c, f x) ↔ 
  StrictMono f := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_integral_inequality_l3839_383964


namespace NUMINAMATH_CALUDE_linda_needs_two_more_batches_l3839_383923

/-- The number of additional batches of cookies Linda needs to bake --/
def additional_batches (classmates : ℕ) (cookies_per_student : ℕ) (dozens_per_batch : ℕ) 
  (chocolate_chip_batches : ℕ) (oatmeal_raisin_batches : ℕ) : ℕ :=
  let total_cookies_needed := classmates * cookies_per_student
  let cookies_per_batch := dozens_per_batch * 12
  let cookies_already_made := (chocolate_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let cookies_still_needed := total_cookies_needed - cookies_already_made
  (cookies_still_needed + cookies_per_batch - 1) / cookies_per_batch

/-- Theorem stating that Linda needs to bake 2 more batches of cookies --/
theorem linda_needs_two_more_batches :
  additional_batches 24 10 4 2 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_linda_needs_two_more_batches_l3839_383923


namespace NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_discount_l3839_383919

-- Define the successive discounts
def discount1 : ℝ := 0.10
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.25

-- Define the equivalent single discount
def equivalent_discount : ℝ := 0.426

-- Theorem statement
theorem successive_discounts_equivalent_to_single_discount :
  (1 - discount1) * (1 - discount2) * (1 - discount3) = 1 - equivalent_discount :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_discount_l3839_383919


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3839_383951

theorem largest_constant_inequality (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^3 + b^3 + c^3 + d^3)) ∧
  ∀ k > 2, ∃ a' b' c' d' : ℝ, 
    (0 ≤ a' ∧ a' ≤ 1) ∧ (0 ≤ b' ∧ b' ≤ 1) ∧ (0 ≤ c' ∧ c' ≤ 1) ∧ (0 ≤ d' ∧ d' ≤ 1) ∧
    (a'^2 * b' + b'^2 * c' + c'^2 * d' + d'^2 * a' + 4 < k * (a'^3 + b'^3 + c'^3 + d'^3)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3839_383951


namespace NUMINAMATH_CALUDE_not_invertible_sum_of_squares_l3839_383974

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem not_invertible_sum_of_squares (M N : Matrix n n ℝ) 
  (h_neq : M ≠ N) 
  (h_cube : M ^ 3 = N ^ 3) 
  (h_comm : M ^ 2 * N = N ^ 2 * M) : 
  ¬(IsUnit (M ^ 2 + N ^ 2)) := by
sorry

end NUMINAMATH_CALUDE_not_invertible_sum_of_squares_l3839_383974


namespace NUMINAMATH_CALUDE_min_three_digit_quotient_l3839_383971

theorem min_three_digit_quotient :
  ∃ (a b c : ℕ), 
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    (∀ (x y z : ℕ), x ≤ 9 → y ≤ 9 → z ≤ 9 →
      (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ) ≤ (100 * x + 10 * y + z : ℚ) / (x + y + z : ℚ)) ∧
    (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ) = 50.5 := by
  sorry

end NUMINAMATH_CALUDE_min_three_digit_quotient_l3839_383971


namespace NUMINAMATH_CALUDE_correct_num_arrangements_l3839_383975

/-- The number of different arrangements for 7 students in a row,
    where one student must stand in the center and two other students must stand together. -/
def num_arrangements : ℕ := 192

/-- The number of students -/
def total_students : ℕ := 7

/-- The number of students that must stand together (excluding the center student) -/
def students_together : ℕ := 2

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_num_arrangements :
  num_arrangements = 
    2 * (Nat.factorial students_together) * 
    (Nat.choose (total_students - 3) 1) * 
    (Nat.factorial 2) * 
    (Nat.factorial 3) :=
by
  sorry


end NUMINAMATH_CALUDE_correct_num_arrangements_l3839_383975


namespace NUMINAMATH_CALUDE_logarithmic_equation_solutions_l3839_383908

theorem logarithmic_equation_solutions (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, x > 0 → x ≠ 1 →
    ((3 * (Real.log x / Real.log a) - 2) * (Real.log a / Real.log x)^2 = Real.log x / (Real.log a / 2) - 3) ↔
    (x = 1/a ∨ x = Real.sqrt a ∨ x = a^2) :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solutions_l3839_383908


namespace NUMINAMATH_CALUDE_ruth_math_class_time_l3839_383941

/-- Represents Ruth's school schedule and math class time --/
structure RuthSchedule where
  hours_per_day : ℝ
  days_per_week : ℝ
  math_class_percentage : ℝ

/-- Calculates the number of hours Ruth spends in math class per week --/
def math_class_hours (schedule : RuthSchedule) : ℝ :=
  schedule.hours_per_day * schedule.days_per_week * schedule.math_class_percentage

/-- Theorem stating that Ruth spends 10 hours per week in math class --/
theorem ruth_math_class_time :
  ∃ (schedule : RuthSchedule),
    schedule.hours_per_day = 8 ∧
    schedule.days_per_week = 5 ∧
    schedule.math_class_percentage = 0.25 ∧
    math_class_hours schedule = 10 := by
  sorry

end NUMINAMATH_CALUDE_ruth_math_class_time_l3839_383941


namespace NUMINAMATH_CALUDE_function_symmetry_and_translation_l3839_383963

-- Define a function that represents a horizontal translation
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x ↦ f (x - h)

-- Define symmetry with respect to y-axis
def symmetric_to_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- State the theorem
theorem function_symmetry_and_translation :
  ∀ f : ℝ → ℝ,
  symmetric_to_y_axis (translate f 1) (λ x ↦ Real.exp x) →
  f = λ x ↦ Real.exp (-x - 1) :=
sorry

end NUMINAMATH_CALUDE_function_symmetry_and_translation_l3839_383963


namespace NUMINAMATH_CALUDE_slope_of_line_l3839_383935

/-- The slope of a line given by the equation x/4 + y/3 = 1 is -3/4 -/
theorem slope_of_line (x y : ℝ) : 
  (x / 4 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l3839_383935


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3839_383916

theorem sum_of_coefficients_equals_one (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^11 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                           a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3839_383916


namespace NUMINAMATH_CALUDE_limit_calculation_l3839_383957

open Real

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

theorem limit_calculation :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f (1 + Δx) - f (1 - 2*Δx)) / Δx + 3/exp 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_calculation_l3839_383957


namespace NUMINAMATH_CALUDE_triangle_side_length_l3839_383922

theorem triangle_side_length (X Y Z : ℝ) (x y z : ℝ) :
  y = 7 →
  z = 5 →
  Real.cos (Y - Z) = 21 / 32 →
  x^2 = 47.75 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3839_383922


namespace NUMINAMATH_CALUDE_rectangle_dimension_l3839_383924

/-- A rectangle with vertices at (0, 0), (0, 6), (x, 6), and (x, 0) has a perimeter of 40 units. -/
def rectangle_perimeter (x : ℝ) : Prop :=
  x > 0 ∧ 2 * (x + 6) = 40

/-- The value of x for which the rectangle has a perimeter of 40 units is 14. -/
theorem rectangle_dimension : ∃ x : ℝ, rectangle_perimeter x ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_l3839_383924


namespace NUMINAMATH_CALUDE_cheryl_same_color_probability_l3839_383939

def total_marbles : ℕ := 6
def red_marbles : ℕ := 3
def green_marbles : ℕ := 2
def yellow_marbles : ℕ := 1

def carol_draw : ℕ := 2
def claudia_draw : ℕ := 2
def cheryl_draw : ℕ := 2

theorem cheryl_same_color_probability :
  let total_outcomes := (total_marbles.choose carol_draw) * ((total_marbles - carol_draw).choose claudia_draw) * ((total_marbles - carol_draw - claudia_draw).choose cheryl_draw)
  let favorable_outcomes := red_marbles.choose cheryl_draw * ((total_marbles - cheryl_draw).choose carol_draw) * ((total_marbles - cheryl_draw - carol_draw).choose claudia_draw)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_same_color_probability_l3839_383939
