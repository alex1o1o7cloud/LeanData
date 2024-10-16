import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l2346_234603

theorem max_value_of_sum_of_square_roots (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) 
  (sum_constraint : a + b + c = 8) : 
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ 3 * Real.sqrt 26 ∧ 
  ∃ a' b' c' : ℝ, a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ a' + b' + c' = 8 ∧
  Real.sqrt (3 * a' + 2) + Real.sqrt (3 * b' + 2) + Real.sqrt (3 * c' + 2) = 3 * Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l2346_234603


namespace NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l2346_234656

/-- Proves that for a rectangle with breadth 10 meters and length 10 meters greater than its breadth,
    the ratio of its area to its breadth is 20:1. -/
theorem rectangle_area_breadth_ratio :
  ∀ (breadth length area : ℝ),
    breadth = 10 →
    length = breadth + 10 →
    area = length * breadth →
    area / breadth = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l2346_234656


namespace NUMINAMATH_CALUDE_earth_sun_distance_scientific_notation_l2346_234612

/-- The distance from Earth to Sun in kilometers -/
def earth_sun_distance : ℕ := 150000000

/-- Represents a number in scientific notation as a pair (coefficient, exponent) -/
def scientific_notation := ℝ × ℤ

/-- Converts a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : scientific_notation :=
  sorry

theorem earth_sun_distance_scientific_notation :
  to_scientific_notation earth_sun_distance = (1.5, 8) :=
sorry

end NUMINAMATH_CALUDE_earth_sun_distance_scientific_notation_l2346_234612


namespace NUMINAMATH_CALUDE_min_distinct_prime_factors_l2346_234605

theorem min_distinct_prime_factors (m n : ℕ) :
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  (p ∣ (m * (n + 9) * (m + 2 * n^2 + 3))) ∧
  (q ∣ (m * (n + 9) * (m + 2 * n^2 + 3))) :=
sorry

end NUMINAMATH_CALUDE_min_distinct_prime_factors_l2346_234605


namespace NUMINAMATH_CALUDE_triangle_equilateral_l2346_234629

-- Define a triangle with angles A, B, C
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define the property of angles forming an arithmetic sequence
def angles_arithmetic_sequence (t : Triangle) : Prop :=
  ∃ d : Real, t.B = t.A + d ∧ t.C = t.B + d

-- Define the property of log(sin) of angles forming an arithmetic sequence
def log_sin_arithmetic_sequence (t : Triangle) : Prop :=
  ∃ d : Real, Real.log (Real.sin t.B) = Real.log (Real.sin t.A) + d ∧
              Real.log (Real.sin t.C) = Real.log (Real.sin t.B) + d

-- Theorem statement
theorem triangle_equilateral (t : Triangle) 
  (h1 : angles_arithmetic_sequence t) 
  (h2 : log_sin_arithmetic_sequence t) : 
  t.A = Real.pi / 3 ∧ t.B = Real.pi / 3 ∧ t.C = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l2346_234629


namespace NUMINAMATH_CALUDE_find_a_and_b_l2346_234643

-- Define the sets A and B
def A : Set ℝ := {x | x^3 + 3*x^2 + 2*x > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ),
    (A ∩ B a b = {x | 0 < x ∧ x ≤ 2}) ∧
    (A ∪ B a b = {x | x > -2}) ∧
    a = -1 ∧
    b = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l2346_234643


namespace NUMINAMATH_CALUDE_fruit_condition_percentage_l2346_234620

theorem fruit_condition_percentage (oranges bananas : ℕ) 
  (rotten_oranges_percent rotten_bananas_percent : ℚ) :
  oranges = 600 →
  bananas = 400 →
  rotten_oranges_percent = 15 / 100 →
  rotten_bananas_percent = 8 / 100 →
  let total_fruits := oranges + bananas
  let rotten_oranges := (rotten_oranges_percent * oranges).num
  let rotten_bananas := (rotten_bananas_percent * bananas).num
  let total_rotten := rotten_oranges + rotten_bananas
  let good_fruits := total_fruits - total_rotten
  (good_fruits : ℚ) / total_fruits * 100 = 87.8 := by
sorry

end NUMINAMATH_CALUDE_fruit_condition_percentage_l2346_234620


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2346_234699

theorem inequality_solution_set (x : ℝ) : 
  (2 * x / 5 ≤ 3 + x ∧ 3 + x < 4 - x / 3) ↔ -5 ≤ x ∧ x < 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2346_234699


namespace NUMINAMATH_CALUDE_picture_frame_dimensions_l2346_234624

theorem picture_frame_dimensions (a b : ℕ+) : 
  (a : ℤ) * b = ((a + 2) * (b + 2) : ℤ) - a * b → 
  ((a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4)) :=
by sorry

end NUMINAMATH_CALUDE_picture_frame_dimensions_l2346_234624


namespace NUMINAMATH_CALUDE_even_decreasing_function_inequality_l2346_234615

noncomputable def e : ℝ := Real.exp 1

theorem even_decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = f (-x)) →
  (∀ x ≥ 0, ∀ y ≥ x, f x ≥ f y) →
  (∀ x ∈ Set.Icc 1 3, f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) →
  a ∈ Set.Icc (1 / e) ((2 + Real.log 3) / 3) :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_function_inequality_l2346_234615


namespace NUMINAMATH_CALUDE_park_area_theorem_l2346_234681

/-- Represents a rectangular park with a given perimeter where the width is one-third of the length -/
structure RectangularPark where
  perimeter : ℝ
  width : ℝ
  length : ℝ
  width_length_relation : width = length / 3
  perimeter_constraint : perimeter = 2 * (width + length)

/-- Calculates the area of a rectangular park -/
def parkArea (park : RectangularPark) : ℝ :=
  park.width * park.length

/-- Theorem stating that a rectangular park with a perimeter of 90 meters and width one-third of its length has an area of 379.6875 square meters -/
theorem park_area_theorem (park : RectangularPark) (h : park.perimeter = 90) : 
  parkArea park = 379.6875 := by
  sorry

end NUMINAMATH_CALUDE_park_area_theorem_l2346_234681


namespace NUMINAMATH_CALUDE_pizza_slice_difference_l2346_234651

/-- Given a pizza with 78 slices shared in a ratio of 5:8, prove that the difference
    between the waiter's share and 20 less than the waiter's share is 20 slices. -/
theorem pizza_slice_difference (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ) : 
  total_slices = 78 → 
  buzz_ratio = 5 → 
  waiter_ratio = 8 → 
  let waiter_share := (waiter_ratio * total_slices) / (buzz_ratio + waiter_ratio)
  waiter_share - (waiter_share - 20) = 20 :=
by sorry

end NUMINAMATH_CALUDE_pizza_slice_difference_l2346_234651


namespace NUMINAMATH_CALUDE_max_value_of_function_l2346_234622

theorem max_value_of_function (x : ℝ) (h : x < 0) :
  2 * x + 2 / x ≤ -4 ∧ ∃ x₀, x₀ < 0 ∧ 2 * x₀ + 2 / x₀ = -4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2346_234622


namespace NUMINAMATH_CALUDE_ellipse_triangle_problem_l2346_234698

-- Define the ellipse
def ellipse (x y : ℝ) (b : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

-- Define the line L
def line_L (x y : ℝ) : Prop := y = x + 2

-- Define parallel lines
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the problem statement
theorem ellipse_triangle_problem 
  (b : ℝ) 
  (ABC : Triangle) 
  (h_ellipse : ellipse ABC.A.1 ABC.A.2 b ∧ ellipse ABC.B.1 ABC.B.2 b)
  (h_C_on_L : line_L ABC.C.1 ABC.C.2)
  (h_AB_parallel_L : parallel ((ABC.B.2 - ABC.A.2) / (ABC.B.1 - ABC.A.1)) 1)
  (h_eccentricity : b^2 = 4/3) :
  (∀ (O : ℝ × ℝ), O = (0, 0) → (ABC.A.1 - O.1) * (ABC.B.2 - O.2) = (ABC.A.2 - O.2) * (ABC.B.1 - O.1) →
    (ABC.B.1 - ABC.A.1)^2 + (ABC.B.2 - ABC.A.2)^2 = 8 ∧ 
    (ABC.B.1 - ABC.A.1) * (ABC.C.2 - ABC.A.2) - (ABC.B.2 - ABC.A.2) * (ABC.C.1 - ABC.A.1) = 4) ∧
  (∀ (m : ℝ), (ABC.B.1 - ABC.A.1)^2 + (ABC.B.2 - ABC.A.2)^2 = (ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2 →
    (ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2 ≥ (ABC.C.1 - ABC.B.1)^2 + (ABC.C.2 - ABC.B.2)^2 →
    ABC.B.2 - ABC.A.2 = ABC.B.1 - ABC.A.1 - (ABC.B.1 - ABC.A.1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_problem_l2346_234698


namespace NUMINAMATH_CALUDE_min_value_cos_sin_l2346_234640

theorem min_value_cos_sin (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 3 * Real.pi / 2) :
  ∃ m : Real, m = -1/2 ∧ ∀ θ' : Real, 0 ≤ θ' ∧ θ' ≤ 3 * Real.pi / 2 →
    m ≤ Real.cos (θ' / 3) * (1 - Real.sin θ') :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_sin_l2346_234640


namespace NUMINAMATH_CALUDE_modular_inverse_37_mod_39_l2346_234674

theorem modular_inverse_37_mod_39 : 
  ∃ x : ℕ, x < 39 ∧ (37 * x) % 39 = 1 ∧ x = 19 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_37_mod_39_l2346_234674


namespace NUMINAMATH_CALUDE_result_line_properties_l2346_234654

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

/-- The given line equation -/
def given_line_eq (x y : ℝ) : Prop := 2*x + 3*y = 0

/-- The resulting line equation -/
def result_line_eq (x y : ℝ) : Prop := 3*x - 2*y + 7 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the resulting line passes through the center of the circle
    and is perpendicular to the given line -/
theorem result_line_properties :
  result_line_eq (circle_center.1) (circle_center.2) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), given_line_eq x₁ y₁ → given_line_eq x₂ y₂ →
    result_line_eq x₁ y₁ → result_line_eq x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * 2 + (y₂ - y₁) * 3) * ((x₂ - x₁) * 3 + (y₂ - y₁) * (-2)) = 0) :=
sorry

end NUMINAMATH_CALUDE_result_line_properties_l2346_234654


namespace NUMINAMATH_CALUDE_cover_rectangles_l2346_234626

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circle with a center point and radius -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- Returns the number of circles needed to cover a rectangle -/
def circles_to_cover (r : Rectangle) (circle_radius : ℝ) : ℕ :=
  sorry

theorem cover_rectangles :
  let r1 := Rectangle.mk 6 3
  let r2 := Rectangle.mk 5 3
  let circle_radius := Real.sqrt 2
  (circles_to_cover r1 circle_radius = 6) ∧
  (circles_to_cover r2 circle_radius = 5) := by
  sorry

end NUMINAMATH_CALUDE_cover_rectangles_l2346_234626


namespace NUMINAMATH_CALUDE_binary_110110011_to_octal_l2346_234671

def binary_to_octal (b : Nat) : Nat :=
  sorry

theorem binary_110110011_to_octal :
  binary_to_octal 110110011 = 163 := by
  sorry

end NUMINAMATH_CALUDE_binary_110110011_to_octal_l2346_234671


namespace NUMINAMATH_CALUDE_hypotenuse_length_l2346_234692

/-- A right triangle with specific medians -/
structure RightTriangleWithMedians where
  /-- First leg of the triangle -/
  a : ℝ
  /-- Second leg of the triangle -/
  b : ℝ
  /-- First median (from vertex of acute angle) -/
  m₁ : ℝ
  /-- Second median (from vertex of acute angle) -/
  m₂ : ℝ
  /-- The first median is 6 -/
  h₁ : m₁ = 6
  /-- The second median is 3√13 -/
  h₂ : m₂ = 3 * Real.sqrt 13
  /-- Relationship between first leg and first median -/
  h₃ : m₁^2 = a^2 + (3*b/2)^2
  /-- Relationship between second leg and second median -/
  h₄ : m₂^2 = b^2 + (3*a/2)^2

/-- The theorem stating that the hypotenuse of the triangle is 3√23 -/
theorem hypotenuse_length (t : RightTriangleWithMedians) : 
  Real.sqrt (9 * (t.a^2 + t.b^2)) = 3 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l2346_234692


namespace NUMINAMATH_CALUDE_average_sale_per_month_l2346_234693

def sales : List ℝ := [2435, 2920, 2855, 3230, 2560, 1000]

theorem average_sale_per_month :
  (sales.sum / sales.length : ℝ) = 2500 := by sorry

end NUMINAMATH_CALUDE_average_sale_per_month_l2346_234693


namespace NUMINAMATH_CALUDE_special_shape_is_regular_tetrahedron_l2346_234660

/-- A 3D shape with the property that the angle between diagonals of adjacent sides is 60 degrees -/
structure SpecialShape :=
  (is_3d : Bool)
  (diagonal_angle : ℝ)
  (angle_property : diagonal_angle = 60)

/-- Definition of a regular tetrahedron -/
structure RegularTetrahedron :=
  (is_3d : Bool)
  (num_faces : Nat)
  (face_type : String)
  (num_faces_property : num_faces = 4)
  (face_type_property : face_type = "equilateral triangle")

/-- Theorem stating that a SpecialShape is equivalent to a RegularTetrahedron -/
theorem special_shape_is_regular_tetrahedron (s : SpecialShape) : 
  ∃ (t : RegularTetrahedron), true :=
sorry

end NUMINAMATH_CALUDE_special_shape_is_regular_tetrahedron_l2346_234660


namespace NUMINAMATH_CALUDE_existence_of_equal_sums_l2346_234672

theorem existence_of_equal_sums (m n : ℕ) (a : Fin m → ℕ) (b : Fin n → ℕ) 
  (ha : ∀ i j : Fin m, i ≤ j → a i ≤ a j) 
  (hb : ∀ i j : Fin n, i ≤ j → b i ≤ b j)
  (ha_bound : ∀ i : Fin m, a i ≤ n)
  (hb_bound : ∀ i : Fin n, b i ≤ m) :
  ∃ (i : Fin m) (j : Fin n), a i + i.val + 1 = b j + j.val + 1 := by
sorry

end NUMINAMATH_CALUDE_existence_of_equal_sums_l2346_234672


namespace NUMINAMATH_CALUDE_triangle_properties_l2346_234634

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfies_equation (t : Triangle) : Prop :=
  (Real.sqrt 3 * t.c) / (t.b * Real.cos t.A) = Real.tan t.A + Real.tan t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h_acute : is_acute_triangle t)
  (h_eq : satisfies_equation t) :
  t.B = Real.pi/3 ∧ 
  (t.c = 4 → 2 * Real.sqrt 3 < (1/2 * t.a * t.c * Real.sin t.B) ∧ 
                (1/2 * t.a * t.c * Real.sin t.B) < 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2346_234634


namespace NUMINAMATH_CALUDE_vacant_seats_l2346_234657

def total_seats : ℕ := 600
def filled_percentage : ℚ := 45 / 100

theorem vacant_seats : 
  ⌊(1 - filled_percentage) * total_seats⌋ = 330 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l2346_234657


namespace NUMINAMATH_CALUDE_quadratic_symmetry_axis_l2346_234670

/-- A quadratic function of the form y = x^2 - bx + 2c with axis of symmetry x = 3 has b = 6 -/
theorem quadratic_symmetry_axis (b c : ℝ) : 
  (∀ x y : ℝ, y = x^2 - b*x + 2*c → (∀ y1 y2 : ℝ, (3 - x)^2 = (3 + x)^2 → y1 = y2)) → 
  b = 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_axis_l2346_234670


namespace NUMINAMATH_CALUDE_orange_harvest_duration_l2346_234614

/-- The number of sacks of oranges harvested per day -/
def sacks_per_day : ℕ := 14

/-- The total number of sacks of oranges harvested -/
def total_sacks : ℕ := 56

/-- The number of days the harvest lasts -/
def harvest_days : ℕ := total_sacks / sacks_per_day

theorem orange_harvest_duration :
  harvest_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_duration_l2346_234614


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2346_234623

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m α) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2346_234623


namespace NUMINAMATH_CALUDE_correct_regression_sequence_l2346_234608

/-- Represents the steps in linear regression analysis -/
inductive RegressionStep
  | InterpretEquation
  | CollectData
  | CalculateEquation
  | ComputeCorrelation
  | PlotScatterDiagram

/-- Represents a sequence of regression steps -/
def RegressionSequence := List RegressionStep

/-- The correct sequence of regression steps -/
def correctSequence : RegressionSequence := [
  RegressionStep.CollectData,
  RegressionStep.PlotScatterDiagram,
  RegressionStep.ComputeCorrelation,
  RegressionStep.CalculateEquation,
  RegressionStep.InterpretEquation
]

/-- Predicate to check if a sequence is valid for determining linear relationship -/
def isValidSequence (seq : RegressionSequence) : Prop := 
  seq = correctSequence

/-- Theorem stating that the correct sequence is valid for linear regression analysis -/
theorem correct_regression_sequence : 
  isValidSequence correctSequence := by sorry

end NUMINAMATH_CALUDE_correct_regression_sequence_l2346_234608


namespace NUMINAMATH_CALUDE_min_value_theorem_l2346_234617

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9*x + 1/x^6 ≥ 10 ∧ (9*x + 1/x^6 = 10 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2346_234617


namespace NUMINAMATH_CALUDE_range_of_shifted_f_l2346_234665

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x, f x ∈ Set.Icc 1 2)
variable (h2 : Set.range f = Set.Icc 1 2)

-- State the theorem
theorem range_of_shifted_f :
  Set.range (fun x ↦ f (x + 1)) = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_shifted_f_l2346_234665


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l2346_234663

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l2346_234663


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2346_234618

theorem rationalize_denominator : 15 / Real.sqrt 45 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2346_234618


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l2346_234628

theorem unique_solution_quadratic_equation :
  ∃! (x y z v : ℝ),
    x^2 + y^2 + z^2 + v^2 - x*y - y*z - z*v - v + 2/5 = 0 ∧
    x = 1/5 ∧ y = 2/5 ∧ z = 3/5 ∧ v = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l2346_234628


namespace NUMINAMATH_CALUDE_product_inspection_problem_l2346_234695

def total_products : ℕ := 100
def defective_products : ℕ := 3
def drawn_products : ℕ := 4
def defective_in_sample : ℕ := 2

theorem product_inspection_problem :
  (Nat.choose defective_products defective_in_sample) *
  (Nat.choose (total_products - defective_products) (drawn_products - defective_in_sample)) = 13968 := by
  sorry

end NUMINAMATH_CALUDE_product_inspection_problem_l2346_234695


namespace NUMINAMATH_CALUDE_no_solution_to_system_l2346_234627

theorem no_solution_to_system :
  ¬ ∃ (x y z : ℝ), 
    (3 * x - 4 * y + z = 10) ∧ 
    (6 * x - 8 * y + 2 * z = 16) ∧ 
    (x + y - z = 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l2346_234627


namespace NUMINAMATH_CALUDE_chord_angle_cosine_l2346_234625

theorem chord_angle_cosine (r : ℝ) (α β : ℝ) : 
  r > 0 ∧ 
  2 * r * Real.sin (α / 2) = 2 ∧
  2 * r * Real.sin (β / 2) = 3 ∧
  2 * r * Real.sin ((α + β) / 2) = 4 ∧
  α + β < π →
  Real.cos α = 17 / 32 := by
sorry

end NUMINAMATH_CALUDE_chord_angle_cosine_l2346_234625


namespace NUMINAMATH_CALUDE_polar_coordinate_equivalence_l2346_234683

/-- Given a point in polar coordinates (-5, 5π/6), prove that it is equivalent to (5, 11π/6) in standard polar coordinate representation. -/
theorem polar_coordinate_equivalence :
  let given_point : ℝ × ℝ := (-5, 5 * Real.pi / 6)
  let standard_point : ℝ × ℝ := (5, 11 * Real.pi / 6)
  (∀ (r θ : ℝ), r > 0 → 0 ≤ θ → θ < 2 * Real.pi →
    (r * (Real.cos θ), r * (Real.sin θ)) =
    (given_point.1 * (Real.cos given_point.2), given_point.1 * (Real.sin given_point.2))) →
  (standard_point.1 * (Real.cos standard_point.2), standard_point.1 * (Real.sin standard_point.2)) =
  (given_point.1 * (Real.cos given_point.2), given_point.1 * (Real.sin given_point.2)) :=
by sorry


end NUMINAMATH_CALUDE_polar_coordinate_equivalence_l2346_234683


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2346_234613

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 - 12) / a = (b^3 - 12) / b ∧ (b^3 - 12) / b = (c^3 - 12) / c) : 
  a^3 + b^3 + c^3 = 36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2346_234613


namespace NUMINAMATH_CALUDE_two_and_one_third_symbiotic_neg_one_third_and_neg_two_symbiotic_symbiotic_pair_negation_l2346_234647

-- Definition of symbiotic rational number pair
def is_symbiotic_pair (a b : ℚ) : Prop := a - b = a * b + 1

-- Theorem 1: (2, 1/3) is a symbiotic rational number pair
theorem two_and_one_third_symbiotic : is_symbiotic_pair 2 (1/3) := by sorry

-- Theorem 2: (-1/3, -2) is a symbiotic rational number pair
theorem neg_one_third_and_neg_two_symbiotic : is_symbiotic_pair (-1/3) (-2) := by sorry

-- Theorem 3: If (m, n) is a symbiotic rational number pair, then (-n, -m) is also a symbiotic rational number pair
theorem symbiotic_pair_negation (m n : ℚ) : 
  is_symbiotic_pair m n → is_symbiotic_pair (-n) (-m) := by sorry

end NUMINAMATH_CALUDE_two_and_one_third_symbiotic_neg_one_third_and_neg_two_symbiotic_symbiotic_pair_negation_l2346_234647


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l2346_234696

theorem ratio_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y = 7 * (x - y)) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l2346_234696


namespace NUMINAMATH_CALUDE_equation_solver_l2346_234659

theorem equation_solver (a b x y : ℝ) (h1 : x^2 / y + y^2 / x = a) (h2 : x / y + y / x = b) :
  (x = (a * (b + 2 + Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2)) ∧
   y = (a * (b + 2 - Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2))) ∨
  (x = (a * (b + 2 - Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2)) ∧
   y = (a * (b + 2 + Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solver_l2346_234659


namespace NUMINAMATH_CALUDE_license_plate_combinations_count_l2346_234661

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of choices for the last character (letters + digits) -/
def last_char_choices : ℕ := num_letters + num_digits

/-- A function to calculate the number of valid license plate combinations -/
def license_plate_combinations : ℕ :=
  num_letters * last_char_choices * 2

/-- Theorem stating that the number of valid license plate combinations is 1872 -/
theorem license_plate_combinations_count :
  license_plate_combinations = 1872 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_count_l2346_234661


namespace NUMINAMATH_CALUDE_train_length_l2346_234680

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 180 → time = 8 → speed * time * (1000 / 3600) = 400 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2346_234680


namespace NUMINAMATH_CALUDE_calculation_proof_l2346_234632

theorem calculation_proof : 211 * 555 + 445 * 789 + 555 * 789 + 211 * 445 = 10^6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2346_234632


namespace NUMINAMATH_CALUDE_typing_orders_count_l2346_234669

/-- Represents the order of letters delivered by the boss -/
def letterOrder : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Represents that letter 8 has been typed -/
def letter8Typed : Nat := 8

/-- The number of letters that can be either typed or not typed after lunch -/
def remainingLetters : Nat := 8

/-- Theorem: The number of possible after-lunch typing orders is 2^8 = 256 -/
theorem typing_orders_count : 
  (2 : Nat) ^ remainingLetters = 256 := by
  sorry

#check typing_orders_count

end NUMINAMATH_CALUDE_typing_orders_count_l2346_234669


namespace NUMINAMATH_CALUDE_printer_time_345_pages_l2346_234649

/-- The time (in minutes) it takes to print a given number of pages at a given rate -/
def print_time (pages : ℕ) (rate : ℕ) : ℚ :=
  pages / rate

theorem printer_time_345_pages : 
  let pages := 345
  let rate := 23
  Int.floor (print_time pages rate) = 15 := by
  sorry

end NUMINAMATH_CALUDE_printer_time_345_pages_l2346_234649


namespace NUMINAMATH_CALUDE_petri_dishes_count_l2346_234650

/-- The number of petri dishes in a biology lab -/
def number_of_petri_dishes : ℕ :=
  let total_germs : ℕ := 3600  -- 0.036 * 10^5 = 3600
  let germs_per_dish : ℕ := 80 -- Approximating 79.99999999999999 to 80
  total_germs / germs_per_dish

theorem petri_dishes_count : number_of_petri_dishes = 45 := by
  sorry

end NUMINAMATH_CALUDE_petri_dishes_count_l2346_234650


namespace NUMINAMATH_CALUDE_sum_of_cubes_plus_one_divisible_by_5_l2346_234601

def sum_of_cubes_plus_one (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => (i + 1)^3 + 1)

theorem sum_of_cubes_plus_one_divisible_by_5 :
  5 ∣ sum_of_cubes_plus_one 50 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_plus_one_divisible_by_5_l2346_234601


namespace NUMINAMATH_CALUDE_equation_solution_l2346_234686

theorem equation_solution : 
  {x : ℝ | x + 60 / (x - 5) = -12} = {0, -7} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2346_234686


namespace NUMINAMATH_CALUDE_boat_travel_distance_l2346_234677

/-- Proves that a boat traveling upstream and downstream with given conditions travels 91.25 miles -/
theorem boat_travel_distance (v : ℝ) (d : ℝ) : 
  d / (v - 3) = d / (v + 3) + 0.5 →
  d / (v + 3) = 2.5191640969412834 →
  d = 91.25 := by
  sorry

end NUMINAMATH_CALUDE_boat_travel_distance_l2346_234677


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2346_234667

/-- Proves that mixing equal volumes of 10% and 30% alcohol solutions results in a 20% solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 200
  let y_volume : ℝ := 200
  let x_concentration : ℝ := 0.1
  let y_concentration : ℝ := 0.3
  let target_concentration : ℝ := 0.2
  x_volume * x_concentration + y_volume * y_concentration = 
    (x_volume + y_volume) * target_concentration :=
by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2346_234667


namespace NUMINAMATH_CALUDE_barry_fifth_game_yards_l2346_234641

theorem barry_fifth_game_yards (game1 game2 game3 game4 game6 : ℕ) 
  (h1 : game1 = 98)
  (h2 : game2 = 107)
  (h3 : game3 = 85)
  (h4 : game4 = 89)
  (h5 : game6 ≥ 130)
  (h6 : (game1 + game2 + game3 + game4 + game6 : ℚ) / 6 > 100) :
  ∃ game5 : ℕ, game5 = 91 ∧ (game1 + game2 + game3 + game4 + game5 + game6 : ℚ) / 6 > 100 := by
sorry

end NUMINAMATH_CALUDE_barry_fifth_game_yards_l2346_234641


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2346_234637

/-- Given a geometric sequence {a_n} with positive terms where a_4 * a_10 = 16, prove a_7 = 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ r : ℝ, ∀ n, a (n + 1) = r * a n)
  (h_product : a 4 * a 10 = 16) : 
  a 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2346_234637


namespace NUMINAMATH_CALUDE_brookes_science_problems_l2346_234687

/-- Represents the number of problems and time for each subject in Brooke's homework --/
structure Homework where
  math_problems : ℕ
  social_studies_problems : ℕ
  science_problems : ℕ
  math_time_per_problem : ℚ
  social_studies_time_per_problem : ℚ
  science_time_per_problem : ℚ
  total_time : ℚ

/-- Calculates the total time spent on homework --/
def total_homework_time (hw : Homework) : ℚ :=
  hw.math_problems * hw.math_time_per_problem +
  hw.social_studies_problems * hw.social_studies_time_per_problem +
  hw.science_problems * hw.science_time_per_problem

/-- Theorem stating that Brooke has 10 science problems --/
theorem brookes_science_problems (hw : Homework)
  (h1 : hw.math_problems = 15)
  (h2 : hw.social_studies_problems = 6)
  (h3 : hw.math_time_per_problem = 2)
  (h4 : hw.social_studies_time_per_problem = 1/2)
  (h5 : hw.science_time_per_problem = 3/2)
  (h6 : hw.total_time = 48)
  (h7 : total_homework_time hw = hw.total_time) :
  hw.science_problems = 10 := by
  sorry


end NUMINAMATH_CALUDE_brookes_science_problems_l2346_234687


namespace NUMINAMATH_CALUDE_bond_face_value_l2346_234631

/-- Proves that the face value of a bond is 5000 given specific conditions --/
theorem bond_face_value (F : ℝ) : 
  (0.10 * F = 0.065 * 7692.307692307692) → F = 5000 := by
  sorry

end NUMINAMATH_CALUDE_bond_face_value_l2346_234631


namespace NUMINAMATH_CALUDE_bowling_ball_volume_l2346_234688

/-- The volume of a sphere with cylindrical holes drilled into it -/
theorem bowling_ball_volume (d : ℝ) (r1 r2 r3 h1 h2 h3 : ℝ) : 
  d = 36 → r1 = 1 → r2 = 1 → r3 = 2 → h1 = 9 → h2 = 10 → h3 = 9 → 
  (4 / 3 * π * (d / 2)^3) - (π * r1^2 * h1) - (π * r2^2 * h2) - (π * r3^2 * h3) = 7721 * π := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_volume_l2346_234688


namespace NUMINAMATH_CALUDE_composite_8n_plus_3_l2346_234685

theorem composite_8n_plus_3 (n : ℕ) (x y : ℕ) 
  (h1 : 8 * n + 1 = x^2) 
  (h2 : 24 * n + 1 = y^2) 
  (h3 : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 8 * n + 3 = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_8n_plus_3_l2346_234685


namespace NUMINAMATH_CALUDE_cone_height_ratio_l2346_234662

/-- Proves the ratio of heights for a cone with reduced height and constant base --/
theorem cone_height_ratio (original_height : ℝ) (base_circumference : ℝ) (shorter_volume : ℝ) :
  original_height = 15 →
  base_circumference = 10 * Real.pi →
  shorter_volume = 50 * Real.pi →
  ∃ (shorter_height : ℝ),
    (1 / 3) * Real.pi * (base_circumference / (2 * Real.pi))^2 * shorter_height = shorter_volume ∧
    shorter_height / original_height = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l2346_234662


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2346_234635

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem stating the general formula for the n-th term of the sequence -/
theorem arithmetic_sequence_formula (seq : ArithmeticSequence) 
  (h3 : (S seq 3)^2 = 9 * (S seq 2))
  (h4 : S seq 4 = 4 * (S seq 2)) :
  ∀ n : ℕ, seq.a n = (4 : ℚ) / 9 * (2 * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2346_234635


namespace NUMINAMATH_CALUDE_product_ratio_equals_one_l2346_234648

theorem product_ratio_equals_one
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_equals_one_l2346_234648


namespace NUMINAMATH_CALUDE_cube_root_of_cube_l2346_234691

theorem cube_root_of_cube (x : ℝ) : x^(1/3)^3 = x := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_cube_l2346_234691


namespace NUMINAMATH_CALUDE_range_of_z_l2346_234606

-- Define the variables and their constraints
def a : ℝ := sorry
def b : ℝ := sorry

-- Define the function z
def z (a b : ℝ) : ℝ := 2 * a - b

-- State the theorem
theorem range_of_z :
  (2 < a ∧ a < 3) → (-2 < b ∧ b < -1) →
  ∀ z₀ : ℝ, (∃ a₀ b₀ : ℝ, (2 < a₀ ∧ a₀ < 3) ∧ (-2 < b₀ ∧ b₀ < -1) ∧ z a₀ b₀ = z₀) ↔ (5 < z₀ ∧ z₀ < 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_l2346_234606


namespace NUMINAMATH_CALUDE_sqrt_sum_quotient_l2346_234668

theorem sqrt_sum_quotient : (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 48 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_quotient_l2346_234668


namespace NUMINAMATH_CALUDE_range_of_m_l2346_234619

def f (x : ℝ) := x^2 - 4*x + 5

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f x ≤ 10) ∧
  (∃ x ∈ Set.Icc (-1) m, f x = 10) ∧
  (∀ x ∈ Set.Icc (-1) m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) m, f x = 1) →
  m ∈ Set.Icc 2 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2346_234619


namespace NUMINAMATH_CALUDE_binary_111011_equals_59_l2346_234633

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_111011_equals_59 :
  binary_to_decimal [true, true, false, true, true, true] = 59 := by
  sorry

end NUMINAMATH_CALUDE_binary_111011_equals_59_l2346_234633


namespace NUMINAMATH_CALUDE_function_range_contained_in_unit_interval_l2346_234689

/-- Given a function f: ℝ → ℝ satisfying (f x)^2 ≤ f y for all x > y,
    prove that the range of f is contained in [0, 1]. -/
theorem function_range_contained_in_unit_interval
  (f : ℝ → ℝ) (h : ∀ x y, x > y → (f x)^2 ≤ f y) :
  ∀ x, 0 ≤ f x ∧ f x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_contained_in_unit_interval_l2346_234689


namespace NUMINAMATH_CALUDE_purchase_system_of_equations_l2346_234621

/-- Represents the purchase of basketballs and soccer balls -/
structure PurchaseInfo where
  basketball_price : ℝ
  soccer_ball_price : ℝ
  basketball_count : ℕ
  soccer_ball_count : ℕ
  total_cost : ℝ
  price_difference : ℝ

/-- The system of equations for the purchase -/
def purchase_equations (p : PurchaseInfo) : Prop :=
  p.basketball_count * p.basketball_price + p.soccer_ball_count * p.soccer_ball_price = p.total_cost ∧
  p.basketball_price - p.soccer_ball_price = p.price_difference

theorem purchase_system_of_equations (p : PurchaseInfo) 
  (h1 : p.basketball_count = 3)
  (h2 : p.soccer_ball_count = 2)
  (h3 : p.total_cost = 474)
  (h4 : p.price_difference = 8) :
  purchase_equations p ↔ 
  (3 * p.basketball_price + 2 * p.soccer_ball_price = 474 ∧
   p.basketball_price - p.soccer_ball_price = 8) :=
by sorry

end NUMINAMATH_CALUDE_purchase_system_of_equations_l2346_234621


namespace NUMINAMATH_CALUDE_tom_reading_speed_l2346_234655

/-- Given that Tom reads 10 hours over 5 days, reads the same amount every day,
    and reads 700 pages in 7 days, prove that he can read 50 pages per hour. -/
theorem tom_reading_speed :
  ∀ (total_hours : ℕ) (days : ℕ) (total_pages : ℕ) (week_days : ℕ),
    total_hours = 10 →
    days = 5 →
    total_pages = 700 →
    week_days = 7 →
    (total_hours / days) * week_days ≠ 0 →
    total_pages / ((total_hours / days) * week_days) = 50 := by
  sorry

end NUMINAMATH_CALUDE_tom_reading_speed_l2346_234655


namespace NUMINAMATH_CALUDE_expression_evaluation_l2346_234673

theorem expression_evaluation : 
  let f (x : ℝ) := (x - 1) / (x + 1)
  let expr (x : ℝ) := (f x + 1) / (f x - 1)
  expr 2 = -2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2346_234673


namespace NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l2346_234676

/-- The coefficient of x^5 in the expansion of (x^3 + 1/x)^7 is 35 -/
theorem coefficient_x5_in_expansion : Nat := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l2346_234676


namespace NUMINAMATH_CALUDE_coefficient_x2_implies_a_eq_2_l2346_234607

/-- The coefficient of x^2 in the expansion of (x+a)^5 -/
def coefficient_x2 (a : ℝ) : ℝ := 10 * a^3

/-- Theorem stating that if the coefficient of x^2 in (x+a)^5 is 80, then a = 2 -/
theorem coefficient_x2_implies_a_eq_2 :
  coefficient_x2 2 = 80 ∧ (∀ a : ℝ, coefficient_x2 a = 80 → a = 2) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x2_implies_a_eq_2_l2346_234607


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l2346_234684

theorem sum_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l2346_234684


namespace NUMINAMATH_CALUDE_integral_sin_cos_sin_l2346_234664

open Real

theorem integral_sin_cos_sin (x : ℝ) :
  ∃ C : ℝ, ∫ t, sin t * cos (2*t) * sin (5*t) = 
    (1/24) * sin (6*x) - (1/32) * sin (8*x) - (1/8) * sin (2*x) + (1/16) * sin (4*x) + C :=
by
  sorry

end NUMINAMATH_CALUDE_integral_sin_cos_sin_l2346_234664


namespace NUMINAMATH_CALUDE_emily_necklaces_l2346_234697

def beads_per_necklace : ℕ := 5
def total_beads_used : ℕ := 20

theorem emily_necklaces :
  total_beads_used / beads_per_necklace = 4 :=
by sorry

end NUMINAMATH_CALUDE_emily_necklaces_l2346_234697


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2346_234609

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 13) 
  (h2 : x*y + x*z + y*z = 32) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 949 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2346_234609


namespace NUMINAMATH_CALUDE_sum_of_products_power_inequality_l2346_234602

theorem sum_of_products_power_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  (a * b) ^ (5/4 : ℝ) + (b * c) ^ (5/4 : ℝ) + (c * a) ^ (5/4 : ℝ) < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_power_inequality_l2346_234602


namespace NUMINAMATH_CALUDE_jiuquan_location_accuracy_l2346_234611

-- Define the possible location descriptions
inductive LocationDescription
  | NorthwestOfBeijing
  | LatitudeOnly (lat : Float)
  | LongitudeOnly (long : Float)
  | LatitudeLongitude (lat : Float) (long : Float)

-- Define the accuracy of a location description
def isAccurateLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.LatitudeLongitude _ _ => True
  | _ => False

-- Theorem statement
theorem jiuquan_location_accuracy :
  ∀ (desc : LocationDescription),
    isAccurateLocation desc ↔
      desc = LocationDescription.LatitudeLongitude 39.75 98.52 :=
by sorry

end NUMINAMATH_CALUDE_jiuquan_location_accuracy_l2346_234611


namespace NUMINAMATH_CALUDE_one_ounce_bottle_caps_count_l2346_234636

/-- The number of one-ounce bottle caps in a collection -/
def oneOunceBottleCaps (totalWeight : ℕ) (totalCaps : ℕ) : ℕ :=
  totalWeight * 16

/-- Theorem: The number of one-ounce bottle caps is equal to the total weight in ounces -/
theorem one_ounce_bottle_caps_count 
  (totalWeight : ℕ) 
  (totalCaps : ℕ) 
  (h1 : totalWeight = 18) 
  (h2 : totalCaps = 2016) : 
  oneOunceBottleCaps totalWeight totalCaps = totalWeight * 16 :=
by sorry

end NUMINAMATH_CALUDE_one_ounce_bottle_caps_count_l2346_234636


namespace NUMINAMATH_CALUDE_solve_money_problem_l2346_234690

def money_problem (mildred_spent candice_spent amount_left : ℕ) : Prop :=
  let total_spent := mildred_spent + candice_spent
  let mom_gave := total_spent + amount_left
  mom_gave = mildred_spent + candice_spent + amount_left

theorem solve_money_problem :
  ∀ (mildred_spent candice_spent amount_left : ℕ),
  money_problem mildred_spent candice_spent amount_left :=
by
  sorry

end NUMINAMATH_CALUDE_solve_money_problem_l2346_234690


namespace NUMINAMATH_CALUDE_find_divisor_l2346_234658

theorem find_divisor (n m d : ℕ) (h1 : n - 7 = m) (h2 : m % d = 0) (h3 : ∀ k < 7, (n - k) % d ≠ 0) : d = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2346_234658


namespace NUMINAMATH_CALUDE_martha_cards_l2346_234646

/-- The number of cards Martha ends up with after receiving more cards -/
def final_cards (start : ℕ) (received : ℕ) : ℕ :=
  start + received

/-- Theorem stating that Martha ends up with 79 cards -/
theorem martha_cards : final_cards 3 76 = 79 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l2346_234646


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l2346_234638

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 28)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  : (throwers + ((total_players - throwers) * 2) / 3) = 56 := by
  sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l2346_234638


namespace NUMINAMATH_CALUDE_total_players_l2346_234666

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) : 
  kabadi = 10 → kho_kho_only = 35 → both = 5 → kabadi + kho_kho_only - both = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l2346_234666


namespace NUMINAMATH_CALUDE_additional_workers_for_wall_project_l2346_234639

/-- Calculates the number of additional workers needed to complete a project on time -/
def additional_workers_needed (total_days : ℕ) (initial_workers : ℕ) (days_passed : ℕ) (work_completed_percentage : ℚ) : ℕ :=
  let total_work := total_days * initial_workers
  let remaining_work := total_work * (1 - work_completed_percentage)
  let remaining_days := total_days - days_passed
  let work_by_existing := initial_workers * remaining_days
  let additional_work_needed := remaining_work - work_by_existing
  (additional_work_needed / remaining_days).ceil.toNat

/-- Proves that given the initial conditions, 12 additional workers are needed -/
theorem additional_workers_for_wall_project : 
  additional_workers_needed 50 60 25 (2/5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_workers_for_wall_project_l2346_234639


namespace NUMINAMATH_CALUDE_probability_all_white_is_correct_l2346_234644

def total_balls : ℕ := 18
def white_balls : ℕ := 8
def black_balls : ℕ := 10
def drawn_balls : ℕ := 7

def probability_all_white : ℚ :=
  (Nat.choose white_balls drawn_balls) / (Nat.choose total_balls drawn_balls)

theorem probability_all_white_is_correct :
  probability_all_white = 1 / 3980 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_white_is_correct_l2346_234644


namespace NUMINAMATH_CALUDE_polygon_sides_l2346_234694

theorem polygon_sides (n : ℕ) (n_pos : n > 0) :
  (((n - 2) * 180) / n = 108) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2346_234694


namespace NUMINAMATH_CALUDE_fruit_salad_composition_l2346_234679

/-- Fruit salad composition problem -/
theorem fruit_salad_composition (total : ℕ) (b r g c : ℕ) : 
  total = 360 ∧ 
  r = 3 * b ∧ 
  g = 4 * c ∧ 
  c = 5 * r ∧ 
  total = b + r + g + c → 
  c = 68 := by
sorry

end NUMINAMATH_CALUDE_fruit_salad_composition_l2346_234679


namespace NUMINAMATH_CALUDE_translation_result_l2346_234604

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation operation
def translate (p : Point) (dx dy : ℝ) : Point :=
  (p.1 + dx, p.2 + dy)

-- Theorem statement
theorem translation_result :
  let A : Point := (-1, 4)
  let B : Point := translate A 5 3
  B = (4, 7) := by sorry

end NUMINAMATH_CALUDE_translation_result_l2346_234604


namespace NUMINAMATH_CALUDE_prime_divisibility_l2346_234652

theorem prime_divisibility (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r →
  (pqr : ℕ) = p * q * r →
  (pqr ∣ (p * q)^r + (q * r)^p + (r * p)^q - 1) →
  ((pqr)^3 ∣ 3 * ((p * q)^r + (q * r)^p + (r * p)^q - 1)) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2346_234652


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2346_234678

-- Problem 1
theorem problem_1 (a b : ℚ) (h1 : a = 2) (h2 : b = 1/3) :
  3 * (a^2 - a*b + 7) - 2 * (3*a*b - a^2 + 1) + 3 = 36 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h : (x + 2)^2 + |y - 1/2| = 0) :
  5*x^2 - (2*x*y - 3*(1/3*x*y + 2) + 4*x^2) = 11 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2346_234678


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l2346_234653

/-- The smallest positive integer divisible by all positive integers less than 8 -/
def M : ℕ := Nat.lcm 7 (Nat.lcm 6 (Nat.lcm 5 (Nat.lcm 4 (Nat.lcm 3 (Nat.lcm 2 1)))))

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_M :
  sum_of_digits M = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l2346_234653


namespace NUMINAMATH_CALUDE_divisor_count_equals_equation_solutions_l2346_234600

/-- The prime factorization of 2310 -/
def prime_factors : List Nat := [2, 3, 5, 7, 11]

/-- The exponent of 2310 in the number we're considering -/
def exponent : Nat := 2310

/-- A function that counts the number of positive integer divisors of n^exponent 
    that are divisible by exactly 48 positive integers, 
    where n is the product of the prime factors -/
def count_specific_divisors (prime_factors : List Nat) (exponent : Nat) : Nat :=
  sorry

/-- A function that counts the number of solutions (a,b,c,d,e) to the equation 
    (a+1)(b+1)(c+1)(d+1)(e+1) = 48, where a,b,c,d,e are non-negative integers -/
def count_equation_solutions : Nat :=
  sorry

/-- The main theorem stating the equality of the two counting functions -/
theorem divisor_count_equals_equation_solutions : 
  count_specific_divisors prime_factors exponent = count_equation_solutions :=
  sorry

end NUMINAMATH_CALUDE_divisor_count_equals_equation_solutions_l2346_234600


namespace NUMINAMATH_CALUDE_laura_weekly_mileage_l2346_234610

/-- Represents the total miles driven by Laura in a week -/
def total_miles_per_week (
  house_school_round_trip : ℕ)
  (supermarket_extra_distance : ℕ)
  (gym_distance : ℕ)
  (friend_distance : ℕ)
  (workplace_distance : ℕ)
  (school_days : ℕ)
  (supermarket_trips : ℕ)
  (gym_trips : ℕ)
  (friend_trips : ℕ) : ℕ :=
  -- Weekday trips (work and school)
  (workplace_distance + (house_school_round_trip / 2 - workplace_distance) + (house_school_round_trip / 2)) * school_days +
  -- Supermarket trips
  ((house_school_round_trip / 2 + supermarket_extra_distance) * 2) * supermarket_trips +
  -- Gym trips
  (gym_distance * 2) * gym_trips +
  -- Friend's house trips
  (friend_distance * 2) * friend_trips

/-- Theorem stating that Laura drives 234 miles per week -/
theorem laura_weekly_mileage :
  total_miles_per_week 20 10 5 12 8 5 2 3 1 = 234 := by
  sorry

end NUMINAMATH_CALUDE_laura_weekly_mileage_l2346_234610


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2346_234645

theorem inscribed_circle_area_ratio (h r b : ℝ) : 
  h > 0 → r > 0 → b > 0 →
  (b + r)^2 + b^2 = h^2 →
  (2 * π * r^2) / ((b + r + h) * r) = 2 * π * r / (2 * b + r + h) := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2346_234645


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l2346_234642

-- Define the reduction factors
def first_reduction : ℝ := 0.85  -- 1 - 0.15
def second_reduction : ℝ := 0.90 -- 1 - 0.10

-- Theorem statement
theorem price_reduction_theorem :
  first_reduction * second_reduction * 100 = 76.5 := by
  sorry

#eval first_reduction * second_reduction * 100

end NUMINAMATH_CALUDE_price_reduction_theorem_l2346_234642


namespace NUMINAMATH_CALUDE_sum_series_equals_three_halves_l2346_234675

/-- The sum of the series (4n-3)/3^n from n=1 to infinity equals 3/2 -/
theorem sum_series_equals_three_halves :
  (∑' n : ℕ, (4 * n - 3 : ℝ) / 3^n) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_series_equals_three_halves_l2346_234675


namespace NUMINAMATH_CALUDE_total_pawns_left_l2346_234682

/-- The number of pawns each player starts with in a chess game -/
def initial_pawns : ℕ := 8

/-- The number of pawns Kennedy has lost -/
def kennedy_lost : ℕ := 4

/-- The number of pawns Riley has lost -/
def riley_lost : ℕ := 1

/-- Theorem: The total number of pawns left in the game is 11 -/
theorem total_pawns_left : 
  (initial_pawns - kennedy_lost) + (initial_pawns - riley_lost) = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_pawns_left_l2346_234682


namespace NUMINAMATH_CALUDE_sally_shirts_wednesday_l2346_234630

/-- The number of shirts Sally sewed on Wednesday -/
def shirts_on_wednesday (monday_shirts : ℕ) (tuesday_shirts : ℕ) (buttons_per_shirt : ℕ) (total_buttons : ℕ) : ℕ :=
  (total_buttons - (monday_shirts + tuesday_shirts) * buttons_per_shirt) / buttons_per_shirt

theorem sally_shirts_wednesday : 
  shirts_on_wednesday 4 3 5 45 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sally_shirts_wednesday_l2346_234630


namespace NUMINAMATH_CALUDE_sum_eight_fib_not_fib_l2346_234616

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the sum of eight consecutive Fibonacci numbers
def sum_eight_fib (k : ℕ) : ℕ :=
  (fib (k + 1)) + (fib (k + 2)) + (fib (k + 3)) + (fib (k + 4)) +
  (fib (k + 5)) + (fib (k + 6)) + (fib (k + 7)) + (fib (k + 8))

-- Theorem statement
theorem sum_eight_fib_not_fib (k : ℕ) :
  (sum_eight_fib k > fib (k + 9)) ∧ (sum_eight_fib k < fib (k + 10)) :=
by sorry

end NUMINAMATH_CALUDE_sum_eight_fib_not_fib_l2346_234616
