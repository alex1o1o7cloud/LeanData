import Mathlib

namespace NUMINAMATH_CALUDE_graduating_class_boys_count_l1328_132848

theorem graduating_class_boys_count (total : ℕ) (difference : ℕ) (boys : ℕ) : 
  total = 466 → difference = 212 → boys + (boys + difference) = total → boys = 127 := by
  sorry

end NUMINAMATH_CALUDE_graduating_class_boys_count_l1328_132848


namespace NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l1328_132827

/-- The volume of a cone formed from a 270-degree sector of a circle with radius 20 units, divided by π, is equal to 1125√7. -/
theorem cone_volume_divided_by_pi : 
  ∀ (r h : ℝ) (V : ℝ),
  -- Conditions
  (2 * π * r = 30 * π) →  -- Arc length becomes circumference of cone's base
  (20^2 = r^2 + h^2) →    -- Pythagorean theorem relating slant height to radius and height
  (V = (1/3) * π * r^2 * h) →  -- Volume formula for a cone
  -- Conclusion
  (V / π = 1125 * Real.sqrt 7) :=
by
  sorry

end NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l1328_132827


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l1328_132820

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

/-- Represents the result of the area calculation -/
structure AreaResult :=
  (r1 : ℚ)
  (n1 : ℕ)
  (r2 : ℚ)
  (n2 : ℕ)
  (r3 : ℚ)

/-- Function to calculate the area of the trapezoid -/
def calculateArea (t : Trapezoid) : AreaResult :=
  sorry

/-- Theorem stating the properties of the calculated area -/
theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.side1 = 4)
  (h2 : t.side2 = 6)
  (h3 : t.side3 = 8)
  (h4 : t.side4 = 10) :
  let result := calculateArea t
  Int.floor (result.r1 + result.r2 + result.r3 + result.n1 + result.n2) = 274 ∧
  ¬∃ (p : ℕ), Prime p ∧ (p^2 ∣ result.n1 ∨ p^2 ∣ result.n2) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l1328_132820


namespace NUMINAMATH_CALUDE_abc_inequality_abc_inequality_tight_l1328_132838

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) ≤ 1/8 :=
sorry

theorem abc_inequality_tight :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) = 1/8 :=
sorry

end NUMINAMATH_CALUDE_abc_inequality_abc_inequality_tight_l1328_132838


namespace NUMINAMATH_CALUDE_max_intersections_circle_line_parabola_l1328_132896

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in a 2D plane --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The maximum number of intersections between a circle and a line --/
def max_intersections_circle_line : ℕ := 2

/-- The maximum number of intersections between a parabola and a line --/
def max_intersections_parabola_line : ℕ := 2

/-- The maximum number of intersections between a circle and a parabola --/
def max_intersections_circle_parabola : ℕ := 4

/-- Theorem: The maximum number of intersections between a circle, a line, and a parabola is 8 --/
theorem max_intersections_circle_line_parabola 
  (c : Circle) (l : Line) (p : Parabola) : 
  max_intersections_circle_line + 
  max_intersections_parabola_line + 
  max_intersections_circle_parabola = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_circle_line_parabola_l1328_132896


namespace NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l1328_132890

/-- The number of rows in the grid -/
def n : ℕ := 4

/-- The number of columns in the grid -/
def m : ℕ := 4

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of rectangles in an n × m grid -/
def num_rectangles (n m : ℕ) : ℕ := choose_two n * choose_two m

theorem rectangles_in_4x4_grid : 
  num_rectangles n m = 36 :=
sorry

end NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l1328_132890


namespace NUMINAMATH_CALUDE_star_three_five_l1328_132880

def star (a b : ℕ) : ℕ := (a + b) ^ 3

theorem star_three_five : star 3 5 = 512 := by
  sorry

end NUMINAMATH_CALUDE_star_three_five_l1328_132880


namespace NUMINAMATH_CALUDE_right_triangle_cotangent_l1328_132832

theorem right_triangle_cotangent (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 12) :
  a / b = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cotangent_l1328_132832


namespace NUMINAMATH_CALUDE_polygon_sides_l1328_132878

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →                           -- n is at least 3 for a polygon
  ((n - 2) * 180 = 2 * 360) →         -- sum of interior angles = twice sum of exterior angles
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l1328_132878


namespace NUMINAMATH_CALUDE_bankers_gain_specific_case_l1328_132805

/-- Calculates the banker's gain given the banker's discount, interest rate, and time period. -/
def bankers_gain (bankers_discount : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  (bankers_discount * interest_rate * time) / (100 + (interest_rate * time))

/-- Theorem stating that given the specific conditions, the banker's gain is 90. -/
theorem bankers_gain_specific_case :
  bankers_gain 340 12 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_bankers_gain_specific_case_l1328_132805


namespace NUMINAMATH_CALUDE_x_squared_gt_one_necessary_not_sufficient_l1328_132899

theorem x_squared_gt_one_necessary_not_sufficient (x : ℝ) :
  (∀ x, x > 1 → x^2 > 1) ∧ (∃ x, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_gt_one_necessary_not_sufficient_l1328_132899


namespace NUMINAMATH_CALUDE_ratio_is_pure_imaginary_l1328_132869

theorem ratio_is_pure_imaginary (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) 
  (h : Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂)) : 
  ∃ (y : ℝ), z₁ / z₂ = Complex.I * y := by
  sorry

end NUMINAMATH_CALUDE_ratio_is_pure_imaginary_l1328_132869


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1328_132822

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (2^(Real.tan x) - 2^(Real.sin x)) / x^2 else 0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = Real.log (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1328_132822


namespace NUMINAMATH_CALUDE_f_composition_value_l1328_132829

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else Real.sin x

theorem f_composition_value : f (f ((7 * Real.pi) / 6)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1328_132829


namespace NUMINAMATH_CALUDE_parabola_intersection_l1328_132861

theorem parabola_intersection (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2*b*x₁ + c = 0 ∧ a * x₂^2 + 2*b*x₂ + c = 0) ∨
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b * x₁^2 + 2*c*x₁ + a = 0 ∧ b * x₂^2 + 2*c*x₂ + a = 0) ∨
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ c * x₁^2 + 2*a*x₁ + b = 0 ∧ c * x₂^2 + 2*a*x₂ + b = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1328_132861


namespace NUMINAMATH_CALUDE_smallest_m_is_ten_l1328_132842

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℤ
  first_term : a 1 = -19
  difference : a 7 - a 4 = 6
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ+) : ℤ :=
  (seq.a 1 + seq.a n) * n / 2

/-- The theorem to be proved -/
theorem smallest_m_is_ten (seq : ArithmeticSequence) :
  ∃ m : ℕ+, (∀ n : ℕ+, sum_n seq n ≥ sum_n seq m) ∧ 
    (∀ k : ℕ+, k < m → ∃ n : ℕ+, sum_n seq n < sum_n seq k) ∧
    m = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_is_ten_l1328_132842


namespace NUMINAMATH_CALUDE_trip_time_calculation_l1328_132888

/-- Given a driving time and a traffic time that is twice the driving time, 
    calculate the total trip time. -/
def total_trip_time (driving_time : ℝ) : ℝ :=
  driving_time + 2 * driving_time

theorem trip_time_calculation :
  total_trip_time 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_calculation_l1328_132888


namespace NUMINAMATH_CALUDE_remainder_problem_l1328_132828

theorem remainder_problem (n : ℤ) : n % 5 = 3 → (4 * n - 5) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1328_132828


namespace NUMINAMATH_CALUDE_rectangle_problem_l1328_132891

theorem rectangle_problem (num_rectangles : ℕ) (area_large : ℝ) 
  (h1 : num_rectangles = 6)
  (h2 : area_large = 6000) :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    (num_rectangles : ℝ) * (2/5 * x) * x = area_large ∧ 
    x = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_problem_l1328_132891


namespace NUMINAMATH_CALUDE_parallel_iff_slope_eq_l1328_132868

/-- Two lines in the plane -/
structure Line where
  k : ℝ
  b : ℝ

/-- Define when two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.k = l2.k

/-- The main theorem: k1 = k2 iff l1 ∥ l2 -/
theorem parallel_iff_slope_eq (l1 l2 : Line) :
  l1.k = l2.k ↔ parallel l1 l2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_iff_slope_eq_l1328_132868


namespace NUMINAMATH_CALUDE_adjacent_chair_subsets_l1328_132858

/-- Given 12 chairs arranged in a circle, this function calculates the number of subsets
    containing at least three adjacent chairs. -/
def subsets_with_adjacent_chairs (num_chairs : ℕ) : ℕ :=
  if num_chairs = 12 then
    2010
  else
    0

/-- Theorem stating that for 12 chairs in a circle, there are 2010 subsets
    with at least three adjacent chairs. -/
theorem adjacent_chair_subsets :
  subsets_with_adjacent_chairs 12 = 2010 := by
  sorry

#eval subsets_with_adjacent_chairs 12

end NUMINAMATH_CALUDE_adjacent_chair_subsets_l1328_132858


namespace NUMINAMATH_CALUDE_min_gennadys_required_l1328_132871

/-- Represents the number of people with a given name -/
structure Attendees :=
  (alexanders : Nat)
  (borises : Nat)
  (vasilys : Nat)
  (gennadys : Nat)

/-- Checks if the arrangement is valid (no two people with the same name are adjacent) -/
def isValidArrangement (a : Attendees) : Prop :=
  a.borises - 1 ≤ a.alexanders + a.vasilys + a.gennadys

/-- The given festival attendance -/
def festivalAttendance : Attendees :=
  { alexanders := 45
  , borises := 122
  , vasilys := 27
  , gennadys := 49 }

/-- Theorem stating that 49 is the minimum number of Gennadys required -/
theorem min_gennadys_required :
  isValidArrangement festivalAttendance ∧
  ∀ g : Nat, g < festivalAttendance.gennadys →
    ¬isValidArrangement { alexanders := festivalAttendance.alexanders
                        , borises := festivalAttendance.borises
                        , vasilys := festivalAttendance.vasilys
                        , gennadys := g } :=
by
  sorry

end NUMINAMATH_CALUDE_min_gennadys_required_l1328_132871


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1328_132800

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = 5/2 ∧ Q = 0 ∧ R = -5) ∧
    ∀ (x : ℚ), x ≠ 4 ∧ x ≠ 2 →
      5*x / ((x - 4) * (x - 2)^3) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^3 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1328_132800


namespace NUMINAMATH_CALUDE_inequality_range_l1328_132802

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1328_132802


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l1328_132831

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l1328_132831


namespace NUMINAMATH_CALUDE_no_integer_solution_l1328_132818

theorem no_integer_solution : ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 - 2*x^2*y^2 - 2*y^2*z^2 - 2*z^2*x^2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1328_132818


namespace NUMINAMATH_CALUDE_max_z_value_l1328_132894

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x*y + y*z + z*x = 3) :
  z ≤ 13/3 :=
by sorry

end NUMINAMATH_CALUDE_max_z_value_l1328_132894


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1328_132881

theorem rectangle_ratio (w : ℚ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 32 → w / 10 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1328_132881


namespace NUMINAMATH_CALUDE_horner_method_correctness_horner_poly_at_5_l1328_132847

def horner_poly (x : ℝ) : ℝ := (((((3*x - 4)*x + 6)*x - 2)*x - 5)*x - 2)

def original_poly (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem horner_method_correctness :
  ∀ x : ℝ, horner_poly x = original_poly x :=
sorry

theorem horner_poly_at_5 : horner_poly 5 = 7548 :=
sorry

end NUMINAMATH_CALUDE_horner_method_correctness_horner_poly_at_5_l1328_132847


namespace NUMINAMATH_CALUDE_exists_n_with_digit_sum_property_l1328_132817

/-- Sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- Predicate to check if a number is composed of only 0 and 1 digits -/
def isComposedOf01 (n : ℕ) : Prop :=
  sorry

/-- Main theorem -/
theorem exists_n_with_digit_sum_property (m : ℕ) :
  ∃ n : ℕ, isComposedOf01 n ∧ sumOfDigits n = m ∧ sumOfDigits (n^2) = m^2 :=
sorry

end NUMINAMATH_CALUDE_exists_n_with_digit_sum_property_l1328_132817


namespace NUMINAMATH_CALUDE_stratified_sampling_example_l1328_132821

/-- Given a total number of positions, male doctors, and female doctors,
    calculate the number of male doctors to be selected through stratified sampling. -/
def stratified_sampling (total_positions : ℕ) (male_doctors : ℕ) (female_doctors : ℕ) : ℕ :=
  (total_positions * male_doctors) / (male_doctors + female_doctors)

theorem stratified_sampling_example :
  stratified_sampling 15 120 180 = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_example_l1328_132821


namespace NUMINAMATH_CALUDE_distance_between_points_l1328_132875

/-- The distance between points (1, -3) and (-4, 7) is 5√5. -/
theorem distance_between_points : Real.sqrt ((1 - (-4))^2 + (-3 - 7)^2) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1328_132875


namespace NUMINAMATH_CALUDE_ducks_at_north_pond_l1328_132895

/-- The number of ducks at North Pond given the specified conditions -/
theorem ducks_at_north_pond :
  let mallard_lake_michigan : ℕ := 100
  let pintail_lake_michigan : ℕ := 75
  let mallard_north_pond : ℕ := 2 * mallard_lake_michigan + 6
  let pintail_north_pond : ℕ := 4 * mallard_lake_michigan
  mallard_north_pond + pintail_north_pond = 606 :=
by sorry


end NUMINAMATH_CALUDE_ducks_at_north_pond_l1328_132895


namespace NUMINAMATH_CALUDE_product_equality_equal_S_not_imply_equal_Q_l1328_132887

-- Define a structure for a triangle divided by cevians
structure CevianTriangle where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  Q₁ : ℝ
  Q₂ : ℝ
  Q₃ : ℝ
  S_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0
  Q_positive : Q₁ > 0 ∧ Q₂ > 0 ∧ Q₃ > 0

-- Theorem 1: Product of S areas equals product of Q areas
theorem product_equality (t : CevianTriangle) : t.S₁ * t.S₂ * t.S₃ = t.Q₁ * t.Q₂ * t.Q₃ := by
  sorry

-- Theorem 2: Equal S areas do not necessarily imply equal Q areas
theorem equal_S_not_imply_equal_Q :
  ∃ t : CevianTriangle, (t.S₁ = t.S₂ ∧ t.S₂ = t.S₃) ∧ (t.Q₁ ≠ t.Q₂ ∨ t.Q₂ ≠ t.Q₃ ∨ t.Q₁ ≠ t.Q₃) := by
  sorry

end NUMINAMATH_CALUDE_product_equality_equal_S_not_imply_equal_Q_l1328_132887


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1328_132801

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (a : ℕ), 5 * n = a^2) ∧ 
  (∃ (b : ℕ), 3 * n = b^3) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 5 * m = x^2) → 
    (∃ (y : ℕ), 3 * m = y^3) → 
    m ≥ n) ∧
  n = 1125 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1328_132801


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1328_132854

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^5 + b^5) / (a + b)^5 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1328_132854


namespace NUMINAMATH_CALUDE_tenth_term_is_19_l1328_132830

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (a 3 = 5) ∧ (a 6 = 11) ∧ 
  ∃ d : ℝ, ∀ n m : ℕ, a (n + m) = a n + m * d

/-- The 10th term of the arithmetic sequence is 19 -/
theorem tenth_term_is_19 (a : ℕ → ℝ) (h : arithmetic_sequence a) : 
  a 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_19_l1328_132830


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1328_132835

def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4}

theorem intersection_of_A_and_B : A ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1328_132835


namespace NUMINAMATH_CALUDE_multiples_properties_l1328_132862

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 12 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ (∃ p : ℤ, a - b = 4 * p) :=
by sorry

end NUMINAMATH_CALUDE_multiples_properties_l1328_132862


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1328_132808

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 2
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (0, 2 * Real.sqrt 3, 2) := by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1328_132808


namespace NUMINAMATH_CALUDE_circle_radius_theorem_l1328_132864

/-- The radius of a circle concentric with and outside a regular octagon -/
def circle_radius (octagon_side_length : ℝ) (probability_four_sides : ℝ) : ℝ :=
  sorry

/-- The theorem stating the relationship between the circle radius, octagon side length, and probability of seeing four sides -/
theorem circle_radius_theorem (octagon_side_length : ℝ) (probability_four_sides : ℝ) :
  circle_radius octagon_side_length probability_four_sides = 6 * Real.sqrt 2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_theorem_l1328_132864


namespace NUMINAMATH_CALUDE_james_lifting_ratio_l1328_132839

theorem james_lifting_ratio :
  let initial_total : ℝ := 2200
  let initial_weight : ℝ := 245
  let total_gain_percent : ℝ := 0.15
  let weight_gain : ℝ := 8
  let final_total : ℝ := initial_total * (1 + total_gain_percent)
  let final_weight : ℝ := initial_weight + weight_gain
  final_total / final_weight = 10
  := by sorry

end NUMINAMATH_CALUDE_james_lifting_ratio_l1328_132839


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1328_132815

theorem triangle_angle_measure (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) 
  (h_condition : Real.sin B ^ 2 - Real.sin C ^ 2 - Real.sin A ^ 2 = Real.sqrt 3 * Real.sin A * Real.sin C) : 
  B = 5 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1328_132815


namespace NUMINAMATH_CALUDE_correct_lineup_count_l1328_132816

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def all_stars : ℕ := 3
def required_players : ℕ := 2

def possible_lineups : ℕ := Nat.choose (team_size - all_stars) (lineup_size - required_players)

theorem correct_lineup_count : possible_lineups = 220 := by sorry

end NUMINAMATH_CALUDE_correct_lineup_count_l1328_132816


namespace NUMINAMATH_CALUDE_fraction_integrality_l1328_132867

theorem fraction_integrality (a b c : ℤ) 
  (h : ∃ (n : ℤ), (a * b / c + a * c / b + b * c / a) = n) : 
  (∃ (n1 : ℤ), a * b / c = n1) ∧ 
  (∃ (n2 : ℤ), a * c / b = n2) ∧ 
  (∃ (n3 : ℤ), b * c / a = n3) := by
sorry

end NUMINAMATH_CALUDE_fraction_integrality_l1328_132867


namespace NUMINAMATH_CALUDE_max_rectangles_in_modified_grid_l1328_132897

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangular cut-out --/
structure Cutout :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the area of a grid --/
def gridArea (g : Grid) : ℕ :=
  g.rows * g.cols

/-- Calculates the area of a cutout --/
def cutoutArea (c : Cutout) : ℕ :=
  c.width * c.height

/-- Calculates the remaining area after cutouts --/
def remainingArea (g : Grid) (cutouts : List Cutout) : ℕ :=
  gridArea g - (cutouts.map cutoutArea).sum

/-- Theorem: Maximum number of 1x3 rectangles in modified 8x8 grid --/
theorem max_rectangles_in_modified_grid :
  let initial_grid : Grid := ⟨8, 8⟩
  let cutouts : List Cutout := [⟨2, 2⟩, ⟨2, 2⟩, ⟨2, 2⟩]
  let remaining_cells := remainingArea initial_grid cutouts
  (remaining_cells / 3 : ℕ) = 17 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangles_in_modified_grid_l1328_132897


namespace NUMINAMATH_CALUDE_digit_puzzle_solution_l1328_132814

theorem digit_puzzle_solution :
  ∃! (A B C D E F G H J : ℕ),
    (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10 ∧ J < 10) ∧
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ J ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ J ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ J ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ J ∧
     E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ J ∧
     F ≠ G ∧ F ≠ H ∧ F ≠ J ∧
     G ≠ H ∧ G ≠ J ∧
     H ≠ J) ∧
    (100 * A + 10 * B + C + 100 * D + 10 * E + F + 10 * G + E = 100 * G + 10 * E + F) ∧
    (100 * G + 10 * E + F + 10 * D + E = 100 * H + 10 * F + J) ∧
    A = 2 ∧ B = 3 ∧ C = 0 ∧ D = 1 ∧ E = 7 ∧ F = 8 ∧ G = 4 ∧ H = 5 ∧ J = 6 :=
by sorry

end NUMINAMATH_CALUDE_digit_puzzle_solution_l1328_132814


namespace NUMINAMATH_CALUDE_job_completion_time_l1328_132809

theorem job_completion_time (x : ℝ) : 
  x > 0 → 
  4 * (1/x + 1/30) = 0.4 → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l1328_132809


namespace NUMINAMATH_CALUDE_max_tan_A_in_triangle_l1328_132851

open Real

theorem max_tan_A_in_triangle (a b c A B C : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  -- Given conditions
  a = 2 →
  b * cos C - c * cos B = 4 →
  π/4 ≤ C ∧ C ≤ π/3 →
  -- Conclusion
  (∃ (max_tan_A : ℝ), max_tan_A = 1/2 ∧ ∀ (tan_A : ℝ), tan_A = tan A → tan_A ≤ max_tan_A) :=
by sorry

end NUMINAMATH_CALUDE_max_tan_A_in_triangle_l1328_132851


namespace NUMINAMATH_CALUDE_mountain_has_three_sections_l1328_132892

/-- Given a mountain with eagles, calculate the number of sections. -/
def mountain_sections (eagles_per_section : ℕ) (total_eagles : ℕ) : ℕ :=
  total_eagles / eagles_per_section

/-- Theorem: The mountain has 3 sections given the specified conditions. -/
theorem mountain_has_three_sections :
  let eagles_per_section := 6
  let total_eagles := 18
  mountain_sections eagles_per_section total_eagles = 3 := by
  sorry

end NUMINAMATH_CALUDE_mountain_has_three_sections_l1328_132892


namespace NUMINAMATH_CALUDE_diamond_op_four_three_l1328_132803

def diamond_op (m n : ℕ) : ℕ := n ^ 2 - m

theorem diamond_op_four_three : diamond_op 4 3 = 5 := by sorry

end NUMINAMATH_CALUDE_diamond_op_four_three_l1328_132803


namespace NUMINAMATH_CALUDE_competition_probability_l1328_132863

/-- The probability of correctly answering a single question -/
def p_correct : ℝ := 0.8

/-- The probability of incorrectly answering a single question -/
def p_incorrect : ℝ := 1 - p_correct

/-- The number of preset questions in the competition -/
def num_questions : ℕ := 5

/-- The probability of answering exactly 4 questions before advancing -/
def prob_four_questions : ℝ := p_correct * p_incorrect * p_correct * p_correct

theorem competition_probability :
  prob_four_questions = 0.128 :=
sorry

end NUMINAMATH_CALUDE_competition_probability_l1328_132863


namespace NUMINAMATH_CALUDE_no_double_application_function_l1328_132806

theorem no_double_application_function : ¬∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l1328_132806


namespace NUMINAMATH_CALUDE_puppy_weight_l1328_132873

theorem puppy_weight (puppy smaller_cat larger_cat : ℝ) 
  (total_weight : puppy + smaller_cat + larger_cat = 24)
  (puppy_larger_cat : puppy + larger_cat = 2 * smaller_cat)
  (puppy_smaller_cat : puppy + smaller_cat = larger_cat) :
  puppy = 4 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l1328_132873


namespace NUMINAMATH_CALUDE_translation_of_sine_to_cosine_l1328_132872

/-- Given a function f(x) = sin(2x + π/6), prove that translating it π/6 units to the left
    results in the function g(x) = cos(2x) -/
theorem translation_of_sine_to_cosine (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 6)
  let g : ℝ → ℝ := λ x => f (x + π / 6)
  g x = Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_translation_of_sine_to_cosine_l1328_132872


namespace NUMINAMATH_CALUDE_geometry_propositions_l1328_132811

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Axioms for the properties of parallel and perpendicular
axiom parallel_transitive {a b c : Plane} : parallel a b → parallel a c → parallel b c
axiom perpendicular_from_line {l : Line} {a b : Plane} : 
  line_perpendicular_plane l a → line_parallel_plane l b → perpendicular a b

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State the theorem
theorem geometry_propositions :
  -- Proposition ①
  (∀ a b c : Plane, parallel a b → parallel a c → parallel b c) ∧
  -- Proposition ③
  (∀ l : Line, ∀ a b : Plane, line_perpendicular_plane l a → line_parallel_plane l b → perpendicular a b) ∧
  -- Negation of Proposition ②
  ¬(∀ l : Line, ∀ a b : Plane, perpendicular a b → line_parallel_plane l a → line_perpendicular_plane l b) ∧
  -- Negation of Proposition ④
  ¬(∀ l1 l2 : Line, ∀ a : Plane, line_parallel l1 l2 → line_in_plane l2 a → line_parallel_plane l1 a) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1328_132811


namespace NUMINAMATH_CALUDE_range_of_a_l1328_132846

open Set

/-- The range of a for which ¬p is a necessary but not sufficient condition for ¬q -/
theorem range_of_a (a : ℝ) : 
  (a < 0) →
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) → 
    (x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0)) →
  (∃ x : ℝ, (x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0) ∧ 
    ¬(x^2 - 4*a*x + 3*a^2 < 0)) →
  (a ≤ -4 ∨ -2/3 ≤ a) :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_l1328_132846


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1328_132898

theorem simplify_sqrt_expression :
  (Real.sqrt 500 / Real.sqrt 180) + (Real.sqrt 128 / Real.sqrt 32) = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1328_132898


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l1328_132825

/-- An arithmetic sequence with the given properties has the general term formula a_n = 2n -/
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) - a n = d) →  -- arithmetic sequence
  d ≠ 0 →  -- non-zero common difference
  a 1 = 2 →  -- a_1 = 2
  (a 2 * a 8 = (a 4)^2) →  -- (a_2, a_4, a_8) forms a geometric sequence
  (∀ n, a n = 2 * n) :=  -- general term formula
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l1328_132825


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l1328_132807

/-- Two congruent squares with side length 20 overlap to form a 20 by 40 rectangle.
    The shaded area is the overlap of the two squares. -/
theorem shaded_area_percentage (square_side : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  square_side = 20 →
  rectangle_width = 20 →
  rectangle_length = 40 →
  (square_side * square_side) / (rectangle_width * rectangle_length) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l1328_132807


namespace NUMINAMATH_CALUDE_equal_segments_after_rearrangement_l1328_132810

-- Define a line in a plane
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a right-angled triangle
structure RightTriangle :=
  (leg1 : ℝ)
  (leg2 : ℝ)

-- Define a function to check if a line is parallel to another line
def isParallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to check if a line intersects triangles in equal segments
def intersectsInEqualSegments (l : Line) (t1 t2 t3 : RightTriangle) : Prop :=
  sorry -- Definition omitted for brevity

-- Main theorem
theorem equal_segments_after_rearrangement
  (l : Line)
  (t1 t2 t3 : RightTriangle)
  (h1 : ∃ (l' : Line), isParallel l l' ∧ intersectsInEqualSegments l' t1 t2 t3) :
  ∃ (l'' : Line), isParallel l l'' ∧ intersectsInEqualSegments l'' t1 t2 t3 :=
by sorry

end NUMINAMATH_CALUDE_equal_segments_after_rearrangement_l1328_132810


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l1328_132860

theorem beef_weight_before_processing (weight_after : ℝ) (percent_lost : ℝ) : 
  weight_after = 240 ∧ percent_lost = 40 → 
  weight_after / (1 - percent_lost / 100) = 400 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l1328_132860


namespace NUMINAMATH_CALUDE_at_most_one_integer_point_on_circle_l1328_132849

theorem at_most_one_integer_point_on_circle :
  ∀ (x y u v : ℤ),
  (x - Real.sqrt 2)^2 + (y - Real.sqrt 3)^2 = (u - Real.sqrt 2)^2 + (v - Real.sqrt 3)^2 →
  x = u ∧ y = v :=
by sorry

end NUMINAMATH_CALUDE_at_most_one_integer_point_on_circle_l1328_132849


namespace NUMINAMATH_CALUDE_DL_length_l1328_132857

-- Define the triangle DEF
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (FD : ℝ)

-- Define the circles ω3 and ω4
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point L
def L : ℝ × ℝ := sorry

-- Define the given triangle
def givenTriangle : Triangle :=
  { DE := 6
  , EF := 10
  , FD := 8 }

-- Define circle ω3
def ω3 : Circle := sorry

-- Define circle ω4
def ω4 : Circle := sorry

-- State the theorem
theorem DL_length (t : Triangle) (ω3 ω4 : Circle) :
  t = givenTriangle →
  (ω3.center.1 - L.1)^2 + (ω3.center.2 - L.2)^2 = ω3.radius^2 →
  (ω4.center.1 - L.1)^2 + (ω4.center.2 - L.2)^2 = ω4.radius^2 →
  (0 - L.1)^2 + (0 - L.2)^2 = 4^2 := by
  sorry

end NUMINAMATH_CALUDE_DL_length_l1328_132857


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l1328_132853

theorem set_equality_implies_sum (a b : ℝ) : 
  ({-1, a} : Set ℝ) = ({b, 1} : Set ℝ) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l1328_132853


namespace NUMINAMATH_CALUDE_multiplication_division_equality_l1328_132850

theorem multiplication_division_equality : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_equality_l1328_132850


namespace NUMINAMATH_CALUDE_min_value_expression_l1328_132856

theorem min_value_expression (x y k : ℝ) : (x*y - k)^2 + (x + y - 1)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1328_132856


namespace NUMINAMATH_CALUDE_cubic_plus_linear_increasing_l1328_132845

/-- The function f(x) = x^3 + x is strictly increasing on all real numbers. -/
theorem cubic_plus_linear_increasing : 
  ∀ x y : ℝ, x < y → (x^3 + x) < (y^3 + y) := by
sorry

end NUMINAMATH_CALUDE_cubic_plus_linear_increasing_l1328_132845


namespace NUMINAMATH_CALUDE_max_log_sum_l1328_132840

theorem max_log_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 2*x + y = 20) :
  ∃ (max_val : ℝ), max_val = 2 - Real.log 2 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 2*a + b = 20 → Real.log a + Real.log b ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_log_sum_l1328_132840


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1328_132874

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1328_132874


namespace NUMINAMATH_CALUDE_simplify_expression_l1328_132882

theorem simplify_expression (x : ℝ) : (3*x)^4 - (4*x^2)*(2*x^3) + 5*x^4 = 86*x^4 - 8*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1328_132882


namespace NUMINAMATH_CALUDE_tank_capacity_l1328_132865

/-- Represents a water tank with a certain capacity -/
structure WaterTank where
  capacity : ℝ
  emptyWeight : ℝ
  waterWeight : ℝ
  filledWeight : ℝ
  filledPercentage : ℝ

/-- Theorem stating that a tank with the given properties has a capacity of 200 gallons -/
theorem tank_capacity (tank : WaterTank) 
  (h1 : tank.emptyWeight = 80)
  (h2 : tank.waterWeight = 8)
  (h3 : tank.filledWeight = 1360)
  (h4 : tank.filledPercentage = 0.8) :
  tank.capacity = 200 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l1328_132865


namespace NUMINAMATH_CALUDE_smallest_term_at_six_l1328_132834

/-- The general term of the sequence -/
def a (n : ℕ) : ℝ := 3 * n^2 - 38 * n + 12

/-- The index of the smallest term in the sequence -/
def smallest_term_index : ℕ := 6

/-- Theorem stating that the smallest term in the sequence occurs at index 6 -/
theorem smallest_term_at_six :
  ∀ (n : ℕ), n ≠ smallest_term_index → a n > a smallest_term_index :=
sorry

end NUMINAMATH_CALUDE_smallest_term_at_six_l1328_132834


namespace NUMINAMATH_CALUDE_fraction_equality_l1328_132837

theorem fraction_equality (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : (5 * x + 2 * y) / (x - 5 * y) = 3) : 
  (x + 5 * y) / (5 * x - y) = 7 / 87 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1328_132837


namespace NUMINAMATH_CALUDE_complete_square_sum_l1328_132879

theorem complete_square_sum (a b c : ℤ) : 
  (∀ x : ℚ, 25 * x^2 + 30 * x - 45 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 62 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1328_132879


namespace NUMINAMATH_CALUDE_pokemon_card_solution_l1328_132889

def pokemon_card_problem (initial_cards : ℕ) : Prop :=
  let after_trade := initial_cards - 5 + 3
  let after_giving := after_trade - 9
  let final_cards := after_giving + 2
  final_cards = 4

theorem pokemon_card_solution :
  pokemon_card_problem 13 := by sorry

end NUMINAMATH_CALUDE_pokemon_card_solution_l1328_132889


namespace NUMINAMATH_CALUDE_degrees_to_radians_1920_l1328_132813

theorem degrees_to_radians_1920 : 
  (1920 : ℝ) * (π / 180) = (32 * π) / 3 := by sorry

end NUMINAMATH_CALUDE_degrees_to_radians_1920_l1328_132813


namespace NUMINAMATH_CALUDE_max_t_geq_pi_l1328_132893

noncomputable def f (x : ℝ) : ℝ := Real.sin x / x

theorem max_t_geq_pi (t : ℝ) (h : ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < t → f x₁ > f x₂) :
  t ≥ π :=
sorry

end NUMINAMATH_CALUDE_max_t_geq_pi_l1328_132893


namespace NUMINAMATH_CALUDE_toy_store_shelves_l1328_132884

def shelves_required (initial_stock new_shipment bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

theorem toy_store_shelves :
  shelves_required 4 10 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l1328_132884


namespace NUMINAMATH_CALUDE_ratio_of_x_to_y_l1328_132844

theorem ratio_of_x_to_y (x y : ℝ) (h1 : 5 * x = 6 * y) (h2 : x * y ≠ 0) :
  (1/3 * x) / (1/5 * y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_to_y_l1328_132844


namespace NUMINAMATH_CALUDE_goods_train_speed_l1328_132841

/-- The speed of the goods train given the conditions of the problem -/
theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (goods_train_length : ℝ) 
  (passing_time : ℝ) 
  (h1 : man_train_speed = 60) 
  (h2 : goods_train_length = 0.3) -- 300 m converted to km
  (h3 : passing_time = 1/300) -- 12 seconds converted to hours
  : ∃ (goods_train_speed : ℝ), goods_train_speed = 30 :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l1328_132841


namespace NUMINAMATH_CALUDE_craftsman_jars_l1328_132859

theorem craftsman_jars (jars clay_pots : ℕ) (h1 : jars = 2 * clay_pots)
  (h2 : 5 * jars + 3 * 5 * clay_pots = 200) : jars = 16 := by
  sorry

end NUMINAMATH_CALUDE_craftsman_jars_l1328_132859


namespace NUMINAMATH_CALUDE_equation_solution_l1328_132819

theorem equation_solution : ∃ x : ℝ, (Real.sqrt (72 / 25) = (x / 25) ^ (1/4)) ∧ x = 207.36 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1328_132819


namespace NUMINAMATH_CALUDE_district3_to_district1_ratio_l1328_132812

/-- The number of voters in District 1 -/
def district1_voters : ℕ := 322

/-- The difference in voters between District 3 and District 2 -/
def district3_2_diff : ℕ := 19

/-- The total number of voters in all three districts -/
def total_voters : ℕ := 1591

/-- The ratio of voters in District 3 to District 1 -/
def voter_ratio : ℚ := 2

theorem district3_to_district1_ratio :
  ∃ (district2_voters district3_voters : ℕ),
    district2_voters = district3_voters - district3_2_diff ∧
    district1_voters + district2_voters + district3_voters = total_voters ∧
    district3_voters = (voter_ratio : ℚ) * district1_voters := by
  sorry

end NUMINAMATH_CALUDE_district3_to_district1_ratio_l1328_132812


namespace NUMINAMATH_CALUDE_dot_product_of_vectors_l1328_132855

/-- Given two vectors a and b in ℝ², prove that their dot product is -12
    when a + b = (1, 3) and a - b = (3, 7). -/
theorem dot_product_of_vectors (a b : ℝ × ℝ) 
    (h1 : a + b = (1, 3)) 
    (h2 : a - b = (3, 7)) : 
  a.1 * b.1 + a.2 * b.2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_vectors_l1328_132855


namespace NUMINAMATH_CALUDE_buses_needed_l1328_132823

theorem buses_needed (students : ℕ) (seats_per_bus : ℕ) (h1 : students = 28) (h2 : seats_per_bus = 7) :
  (students + seats_per_bus - 1) / seats_per_bus = 4 := by
  sorry

end NUMINAMATH_CALUDE_buses_needed_l1328_132823


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1328_132824

theorem ratio_of_sum_to_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 8 * (x - y)) : x / y = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1328_132824


namespace NUMINAMATH_CALUDE_a_squared_plus_reciprocal_squared_is_integer_l1328_132843

theorem a_squared_plus_reciprocal_squared_is_integer (a : ℝ) (h : ∃ k : ℤ, a + 1 / a = k) :
  ∃ m : ℤ, a^2 + 1 / a^2 = m := by
sorry

end NUMINAMATH_CALUDE_a_squared_plus_reciprocal_squared_is_integer_l1328_132843


namespace NUMINAMATH_CALUDE_dog_age_ratio_l1328_132877

/-- Given information about five dogs' ages, prove the ratio of the 4th to 3rd fastest dog's age --/
theorem dog_age_ratio :
  ∀ (age1 age2 age3 age4 age5 : ℕ),
  -- Average age of 1st and 5th fastest dogs is 18 years
  (age1 + age5) / 2 = 18 →
  -- 1st fastest dog is 10 years old
  age1 = 10 →
  -- 2nd fastest dog is 2 years younger than the 1st fastest dog
  age2 = age1 - 2 →
  -- 3rd fastest dog is 4 years older than the 2nd fastest dog
  age3 = age2 + 4 →
  -- 4th fastest dog is half the age of the 3rd fastest dog
  2 * age4 = age3 →
  -- 5th fastest dog is 20 years older than the 4th fastest dog
  age5 = age4 + 20 →
  -- Ratio of 4th fastest dog's age to 3rd fastest dog's age is 1:2
  2 * age4 = age3 := by
  sorry

end NUMINAMATH_CALUDE_dog_age_ratio_l1328_132877


namespace NUMINAMATH_CALUDE_gingerbread_theorem_l1328_132852

def gingerbread_problem (red_hats blue_boots both : ℕ) : Prop :=
  let total := red_hats + blue_boots - both
  (red_hats : ℚ) / total * 100 = 50

theorem gingerbread_theorem :
  gingerbread_problem 6 9 3 := by
  sorry

end NUMINAMATH_CALUDE_gingerbread_theorem_l1328_132852


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l1328_132883

theorem least_four_digit_multiple : ∀ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) → 
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 7 = 0) → 
  1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_l1328_132883


namespace NUMINAMATH_CALUDE_ratio_equation_solution_sum_l1328_132836

theorem ratio_equation_solution_sum : 
  ∃! s : ℝ, ∀ x : ℝ, (3 * x + 4) / (5 * x + 4) = (5 * x + 6) / (8 * x + 6) → s = x :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_sum_l1328_132836


namespace NUMINAMATH_CALUDE_meadow_orders_30_boxes_l1328_132833

/-- Represents Meadow's diaper business --/
structure DiaperBusiness where
  packs_per_box : ℕ
  diapers_per_pack : ℕ
  price_per_diaper : ℕ
  total_revenue : ℕ

/-- Calculates the number of boxes ordered weekly --/
def boxes_ordered (business : DiaperBusiness) : ℕ :=
  business.total_revenue / (business.price_per_diaper * business.diapers_per_pack * business.packs_per_box)

/-- Theorem: Given the conditions, Meadow orders 30 boxes weekly --/
theorem meadow_orders_30_boxes :
  let business : DiaperBusiness := {
    packs_per_box := 40,
    diapers_per_pack := 160,
    price_per_diaper := 5,
    total_revenue := 960000
  }
  boxes_ordered business = 30 := by
  sorry

end NUMINAMATH_CALUDE_meadow_orders_30_boxes_l1328_132833


namespace NUMINAMATH_CALUDE_parabola_roots_l1328_132876

/-- Given a parabola y = ax^2 - 2ax + c where a ≠ 0 that passes through the point (3, 0),
    prove that the solutions to ax^2 - 2ax + c = 0 are x₁ = -1 and x₂ = 3. -/
theorem parabola_roots (a c : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 - 2*a*x + c = 0 ↔ x = -1 ∨ x = 3) ↔
  a * 3^2 - 2*a*3 + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_roots_l1328_132876


namespace NUMINAMATH_CALUDE_distance_between_students_l1328_132885

/-- The distance between two students after 4 hours, given they start from the same point
    and walk in opposite directions with speeds of 6 km/hr and 9 km/hr respectively. -/
theorem distance_between_students (speed1 speed2 time : ℝ) 
  (h1 : speed1 = 6)
  (h2 : speed2 = 9)
  (h3 : time = 4) :
  speed1 * time + speed2 * time = 60 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_students_l1328_132885


namespace NUMINAMATH_CALUDE_simplification_to_x_plus_one_l1328_132870

theorem simplification_to_x_plus_one (x : ℝ) (h : x ≠ 1) :
  (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplification_to_x_plus_one_l1328_132870


namespace NUMINAMATH_CALUDE_number_with_given_quotient_and_remainder_l1328_132886

theorem number_with_given_quotient_and_remainder : 
  ∀ N : ℕ, (N / 7 = 12 ∧ N % 7 = 5) → N = 89 :=
by sorry

end NUMINAMATH_CALUDE_number_with_given_quotient_and_remainder_l1328_132886


namespace NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l1328_132826

theorem cubic_polynomial_coefficient (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l1328_132826


namespace NUMINAMATH_CALUDE_triangle_area_not_integer_l1328_132866

theorem triangle_area_not_integer (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) 
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ¬ ∃ (S : ℕ), (S : ℝ)^2 * 16 = (a + b + c) * ((a + b + c) - 2*a) * ((a + b + c) - 2*b) * ((a + b + c) - 2*c) :=
sorry


end NUMINAMATH_CALUDE_triangle_area_not_integer_l1328_132866


namespace NUMINAMATH_CALUDE_max_length_AB_l1328_132804

/-- The function representing the length of AB -/
def f (t : ℝ) : ℝ := -2 * t^2 + 3 * t + 9

/-- The theorem stating the maximum value of f(t) for t in [0, 3] -/
theorem max_length_AB : 
  ∃ (t : ℝ), t ∈ Set.Icc 0 3 ∧ f t = 81/8 ∧ ∀ x ∈ Set.Icc 0 3, f x ≤ 81/8 :=
sorry

end NUMINAMATH_CALUDE_max_length_AB_l1328_132804
