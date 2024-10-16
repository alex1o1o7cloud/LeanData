import Mathlib

namespace NUMINAMATH_CALUDE_cylinder_volume_l4112_411285

/-- Given a sphere and a cylinder with specific properties, prove the volume of the cylinder --/
theorem cylinder_volume (sphere_volume : ℝ) (cylinder_base_diameter : ℝ) :
  sphere_volume = (500 * Real.pi) / 3 →
  cylinder_base_diameter = 8 →
  ∃ (cylinder_volume : ℝ),
    cylinder_volume = 96 * Real.pi ∧
    (∃ (sphere_radius : ℝ) (cylinder_height : ℝ),
      (4 / 3) * Real.pi * sphere_radius ^ 3 = sphere_volume ∧
      cylinder_height ^ 2 = sphere_radius ^ 2 - (cylinder_base_diameter / 2) ^ 2 ∧
      cylinder_volume = Real.pi * (cylinder_base_diameter / 2) ^ 2 * cylinder_height) :=
by sorry


end NUMINAMATH_CALUDE_cylinder_volume_l4112_411285


namespace NUMINAMATH_CALUDE_black_hen_day_probability_l4112_411292

/-- Represents the color of a hen -/
inductive HenColor
| Black
| White

/-- Represents a program type -/
inductive ProgramType
| Day
| Evening

/-- Represents the state of available spots -/
structure AvailableSpots :=
  (day : Nat)
  (evening : Nat)

/-- Represents a hen's application -/
structure Application :=
  (color : HenColor)
  (program : ProgramType)

/-- The probability of at least one black hen in the daytime program -/
def prob_black_hen_day (total_spots : Nat) (day_spots : Nat) (evening_spots : Nat) 
                       (black_hens : Nat) (white_hens : Nat) : ℚ :=
  sorry

theorem black_hen_day_probability :
  let total_spots := 5
  let day_spots := 2
  let evening_spots := 3
  let black_hens := 3
  let white_hens := 1
  prob_black_hen_day total_spots day_spots evening_spots black_hens white_hens = 59 / 64 :=
by sorry

end NUMINAMATH_CALUDE_black_hen_day_probability_l4112_411292


namespace NUMINAMATH_CALUDE_z₂_value_l4112_411260

-- Define the complex numbers
variable (z₁ z₂ : ℂ)

-- Define the conditions
axiom h₁ : (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I
axiom h₂ : z₂.im = 2
axiom h₃ : (z₁ * z₂).im = 0

-- Theorem statement
theorem z₂_value : z₂ = 4 + 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_z₂_value_l4112_411260


namespace NUMINAMATH_CALUDE_first_share_rate_is_9_percent_l4112_411217

-- Define the total investment
def total_investment : ℝ := 10000

-- Define the interest rate of the second share
def second_share_rate : ℝ := 0.11

-- Define the total interest rate after one year
def total_interest_rate : ℝ := 0.0975

-- Define the amount invested in the second share
def second_share_investment : ℝ := 3750

-- Define the amount invested in the first share
def first_share_investment : ℝ := total_investment - second_share_investment

-- Theorem: The interest rate of the first share is 9%
theorem first_share_rate_is_9_percent :
  ∃ r : ℝ, r = 0.09 ∧
  r * first_share_investment + second_share_rate * second_share_investment =
  total_interest_rate * total_investment :=
by sorry

end NUMINAMATH_CALUDE_first_share_rate_is_9_percent_l4112_411217


namespace NUMINAMATH_CALUDE_bounded_expression_l4112_411286

theorem bounded_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end NUMINAMATH_CALUDE_bounded_expression_l4112_411286


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l4112_411293

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l4112_411293


namespace NUMINAMATH_CALUDE_tracy_candies_l4112_411238

theorem tracy_candies (x : ℕ) (h1 : x % 4 = 0) 
  (h2 : x / 2 % 2 = 0) 
  (h3 : 4 ≤ x / 2 - 20) (h4 : x / 2 - 20 ≤ 8) 
  (h5 : ∃ (b : ℕ), 2 ≤ b ∧ b ≤ 6 ∧ x / 2 - 20 - b = 4) : x = 48 := by
  sorry

end NUMINAMATH_CALUDE_tracy_candies_l4112_411238


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l4112_411226

theorem polar_to_cartesian :
  let r : ℝ := 2
  let θ : ℝ := π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = Real.sqrt 3 ∧ y = 1) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l4112_411226


namespace NUMINAMATH_CALUDE_night_crew_ratio_l4112_411207

theorem night_crew_ratio (D N : ℕ) (B : ℝ) (h1 : D > 0) (h2 : N > 0) (h3 : B > 0) :
  (D * B) / ((D * B) + (N * (B / 2))) = 5 / 7 →
  (N : ℝ) / D = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_night_crew_ratio_l4112_411207


namespace NUMINAMATH_CALUDE_decagon_triangles_l4112_411239

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- Theorem stating that the number of triangles formed by the vertices of a regular decagon is 120 -/
theorem decagon_triangles : trianglesInDecagon = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l4112_411239


namespace NUMINAMATH_CALUDE_sqrt_simplification_l4112_411209

theorem sqrt_simplification (a : ℝ) (ha : a > 0) : a^2 * Real.sqrt a = a^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l4112_411209


namespace NUMINAMATH_CALUDE_v_2010_equals_0_l4112_411295

-- Define the function g
def g : ℕ → ℕ
| 0 => 2
| 1 => 5
| 2 => 3
| 3 => 0
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, although not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 0
| (n + 1) => g (v n)

-- Theorem to prove
theorem v_2010_equals_0 : v 2010 = 0 := by
  sorry

end NUMINAMATH_CALUDE_v_2010_equals_0_l4112_411295


namespace NUMINAMATH_CALUDE_intersection_line_slope_l4112_411201

/-- Given two lines y = 4 - 3x and y = 2x - 1, and a third line y = ax + 7 that passes through 
    their intersection point, prove that a = -6. -/
theorem intersection_line_slope (a : ℝ) : 
  (∃ x y : ℝ, y = 4 - 3*x ∧ y = 2*x - 1 ∧ y = a*x + 7) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l4112_411201


namespace NUMINAMATH_CALUDE_symmetric_abs_function_l4112_411213

/-- A function f is symmetric about a point c if f(c+x) = f(c-x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetric_abs_function (m n : ℝ) :
  SymmetricAbout (fun x ↦ |x + m| + |n * x + 1|) 2 → m + n = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_abs_function_l4112_411213


namespace NUMINAMATH_CALUDE_line_point_x_coordinate_l4112_411265

/-- Given a line passing through (10, 3) with x-intercept 4,
    the x-coordinate of the point on this line with y-coordinate -3 is -2. -/
theorem line_point_x_coordinate 
  (line : ℝ → ℝ) 
  (passes_through_10_3 : line 10 = 3)
  (x_intercept_4 : line 4 = 0) :
  ∃ x : ℝ, line x = -3 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_point_x_coordinate_l4112_411265


namespace NUMINAMATH_CALUDE_angle_B_measure_l4112_411214

theorem angle_B_measure :
  ∀ (A B : ℝ),
  A + B = 180 →  -- complementary angles sum to 180°
  B = 4 * A →    -- B is 4 times A
  B = 144 :=     -- B measures 144°
by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l4112_411214


namespace NUMINAMATH_CALUDE_count_square_family_with_range_14_l4112_411230

/-- A function family is characterized by its analytic expression and range -/
structure FunctionFamily where
  expression : ℝ → ℝ
  range : Set ℝ

/-- Count the number of functions in a family with different domains -/
def countFunctionsInFamily (f : FunctionFamily) : ℕ :=
  sorry

/-- The specific function family we're interested in -/
def squareFamilyWithRange14 : FunctionFamily :=
  { expression := fun x ↦ x^2,
    range := {1, 4} }

/-- Theorem stating that the number of functions in our specific family is 9 -/
theorem count_square_family_with_range_14 :
  countFunctionsInFamily squareFamilyWithRange14 = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_square_family_with_range_14_l4112_411230


namespace NUMINAMATH_CALUDE_symmetric_point_about_line_l4112_411246

/-- The symmetric point of (x₁, y₁) about the line ax + by + c = 0 is (x₂, y₂) -/
def is_symmetric_point (x₁ y₁ x₂ y₂ a b c : ℝ) : Prop :=
  -- The line connecting the points is perpendicular to the line of symmetry
  (y₂ - y₁) * a = -(x₂ - x₁) * b ∧
  -- The midpoint of the two points lies on the line of symmetry
  (a * ((x₁ + x₂) / 2) + b * ((y₁ + y₂) / 2) + c = 0)

theorem symmetric_point_about_line :
  is_symmetric_point (-1) 2 (-6) (-3) 1 1 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_about_line_l4112_411246


namespace NUMINAMATH_CALUDE_circle_equation_correct_l4112_411216

/-- The standard equation of a circle with center (a, b) and radius r -/
def CircleEquation (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

/-- Theorem: The equation (x - 2)^2 + (y + 1)^2 = 4 represents a circle with center (2, -1) and radius 2 -/
theorem circle_equation_correct :
  ∀ x y : ℝ, CircleEquation x y 2 (-1) 2 ↔ (x - 2)^2 + (y + 1)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l4112_411216


namespace NUMINAMATH_CALUDE_two_digit_sum_square_property_l4112_411227

/-- Returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Checks if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- The set of numbers satisfying the condition -/
def validSet : Set ℕ :=
  {10, 20, 11, 30, 21, 12, 31, 22, 13}

/-- The main theorem -/
theorem two_digit_sum_square_property (A : ℕ) :
  isTwoDigit A →
  ((sumOfDigits A)^2 = sumOfDigits (A^2)) ↔ A ∈ validSet :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_square_property_l4112_411227


namespace NUMINAMATH_CALUDE_octagonal_pyramid_sum_l4112_411202

-- Define the structure of an octagonal pyramid
structure OctagonalPyramid where
  base_vertices : Nat
  base_edges : Nat
  triangular_faces : Nat
  apex_vertex : Nat
  apex_edges : Nat

-- Define the properties of an octagonal pyramid
def is_octagonal_pyramid (p : OctagonalPyramid) : Prop :=
  p.base_vertices = 8 ∧
  p.base_edges = 8 ∧
  p.triangular_faces = 8 ∧
  p.apex_vertex = 1 ∧
  p.apex_edges = 8

-- Calculate the total number of faces
def total_faces (p : OctagonalPyramid) : Nat :=
  1 + p.triangular_faces

-- Calculate the total number of edges
def total_edges (p : OctagonalPyramid) : Nat :=
  p.base_edges + p.apex_edges

-- Calculate the total number of vertices
def total_vertices (p : OctagonalPyramid) : Nat :=
  p.base_vertices + p.apex_vertex

-- Theorem: The sum of faces, edges, and vertices of an octagonal pyramid is 34
theorem octagonal_pyramid_sum (p : OctagonalPyramid) 
  (h : is_octagonal_pyramid p) : 
  total_faces p + total_edges p + total_vertices p = 34 := by
  sorry

end NUMINAMATH_CALUDE_octagonal_pyramid_sum_l4112_411202


namespace NUMINAMATH_CALUDE_f_min_max_l4112_411241

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem f_min_max :
  let I : Set ℝ := Set.Icc 0 (2 * Real.pi)
  ∃ (min_val max_val : ℝ),
    (∀ x ∈ I, f x ≥ min_val) ∧
    (∀ x ∈ I, f x ≤ max_val) ∧
    (∃ x₁ ∈ I, f x₁ = min_val) ∧
    (∃ x₂ ∈ I, f x₂ = max_val) ∧
    min_val = -3 * Real.pi / 2 ∧
    max_val = Real.pi / 2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_f_min_max_l4112_411241


namespace NUMINAMATH_CALUDE_at_least_13_blondes_identifiable_l4112_411287

/-- Represents a woman in the factory -/
inductive Woman
| Blonde
| Brunette

/-- The total number of women in the factory -/
def total_women : ℕ := 217

/-- The number of brunettes in the factory -/
def num_brunettes : ℕ := 17

/-- The number of blondes in the factory -/
def num_blondes : ℕ := 200

/-- The number of women each woman lists as blonde -/
def list_size : ℕ := 200

/-- A function representing a woman's list of supposed blondes -/
def list_blondes (w : Woman) : Finset Woman := sorry

theorem at_least_13_blondes_identifiable :
  ∃ (identified_blondes : Finset Woman),
    (∀ w ∈ identified_blondes, w = Woman.Blonde) ∧
    identified_blondes.card ≥ 13 := by sorry

end NUMINAMATH_CALUDE_at_least_13_blondes_identifiable_l4112_411287


namespace NUMINAMATH_CALUDE_specific_quilt_shaded_fraction_l4112_411256

/-- Represents a square quilt composed of smaller squares -/
structure Quilt :=
  (side_length : ℕ)
  (shaded_triangles : ℕ)
  (shaded_squares : ℕ)

/-- Calculates the fraction of a quilt that is shaded -/
def shaded_fraction (q : Quilt) : ℚ :=
  let total_area : ℚ := (q.side_length * q.side_length : ℚ)
  let shaded_area : ℚ := (q.shaded_squares : ℚ) + (q.shaded_triangles : ℚ) / 2
  shaded_area / total_area

/-- Theorem stating that for a specific quilt configuration, the shaded fraction is 5/18 -/
theorem specific_quilt_shaded_fraction :
  let q := Quilt.mk 3 3 1
  shaded_fraction q = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_specific_quilt_shaded_fraction_l4112_411256


namespace NUMINAMATH_CALUDE_infinite_points_in_region_l4112_411221

/-- The set of points with positive rational coordinates satisfying x + 2y ≤ 6 is infinite -/
theorem infinite_points_in_region : 
  Set.Infinite {p : ℚ × ℚ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 + 2 * p.2 ≤ 6} := by sorry

end NUMINAMATH_CALUDE_infinite_points_in_region_l4112_411221


namespace NUMINAMATH_CALUDE_calculation_proofs_l4112_411250

theorem calculation_proofs :
  (∃ (x : ℝ), x = (1/2 * Real.sqrt 24 - 2 * Real.sqrt 2 * Real.sqrt 3) ∧ x = -Real.sqrt 6) ∧
  (∃ (y : ℝ), y = ((Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) + Real.sqrt 8 - Real.sqrt (9/2)) ∧ y = -1 + Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proofs_l4112_411250


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l4112_411251

/-- The length of the diagonal of a rectangular solid with edges of length 2, 3, and 4 is √29. -/
theorem rectangular_solid_diagonal : 
  let a : ℝ := 2
  let b : ℝ := 3
  let c : ℝ := 4
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l4112_411251


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l4112_411294

/-- Given a train of length 100 meters traveling at 45 km/hr that crosses a bridge in 30 seconds, 
    the length of the bridge is 275 meters. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 275 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l4112_411294


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l4112_411219

theorem quadratic_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁^2 + p*r₁ + 15 = 0 → r₂^2 + p*r₂ + 15 = 0 → |r₁ + r₂| > 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l4112_411219


namespace NUMINAMATH_CALUDE_cubic_inequality_l4112_411222

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a*b^2 + a^2*b + b*c^2 + b^2*c + a*c^2 + a^2*c := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l4112_411222


namespace NUMINAMATH_CALUDE_divisible_by_six_l4112_411255

theorem divisible_by_six (n : ℤ) : ∃ k : ℤ, n^3 + 5*n = 6*k := by sorry

end NUMINAMATH_CALUDE_divisible_by_six_l4112_411255


namespace NUMINAMATH_CALUDE_vector_perpendicular_problem_l4112_411232

theorem vector_perpendicular_problem (x : ℝ) : 
  let a : ℝ × ℝ := (-2, x)
  let b : ℝ × ℝ := (1, Real.sqrt 3)
  (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 → x = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_problem_l4112_411232


namespace NUMINAMATH_CALUDE_painting_cost_cny_l4112_411220

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℚ := 7

/-- Cost of the painting in Namibian dollars -/
def painting_cost_nad : ℚ := 160

/-- Theorem stating the cost of the painting in Chinese yuan -/
theorem painting_cost_cny : 
  (painting_cost_nad / usd_to_nad) * usd_to_cny = 140 := by
  sorry

end NUMINAMATH_CALUDE_painting_cost_cny_l4112_411220


namespace NUMINAMATH_CALUDE_tan_sum_equation_l4112_411267

theorem tan_sum_equation : ∀ (x y : Real),
  x + y = 60 * π / 180 →
  Real.tan (60 * π / 180) = Real.sqrt 3 →
  Real.tan x + Real.tan y + Real.sqrt 3 * Real.tan x * Real.tan y = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_equation_l4112_411267


namespace NUMINAMATH_CALUDE_pond_diameter_l4112_411205

/-- The diameter of a circular pond given specific conditions -/
theorem pond_diameter : ∃ (h k r : ℝ),
  (4 - h)^2 + (11 - k)^2 = r^2 ∧
  (12 - h)^2 + (9 - k)^2 = r^2 ∧
  (2 - h)^2 + (7 - k)^2 = (r - 1)^2 ∧
  2 * r = 9.2 := by
  sorry

end NUMINAMATH_CALUDE_pond_diameter_l4112_411205


namespace NUMINAMATH_CALUDE_square_perimeter_side_length_l4112_411261

theorem square_perimeter_side_length (perimeter : ℝ) (side : ℝ) : 
  perimeter = 8 → side ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_side_length_l4112_411261


namespace NUMINAMATH_CALUDE_abs_diff_ge_one_l4112_411297

theorem abs_diff_ge_one (a b c : ℝ) 
  (sum_eq : a + b + c = 2) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 2) : 
  max (|a - b|) (max (|b - c|) (|c - a|)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_ge_one_l4112_411297


namespace NUMINAMATH_CALUDE_exponential_inequality_l4112_411288

theorem exponential_inequality (x y a b : ℝ) (h1 : x > y) (h2 : y > a) (h3 : a > b) (h4 : b > 1) :
  a^x > b^y := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l4112_411288


namespace NUMINAMATH_CALUDE_three_planes_solutions_two_planes_one_point_solutions_one_plane_two_points_solutions_three_points_solutions_l4112_411268

-- Define the basic types
structure Point
structure Plane

-- Define the distance function
def distance (p : Point) (x : Point ⊕ Plane) : ℝ := sorry

-- Define the function to count solutions
def countSolutions (objects : List (Point ⊕ Plane)) (d : ℝ) : ℕ := sorry

-- Theorem for case (a)
theorem three_planes_solutions (p1 p2 p3 : Plane) (d : ℝ) :
  countSolutions [Sum.inr p1, Sum.inr p2, Sum.inr p3] d = 8 := sorry

-- Theorem for case (b)
theorem two_planes_one_point_solutions (p1 p2 : Plane) (pt : Point) (d : ℝ) :
  countSolutions [Sum.inr p1, Sum.inr p2, Sum.inl pt] d = 8 := sorry

-- Theorem for case (c)
theorem one_plane_two_points_solutions (p : Plane) (pt1 pt2 : Point) (d : ℝ) :
  countSolutions [Sum.inr p, Sum.inl pt1, Sum.inl pt2] d = 4 := sorry

-- Theorem for case (d)
theorem three_points_solutions (pt1 pt2 pt3 : Point) (d : ℝ) :
  let n := countSolutions [Sum.inl pt1, Sum.inl pt2, Sum.inl pt3] d
  n = 0 ∨ n = 1 ∨ n = 2 := sorry

end NUMINAMATH_CALUDE_three_planes_solutions_two_planes_one_point_solutions_one_plane_two_points_solutions_three_points_solutions_l4112_411268


namespace NUMINAMATH_CALUDE_group_size_l4112_411298

theorem group_size (n : ℕ) 
  (avg_increase : ℝ) 
  (old_weight new_weight : ℝ) 
  (h1 : avg_increase = 1.5)
  (h2 : old_weight = 65)
  (h3 : new_weight = 74)
  (h4 : n * avg_increase = new_weight - old_weight) : 
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_group_size_l4112_411298


namespace NUMINAMATH_CALUDE_line_point_k_value_l4112_411218

/-- A line contains the points (6,8), (-2,k), and (-10,4). Prove that k = 6. -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 8 = m * 6 + b ∧ k = m * (-2) + b ∧ 4 = m * (-10) + b) → k = 6 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l4112_411218


namespace NUMINAMATH_CALUDE_divisibility_implication_l4112_411281

theorem divisibility_implication (n m : ℤ) : 
  (31 ∣ (6 * n + 11 * m)) → (31 ∣ (n + 7 * m)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_implication_l4112_411281


namespace NUMINAMATH_CALUDE_factor_polynomial_l4112_411262

theorem factor_polynomial (x : ℝ) : 45 * x^3 - 135 * x^7 = 45 * x^3 * (1 - 3 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l4112_411262


namespace NUMINAMATH_CALUDE_proportional_function_and_point_l4112_411210

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 2

-- State the theorem
theorem proportional_function_and_point :
  -- Conditions
  (∃ k : ℝ, ∀ x y : ℝ, y - 2 = k * x) →  -- y-2 is directly proportional to x
  (f 1 = 6) →                           -- When x=1, y=6
  (f (-3/4) = -1) →                     -- Point P(a, -1) is on the graph of the function
  -- Conclusions
  ((∀ x : ℝ, f x = 4 * x + 2) ∧         -- The function expression is y = 4x + 2
   (-3/4 : ℝ) = -3/4)                   -- The value of a is -3/4
  := by sorry

end NUMINAMATH_CALUDE_proportional_function_and_point_l4112_411210


namespace NUMINAMATH_CALUDE_modulus_of_complex_expression_l4112_411233

theorem modulus_of_complex_expression : 
  Complex.abs ((1 - 2 * Complex.I)^2 / Complex.I) = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_expression_l4112_411233


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l4112_411243

/-- Given an arithmetic sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

/-- The theorem states that for an arithmetic sequence where a₄ = 2a₃, 
    the ratio of S₇ to S₅ is equal to 14/5 -/
theorem arithmetic_sequence_sum_ratio 
  (a : ℕ → ℚ) 
  (h : a 4 = 2 * a 3) : 
  S 7 a / S 5 a = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l4112_411243


namespace NUMINAMATH_CALUDE_max_inequality_constant_l4112_411206

theorem max_inequality_constant : ∃ (M : ℝ), (∀ (x y : ℝ), x + y ≥ 0 → 
  (x^2 + y^2)^3 ≥ M * (x^3 + y^3) * (x*y - x - y)) ∧ 
  (∀ (M' : ℝ), (∀ (x y : ℝ), x + y ≥ 0 → 
    (x^2 + y^2)^3 ≥ M' * (x^3 + y^3) * (x*y - x - y)) → M' ≤ M) ∧
  M = 32 :=
by sorry

end NUMINAMATH_CALUDE_max_inequality_constant_l4112_411206


namespace NUMINAMATH_CALUDE_intersection_M_N_l4112_411231

-- Define set M
def M : Set ℝ := {x | x^2 ≥ x}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 3^x + 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4112_411231


namespace NUMINAMATH_CALUDE_pet_food_difference_l4112_411236

theorem pet_food_difference (dog_food cat_food : ℕ) 
  (h1 : dog_food = 600) 
  (h2 : cat_food = 327) : 
  dog_food - cat_food = 273 := by
sorry

end NUMINAMATH_CALUDE_pet_food_difference_l4112_411236


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l4112_411248

/-- Represents the points on the circle --/
inductive Point
| One
| Two
| Three
| Four
| Five
| Six

/-- Defines the next point for a jump --/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.One => Point.Three
  | Point.Two => Point.Three
  | Point.Three => Point.Five
  | Point.Four => Point.Five
  | Point.Five => Point.One
  | Point.Six => Point.One

/-- Calculates the position after n jumps --/
def positionAfterJumps (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | Nat.succ m => nextPoint (positionAfterJumps start m)

theorem bug_position_after_2023_jumps :
  positionAfterJumps Point.Six 2023 = Point.One := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l4112_411248


namespace NUMINAMATH_CALUDE_fraction_division_addition_l4112_411253

theorem fraction_division_addition : (3 / 7 : ℚ) / 4 + 2 / 7 = 11 / 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_addition_l4112_411253


namespace NUMINAMATH_CALUDE_horner_V₁_value_l4112_411254

/-- Horner's method for polynomial evaluation -/
def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = 3x⁴ + 2x² + x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

/-- V₁ in Horner's method for f(x) at x = 10 -/
def V₁ : ℝ := horner_step 2 10 3

theorem horner_V₁_value : V₁ = 32 := by sorry

end NUMINAMATH_CALUDE_horner_V₁_value_l4112_411254


namespace NUMINAMATH_CALUDE_simplification_equivalence_simplified_is_quadratic_trinomial_l4112_411271

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ := 2*x^2 - 5*x + x^2 - 4*x + 5

-- Define the simplified polynomial
def simplified_polynomial (x : ℝ) : ℝ := 3*x^2 - 9*x + 5

-- Theorem stating that the simplified polynomial is equivalent to the original
theorem simplification_equivalence :
  ∀ x, original_polynomial x = simplified_polynomial x :=
by sorry

-- Define what it means for a polynomial to be quadratic
def is_quadratic (p : ℝ → ℝ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ ∀ x, p x = a*x^2 + b*x + c

-- Define what it means for a polynomial to have exactly three terms
def has_three_terms (p : ℝ → ℝ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ∀ x, p x = a*x^2 + b*x + c

-- Theorem stating that the simplified polynomial is a quadratic trinomial
theorem simplified_is_quadratic_trinomial :
  is_quadratic simplified_polynomial ∧ has_three_terms simplified_polynomial :=
by sorry

end NUMINAMATH_CALUDE_simplification_equivalence_simplified_is_quadratic_trinomial_l4112_411271


namespace NUMINAMATH_CALUDE_video_recorder_markup_percentage_l4112_411299

/-- Proves that the markup percentage is 20% given the problem conditions -/
theorem video_recorder_markup_percentage
  (wholesale_cost : ℝ)
  (employee_price : ℝ)
  (employee_discount : ℝ)
  (markup_percentage : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : employee_price = 192)
  (h3 : employee_discount = 0.20)
  (h4 : employee_price = (1 - employee_discount) * (wholesale_cost * (1 + markup_percentage / 100)))
  : markup_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_markup_percentage_l4112_411299


namespace NUMINAMATH_CALUDE_hiking_distance_proof_l4112_411257

theorem hiking_distance_proof (total_distance : ℝ) : 
  (total_distance / 3 + (2 * total_distance / 3) / 3 + (4 * total_distance / 9) / 4 + 24 = total_distance) →
  (total_distance = 72 ∧ 
   total_distance / 3 = 24 ∧ 
   (2 * total_distance / 3) / 3 = 16 ∧ 
   (4 * total_distance / 9) / 4 = 8) :=
by
  sorry

#check hiking_distance_proof

end NUMINAMATH_CALUDE_hiking_distance_proof_l4112_411257


namespace NUMINAMATH_CALUDE_lines_parallel_in_intersecting_planes_l4112_411276

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation between a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- State the theorem
theorem lines_parallel_in_intersecting_planes
  (l m n : Line) (α β γ : Plane)
  (distinct_lines : l ≠ m ∧ m ≠ n ∧ n ≠ l)
  (distinct_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)
  (h1 : intersect α β = l)
  (h2 : intersect β γ = m)
  (h3 : intersect γ α = n)
  (h4 : lineParallelPlane l γ) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_in_intersecting_planes_l4112_411276


namespace NUMINAMATH_CALUDE_number_of_men_l4112_411224

theorem number_of_men (max_handshakes : ℕ) : max_handshakes = 153 → ∃ n : ℕ, n = 18 ∧ max_handshakes = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_l4112_411224


namespace NUMINAMATH_CALUDE_aquarium_solution_l4112_411289

def aquarium_animals (otters seals sea_lions : ℕ) : Prop :=
  (otters + seals = 7 ∨ otters = 7 ∨ seals = 7) ∧
  (sea_lions + seals = 6 ∨ sea_lions = 6 ∨ seals = 6) ∧
  (otters + sea_lions = 5 ∨ otters = 5 ∨ sea_lions = 5) ∧
  (otters ≤ seals ∨ seals ≤ otters) ∧
  (otters ≤ sea_lions ∧ seals ≤ sea_lions)

theorem aquarium_solution :
  ∃! (otters seals sea_lions : ℕ),
    aquarium_animals otters seals sea_lions ∧
    otters = 5 ∧ seals = 7 ∧ sea_lions = 6 :=
sorry

end NUMINAMATH_CALUDE_aquarium_solution_l4112_411289


namespace NUMINAMATH_CALUDE_min_lines_8x8_grid_l4112_411235

/-- Represents a grid with points at the center of each square -/
structure Grid :=
  (size : ℕ)
  (points : ℕ)

/-- Calculates the minimum number of lines needed to separate all points in a grid -/
def min_lines (g : Grid) : ℕ :=
  2 * (g.size - 1)

/-- Theorem: For an 8x8 grid with 64 points, the minimum number of lines to separate all points is 14 -/
theorem min_lines_8x8_grid :
  let g : Grid := ⟨8, 64⟩
  min_lines g = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_lines_8x8_grid_l4112_411235


namespace NUMINAMATH_CALUDE_sqrt_simplification_l4112_411204

theorem sqrt_simplification :
  (Real.sqrt 18 - Real.sqrt 2) / Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l4112_411204


namespace NUMINAMATH_CALUDE_average_youtube_viewer_videos_l4112_411245

theorem average_youtube_viewer_videos (video_length : ℕ) (ad_time : ℕ) (total_time : ℕ) :
  video_length = 7 →
  ad_time = 3 →
  total_time = 17 →
  ∃ (num_videos : ℕ), num_videos * video_length + ad_time = total_time ∧ num_videos = 2 :=
by sorry

end NUMINAMATH_CALUDE_average_youtube_viewer_videos_l4112_411245


namespace NUMINAMATH_CALUDE_intersection_sum_l4112_411252

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 5*x + 2
def g (x y : ℝ) : Prop := x + 5*y = 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {p | f p.1 = p.2 ∧ g p.1 p.2}

-- State the theorem
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 0 ∧
    y₁ + y₂ + y₃ = 3 :=
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l4112_411252


namespace NUMINAMATH_CALUDE_divisibility_sequence_l4112_411278

theorem divisibility_sequence (t : ℤ) (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ (p : ℤ) ∣ ((3 - 7*t) * 2^n + (18*t - 9) * 3^n + (6 - 10*t) * 4^n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_sequence_l4112_411278


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l4112_411229

/-- Given a cistern with specific properties, prove the time it takes to empty -/
theorem cistern_emptying_time 
  (capacity : ℝ)
  (leak_empty_time : ℝ)
  (tap_rate : ℝ)
  (h1 : capacity = 480)
  (h2 : leak_empty_time = 20)
  (h3 : tap_rate = 4)
  : (capacity / (capacity / leak_empty_time - tap_rate) = 24) :=
by
  sorry

end NUMINAMATH_CALUDE_cistern_emptying_time_l4112_411229


namespace NUMINAMATH_CALUDE_equation_solution_l4112_411291

theorem equation_solution :
  ∃! x : ℚ, (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 :=
by
  use (-2/3)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4112_411291


namespace NUMINAMATH_CALUDE_sqrt_five_power_calculation_l4112_411242

theorem sqrt_five_power_calculation : 
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 125 * 5 ^ (3/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_power_calculation_l4112_411242


namespace NUMINAMATH_CALUDE_workshop_average_salary_l4112_411259

/-- Proves that the average salary of all workers in a workshop is 8000,
    given the specified conditions. -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℕ)
  (other_salary : ℕ)
  (h1 : total_workers = 49)
  (h2 : technicians = 7)
  (h3 : technician_salary = 20000)
  (h4 : other_salary = 6000) :
  (technicians * technician_salary + (total_workers - technicians) * other_salary) / total_workers = 8000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l4112_411259


namespace NUMINAMATH_CALUDE_function_properties_unique_proportional_function_l4112_411258

/-- A proportional function passing through the point (3, 6) -/
def f (x : ℝ) : ℝ := 2 * x

/-- Theorem stating the properties of the function f -/
theorem function_properties :
  (f 3 = 6) ∧
  (f 4 ≠ -2) ∧
  (f (-1.5) ≠ 3) := by
  sorry

/-- Theorem proving that f is the unique proportional function passing through (3, 6) -/
theorem unique_proportional_function (g : ℝ → ℝ) (h : ∀ x, g x = k * x) :
  g 3 = 6 → g = f := by
  sorry

end NUMINAMATH_CALUDE_function_properties_unique_proportional_function_l4112_411258


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l4112_411270

theorem max_sqrt_sum (x : ℝ) (h : 0 ≤ x ∧ x ≤ 20) :
  Real.sqrt (x + 16) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ 8 * Real.sqrt 3 / 3 ∧
  ∃ y : ℝ, 0 ≤ y ∧ y ≤ 20 ∧ Real.sqrt (y + 16) + Real.sqrt (20 - y) + 2 * Real.sqrt y = 8 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l4112_411270


namespace NUMINAMATH_CALUDE_solve_cab_driver_income_l4112_411266

def cab_driver_income_problem (day1 day2 day3 day5 average : ℕ) : Prop :=
  let total := 5 * average
  let known_sum := day1 + day2 + day3 + day5
  let day4 := total - known_sum
  (day1 = 45 ∧ day2 = 50 ∧ day3 = 60 ∧ day5 = 70 ∧ average = 58) →
  day4 = 65

theorem solve_cab_driver_income :
  cab_driver_income_problem 45 50 60 70 58 :=
sorry

end NUMINAMATH_CALUDE_solve_cab_driver_income_l4112_411266


namespace NUMINAMATH_CALUDE_bolt_nut_balance_l4112_411277

theorem bolt_nut_balance (total_workers : ℕ) (bolts_per_worker : ℕ) (nuts_per_worker : ℕ) 
  (nuts_per_bolt : ℕ) (bolt_workers : ℕ) : 
  total_workers = 56 →
  bolts_per_worker = 16 →
  nuts_per_worker = 24 →
  nuts_per_bolt = 2 →
  bolt_workers ≤ total_workers →
  (2 * bolts_per_worker * bolt_workers = nuts_per_worker * (total_workers - bolt_workers)) ↔
  (bolts_per_worker * bolt_workers * nuts_per_bolt = nuts_per_worker * (total_workers - bolt_workers)) :=
by sorry

end NUMINAMATH_CALUDE_bolt_nut_balance_l4112_411277


namespace NUMINAMATH_CALUDE_complex_equation_sum_l4112_411240

theorem complex_equation_sum (x y : ℝ) :
  (x - 3*y : ℂ) + (2*x + 3*y)*I = 5 + I →
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l4112_411240


namespace NUMINAMATH_CALUDE_five_from_six_circular_seating_l4112_411275

/-- The number of ways to seat 5 people from a group of 6 around a circular table -/
def circular_seating_arrangements (total_people : ℕ) (seated_people : ℕ) : ℕ :=
  (total_people.choose seated_people) * (seated_people - 1).factorial

/-- Theorem stating that the number of ways to seat 5 people from a group of 6 around a circular table is 144 -/
theorem five_from_six_circular_seating :
  circular_seating_arrangements 6 5 = 144 := by
  sorry

end NUMINAMATH_CALUDE_five_from_six_circular_seating_l4112_411275


namespace NUMINAMATH_CALUDE_point_D_transformation_l4112_411272

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def transform_point (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_axis (reflect_x_axis (rotate_90_clockwise p))

theorem point_D_transformation :
  transform_point (4, -3) = (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_D_transformation_l4112_411272


namespace NUMINAMATH_CALUDE_tom_typing_time_l4112_411234

/-- Calculates the time required to type a given number of pages -/
def typing_time (words_per_minute : ℕ) (words_per_page : ℕ) (num_pages : ℕ) : ℕ :=
  (words_per_page * num_pages) / words_per_minute

theorem tom_typing_time :
  typing_time 90 450 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tom_typing_time_l4112_411234


namespace NUMINAMATH_CALUDE_cookie_sharing_proof_l4112_411264

/-- Given a total number of cookies and the number of cookies each person gets,
    calculate the number of people sharing the cookies. -/
def number_of_people (total_cookies : ℕ) (cookies_per_person : ℕ) : ℕ :=
  total_cookies / cookies_per_person

/-- Prove that when sharing 24 cookies equally among people,
    with each person getting 4 cookies, the number of people is 6. -/
theorem cookie_sharing_proof :
  number_of_people 24 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sharing_proof_l4112_411264


namespace NUMINAMATH_CALUDE_monogram_count_l4112_411244

/-- The number of letters in the alphabet excluding 'A' -/
def n : ℕ := 25

/-- The number of letters we need to choose for middle and last initials -/
def k : ℕ := 2

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of ways to choose two distinct letters from 25 letters in alphabetical order is 300 -/
theorem monogram_count : choose n k = 300 := by sorry

end NUMINAMATH_CALUDE_monogram_count_l4112_411244


namespace NUMINAMATH_CALUDE_rotated_point_x_coordinate_l4112_411280

/-- Given a point P(1,2) in the Cartesian plane, prove that when the vector OP
    is rotated counterclockwise by 5π/6 around the origin O to obtain vector OQ,
    the x-coordinate of Q is -√3/2 - 2√5. -/
theorem rotated_point_x_coordinate (P Q : ℝ × ℝ) (h1 : P = (1, 2)) :
  (∃ θ : ℝ, θ = 5 * π / 6 ∧
   Q.1 = P.1 * Real.cos θ - P.2 * Real.sin θ ∧
   Q.2 = P.1 * Real.sin θ + P.2 * Real.cos θ) →
  Q.1 = -Real.sqrt 3 / 2 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rotated_point_x_coordinate_l4112_411280


namespace NUMINAMATH_CALUDE_division_problem_l4112_411223

theorem division_problem (divisor : ℕ) : 
  22 = divisor * 7 + 1 → divisor = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4112_411223


namespace NUMINAMATH_CALUDE_water_remaining_l4112_411215

theorem water_remaining (initial_water : ℚ) (used_water : ℚ) : 
  initial_water = 7/2 ∧ used_water = 7/3 → initial_water - used_water = 7/6 := by
  sorry

#check water_remaining

end NUMINAMATH_CALUDE_water_remaining_l4112_411215


namespace NUMINAMATH_CALUDE_power_calculation_l4112_411274

theorem power_calculation : (4^4 / 4^3) * 2^8 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l4112_411274


namespace NUMINAMATH_CALUDE_solution_set_nonempty_iff_m_in_range_inequality_holds_for_interval_iff_m_in_range_l4112_411290

-- Part 1
theorem solution_set_nonempty_iff_m_in_range (m : ℝ) :
  (∃ x : ℝ, m * x^2 + m * x + m - 6 < 0) ↔ m < 8 :=
sorry

-- Part 2
theorem inequality_holds_for_interval_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → m * x^2 - m * x < -m + 2) ↔ m < 2/7 :=
sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_iff_m_in_range_inequality_holds_for_interval_iff_m_in_range_l4112_411290


namespace NUMINAMATH_CALUDE_sandwich_non_condiments_percentage_l4112_411247

theorem sandwich_non_condiments_percentage 
  (total_weight : ℝ) 
  (condiments_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : condiments_weight = 50) : 
  (total_weight - condiments_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_non_condiments_percentage_l4112_411247


namespace NUMINAMATH_CALUDE_unique_solution_l4112_411283

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (a + b) - f (a * b) = f a * f b - 1

/-- The theorem stating that the only function satisfying the equation is f(n) = n + 1 -/
theorem unique_solution (f : ℤ → ℤ) (h : SatisfiesEquation f) :
  ∀ n : ℤ, f n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4112_411283


namespace NUMINAMATH_CALUDE_winning_percentage_correct_l4112_411279

/-- Represents the percentage of votes secured by the winning candidate -/
def winning_percentage : ℝ := 70

/-- Represents the total number of valid votes -/
def total_votes : ℕ := 455

/-- Represents the majority of votes by which the winning candidate won -/
def vote_majority : ℕ := 182

/-- Theorem stating that the winning percentage is correct given the conditions -/
theorem winning_percentage_correct : 
  (winning_percentage / 100 * total_votes : ℝ) - 
  ((100 - winning_percentage) / 100 * total_votes : ℝ) = vote_majority := by
  sorry

end NUMINAMATH_CALUDE_winning_percentage_correct_l4112_411279


namespace NUMINAMATH_CALUDE_possible_y_values_l4112_411269

-- Define the relationship between x and y
def relation (x y : ℝ) : Prop := x^2 = y - 5

-- Theorem statement
theorem possible_y_values :
  (∃ y : ℝ, relation (-7) y ∧ y = 54) ∧
  (∃ y : ℝ, relation 2 y ∧ y = 9) := by
  sorry

end NUMINAMATH_CALUDE_possible_y_values_l4112_411269


namespace NUMINAMATH_CALUDE_characterization_of_C_l4112_411200

def A : Set ℝ := {x | x^2 - 6*x + 8 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 4 = 0}
def C : Set ℝ := {m | B m ∩ A = B m}

theorem characterization_of_C : C = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_characterization_of_C_l4112_411200


namespace NUMINAMATH_CALUDE_function_range_l4112_411284

theorem function_range (x : ℝ) (h : x > 1) : 
  let y := x + 1 / (x - 1)
  (∀ x > 1, y ≥ 3) ∧ (∃ x > 1, y = 3) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l4112_411284


namespace NUMINAMATH_CALUDE_license_plate_count_l4112_411237

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The number of odd digits -/
def num_odd_digits : ℕ := 5

/-- The total number of license plates -/
def total_plates : ℕ := num_letters ^ 3 * num_even_digits * num_odd_digits * num_even_digits

theorem license_plate_count :
  total_plates = 2197000 := by sorry

end NUMINAMATH_CALUDE_license_plate_count_l4112_411237


namespace NUMINAMATH_CALUDE_characterization_of_S_l4112_411225

open Set

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 4 = 0
def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the set of a values that satisfy both p and q
def S : Set ℝ := {a : ℝ | p a ∧ q a}

-- State the theorem
theorem characterization_of_S : S = Iic (-2) := by sorry

end NUMINAMATH_CALUDE_characterization_of_S_l4112_411225


namespace NUMINAMATH_CALUDE_reciprocal_sum_one_l4112_411296

theorem reciprocal_sum_one (x y z : ℕ+) : 
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 1 ↔ 
  ((x = 2 ∧ y = 4 ∧ z = 4) ∨ 
   (x = 2 ∧ y = 3 ∧ z = 6) ∨ 
   (x = 3 ∧ y = 3 ∧ z = 3)) :=
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_one_l4112_411296


namespace NUMINAMATH_CALUDE_marcel_shopping_cost_l4112_411211

def pen_cost : ℝ := 4

def briefcase_cost : ℝ := 5 * pen_cost

def notebook_cost : ℝ := 2 * pen_cost

def calculator_cost : ℝ := 3 * notebook_cost

def total_cost_before_tax : ℝ := pen_cost + briefcase_cost + notebook_cost + calculator_cost

def tax_rate : ℝ := 0.1

def tax_amount : ℝ := tax_rate * total_cost_before_tax

def total_cost_with_tax : ℝ := total_cost_before_tax + tax_amount

theorem marcel_shopping_cost : total_cost_with_tax = 61.60 := by sorry

end NUMINAMATH_CALUDE_marcel_shopping_cost_l4112_411211


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l4112_411273

theorem surrounding_circles_radius (r : ℝ) : r = 4 := by
  -- Given a central circle of radius 2
  -- Surrounded by 4 circles of radius r
  -- The surrounding circles touch the central circle and each other
  -- We need to prove that r = 4
  sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l4112_411273


namespace NUMINAMATH_CALUDE_line_equation_proof_l4112_411203

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point (x, y) lies on the line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The line we're interested in -/
def ourLine : Line :=
  { slope := 2, yIntercept := 5 }

theorem line_equation_proof :
  (∀ x y : ℝ, ourLine.containsPoint x y ↔ -2 * x + y = 1) ∧
  ourLine.containsPoint (-2) 3 :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l4112_411203


namespace NUMINAMATH_CALUDE_intersection_A_B_when_m_3_range_of_m_when_A_subset_B_l4112_411249

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (6 + 5*x - x^2)}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | (x - 1 + m) * (x - 1 - m) ≤ 0}

-- Theorem for part (1)
theorem intersection_A_B_when_m_3 : 
  A ∩ B 3 = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for part (2)
theorem range_of_m_when_A_subset_B : 
  ∀ m : ℝ, m > 0 → (A ⊆ B m) → m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_m_3_range_of_m_when_A_subset_B_l4112_411249


namespace NUMINAMATH_CALUDE_odot_inequality_range_l4112_411208

-- Define the operation ⊙
def odot (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem odot_inequality_range (a : ℝ) :
  (∀ x : ℝ, odot (x - a) (x + a) < 1) ↔ -1/2 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_odot_inequality_range_l4112_411208


namespace NUMINAMATH_CALUDE_percentage_calculation_l4112_411282

theorem percentage_calculation : 
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l4112_411282


namespace NUMINAMATH_CALUDE_inequality_proof_l4112_411212

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4112_411212


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l4112_411228

/-- Represents a domino tile with two numbers -/
structure Domino :=
  (upper : Nat)
  (lower : Nat)
  (upper_bound : upper ≤ 6)
  (lower_bound : lower ≤ 6)

/-- The set of all possible domino tiles -/
def dominoSet : Finset Domino := sorry

/-- The game state, including the numbers written on the blackboard and remaining tiles -/
structure GameState :=
  (written : Finset Nat)
  (remaining : Finset Domino)

/-- A player's strategy for selecting a domino -/
def Strategy := GameState → Option Domino

/-- Determines if a strategy is winning for the second player -/
def isWinningStrategy (s : Strategy) : Prop := sorry

/-- The main theorem stating that there exists a winning strategy for the second player -/
theorem second_player_winning_strategy :
  ∃ (s : Strategy), isWinningStrategy s := by sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l4112_411228


namespace NUMINAMATH_CALUDE_remainder_proof_l4112_411263

theorem remainder_proof : 123456789012 % 252 = 84 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l4112_411263
