import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l511_51126

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Side lengths
variable (S : ℝ) -- Area of the triangle

-- Define vectors
def vec_a : ℝ × ℝ := (Real.cos A, Real.cos B)
def vec_b : ℝ × ℝ := (a, 2*c - b)

-- State the theorem
theorem triangle_properties :
  (∃ (k : ℝ), vec_a A B = k • vec_b a b c) → -- Vectors are parallel
  b = 3 →
  S = 3 * Real.sqrt 3 →
  A = π / 3 ∧ a = Real.sqrt 13 := by
    sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l511_51126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_min_area_and_line_equation_l511_51144

/-- The line l: kx - y + 1 + 2k = 0 (k ∈ ℝ) -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

/-- Point that the line passes through -/
def fixed_point : ℝ × ℝ := (-2, 1)

/-- Area of the triangle formed by the line and the coordinate axes -/
noncomputable def triangle_area (k : ℝ) : ℝ := (2 + 1/k) * (2*k + 1) / 2

/-- Theorem stating that the line passes through a fixed point for all k -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_l k (fixed_point.1) (fixed_point.2) :=
by sorry

/-- Theorem stating the minimum area of the triangle and the corresponding line equation -/
theorem min_area_and_line_equation :
  (∀ k : ℝ, triangle_area k ≥ 4) ∧
  (∃ k : ℝ, triangle_area k = 4 ∧ line_l k = λ x y ↦ x - 2*y + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_min_area_and_line_equation_l511_51144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_not_same_shape_l511_51131

-- Define a type for the different options
inductive ShapeOption : Type
  | A : ShapeOption
  | B : ShapeOption
  | C : ShapeOption
  | D : ShapeOption

-- Define a predicate for whether an option represents figures of the same shape
def sameShape (o : ShapeOption) : Prop :=
  match o with
  | ShapeOption.A => True  -- Two photos from the same negative
  | ShapeOption.B => True  -- Original and magnified patterns
  | ShapeOption.C => False -- Profile and frontal photos
  | ShapeOption.D => True  -- Tree and its reflection

-- Theorem stating that C is the only option not representing the same shape
theorem only_C_not_same_shape :
  ∀ o : ShapeOption, ¬(sameShape o) ↔ o = ShapeOption.C :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_not_same_shape_l511_51131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_roots_l511_51106

-- Define the polynomial equation as noncomputable
noncomputable def P (t : ℝ) : ℝ := 32 * t^5 - 40 * t^3 + 10 * t - Real.sqrt 3

-- State the theorem
theorem cosine_roots :
  P (Real.cos (6 * π / 180)) = 0 →
  P (Real.cos (66 * π / 180)) = 0 ∧
  P (Real.cos (78 * π / 180)) = 0 ∧
  P (Real.cos (138 * π / 180)) = 0 ∧
  P (Real.cos (150 * π / 180)) = 0 :=
by
  sorry

-- You can add additional lemmas or helper functions here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_roots_l511_51106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l511_51100

def f : Set ℝ := Set.Icc (-10) 6

def g (x : ℝ) : ℝ := -3 * x

theorem domain_of_g :
  {x : ℝ | g x ∈ f} = Set.Icc (-2) (10/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l511_51100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_vectors_cos_angle_between_lines_l511_51173

noncomputable def vector1 : ℝ × ℝ := (4, 5)
noncomputable def vector2 : ℝ × ℝ := (2, 7)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem cos_angle_between_vectors (v w : ℝ × ℝ) :
  let θ := Real.arccos ((dot_product v w) / (magnitude v * magnitude w))
  (0 ≤ θ) ∧ (θ ≤ Real.pi / 2) →
  Real.cos θ = (dot_product v w) / (magnitude v * magnitude w) :=
by sorry

theorem cos_angle_between_lines :
  let θ := Real.arccos ((dot_product vector1 vector2) / (magnitude vector1 * magnitude vector2))
  (0 ≤ θ) ∧ (θ ≤ Real.pi / 2) →
  Real.cos θ = 43 / (Real.sqrt 41 * Real.sqrt 53) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_vectors_cos_angle_between_lines_l511_51173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_hold_l511_51115

theorem inequalities_hold (x y z a b c : ℝ) 
  (h1 : x < a) (h2 : y < b) (h3 : z < c) (h4 : x + y < a + b) :
  (x*y + y*z + z*x < a*b + b*c + c*a) ∧ 
  (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧ 
  (x*y*z < a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_hold_l511_51115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goose_eggs_calculation_goose_eggs_laid_l511_51104

/-- The number of goose eggs laid at a certain pond -/
def eggs_laid : ℕ := 630

/-- The fraction of eggs that hatched -/
def hatch_rate : ℚ := 2/3

/-- The fraction of hatched geese that survived the first month -/
def first_month_survival_rate : ℚ := 3/4

/-- The fraction of geese that survived the first month but did not survive the first year -/
def first_year_mortality_rate : ℚ := 3/5

/-- The number of geese that survived the first year -/
def geese_survived_first_year : ℕ := 126

theorem goose_eggs_calculation :
  (↑eggs_laid : ℚ) * hatch_rate * first_month_survival_rate * (1 - first_year_mortality_rate) = ↑geese_survived_first_year := by
  sorry

/-- Each egg produces at most one goose -/
axiom one_goose_per_egg : ∀ n : ℕ, n ≤ eggs_laid → n ≤ (↑eggs_laid * hatch_rate).floor

/-- The main theorem stating that 630 goose eggs were laid at the pond -/
theorem goose_eggs_laid : 
  ∃! n : ℕ, n = eggs_laid ∧ 
    (↑n : ℚ) * hatch_rate * first_month_survival_rate * (1 - first_year_mortality_rate) = ↑geese_survived_first_year ∧
    (∀ m : ℕ, m ≤ n → m ≤ (↑n * hatch_rate).floor) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goose_eggs_calculation_goose_eggs_laid_l511_51104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_for_sum_fractions_l511_51169

/-- The center of symmetry for a function f(x) --/
structure CenterOfSymmetry where
  x : ℝ
  y : ℝ

/-- A function representing the sum of fractions --/
noncomputable def sumFractions (n : ℕ) (x : ℝ) : ℝ :=
  (Finset.range n).sum (fun i => (x + i + 1) / (x + i))

/-- The theorem stating the center of symmetry for the given function --/
theorem center_of_symmetry_for_sum_fractions :
  /- Given the centers of symmetry for the first three simpler functions -/
  (CenterOfSymmetry.mk 0 0).x = 0 →
  (CenterOfSymmetry.mk (-1/2) 0).x = -1/2 →
  (CenterOfSymmetry.mk (-1) 0).x = -1 →
  /- The center of symmetry for the sum of 2019 fractions -/
  ∃ c : CenterOfSymmetry, 
    c.x = -1009 ∧ 
    c.y = 2019 ∧ 
    (∀ x : ℝ, sumFractions 2019 x = sumFractions 2019 (2 * c.x - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_for_sum_fractions_l511_51169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_l511_51154

-- Define the triangle ABC
def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  A + B + C = Real.pi ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0

-- State the theorem
theorem shortest_side_length 
  (A B C : Real) (a b c : Real) 
  (h_triangle : triangle_ABC A B C a b c)
  (h_B : B = Real.pi/4)  -- 45°
  (h_C : C = Real.pi/3)  -- 60°
  (h_c : c = 1) :
  b = Real.sqrt 6 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_l511_51154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_angle_l511_51146

open Real

theorem least_positive_angle (a b : ℝ) (x y : ℝ) 
  (h1 : tan x = a / b)
  (h2 : tan (x + y) = b / (a + b))
  (h3 : y = x)
  (h4 : a > 0)
  (h5 : b > 0) :
  x = arctan (1 / (a + 2)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_angle_l511_51146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_to_white_ratio_l511_51145

/-- Represents a square in the nested square diagram -/
structure NestedSquare where
  side_length : ℝ
  is_largest : Bool

/-- Represents the entire diagram of nested squares -/
structure NestedSquareDiagram where
  squares : List NestedSquare
  /-- Ensures that all squares except the largest have vertices at midpoints -/
  midpoint_property : ∀ s ∈ squares, ¬s.is_largest → 
    ∃ larger_s ∈ squares, s.side_length = larger_s.side_length / 2

/-- Represents a quarter of the entire diagram -/
structure QuarterSquare (d : NestedSquareDiagram) where

/-- The number of identical triangles in the shaded area of a quarter square -/
def shaded_triangles : ℕ := 5

/-- The number of identical triangles in the white area of a quarter square -/
def white_triangles : ℕ := 3

/-- The theorem stating the ratio of shaded to white area -/
theorem shaded_to_white_ratio (d : NestedSquareDiagram) :
  (shaded_triangles : ℚ) / (white_triangles : ℚ) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_to_white_ratio_l511_51145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l511_51133

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if sinA, sinB, sinC form a geometric sequence and c = 2a, then cosB = 3/4 -/
theorem triangle_cosine_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  c = 2 * a →
  (∃ r : ℝ, r ≠ 0 ∧ (Real.sin B)^2 = (Real.sin A) * (Real.sin C)) →
  Real.cos B = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l511_51133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_extremum_quadratic_minimum_quadratic_maximum_l511_51142

/-- A quadratic function with a specific condition on c -/
noncomputable def QuadraticFunction (a b : ℝ) : ℝ → ℝ := 
  fun x => a * x^2 + b * x - b^2 / (4 * a)

/-- Theorem stating that the quadratic function has either a minimum or maximum -/
theorem quadratic_extremum (a b : ℝ) (h : a ≠ 0) :
  (∀ x, QuadraticFunction a b x ≥ QuadraticFunction a b (-b / (2 * a)) ∧ a > 0) ∨
  (∀ x, QuadraticFunction a b x ≤ QuadraticFunction a b (-b / (2 * a)) ∧ a < 0) := by
  sorry

/-- Corollary: The quadratic function has a minimum if a > 0 -/
theorem quadratic_minimum (a b : ℝ) (h : a > 0) :
  ∀ x, QuadraticFunction a b x ≥ QuadraticFunction a b (-b / (2 * a)) := by
  sorry

/-- Corollary: The quadratic function has a maximum if a < 0 -/
theorem quadratic_maximum (a b : ℝ) (h : a < 0) :
  ∀ x, QuadraticFunction a b x ≤ QuadraticFunction a b (-b / (2 * a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_extremum_quadratic_minimum_quadratic_maximum_l511_51142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l511_51150

def M : Set ℕ := {1, 3, 5, 7, 9}

def N : Set ℕ := {x : ℕ | 2 * x > 7}

theorem intersection_M_N : M ∩ N = {5, 7, 9} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l511_51150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_cube_of_382_l511_51121

theorem closest_to_cube_of_382 : 
  ∀ x ∈ ({0.033, 0.040, 0.050, 0.060} : Set ℝ), 
    |0.382^3 - 0.037| < |0.382^3 - x| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_cube_of_382_l511_51121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l511_51101

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The line y = x -/
def line_y_eq_x (x y : ℝ) : Prop := y = x

/-- The first quadrant -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- A point on a circle -/
def on_circle (c : Circle) (x y : ℝ) : Prop :=
  c.equation x y

/-- The length of a chord intercepted by the x-axis -/
noncomputable def chord_length (c : Circle) : ℝ :=
  2 * Real.sqrt (c.radius^2 - c.center.2^2)

/-- The main theorem -/
theorem circle_equation_proof (c : Circle) :
  (∃ x y : ℝ, line_y_eq_x x y ∧ first_quadrant x y ∧ c.center = (x, y)) ∧
  on_circle c (-1) 2 ∧
  chord_length c = 4 * Real.sqrt 2 →
  c.center = (3, 3) ∧ c.radius^2 = 17 := by
  sorry

#check circle_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l511_51101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equality_l511_51125

noncomputable def f_B (x : ℝ) : ℝ := (Real.sqrt x)^2 / x
noncomputable def g_B (x : ℝ) : ℝ := x / (Real.sqrt x)^2

def f_D (x : ℝ) : ℝ := abs x
noncomputable def g_D (x : ℝ) : ℝ := Real.sqrt (x^2)

theorem functions_equality :
  (∀ x > 0, f_B x = g_B x) ∧
  (∀ x : ℝ, f_D x = g_D x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equality_l511_51125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l511_51167

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (ω * x) * Real.sin (ω * x + Real.pi / 2) - Real.cos (ω * x)^2 + 1 / 2

noncomputable def g (x : ℝ) : ℝ := Real.sin (1 / 2 * x + Real.pi / 6)

theorem function_properties (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi) = f ω x) :
  ω = 1 ∧ ∀ x, g x = f ω (x / 4 + Real.pi / 6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l511_51167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l511_51130

-- Define the original price and sale price
noncomputable def original_price : ℝ := 100
noncomputable def sale_price : ℝ := 55

-- Define the percent decrease formula
noncomputable def percent_decrease (original : ℝ) (sale : ℝ) : ℝ :=
  ((original - sale) / original) * 100

-- Theorem statement
theorem price_decrease_percentage :
  percent_decrease original_price sale_price = 45 := by
  -- Unfold the definitions
  unfold percent_decrease original_price sale_price
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l511_51130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_R2_l511_51139

-- Define the properties of rectangle R1
def side_R1 : ℝ := 4
def area_R1 : ℝ := 32

-- Define the diagonal of rectangle R2
def diagonal_R2 : ℝ := 20

-- Theorem to prove
theorem area_of_R2 : ∃ (side_a_R2 side_b_R2 : ℝ), side_a_R2 * side_b_R2 = 160 := by
  let other_side_R1 := area_R1 / side_R1
  let ratio := other_side_R1 / side_R1
  let side_a_R2 := Real.sqrt (diagonal_R2^2 / (1 + ratio^2))
  let side_b_R2 := ratio * side_a_R2
  use side_a_R2, side_b_R2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_R2_l511_51139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_perimeter_eq_third_circle_circumference_shaded_perimeter_approx_l511_51108

/-- The perimeter of the shaded area in a square with side length 1,
    where quarter circles with radius 1 are drawn at each corner. -/
noncomputable def shaded_perimeter : ℝ :=
  (1 / 3) * (2 * Real.pi)

/-- Theorem: The perimeter of the shaded area is 1/3 of the circumference of a circle with radius 1. -/
theorem shaded_perimeter_eq_third_circle_circumference :
  shaded_perimeter = (1 / 3) * (2 * Real.pi) := by
  -- Unfold the definition of shaded_perimeter
  unfold shaded_perimeter
  -- The equality follows directly from the definition
  rfl

/-- Theorem: The perimeter of the shaded area is approximately 2.094 (using π ≈ 3.141). -/
theorem shaded_perimeter_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |shaded_perimeter - 2.094| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_perimeter_eq_third_circle_circumference_shaded_perimeter_approx_l511_51108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l511_51178

-- Define a circle passing through three points
def CircleThroughThreePoints (p1 p2 p3 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (c : ℝ × ℝ) (r : ℝ), 
    (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 ∧
    (p1.1 - c.1)^2 + (p1.2 - c.2)^2 = r^2 ∧
    (p2.1 - c.1)^2 + (p2.2 - c.2)^2 = r^2 ∧
    (p3.1 - c.1)^2 + (p3.2 - c.2)^2 = r^2}

-- Define the equation of a circle
def CircleEquation (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + a*p.1 + b*p.2 + c = 0}

-- Theorem stating the properties of the circle
theorem circle_properties :
  let C := CircleThroughThreePoints (2, 0) (0, 4) (0, 2)
  ∃ (a b c : ℝ),
    -- The equation of the circle
    C = CircleEquation a b c ∧
    -- The center of the circle
    (∃ (center : ℝ × ℝ), center = (3, 3) ∧
      ∀ (p : ℝ × ℝ), p ∈ C → (p.1 - center.1)^2 + (p.2 - center.2)^2 = 10) ∧
    -- The radius of the circle
    (∃ (radius : ℝ), radius^2 = 10 ∧
      ∀ (p : ℝ × ℝ), p ∈ C → (p.1 - 3)^2 + (p.2 - 3)^2 = radius^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l511_51178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sum_l511_51110

theorem inequality_solution_sum : ∃ (S : Finset ℤ),
  (∀ x ∈ S, (Real.sqrt (10 * x - 21) - Real.sqrt (5 * x^2 - 21 * x + 21) ≥ 5 * x^2 - 31 * x + 42) ∧ 
             (10 * x - 21 ≥ 0) ∧ 
             (5 * x^2 - 21 * x + 21 ≥ 0)) ∧
  (∀ x : ℤ, (Real.sqrt (10 * x - 21) - Real.sqrt (5 * x^2 - 21 * x + 21) ≥ 5 * x^2 - 31 * x + 42) ∧ 
             (10 * x - 21 ≥ 0) ∧ 
             (5 * x^2 - 21 * x + 21 ≥ 0) → x ∈ S) ∧
  (S.sum id = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sum_l511_51110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_length_l511_51164

/-- A settlement is a 3x3 grid of square blocks -/
structure Settlement where
  blockSize : ℝ
  gridSize : ℕ
  gridSize_eq : gridSize = 3

/-- A path in the settlement -/
structure SettlementPath (s : Settlement) where
  length : ℝ
  coversAllStreets : Prop
  startsAndEndsAtCorner : Prop

/-- The theorem stating the shortest path length -/
theorem shortest_path_length (s : Settlement) :
  ∃ (p : SettlementPath s), p.coversAllStreets ∧ p.startsAndEndsAtCorner ∧
    p.length = 28 * s.blockSize ∧
    ∀ (q : SettlementPath s), q.coversAllStreets → q.startsAndEndsAtCorner → q.length ≥ p.length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_length_l511_51164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l511_51170

def cherry_gum : ℕ := 25
def grape_gum : ℕ := 40

def pack_size (x : ℕ) : Prop := 
  (cherry_gum - 2 * x) / grape_gum = cherry_gum / (grape_gum + 4 * x) ∧ 
  x > 0

theorem gum_pack_size : ∃ x : ℕ, pack_size x ∧ x = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l511_51170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_armans_hourly_rate_increase_l511_51111

/-- Calculates the increase in hourly rate for Arman's second week of work -/
theorem armans_hourly_rate_increase 
  (hours_week1 : ℕ) 
  (hours_week2 : ℕ) 
  (hourly_rate_week1 : ℚ) 
  (total_earnings : ℚ) 
  (h1 : hours_week1 = 35)
  (h2 : hours_week2 = 40)
  (h3 : hourly_rate_week1 = 10)
  (h4 : total_earnings = 770) : 
  (total_earnings - (hours_week1 : ℚ) * hourly_rate_week1) / (hours_week2 : ℚ) - hourly_rate_week1 = 1/2 := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_armans_hourly_rate_increase_l511_51111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l511_51177

-- Define the points and their properties
def H : ℝ × ℝ := (-3, 0)
def P (y : ℝ) : ℝ × ℝ := (0, y)
def Q (x : ℝ) : ℝ × ℝ := (x, 0)
def M (x y t : ℝ) : ℝ × ℝ := ((1 - t) * (P y).1 + t * (Q x).1, (1 - t) * (P y).2 + t * (Q x).2)

-- Define the conditions
def condition1 (y : ℝ) (m : ℝ × ℝ) : Prop :=
  (H.1 - (P y).1) * (m.1 - (P y).1) + (H.2 - (P y).2) * (m.2 - (P y).2) = 0

def condition2 (y x : ℝ) (m : ℝ × ℝ) : Prop :=
  (m.1 - (P y).1, m.2 - (P y).2) = (-3/2) * ((Q x).1 - m.1, (Q x).2 - m.2)

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1 ∧ p.1 > 0}

-- Define circle N
def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2 * p.1}

-- Define line l
def l (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 = m * p.2 + 1}

-- Define the theorem
theorem line_equation :
  ∀ y x t m : ℝ,
  ∀ a b c d : ℝ × ℝ,
  0 < t → t < 1 →
  condition1 y (M x y t) →
  condition2 y x (M x y t) →
  a ∈ N ∩ C ∩ l m →
  b ∈ N ∩ C ∩ l m →
  c ∈ N ∩ C ∩ l m →
  d ∈ N ∩ C ∩ l m →
  a.2 > b.2 ∧ b.2 > c.2 ∧ c.2 > d.2 →
  (a.1 - b.1)^2 + (a.2 - b.2)^2 - (b.1 - c.1)^2 - (b.2 - c.2)^2 =
  (b.1 - c.1)^2 + (b.2 - c.2)^2 - (c.1 - d.1)^2 - (c.2 - d.2)^2 →
  m = Real.sqrt 2 / 2 ∨ m = -Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l511_51177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_two_l511_51137

/-- A triangular pyramid with pairwise perpendicular lateral edges -/
structure TriangularPyramid where
  /-- Length of the first lateral edge -/
  a : ℝ
  /-- Length of the second lateral edge -/
  b : ℝ
  /-- Length of the third lateral edge -/
  c : ℝ
  /-- The lateral edges are pairwise perpendicular -/
  perpendicular : a * b = 3 ∧ b * c = 4 ∧ a * c = 12

/-- The volume of a triangular pyramid -/
noncomputable def volume (pyramid : TriangularPyramid) : ℝ :=
  (1 / 6) * pyramid.a * pyramid.b * pyramid.c

/-- Theorem: The volume of the specific triangular pyramid is 2 -/
theorem volume_is_two (pyramid : TriangularPyramid) : volume pyramid = 2 := by
  sorry

#check volume_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_two_l511_51137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_little_theorem_l511_51160

theorem fermat_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) (ha : a.natAbs ≠ 0) :
  a ^ (p - 1) ≡ 1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_little_theorem_l511_51160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_17_two_ten_sided_dice_l511_51192

def ten_sided_die : Finset ℕ := Finset.range 10

theorem probability_sum_17_two_ten_sided_dice :
  let outcomes := ten_sided_die.product ten_sided_die
  let favorable := outcomes.filter (fun p => p.1 + p.2 = 17)
  (favorable.card : ℚ) / (outcomes.card : ℚ) = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_17_two_ten_sided_dice_l511_51192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_property_l511_51120

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a/x + b*x - 3

-- State the theorem
theorem f_symmetric_property (a b : ℝ) :
  f a b (-2023) = 2023 → f a b 2023 = -2029 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_property_l511_51120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_triangles_isosceles_l511_51193

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Triangle type
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := distance t.a t.b
  let d2 := distance t.b t.c
  let d3 := distance t.c t.a
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the four triangles
def triangle1 : Triangle :=
  { a := { x := 1, y := 4 },
    b := { x := 3, y := 4 },
    c := { x := 2, y := 2 } }

def triangle2 : Triangle :=
  { a := { x := 4, y := 3 },
    b := { x := 4, y := 5 },
    c := { x := 6, y := 3 } }

def triangle3 : Triangle :=
  { a := { x := 0, y := 1 },
    b := { x := 2, y := 2 },
    c := { x := 4, y := 1 } }

def triangle4 : Triangle :=
  { a := { x := 5, y := 1 },
    b := { x := 6, y := 3 },
    c := { x := 7, y := 0 } }

-- Theorem statement
theorem all_triangles_isosceles :
  isIsosceles triangle1 ∧
  isIsosceles triangle2 ∧
  isIsosceles triangle3 ∧
  isIsosceles triangle4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_triangles_isosceles_l511_51193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l511_51153

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle --/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

/-- Theorem about the area and side length of a specific triangle --/
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos (t.A / 2) = 2 * Real.sqrt 5 / 5)
  (h2 : t.b * t.c * Real.cos t.A = 3)
  (h3 : t.b + t.c = 4 * Real.sqrt 2) : 
  area t = 2 ∧ t.a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l511_51153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_equation_temperature_is_8_l511_51129

/-- The temperature in degrees -/
noncomputable def T : ℝ := sorry

/-- The chance of skidding increases by 5% for every 3 degrees below 32 -/
noncomputable def chance_of_skidding : ℝ := (32 - T) / 3 * 0.05

/-- The chance of having a serious accident when skidding is 60% -/
def chance_of_accident_when_skidding : ℝ := 0.60

/-- The overall chance of having a serious accident is 24% -/
def overall_chance_of_accident : ℝ := 0.24

/-- Theorem stating that the temperature T satisfies the equation -/
theorem temperature_equation : 
  chance_of_skidding * chance_of_accident_when_skidding = overall_chance_of_accident := by
  sorry

/-- The temperature is 8 degrees -/
theorem temperature_is_8 : T = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_equation_temperature_is_8_l511_51129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_proof_l511_51149

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  p : ℝ

/-- Represents a hyperbola -/
def Hyperbola : Set (ℝ × ℝ) := {(x, y) | x^2 - y^2 = 2}

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (a b c : Point) : Prop := sorry

/-- The directrix of a parabola -/
def directrix (p : Parabola) : Set (ℝ × ℝ) := sorry

/-- Intersection points of a set and a hyperbola -/
def intersectionPoints (s : Set (ℝ × ℝ)) (h : Set (ℝ × ℝ)) : Set Point := sorry

/-- The equation of a parabola -/
def parabolaEquation (p : Parabola) : ℝ → ℝ → Prop := sorry

/-- Main theorem -/
theorem parabola_equation_proof (C : Parabola) (h : C.p > 0) :
  C.focus = Point.mk 0 (C.p / 2) →
  let M := intersectionPoints (directrix C) Hyperbola
  (∃ m n : Point, m ∈ M ∧ n ∈ M ∧ m ≠ n ∧ isEquilateral C.focus m n) →
  parabolaEquation C = (fun x y => x^2 = 4 * Real.sqrt 6 * y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_proof_l511_51149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sum_l511_51147

/-- Given a quadratic function g(x) = ax^2 + bx + c that passes through
    the points (1, 5) and (-1, 9), prove that a + 3b + c = 1 -/
theorem quadratic_sum (a b c : ℝ) (g : ℝ → ℝ) : 
  (∀ x, g x = a * x^2 + b * x + c) →
  g 1 = 5 →
  g (-1) = 9 →
  a + 3 * b + c = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sum_l511_51147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_painting_possibilities_l511_51134

theorem rectangle_painting_possibilities :
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ ↦ 
      p.2 > p.1 ∧ 
      (p.1 - 4) * (p.2 - 4) = p.1 * p.2 / 2)
    (Finset.product (Finset.range 100) (Finset.range 100))).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_painting_possibilities_l511_51134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l511_51179

theorem tan_phi_value (φ : Real) 
  (h1 : Real.sin (Real.pi/2 + φ) = Real.sqrt 3/2) 
  (h2 : 0 < φ) (h3 : φ < Real.pi) : 
  Real.tan φ = Real.sqrt 3/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l511_51179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_questionnaire_B_count_l511_51156

theorem questionnaire_B_count :
  let a : ℕ → ℕ := λ n => 30 * n - 21
  let lower_bound : ℕ := 451
  let upper_bound : ℕ := 750
  let count := (Finset.range 32).filter (λ n => lower_bound ≤ a (n + 1) ∧ a (n + 1) ≤ upper_bound)
  count.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_questionnaire_B_count_l511_51156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_g_is_odd_l511_51118

noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 3) + Real.exp (-x * Real.log 3)
noncomputable def g (x : ℝ) : ℝ := Real.exp (x * Real.log 3) - Real.exp (-x * Real.log 3)

theorem f_is_even_and_g_is_odd :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, g (-x) = -g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_g_is_odd_l511_51118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_distance_proof_l511_51135

/-- Proves that given the conditions of the flight problem, the distance flown each way is 1500 miles -/
theorem flight_distance_proof (speed_out speed_return total_time : ℝ) 
  (h1 : speed_out = 300)
  (h2 : speed_return = 500)
  (h3 : total_time = 8) :
  (total_time * speed_out * speed_return) / (speed_out + speed_return) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_distance_proof_l511_51135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_infinite_l511_51138

/-- A power is a positive integer of the form a^k, where a and k are positive integers with k ≥ 2. -/
def IsPower (n : ℕ) : Prop :=
  ∃ (a k : ℕ), k ≥ 2 ∧ n = a ^ k

/-- S is the set of positive integers which cannot be expressed as the sum of two powers. -/
def S : Set ℕ :=
  {n | n > 0 ∧ ¬∃ (x y : ℕ), IsPower x ∧ IsPower y ∧ n = x + y}

/-- The set S is infinite. -/
theorem S_is_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_infinite_l511_51138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_and_expression_value_l511_51194

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem cosine_function_and_expression_value 
  (ω : ℝ) (φ : ℝ) (α : ℝ)
  (h1 : ω > 0)
  (h2 : 0 ≤ φ ∧ φ ≤ π)
  (h3 : ∀ x, f ω φ x = f ω φ (-x))  -- even function
  (h4 : ∀ x, f ω φ (x + π/ω) = f ω φ x)  -- symmetry axes distance
  (h5 : Real.sin α + f ω φ α = 2/3) :
  (∀ x, f ω φ x = Real.cos x) ∧ 
  ((Real.sqrt 2 * Real.sin (2*α - π/4) + 1) / (1 + Real.tan α) = -5/9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_and_expression_value_l511_51194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_evaporation_problem_l511_51119

/-- The original amount of water in a glass, given evaporation conditions --/
noncomputable def original_water_amount (daily_evaporation : ℝ) (days : ℕ) (evaporation_percentage : ℝ) : ℝ :=
  (daily_evaporation * (days : ℝ)) / evaporation_percentage

/-- Theorem stating that under the given conditions, the original amount of water was 10 ounces --/
theorem water_evaporation_problem :
  let daily_evaporation : ℝ := 0.06
  let days : ℕ := 20
  let evaporation_percentage : ℝ := 0.12
  original_water_amount daily_evaporation days evaporation_percentage = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_evaporation_problem_l511_51119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l511_51123

theorem unique_solution_condition (a : ℝ) : 
  (∃! p : ℝ × ℝ, (p.1 - 7 * Real.cos a)^2 + (p.2 - 7 * Real.sin a)^2 = 1 ∧ 
                  |p.1| + |p.2| = 8) ↔ 
  (∃ k : ℤ, a = Real.arcsin ((4 * Real.sqrt 2 + 1) / 7) + (2 * k - 1) * π / 4 ∨
            a = -Real.arcsin ((4 * Real.sqrt 2 + 1) / 7) + (2 * k - 1) * π / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l511_51123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_tan_l511_51189

theorem right_triangle_tan (PQ QR : ℝ) (h1 : PQ = 30) (h2 : QR = 54) :
  let PR := Real.sqrt (QR^2 - PQ^2)
  Real.tan (Real.arcsin (PR / QR)) = 2 * Real.sqrt 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_tan_l511_51189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l511_51158

noncomputable section

-- Define the problem setup
def angle_XOY : Real := Real.pi / 2
def angle_XOP : Real := Real.pi / 6
def OP : Real := 1

-- Define the function to be maximized
def f (M N : Real × Real) : Real :=
  Real.sqrt (M.1^2 + M.2^2) + Real.sqrt (N.1^2 + N.2^2) - Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)

-- State the theorem
theorem max_value_of_f :
  ∃ (M N : Real × Real),
    -- Conditions
    (0 ≤ M.1 ∧ M.2 = 0) ∧  -- M is on OX
    (N.1 = 0 ∧ 0 ≤ N.2) ∧  -- N is on OY
    (∃ (t : Real), M = (t * Real.cos angle_XOP, t * Real.sin angle_XOP) ∨
                N = ((1 - t) * Real.cos angle_XOP, (1 - t) * Real.sin angle_XOP)) ∧  -- Line passes through P
    -- Maximum value
    (∀ (M' N' : Real × Real), f M' N' ≤ 1 + Real.sqrt 3 - Real.sqrt (Real.sqrt 12)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l511_51158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l511_51198

-- Define the function f
def f (x : ℝ) : ℝ := 6 - 8*x + x^2

-- Define the proposed inverse function g
noncomputable def g (x : ℝ) : ℝ := 4 + Real.sqrt (10 + x)

-- Theorem statement
theorem inverse_function_proof :
  ∀ x : ℝ, f (g x) = x ∧ g (f x) = x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l511_51198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antiderivative_correct_l511_51116

-- Define the function representing the antiderivative
noncomputable def F (x : ℝ) : ℝ := 
  x * Real.arctan (Real.sqrt (4 * x - 1)) - (1 / 4) * Real.sqrt (4 * x - 1)

-- State the theorem
theorem antiderivative_correct (x : ℝ) (h : 4 * x - 1 > 0) :
  deriv F x = Real.arctan (Real.sqrt (4 * x - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_antiderivative_correct_l511_51116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l511_51183

open Real

/-- The tangent line to both f(x) = ln x + 2 and g(x) = ln(x+1) -/
def tangent_line (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

/-- The first curve -/
noncomputable def f (x : ℝ) : ℝ := log x + 2

/-- The second curve -/
noncomputable def g (x : ℝ) : ℝ := log (x + 1)

/-- Theorem stating that if a line is tangent to both curves, then k - b = 1 + ln 2 -/
theorem tangent_line_property (k b : ℝ) :
  (∃ x₁ > 0, tangent_line k b x₁ = f x₁ ∧ k = (deriv f) x₁) ∧
  (∃ x₂ > -1, tangent_line k b x₂ = g x₂ ∧ k = (deriv g) x₂) →
  k - b = 1 + log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l511_51183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_half_of_eight_eq_neg_three_l511_51128

-- Define the logarithm function for base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- State the theorem
theorem log_half_of_eight_eq_neg_three : log_half 8 = -3 := by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_half_of_eight_eq_neg_three_l511_51128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_exam_results_l511_51152

-- Define the normal distribution
structure NormalDist (μ σ : ℝ) where

-- Define the probability function
noncomputable def prob (X : NormalDist μ σ) (a b : ℝ) : ℝ := sorry

-- Define the inverse cumulative distribution function (quantile function)
noncomputable def quantile (X : NormalDist μ σ) (p : ℝ) : ℝ := sorry

theorem math_exam_results (X : NormalDist 90 10) (n : ℕ) 
  (h1 : n = 5000) :
  -- Part 1: Number of candidates scoring between 100 and 120
  ∃ (k : ℕ), abs (k - (n * (prob X 100 120))) < 1 ∧ k = 3413 ∧
  -- Part 2: Admission score cut-off
  ∃ (s : ℝ), abs (s - (quantile X 0.9772)) < 1 ∧ s = 290 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_exam_results_l511_51152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_8_l511_51166

-- Define the functions t and f as noncomputable
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
noncomputable def f (x : ℝ) : ℝ := 8 - t x

-- State the theorem
theorem t_of_f_8 : t (f 8) = Real.sqrt (42 - 5 * Real.sqrt 42) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_8_l511_51166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Al_percentage_in_AlBr3_l511_51181

/-- Represents the mass percentage of an element in a compound -/
noncomputable def mass_percentage (element_mass : ℝ) (compound_mass : ℝ) : ℝ :=
  (element_mass / compound_mass) * 100

/-- The atomic mass of aluminum in g/mol -/
def Al_mass : ℝ := 26.98

/-- The atomic mass of bromine in g/mol -/
def Br_mass : ℝ := 79.90

/-- The molar mass of aluminum bromide (AlBr₃) in g/mol -/
def AlBr3_mass : ℝ := Al_mass + 3 * Br_mass

/-- Theorem stating that the mass percentage of aluminum in aluminum bromide is approximately 10.11% -/
theorem Al_percentage_in_AlBr3 : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |mass_percentage Al_mass AlBr3_mass - 10.11| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Al_percentage_in_AlBr3_l511_51181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_theorem_l511_51182

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| * Real.exp (-1 / x) - a

theorem three_roots_theorem (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  a > 0 ∧ (∀ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 →
    x₂ - x₁ < a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_theorem_l511_51182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_number_greater_probability_l511_51174

-- Define the random variables X and Y as functions
noncomputable def X : Ω → ℝ := sorry
noncomputable def Y : Ω → ℝ := sorry

-- Define the probability space
def Ω : Type := ℝ × ℝ

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- State the theorem
theorem bobs_number_greater_probability :
  (∀ ω : Ω, X ω ∈ Set.Ioo (1/2 : ℝ) (7/8 : ℝ)) →
  (∀ ω : Ω, Y ω ∈ Set.Ioo (3/4 : ℝ) 1) →
  P {ω : Ω | Y ω > X ω} = 11/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_number_greater_probability_l511_51174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l511_51143

/-- A parabola with focus on the x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- The line that intersects the parabola -/
def intersecting_line (x y : ℝ) : Prop := y = 2 * x - 4

/-- The chord length of the intersection -/
noncomputable def chord_length (parabola : Parabola) : ℝ := 3 * Real.sqrt 5

/-- Theorem stating the possible standard equations of the parabola -/
theorem parabola_equation (parabola : Parabola) :
  (∀ x y, parabola.equation x y ↔ y^2 = 4*x) ∨
  (∀ x y, parabola.equation x y ↔ y^2 = -36*x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l511_51143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l511_51113

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  -- Add triangle conditions
  angle_sum : A + B + C = Real.pi
  sine_law : a / Real.sin A = b / Real.sin B
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos (2 * t.C) - Real.cos (2 * t.A) = 
        2 * Real.sin (Real.pi/3 + t.C) * Real.sin (Real.pi/3 - t.C))
  (h2 : t.a = Real.sqrt 3)
  (h3 : t.b ≥ t.a) : 
  ((t.A = Real.pi/3 ∨ t.A = 2*Real.pi/3) ∧ 
   (t.A = Real.pi/3 → 2*t.b - t.c ≥ Real.sqrt 3 ∧ 2*t.b - t.c < 2*Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l511_51113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_old_machine_rate_proof_l511_51197

/-- The rate of the new machine in bolts per hour -/
noncomputable def new_machine_rate : ℝ := 150

/-- The time both machines work together in hours -/
noncomputable def work_time : ℝ := 132 / 60

/-- The total number of bolts produced by both machines -/
noncomputable def total_bolts : ℝ := 550

/-- The rate of the old machine in bolts per hour -/
noncomputable def old_machine_rate : ℝ := 100

/-- Theorem stating that the combined rate of both machines multiplied by the work time equals the total bolts produced -/
theorem old_machine_rate_proof :
  (old_machine_rate + new_machine_rate) * work_time = total_bolts :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_old_machine_rate_proof_l511_51197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_product_range_l511_51159

/-- In a triangle ABC, given that 3a² = c² - b², prove that 0 < tan(A) * tan(B) < 1/2 -/
theorem triangle_tangent_product_range (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_angles : A + B + C = Real.pi) (h_sides : 3 * a^2 = c^2 - b^2) : 
  0 < Real.tan A * Real.tan B ∧ Real.tan A * Real.tan B < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_product_range_l511_51159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cone_altitude_of_given_frustum_l511_51132

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the small cone removed from a frustum -/
noncomputable def small_cone_altitude (f : Frustum) : ℝ :=
  (f.altitude * (Real.sqrt (f.upper_base_area / Real.pi)) / 
   (Real.sqrt (f.lower_base_area / Real.pi) - Real.sqrt (f.upper_base_area / Real.pi)))

/-- Theorem stating the altitude of the small cone removed from the given frustum -/
theorem small_cone_altitude_of_given_frustum :
  let f : Frustum := { altitude := 30, lower_base_area := 400 * Real.pi, upper_base_area := 36 * Real.pi }
  small_cone_altitude f = 90 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cone_altitude_of_given_frustum_l511_51132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_one_l511_51180

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the sequences a_n and b_n
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

-- State the condition relating a_n and b_n to (3 + i)^(2n)
axiom sequence_condition : ∀ n : ℕ, (3 + i) ^ (2 * n) = (a n : ℂ) + (b n : ℂ) * i

-- State the theorem to be proved
theorem sum_equals_one :
  2 * (∑' n, (a n * b n) / (10 ^ n)) = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_one_l511_51180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l511_51163

/-- A hyperbola with foci on the x-axis and asymptotes y = ± 1/2 x has eccentricity √5/2 -/
theorem hyperbola_eccentricity (h : Real → Real → Prop) 
  (foci_on_x : ∃ c, ∀ x y, h x y → (x = c ∨ x = -c) ∧ y = 0)
  (asymptotes : ∀ x y, h x y → (y = (1/2) * x ∨ y = -(1/2) * x)) : 
  ∃ e, e = Real.sqrt 5 / 2 ∧ 
    ∀ x y, h x y → (x^2 / (e^2 - 1) - y^2 / (e^2 - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l511_51163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chloe_rate_l511_51157

/-- The cycling rates of George, Lucy, Max, and Chloe -/
noncomputable def cycling_rates (george_rate : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let lucy_rate := (3/4) * george_rate
  let max_rate := (4/3) * lucy_rate
  let chloe_rate := (5/6) * max_rate
  (george_rate, lucy_rate, max_rate, chloe_rate)

/-- Theorem stating Chloe's cycling rate -/
theorem chloe_rate (george_rate : ℝ) (h : george_rate = 6) :
  (cycling_rates george_rate).2.2.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chloe_rate_l511_51157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_three_l511_51172

noncomputable def g (x : ℝ) : ℝ :=
  if -5 ≤ x ∧ x ≤ -1 then -(x+3)^2 + 4
  else if -1 < x ∧ x ≤ 3 then x - 1
  else if 3 < x ∧ x ≤ 5 then (x-4)^2 + 1
  else 0  -- undefined outside [-5, 5]

-- Theorem statement
theorem g_composition_equals_three :
  ∃ (S : Finset ℝ), S.card = 2 ∧ 
    (∀ y ∈ S, -5 ≤ y ∧ y ≤ 5 ∧ g (g y) = 3) ∧
    (∀ z, -5 ≤ z ∧ z ≤ 5 → g (g z) = 3 → z ∈ S) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_three_l511_51172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_47_to_17_l511_51151

theorem digits_of_47_to_17 (h : ∃ a : ℝ, 0 ≤ a ∧ a < 1 ∧ Real.log (47^100) = 167 + a) :
  ∃ b : ℝ, 0 ≤ b ∧ b < 1 ∧ Real.log (47^17) = 28 + b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_47_to_17_l511_51151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_selection_l511_51185

def number_of_ways_to_select (m n k : ℕ) : ℕ :=
  Nat.choose (m + n) k - Nat.choose m k - Nat.choose n k

theorem summer_camp_selection (m n k : ℕ) (hm : m = 7) (hn : n = 5) (hk : k = 4) :
  number_of_ways_to_select m n k =
  Nat.choose (m + n) k - Nat.choose m k - Nat.choose n k :=
by
  rfl

#eval number_of_ways_to_select 7 5 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_selection_l511_51185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_sandwich_loaves_needed_l511_51114

/-- Calculates the number of loaves needed for club sandwiches over 6 months -/
theorem club_sandwich_loaves_needed : ℕ := by
  let weeks : ℕ := 26
  let slices_per_loaf : ℕ := 16
  let saturday_consumption : ℕ := 6
  let sunday_consumption : ℕ := 0
  let weekend_consumption : ℕ := saturday_consumption + sunday_consumption
  let total_slices : ℕ := weekend_consumption * weeks
  have h : (total_slices + slices_per_loaf - 1) / slices_per_loaf = 10 := by
    sorry
  exact 10


end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_sandwich_loaves_needed_l511_51114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l511_51165

theorem diophantine_equation_solution :
  ∀ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 → 3^x + 4^y = 5^z → x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l511_51165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poster_enlargement_l511_51127

/-- Represents the dimensions of a rectangular poster -/
structure PosterDimensions where
  width : ℚ
  height : ℚ

/-- Calculates the new height of a proportionally enlarged poster -/
def enlargedHeight (original : PosterDimensions) (newWidth : ℚ) : ℚ :=
  (newWidth / original.width) * original.height

theorem poster_enlargement (original : PosterDimensions) (newWidth : ℚ) 
  (h1 : original.width = 3)
  (h2 : original.height = 2)
  (h3 : newWidth = 12) :
  enlargedHeight original newWidth = 8 := by
  sorry

#eval enlargedHeight ⟨3, 2⟩ 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poster_enlargement_l511_51127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_satisfying_inequality_l511_51107

/-- Given a function f and a real number m, this theorem states the conditions
    for m to satisfy f(√(-m² + 2m + 3)) > f(√(-m² + 4)) -/
theorem range_of_m_satisfying_inequality (f : ℝ → ℝ) (m : ℝ) : 
  (f = λ x ↦ 3 * Real.sin (-1/5 * x + 3 * Real.pi / 10)) →
  (f (Real.sqrt (-m^2 + 2*m + 3)) > f (Real.sqrt (-m^2 + 4))) ↔ 
  (m ≥ -1 ∧ m < 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_satisfying_inequality_l511_51107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_difference_l511_51109

-- Define the theorem
theorem tangent_difference (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = Real.sqrt 17 / 17) :
  Real.tan (α - π / 4) = -5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_difference_l511_51109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_projection_is_orthocenter_l511_51117

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the triangle and trihedral angle
def Triangle (A B C : Point3D) : Prop := True
def TrihedralAngle (O : Point3D) (x y z : Point3D → Prop) : Prop := True

-- Define the orthogonal projection
noncomputable def OrthogonalProjection (P : Point3D) (π : Plane) : Point3D := sorry

-- Define the orthocenter of a triangle
noncomputable def Orthocenter (A B C : Point3D) : Point3D := sorry

-- Define a membership relation for Point3D and Plane
def PointInPlane (P : Point3D) (π : Plane) : Prop := 
  π.a * P.x + π.b * P.y + π.c * P.z + π.d = 0

-- The main theorem
theorem orthogonal_projection_is_orthocenter 
  (α : Plane) (O A B C : Point3D) 
  (x y z : Point3D → Prop)
  (h1 : Triangle A B C)
  (h2 : TrihedralAngle O x y z)
  (h3 : PointInPlane A α ∧ x A)
  (h4 : PointInPlane B α ∧ y B)
  (h5 : PointInPlane C α ∧ z C) :
  OrthogonalProjection O α = Orthocenter A B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_projection_is_orthocenter_l511_51117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_cos_range_l511_51191

theorem sin_greater_cos_range (x : ℝ) :
  x ∈ Set.Ioo 0 (2 * Real.pi) →
  (Real.sin x > Real.cos x) ↔ x ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_cos_range_l511_51191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_for_xiaowen_l511_51196

/-- Represents the cost of items at a store -/
structure StoreCost where
  penPrice : ℕ
  notebookPrice : ℕ
  penDiscount : ℕ → ℕ
  notebookDiscount : ℕ → ℕ

/-- Calculate the total cost at a store -/
def totalCost (store : StoreCost) (pens : ℕ) (notebooks : ℕ) : ℕ :=
  store.penDiscount (pens * store.penPrice) + store.notebookDiscount (notebooks * store.notebookPrice)

/-- Store A's pricing strategy -/
def storeA : StoreCost :=
  { penPrice := 10
  , notebookPrice := 2
  , penDiscount := id
  , notebookDiscount := fun n => max 0 (n - 2 * 10) }

/-- Store B's pricing strategy -/
def storeB : StoreCost :=
  { penPrice := 10
  , notebookPrice := 2
  , penDiscount := fun n => n * 9 / 10
  , notebookDiscount := fun n => n * 9 / 10 }

/-- The minimum cost for Xiaowen's purchase -/
theorem min_cost_for_xiaowen :
  min
    (totalCost storeA 4 24)
    (min
      (totalCost storeB 4 24)
      (totalCost storeA 4 0 + totalCost storeB 0 20)) = 76 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_for_xiaowen_l511_51196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_nth_root_of_unity_l511_51188

def S : Set ℂ :=
  {z | ∃ (x y : ℝ), z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 2 / 2 ∧ y ≥ 1/2}

theorem smallest_m_for_nth_root_of_unity (m : ℕ) : m = 24 ↔
  (∀ n : ℕ, n ≥ m → ∃ z ∈ S, z^n = 1) ∧
  (∀ k : ℕ, k < m → ∃ n : ℕ, n ≥ k ∧ ∀ z ∈ S, z^n ≠ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_nth_root_of_unity_l511_51188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l511_51105

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.cos (2*x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem symmetry_center_of_f :
  ∃ (k : ℤ), (∀ x, f (x + (-π/12)) = f (-x + (-π/12))) ∧ 
  ∀ y, (∀ x, f (x + y) = f (-x + y)) → y = -π/12 + k * π/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l511_51105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_to_appear_l511_51199

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 2
  | n + 2 => (modifiedFibonacci (n + 1) + modifiedFibonacci n) % 10

def appearsInSequence (d : ℕ) : Prop :=
  ∃ n : ℕ, modifiedFibonacci n % 10 = d

theorem last_digit_to_appear :
  (∀ d : ℕ, d < 10 → d ≠ 1 → appearsInSequence d) ∧
  (¬ appearsInSequence 1 ∨
   (appearsInSequence 1 ∧ ∀ d : ℕ, d < 10 → d ≠ 1 → 
     ∃ n m : ℕ, n < m ∧ modifiedFibonacci n % 10 = d ∧ modifiedFibonacci m % 10 = 1)) :=
by sorry

#check last_digit_to_appear

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_to_appear_l511_51199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_of_squares_solutions_l511_51190

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def is_solution (a b n : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ n > 0 ∧ n < 14 ∧ a ≤ b ∧ a^2 + b^2 = factorial n

theorem factorial_sum_of_squares_solutions :
  ∀ a b n : ℕ, is_solution a b n ↔ 
    ((a = 1 ∧ b = 1 ∧ n = 1) ∨ 
     (a = 1 ∧ b = 1 ∧ n = 2) ∨ 
     (a = 12 ∧ b = 24 ∧ n = 6)) :=
by sorry

#check factorial_sum_of_squares_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_of_squares_solutions_l511_51190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_popsicle_melting_rate_l511_51124

theorem popsicle_melting_rate :
  ∀ (t : ℝ), t > 0 →
  let sequence := λ (i : ℕ) => t / (2 ^ i)
  (sequence 0) / (sequence 5) = 32 :=
by
  intro t ht
  simp [sequence]
  field_simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_popsicle_melting_rate_l511_51124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_problem_l511_51136

-- Define the ellipse and hyperbola
def ellipse (c d : ℝ) : Set (ℝ × ℝ) := {(x, y) | x^2 / c^2 + y^2 / d^2 = 1}
def hyperbola (c d : ℝ) : Set (ℝ × ℝ) := {(x, y) | x^2 / c^2 - y^2 / d^2 = 1}

-- Define the foci of the ellipse and hyperbola
def ellipse_foci (c d : ℝ) : Set (ℝ × ℝ) := {(0, 5), (0, -5)}
def hyperbola_foci (c d : ℝ) : Set (ℝ × ℝ) := {(7, 0), (-7, 0)}

-- Theorem statement
theorem conic_sections_problem (c d : ℝ) 
  (h1 : ellipse_foci c d = {(0, 5), (0, -5)})
  (h2 : hyperbola_foci c d = {(7, 0), (-7, 0)}) :
  |c * d| = 2 * Real.sqrt 111 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_problem_l511_51136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l511_51162

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

/-- Area of a triangle given two sides and the included angle -/
noncomputable def area (t : Triangle) : ℝ := 
  (1 / 2) * t.b * t.c * sin t.A

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
    (h : t.a * sin t.C = Real.sqrt 3 * t.c * cos t.A) :
    t.A = π / 3 ∧ 
    (t.a = Real.sqrt 13 ∧ t.c = 3 → area t = 3 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l511_51162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_abc_ratio_l511_51141

noncomputable def m (x y : ℝ) : ℝ × ℝ := (2 * Real.cos x, y - 2 * Real.sqrt 3 * Real.sin x * Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + 1

theorem range_of_abc_ratio 
  (x y : ℝ) 
  (h1 : ∃ (k : ℝ), m x y = k • (n x)) 
  (A B C : ℝ) 
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h3 : A + B + C = Real.pi)
  (h4 : f (C / 2) = 3) :
  ∃ (r : Set ℝ), r = Set.Ioo 1 2 ∧ 
  ∀ (a b c : ℝ), (a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) → 
  (a + b) / c ∈ r :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_abc_ratio_l511_51141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wren_population_below_10_percent_in_2009_l511_51186

/-- The year when the wren population falls below 10% of its initial value -/
def year_below_10_percent (n : ℕ) : Prop :=
  (0.6 : ℝ)^n < 0.1 ∧ ∀ k < n, (0.6 : ℝ)^k ≥ 0.1

/-- Theorem stating that the population falls below 10% in 2009 (5 years after 2004) -/
theorem wren_population_below_10_percent_in_2009 :
  year_below_10_percent 5 := by
  sorry

#check wren_population_below_10_percent_in_2009

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wren_population_below_10_percent_in_2009_l511_51186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l511_51161

theorem min_value_trig_expression (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.tan x + 1 / Real.tan x)^2 + (1 / Real.cos x + 1 / Real.sin x)^2 ≥ 8 ∧
  ((Real.tan x + 1 / Real.tan x)^2 + (1 / Real.cos x + 1 / Real.sin x)^2 = 8 ↔ x = Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l511_51161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_condition_l511_51175

/-- Given two vectors a and b in R^2, if a = (x, 2), b = (x-1, 1), 
    and (a + b) is perpendicular to (a - b), then x = -1 -/
theorem vector_perpendicular_condition (x : ℝ) : 
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (x - 1, 1)
  (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_condition_l511_51175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_600_plus_tan_240_l511_51168

theorem sin_600_plus_tan_240 : 
  Real.sin (600 * Real.pi / 180) + Real.tan (240 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_600_plus_tan_240_l511_51168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_and_bounded_l511_51155

theorem sqrt_two_irrational_and_bounded : 
  Irrational (Real.sqrt 2) ∧ 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_and_bounded_l511_51155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OMN_l511_51102

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the focus F
noncomputable def F : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define endpoints A and B of the major axis
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the dot product condition
axiom dot_product_condition : (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = -1

-- Define a line l intersecting E at points M and N
def line_intersects_E (l : ℝ → ℝ) (M N : ℝ × ℝ) : Prop :=
  E M.1 M.2 ∧ E N.1 N.2 ∧ ∃ t, (M.1, M.2) = (t, l t) ∧ ∃ s, (N.1, N.2) = (s, l s)

-- Define the area of triangle OMN
noncomputable def area_OMN (M N : ℝ × ℝ) : ℝ :=
  abs (M.1 * N.2 - M.2 * N.1) / 2

-- Theorem statement
theorem max_area_OMN :
  ∃ (max_area : ℝ), max_area = 1 ∧
  ∀ (l : ℝ → ℝ) (M N : ℝ × ℝ),
    line_intersects_E l M N →
    area_OMN M N ≤ max_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OMN_l511_51102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_of_f_l511_51187

/-- The function f(x) = 2x - 5 - ln x has exactly one zero for x > 0 -/
theorem unique_zero_of_f (f : ℝ → ℝ) : 
  (∀ x, x > 0 → f x = 2 * x - 5 - Real.log x) → 
  ∃! x, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_of_f_l511_51187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_increase_approx_l511_51195

/-- Calculate the percent increase between two values -/
noncomputable def percentIncrease (original : ℝ) (new : ℝ) : ℝ :=
  (new - original) / original * 100

/-- The original cost per minute of a long-distance call in 2000 -/
def originalCost : ℝ := 15

/-- The new cost per minute of a long-distance call in 2020 -/
def newCost : ℝ := 25

/-- Theorem stating that the percent increase in cost is approximately 66.67% -/
theorem cost_increase_approx : 
  abs (percentIncrease originalCost newCost - 66.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_increase_approx_l511_51195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_g_greater_than_one_l511_51184

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x - 1)
noncomputable def g (a x : ℝ) : ℝ := a^x + x^a

-- Theorem 1: f is monotonically decreasing on (0, 1)
theorem f_monotone_decreasing :
  ∀ x, 0 < x → x < 1 → (deriv f) x < 0 := by
  sorry

-- Theorem 2: g(x) > 1 when 0 < a < x < 1
theorem g_greater_than_one :
  ∀ a x, 0 < a → a < x → x < 1 → g a x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_g_greater_than_one_l511_51184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_a_b_less_than_500_l511_51103

theorem max_sum_a_b_less_than_500 :
  ∃ (a b : ℕ), (b > 1) ∧ (a^b < 500) ∧
  (∀ (c d : ℕ), (d > 1) ∧ (c^d < 500) → a + b ≥ c + d) ∧
  (a + b = 24) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_a_b_less_than_500_l511_51103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l511_51171

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 / Real.exp x

theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 4 ∧ ∃ x₀ : ℝ, f x₀ = 4 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l511_51171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_car_insurance_cost_l511_51176

/-- Represents the total cost of car insurance over a decade in dollars -/
noncomputable def total_cost : ℚ := 20000

/-- Represents the number of years in a decade -/
noncomputable def num_years : ℚ := 10

/-- Represents the annual cost of car insurance in dollars -/
noncomputable def annual_cost : ℚ := total_cost / num_years

/-- Theorem stating that the annual cost of car insurance is 2000 dollars -/
theorem annual_car_insurance_cost : annual_cost = 2000 := by
  rw [annual_cost, total_cost, num_years]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_car_insurance_cost_l511_51176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_r_plus_s_l511_51122

-- Define the triangle DEF
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define the area of the triangle
def area (t : Triangle) : ℝ := 50

-- Define the slope of the median to side DE
def median_slope : ℝ := -3

-- Define the theorem
theorem max_r_plus_s :
  ∀ (r s : ℝ),
  let t : Triangle := { D := (10, 15), E := (20, 18), F := (r, s) }
  area t = 50 →
  median_slope = -3 →
  r + s ≤ 565/33 + 609/66 :=
by
  intros r s t_def area_eq slope_eq
  sorry

#check max_r_plus_s

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_r_plus_s_l511_51122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l511_51148

theorem infinitely_many_solutions :
  ∃ f : ℕ → ℕ × ℕ, 
    Function.Injective f ∧ 
    ∀ n : ℕ, 
      let (a, b) := f n
      2 * a^2 - 3 * a + 1 = 3 * b^2 + b ∧ a > 0 ∧ b > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l511_51148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l511_51112

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x + m)

theorem min_translation_for_symmetry (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x, g m x = g m (π/4 - x)) : m ≥ π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l511_51112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_interest_rate_problem_l511_51140

/-- Compound interest calculation -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r/n)^(n*t) - P

/-- Theorem for the annual interest rate problem -/
theorem annual_interest_rate_problem (P t CI : ℝ) :
  P = 3000 →
  t = 2 →
  CI = 630 →
  ∃ r : ℝ, compound_interest P r 1 t = CI ∧ r = 0.1 := by
  sorry

#check annual_interest_rate_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_interest_rate_problem_l511_51140
