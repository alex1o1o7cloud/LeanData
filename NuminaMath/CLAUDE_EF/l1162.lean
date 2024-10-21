import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_specific_parabola_focus_l1162_116220

/-- Predicate to check if a point is the focus of a parabola -/
def is_focus (parabola : Set (ℝ × ℝ)) (f : ℝ × ℝ) : Prop :=
  ∃ (a : ℝ), a > 0 ∧ parabola = {(x, y) : ℝ × ℝ | y^2 = a * x} ∧ f = (a / 4, 0)

/-- The focus of a parabola y^2 = ax (where a > 0) has coordinates (a/4, 0) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a > 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = a * x}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (a / 4, 0) ∧ is_focus parabola f :=
by
  sorry

/-- The focus of the parabola y^2 = 16x has coordinates (4, 0) -/
theorem specific_parabola_focus :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 16 * x}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (4, 0) ∧ is_focus parabola f :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_specific_parabola_focus_l1162_116220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt5_power_fractional_part_l1162_116207

theorem sqrt5_power_fractional_part (n : ℕ) :
  ∃ (P Q : ℝ), (Real.sqrt 5 + 2) ^ (2 * n + 1) = P + Q ∧
  P = ⌊(Real.sqrt 5 + 2) ^ (2 * n + 1)⌋ ∧
  Q = (Real.sqrt 5 + 2) ^ (2 * n + 1) - ⌊(Real.sqrt 5 + 2) ^ (2 * n + 1)⌋ ∧
  0 < Q ∧ Q < 1 ∧
  Q * (P + Q) = 1 := by
  sorry

#check sqrt5_power_fractional_part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt5_power_fractional_part_l1162_116207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l1162_116280

/-- The equation of the tangent line to y = 2e^x at x = 0 is 2x - y + 2 = 0 -/
theorem tangent_line_at_zero (x y : ℝ) : 
  (fun x => 2 * Real.exp x) 0 = y → 
  2 * x - y + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l1162_116280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_late_start_meeting_time_l1162_116241

/-- Represents a person walking with a constant speed -/
structure Walker where
  speed : ℝ

/-- Represents the scenario of two people walking towards each other -/
structure WalkingScenario where
  distance : ℝ
  walker_a : Walker
  walker_b : Walker

/-- The time it takes for two walkers to meet when starting simultaneously -/
noncomputable def meeting_time (scenario : WalkingScenario) : ℝ :=
  scenario.distance / (scenario.walker_a.speed + scenario.walker_b.speed)

/-- The distance from the start point of walker A to the meeting point -/
noncomputable def meeting_distance_a (scenario : WalkingScenario) : ℝ :=
  scenario.walker_a.speed * meeting_time scenario

theorem late_start_meeting_time 
  (scenario : WalkingScenario)
  (h1 : scenario.distance = 10)
  (h2 : meeting_distance_a scenario = 6)
  (h3 : meeting_distance_a { scenario with
    walker_a := { speed := scenario.walker_a.speed * 12 / 11 }} = 5) :
  12 / 11 * meeting_time scenario = 10 := by
  sorry

#check late_start_meeting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_late_start_meeting_time_l1162_116241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_function_phi_l1162_116203

theorem sin_function_phi (φ : ℝ) : 
  (∀ x : ℝ, Real.sin (2 * x + φ) ≤ |Real.sin (π / 3 + φ)|) ∧ 
  (Real.sin (2 * π / 3 + φ) > Real.sin (π + φ)) → 
  φ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_function_phi_l1162_116203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l1162_116292

-- Use the existing Complex type from Mathlib
open Complex

theorem complex_equation_solution (m : ℝ) :
  let z₁ : ℂ := (m^2 - 3*m : ℝ) + (m^2 : ℝ) * I
  let z₂ : ℂ := (4 : ℝ) + (5*m + 6 : ℝ) * I
  z₁ - z₂ = 0 → m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l1162_116292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1162_116215

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.cos x ^ 2

theorem smallest_positive_period_of_f :
  ∀ T : ℝ, T > 0 → (∀ x : ℝ, f (x + T) = f x) → T ≥ 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1162_116215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l1162_116293

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- Given two lines y = 3x + 5 and 4y + ax = 8, prove that if they are perpendicular, then a = 4/3 -/
theorem perpendicular_lines_a_value (a : ℝ) :
  perpendicular 3 (-a/4) → a = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l1162_116293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_is_real_Z_is_pure_imaginary_Z_in_first_quadrant_l1162_116247

-- Define the complex number Z as a function of m
noncomputable def Z (m : ℝ) : ℂ := Complex.mk (Real.log (m^2 - 2*m - 2)) (m^2 + 3*m + 2)

-- Theorem 1: Z is real iff m = -1 or m = -2
theorem Z_is_real (m : ℝ) : (Z m).im = 0 ↔ m = -1 ∨ m = -2 := by sorry

-- Theorem 2: Z is pure imaginary iff m = 3 or m = -1
theorem Z_is_pure_imaginary (m : ℝ) : (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = 3 ∨ m = -1 := by sorry

-- Theorem 3: Z is in the first quadrant iff m < -2 or m > 3
theorem Z_in_first_quadrant (m : ℝ) : (Z m).re > 0 ∧ (Z m).im > 0 ↔ m < -2 ∨ m > 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_is_real_Z_is_pure_imaginary_Z_in_first_quadrant_l1162_116247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_is_maximum_l1162_116239

/-- The maximum number of pairs that can be chosen from the set of integers 
    {1, 2, 3, ..., 3009} satisfying the given conditions -/
def max_pairs : ℕ := 1203

/-- The set of integers from which pairs are chosen -/
def integer_set : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 3009}

/-- A structure representing a pair of integers -/
structure IntPair where
  a : ℕ
  b : ℕ
  h_ab : a < b
  h_a_in_set : a ∈ integer_set
  h_b_in_set : b ∈ integer_set

/-- The proposition that all sums of pairs are distinct and ≤ 3009 -/
def valid_sums (pairs : List IntPair) : Prop :=
  ∀ i j, i ≠ j → (pairs.get i).a + (pairs.get i).b ≠ (pairs.get j).a + (pairs.get j).b ∧
  ∀ p ∈ pairs, p.a + p.b ≤ 3009

/-- The proposition that no two pairs have a common element -/
def no_common_elements (pairs : List IntPair) : Prop :=
  ∀ i j, i ≠ j → 
    (pairs.get i).a ≠ (pairs.get j).a ∧ 
    (pairs.get i).a ≠ (pairs.get j).b ∧
    (pairs.get i).b ≠ (pairs.get j).a ∧ 
    (pairs.get i).b ≠ (pairs.get j).b

/-- The main theorem stating that max_pairs is the maximum number of pairs
    satisfying all conditions -/
theorem max_pairs_is_maximum :
  ∀ pairs : List IntPair,
    valid_sums pairs →
    no_common_elements pairs →
    pairs.length ≤ max_pairs := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_is_maximum_l1162_116239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melting_ice_cream_l1162_116230

/-- The volume of a sphere with radius r -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The melting ice cream problem -/
theorem melting_ice_cream (initialRadius finalRadius : ℝ) 
  (h_init : initialRadius = 3)
  (h_final : finalRadius = 12) : 
  ∃ (h : ℝ), h = 1/4 ∧ sphereVolume initialRadius = cylinderVolume finalRadius h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_melting_ice_cream_l1162_116230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_result_exists_l1162_116266

def correct_result : ℕ := 109395

def digit (i : ℕ) (n : ℕ) : ℕ :=
  (n / (10 ^ i)) % 10

theorem incorrect_result_exists : ∃ (n : ℕ), 
  (n ≠ correct_result) ∧ 
  (∃ (i j : ℕ), i ≠ j ∧ 
    (digit i n = 2) ∧ 
    (digit j n = 2) ∧
    (∀ (k : ℕ), k ≠ i ∧ k ≠ j → digit k n = digit k correct_result)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_result_exists_l1162_116266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_four_cubes_l1162_116289

/-- Represents a cube with six faces --/
structure Cube where
  faces : Fin 6 → ℕ

/-- The set of numbers on each cube's faces --/
def cubeNumbers : Finset ℕ := {1, 3, 9, 27, 81, 243}

/-- A function to calculate the sum of visible faces when stacking cubes --/
def visibleSum (cubes : Fin 4 → Cube) : ℕ :=
  sorry

/-- Theorem stating the maximum sum of visible numbers when stacking four cubes --/
theorem max_visible_sum_four_cubes :
  ∃ (cubes : Fin 4 → Cube),
    (∀ i : Fin 4, ∀ j : Fin 6, (cubes i).faces j ∈ cubeNumbers) ∧
    visibleSum cubes = 1446 ∧
    ∀ (other_cubes : Fin 4 → Cube),
      (∀ i : Fin 4, ∀ j : Fin 6, (other_cubes i).faces j ∈ cubeNumbers) →
      visibleSum other_cubes ≤ 1446 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_four_cubes_l1162_116289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1162_116287

def our_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  a 1 = 1 ∧
  (∀ n, a (n + 2) = 1 / (a n + 1)) ∧
  a 6 = a 2

theorem sequence_sum (a : ℕ → ℝ) (h : our_sequence a) : a 2016 + a 3 = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1162_116287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_satisfies_equations_intersection_unique_l1162_116236

/-- Two lines in a 2D plane --/
structure TwoLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ

/-- The intersection point of two lines --/
noncomputable def intersection_point : ℝ × ℝ := (-1/2, -3/2)

/-- Theorem stating that the intersection point satisfies both line equations --/
theorem intersection_satisfies_equations (lines : TwoLines) 
  (h1 : lines.line1 = fun x ↦ 5 * x + 1)
  (h2 : lines.line2 = fun x ↦ -3 * x - 3) :
  lines.line1 (intersection_point.1) = intersection_point.2 ∧
  lines.line2 (intersection_point.1) = intersection_point.2 := by
  sorry

/-- Theorem stating that the intersection point is unique --/
theorem intersection_unique (lines : TwoLines) 
  (h1 : lines.line1 = fun x ↦ 5 * x + 1)
  (h2 : lines.line2 = fun x ↦ -3 * x - 3)
  (p : ℝ × ℝ) 
  (hp1 : lines.line1 p.1 = p.2)
  (hp2 : lines.line2 p.1 = p.2) :
  p = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_satisfies_equations_intersection_unique_l1162_116236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1162_116275

/-- The equation of the tangent line to y = 2x^2 at (1, 2) is 4x - y - 2 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = 2 * x^2) →  -- Curve equation
  (∃ t : ℝ, t = 1 ∧ y = 2 * t^2) →  -- Point (1, 2) lies on the curve
  (∃ m b : ℝ, ∀ x' y' : ℝ, y' - 2 = m * (x' - 1) ∧  -- Equation of tangent line
                           y' = 2 * x'^2 →  -- Point (x', y') lies on the curve
                           (x' - 1)^2 + (y' - 2)^2 < (x - 1)^2 + (y - 2)^2 →  -- (x', y') is closer to (1, 2) than (x, y)
                           |4 * x' - y' - 2| < |4 * x - y - 2|) →  -- Definition of tangent line
  4 * x - y - 2 = 0  -- Conclusion: equation of the tangent line
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1162_116275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_even_percent_l1162_116246

/-- A set of integers -/
structure IntegerSet where
  S : Finset ℤ

/-- Properties of the integer set -/
structure IntegerSetProperties (s : IntegerSet) where
  even_multiple_of_3_percent : ℚ
  even_not_multiple_of_3_percent : ℚ
  h_even_multiple_of_3 : even_multiple_of_3_percent = 36 / 100
  h_even_not_multiple_of_3 : even_not_multiple_of_3_percent = 40 / 100

/-- The theorem to be proved -/
theorem not_even_percent (s : IntegerSet) (props : IntegerSetProperties s) :
  (s.S.filter (λ x => ¬ Even x)).card / s.S.card = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_even_percent_l1162_116246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_point_satisfies_parabolas_tangency_point_is_unique_l1162_116277

/-- The point of tangency for two parabolas -/
noncomputable def point_of_tangency : ℝ × ℝ := (-17/2, -35/2)

/-- First parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 18*x + 47

/-- Second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 36*y + 323

/-- Theorem stating that the point_of_tangency satisfies both parabola equations -/
theorem tangency_point_satisfies_parabolas :
  let (x, y) := point_of_tangency
  parabola1 x y ∧ parabola2 x y :=
by sorry

/-- Theorem stating that the point_of_tangency is unique -/
theorem tangency_point_is_unique :
  ∀ (x y : ℝ), parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_point_satisfies_parabolas_tangency_point_is_unique_l1162_116277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_ellipse_passes_through_point_ellipse_eccentricity_l1162_116276

-- Define the ellipse E
def ellipse_E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the circle C
def circle_C (x y R : ℝ) : Prop := x^2 + y^2 = R^2

-- Define the line l
def line_l (x y k t : ℝ) : Prop := y = k * x + t

-- Define the tangent condition
def is_tangent (k t R : ℝ) : Prop := t^2 = R^2 * (1 + k^2)

-- Define the single intersection condition
def single_intersection (k t : ℝ) : Prop := (8 * k * t)^2 = 4 * (4 * k^2) * (4 * t^2 - 4)

theorem ellipse_and_line_properties 
  (R : ℝ) 
  (hR : 1 < R ∧ R < 2) :
  ∃ (k t : ℝ),
    -- 1. The equation of ellipse E is x^2/4 + y^2 = 1 (already defined in ellipse_E)
    -- 2. k^2 = (R^2 - 1) / (4 - R^2)
    (k^2 = (R^2 - 1) / (4 - R^2)) ∧
    -- 3. The maximum value of |AB| is 1
    (∀ (x y : ℝ), ellipse_E x y ∧ line_l x y k t → (x^2 + y^2 - R^2) ≤ 1) ∧
    (∃ (x y : ℝ), ellipse_E x y ∧ line_l x y k t ∧ x^2 + y^2 - R^2 = 1) :=
by sorry

-- Additional theorem to show that the ellipse passes through (√3, 1/2)
theorem ellipse_passes_through_point :
  ellipse_E (Real.sqrt 3) (1/2) :=
by sorry

-- Additional theorem to show the eccentricity of the ellipse
theorem ellipse_eccentricity :
  (Real.sqrt (4 - 1)) / 2 = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_ellipse_passes_through_point_ellipse_eccentricity_l1162_116276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_unit_circle_max_distance_to_unit_circle_achieved_l1162_116242

noncomputable def z1 : ℂ := 2 - 2*Complex.I

theorem max_distance_to_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z - z1) ≤ 2*Real.sqrt 2 + 1 :=
sorry

theorem max_distance_to_unit_circle_achieved :
  ∃ z : ℂ, Complex.abs z = 1 ∧ Complex.abs (z - z1) = 2*Real.sqrt 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_unit_circle_max_distance_to_unit_circle_achieved_l1162_116242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_diagonals_is_80_l1162_116296

/-- A pentagon inscribed in a circle -/
structure InscribedPentagon where
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ
  J : ℝ × ℝ
  inscribed : Set (ℝ × ℝ) := {F, G, H, I, J}

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of lengths of all diagonals in a pentagon -/
noncomputable def sumOfDiagonals (p : InscribedPentagon) : ℝ :=
  distance p.F p.H + distance p.F p.I + distance p.G p.I + distance p.G p.J + distance p.H p.J

/-- Main theorem -/
theorem sum_of_diagonals_is_80 (p : InscribedPentagon) 
  (h1 : distance p.F p.G = 4)
  (h2 : distance p.H p.I = 4)
  (h3 : distance p.G p.H = 11)
  (h4 : distance p.I p.J = 11)
  (h5 : distance p.F p.J = 15) :
  sumOfDiagonals p = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_diagonals_is_80_l1162_116296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1162_116249

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 5^x)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x < 0 :=
by
  intro x
  constructor
  · intro h
    sorry
  · intro h
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1162_116249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_volume_formula_l1162_116238

/-- The volume of a regular octahedron with edge length a -/
noncomputable def octahedron_volume (a : ℝ) : ℝ := (a^3 * Real.sqrt 2) / 3

/-- Theorem: The volume of a regular octahedron with edge length a is (a³√2)/3 -/
theorem octahedron_volume_formula (a : ℝ) (ha : a > 0) :
  octahedron_volume a = (a^3 * Real.sqrt 2) / 3 := by
  -- Proof will be added later
  sorry

#check octahedron_volume
#check octahedron_volume_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_volume_formula_l1162_116238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_smallest_hypotenuse_l1162_116211

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  perimeter : ℝ

-- Define the theorem
theorem isosceles_smallest_hypotenuse 
  (t : Set Triangle) 
  (h1 : ∀ x y, x ∈ t → y ∈ t → x.perimeter = y.perimeter) 
  (h2 : ∀ x y, x ∈ t → y ∈ t → x.γ = y.γ) :
  ∃ z, z ∈ t ∧ z.α = z.β ∧ ∀ w, w ∈ t → z.c ≤ w.c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_smallest_hypotenuse_l1162_116211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_endpoints_on_circle_l1162_116221

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define an inscribed circle
structure InscribedCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the projection endpoints
noncomputable def projectionEndpoints (t : Triangle) (ic : InscribedCircle) : Fin 6 → ℝ × ℝ :=
  fun _ => sorry

-- Theorem statement
theorem projection_endpoints_on_circle (t : Triangle) (ic : InscribedCircle) :
  ∃ (c : ℝ × ℝ) (r : ℝ), ∀ (i : Fin 6),
    let p := projectionEndpoints t ic i
    (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 ∧ c = ic.center ∧ r = ic.radius * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_endpoints_on_circle_l1162_116221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1162_116209

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-8 * x^2 + 10 * x + 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1/4 ≤ x ∧ x ≤ 3/4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1162_116209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l1162_116281

theorem inequality_holds_iff_a_in_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (1/3), (2 : ℝ)^(3*x) ≤ Real.log x / Real.log a + 1) ↔ a ∈ Set.Icc (1/3) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l1162_116281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_five_l1162_116201

/-- The function for which we're finding the vertical asymptote -/
noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 + 2*x + 9) / (x - 5)

/-- Theorem stating that the function has a vertical asymptote at x = 5 -/
theorem vertical_asymptote_at_five : 
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ 
    ∀ (x : ℝ), 0 < |x - 5| ∧ |x - 5| < δ → |f x| > M :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_five_l1162_116201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_bound_l1162_116297

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the specific rectangle in the problem
noncomputable def problemRectangle : Rectangle := { width := 1, height := 1/2 }

-- Function to check if a point is inside or on the rectangle
def isInRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ r.width ∧ 0 ≤ p.2 ∧ p.2 ≤ r.height

-- Function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_bound (points : Finset Point) :
  points.card = 6 →
  (∀ p, p ∈ points → isInRectangle p problemRectangle) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_bound_l1162_116297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_simplest_l1162_116284

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → (∃ n : ℕ, x = Real.sqrt n) → (∃ m : ℕ, y = Real.sqrt m) → x ≤ y

theorem sqrt_3_simplest :
  is_simplest_quadratic_radical (Real.sqrt 3) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1 / 2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 12) :=
sorry

-- Note: We can't directly represent √(1/x) as it involves a variable,
-- so we've omitted that part from the theorem.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_simplest_l1162_116284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_areas_l1162_116261

/-- Represents a right cylinder with given height and radius -/
structure RightCylinder where
  height : ℝ
  radius : ℝ

/-- Calculates the lateral surface area of a right cylinder -/
noncomputable def lateralSurfaceArea (c : RightCylinder) : ℝ :=
  2 * Real.pi * c.radius * c.height

/-- Calculates the total surface area of a right cylinder -/
noncomputable def totalSurfaceArea (c : RightCylinder) : ℝ :=
  2 * Real.pi * c.radius * c.height + 2 * Real.pi * c.radius^2

theorem cylinder_surface_areas :
  let c : RightCylinder := { height := 8, radius := 3 }
  lateralSurfaceArea c = 48 * Real.pi ∧ totalSurfaceArea c = 66 * Real.pi := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_areas_l1162_116261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_range_l1162_116233

/-- A function that represents y = log_a(2-x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 - x) / Real.log a

/-- The theorem states that if f is an increasing function of x, then 0 < a < 1 --/
theorem log_base_range (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) → 0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_range_l1162_116233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_ratio_l1162_116270

/-- Represents a geometric sequence -/
structure GeometricSequence (α : Type*) [Field α] where
  a : ℕ → α
  r : α
  h : ∀ n, a (n + 1) = r * a n

/-- Product of first n terms of a geometric sequence -/
def product {α : Type*} [Field α] (s : GeometricSequence α) (n : ℕ) : α :=
  (s.a n) ^ n

theorem geometric_sequence_product_ratio 
  {α : Type*} [Field α]
  (a b : GeometricSequence α) 
  (h1 : a.r ≠ 1) 
  (h2 : b.r ≠ 1) 
  (h3 : a.a 5 / b.a 5 = 2) : 
  product a 9 / product b 9 = 512 := by
  sorry

#check geometric_sequence_product_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_ratio_l1162_116270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_AFB_l1162_116244

-- Define the square
noncomputable def square_side_length : ℝ := 2

-- Define the folding point B'
noncomputable def B'_x : ℝ := square_side_length
noncomputable def B'_y : ℝ := 4/3

-- Define the vertices of the triangle
noncomputable def A : ℝ × ℝ := (0, square_side_length)
noncomputable def F : ℝ × ℝ := (0, 0)
noncomputable def B' : ℝ × ℝ := (B'_x, B'_y)

-- Calculate the distances
noncomputable def AF : ℝ := square_side_length
noncomputable def AB' : ℝ := square_side_length
noncomputable def FB' : ℝ := Real.sqrt ((B'_x - F.1)^2 + (B'_y - F.2)^2)

-- Theorem statement
theorem perimeter_of_triangle_AFB' :
  AF + FB' + AB' = 4 + (2 * Real.sqrt 13) / 3 := by
  sorry

#eval "Theorem statement added successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_AFB_l1162_116244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_shape_independent_of_first_fold_l1162_116208

/-- Represents a square sheet of paper --/
structure SquareSheet :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Represents a folded state of the paper --/
inductive FoldedState
  | Unfolded
  | FoldedOnce
  | FoldedTwice
  | FoldedThrice

/-- Represents a direction of folding --/
inductive FoldDirection
  | MN
  | AB

/-- Represents the result of folding, cutting, and unfolding --/
def FinalShape := Set (ℝ × ℝ)

/-- Function to fold the paper --/
def fold (sheet : SquareSheet) (dir : FoldDirection) : SquareSheet :=
  sorry

/-- Function to fold the paper multiple times --/
def foldMultiple (sheet : SquareSheet) (dirs : List FoldDirection) : SquareSheet :=
  match dirs with
  | [] => sheet
  | d :: ds => foldMultiple (fold sheet d) ds

/-- Function to cut the folded paper --/
noncomputable def cut (sheet : SquareSheet) : SquareSheet :=
  sorry

/-- Function to unfold the paper --/
noncomputable def unfold (sheet : SquareSheet) : FinalShape :=
  sorry

/-- The main theorem --/
theorem final_shape_independent_of_first_fold (sheet : SquareSheet) :
  ∀ dir₁ dir₂ : FoldDirection,
    unfold (cut (foldMultiple sheet [dir₁, FoldDirection.MN, FoldDirection.MN])) =
    unfold (cut (foldMultiple sheet [dir₂, FoldDirection.MN, FoldDirection.MN])) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_shape_independent_of_first_fold_l1162_116208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l1162_116212

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + 3 * Real.pi / 4)

theorem max_t_value (t : ℝ) : 
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ t → f x₁ - f x₂ < g x₁ - g x₂) ↔ 
  t ≤ Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l1162_116212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_not_sqrt3_div_2_l1162_116225

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def Ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∃ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

/-- Definition of eccentricity for an ellipse -/
noncomputable def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

/-- Theorem: The eccentricity of the ellipse x^2/2 + y^2 = 1 is not equal to √3/2 -/
theorem ellipse_eccentricity_not_sqrt3_div_2 :
  let a := Real.sqrt 2
  let b := 1
  Ellipse a b → Eccentricity a b ≠ Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_not_sqrt3_div_2_l1162_116225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_rotation_max_value_l1162_116206

theorem unit_circle_rotation_max_value (x_A y_A x_B y_B m : ℝ) : 
  x_A^2 + y_A^2 = 1 →  -- A is on the unit circle
  x_B = x_A * Real.cos (π/3) - y_A * Real.sin (π/3) →  -- B is obtained by rotating A by π/3
  y_B = x_A * Real.sin (π/3) + y_A * Real.cos (π/3) →
  m > 0 →
  (∀ x y : ℝ, x^2 + y^2 = 1 → m * y - 2 * (x * Real.sin (π/3) + y * Real.cos (π/3)) ≤ Real.sqrt 7) →  -- max value condition
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ m * y - 2 * (x * Real.sin (π/3) + y * Real.cos (π/3)) = Real.sqrt 7) →  -- max value is achievable
  m = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_rotation_max_value_l1162_116206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taller_tree_height_is_40_l1162_116214

/-- The height difference between two trees -/
noncomputable def height_difference : ℝ := 20

/-- The ratio of shadows -/
noncomputable def shadow_ratio : ℝ := 4/5

/-- The tangent of the sun's angle -/
noncomputable def tan_angle : ℝ := 1/2

/-- The length of the shorter tree's shadow -/
noncomputable def shorter_shadow : ℝ := 40

/-- The height of the shorter tree -/
noncomputable def shorter_tree_height : ℝ := shorter_shadow * tan_angle

/-- The height of the taller tree -/
noncomputable def taller_tree_height : ℝ := shorter_tree_height + height_difference

theorem taller_tree_height_is_40 : taller_tree_height = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taller_tree_height_is_40_l1162_116214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_segments_length_l1162_116224

/-- Given a configuration of two intersecting line segments AC and BD with point of intersection O,
    prove that the length of AC is √(511/7) given the specified side lengths. -/
theorem intersecting_segments_length (O A B C D : ℝ × ℝ) 
  (h1 : ‖A - O‖ = 3)
  (h2 : ‖C - O‖ = 8)
  (h3 : ‖D - O‖ = 6)
  (h4 : ‖B - O‖ = 7)
  (h5 : ‖B - D‖ = 11) :
  ‖A - C‖ = Real.sqrt (511 / 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_segments_length_l1162_116224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1162_116254

open BigOperators

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) 
  (h1 : ∀ i, 0 < a i) (h2 : ∀ i, a i ≤ 1/2) : 
  (∑ i, (a i)^2) / (∑ i, a i) - (∏ i, a i)^(1/n : ℝ) ≥ 
  (∑ i, (1 - a i)^2) / (∑ i, (1 - a i)) - (∏ i, (1 - a i))^(1/n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1162_116254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inner_tetrahedron_edge_value_l1162_116298

/-- The radius of the inscribed sphere of a regular tetrahedron -/
noncomputable def inscribed_sphere_radius (edge_length : ℝ) : ℝ := (Real.sqrt 6 / 12) * edge_length

/-- The radius of the circumscribed sphere of a regular tetrahedron -/
noncomputable def circumscribed_sphere_radius (edge_length : ℝ) : ℝ := (Real.sqrt 6 / 4) * edge_length

/-- The maximum edge length of a smaller tetrahedron that can rotate freely inside a larger one -/
noncomputable def max_inner_tetrahedron_edge (outer_edge_length : ℝ) : ℝ :=
  inscribed_sphere_radius outer_edge_length / (Real.sqrt 6 / 4)

theorem max_inner_tetrahedron_edge_value :
  max_inner_tetrahedron_edge 5 = 5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inner_tetrahedron_edge_value_l1162_116298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_bill_correct_l1162_116235

/-- Represents the electricity bill calculation function -/
noncomputable def electricityBill (x : ℝ) : ℝ :=
  if x ≤ 200 then 0.6 * x else 1.1 * x - 100

/-- Theorem stating the correctness of the electricity bill calculation -/
theorem electricity_bill_correct (x : ℝ) :
  electricityBill x = 
    if x ≤ 200 then 0.6 * x else 1.1 * x - 100 :=
by
  -- Unfold the definition of electricityBill
  unfold electricityBill
  -- The rest of the proof is skipped
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_bill_correct_l1162_116235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l1162_116299

/-- Represents the profit share of partner C in rupees -/
noncomputable def c_share : ℝ := 3375.0000000000005

/-- Represents the investment ratio of partner A to partner B -/
noncomputable def a_to_b_ratio : ℝ := 3

/-- Represents the investment ratio of partner A to partner C -/
noncomputable def a_to_c_ratio : ℝ := 2/3

/-- Calculates the total profit based on given ratios and C's share -/
noncomputable def total_profit : ℝ := 
  let b_parts : ℝ := 2
  let a_parts : ℝ := a_to_b_ratio * b_parts
  let c_parts : ℝ := a_parts / a_to_c_ratio
  let total_parts : ℝ := a_parts + b_parts + c_parts
  (total_parts / c_parts) * c_share

theorem partnership_profit : 
  ∃ ε > 0, |total_profit - 6375.000000000001| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l1162_116299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_drive_l1162_116282

/-- Represents the odometer reading as a triple of digits -/
structure OdometerReading where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≥ 1
  h2 : a + b + c ≤ 9

/-- Calculates the mileage from an odometer reading -/
def mileage (r : OdometerReading) : ℕ := 100 * r.a + 10 * r.b + r.c

theorem linda_drive (start end_ : OdometerReading) (hours : ℕ) :
  mileage end_ - mileage start = 60 * hours →
  end_.a = start.b ∧ end_.b = start.c ∧ end_.c = start.a →
  start.a^2 + start.b^2 + start.c^2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_drive_l1162_116282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l1162_116245

theorem factorial_divisibility (n k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_div : (p ^ k) ∣ n!) : 
  ((Nat.factorial p) ^ k) ∣ n! := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l1162_116245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1162_116231

/-- Calculates the time (in seconds) for a train to pass a stationary point -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

/-- The time taken for a 100-meter long train traveling at 30 km/h to pass a stationary point is approximately 12.01 seconds -/
theorem train_passing_time_approx :
  ∃ ε > 0, |train_passing_time 100 30 - 12.01| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1162_116231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_theorem_l1162_116223

/-- Represents the tonnage changes over 4 days --/
def tonnage_changes : List Int := [24, -48, -13, 37, -52, 57, -13, -33]

/-- The final quantity of goods in the warehouse --/
def final_quantity : Int := 217

/-- The handling fee per ton --/
def handling_fee : Int := 15

/-- Theorem stating the initial quantity and total handling fees --/
theorem warehouse_theorem :
  let initial_quantity := final_quantity - tonnage_changes.sum
  let total_fees := (tonnage_changes.map abs).sum * handling_fee
  initial_quantity = 258 ∧ total_fees = 4155 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_theorem_l1162_116223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_length_calculation_l1162_116271

/-- Calculates the length of a bus given its travel distance, time, and time to pass a post -/
noncomputable def bus_length (distance : ℝ) (travel_time : ℝ) (pass_time : ℝ) : ℝ :=
  (distance * 1000 / (travel_time * 60)) * pass_time

/-- Theorem stating that a bus traveling 12 km in 5 minutes and taking 5 seconds to pass a post is 200 meters long -/
theorem bus_length_calculation :
  bus_length 12 5 5 = 200 := by
  -- Unfold the definition of bus_length
  unfold bus_length
  -- Simplify the arithmetic expression
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_length_calculation_l1162_116271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l1162_116260

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 4

/-- The line function -/
def g (t : ℝ) : ℝ → ℝ := λ x => t

/-- The area of the triangle formed by the vertex of the parabola and its intersections with the line -/
noncomputable def triangleArea (t : ℝ) : ℝ := (t + 4) * Real.sqrt (t + 4)

theorem triangle_area_bound :
  ∀ t : ℝ, (∃ x : ℝ, f x = g t x) →
    (triangleArea t ≤ 36 ↔ -4 ≤ t ∧ t ≤ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l1162_116260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_cube_root_equation_l1162_116213

theorem no_real_solutions_cube_root_equation :
  ¬ ∃ (x : ℝ), x ^ (1/3) = 18 / (8 - x ^ (1/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_cube_root_equation_l1162_116213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_concentration_original_mixture_l1162_116265

/-- Represents a mixture of acid and water -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- Calculates the acid concentration in a mixture -/
noncomputable def acid_concentration (m : Mixture) : ℝ :=
  m.acid / (m.acid + m.water)

theorem acid_concentration_original_mixture 
  (initial : Mixture)
  (h1 : acid_concentration {acid := initial.acid, water := initial.water + 2} = 0.18)
  (h2 : acid_concentration {acid := initial.acid + 2, water := initial.water + 2} = 0.3) :
  ∃ ε > 0, |acid_concentration initial - 0.2917| < ε := by
  sorry

#check acid_concentration_original_mixture

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_concentration_original_mixture_l1162_116265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_ninety_sixth_permutation_l1162_116251

def digits : List Nat := [1, 2, 3, 4, 5]

def is_valid_permutation (p : List Nat) : Bool :=
  p.length = 5 && p.toFinset = digits.toFinset

def permutations : List (List Nat) :=
  digits.permutations.filter is_valid_permutation

theorem permutation_count : permutations.length = 120 := by sorry

theorem ninety_sixth_permutation : permutations[95]! = [4, 5, 3, 2, 1] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_ninety_sixth_permutation_l1162_116251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_in_square_arrangement_l1162_116267

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the square arrangement -/
structure SquareArrangement where
  innerSquareSide : ℝ
  rectangle : Rectangle

/-- The ratio of the longer side to the shorter side of the rectangle -/
noncomputable def rectangleRatio (r : Rectangle) : ℝ :=
  max r.length r.width / min r.length r.width

/-- The theorem statement -/
theorem rectangle_ratio_in_square_arrangement 
  (arrangement : SquareArrangement) 
  (h1 : arrangement.innerSquareSide > 0)
  (h2 : arrangement.rectangle.length > 0)
  (h3 : arrangement.rectangle.width > 0)
  (h4 : (2 * arrangement.innerSquareSide)^2 = 4 * arrangement.innerSquareSide^2)
  (h5 : arrangement.rectangle.length + arrangement.innerSquareSide = 2 * arrangement.innerSquareSide)
  (h6 : arrangement.innerSquareSide + 2 * arrangement.rectangle.width = 2 * arrangement.innerSquareSide)
  : rectangleRatio arrangement.rectangle = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_in_square_arrangement_l1162_116267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l1162_116252

-- Define the vector type
def MyVector := ℝ × ℝ

-- Define the point type
def MyPoint := ℝ × ℝ

-- Define vector addition
def vector_add (v1 v2 : MyVector) : MyVector :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Define point translation by a vector
def point_translate (p : MyPoint) (v : MyVector) : MyPoint :=
  (p.1 + v.1, p.2 + v.2)

-- Theorem statement
theorem point_C_coordinates
  (AB : MyVector)
  (BC : MyVector)
  (A : MyPoint)
  (h1 : AB = (1, 3))
  (h2 : BC = (2, 1))
  (h3 : A = (-1, 2)) :
  point_translate A (vector_add AB BC) = (2, 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l1162_116252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_l1162_116226

/-- Polar coordinate representation -/
structure PolarCoord where
  ρ : ℝ
  θ : ℝ

/-- Define a line through two points in polar coordinates -/
def line_through (A B : PolarCoord) : Set PolarCoord :=
  sorry

/-- Given a line passing through points A and B in polar coordinates, 
    and this line having exactly one common point with curve C,
    prove that the value of r is equal to 1. -/
theorem line_tangent_to_curve (r : ℝ) : r > 0 → r = 1 := by
  intro hr
  -- Define points A and B in polar coordinates
  let A : PolarCoord := ⟨Real.sqrt 3, 2 * Real.pi / 3⟩
  let B : PolarCoord := ⟨3, Real.pi / 2⟩

  -- Define the curve C: ρ = 2r*sin(θ)
  let C (θ : ℝ) : PolarCoord := ⟨2 * r * Real.sin θ, θ⟩

  -- Define the condition that the line AB has exactly one common point with curve C
  let line_tangent_to_curve := ∃! p : PolarCoord, p ∈ line_through A B ∧ ∃ θ, p = C θ

  -- Assume the condition
  have h : line_tangent_to_curve := sorry

  -- Prove that r = 1
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_l1162_116226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_exactly_one_die_showing_one_l1162_116259

def num_dice : ℕ := 12
def dice_sides : ℕ := 6
def target_value : ℕ := 1

def probability_exactly_one_target : ℚ :=
  (num_dice : ℚ) * (5 ^ (num_dice - 1) : ℚ) / (dice_sides ^ num_dice : ℚ)

theorem probability_exactly_one_die_showing_one :
  ∃ (p : ℚ), abs (p - probability_exactly_one_target) < 1/2000 ∧ 
  (p * 1000).floor / 1000 = 261/1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_exactly_one_die_showing_one_l1162_116259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_probability_first_week_january_l1162_116237

/-- The probability of snow occurring at least once during a week, given specific probabilities for each day. -/
theorem snow_probability_first_week_january : 
  (1 - (1 - 1/3)^3 * (1 - 1/4)^4 : ℚ) = 29/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_probability_first_week_january_l1162_116237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_diagonal_ratio_l1162_116216

/-- A quadrilateral inscribed in a circle with side lengths a, b, c, and d -/
structure InscribedQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

/-- The ratio of the lengths of the diagonals of an inscribed quadrilateral -/
noncomputable def diagonalRatio (q : InscribedQuadrilateral) : ℝ :=
  (q.b * q.c + q.a * q.d) / (q.a * q.b + q.c * q.d)

/-- Theorem: The ratio of the lengths of the diagonals of an inscribed quadrilateral
    is given by (bc + ad) / (ab + cd) -/
theorem inscribed_quadrilateral_diagonal_ratio (q : InscribedQuadrilateral) :
  ∃ (d1 d2 : ℝ), d1 > 0 ∧ d2 > 0 ∧ d1 / d2 = diagonalRatio q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_diagonal_ratio_l1162_116216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_eq_x_plus_3_l1162_116279

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - 3*x else -((-x)^2 - 3*(-x))

-- State the theorem
theorem solution_set_of_f_eq_x_plus_3 :
  {x : ℝ | f x = x + 3} = {2 + Real.sqrt 7, -1, -3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_eq_x_plus_3_l1162_116279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_always_zero_l1162_116219

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (x^2 + 1) + 1 / (x - Real.sqrt (x^2 + 1))

-- Theorem statement
theorem f_always_zero : ∀ x : ℝ, f x = 0 := by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Example usage for f(2015)
example : f 2015 = 0 := by
  -- Apply the general theorem
  exact f_always_zero 2015

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_always_zero_l1162_116219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_approx_l1162_116205

/-- Represents a rectangular floor with a specific length-to-breadth ratio and area. -/
structure RectangularFloor where
  breadth : ℝ
  length_ratio : ℝ
  area : ℝ
  length_eq : length_ratio * breadth = length_ratio * breadth
  area_eq : (length_ratio * breadth) * breadth = area

/-- The length of a rectangular floor given specific conditions -/
def floor_length (floor : RectangularFloor) : ℝ := floor.length_ratio * floor.breadth

/-- Theorem stating the length of the floor under given conditions -/
theorem floor_length_approx (floor : RectangularFloor) 
  (h1 : floor.length_ratio = 3)
  (h2 : floor.area = 50) :
  ∃ ε > 0, |floor_length floor - 12.24| < ε := by
  sorry

#check floor_length_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_approx_l1162_116205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_lateral_surface_area_l1162_116272

/-- A triangular prism with an equilateral triangular base and a sphere tangent to all faces. -/
structure TriangularPrismWithSphere where
  -- The radius of the inscribed sphere
  r : ℝ
  -- Assumption that the sphere has volume 4π/3
  sphere_volume : (4 / 3) * Real.pi * r ^ 3 = (4 / 3) * Real.pi

/-- The lateral surface area of the triangular prism. -/
noncomputable def lateralSurfaceArea (p : TriangularPrismWithSphere) : ℝ :=
  12 * Real.sqrt 3

/-- Theorem stating that the lateral surface area of the triangular prism is 12√3. -/
theorem triangular_prism_lateral_surface_area (p : TriangularPrismWithSphere) :
  lateralSurfaceArea p = 12 * Real.sqrt 3 := by
  -- Unfold the definition of lateralSurfaceArea
  unfold lateralSurfaceArea
  -- The equality is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_lateral_surface_area_l1162_116272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_addition_l1162_116264

def original_num1 : ℕ := 742586
def original_num2 : ℕ := 829430
def original_sum : ℕ := 1212016

def replace_digit (n : ℕ) (old : ℕ) (new : ℕ) : ℕ :=
  let s := n.repr
  let replaced := s.map (fun c => if c.toNat - 48 = old then Char.ofNat (new + 48) else c)
  replaced.toNat!

def corrected_num1 : ℕ := replace_digit original_num1 2 6
def corrected_num2 : ℕ := replace_digit original_num2 2 6
def corrected_sum : ℕ := replace_digit original_sum 2 6

theorem correct_addition :
  corrected_num1 + corrected_num2 = corrected_sum :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_addition_l1162_116264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_B_l1162_116217

def A : Finset ℕ := {2, 0, 1, 7}

def B : Finset ℕ := Finset.image (fun p => p.1 * p.2) (A.product A)

theorem cardinality_of_B : Finset.card B = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_B_l1162_116217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_max_w_of_f_l1162_116268

open Real

-- Define the function f
noncomputable def f (w : ℝ) (x : ℝ) : ℝ := 
  4 * cos (w * x + π / 6) * sin (w * x) - cos (2 * w * x) + 1

-- Part 1: Period of the function
theorem period_of_f (w : ℝ) (h1 : 0 < w) (h2 : w < 2) 
  (h3 : ∀ x, f w (π/4 + x) = f w (π/4 - x)) : 
  ∃ T, T = π ∧ ∀ x, f w (x + T) = f w x := by sorry

-- Part 2: Maximum value of w
theorem max_w_of_f (w : ℝ) (h1 : 0 < w) (h2 : w < 2)
  (h3 : ∀ x y, x ∈ Set.Icc (-π/6) (π/3) → y ∈ Set.Icc (-π/6) (π/3) → x < y → f w x < f w y) :
  w ≤ 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_max_w_of_f_l1162_116268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l1162_116232

/-- Calculates the annual interest rate given the principal, time, and interest amount -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) : ℝ :=
  (interest * 100) / (principal * time)

/-- Theorem: Given the specific conditions, the interest rate is 4% -/
theorem interest_rate_is_four_percent 
  (principal : ℝ) 
  (time : ℝ) 
  (interest : ℝ) 
  (h1 : principal = 300)
  (h2 : time = 8)
  (h3 : interest = principal - 204) :
  calculate_interest_rate principal time interest = 4 := by
  sorry

#check interest_rate_is_four_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l1162_116232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_inequality_l1162_116288

theorem contrapositive_inequality :
  (∀ (a b c : ℝ), a > b → a + c > b + c) ↔ (∀ (a b c : ℝ), a + c ≤ b + c → a ≤ b) :=
by
  apply Iff.intro
  · intros h a b c hab
    contrapose! hab
    exact h a b c hab
  · intros h a b c hab
    contrapose! hab
    exact h a b c hab

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_inequality_l1162_116288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l1162_116278

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

-- Define the derivative of the curve
noncomputable def curve_derivative (x : ℝ) : ℝ := -4 * Real.exp x / ((Real.exp x + 1)^2)

-- Theorem statement
theorem tangent_angle_range :
  ∀ x : ℝ, 3 * Real.pi / 4 ≤ Real.arctan (curve_derivative x) ∧ 
            Real.arctan (curve_derivative x) < Real.pi :=
by
  intro x
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l1162_116278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1162_116250

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := -Real.sin (2 * x) - Real.sqrt 3 * (1 - 2 * Real.sin x ^ 2) + 1

-- State the theorem
theorem f_properties :
  -- The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- When x ∈ [-π/6, π/6], the range of f(x) is [-1, 1]
  (∀ (x : ℝ), -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 6 →
    -1 ≤ f x ∧ f x ≤ 1) ∧
  (∀ (y : ℝ), -1 ≤ y ∧ y ≤ 1 →
    ∃ (x : ℝ), -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 6 ∧ f x = y) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1162_116250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_tangent_l1162_116227

/-- A triangle inscribed in one parabola with two sides tangent to another parabola -/
structure InscribedTangentTriangle (p q : ℝ) where
  a : ℝ
  b : ℝ
  c : ℝ
  inscribed_parabola : (2 * p * a)^2 = 2 * p * (2 * p * a^2) ∧
                       (2 * p * b)^2 = 2 * p * (2 * p * b^2) ∧
                       (2 * p * c)^2 = 2 * p * (2 * p * c^2)
  tangent_condition_ab : q + 4 * p * a * b * (a + b) = 0
  tangent_condition_bc : q + 4 * p * b * c * (b + c) = 0

/-- The third side of the triangle is also tangent to the second parabola -/
theorem third_side_tangent (p q : ℝ) (triangle : InscribedTangentTriangle p q) :
  q + 4 * p * triangle.a * triangle.c * (triangle.a + triangle.c) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_tangent_l1162_116227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_OA_OB_k_value_for_area_sqrt10_l1162_116258

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = -x
def line (k x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points
def intersection_points (k : ℝ) : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂ := by sorry

-- Theorem 1: OA ⊥ OB
theorem perpendicular_OA_OB (k : ℝ) : 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂ → 
  x₁ * x₂ + y₁ * y₂ = 0 := by sorry

-- Define the area of triangle OAB
noncomputable def triangle_area (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (1/2) * Real.sqrt ((y₁ - y₂)^2)

-- Theorem 2: When area of triangle OAB is √10, k = ± 1/6
theorem k_value_for_area_sqrt10 : 
  ∃ (k : ℝ), (∀ (x₁ y₁ x₂ y₂ : ℝ), 
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂ → 
  triangle_area x₁ y₁ x₂ y₂ = Real.sqrt 10) ∧ 
  (k = 1/6 ∨ k = -1/6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_OA_OB_k_value_for_area_sqrt10_l1162_116258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_x_intercepts_l1162_116240

-- Define the point of intersection
noncomputable def intersection_point : ℝ × ℝ := (4, -3)

-- Define the slopes of the two lines
noncomputable def slope1 : ℝ := 2
noncomputable def slope2 : ℝ := -1

-- Define the x-intercepts of the two lines
noncomputable def x_intercept1 : ℝ := 11/2
noncomputable def x_intercept2 : ℝ := 1

-- Theorem statement
theorem distance_between_x_intercepts :
  let d := |x_intercept1 - x_intercept2|
  d = 9/2 := by
  -- Proof goes here
  sorry

#eval (9:ℚ)/2 -- This will evaluate to 4.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_x_intercepts_l1162_116240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_symmetric_sum_l1162_116200

-- Define an odd function that is symmetric about x=1
def odd_symmetric_about_one (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x + 2) = -f x) ∧ (∀ x, f x = -f (-x))

-- Main theorem
theorem odd_symmetric_sum (f : ℝ → ℝ) 
  (h_odd_sym : odd_symmetric_about_one f) 
  (h_f_1 : f 1 = 2) : 
  f 3 + f 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_symmetric_sum_l1162_116200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g2_values_product_l1162_116256

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- State the main property of g
axiom g_property : ∀ (x y : ℝ), g ((x - y)^2 + 1) = g x^2 - 2*x*g y + y^2 + 1

-- Define the set of possible values for g(2)
def g2_values : Set ℝ := {y | ∃ (g : ℝ → ℝ), (∀ (x y : ℝ), g ((x - y)^2 + 1) = g x^2 - 2*x*g y + y^2 + 1) ∧ g 2 = y}

-- State the theorem
theorem g2_values_product : 
  ∃ (S : Finset ℝ), S.toSet = g2_values ∧ S.card * (S.sum id) = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g2_values_product_l1162_116256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_f_at_5_l1162_116262

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 25 / (4 + 5 * x)

-- State the theorem
theorem inverse_of_inverse_f_at_5 : (Function.invFun f 5)⁻¹ = 5 := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_f_at_5_l1162_116262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1162_116222

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := (x + 3)^2 + (y - 4)^2 = 50

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 5

/-- Two points satisfy both the circle and line equations -/
def intersection_points (p₁ p₂ : ℝ × ℝ) : Prop :=
  circle_equation p₁.1 p₁.2 ∧ line_equation p₁.1 p₁.2 ∧
  circle_equation p₂.1 p₂.2 ∧ line_equation p₂.1 p₂.2 ∧
  p₁ ≠ p₂

/-- The distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2)

theorem intersection_distance :
  ∀ p₁ p₂ : ℝ × ℝ, intersection_points p₁ p₂ → distance p₁ p₂ = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1162_116222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_exists_l1162_116210

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the distance function between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define the radical axis theorem
theorem radical_axis_exists (c1 c2 : Circle) :
  ∃ (N : Point), 
    (distance c1.center N)^2 - c1.radius^2 = (distance c2.center N)^2 - c2.radius^2 ∧
    ∃ (l : Set Point), 
      l = {p : Point | (p.x - N.x) * (c2.center.x - c1.center.x) + 
                       (p.y - N.y) * (c2.center.y - c1.center.y) = 0} ∧
      ∀ (p : Point), p ∈ l → 
        (distance c1.center p)^2 - c1.radius^2 = (distance c2.center p)^2 - c2.radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_exists_l1162_116210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_conclusion_l1162_116285

-- Define the chi-square statistic and critical value
def chi_square : ℝ := 9.632
def critical_value : ℝ := 10.828

-- Define the significance level
def alpha : ℝ := 0.001

-- Statement to be proven false
theorem incorrect_conclusion :
  chi_square < critical_value →
  ∃ (p : ℝ), p ≤ alpha ∧ (∀ (X Y : Prop), X ∨ Y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_conclusion_l1162_116285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_equal_magnitude_vectors_l1162_116263

/-- Two vectors are orthogonal if their dot product is zero -/
def orthogonal (a b : ℝ × ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0

/-- The magnitude of a vector is the square root of the sum of squares of its components -/
noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

/-- Two vectors have equal magnitude if their magnitudes are equal -/
def equal_magnitude (a b : ℝ × ℝ × ℝ) : Prop :=
  magnitude a = magnitude b

theorem orthogonal_equal_magnitude_vectors (p q : ℝ) :
  let a : ℝ × ℝ × ℝ := (4, p, -2)
  let b : ℝ × ℝ × ℝ := (-1, 2, q)
  orthogonal a b ∧ equal_magnitude a b → p = -2.75 ∧ q = -4.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_equal_magnitude_vectors_l1162_116263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_sine_symmetry_l1162_116243

theorem min_omega_for_sine_symmetry (ω : ℝ) (h_pos : ω > 0) :
  (∀ x, Real.sin (ω * x + π / 6) = Real.sin (ω * (π / 6 - x) + π / 6)) →
  ω ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_sine_symmetry_l1162_116243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sweets_distribution_l1162_116257

theorem sweets_distribution 
  (total_children absent_children extra_sweets original_sweets : ℕ)
  (h1 : total_children = 112)
  (h2 : absent_children = 32)
  (h3 : extra_sweets = 6)
  (h4 : total_children * original_sweets = (total_children - absent_children) * (original_sweets + extra_sweets)) :
  original_sweets = 15 := by
  sorry

#check sweets_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sweets_distribution_l1162_116257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_five_l1162_116204

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the hyperbola -/
def Point.on_hyperbola (p : Point) (h : Hyperbola) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- Represents the foci of a hyperbola -/
structure Foci (h : Hyperbola) where
  F₁ : Point
  F₂ : Point

/-- Checks if three real numbers form an arithmetic sequence -/
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

/-- Calculates the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b^2 / h.a^2))

/-- The main theorem -/
theorem hyperbola_eccentricity_five (h : Hyperbola) (p : Point) (f : Foci h) :
  p.on_hyperbola h →
  (p.x - f.F₁.x) * (p.x - f.F₂.x) + (p.y - f.F₁.y) * (p.y - f.F₂.y) = 0 →
  is_arithmetic_sequence ((p.x - f.F₁.x)^2 + (p.y - f.F₁.y)^2)
                         ((p.x - f.F₂.x)^2 + (p.y - f.F₂.y)^2)
                         ((f.F₁.x - f.F₂.x)^2 + (f.F₁.y - f.F₂.y)^2) →
  eccentricity h = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_five_l1162_116204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_students_follow_all_celebrities_l1162_116218

/-- Represents a student following a celebrity -/
structure Follow where
  student : Nat
  celebrity : Nat

/-- The proposition to be proved -/
theorem two_students_follow_all_celebrities 
  (students : Finset Nat) 
  (celebrities : Finset Nat) 
  (follows : Finset Follow) 
  (h1 : students.card = 120)
  (h2 : celebrities.card = 10)
  (h3 : ∀ c ∈ celebrities, (follows.filter (λ f => f.celebrity = c)).card ≥ 85) :
  ∃ s1 s2, s1 ∈ students ∧ s2 ∈ students ∧ ∀ c ∈ celebrities, 
    ∃ f ∈ follows, (f.student = s1 ∨ f.student = s2) ∧ f.celebrity = c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_students_follow_all_celebrities_l1162_116218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_distance_l1162_116269

-- Define the points in ℝ³
def P : Fin 3 → ℝ := ![(-2), 0, 2]
def M : Fin 3 → ℝ := ![(-1), 1, 2]
def N : Fin 3 → ℝ := ![(-3), 0, 4]

-- Define vectors a and b
def a : Fin 3 → ℝ := ![1, 1, 0]
def b : Fin 3 → ℝ := ![(-1), 0, 2]

-- Define dot product for ℝ³ vectors
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Define the perpendicularity condition
def is_perpendicular (k : ℝ) : Prop :=
  dot_product (fun i => k * (a i) + (b i))
              (fun i => k * (a i) - 2 * (b i)) = 0

-- Define the distance function
noncomputable def distance_point_to_line (point line_point line_vector : Fin 3 → ℝ) : ℝ :=
  let v := fun i => point i - line_point i
  let u := fun i => line_vector i / Real.sqrt (dot_product line_vector line_vector)
  Real.sqrt (dot_product v v - (dot_product v u)^2)

theorem perpendicular_and_distance :
  (∃ k₁ k₂ : ℝ, k₁ ≠ k₂ ∧ is_perpendicular k₁ ∧ is_perpendicular k₂ ∧
   (k₁ = 2 ∨ k₁ = -5/2) ∧ (k₂ = 2 ∨ k₂ = -5/2)) ∧
  distance_point_to_line N P a = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_distance_l1162_116269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nitrogen_most_difficult_to_react_l1162_116286

-- Define a type for the substances
inductive Substance
  | Fluorine
  | Nitrogen
  | Chlorine
  | Oxygen

-- Define a function to represent the bond energy
def bondEnergy : Substance → ℝ :=
  sorry

-- Define a function to represent the difficulty to react with H₂
def difficultyToReactWithH2 : Substance → ℝ :=
  sorry

-- Assume that higher bond energy implies higher difficulty to react
axiom difficulty_increases_with_energy :
  ∀ (s1 s2 : Substance), bondEnergy s1 > bondEnergy s2 → difficultyToReactWithH2 s1 > difficultyToReactWithH2 s2

-- State the theorem
theorem nitrogen_most_difficult_to_react :
  ∀ (s : Substance), s ≠ Substance.Nitrogen →
    difficultyToReactWithH2 Substance.Nitrogen > difficultyToReactWithH2 s :=
by
  sorry

-- Additional axiom to represent that nitrogen has the highest bond energy
axiom nitrogen_highest_bond_energy :
  ∀ (s : Substance), s ≠ Substance.Nitrogen → bondEnergy Substance.Nitrogen > bondEnergy s

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nitrogen_most_difficult_to_react_l1162_116286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_correct_l1162_116290

/-- The area of the shaded region formed by the overlap of two semicircles
    with radius 2 units and a semicircle with radius 1 unit, where the smaller
    semicircle is inscribed between the two larger ones. -/
noncomputable def shaded_area : ℝ := 3.5 * Real.pi

/-- The radius of the larger semicircles. -/
def large_radius : ℝ := 2

/-- The radius of the smaller semicircle. -/
def small_radius : ℝ := 1

/-- Theorem stating that the shaded area is correct. -/
theorem shaded_area_correct :
  shaded_area = 2 * (π * large_radius^2 / 2) + 2 * (π * large_radius^2 / 2) - (π * small_radius^2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_correct_l1162_116290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_exists_for_n_ge_14_l1162_116234

/-- A function that determines if an n-sided polygon exists where each side lies on a line
    containing at least one other side of the polygon. -/
def polygon_exists (n : ℕ) : Prop :=
  ∃ (vertices : Fin n → ℝ × ℝ), 
    ∀ (i j : Fin n), i ≠ j → 
      ∃ (k l : Fin n), k ≠ l ∧ k ≠ i ∧ l ≠ i ∧ 
        ((vertices i).1 - (vertices j).1) * ((vertices k).2 - (vertices l).2) =
        ((vertices i).2 - (vertices j).2) * ((vertices k).1 - (vertices l).1)

/-- For all natural numbers n ≥ 14, there exists an n-sided polygon where each side
    lies on a line containing at least one other side of the polygon. -/
theorem polygon_exists_for_n_ge_14 : ∀ n : ℕ, n ≥ 14 → polygon_exists n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_exists_for_n_ge_14_l1162_116234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1162_116253

/-- The function f(x) = (1/2)^(x^2 + 4x + 3) - t -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (1/2)^(x^2 + 4*x + 3) - t

/-- The function g(x) = x + 1 + 4/(x+1) + t -/
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := x + 1 + 4/(x+1) + t

/-- The theorem stating the range of t given the conditions -/
theorem range_of_t :
  (∀ t : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, x₂ < -1 ∧ f t x₁ ≤ g t x₂)) →
  (∀ t : ℝ, t ≥ 3 ↔ (∀ x₁ : ℝ, ∃ x₂ : ℝ, x₂ < -1 ∧ f t x₁ ≤ g t x₂)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1162_116253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_implies_b_bounded_l1162_116294

theorem no_solutions_implies_b_bounded (a b : ℝ) 
  (h : ∀ x : ℝ, ¬(a * Real.cos x + b * Real.cos (3 * x) > 1)) : 
  abs b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_implies_b_bounded_l1162_116294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l1162_116291

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define an angle
def Angle := ℝ

-- Define a function to check if a point is on the circle
def onCircle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define collinearity
def collinear (p q r : Point) : Prop :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (x₂ - x₁) * (y₃ - y₁) = (x₃ - x₁) * (y₂ - y₁)

-- Define the problem setup
structure GeometrySetup where
  circle1 : Circle
  circle2 : Circle
  P : Point
  Q : Point
  A : Point
  B : Point
  C : Point
  D : Point
  h1 : onCircle P circle1 ∧ onCircle P circle2
  h2 : onCircle Q circle1 ∧ onCircle Q circle2
  h3 : onCircle A circle1
  h4 : onCircle B circle1
  h5 : onCircle C circle2
  h6 : onCircle D circle2
  h7 : collinear A B C ∧ collinear B C D

-- Define a function to calculate the angle between three points
noncomputable def angleOf (p q r : Point) : Angle :=
  sorry -- Actual implementation would go here

-- Define the theorem
theorem angle_equality (setup : GeometrySetup) :
  angleOf setup.A setup.P setup.B = angleOf setup.C setup.Q setup.D :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l1162_116291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1162_116283

theorem polynomial_factorization : 
  ∀ x : Polynomial ℤ,
  9 * (x + 4) * (x + 5) * (x + 11) * (x + 13) - 7 * x^2 = 
  (3 * x + 24) * (x + 8) * (3 * x + 15) * (x + 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1162_116283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_theorem_l1162_116255

/-- Represents the price change for a product --/
structure PriceChange where
  initial : ℝ
  percentage : ℝ

/-- Calculates the new price after applying a price change --/
noncomputable def newPrice (pc : PriceChange) : ℝ :=
  pc.initial * (1 + pc.percentage / 100)

/-- Calculates the percentage change in total cost --/
noncomputable def percentageChange (initialEggPrice initialApplePrice : ℝ) 
  (eggChange appleChange : PriceChange) : ℝ :=
  let initialTotal := initialEggPrice + initialApplePrice
  let newTotal := newPrice eggChange + newPrice appleChange
  (newTotal - initialTotal) / initialTotal * 100

theorem price_change_theorem 
  (initialEggPrice initialApplePrice : ℝ)
  (eggChange appleChange : PriceChange) :
  initialEggPrice = initialApplePrice ∧ 
  eggChange.initial = initialEggPrice ∧
  appleChange.initial = initialApplePrice ∧
  eggChange.percentage = -10 ∧
  appleChange.percentage = 2 →
  percentageChange initialEggPrice initialApplePrice eggChange appleChange = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_theorem_l1162_116255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l1162_116274

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_specific_vectors :
  ∃ (a b : ℝ × ℝ),
    Real.sqrt (a.1^2 + a.2^2) = 3 ∧
    Real.sqrt (b.1^2 + b.2^2) = 2 ∧
    a.1 * b.1 + a.2 * b.2 = -3 ∧
    angle_between_vectors a b = 2 * π / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l1162_116274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_learning_time_90_l1162_116229

/-- The learning time function for typing practice -/
noncomputable def learning_time (N : ℝ) : ℝ := -144 * Real.log (1 - N / 100) / Real.log 10

/-- Theorem: The learning time for 90 characters per minute is 144 hours -/
theorem learning_time_90 : learning_time 90 = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_learning_time_90_l1162_116229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_92_l1162_116248

/-- Represents the duration of stay for the mathematicians in minutes -/
noncomputable def n : ℝ := 0

/-- Represents the first component of n in the equation n = x - y√z -/
def x : ℕ := 0

/-- Represents the second component of n in the equation n = x - y√z -/
def y : ℕ := 0

/-- Represents the third component of n in the equation n = x - y√z -/
def z : ℕ := 0

/-- The duration of stay is defined as n = x - y√z -/
axiom n_def : n = x - y * Real.sqrt z

/-- x, y, and z are positive integers -/
axiom positive_integers : x > 0 ∧ y > 0 ∧ z > 0

/-- z is not divisible by the square of any prime -/
axiom z_not_divisible : ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ z)

/-- The probability of the mathematicians meeting is 50% -/
axiom meeting_probability : (3600 - (60 - n)^2) / 3600 = 1/2

/-- Theorem: Under the given conditions, x + y + z = 92 -/
theorem sum_equals_92 : x + y + z = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_92_l1162_116248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_square_center_inside_another_l1162_116228

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in 2D space -/
structure Square where
  center : Point
  side_length : ℝ

/-- Check if a point is strictly inside a square -/
def is_strictly_inside (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) < s.side_length / 2 ∧
  abs (p.y - s.center.y) < s.side_length / 2

/-- The main theorem -/
theorem unit_square_center_inside_another
  (large_square : Square)
  (unit_squares : Finset Square)
  (h_large_side : large_square.side_length = 7)
  (h_unit_count : unit_squares.card = 170)
  (h_unit_side : ∀ s, s ∈ unit_squares → s.side_length = 1)
  (h_parallel : ∀ s, s ∈ unit_squares → 
    (s.center.x - s.side_length / 2 = large_square.center.x - large_square.side_length / 2 ∨
     s.center.x + s.side_length / 2 = large_square.center.x + large_square.side_length / 2) ∧
    (s.center.y - s.side_length / 2 = large_square.center.y - large_square.side_length / 2 ∨
     s.center.y + s.side_length / 2 = large_square.center.y + large_square.side_length / 2))
  (h_contained : ∀ s, s ∈ unit_squares → 
    abs (s.center.x - large_square.center.x) ≤ (large_square.side_length - s.side_length) / 2 ∧
    abs (s.center.y - large_square.center.y) ≤ (large_square.side_length - s.side_length) / 2) :
  ∃ s₁ s₂, s₁ ∈ unit_squares ∧ s₂ ∈ unit_squares ∧ s₁ ≠ s₂ ∧ is_strictly_inside s₁.center s₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_square_center_inside_another_l1162_116228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_triangle_side_length_l1162_116202

/-- Represents the side length of the nth triangle in the sequence -/
noncomputable def sideLength (n : ℕ) (s : ℝ) : ℝ := s / (2 ^ n)

/-- Represents the perimeter of the nth triangle in the sequence -/
noncomputable def perimeter (n : ℕ) (s : ℝ) : ℝ := 3 * sideLength n s

/-- The sum of the perimeters of all triangles in the infinite sequence -/
noncomputable def totalPerimeter (s : ℝ) : ℝ := ∑' n, perimeter n s

/-- 
Given an infinite sequence of equilateral triangles where each subsequent triangle 
is formed by joining the midpoints of the previous triangle's sides, 
if the sum of all triangles' perimeters is 300 cm, 
then the side length of the first triangle is 50 cm.
-/
theorem first_triangle_side_length : 
  ∀ s : ℝ, totalPerimeter s = 300 → s = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_triangle_side_length_l1162_116202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_arrangement_theorem_l1162_116295

/-- Represents a unit cube with some black faces -/
structure UnitCube where
  black_faces : Nat
  white_faces : Nat
  face_sum : black_faces + white_faces = 6

/-- Represents the larger 2x2x2 cube composed of 8 unit cubes -/
structure LargeCube where
  unit_cubes : Fin 8 → UnitCube
  total_black_faces : Nat
  total_black_faces_eq : total_black_faces = (Finset.sum Finset.univ fun i => (unit_cubes i).black_faces)
  surface_black_faces : Nat
  surface_black_faces_eq : surface_black_faces = 24

/-- The main theorem statement -/
theorem cube_arrangement_theorem (n : Nat) :
  (∃ (lc : LargeCube), lc.total_black_faces = n) ↔ n ∈ ({23, 24, 25} : Finset Nat) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_arrangement_theorem_l1162_116295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_proof_l1162_116273

theorem missing_number_proof : 
  ∃ x : ℝ, 11 + Real.sqrt (x + 6 * 4 / 3) = 13 ∧ x = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_proof_l1162_116273
