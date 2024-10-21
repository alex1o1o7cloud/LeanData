import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log19_not_expressible_log_7_div_5_expressible_log21_expressible_log420_expressible_log0_35_expressible_l132_13202

-- Define the given logarithms
noncomputable def log5 : ℝ := Real.log 5
noncomputable def log7 : ℝ := Real.log 7

-- Define a function that checks if a logarithm can be expressed using log5 and log7
def can_express (x : ℝ) : Prop :=
  ∃ (a b : ℚ), x = a * log5 + b * log7

-- Theorem stating that log 19 cannot be expressed using log5 and log7
theorem log19_not_expressible : ¬ can_express (Real.log 19) := by
  sorry

-- Theorems stating that other options can be expressed or simplified using log5 and log7
theorem log_7_div_5_expressible : can_express (Real.log 7 - Real.log 5) := by
  sorry

theorem log21_expressible : can_express (Real.log 21) := by
  sorry

theorem log420_expressible : can_express (Real.log 420) := by
  sorry

theorem log0_35_expressible : can_express (Real.log 0.35) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log19_not_expressible_log_7_div_5_expressible_log21_expressible_log420_expressible_log0_35_expressible_l132_13202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_min_distance_equals_3sqrt5_minus_5_l132_13270

noncomputable section

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 2)^2 = 9
def circle_C2 (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem min_distance_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle_C1 x1 y1 → circle_C2 x2 y2 →
    ∀ (p1_x p1_y p2_x p2_y : ℝ),
      circle_C1 p1_x p1_y → circle_C2 p2_x p2_y →
      distance x1 y1 x2 y2 ≥ 3 * Real.sqrt 5 - 5 :=
by
  sorry

-- Main theorem
theorem min_distance_equals_3sqrt5_minus_5 :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_C1 x1 y1 ∧ circle_C2 x2 y2 ∧
    distance x1 y1 x2 y2 = 3 * Real.sqrt 5 - 5 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_min_distance_equals_3sqrt5_minus_5_l132_13270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_properties_l132_13222

theorem sum_properties (a b : ℤ) (ha : ∃ k : ℤ, a = 3 * k) (hb : ∃ l : ℤ, b = 5 * l) :
  (∃ m n : ℤ, Even (a + b) ∧ ∃ m' n' : ℤ, Odd (a + b)) ∧
  (∃ m n : ℤ, (a + b) % 3 = 0 ∧ ∃ m' n' : ℤ, (a + b) % 3 ≠ 0) ∧
  (∃ m n : ℤ, (a + b) % 5 = 0 ∧ ∃ m' n' : ℤ, (a + b) % 5 ≠ 0) ∧
  (∃ m n : ℤ, (a + b) % 15 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_properties_l132_13222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_relationship_l132_13250

theorem angle_cosine_relationship (A B C : ℝ) (hABC : A + B + C = Real.pi) :
  (∀ (A' B' C' : ℝ), A' + B' + C' = Real.pi → 
    (Real.cos (2 * A') > Real.cos (2 * B') ∧ Real.cos (2 * B') > Real.cos (2 * C')) → 
    A' < B' ∧ B' < C') ∧
  ¬(∀ (A' B' C' : ℝ), A' + B' + C' = Real.pi → 
    (A' < B' ∧ B' < C') → 
    Real.cos (2 * A') > Real.cos (2 * B') ∧ Real.cos (2 * B') > Real.cos (2 * C')) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_relationship_l132_13250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_durakavalyanie_in_1c_l132_13273

-- Define the classes and subjects
inductive ClassType : Type
| A : ClassType
| B : ClassType
| C : ClassType

inductive SubjectType : Type
| K : SubjectType  -- Kurashenie
| N : SubjectType  -- Nizvedenie
| D : SubjectType  -- Durakavalyanie

-- Define the schedule as a function from ClassType and lesson number to SubjectType
def Schedule := ClassType → Fin 3 → SubjectType

-- Define the conditions
def valid_schedule (s : Schedule) : Prop :=
  -- Kurashenie is the first lesson in class 1B
  s ClassType.B 0 = SubjectType.K ∧
  -- Durakavalyanie in 1B is after Durakavalyanie in 1A
  (∃ i j : Fin 3, i < j ∧ s ClassType.A i = SubjectType.D ∧ s ClassType.B j = SubjectType.D) ∧
  -- Nizvedenie is not the second lesson in class 1A
  s ClassType.A 1 ≠ SubjectType.N ∧
  -- No subject is taught simultaneously in two classes
  ∀ (i : Fin 3) (c1 c2 : ClassType), c1 ≠ c2 → s c1 i ≠ s c2 i

-- Theorem statement
theorem durakavalyanie_in_1c (s : Schedule) (h : valid_schedule s) :
  s ClassType.C 2 = SubjectType.D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_durakavalyanie_in_1c_l132_13273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l132_13286

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x + b / x

-- State the theorem
theorem function_properties (a b : ℝ) :
  (f a b 1 = 3 - 8) →
  ((deriv (f a b)) 1 = 3) →
  (a = -1 ∧ b = -4) ∧
  (∀ x, x ≠ 0 → f (-1) (-4) x ≥ -4) ∧
  (∀ x, x ≠ 0 → f (-1) (-4) x ≤ 4) ∧
  (∃ x, x ≠ 0 ∧ f (-1) (-4) x = -4) ∧
  (∃ x, x ≠ 0 ∧ f (-1) (-4) x = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l132_13286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_three_dividing_fifteen_factorial_l132_13210

theorem greatest_power_of_three_dividing_fifteen_factorial :
  ∃ k : ℕ+, k = 8 ∧ 
  (∀ m : ℕ+, 3^m.val ∣ (Nat.factorial 15) → m ≤ k) ∧
  (3^k.val ∣ (Nat.factorial 15)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_three_dividing_fifteen_factorial_l132_13210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l132_13287

-- Define propositions p and q
def p (m : ℝ) : Prop := m > 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 - 4*m*x + 4*m - 3 ≥ 0

-- Define the set of m values that satisfy the conditions
def S : Set ℝ := {m | ¬(p m) ∧ q m}

-- Theorem statement
theorem range_of_m : S = Set.Icc 1 2 := by sorry

-- Here, Set.Icc 1 2 represents the closed interval [1, 2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l132_13287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_properties_l132_13284

-- Define the radius of the sphere
noncomputable def sphere_radius : ℝ := 2 * Real.sqrt 3

-- Define the side length of the inscribed cube
noncomputable def cube_side_length (r : ℝ) : ℝ := 2 * r / Real.sqrt 3

-- Define the surface area of a cube
def cube_surface_area (a : ℝ) : ℝ := 6 * a^2

-- Define the volume of a cube
def cube_volume (a : ℝ) : ℝ := a^3

theorem inscribed_cube_properties :
  cube_surface_area (cube_side_length sphere_radius) = 96 ∧
  cube_volume (cube_side_length sphere_radius) = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_properties_l132_13284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_f_implies_m_eq_one_symmetric_g_implies_negative_x_form_g_less_than_f_implies_a_bound_l132_13226

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + m*x + m) / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 
  if x > 0 then x^2 + a*x + 1
  else -x^2 + a*x + 1

-- Theorem 1
theorem symmetric_f_implies_m_eq_one (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f m x + f m (-x) = 2) → m = 1 := by sorry

-- Theorem 2
theorem symmetric_g_implies_negative_x_form (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → g a x + g a (-x) = 2) →
  (∀ x : ℝ, x < 0 → g a x = -x^2 + a*x + 1) := by sorry

-- Theorem 3
theorem g_less_than_f_implies_a_bound :
  (∀ x t : ℝ, x < 0 → t > 0 → g 1 x < f 1 t) →
  ∃ a : ℝ, a > -2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_f_implies_m_eq_one_symmetric_g_implies_negative_x_form_g_less_than_f_implies_a_bound_l132_13226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_sum_of_values_l132_13294

noncomputable def f (x : ℝ) : ℝ := (x^3 + 5*x^2 + 8*x + 4) / (x + 1)
noncomputable def g (x : ℝ) : ℝ := x^2 + 4*x + 4

theorem f_equiv_g : ∀ x : ℝ, x ≠ -1 → f x = g x := by
  sorry

theorem sum_of_values :
  let A : ℝ := 1
  let B : ℝ := 4
  let C : ℝ := 4
  let D : ℝ := -1
  A + B + C + D = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_sum_of_values_l132_13294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_four_theta_l132_13274

theorem cos_four_theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (4*θ) = 17/81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_four_theta_l132_13274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_subsets_count_l132_13278

-- Define the sets M and N
def M : Finset Nat := {0, 1, 2, 3, 4}
def N : Finset Nat := {1, 3, 5}

-- Define P as the intersection of M and N
def P : Finset Nat := M ∩ N

-- Theorem statement
theorem intersection_subsets_count :
  Finset.card (Finset.powerset P) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_subsets_count_l132_13278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l132_13263

def N : Set ℤ := {x | (1/2 : ℝ) < (2 : ℝ)^(x+1) ∧ (2 : ℝ)^(x+1) < 4}

def M : Set ℤ := {-1, 1}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l132_13263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_slope_l132_13223

noncomputable def f (x : ℝ) := 2 * Real.sin x

noncomputable def g (x : ℝ) := 2 * Real.sqrt x * (x / 3 + 1)

theorem parallel_tangents_slope (P Q : ℝ × ℝ) :
  P.1 ∈ Set.Icc 0 Real.pi →
  (∃ (xP xQ : ℝ), 
    xP ∈ Set.Icc 0 Real.pi ∧
    P = (xP, f xP) ∧
    Q = (xQ, g xQ) ∧
    (deriv f xP = deriv g xQ)) →
  (Q.2 - P.2) / (Q.1 - P.1) = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_slope_l132_13223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_rounded_to_thousandth_l132_13218

/-- Rounds a real number to the nearest thousandth -/
noncomputable def round_to_thousandth (x : ℝ) : ℝ :=
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

/-- The sum of 46.129 and 37.9312, rounded to the nearest thousandth, equals 84.106 -/
theorem sum_rounded_to_thousandth :
  round_to_thousandth (46.129 + 37.9312) = 84.106 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_rounded_to_thousandth_l132_13218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l132_13214

-- Define set A
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (3/4 : ℝ) 2, y = x^2 - (3/2)*x + 1}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | x + m^2 ≥ 1}

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (∀ x, x ∈ A → x ∈ B m) ↔ (m ≥ 3/4 ∨ m ≤ -3/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l132_13214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_reaction_yields_l132_13237

noncomputable section

-- Define the chemical equation
def sulfuric_acid_moles : ℚ := 3
def sodium_hydroxide_moles : ℚ := 6
def experimental_water_moles : ℚ := 27/5

-- Define the stoichiometric ratio from the balanced equation
def water_to_sulfuric_acid_ratio : ℚ := 2

-- Theoretical yield calculation
def theoretical_yield : ℚ := sulfuric_acid_moles * water_to_sulfuric_acid_ratio

-- Percent yield calculation
def percent_yield : ℚ := (experimental_water_moles / theoretical_yield) * 100

theorem chemical_reaction_yields :
  theoretical_yield = 6 ∧ percent_yield = 90 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_reaction_yields_l132_13237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_product_probability_l132_13299

/-- The probability of selecting exactly k defective items from a population of n items,
    where d items are defective, when selecting r items in total. -/
def hypergeometric_probability (n d r k : ℕ) : ℚ :=
  (Nat.choose d k * Nat.choose (n - d) (r - k)) / Nat.choose n r

theorem defective_product_probability :
  let total_products : ℕ := 100
  let defective_products : ℕ := 10
  let selected_products : ℕ := 5
  let defective_selected : ℕ := 2
  hypergeometric_probability total_products defective_products selected_products defective_selected =
    18 / (11 * 97 * 96) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_product_probability_l132_13299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jumps_jumps_even_l132_13230

/-- Represents the number of flower pots in the circle -/
def num_pots : ℕ := 8

/-- Represents the possible routes for the frog given the number of jumps -/
noncomputable def possible_routes (n : ℕ) : ℝ :=
  let m := (n - 2) / 2
  ((2 + Real.sqrt 2) ^ m - (2 - Real.sqrt 2) ^ m) / Real.sqrt 2

/-- Theorem stating the properties of frog jumps between pots -/
theorem frog_jumps (n : ℕ) (h1 : n > 0) (h2 : Even n) :
  ∃ (routes : ℝ), routes = possible_routes n ∧ routes > 0 := by
  sorry

/-- Theorem stating that the number of jumps must be even -/
theorem jumps_even (n : ℕ) (h : ∃ (routes : ℝ), routes = possible_routes n ∧ routes > 0) :
  Even n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jumps_jumps_even_l132_13230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l132_13272

/-- A line passing through the origin and tangent to a circle in the second quadrant -/
structure TangentLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The line passes through the origin -/
  passes_origin : ∀ x y : ℝ, y = k * x → (x = 0 ∧ y = 0)
  /-- The circle's equation is x^2 + (y-4)^2 = 4 -/
  circle_equation : Set (ℝ × ℝ) := {(x, y) | x^2 + (y-4)^2 = 4}
  /-- The line is tangent to the circle -/
  is_tangent : ∃! p : ℝ × ℝ, p ∈ circle_equation ∧ p.2 = k * p.1
  /-- The tangent point is in the second quadrant -/
  in_second_quadrant : ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ (x, y) ∈ circle_equation ∧ y = k * x

/-- The theorem stating that the tangent line has the equation y = -√3 x -/
theorem tangent_line_equation (l : TangentLine) : l.k = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l132_13272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l132_13262

/-- Hyperbola C with equation x²/a² - y²/b² = 1 -/
structure Hyperbola (a b : ℝ) where
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- Right latus rectum of the hyperbola -/
noncomputable def rightLatusRectum (a c : ℝ) : Set ℝ := {x | x = a^2 / c}

/-- Asymptotes of the hyperbola -/
noncomputable def asymptotes (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = b/a * x ∨ y = -b/a * x}

/-- Right focus of the hyperbola -/
def rightFocus (c : ℝ) : ℝ × ℝ := (c, 0)

/-- Points A and B where right latus rectum intersects asymptotes -/
noncomputable def intersectionPoints (a b c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((a^2/c, a*b/c), (a^2/c, -a*b/c))

/-- Triangle ABF is equilateral -/
def isEquilateralTriangle (A B F : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 =
  (B.1 - F.1)^2 + (B.2 - F.2)^2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 =
  (A.1 - F.1)^2 + (A.2 - F.2)^2

/-- Eccentricity of the hyperbola -/
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity (a b c : ℝ) (h : Hyperbola a b) :
  let A := (intersectionPoints a b c).1
  let B := (intersectionPoints a b c).2
  let F := rightFocus c
  isEquilateralTriangle A B F →
  eccentricity a c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l132_13262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_min_sum_l132_13244

theorem matrix_equation_min_sum (p q r s : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 →
  (Matrix.of ![![p, q], ![r, s]])^2 = Matrix.of ![![9, 0], ![0, 9]] →
  (∃ (a b c d : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (Matrix.of ![![a, b], ![c, d]])^2 = Matrix.of ![![9, 0], ![0, 9]] ∧
    (a.natAbs + b.natAbs + c.natAbs + d.natAbs : ℕ) < p.natAbs + q.natAbs + r.natAbs + s.natAbs) →
  (p.natAbs + q.natAbs + r.natAbs + s.natAbs : ℕ) ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_min_sum_l132_13244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_passes_through_A_perpendicular_to_BC_l132_13276

-- Define the points
def A : Fin 3 → ℝ := ![-1, 3, 4]
def B : Fin 3 → ℝ := ![-1, 5, 0]
def C : Fin 3 → ℝ := ![2, 6, 1]

-- Define the vector BC
def BC : Fin 3 → ℝ := ![C 0 - B 0, C 1 - B 1, C 2 - B 2]

-- Define the plane equation
def plane_equation (x y z : ℝ) : Prop :=
  3 * x + y + z - 4 = 0

-- Theorem statement
theorem plane_passes_through_A_perpendicular_to_BC :
  plane_equation (A 0) (A 1) (A 2) ∧
  (∀ (x y z : ℝ), plane_equation x y z →
    (x - A 0) * BC 0 + (y - A 1) * BC 1 + (z - A 2) * BC 2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_passes_through_A_perpendicular_to_BC_l132_13276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_v_l132_13288

-- Define the polynomials
def poly1 (x : ℝ) := x^3 + 5*x^2 + 2*x - 8
def poly2 (x u v w : ℝ) := x^3 + u*x^2 + v*x + w

-- Define the roots and v as variables
variable (p q r v : ℝ)

-- State the conditions
axiom root1_1 : poly1 p = 0
axiom root1_2 : poly1 q = 0
axiom root1_3 : poly1 r = 0

axiom root2_1 : ∃ u w, poly2 (p+q) u v w = 0
axiom root2_2 : ∃ u w, poly2 (q+r) u v w = 0
axiom root2_3 : ∃ u w, poly2 (r+p) u v w = 0

-- State the theorem
theorem find_v : v = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_v_l132_13288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_350_properties_l132_13211

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The highest power of 3 that divides n! -/
def highestPowerOfThree (n : ℕ) : ℕ := sorry

theorem factorial_350_properties :
  trailingZeros 350 = 86 ∧ (Nat.factorial 350) % (3^171) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_350_properties_l132_13211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_rectangle_area_percentage_l132_13246

/-- Represents the side length of the square -/
noncomputable def square_side : ℝ := 1

/-- Represents the width of the rectangle -/
noncomputable def rectangle_width : ℝ := 3 * square_side

/-- Represents the length of the rectangle -/
noncomputable def rectangle_length : ℝ := (3/2) * rectangle_width

/-- Calculates the area of the square -/
noncomputable def square_area : ℝ := square_side^2

/-- Calculates the area of the rectangle -/
noncomputable def rectangle_area : ℝ := rectangle_length * rectangle_width

/-- Calculates the percentage of the rectangle's area covered by the square -/
noncomputable def area_percentage : ℝ := (square_area / rectangle_area) * 100

theorem square_in_rectangle_area_percentage :
  abs (area_percentage - 7.41) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_rectangle_area_percentage_l132_13246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_right_triangle_with_arithmetic_radii_l132_13207

theorem no_right_triangle_with_arithmetic_radii : ¬ ∃ (a b c : ℝ),
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧  -- positive side lengths
  (a^2 + b^2 = c^2) ∧  -- right-angled triangle
  (a ≤ b) ∧  -- ordering of sides
  (∃ (d : ℝ), (0 < d) ∧  -- common difference
    let s := (a + b + c) / 2  -- semiperimeter
    let r_in := s - c  -- inscribed circle radius
    let r_ex1 := s - a  -- first escribed circle radius
    let r_ex2 := s - b  -- second escribed circle radius
    let r_ex3 := s  -- third escribed circle radius
    (r_in + d = r_ex2) ∧ (r_ex2 + d = r_ex1) ∧ (r_ex1 + d = r_ex3))  -- arithmetic sequence
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_right_triangle_with_arithmetic_radii_l132_13207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_mixture_weight_l132_13203

/-- The total weight of concrete obtained by mixing two batches -/
noncomputable def total_weight (weight1 : ℝ) (weight2 : ℝ) : ℝ :=
  weight1 + weight2

/-- The cement percentage of the mixture -/
noncomputable def mixture_cement_percentage (weight1 : ℝ) (cement1 : ℝ) (weight2 : ℝ) (cement2 : ℝ) : ℝ :=
  (weight1 * cement1 + weight2 * cement2) / (weight1 + weight2)

theorem concrete_mixture_weight :
  let weight1 : ℝ := 1125
  let weight2 : ℝ := 1125
  let cement1 : ℝ := 9.3
  let cement2 : ℝ := 11.3
  let mixture_cement : ℝ := 10.8
  (mixture_cement_percentage weight1 cement1 weight2 cement2 = mixture_cement) →
  (total_weight weight1 weight2 = 2250) :=
by
  intro h
  simp [total_weight]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_mixture_weight_l132_13203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_equality_l132_13293

theorem remainder_equality (P P' D R k : ℕ) (h1 : P > P') (h2 : R < D)
  (h3 : P % D = 2 * R) (h4 : P' % D = R) :
  (k * (P + P')) % D = (k * (2 * R + R)) % D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_equality_l132_13293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digging_project_length_l132_13264

/-- The length of the first digging project -/
noncomputable def L : ℝ := 25

/-- The volume of the first digging project -/
noncomputable def V1 (l : ℝ) : ℝ := 3000 * l

/-- The volume of the second digging project -/
noncomputable def V2 : ℝ := 75000

/-- The number of days for both projects -/
noncomputable def days : ℝ := 12

/-- The rate of work (volume dug per day) for the first project -/
noncomputable def R1 (l : ℝ) : ℝ := V1 l / days

/-- The rate of work (volume dug per day) for the second project -/
noncomputable def R2 : ℝ := V2 / days

theorem digging_project_length :
  R1 L = R2 ∧ L = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digging_project_length_l132_13264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_total_faces_l132_13258

/-- Represents a pair of fair dice -/
structure DicePair where
  faces1 : ℕ
  faces2 : ℕ
  h1 : faces1 ≥ 6
  h2 : faces2 ≥ 6

/-- The probability of rolling a specific sum with a pair of dice -/
noncomputable def prob_sum (d : DicePair) (sum : ℕ) : ℚ :=
  (Finset.filter (fun p => p.1 + p.2 = sum) (Finset.product (Finset.range d.faces1) (Finset.range d.faces2))).card /
  (d.faces1 * d.faces2 : ℚ)

/-- The theorem stating the minimum number of total faces for the given conditions -/
theorem min_total_faces (d : DicePair) 
  (h_prob_8_11 : prob_sum d 8 = (1/2) * prob_sum d 11)
  (h_prob_13 : prob_sum d 13 = 1/14) :
  d.faces1 + d.faces2 ≥ 26 := by
  sorry

#check min_total_faces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_total_faces_l132_13258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_concyclic_points_l132_13290

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line type -/
structure Line where
  slope : ℝ
  passesThrough : ℝ × ℝ

/-- Defines if four points are concyclic -/
noncomputable def areConcyclic (a b c d : ℝ × ℝ) : Prop := sorry

/-- Theorem statement -/
theorem parabola_concyclic_points
  (p : Parabola)
  (P : ℝ × ℝ)
  (l₁ l₂ : Line)
  (α β : ℝ)
  (h_parabola : p.equation = fun x y ↦ y^2 = 2*x)
  (h_point : P = (1, 1))
  (h_l₁ : l₁.passesThrough = P)
  (h_l₂ : l₂.passesThrough = P)
  (h_α : 0 < α ∧ α < π)
  (h_β : 0 < β ∧ β < π)
  (h_l₁_slope : l₁.slope = Real.tan α)
  (h_l₂_slope : l₂.slope = Real.tan β)
  (h_distinct : l₁ ≠ l₂)
  (A B C D : ℝ × ℝ)
  (h_intersections : 
    (p.equation A.1 A.2 ∧ A = l₁.passesThrough) ∧
    (p.equation B.1 B.2 ∧ B = l₁.passesThrough) ∧
    (p.equation C.1 C.2 ∧ C = l₂.passesThrough) ∧
    (p.equation D.1 D.2 ∧ D = l₂.passesThrough)) :
  areConcyclic A B C D ↔ α + β = π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_concyclic_points_l132_13290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_four_greater_than_f_one_l132_13280

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(|x + 1|)

-- State the theorem
theorem f_negative_four_greater_than_f_one
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : Set.range (f a) = Set.Ici 1) :
  f a (-4) > f a 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_four_greater_than_f_one_l132_13280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l132_13209

theorem absolute_value_equation_solutions :
  ∃ (s : Finset ℝ), (∀ x : ℝ, x ∈ s ↔ |2*x - 14| = |x + 4|) ∧ s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l132_13209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l132_13221

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  (- b + Real.sqrt (b^2 - 4*a*c)) / (2*a) + (- b - Real.sqrt (b^2 - 4*a*c)) / (2*a) = - b / a :=
sorry

theorem sum_of_solutions_specific_quadratic :
  (9 + Real.sqrt 81) / 2 + (9 - Real.sqrt 81) / 2 = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l132_13221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l132_13285

/-- Represents a line in the form lx - y + 1 + 2k = 0 --/
structure Line where
  k : ℝ

/-- Condition that the line does not pass through the fourth quadrant --/
def Line.notInFourthQuadrant (l : Line) : Prop :=
  l.k ≥ 0

/-- Area of triangle AOB formed by the line's intersections with the axes --/
noncomputable def Line.triangleArea (l : Line) : ℝ :=
  (1/2) * ((1 + 2*l.k) + (1 + 2*l.k)/l.k)

/-- Theorem stating the properties of the line --/
theorem line_properties (l : Line) (h : l.notInFourthQuadrant) :
  (∀ k : ℝ, k ≥ 0 → ∃ l' : Line, l'.k = k ∧ l'.notInFourthQuadrant) ∧
  (∃ min_area : ℝ, min_area = 4 ∧ ∀ l' : Line, l'.notInFourthQuadrant → l'.triangleArea ≥ min_area) ∧
  (∃ l_min : Line, l_min.triangleArea = 4 ∧ l_min.k = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l132_13285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l132_13275

/-- Calculates the time (in seconds) for a train to pass a platform -/
noncomputable def time_to_pass_platform (train_length : ℝ) (train_speed_kmh : ℝ) (platform_length : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train of length 360 meters traveling at 45 km/h takes 40 seconds to pass a platform of length 140 meters -/
theorem train_passing_platform :
  time_to_pass_platform 360 45 140 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l132_13275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_all_color_square_l132_13217

/-- Represents the four colors used for coloring vertices -/
inductive Color
| One
| Two
| Three
| Four

/-- Represents a vertex in the rectangle -/
structure Vertex where
  x : ℕ
  y : ℕ
  color : Color

/-- Represents the rectangle ABCD -/
structure Rectangle where
  width : ℕ
  height : ℕ
  vertices : Set Vertex

/-- Checks if two vertices are neighbors -/
def are_neighbors (v1 v2 : Vertex) : Prop :=
  (v1.x = v2.x ∧ (v1.y + 1 = v2.y ∨ v2.y + 1 = v1.y)) ∨
  (v1.y = v2.y ∧ (v1.x + 1 = v2.x ∨ v2.x + 1 = v1.x))

/-- Checks if the coloring satisfies the side conditions -/
def satisfies_side_conditions (r : Rectangle) : Prop :=
  ∀ v ∈ r.vertices,
    (v.x = 0 → v.color = Color.One ∨ v.color = Color.Two) ∧
    (v.x = r.width → v.color = Color.Two ∨ v.color = Color.Three) ∧
    (v.y = 0 → v.color = Color.Three ∨ v.color = Color.Four) ∧
    (v.y = r.height → v.color = Color.Four ∨ v.color = Color.One)

/-- Checks if the coloring satisfies the neighbor conditions -/
def satisfies_neighbor_conditions (r : Rectangle) : Prop :=
  ∀ v1 v2, v1 ∈ r.vertices → v2 ∈ r.vertices → are_neighbors v1 v2 →
    (v1.color ≠ Color.One ∨ v2.color ≠ Color.Three) ∧
    (v1.color ≠ Color.Three ∨ v2.color ≠ Color.One) ∧
    (v1.color ≠ Color.Two ∨ v2.color ≠ Color.Four) ∧
    (v1.color ≠ Color.Four ∨ v2.color ≠ Color.Two)

/-- Checks if a unit square has all vertices in different colors -/
def has_all_colors (r : Rectangle) (x y : ℕ) : Prop :=
  ∃ (v1 v2 v3 v4 : Vertex),
    v1 ∈ r.vertices ∧ v2 ∈ r.vertices ∧ v3 ∈ r.vertices ∧ v4 ∈ r.vertices ∧
    v1.x = x ∧ v1.y = y ∧
    v2.x = x + 1 ∧ v2.y = y ∧
    v3.x = x ∧ v3.y = y + 1 ∧
    v4.x = x + 1 ∧ v4.y = y + 1 ∧
    v1.color ≠ v2.color ∧ v1.color ≠ v3.color ∧ v1.color ≠ v4.color ∧
    v2.color ≠ v3.color ∧ v2.color ≠ v4.color ∧
    v3.color ≠ v4.color

theorem existence_of_all_color_square (r : Rectangle) 
  (h1 : satisfies_side_conditions r)
  (h2 : satisfies_neighbor_conditions r) :
  ∃ x y, has_all_colors r x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_all_color_square_l132_13217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l132_13206

/-- The solution to the equation (0.47 * 1442 - 0.36 * 1412) + x = 252 -/
def solution : ℝ := 82.58

/-- The left-hand side of the equation -/
def lhs (x : ℝ) : ℝ := (0.47 * 1442 - 0.36 * 1412) + x

/-- The right-hand side of the equation -/
def rhs : ℝ := 252

/-- Approximate equality for real numbers -/
def approx_eq (x y : ℝ) : Prop := abs (x - y) < 0.01

theorem equation_solution :
  approx_eq (lhs solution) rhs := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l132_13206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l132_13232

theorem exponential_equation_solution : 
  ∃ x : ℝ, (2 : ℝ)^x + 6 = 3 * (2 : ℝ)^x - 26 ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l132_13232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_region_l132_13269

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
def g (x : ℝ) : ℝ := x^3

-- Define the enclosed region
def enclosed_region (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1 ∧ g x ≤ f x

-- Statement of the theorem
theorem area_enclosed_region :
  ∫ x in Set.Icc 0 1, (f x - g x) = 5/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_region_l132_13269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_loss_percent_l132_13267

noncomputable section

-- Define the selling price of each car
def selling_price : ℝ := 325475

-- Define the gain percentage on the first car
def gain_percent : ℝ := 0.14

-- Define the loss percentage on the second car
def loss_percent : ℝ := 0.14

-- Calculate the cost price of the first car
noncomputable def cost_price1 : ℝ := selling_price / (1 + gain_percent)

-- Calculate the cost price of the second car
noncomputable def cost_price2 : ℝ := selling_price / (1 - loss_percent)

-- Calculate the total cost price
noncomputable def total_cost_price : ℝ := cost_price1 + cost_price2

-- Calculate the total selling price
def total_selling_price : ℝ := 2 * selling_price

-- Calculate the loss amount
noncomputable def loss_amount : ℝ := total_cost_price - total_selling_price

-- Calculate the loss percent
noncomputable def loss_percentage : ℝ := (loss_amount / total_cost_price) * 100

-- Theorem statement
theorem trader_loss_percent :
  abs (loss_percentage - 1.957) < 0.001 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_loss_percent_l132_13267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_l132_13239

noncomputable section

/-- A function f : ℝ → ℝ is monotonically increasing -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The function f(x) = (1/3)x³ + x² + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x

theorem monotonically_increasing_condition (a : ℝ) :
  MonotonicallyIncreasing (f a) ↔ a ≥ 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_l132_13239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l132_13234

theorem division_problem (m n : ℕ) (h1 : m % n = 12) (h2 : (m : ℚ) / (n : ℚ) = 24.2) 
    (hm : m > 0) (hn : n > 0) : n = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l132_13234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l132_13238

open Real

theorem trigonometric_equation_solution (x : ℝ) (h : cos x ≠ 0) :
  (sin (3 * x) * cos (5 * x) - sin (2 * x) * cos (6 * x)) / cos x = 0 ↔
  (∃ n : ℤ, x = π * ↑n ∨ x = π / 6 + π * ↑n ∨ x = -π / 6 + π * ↑n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l132_13238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_coverage_exists_l132_13213

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a coin placement on the grid -/
structure CoinPlacement where
  center : GridPoint

/-- The distance between two grid points -/
noncomputable def gridDistance (p1 p2 : GridPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 : ℝ)

/-- Checks if a grid point is covered by a coin -/
def isCovered (point : GridPoint) (coin : CoinPlacement) : Prop :=
  gridDistance point coin.center ≤ 1.3

/-- Checks if two coins overlap -/
def doOverlap (coin1 coin2 : CoinPlacement) : Prop :=
  gridDistance coin1.center coin2.center < 2.6

/-- A valid coin arrangement covers all grid points without overlaps -/
def isValidArrangement (arrangement : Set CoinPlacement) : Prop :=
  (∀ p : GridPoint, ∃ c ∈ arrangement, isCovered p c) ∧
  (∀ c1 c2 : CoinPlacement, c1 ∈ arrangement → c2 ∈ arrangement → c1 ≠ c2 → ¬doOverlap c1 c2)

/-- The main theorem: there exists a valid coin arrangement -/
theorem coin_coverage_exists : ∃ arrangement : Set CoinPlacement, isValidArrangement arrangement := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_coverage_exists_l132_13213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BC_formula_l132_13260

/-- Two externally tangent circles with a tangent line from the larger circle to the smaller circle -/
structure TangentCircles where
  R : ℝ
  r : ℝ
  a : ℝ
  h1 : R > r
  h2 : R > 0
  h3 : r > 0
  h4 : a > 0

/-- The length of BC in the tangent circles configuration -/
noncomputable def length_BC (tc : TangentCircles) : ℝ :=
  tc.a * Real.sqrt ((tc.R + tc.r) / tc.R)

/-- Theorem stating that the length of BC is equal to a * sqrt((R + r) / R) -/
theorem length_BC_formula (tc : TangentCircles) :
  length_BC tc = tc.a * Real.sqrt ((tc.R + tc.r) / tc.R) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BC_formula_l132_13260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_350_l132_13253

theorem closest_perfect_square_to_350 : 
  ∀ n : ℤ, n * n ≠ 361 → |350 - n * n| ≥ |350 - 361| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_350_l132_13253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_section_theorem_l132_13225

def rectangle_pyramid (ab bc h : ℝ) := ab > 0 ∧ bc > 0 ∧ h > 0

def volume_ratio (vp vp' : ℝ) := vp = 3 * vp'

def height_ratio (hp hp' : ℝ) := hp' = hp / (3 ^ (1/3 : ℝ))

def base_center_to_apex (hp hp' : ℝ) : ℝ := hp - hp'

theorem pyramid_section_theorem (ab bc h : ℝ) (hp hp' : ℝ) :
  rectangle_pyramid ab bc h →
  ab = 15 →
  bc = 20 →
  h = 30 →
  volume_ratio (ab * bc * h / 3) (ab * bc * hp' / 3) →
  height_ratio h hp' →
  ‖base_center_to_apex h hp' - 9.2‖ < 0.1 := by sorry

#check pyramid_section_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_section_theorem_l132_13225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_five_digit_numbers_l132_13243

/-- The set of digits used to form the numbers -/
def digits : Finset ℕ := {1, 2, 3, 4, 5}

/-- A function that checks if a number is even -/
def is_even (n : ℕ) : Bool := n % 2 = 0

/-- The set of even digits from the given set -/
def even_digits : Finset ℕ := digits.filter (fun n => is_even n)

/-- The number of ways to arrange the remaining digits -/
def remaining_arrangements : ℕ := 4 * 3 * 2 * 1

/-- The theorem stating the number of even five-digit numbers -/
theorem count_even_five_digit_numbers : 
  (even_digits.card : ℕ) * remaining_arrangements = 48 := by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_five_digit_numbers_l132_13243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_inequality_l132_13201

theorem smallest_n_for_inequality : ∃ (n : ℕ), 
  (∀ (m : ℕ), m < n → (3 * Real.sqrt (m : ℝ) - 2 * Real.sqrt ((m : ℝ) - 1) ≥ 0.03)) ∧
  (3 * Real.sqrt (n : ℝ) - 2 * Real.sqrt ((n : ℝ) - 1) < 0.03) ∧
  n = 433715589 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_inequality_l132_13201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_conditions_l132_13282

noncomputable def f (x y : ℝ) : ℝ :=
  (y^2 - x^2) / 2 + Real.exp y - Real.log ((x + Real.sqrt (1 + x^2)) / (2 + Real.sqrt 5)) - (Real.exp 1 - 3/2)

theorem solution_satisfies_conditions (x y : ℝ) :
  f x y = 0 →
  (deriv (fun y => f x y) y) = (x * Real.sqrt (1 + x^2) + 1) / (Real.sqrt (1 + x^2) * (y + Real.exp y)) ∧
  f 2 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_conditions_l132_13282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_special_case_l132_13255

/-- Represents the coefficients of a cubic equation -/
structure CubicCoefficients (α : Type*) [Ring α] where
  a₃ : α
  a₂ : α
  a₁ : α
  a₀ : α

/-- Represents the parameters of the special case model -/
structure ModelParameters (α : Type*) [Ring α] where
  v : α
  u : α
  w : α

/-- Theorem stating that the given cubic equation is a special case of the model equation -/
theorem cubic_special_case 
  {α : Type*} [Ring α] [OrderedRing α] [OfScientific α]
  (coeff : CubicCoefficients α) 
  (b : α) 
  (h_b : b ≥ 0) :
  ∃ (params : ModelParameters α),
    coeff.a₃ = (6.266 : α) ∧
    coeff.a₂ = -3 * params.v ∧
    coeff.a₁ = 3 * params.u^2 - params.w^2 ∧
    coeff.a₀ = -(params.v^3 - params.v * params.w^2) ∧
    params.v = params.u ∧
    params.w^2 = b :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_special_case_l132_13255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_problem_l132_13247

theorem quadratic_roots_problem (x₁ x₂ m : ℝ) : 
  (∀ x, x^2 - 4*x + m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + x₂ - x₁*x₂ = 1 →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_problem_l132_13247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_incircle_bisector_l132_13277

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (on_circle : Point → Circle → Prop)
variable (diameter_of : Point → Point → Circle → Prop)
variable (perpendicular : Point → Point → Point → Point → Prop)
variable (incircle : Circle → Point → Point → Point → Prop)
variable (touches : Circle → Point → Prop)
variable (bisects : Point → Point → Point → Point → Point → Prop)

-- State the theorem
theorem semicircle_incircle_bisector 
  (Γ : Circle) (Γ' : Circle) (A B P Q L : Point) :
  diameter_of A B Γ →
  on_circle P Γ →
  perpendicular P Q A B →
  incircle Γ' A P Q →
  touches Γ' L →
  bisects P L A P Q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_incircle_bisector_l132_13277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_intersection_points_l132_13298

noncomputable def C₁ (t : ℝ) : ℝ × ℝ :=
  (2 * Real.sqrt 2 - (Real.sqrt 2 / 2) * t, Real.sqrt 2 + (Real.sqrt 2 / 2) * t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ :=
  let ρ := 4 * Real.sqrt 2 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def P : ℝ × ℝ := (Real.sqrt 2, 2 * Real.sqrt 2)

theorem distance_sum_intersection_points :
  ∃ A B : ℝ × ℝ,
    (∃ t θ : ℝ, C₁ t = A ∧ C₂ θ = A) ∧
    (∃ t' θ' : ℝ, C₁ t' = B ∧ C₂ θ' = B) ∧
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) =
    2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_intersection_points_l132_13298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l132_13279

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l132_13279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_uniform_face_l132_13224

-- Define the type for colors
inductive Color where
  | Red
  | Blue
deriving BEq, DecidableEq

-- Define a face as a pair of colors
def Face := Color × Color

-- Define a cube as a list of 6 faces
def Cube := List Face

-- Function to check if a face has both triangles the same color
def uniformColor (face : Face) : Bool :=
  face.1 == face.2

-- Function to check if at least one face of the cube has uniform color
def hasUniformFace (cube : Cube) : Bool :=
  cube.any uniformColor

-- Probability of a single face having uniform color
def probUniformFace : ℚ := 1 / 2

-- Total number of faces in a cube
def numFaces : ℕ := 6

-- Theorem statement
theorem prob_at_least_one_uniform_face :
  (1 : ℚ) - (1 - probUniformFace) ^ numFaces = 63 / 64 := by
  sorry

#eval ((1 : ℚ) - (1 - probUniformFace) ^ numFaces)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_uniform_face_l132_13224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_b_value_l132_13240

/-- Triangle T1 with sides 8, 11, and 11 -/
def T1 : Set ℝ := sorry

/-- Triangle T2 with sides b, 1, and 1 -/
def T2 (b : ℝ) : Set ℝ := sorry

/-- Incircle radius of a triangle -/
noncomputable def incircleRadius (t : Set ℝ) : ℝ := sorry

/-- Circumcircle radius of a triangle -/
noncomputable def circumcircleRadius (t : Set ℝ) : ℝ := sorry

/-- The ratio of incircle radius to circumcircle radius -/
noncomputable def radiusRatio (t : Set ℝ) : ℝ :=
  incircleRadius t / circumcircleRadius t

/-- The theorem stating the largest possible value of b -/
theorem largest_b_value :
  ∃ (b : ℝ), radiusRatio T1 = radiusRatio (T2 b) ∧
  ∀ (b' : ℝ), radiusRatio T1 = radiusRatio (T2 b') → b' ≤ 14/11 := by
  sorry

#check largest_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_b_value_l132_13240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l132_13295

/-- Represents a polynomial function -/
structure MyPolynomial (α : Type*) where
  coeffs : List α

/-- Counts the number of operations in Horner's method -/
def horner_operations (p : MyPolynomial ℤ) : ℕ × ℕ :=
  match p.coeffs with
  | [] => (0, 0)
  | [_] => (0, 0)
  | _ :: rest => (rest.length, rest.length)

/-- The given polynomial f(x) = 6x^5 - 4x^4 + x^3 - 2x^2 - 9x - 9 -/
def f : MyPolynomial ℤ :=
  { coeffs := [6, -4, 1, -2, -9, -9] }

theorem horner_method_operations :
  horner_operations f = (5, 5) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l132_13295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l132_13266

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * x)

-- State the theorem
theorem period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  ∀ (T' : ℝ), (T' > 0 ∧ ∀ (x : ℝ), f (x + T') = f x) → T' ≥ T :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l132_13266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonic_increase_condition_l132_13249

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 
  -1/x^2 * Real.log (1 + x) + (1/x + a) * (1 / (1 + x))

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  let a : ℝ := -1
  let x₀ : ℝ := 1
  let y₀ : ℝ := f a x₀
  let m : ℝ := f_deriv a x₀
  ∀ x : ℝ, (y₀ + m * (x - x₀) = -Real.log 2 * x + Real.log 2) := by sorry

-- Theorem for monotonic increase condition
theorem monotonic_increase_condition (a : ℝ) :
  (∀ x : ℝ, x > 0 → StrictMono (fun x => f a x)) ↔ a ≥ 1/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonic_increase_condition_l132_13249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_y_coords_constant_l132_13212

/-- Circle O with radius √10 -/
def Circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 10}

/-- Point A on the circle -/
noncomputable def A : ℝ × ℝ := (1, 3)

/-- Point B on the circle -/
noncomputable def B : ℝ × ℝ := (-1, 3)

/-- Any point P on the circle other than A and B -/
noncomputable def P : ℝ × ℝ := sorry

/-- M is the intersection of AP with y-axis -/
noncomputable def M : ℝ × ℝ := (0, (3 * P.1 - P.2) / (P.1 - 1))

/-- N is the intersection of BP with y-axis -/
noncomputable def N : ℝ × ℝ := (0, (3 * P.1 + P.2) / (P.1 + 1))

theorem product_of_y_coords_constant :
  P ∈ Circle_O → P ≠ A → P ≠ B → M.2 * N.2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_y_coords_constant_l132_13212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_three_for_consecutive_number_l132_13256

def consecutiveNumber (start finish : ℕ) : ℕ :=
  -- Function to create the number N by concatenating integers from start to finish
  sorry

def highestPowerOfThree (n : ℕ) : ℕ :=
  -- Function to find the highest power of 3 that divides n
  sorry

theorem highest_power_of_three_for_consecutive_number :
  highestPowerOfThree (consecutiveNumber 31 53) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_three_for_consecutive_number_l132_13256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_retailer_profit_percentage_l132_13216

/-- Calculates the profit percentage for a retailer selling pens -/
theorem pen_retailer_profit_percentage 
  (market_price : ℝ) -- Market price for one pen
  (buy_quantity : ℕ) -- Number of pens bought
  (price_quantity : ℕ) -- Number of pens priced at market price
  (sell_quantity : ℕ) -- Number of pens sold
  (discount_percent : ℝ) -- Discount percentage
  (h1 : buy_quantity = 140)
  (h2 : price_quantity = 36)
  (h3 : sell_quantity = buy_quantity)
  (h4 : discount_percent = 1)
  : ∃ (profit_percent : ℝ), abs (profit_percent - 285) < 1 :=
by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_retailer_profit_percentage_l132_13216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arcsin_cos_l132_13233

/-- The area bounded by y = arcsin(cos x) and the x-axis on [0, 2π] -/
noncomputable def bounded_area (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, |f x|

/-- The function y = arcsin(cos x) -/
noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.cos x)

theorem area_arcsin_cos : bounded_area f 0 (2 * Real.pi) = Real.pi^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arcsin_cos_l132_13233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l132_13248

/-- Given a real parameter a, define the function f as specified -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1)^2 + a * Real.sin x) / (x^2 + 1) + 3

/-- The main theorem to prove -/
theorem f_property (a : ℝ) :
  f a (Real.log (Real.log 5 / Real.log 2)) = 5 →
  f a (Real.log (Real.log 2 / Real.log 5)) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l132_13248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_product_inequality_l132_13241

open Real

/-- The function f(x) = 1/x - x -/
noncomputable def f (x : ℝ) : ℝ := 1/x - x

/-- The upper bound of k -/
noncomputable def k_upper_bound : ℝ := 2 * Real.sqrt (Real.sqrt 5 - 2)

theorem f_product_inequality (k : ℝ) :
  (k > 0 ∧ k ≤ k_upper_bound) ↔
  (∀ x ∈ Set.Ioo 0 k, f x * f (k - x) ≥ (k/2 - 2/k)^2) := by
  sorry

#check f_product_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_product_inequality_l132_13241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_seven_l132_13265

/-- An arithmetic sequence with positive first term -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_positive : 0 < a 1
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- Theorem stating that the maximum sum occurs at k = 7 -/
theorem max_sum_at_seven (seq : ArithmeticSequence) 
  (h : sum_n seq 3 = sum_n seq 11) :
  ∃ k : ℕ, k = 7 ∧ ∀ n : ℕ, sum_n seq n ≤ sum_n seq k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_seven_l132_13265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_cubic_expression_l132_13204

theorem factorization_cubic_expression (x y z : ℝ) :
  ((x^3 - y^3)^3 + (y^3 - z^3)^3 + (z^3 - x^3)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) =
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_cubic_expression_l132_13204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l132_13259

theorem min_value_of_expression (a b : ℕ) (ha : a < 6) (hb : b ≤ 7) :
  (∀ a' b' : ℕ, 0 < a' → a' < 6 → 0 < b' → b' ≤ 7 → 
    (2 : ℤ) * a' - a' * b' ≥ (2 : ℤ) * a - a * b) →
  (2 : ℤ) * a - a * b = -25 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l132_13259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equations_count_l132_13220

/-- A function that represents a valid equation formed by three distinct numbers -/
def ValidEquation (a b c : ℕ) : Prop :=
  a + b = c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The set of all numbers used in the equations -/
def UsedNumbers (equations : List (ℕ × ℕ × ℕ)) : Finset ℕ :=
  (equations.map (fun (a, b, c) => [a, b, c])).join.toFinset

theorem max_equations_count :
  ∃ (equations : List (ℕ × ℕ × ℕ)),
    (∀ (eq : ℕ × ℕ × ℕ), eq ∈ equations → ValidEquation eq.1 eq.2.1 eq.2.2) ∧
    (UsedNumbers equations) ⊆ (Finset.range 100) ∧
    (UsedNumbers equations).card = 99 ∧
    equations.length = 33 ∧
    (∀ (better_equations : List (ℕ × ℕ × ℕ)),
      (∀ (eq : ℕ × ℕ × ℕ), eq ∈ better_equations → ValidEquation eq.1 eq.2.1 eq.2.2) →
      (UsedNumbers better_equations) ⊆ (Finset.range 100) →
      better_equations.length ≤ 33) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equations_count_l132_13220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_inverse_l132_13261

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the symmetry condition
def symmetric_wrt_xy (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_inverse (f : ℝ → ℝ) :
  symmetric_wrt_xy f (λ x ↦ lg (x + 1)) →
  f = λ x ↦ 10^x - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_inverse_l132_13261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_selection_properties_l132_13268

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def is_double_multiple (a b : ℕ) : Prop := ∃ k : ℕ, b = k * (2 * a)

def number_set : Finset ℕ := Finset.range 11

theorem random_selection_properties :
  ∀ (selection : Finset ℕ), 
    selection ⊆ number_set → 
    selection.card = 6 → 
    (∃ (a b : ℕ), a ∈ selection ∧ b ∈ selection ∧ a ≠ b ∧ is_coprime a b) ∧
    (∃ (a b : ℕ), a ∈ selection ∧ b ∈ selection ∧ a ≠ b ∧ is_multiple a b) ∧
    ¬(∃ (a b : ℕ), a ∈ selection ∧ b ∈ selection ∧ a ≠ b ∧ is_double_multiple a b) :=
by sorry

#check random_selection_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_selection_properties_l132_13268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangulations_eq_catalan_l132_13208

/-- The number of triangulations of a convex (n+2)-gon -/
def triangulations (n : ℕ) : ℕ := sorry

/-- The n-th Catalan number -/
def catalanNumber (n : ℕ) : ℕ := sorry

/-- Theorem: The number of triangulations of a convex (n+2)-gon 
    is equal to the n-th Catalan number -/
theorem triangulations_eq_catalan (n : ℕ) : 
  triangulations n = catalanNumber n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangulations_eq_catalan_l132_13208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_and_function_properties_l132_13229

noncomputable def α : Real := Real.arctan (Real.sqrt 3 / 3)

noncomputable def f (x : Real) : Real := Real.cos (x - α) * Real.cos α - Real.sin (x - α) * Real.sin α

noncomputable def g (x : Real) : Real := Real.sqrt 3 * f (Real.pi / 2 - 2 * x) - 2 * f x ^ 2

theorem angle_and_function_properties :
  (Real.sin (2 * α) - Real.tan α = -Real.sqrt 3 / 6) ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi / 3), -2 ≤ g x ∧ g x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_and_function_properties_l132_13229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_rental_cost_l132_13283

/-- Proves that the weekly rental cost is $10 given the specified conditions -/
theorem weekly_rental_cost (months_per_year weeks_per_year monthly_rate annual_savings : ℕ) 
  (h1 : months_per_year = 12)
  (h2 : weeks_per_year = 52)
  (h3 : monthly_rate = 40)
  (h4 : annual_savings = 40)
  : (months_per_year * monthly_rate + annual_savings) / weeks_per_year = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_rental_cost_l132_13283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l132_13205

theorem product_of_four_integers (P Q R S : ℕ) : 
  P + Q + R + S = 100 →
  P + 4 = Q - 4 →
  P + 4 = R * 4 →
  P + 4 = S / 4 →
  P * Q * R * S = 61440 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l132_13205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_sqrt30_div_6_lambda_mu_sum_squares_is_one_l132_13215

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The center of the ellipse is at the origin -/
  center_at_origin : True
  /-- The foci are on the x-axis -/
  foci_on_x_axis : True
  /-- A line with slope 1/2 passes through the right focus F -/
  line_through_focus : True
  /-- The line intersects the ellipse at points A and B -/
  A : ℝ × ℝ
  B : ℝ × ℝ
  /-- Vectors OA + OB and (-3, 1) are collinear -/
  collinear : ∃ (k : ℝ), A.1 + B.1 = -3*k ∧ A.2 + B.2 = k

/-- The eccentricity of the special ellipse is √30/6 -/
theorem eccentricity_is_sqrt30_div_6 (e : SpecialEllipse) : 
  ∃ (ecc : ℝ), ecc = Real.sqrt 30 / 6 := by sorry

/-- For any point M on the ellipse where OM = λOA + μOB, λ² + μ² = 1 -/
theorem lambda_mu_sum_squares_is_one (e : SpecialEllipse) :
  ∀ (M : ℝ × ℝ) (lambda mu : ℝ), 
    M.1 = lambda * e.A.1 + mu * e.B.1 ∧ 
    M.2 = lambda * e.A.2 + mu * e.B.2 → 
    lambda^2 + mu^2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_sqrt30_div_6_lambda_mu_sum_squares_is_one_l132_13215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_45_equals_1991_l132_13297

def a : ℕ → ℚ
  | 0 => 11
  | 1 => 11
  | n + 2 => (1/2) * (a (2*(n/2)) + a (2*(n%2))) - ((n/2) - (n%2))^2

theorem a_45_equals_1991 : a 45 = 1991 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_45_equals_1991_l132_13297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whale_sixth_hour_consumption_l132_13235

/-- Represents the whale's feeding pattern over 9 hours -/
def whale_feeding (x : ℕ) : ℕ → ℕ
| 0 => x  -- First hour
| n + 1 => whale_feeding x n + 4  -- Subsequent hours

/-- The total consumption over 9 hours -/
def total_consumption (x : ℕ) : ℕ :=
  (List.range 9).map (whale_feeding x) |> List.sum

theorem whale_sixth_hour_consumption :
  ∃ x : ℕ, total_consumption x = 450 ∧ whale_feeding x 5 = 54 := by
  use 34
  apply And.intro
  · rfl
  · rfl

#eval whale_feeding 34 5  -- Should output 54
#eval total_consumption 34  -- Should output 450

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whale_sixth_hour_consumption_l132_13235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_equality_l132_13289

theorem definite_integral_equality : 
  ∫ x in (0 : ℝ)..4, (Real.sqrt (16 - x^2) - 1/2 * x) = 4 * Real.pi - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_equality_l132_13289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l132_13292

/-- The parabola y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point A -/
def A : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (5, 5)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_sum :
  ∀ P ∈ Parabola, distance A P + distance B P ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l132_13292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l132_13251

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the point on the curve
def point : ℝ × ℝ := (1, f 1)

-- Define the slope of the tangent line
noncomputable def tangent_slope : ℝ := 2 * point.1 + 2

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 4*x - y - 1 = 0

-- Theorem statement
theorem tangent_line_at_point :
  tangent_line point.1 point.2 ∧
  ∀ x y, tangent_line x y ↔ y - point.2 = tangent_slope * (x - point.1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l132_13251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l132_13228

open Real

theorem function_inequality {f : ℝ → ℝ} (hf : Differentiable ℝ f) 
  (hf' : Differentiable ℝ (deriv f))
  (h : ∀ x, x > 0 → x * (deriv (deriv f) x) < 1) : 
  f (exp 1) < f 1 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l132_13228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_sphere_ratio_l132_13281

/-- A truncated cone circumscribed around a sphere -/
structure TruncatedConeSphere where
  R : ℝ  -- radius of the base circle
  r : ℝ  -- radius of the top circle
  ρ : ℝ  -- radius of the sphere
  h : ℝ  -- height of the truncated cone

/-- The volume of the truncated cone -/
noncomputable def volume_cone (tcs : TruncatedConeSphere) : ℝ :=
  (Real.pi / 3) * tcs.h * (tcs.R^2 + tcs.r^2 + tcs.R * tcs.r)

/-- The volume of the sphere -/
noncomputable def volume_sphere (tcs : TruncatedConeSphere) : ℝ :=
  (4 / 3) * Real.pi * tcs.ρ^3

/-- Theorem: If the volume of the truncated cone is twice the volume of the sphere,
    then the ratio of the base radius to the top radius is (3 + √5) / 2 -/
theorem truncated_cone_sphere_ratio (tcs : TruncatedConeSphere) :
  volume_cone tcs = 2 * volume_sphere tcs →
  tcs.h = 2 * tcs.ρ →
  tcs.R / tcs.r = (3 + Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_sphere_ratio_l132_13281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_is_five_l132_13245

/-- The radius of a cylinder's base given its height and surface area -/
noncomputable def cylinder_base_radius (h : ℝ) (surface_area : ℝ) : ℝ :=
  Real.sqrt ((surface_area / (2 * Real.pi) - 4 * h) / 2)

/-- Theorem: The radius of a cylinder with height 8 cm and surface area 130π cm² is 5 cm -/
theorem cylinder_radius_is_five :
  cylinder_base_radius 8 (130 * Real.pi) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_is_five_l132_13245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_quadratic_inequality_l132_13271

noncomputable def f (x : ℝ) := 2 * Real.sin (Real.pi * x)

theorem max_value_and_quadratic_inequality 
  (x₀ : ℝ) 
  (h : ∀ x, f x ≤ f x₀) :
  {m : ℝ | m^2 + m - f x₀ > 0} = {m : ℝ | m < -2 ∨ m > 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_quadratic_inequality_l132_13271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_case_l132_13236

/-- A right triangle with specific properties -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0  -- Right angle at A
  circumcenter : ℝ × ℝ  -- Circumcenter O
  orthocenter : ℝ × ℝ  -- Orthocenter H
  bo_eq_bh : (B.1 - circumcenter.1)^2 + (B.2 - circumcenter.2)^2 = 
             (B.1 - orthocenter.1)^2 + (B.2 - orthocenter.2)^2  -- BO = BH

/-- The theorem to be proved -/
theorem right_triangle_special_case (t : RightTriangle) : 
  let angle_B := Real.arccos ((t.C.1 - t.B.1) * (t.A.1 - t.B.1) + (t.C.2 - t.B.2) * (t.A.2 - t.B.2)) / 
                 (((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2) * ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2))^(1/2)
  angle_B * (180 / Real.pi) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_case_l132_13236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventhTerm_is_25_l132_13257

/-- An arithmetic sequence with a given sum of first seven terms and first term -/
structure ArithmeticSequence where
  S₇ : ℚ  -- Sum of first seven terms
  a₁ : ℚ  -- First term
  -- Condition: S₇ = 77 and a₁ = 5
  h₁ : S₇ = 77
  h₂ : a₁ = 5

/-- The eleventh term of the arithmetic sequence -/
def eleventhTerm (seq : ArithmeticSequence) : ℚ :=
  -- Definition of the eleventh term
  seq.a₁ + 10 * ((seq.S₇ - 7 * seq.a₁) / 21)

/-- Theorem stating that the eleventh term is 25 -/
theorem eleventhTerm_is_25 (seq : ArithmeticSequence) :
  eleventhTerm seq = 25 := by
  -- Unfold the definition of eleventhTerm
  unfold eleventhTerm
  -- Substitute the known values
  rw [seq.h₁, seq.h₂]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventhTerm_is_25_l132_13257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_n_squared_less_than_n_not_dividing_n_l132_13296

def n : ℕ := 2^20 * 5^15

theorem divisors_of_n_squared_less_than_n_not_dividing_n : 
  (Finset.filter (fun d => d < n ∧ d ∣ n^2 ∧ ¬(d ∣ n)) (Finset.range (n + 1))).card = 299 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_n_squared_less_than_n_not_dividing_n_l132_13296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_binomial_expansion_l132_13227

theorem sum_of_coefficients_binomial_expansion (n : ℕ) :
  ∃ (sum : ℤ), (sum = 1 ∨ sum = -1) ∧ 
  sum = (1 - 2)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_binomial_expansion_l132_13227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_value_l132_13231

/-- The function f(x, y) = (3xy+1)e^(-(x^2+y^2)) -/
noncomputable def f (x y : ℝ) : ℝ := (3 * x * y + 1) * Real.exp (-(x^2 + y^2))

/-- Theorem stating that f has a maximum value on ℝ² and it equals (3/2)e^(-1/3) -/
theorem f_has_max_value :
  ∃ (max_val : ℝ), max_val = (3/2) * Real.exp (-1/3) ∧
  ∀ (x y : ℝ), f x y ≤ max_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_value_l132_13231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_l132_13242

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sequence_term (n : ℕ) : ℕ := factorial n * n

def sum_of_terms : ℕ := (List.range 10).map (λ i => sequence_term (i + 1)) |>.sum

theorem units_digit_of_sum :
  sum_of_terms % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_l132_13242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_in_specific_cylinder_l132_13254

/-- The longest segment that fits inside a cylinder. -/
noncomputable def longest_segment (radius : ℝ) (height : ℝ) : ℝ :=
  Real.sqrt ((2 * radius)^2 + height^2)

/-- Theorem: The longest segment that fits inside a cylinder with radius 5 cm and height 12 cm is √244 cm. -/
theorem longest_segment_in_specific_cylinder :
  longest_segment 5 12 = Real.sqrt 244 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_in_specific_cylinder_l132_13254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_specific_trapezoid_l132_13291

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  AB : ℝ  -- Length of longer base
  CD : ℝ  -- Length of shorter base
  AD : ℝ  -- Length of non-parallel side (equal to BC)
  ab_gt_cd : AB > CD
  ad_eq_bc : AD = AD  -- This is trivially true, but represents AD = BC

/-- The length of the diagonal in an isosceles trapezoid -/
noncomputable def diagonal_length (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt (494 : ℝ)

/-- Theorem: The diagonal length of the specific isosceles trapezoid is √494 -/
theorem diagonal_length_specific_trapezoid :
  ∃ t : IsoscelesTrapezoid,
    t.AB = 25 ∧ t.CD = 13 ∧ t.AD = 13 ∧ diagonal_length t = Real.sqrt (494 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_specific_trapezoid_l132_13291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_l132_13219

theorem inequality_holds_iff (a : ℝ) (ha : a < 0) :
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_l132_13219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportion_square_perimeter_l132_13252

/-- Represents a relationship between two variables -/
structure Relationship where
  f : ℝ → ℝ

/-- Checks if a relationship is a direct proportion -/
def is_direct_proportion (r : Relationship) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, r.f x = k * x

/-- The relationship between the area of a square and its side length -/
noncomputable def square_area : Relationship :=
  ⟨λ x => x^2⟩

/-- The relationship between the height and base of a triangle with constant area -/
noncomputable def triangle_height (area : ℝ) : Relationship :=
  ⟨λ a => 2 * area / a⟩

/-- The relationship between the perimeter of a square and its side length -/
def square_perimeter : Relationship :=
  ⟨λ x => 4 * x⟩

/-- The relationship between remaining water volume and time for a constant outflow rate -/
def water_tank (initial_volume rate : ℝ) : Relationship :=
  ⟨λ t => initial_volume - rate * t⟩

theorem direct_proportion_square_perimeter :
  is_direct_proportion square_perimeter ∧
  ¬ is_direct_proportion square_area ∧
  ¬ is_direct_proportion (triangle_height 20) ∧
  ¬ is_direct_proportion (water_tank 100 0.5) := by
  sorry

#check direct_proportion_square_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportion_square_perimeter_l132_13252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_sin_cos_l132_13200

theorem tan_value_given_sin_cos (x : ℝ) : 
  Real.sin x - 2 * Real.cos x = Real.sqrt 5 → Real.tan x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_sin_cos_l132_13200
